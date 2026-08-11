"""Initial nested leave-one-subject-out evaluation for :class:`GeoAdaptNet`.

The outer fold withholds all four recordings from one participant.  Of the seven
remaining participants, the next participant in ``VALID_SUBJECTS`` order is the
single inner-validation participant and the other six train the frozen final model.
Checkpoint, temperature, and target-median blend are selected only on the four
inner-validation recordings.  Keeping that exact checkpoint makes the transferred
temperature valid for its logit scale; the validation participant is not refit.

Each validation or target recording owns a separate unlabeled chronological
calibration prefix and a freshly initialized causal adapter.  The prefix ends
immediately before the first scored task event; its task labels are never used and
its task events are excluded from metrics.  No adapter state crosses a recording
boundary.

Important limitation: this is an *initial* nested LOSO estimate with one
deterministically chosen inner-validation participant per outer fold, not a full
inner LOSO sweep.  The same four inner-validation recordings select all three
hyperparameters, so their selection variance is not averaged.  Results must not
be presented as clinical validation: the current cohort contains eight healthy
participants and no independent dataset or prospective patient cohort.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from .config import (
    DEFAULT_DATA_CONFIG,
    EPOCH_WINDOWS,
    PROJECT_ROOT,
    SUBJECT_RUNS,
    VALID_SUBJECTS,
)
from .data import SessionData, SessionKey, dataset_contract, load_sessions
from .engine import (
    CovarianceDataset,
    TrainConfig,
    predict_proba,
    resolve_device,
    set_reproducible_seed,
    train_model,
)
from .experiment import (
    _atomic_json,
    _environment,
    _finite_or_none,
    _repository_state,
    _summary,
    relative_matrix_log,
    session_log_references,
)
from .metrics import classification_metrics
from .model import GeoAdaptNet
from .protocols import ProtocolSplit, validate_split
from .recenter import (
    BoundaryRecenter,
    DualLevelOnlineAdapter,
    LogEuclideanCovarianceRecenter,
)


SCHEMA_VERSION = 2
PROTOCOL_NAME = "nested_loso_train-6_validate_calibrate-1_test-1"
MODEL_NAME = "geoadapt"
NESTED_LIMITATION = (
    "One deterministic inner-validation participant is used per outer fold, "
    "rather than averaging hyperparameter selection over a complete inner LOSO; "
    "all four recordings from that participant jointly select the frozen checkpoint, "
    "temperature, and target-median blend and remain excluded from model fitting."
)
COHORT_LIMITATION = (
    "The eight-person cohort consists of healthy participants; this benchmark is "
    "not clinical validation in disabled or paralyzed people."
)


@dataclass(frozen=True)
class LOSOConfig:
    """Settings that fully determine one nested LOSO run."""

    window: str = "deployment"
    include_rest: bool = True
    calibration_task_events: int = 10
    max_epochs: int = 180
    patience: int = 25
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-3
    device: str = "auto"
    commit_confidence: float = 0.85
    temperature_grid: tuple[float, ...] = (0.5, 0.75, 1.0, 1.5, 2.0)
    target_median_blends: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    # Accuracy knobs (defaults reproduce the locked LOSO baseline); fit-side only.
    lr_swap_prob: float = 0.0
    select_metric: str = "loss"
    covariance_shrinkage_method: str = "fixed"
    deterministic: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "temperature_grid", tuple(self.temperature_grid))
        object.__setattr__(self, "target_median_blends", tuple(self.target_median_blends))
        if self.window not in EPOCH_WINDOWS:
            raise ValueError(f"unknown window {self.window!r}")
        if self.select_metric not in {"loss", "balanced_accuracy", "blend"}:
            raise ValueError("select_metric must be loss, balanced_accuracy, or blend")
        if self.covariance_shrinkage_method not in {"fixed", "oas"}:
            raise ValueError("covariance_shrinkage_method must be 'fixed' or 'oas'")
        if self.calibration_task_events < 2:
            raise ValueError("calibration_task_events must be at least 2")
        if self.max_epochs <= 0 or self.patience <= 0 or self.batch_size <= 0:
            raise ValueError("epoch, patience, and batch settings must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("learning rate must be positive and weight decay non-negative")
        if not 0.5 <= self.commit_confidence <= 1.0:
            raise ValueError("commit_confidence must be in [0.5, 1]")
        if not self.temperature_grid or any(value <= 0.0 for value in self.temperature_grid):
            raise ValueError("temperature_grid must contain positive values")
        if len(set(self.temperature_grid)) != len(self.temperature_grid):
            raise ValueError("temperature_grid must not contain duplicates")
        if not self.target_median_blends or any(
            not 0.0 <= value <= 1.0 for value in self.target_median_blends
        ):
            raise ValueError("target_median_blends must lie in [0, 1]")
        if len(set(self.target_median_blends)) != len(self.target_median_blends):
            raise ValueError("target_median_blends must not contain duplicates")


@dataclass(frozen=True)
class NestedLOSOFold:
    """Materialized rows for one outer fold and its protected inner split."""

    outer_subject: int
    validation_subject: int
    selection_subjects: tuple[int, ...]
    available_source_subjects: tuple[int, ...]
    selection_indices: np.ndarray
    validation_indices: np.ndarray
    outer_indices: np.ndarray

    def __post_init__(self) -> None:
        for name in ("selection_indices", "validation_indices", "outer_indices"):
            values = np.asarray(getattr(self, name), dtype=np.int64).reshape(-1)
            object.__setattr__(self, name, values)
            if len(values) == 0:
                raise ValueError(f"{name} must be non-empty")
            if len(np.unique(values)) != len(values):
                raise ValueError(f"{name} contains duplicate rows")
        object.__setattr__(
            self, "selection_subjects", tuple(int(value) for value in self.selection_subjects)
        )
        object.__setattr__(
            self,
            "available_source_subjects",
            tuple(int(value) for value in self.available_source_subjects),
        )
        if self.outer_subject == self.validation_subject:
            raise ValueError("outer and inner-validation subjects must differ")
        protected = {self.outer_subject, self.validation_subject}
        if protected & set(self.selection_subjects):
            raise ValueError("selection subjects overlap a protected subject")
        if set(self.available_source_subjects) != set(self.selection_subjects) | {
            self.validation_subject
        }:
            raise ValueError(
                "available source subjects must be training plus validation subjects"
            )
        partitions = (
            self.selection_indices,
            self.validation_indices,
            self.outer_indices,
        )
        for left in range(len(partitions)):
            for right in range(left + 1, len(partitions)):
                if len(np.intersect1d(partitions[left], partitions[right])):
                    raise ValueError("nested LOSO row partitions overlap")


@dataclass(frozen=True)
class SessionCalibrationPartition:
    """One recording's unlabeled prefix and strictly later live stream."""

    session_id: str
    calibration_indices: np.ndarray
    stream_indices: np.ndarray

    def __post_init__(self) -> None:
        calibration = np.asarray(self.calibration_indices, dtype=np.int64).reshape(-1)
        stream = np.asarray(self.stream_indices, dtype=np.int64).reshape(-1)
        object.__setattr__(self, "session_id", str(self.session_id))
        object.__setattr__(self, "calibration_indices", calibration)
        object.__setattr__(self, "stream_indices", stream)
        if not self.session_id or len(calibration) == 0 or len(stream) == 0:
            raise ValueError("session ID, calibration prefix, and stream must be non-empty")
        if len(np.intersect1d(calibration, stream)):
            raise ValueError("calibration and live rows overlap")


def _json_config(config: LOSOConfig) -> dict[str, Any]:
    """Normalize tuples exactly as JSON does so resume comparisons are stable."""

    return json.loads(json.dumps(asdict(config), sort_keys=True))


def nested_loso_fold(
    data: SessionData | Any,
    outer_subject: int,
    *,
    subject_order: Sequence[int] = VALID_SUBJECTS,
) -> NestedLOSOFold:
    """Create the deterministic 6/1/1 split and hard-fail on subject leakage."""

    order = tuple(int(subject) for subject in subject_order)
    if len(order) < 3 or len(set(order)) != len(order):
        raise ValueError("subject_order must contain at least three unique subjects")
    outer_subject = int(outer_subject)
    if outer_subject not in order:
        raise ValueError(f"outer subject {outer_subject} is absent from subject_order")
    subjects = np.asarray(data.subject_ids, dtype=np.int64).reshape(-1)
    sessions = np.asarray(data.session_ids, dtype=np.str_).reshape(-1)
    if len(subjects) != len(sessions) or len(subjects) == 0:
        raise ValueError("subject and session metadata must be non-empty and equally sized")
    present = set(np.unique(subjects).tolist())
    if present != set(order):
        raise ValueError(
            f"data subjects {sorted(present)} do not match fixed order {sorted(order)}"
        )
    for subject in order:
        count = len(np.unique(sessions[subjects == subject]))
        if count != 4:
            raise ValueError(f"subject {subject} must contribute exactly four sessions, got {count}")

    position = order.index(outer_subject)
    validation_subject = order[(position + 1) % len(order)]
    selection_subjects = tuple(
        subject for subject in order if subject not in {outer_subject, validation_subject}
    )
    available_source_subjects = tuple(
        subject for subject in order if subject != outer_subject
    )
    selection_indices = np.flatnonzero(np.isin(subjects, selection_subjects))
    validation_indices = np.flatnonzero(subjects == validation_subject)
    outer_indices = np.flatnonzero(subjects == outer_subject)

    inner = ProtocolSplit(
        protocol="nested_loso_inner",
        fold=f"outer-{outer_subject:02d}_validation-{validation_subject:02d}",
        train_indices=selection_indices,
        test_indices=validation_indices,
        held_out_subject=validation_subject,
        held_out_sessions=tuple(sorted(np.unique(sessions[validation_indices]).tolist())),
    )
    validate_split(inner, data, require_subject_disjoint=True)
    outer = ProtocolSplit(
        protocol="nested_loso_outer",
        fold=f"test-subject-{outer_subject:02d}",
        train_indices=np.concatenate((selection_indices, validation_indices)),
        test_indices=outer_indices,
        held_out_subject=outer_subject,
        held_out_sessions=tuple(sorted(np.unique(sessions[outer_indices]).tolist())),
    )
    validate_split(outer, data, require_subject_disjoint=True)
    return NestedLOSOFold(
        outer_subject=outer_subject,
        validation_subject=validation_subject,
        selection_subjects=selection_subjects,
        available_source_subjects=available_source_subjects,
        selection_indices=selection_indices,
        validation_indices=validation_indices,
        outer_indices=outer_indices,
    )


def session_calibration_partitions(
    data: SessionData | Any,
    indices: Sequence[int],
    calibration_task_events: int,
) -> tuple[SessionCalibrationPartition, ...]:
    """Split every recording at the same *task-counted* chronological prefix.

    The calibration prefix includes everything before the ``K+1``-st task event,
    including intervening rest events when present.  Consequently exactly the
    first ``K`` task events are unlabeled calibration observations and none can
    enter a score.  Counting task cues uses only phase identity, never left/right
    direction.
    """

    if calibration_task_events < 1:
        raise ValueError("calibration_task_events must be positive")
    rows = np.asarray(indices, dtype=np.int64).reshape(-1)
    if len(rows) == 0 or len(np.unique(rows)) != len(rows):
        raise ValueError("indices must be non-empty and unique")
    labels = np.asarray(data.labels, dtype=np.int64)
    sessions = np.asarray(data.session_ids, dtype=np.str_)
    event_samples = np.asarray(data.event_samples, dtype=np.int64)
    if rows.min(initial=0) < 0 or rows.max(initial=-1) >= len(labels):
        raise ValueError("partition row is outside the data")

    partitions: list[SessionCalibrationPartition] = []
    selected_sessions = sorted(np.unique(sessions[rows]).tolist())
    for session_id in selected_sessions:
        session_rows = rows[sessions[rows] == session_id]
        ordered = session_rows[
            np.argsort(event_samples[session_rows], kind="stable")
        ]
        task_positions = np.flatnonzero(labels[ordered] >= 0)
        if len(task_positions) <= calibration_task_events:
            raise ValueError(
                f"session {session_id} has {len(task_positions)} task events; "
                f"need more than {calibration_task_events}"
            )
        # Everything before the next task is an unlabeled chronological prefix.
        split_position = int(task_positions[calibration_task_events])
        calibration = ordered[:split_position]
        stream = ordered[split_position:]
        if int(np.sum(labels[calibration] >= 0)) != calibration_task_events:
            raise RuntimeError("internal task-prefix accounting error")
        task_labels = labels[stream][labels[stream] >= 0]
        if len(task_labels) == 0 or np.unique(task_labels).size != 2:
            raise ValueError(
                f"session {session_id} must retain both task classes after calibration"
            )
        partition = SessionCalibrationPartition(session_id, calibration, stream)
        if not np.all(sessions[partition.calibration_indices] == session_id) or not np.all(
            sessions[partition.stream_indices] == session_id
        ):
            raise RuntimeError("session calibration crossed a recording boundary")
        partitions.append(partition)
    if not partitions:
        raise ValueError("no sessions were selected")
    return tuple(partitions)


def _training_config(config: LOSOConfig, seed: int) -> TrainConfig:
    return TrainConfig(
        epochs=config.max_epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        patience=config.patience,
        seed=int(seed),
        device=config.device,
        lr_swap_prob=config.lr_swap_prob,
        select_metric=config.select_metric,
        deterministic=config.deterministic,
    )


def _validation_dataset(
    data: SessionData,
    partitions: Sequence[SessionCalibrationPartition],
) -> CovarianceDataset:
    rows: list[np.ndarray] = []
    references: list[np.ndarray] = []
    for partition in partitions:
        reference = relative_matrix_log(
            data.covariances[partition.calibration_indices]
        ).mean(axis=0).astype(np.float32)
        rows.append(partition.stream_indices)
        references.append(
            np.broadcast_to(
                reference,
                (len(partition.stream_indices),) + reference.shape,
            ).copy()
        )
    selected = np.concatenate(rows)
    return CovarianceDataset(
        data.covariances[selected],
        data.labels[selected],
        log_references=np.concatenate(references),
    )


def _new_adapter(
    *, temperature: float, blend: float, config: LOSOConfig
) -> DualLevelOnlineAdapter:
    covariance = LogEuclideanCovarianceRecenter(
        alpha=0.001,
        robust_clip=1.0,
        eig_floor=1e-15,
        alignment_enabled=True,
        adaptation_enabled=True,
    )
    boundary = BoundaryRecenter(
        alpha=0.01,
        clamp=2.0,
        rest_confidence=0.65,
        commit_threshold=config.commit_confidence,
        temperature=float(temperature),
        class1_label=0,
        class2_label=1,
    )
    # ``blend`` is applied after the session-specific unlabeled median is known.
    del blend
    return DualLevelOnlineAdapter(
        covariance, boundary, covariance_updates_rest_only=True
    )


def _probabilities_from_margin(margin: float, temperature: float) -> tuple[float, float]:
    scaled = float(np.clip(margin / temperature, -40.0, 40.0))
    positive = 1.0 / (1.0 + math.exp(-scaled))
    return 1.0 - positive, positive


@torch.no_grad()
def _one_model_score(
    model: GeoAdaptNet,
    aligned_covariance: np.ndarray,
    zero_reference: np.ndarray,
    *,
    device: torch.device,
) -> tuple[float, float | None]:
    covariance = torch.as_tensor(
        aligned_covariance, dtype=torch.float32, device=device
    ).unsqueeze(0)
    reference = torch.as_tensor(
        zero_reference, dtype=torch.float32, device=device
    ).unsqueeze(0)
    output = model(covariance, log_reference=reference)
    score = float(output.log_odds.reshape(-1)[0].detach().cpu())
    intent = (
        None
        if output.intent_logit is None
        else float(torch.sigmoid(output.intent_logit.reshape(-1)[0]).detach().cpu())
    )
    return score, intent


def _metric_payload(
    labels: Sequence[int],
    probabilities: Sequence[tuple[float, float]],
    committed: Sequence[bool],
    *,
    rest_windows: int,
    rest_false_commits: int,
    commit_confidence: float,
) -> dict[str, Any]:
    truth = np.asarray(labels, dtype=np.int64)
    probability_array = np.asarray(probabilities, dtype=np.float64)
    commit_array = np.asarray(committed, dtype=bool)
    metrics = classification_metrics(
        truth, probability_array, commit_confidence=commit_confidence
    ).to_dict()
    metrics["coverage"] = float(commit_array.mean())
    metrics["selective_accuracy"] = (
        float(np.mean(probability_array[commit_array].argmax(axis=1) == truth[commit_array]))
        if np.any(commit_array)
        else None
    )
    metrics["rest_false_commit_rate"] = (
        float(rest_false_commits / rest_windows) if rest_windows else None
    )
    return {
        key: None if value is None else _finite_or_none(value)
        for key, value in metrics.items()
    }


def evaluate_subject_sessions(
    model: GeoAdaptNet,
    data: SessionData,
    partitions: Sequence[SessionCalibrationPartition],
    *,
    temperature: float,
    target_median_blend: float,
    config: LOSOConfig,
) -> dict[str, Any]:
    """Evaluate independent causal streams and aggregate participant metrics.

    This function deliberately constructs a new adapter inside the session loop.
    Neither the covariance reference nor decision center can carry over to the
    next recording.
    """

    if not partitions:
        raise ValueError("at least one session partition is required")
    resolved = resolve_device(config.device)
    model = model.to(resolved).eval()
    all_truth: list[int] = []
    all_probabilities: list[tuple[float, float]] = []
    all_committed: list[bool] = []
    total_rest = 0
    total_rest_false_commits = 0
    total_covariance_updates = 0
    total_boundary_updates = 0
    session_payloads: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []

    for partition in partitions:
        adapter = _new_adapter(
            temperature=temperature, blend=target_median_blend, config=config
        )
        calibration_covariances = data.covariances[partition.calibration_indices]
        adapter.covariance.calibrate(calibration_covariances)
        aligned_calibration = adapter.covariance.transform(calibration_covariances)
        zero_reference = np.zeros(aligned_calibration.shape[1:], dtype=np.float32)
        calibration_probabilities, _ = predict_proba(
            model,
            CovarianceDataset(
                aligned_calibration,
                data.labels[partition.calibration_indices],
                log_references=zero_reference,
            ),
            device=config.device,
        )
        positive = np.clip(calibration_probabilities[:, 1], 1e-7, 1.0 - 1e-7)
        calibration_scores = np.log(positive) - np.log1p(-positive)
        adapter.boundary.calibrate(calibration_scores)
        adapter.boundary.center *= target_median_blend
        adapter.boundary.seed_center = adapter.boundary.center

        session_truth: list[int] = []
        session_probabilities: list[tuple[float, float]] = []
        session_committed: list[bool] = []
        session_rest = 0
        session_rest_false_commits = 0
        covariance_updates = 0
        boundary_updates = 0
        session_predictions: list[dict[str, Any]] = []

        for row in partition.stream_indices:
            raw_covariance = data.covariances[row]
            # Score under state t.  ``adapter.step`` repeats this alignment before
            # forming its state transition, but does not update until afterward.
            aligned = adapter.covariance.transform(raw_covariance)
            score, intent_probability = _one_model_score(
                model, aligned, zero_reference, device=resolved
            )
            rest_probability = (
                None if intent_probability is None else 1.0 - intent_probability
            )
            decision = adapter.step(
                raw_covariance,
                score=score,
                rest_probability=rest_probability,
                update=True,
            )
            probabilities = _probabilities_from_margin(decision.margin, temperature)
            intent_gate = rest_probability is None or rest_probability < 0.5
            effective_commit = bool(decision.committed and intent_gate)
            label = int(data.labels[row])
            prediction_record = {
                "row_index": int(row),
                "subject": int(data.subject_ids[row]),
                "run": int(data.run_ids[row]),
                "session_id": str(data.session_ids[row]),
                "event_sample": int(data.event_samples[row]),
                "event_onset_seconds": float(data.event_onsets[row]),
                "trial_id": int(data.trial_ids[row]),
                "annotation": str(data.annotations[row]),
                "label": label,
                "probability_left": float(probabilities[0]),
                "probability_right": float(probabilities[1]),
                "intent_probability": intent_probability,
                "rest_probability": rest_probability,
                "prediction": int(decision.prediction),
                "confidence": float(decision.confidence),
                "committed": effective_commit,
                "margin": float(decision.margin),
                "center_before": float(decision.center_before),
                "center_after": float(decision.center_after),
                "covariance_updated": bool(decision.covariance_updated),
                "boundary_updated": bool(decision.boundary_updated),
            }
            session_predictions.append(prediction_record)
            if label >= 0:
                session_truth.append(label)
                session_probabilities.append(probabilities)
                session_committed.append(effective_commit)
            else:
                session_rest += 1
                session_rest_false_commits += int(effective_commit)
            covariance_updates += int(decision.covariance_updated)
            boundary_updates += int(decision.boundary_updated)

        session_metrics = _metric_payload(
            session_truth,
            session_probabilities,
            session_committed,
            rest_windows=session_rest,
            rest_false_commits=session_rest_false_commits,
            commit_confidence=config.commit_confidence,
        )
        session_payloads.append(
            {
                "session_id": partition.session_id,
                "calibration_windows": len(partition.calibration_indices),
                "calibration_task_events": int(
                    np.sum(data.labels[partition.calibration_indices] >= 0)
                ),
                "observed_post_calibration_windows": len(partition.stream_indices),
                "scored_task_windows": len(session_truth),
                "rest_windows": session_rest,
                "rest_false_commits": session_rest_false_commits,
                "covariance_updates": covariance_updates,
                "boundary_updates": boundary_updates,
                "metrics": session_metrics,
                "predictions": session_predictions,
            }
        )
        all_truth.extend(session_truth)
        all_probabilities.extend(session_probabilities)
        all_committed.extend(session_committed)
        total_rest += session_rest
        total_rest_false_commits += session_rest_false_commits
        total_covariance_updates += covariance_updates
        total_boundary_updates += boundary_updates
        all_predictions.extend(session_predictions)

    metrics = _metric_payload(
        all_truth,
        all_probabilities,
        all_committed,
        rest_windows=total_rest,
        rest_false_commits=total_rest_false_commits,
        commit_confidence=config.commit_confidence,
    )
    return {
        "metrics": metrics,
        "predictions": all_predictions,
        "sessions": session_payloads,
        "deployment": {
            "sessions": len(partitions),
            "calibration_task_events_per_session": config.calibration_task_events,
            "calibration_windows": int(
                sum(len(partition.calibration_indices) for partition in partitions)
            ),
            "scored_task_windows": len(all_truth),
            "rest_windows": total_rest,
            "rest_false_commits": total_rest_false_commits,
            "covariance_updates": total_covariance_updates,
            "boundary_updates": total_boundary_updates,
            "adapter_reset_per_session": True,
        },
    }


def _select_adaptation_parameters(
    model: GeoAdaptNet,
    data: SessionData,
    partitions: Sequence[SessionCalibrationPartition],
    config: LOSOConfig,
) -> tuple[float, float, list[dict[str, Any]]]:
    """Select a predeclared temperature/blend pair on the inner subject only."""

    trace: list[dict[str, Any]] = []
    best_key: tuple[float, float, float, float] | None = None
    best_values: tuple[float, float] | None = None
    for temperature in config.temperature_grid:
        for blend in config.target_median_blends:
            evaluated = evaluate_subject_sessions(
                model,
                data,
                partitions,
                temperature=temperature,
                target_median_blend=blend,
                config=config,
            )
            metrics = evaluated["metrics"]
            balanced = float(metrics["balanced_accuracy"])
            brier = float(metrics["brier"])
            trace.append(
                {
                    "temperature": temperature,
                    "target_median_blend": blend,
                    "balanced_accuracy": balanced,
                    "brier": brier,
                    "coverage": metrics["coverage"],
                    "rest_false_commit_rate": metrics["rest_false_commit_rate"],
                }
            )
            # Primary discrimination, then calibration.  Exact ties prefer a
            # temperature nearer one and less target-derived boundary movement.
            key = (balanced, -brier, -abs(math.log(temperature)), -blend)
            if best_key is None or key > best_key:
                best_key = key
                best_values = (float(temperature), float(blend))
    if best_values is None:
        raise RuntimeError("adaptation hyperparameter selection produced no candidate")
    return best_values[0], best_values[1], trace


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.stem}-", suffix=".pt", dir=path.parent, delete=False
        ) as temporary:
            temporary_name = temporary.name
        torch.save(payload, temporary_name)
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _checkpoint_path(output: Path, outer_subject: int, seed: int) -> Path:
    return (
        PROJECT_ROOT
        / "deepnet"
        / "checkpoints"
        / f"{output.stem}_nested_loso"
        / f"outer-{outer_subject:02d}_seed-{seed}.pt"
    )


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _run_fold(
    data: SessionData,
    fold: NestedLOSOFold,
    seed: int,
    config: LOSOConfig,
    *,
    input_contract: dict[str, Any],
    checkpoint_path: Path,
) -> dict[str, Any]:
    validation_partitions = session_calibration_partitions(
        data, fold.validation_indices, config.calibration_task_events
    )
    selection_references = session_log_references(
        data.covariances[fold.selection_indices],
        data.session_ids[fold.selection_indices],
    )
    train_config = _training_config(config, seed)
    set_reproducible_seed(seed, deterministic=config.deterministic)
    selection_model = GeoAdaptNet(auxiliary_intent=config.include_rest)
    selected = train_model(
        selection_model,
        CovarianceDataset(
            data.covariances[fold.selection_indices],
            data.labels[fold.selection_indices],
            log_references=selection_references,
        ),
        _validation_dataset(data, validation_partitions),
        train_config,
    )
    temperature, blend, adaptation_trace = _select_adaptation_parameters(
        selected.model, data, validation_partitions, config
    )

    # Temperature is model/logit-scale specific.  The protected validation
    # participant selects this exact checkpoint, which is then frozen for the
    # untouched outer participant; no differently scaled refit receives it.
    final_model = selected.model

    outer_partitions = session_calibration_partitions(
        data, fold.outer_indices, config.calibration_task_events
    )
    evaluated = evaluate_subject_sessions(
        final_model,
        data,
        outer_partitions,
        temperature=temperature,
        target_median_blend=blend,
        config=config,
    )
    checkpoint = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL_NAME,
        "model": "GeoAdaptNet",
        "model_config": {"auxiliary_intent": config.include_rest},
        "state_dict": {
            key: value.detach().cpu() for key, value in final_model.state_dict().items()
        },
        "outer_subject": fold.outer_subject,
        "inner_validation_subject": fold.validation_subject,
        "training_subjects": list(fold.selection_subjects),
        "available_source_subjects": list(fold.available_source_subjects),
        "seed": int(seed),
        "selected_epochs": selected.best_epoch + 1,
        "temperature": temperature,
        "target_median_blend": blend,
        "loso_config": _json_config(config),
        "data_contract": input_contract,
        "nested_limitation": NESTED_LIMITATION,
    }
    _atomic_torch_save(checkpoint_path, checkpoint)
    return {
        "model": MODEL_NAME,
        "subject": fold.outer_subject,
        "outer_subject": fold.outer_subject,
        "inner_validation_subject": fold.validation_subject,
        "training_subjects": list(fold.selection_subjects),
        "available_source_subjects": list(fold.available_source_subjects),
        "seed": int(seed),
        "metrics": evaluated["metrics"],
        "predictions": evaluated["predictions"],
        "sessions": evaluated["sessions"],
        "selection": {
            "best_epoch_zero_based": selected.best_epoch,
            "selected_epochs": selected.best_epoch + 1,
            "epochs_ran": selected.epochs_ran,
            "validation_loss": _finite_or_none(selected.best_validation_loss),
            "validation_balanced_accuracy": _finite_or_none(
                selected.best_validation_balanced_accuracy
            ),
            "temperature": temperature,
            "target_median_blend": blend,
            "candidate_trace": adaptation_trace,
            "inner_validation_sessions": [
                partition.session_id for partition in validation_partitions
            ],
            "selection_train_seconds": selected.train_seconds,
            "final_model": "protected_validation_checkpoint_no_refit",
        },
        "deployment": {
            **evaluated["deployment"],
            "parameters": final_model.parameter_count,
            "residual_gate": float(final_model.residual_gate.detach().cpu()),
            "checkpoint": _display_path(checkpoint_path),
        },
    }


def run_nested_loso(
    *,
    subjects: Sequence[int],
    seeds: Sequence[int],
    config: LOSOConfig,
    output: Path,
    resume: bool = True,
) -> dict[str, Any]:
    """Run selected outer folds while retaining the complete eight-subject nest."""

    outer_subjects = tuple(int(subject) for subject in subjects)
    seed_values = tuple(int(seed) for seed in seeds)
    if not outer_subjects or len(set(outer_subjects)) != len(outer_subjects):
        raise ValueError("subjects must be non-empty and unique")
    if set(outer_subjects) - set(VALID_SUBJECTS):
        raise ValueError(f"outer subjects must be selected from {VALID_SUBJECTS}")
    if not seed_values or len(set(seed_values)) != len(seed_values):
        raise ValueError("seeds must be non-empty and unique")
    if any(seed < 0 for seed in seed_values):
        raise ValueError("seeds must be non-negative")
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {config.device!r} was requested but torch.cuda.is_available() is false"
        )

    keys = [
        SessionKey(subject, run)
        for subject in VALID_SUBJECTS
        for run in SUBJECT_RUNS[subject]
    ]
    data_config = replace(
        DEFAULT_DATA_CONFIG,
        window_name=config.window,
        include_rest=config.include_rest,
        covariance_shrinkage_method=config.covariance_shrinkage_method,
    )
    data = load_sessions(keys, data_config)
    input_contract = dataset_contract(data, data_config)
    normalized_config = _json_config(config)

    if resume and output.is_file():
        payload = json.loads(output.read_text(encoding="utf-8"))
        if payload.get("schema_version") != SCHEMA_VERSION or payload.get(
            "protocol"
        ) != PROTOCOL_NAME:
            raise ValueError("cannot resume an incompatible LOSO result schema")
        if payload.get("loso_config") != normalized_config:
            raise ValueError("cannot resume with a different LOSO configuration")
        if payload.get("data_contract") != input_contract:
            raise ValueError("cannot resume with a different preprocessing/data contract")
        if payload.get("outer_subjects") != list(outer_subjects):
            raise ValueError("cannot resume with different outer subjects")
        if payload.get("seeds") != list(seed_values):
            raise ValueError("cannot resume with different seeds")
    else:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "protocol": PROTOCOL_NAME,
            "model": MODEL_NAME,
            "loso_config": normalized_config,
            "data_contract": input_contract,
            "subject_order": list(VALID_SUBJECTS),
            "outer_subjects": list(outer_subjects),
            "seeds": list(seed_values),
            "environment": _environment(),
            "repository": _repository_state(),
            "command": list(sys.argv),
            "limitations": [NESTED_LIMITATION, COHORT_LIMITATION],
            "folds": [],
            "summary": {},
        }

    rows: list[dict[str, Any]] = payload["folds"]
    completed = {
        (int(row["outer_subject"]), int(row["seed"])) for row in rows
    }
    for outer_subject in outer_subjects:
        fold = nested_loso_fold(data, outer_subject)
        for seed in seed_values:
            key = (outer_subject, seed)
            if key in completed:
                print(
                    f"skip complete nested LOSO outer={outer_subject} seed={seed}",
                    flush=True,
                )
                continue
            print(
                f"run nested LOSO outer={outer_subject} "
                f"inner_validation={fold.validation_subject} seed={seed}",
                flush=True,
            )
            row = _run_fold(
                data,
                fold,
                seed,
                config,
                input_contract=input_contract,
                checkpoint_path=_checkpoint_path(output, outer_subject, seed),
            )
            rows.append(row)
            completed.add(key)
            payload["updated_at"] = datetime.now(timezone.utc).isoformat()
            payload["summary"] = _summary(rows)
            _atomic_json(output, payload)
            balanced = row["metrics"]["balanced_accuracy"]
            print(
                f"done nested LOSO outer={outer_subject} seed={seed}: "
                f"balanced_accuracy={balanced:.3f}",
                flush=True,
            )
    return payload


def _csv_ints(value: str, *, all_values: Iterable[int] | None = None) -> tuple[int, ...]:
    if value.strip().lower() == "all":
        if all_values is None:
            raise argparse.ArgumentTypeError("'all' is not valid here")
        return tuple(all_values)
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not parsed:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return parsed


def _csv_floats(value: str) -> tuple[float, ...]:
    try:
        parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from error
    if not parsed:
        raise argparse.ArgumentTypeError("at least one number is required")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=f"Nested limitation: {NESTED_LIMITATION}",
    )
    parser.add_argument(
        "--model",
        choices=(MODEL_NAME,),
        default=MODEL_NAME,
        help="initial nested runner currently supports GeoAdaptNet only",
    )
    parser.add_argument("--subjects", default="all", help="outer subjects or 'all'")
    parser.add_argument("--seeds", default="7", help="fixed comma-separated fit seeds")
    parser.add_argument("--window", choices=tuple(EPOCH_WINDOWS), default="deployment")
    parser.add_argument("--task-only", action="store_true")
    parser.add_argument(
        "--calibration-task-events",
        type=int,
        default=LOSOConfig.calibration_task_events,
    )
    parser.add_argument("--max-epochs", type=int, default=180)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument(
        "--device",
        default="auto",
        help="torch device; auto selects CUDA when available and otherwise a safe CPU path",
    )
    parser.add_argument("--commit-confidence", type=float, default=0.85)
    parser.add_argument(
        "--temperature-grid",
        type=_csv_floats,
        default=LOSOConfig.temperature_grid,
    )
    parser.add_argument(
        "--target-median-blends",
        type=_csv_floats,
        default=LOSOConfig.target_median_blends,
    )
    parser.add_argument("--lr-swap-prob", type=float, default=0.0)
    parser.add_argument(
        "--select-metric", choices=("loss", "balanced_accuracy", "blend"), default="loss"
    )
    parser.add_argument(
        "--covariance-shrinkage-method", choices=("fixed", "oas"), default="fixed"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="faster iteration: non-deterministic kernels + TF32 matmul (accuracy ~unchanged)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "deepnet" / "results" / "nested_loso.json",
    )
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    subjects = _csv_ints(args.subjects, all_values=VALID_SUBJECTS)
    seeds = _csv_ints(args.seeds)
    config = LOSOConfig(
        window=args.window,
        include_rest=not args.task_only,
        calibration_task_events=args.calibration_task_events,
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=args.device,
        commit_confidence=args.commit_confidence,
        temperature_grid=args.temperature_grid,
        target_median_blends=args.target_median_blends,
        lr_swap_prob=args.lr_swap_prob,
        select_metric=args.select_metric,
        covariance_shrinkage_method=args.covariance_shrinkage_method,
        deterministic=not args.fast,
    )
    payload = run_nested_loso(
        subjects=subjects,
        seeds=seeds,
        config=config,
        output=args.output,
        resume=not args.no_resume,
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "COHORT_LIMITATION",
    "LOSOConfig",
    "MODEL_NAME",
    "NESTED_LIMITATION",
    "NestedLOSOFold",
    "PROTOCOL_NAME",
    "SCHEMA_VERSION",
    "SessionCalibrationPartition",
    "build_parser",
    "evaluate_subject_sessions",
    "main",
    "nested_loso_fold",
    "run_nested_loso",
    "session_calibration_partitions",
]
