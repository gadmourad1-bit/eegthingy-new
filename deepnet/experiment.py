"""Reproducible, leakage-resistant experiments for the geometric EEG decoder.

The primary benchmark is chronological and subject-specific: two recordings are
used for training, the third recording selects the model checkpoint and calibrates
its output, and that same frozen model is used before the fourth is opened.  An
unlabeled chronological prefix of the fourth recording seeds deployment state and
is excluded from scoring.  No target label is used for fitting or adaptation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from sklearn.metrics import balanced_accuracy_score

from .augment import left_right_swap_index
from .baselines import EAFilterBankCSP, RiemannianTangentLogistic
from .config import CHANNELS, DEFAULT_DATA_CONFIG, EPOCH_WINDOWS, PROJECT_ROOT, SUBJECT_RUNS
from .data import SessionData, SessionKey, dataset_contract, load_sessions
from .engine import (
    CovarianceDataset,
    TrainConfig,
    batch_one_latency_ms,
    predict_proba,
    set_reproducible_seed,
    train_model,
)
from .metrics import classification_metrics
from .model import GeoAdaptNet
from .recenter import (
    BoundaryRecenter,
    DualLevelOnlineAdapter,
    LogEuclideanCovarianceRecenter,
)


SCHEMA_VERSION = 2
DEFAULT_MODELS = ("geoadapt", "riemann", "fbcsp")


@dataclass(frozen=True)
class BenchmarkConfig:
    window: str = "deployment"
    include_rest: bool = True
    calibration_windows: int = 20
    max_epochs: int = 180
    patience: int = 25
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-3
    device: str = "auto"
    commit_confidence: float = 0.85
    # GeoAdaptNet accuracy knobs.  All defaults reproduce the locked schema-v2
    # baseline exactly; they are fit/selection-side only and never touch the
    # outer test recording, so raising them cannot leak.
    deep_supervision_weight: float = 0.0
    residual_gate_init: float = 0.02
    exclude_gate_from_weight_decay: bool = False
    select_metric: str = "loss"
    label_smoothing: float = 0.05
    # Seed the residual BiMaps from supervised CSP filters fitted on the inner
    # training recordings only (never rec3/rec4).  The single mechanism EA-FBCSP
    # has that the net's random projection lacks.
    csp_warm_start: bool = False
    # Fit-side covariance augmentation (training recordings only): left/right
    # electrode swap + label flip, and log-Euclidean same-class mixup.
    lr_swap_prob: float = 0.0
    mixup_alpha: float = 0.0
    # Covariance shrinkage estimator applied identically to every window at fit
    # and deploy: "fixed" (1e-3) or adaptive per-window "oas".
    covariance_shrinkage_method: str = "fixed"
    # Reproducible-but-slower by default; the fast profile flips this.
    deterministic: bool = True
    # Test-time mirror augmentation: average the prediction on each window with
    # the label-flipped prediction on its left/right-mirrored covariance.
    # Unsupervised and applied identically at deploy; costs 2x inference.
    tta_mirror: bool = False

    def __post_init__(self) -> None:
        if self.window not in EPOCH_WINDOWS:
            raise ValueError(f"unknown window {self.window!r}")
        if self.calibration_windows < 2:
            raise ValueError("calibration_windows must be at least 2")
        if self.max_epochs <= 0 or self.patience <= 0 or self.batch_size <= 0:
            raise ValueError("epoch, patience, and batch settings must be positive")
        if not 0.5 <= self.commit_confidence <= 1.0:
            raise ValueError("commit_confidence must be in [0.5, 1]")
        if self.deep_supervision_weight < 0.0:
            raise ValueError("deep_supervision_weight must be non-negative")
        if not 0.0 < self.residual_gate_init < 1.0:
            raise ValueError("residual_gate_init must be strictly between 0 and 1")
        if self.select_metric not in {"loss", "balanced_accuracy", "blend"}:
            raise ValueError("select_metric must be loss, balanced_accuracy, or blend")


def _mirror_covariances(covariances: np.ndarray, swap_index: np.ndarray) -> np.ndarray:
    """Apply the left/right channel permutation ``P C P^T`` to a (..., C, C) batch."""

    return covariances[..., swap_index, :][..., :, swap_index]


def _tta_mirror_probabilities(
    model: Any,
    aligned: np.ndarray,
    labels: np.ndarray,
    zero_reference: np.ndarray,
    swap_index: np.ndarray,
    *,
    device: str,
    batch_size: int,
    branch: str = "full",
) -> tuple[np.ndarray, np.ndarray | None]:
    """Model probabilities averaged with the label-flipped mirrored prediction.

    Mirroring swaps the left/right sensors, so the mirror's class-2 probability
    estimates the original window's class-1 probability.  Averaging the two views
    is a label-free, deploy-consistent variance reducer at 2x inference cost.
    """

    probs, intent = predict_proba(
        model,
        CovarianceDataset(aligned, labels, log_references=zero_reference),
        device=device,
        batch_size=batch_size,
        branch=branch,
    )
    mirror_probs, mirror_intent = predict_proba(
        model,
        CovarianceDataset(
            _mirror_covariances(aligned, swap_index), labels, log_references=zero_reference
        ),
        device=device,
        batch_size=batch_size,
        branch=branch,
    )
    positive = 0.5 * (probs[:, 1] + (1.0 - mirror_probs[:, 1]))
    averaged = np.stack([1.0 - positive, positive], axis=1)
    if intent is not None and mirror_intent is not None:
        intent = 0.5 * (intent + mirror_intent)  # rest/task is mirror-invariant
    return averaged, intent


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + np.swapaxes(matrix, -1, -2))


def relative_matrix_log(matrices: np.ndarray, *, eps: float = 1e-5) -> np.ndarray:
    """NumPy equivalent of the network's scale-relative SPD logarithm."""

    values = np.asarray(matrices, dtype=np.float64)
    if values.ndim < 2 or values.shape[-1] != values.shape[-2]:
        raise ValueError("matrices must end in square matrix dimensions")
    values = _symmetrize(values)
    scale = np.abs(np.diagonal(values, axis1=-2, axis2=-1)).mean(axis=-1)
    fallback = np.linalg.norm(values, axis=(-2, -1)) / math.sqrt(values.shape[-1])
    scale = np.where(scale > np.finfo(np.float64).tiny, scale, fallback)
    scale = np.maximum(scale, np.finfo(np.float64).tiny)
    eigenvalues, eigenvectors = np.linalg.eigh(values / scale[..., None, None])
    logged = np.log(np.maximum(eigenvalues, eps)) + np.log(scale)[..., None]
    result = (eigenvectors * logged[..., None, :]) @ np.swapaxes(eigenvectors, -1, -2)
    return _symmetrize(result)


def session_log_references(
    covariances: np.ndarray, session_ids: Sequence[str]
) -> np.ndarray:
    """Return one label-free log-Euclidean reference per source session/sample."""

    cov = np.asarray(covariances)
    groups = np.asarray(session_ids, dtype=np.str_)
    if len(cov) != len(groups):
        raise ValueError("covariances and session IDs have different lengths")
    references = np.empty_like(cov, dtype=np.float32)
    for group in np.unique(groups):
        mask = groups == group
        references[mask] = relative_matrix_log(cov[mask]).mean(axis=0).astype(np.float32)
    return references


def calibration_partition(
    data: SessionData, indices: Sequence[int], count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split one target recording into an unlabeled prefix and later score rows."""

    rows = np.asarray(indices, dtype=np.int64)
    if len(rows) <= count:
        raise ValueError("target recording is too short for the requested calibration prefix")
    ordered = rows[np.argsort(data.event_samples[rows], kind="stable")]
    calibration = ordered[:count]
    calibration_set = set(calibration.tolist())
    scoring = np.asarray(
        [row for row in ordered if row not in calibration_set and data.labels[row] >= 0],
        dtype=np.int64,
    )
    if len(scoring) == 0 or np.unique(data.labels[scoring]).size != 2:
        raise ValueError("post-calibration scoring rows must contain both MI classes")
    return calibration, scoring


def _temperature_from_probabilities(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Fit one scalar temperature on the inner validation recording only."""

    y = np.asarray(labels, dtype=np.float64)
    p = np.clip(np.asarray(probabilities, dtype=np.float64)[:, 1], 1e-6, 1.0 - 1e-6)
    log_odds = np.log(p) - np.log1p(-p)

    def objective(log_temperature: float) -> float:
        temperature = math.exp(log_temperature)
        scaled = np.clip(log_odds / temperature, -40.0, 40.0)
        probability = 1.0 / (1.0 + np.exp(-scaled))
        return float(
            -np.mean(y * np.log(probability + 1e-12) + (1.0 - y) * np.log1p(-probability + 1e-12))
        )

    fit = minimize_scalar(objective, bounds=(math.log(0.25), math.log(4.0)), method="bounded")
    return float(math.exp(fit.x)) if fit.success else 1.0


def _centered_probabilities(probabilities: np.ndarray, center: float) -> np.ndarray:
    positive = np.clip(np.asarray(probabilities, dtype=np.float64)[:, 1], 1e-7, 1.0 - 1e-7)
    logits = np.log(positive) - np.log1p(-positive) - center
    p1 = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
    return np.column_stack((1.0 - p1, p1))


def _unlabeled_logit_center(probabilities: np.ndarray) -> float:
    positive = np.clip(
        np.asarray(probabilities, dtype=np.float64)[:, 1], 1e-7, 1.0 - 1e-7
    )
    return float(np.median(np.log(positive) - np.log1p(-positive)))


def _select_boundary_blend(
    labels: np.ndarray,
    score_probabilities: np.ndarray,
    calibration_probabilities: np.ndarray,
) -> tuple[float, float]:
    """Choose target-median strength on the protected validation recording.

    A blend of zero keeps the source model's logit threshold; one uses the full
    unlabeled target-prefix median.  Only this small predeclared grid is searched,
    and the chosen value is transferred unchanged to the outer test session.
    """

    target_center = _unlabeled_logit_center(calibration_probabilities)
    candidates = (0.0, 0.25, 0.5, 0.75, 1.0)
    scored: list[tuple[float, float]] = []
    for blend in candidates:
        probabilities = _centered_probabilities(score_probabilities, blend * target_center)
        balanced = float(balanced_accuracy_score(labels, probabilities.argmax(axis=1)))
        scored.append((balanced, blend))
    # Prefer less target adaptation on an exact validation tie.
    best_balanced, best_blend = max(scored, key=lambda item: (item[0], -item[1]))
    return best_blend, best_balanced


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix=f".{path.stem}-", suffix=".json",
            dir=path.parent, delete=False
        ) as temporary:
            temporary_name = temporary.name
            json.dump(payload, temporary, indent=2, sort_keys=True, allow_nan=False)
            temporary.write("\n")
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _finite_or_none(value: Any) -> float | int | None:
    if isinstance(value, (int, np.integer)):
        return int(value)
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _environment() -> dict[str, Any]:
    gpu = None
    if torch.cuda.is_available():
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "cuda": torch.version.cuda,
        }
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "gpu": gpu,
        "accelerators": {
            "cuda_available": bool(torch.cuda.is_available()),
            "mps_built": bool(torch.backends.mps.is_built()),
            "mps_available": bool(torch.backends.mps.is_available()),
        },
    }


def _repository_state() -> dict[str, Any]:
    """Record the exact code revision and dirty paths used for an experiment."""

    def git(*arguments: str) -> str | None:
        completed = subprocess.run(
            ("git", "-C", str(PROJECT_ROOT), *arguments),
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    return {
        "commit": commit,
        "dirty": bool(status),
        "status": [] if not status else status.splitlines(),
    }


def _fold_rows(data: SessionData, subject: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    runs = sorted(np.unique(data.run_ids[data.subject_ids == subject]).tolist())
    expected = list(SUBJECT_RUNS[subject])
    if runs != expected:
        raise ValueError(f"subject {subject} has runs {runs}, expected {expected}")
    inner_train = np.flatnonzero(
        (data.subject_ids == subject) & np.isin(data.run_ids, runs[:2])
    )
    validation = np.flatnonzero(
        (data.subject_ids == subject) & (data.run_ids == runs[2])
    )
    outer_test = np.flatnonzero(
        (data.subject_ids == subject) & (data.run_ids == runs[3])
    )
    return inner_train, validation, outer_test


def _training_config(config: BenchmarkConfig, seed: int) -> TrainConfig:
    return TrainConfig(
        epochs=config.max_epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        patience=config.patience,
        seed=seed,
        device=config.device,
        label_smoothing=config.label_smoothing,
        deep_supervision_weight=config.deep_supervision_weight,
        exclude_gate_from_weight_decay=config.exclude_gate_from_weight_decay,
        select_metric=config.select_metric,
        lr_swap_prob=config.lr_swap_prob,
        mixup_alpha=config.mixup_alpha,
        deterministic=config.deterministic,
    )


def _build_geoadapt(config: BenchmarkConfig) -> GeoAdaptNet:
    """Construct GeoAdaptNet with the run's accuracy knobs (defaults = baseline)."""

    return GeoAdaptNet(
        auxiliary_intent=config.include_rest,
        residual_gate_init=config.residual_gate_init,
    )


def _score_geoadapt(
    data: SessionData,
    subject: int,
    seed: int,
    config: BenchmarkConfig,
    *,
    input_contract: dict[str, Any],
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    inner_rows, validation_rows, test_rows = _fold_rows(data, subject)
    validation_cal, validation_score = calibration_partition(
        data, validation_rows, config.calibration_windows
    )
    source_refs = session_log_references(
        data.covariances[inner_rows], data.session_ids[inner_rows]
    )
    validation_ref = relative_matrix_log(data.covariances[validation_cal]).mean(axis=0).astype(np.float32)
    validation_keep = np.setdiff1d(validation_rows, validation_cal, assume_unique=False)

    train_config = _training_config(config, seed)
    set_reproducible_seed(seed, deterministic=config.deterministic)
    selection_model = _build_geoadapt(config)
    if config.csp_warm_start:
        # Leakage guard: CSP may only see the inner training recordings.  This
        # mirrors exactly the rows the EA-FBCSP baseline fits on for this fold.
        csp_rows = np.asarray(inner_rows)
        forbidden = np.union1d(np.asarray(validation_rows), np.asarray(test_rows))
        if np.intersect1d(csp_rows, forbidden).size:
            raise AssertionError("CSP warm-start rows leaked into validation/test")
        selection_model.init_bimaps_from_csp(
            data.covariances[csp_rows], data.labels[csp_rows]
        )
    selected = train_model(
        selection_model,
        CovarianceDataset(
            data.covariances[inner_rows], data.labels[inner_rows], log_references=source_refs
        ),
        CovarianceDataset(
            data.covariances[validation_keep],
            data.labels[validation_keep],
            log_references=validation_ref,
        ),
        train_config,
    )

    validation_probabilities, _ = predict_proba(
        selected.model,
        CovarianceDataset(
            data.covariances[validation_score],
            data.labels[validation_score],
            log_references=validation_ref,
        ),
        device=config.device,
    )
    validation_calibration_probabilities, _ = predict_proba(
        selected.model,
        CovarianceDataset(
            data.covariances[validation_cal],
            data.labels[validation_cal],
            log_references=validation_ref,
        ),
        device=config.device,
    )
    validation_anchor_probabilities, _ = predict_proba(
        selected.model,
        CovarianceDataset(
            data.covariances[validation_score],
            data.labels[validation_score],
            log_references=validation_ref,
        ),
        device=config.device,
        branch="anchor",
    )
    validation_anchor_calibration, _ = predict_proba(
        selected.model,
        CovarianceDataset(
            data.covariances[validation_cal],
            data.labels[validation_cal],
            log_references=validation_ref,
        ),
        device=config.device,
        branch="anchor",
    )
    boundary_blend, blended_validation_balanced = _select_boundary_blend(
        data.labels[validation_score],
        validation_probabilities,
        validation_calibration_probabilities,
    )
    anchor_boundary_blend, _ = _select_boundary_blend(
        data.labels[validation_score],
        validation_anchor_probabilities,
        validation_anchor_calibration,
    )
    temperature = _temperature_from_probabilities(
        data.labels[validation_score],
        _centered_probabilities(
            validation_probabilities,
            boundary_blend * _unlabeled_logit_center(
                validation_calibration_probabilities
            ),
        ),
    )
    anchor_temperature = _temperature_from_probabilities(
        data.labels[validation_score],
        _centered_probabilities(
            validation_anchor_probabilities,
            anchor_boundary_blend
            * _unlabeled_logit_center(validation_anchor_calibration),
        ),
    )

    # Temperature scaling is specific to a model's logit scale.  Keep the exact
    # checkpoint evaluated on the protected validation recording; transferring
    # its temperature to a separately refit model would not be calibrated.
    final_model = selected.model

    test_cal, test_score = calibration_partition(data, test_rows, config.calibration_windows)
    covariance_adapter = LogEuclideanCovarianceRecenter(
        alpha=0.001,
        robust_clip=1.0,
        eig_floor=1e-15,
        alignment_enabled=True,
        adaptation_enabled=True,
    )
    boundary_adapter = BoundaryRecenter(
        alpha=0.01,
        clamp=2.0,
        rest_confidence=0.65,
        commit_threshold=config.commit_confidence,
        temperature=temperature,
        class1_label=0,
        class2_label=1,
    )
    adapter = DualLevelOnlineAdapter(
        covariance_adapter, boundary_adapter, covariance_updates_rest_only=True
    )
    covariance_adapter.calibrate(data.covariances[test_cal])
    aligned_calibration = covariance_adapter.transform(data.covariances[test_cal])
    zero_reference = np.zeros(aligned_calibration.shape[1:], dtype=np.float32)
    swap_index = np.asarray(left_right_swap_index(CHANNELS), dtype=np.int64)
    if config.tta_mirror:
        calibration_probabilities, calibration_intent = _tta_mirror_probabilities(
            final_model,
            aligned_calibration,
            data.labels[test_cal],
            zero_reference,
            swap_index,
            device=config.device,
            batch_size=256,
        )
    else:
        calibration_probabilities, calibration_intent = predict_proba(
            final_model,
            CovarianceDataset(
                aligned_calibration,
                data.labels[test_cal],
                log_references=zero_reference,
            ),
            device=config.device,
        )
    calibration_anchor_probabilities, _ = predict_proba(
        final_model,
        CovarianceDataset(
            aligned_calibration,
            data.labels[test_cal],
            log_references=zero_reference,
        ),
        device=config.device,
        branch="anchor",
    )
    calibration_positive = np.clip(calibration_probabilities[:, 1], 1e-7, 1.0 - 1e-7)
    calibration_scores = np.log(calibration_positive) - np.log1p(-calibration_positive)
    # Use the complete unlabeled prefix for the scalar neutral.  Selecting only
    # examples that the learned intent head already considers easy can create a
    # class-asymmetric threshold and is therefore deliberately avoided.
    boundary_adapter.calibrate(calibration_scores)
    boundary_adapter.center *= boundary_blend
    boundary_adapter.seed_center = boundary_adapter.center
    anchor_adapter = DualLevelOnlineAdapter.from_state_dict(adapter.state_dict())
    anchor_adapter.boundary.temperature = anchor_temperature
    anchor_positive = np.clip(
        calibration_anchor_probabilities[:, 1], 1e-7, 1.0 - 1e-7
    )
    anchor_scores = np.log(anchor_positive) - np.log1p(-anchor_positive)
    anchor_adapter.boundary.calibrate(anchor_scores)
    anchor_adapter.boundary.center *= anchor_boundary_blend
    anchor_adapter.boundary.seed_center = anchor_adapter.boundary.center

    calibration_set = set(test_cal.tolist())
    ordered_live = test_rows[np.argsort(data.event_samples[test_rows], kind="stable")]
    ordered_live = np.asarray(
        [row for row in ordered_live if row not in calibration_set], dtype=np.int64
    )
    final_model.eval()

    def stream_score(
        stream_adapter: DualLevelOnlineAdapter, *, update: bool, branch: str = "full"
    ) -> tuple[dict[str, Any], int, int, list[dict[str, Any]]]:
        probabilities: list[tuple[float, float]] = []
        task_truth: list[int] = []
        committed: list[bool] = []
        prediction_trace: list[dict[str, Any]] = []
        rest_windows = 0
        rest_false_commits = 0
        covariance_updates = 0
        boundary_updates = 0
        for row in ordered_live:
            raw_covariance = data.covariances[row]
            aligned = stream_adapter.covariance.transform(raw_covariance)
            if config.tta_mirror:
                window_probabilities, intent_probability = _tta_mirror_probabilities(
                    final_model,
                    aligned[np.newaxis],
                    np.asarray([data.labels[row]]),
                    zero_reference,
                    swap_index,
                    device=config.device,
                    batch_size=1,
                    branch=branch,
                )
            else:
                window_probabilities, intent_probability = predict_proba(
                    final_model,
                    CovarianceDataset(
                        aligned[np.newaxis],
                        np.asarray([data.labels[row]]),
                        log_references=zero_reference,
                    ),
                    device=config.device,
                    batch_size=1,
                    branch=branch,
                )
            positive = float(np.clip(window_probabilities[0, 1], 1e-7, 1.0 - 1e-7))
            score = math.log(positive) - math.log1p(-positive)
            rest_probability = (
                None if intent_probability is None else 1.0 - float(intent_probability[0])
            )
            decision = stream_adapter.step(
                raw_covariance,
                score=score,
                rest_probability=rest_probability,
                update=update,
            )
            p1 = 1.0 / (
                1.0 + math.exp(-np.clip(decision.margin / temperature, -40.0, 40.0))
            )
            intent_gate = rest_probability is None or rest_probability < 0.5
            effective_commit = bool(decision.committed and intent_gate)
            label = int(data.labels[row])
            prediction_trace.append(
                {
                    "row_index": int(row),
                    "subject": int(data.subject_ids[row]),
                    "run": int(data.run_ids[row]),
                    "session_id": str(data.session_ids[row]),
                    "event_sample": int(data.event_samples[row]),
                    "event_onset_seconds": float(data.event_onsets[row]),
                    "trial_id": int(data.trial_ids[row]),
                    "annotation": str(data.annotations[row]),
                    "label": label,
                    "probability_left": float(1.0 - p1),
                    "probability_right": float(p1),
                    "intent_probability": (
                        None
                        if rest_probability is None
                        else float(1.0 - rest_probability)
                    ),
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
            )
            if label >= 0:
                probabilities.append((1.0 - p1, p1))
                task_truth.append(label)
                committed.append(effective_commit)
            else:
                rest_windows += 1
                rest_false_commits += int(effective_commit)
            covariance_updates += int(decision.covariance_updated)
            boundary_updates += int(decision.boundary_updated)

        probability_array = np.asarray(probabilities)
        metric_values = classification_metrics(
            np.asarray(task_truth), probability_array,
            commit_confidence=config.commit_confidence,
        ).to_dict()
        metric_values["coverage"] = float(np.mean(committed))
        metric_values["rest_false_commit_rate"] = (
            float(rest_false_commits / rest_windows) if rest_windows else None
        )
        if np.any(committed):
            predictions = probability_array.argmax(axis=1)
            metric_values["selective_accuracy"] = float(
                np.mean(
                    predictions[np.asarray(committed)]
                    == np.asarray(task_truth)[np.asarray(committed)]
                )
            )
        metric_values = {
            key: None if value is None else _finite_or_none(value)
            for key, value in metric_values.items()
        }
        return metric_values, covariance_updates, boundary_updates, prediction_trace

    frozen_adapter = DualLevelOnlineAdapter.from_state_dict(adapter.state_dict())
    frozen_metrics, _, _, frozen_predictions = stream_score(
        frozen_adapter, update=False
    )
    metrics, covariance_updates, boundary_updates, predictions = stream_score(
        adapter, update=True
    )
    anchor_metrics, _, _, anchor_predictions = stream_score(
        anchor_adapter, update=True, branch="anchor"
    )
    scored_tasks = sum(int(item["label"] >= 0) for item in predictions)
    latency_covariance = covariance_adapter.transform(data.covariances[ordered_live[0]])
    latency = batch_one_latency_ms(
        final_model,
        latency_covariance,
        log_reference=zero_reference,
        device=config.device,
        warmup=10,
        repetitions=50,
    )
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "schema_version": SCHEMA_VERSION,
            "model": "GeoAdaptNet",
            "model_config": {"auxiliary_intent": config.include_rest},
            "state_dict": {
                key: value.detach().cpu() for key, value in final_model.state_dict().items()
            },
            "subject": subject,
            "seed": seed,
            "selected_epochs": selected.best_epoch + 1,
            "temperature": temperature,
            "target_median_blend": boundary_blend,
            "benchmark_config": asdict(config),
            "data_contract": input_contract,
        }
        temporary = checkpoint_path.with_name(f".{checkpoint_path.name}.tmp")
        torch.save(checkpoint, temporary)
        os.replace(temporary, checkpoint_path)
    return {
        "model": "geoadapt",
        "subject": subject,
        "seed": seed,
        "metrics": metrics,
        "predictions": predictions,
        "ablations": {
            "frozen_target_recenter": frozen_metrics,
            "anchor_only_logits": anchor_metrics,
        },
        "ablation_predictions": {
            "frozen_target_recenter": frozen_predictions,
            "anchor_only_logits": anchor_predictions,
        },
        "selection": {
            "best_epoch_zero_based": selected.best_epoch,
            "epochs_ran": selected.epochs_ran,
            "validation_loss": selected.best_validation_loss,
            "validation_balanced_accuracy": selected.best_validation_balanced_accuracy,
            "temperature": temperature,
            "anchor_temperature": anchor_temperature,
            "target_median_blend": boundary_blend,
            "anchor_target_median_blend": anchor_boundary_blend,
            "blended_validation_balanced_accuracy": blended_validation_balanced,
            "selection_train_seconds": selected.train_seconds,
            "final_model": "protected_validation_checkpoint_no_refit",
        },
        "deployment": {
            "calibration_windows": len(test_cal),
            "scored_task_windows": scored_tasks,
            "observed_post_calibration_windows": len(ordered_live),
            "covariance_updates": covariance_updates,
            "boundary_updates": boundary_updates,
            "batch_one_model_latency_ms": latency,
            "parameters": final_model.parameter_count,
            "residual_gate": float(final_model.residual_gate.detach().cpu()),
            "checkpoint": (
                None
                if checkpoint_path is None
                else str(checkpoint_path.relative_to(PROJECT_ROOT))
            ),
        },
    }


def _score_baseline(
    name: str,
    data: SessionData,
    subject: int,
    seed: int,
    config: BenchmarkConfig,
) -> dict[str, Any]:
    del seed  # classical estimators are fixed to deterministic internal seeds
    inner_rows, validation_rows, test_rows = _fold_rows(data, subject)
    source_rows = inner_rows[data.labels[inner_rows] >= 0]
    validation_cal, validation_score = calibration_partition(
        data, validation_rows, config.calibration_windows
    )
    test_cal, test_score = calibration_partition(data, test_rows, config.calibration_windows)
    calibration_set = set(test_cal.tolist())
    live_rows = test_rows[np.argsort(data.event_samples[test_rows], kind="stable")]
    live_rows = np.asarray([row for row in live_rows if row not in calibration_set], dtype=np.int64)
    groups = data.session_ids[source_rows]
    started = time.perf_counter()
    if name == "riemann":
        estimator = RiemannianTangentLogistic().fit(
            data.covariances[source_rows], data.labels[source_rows], groups=groups
        )
        features = data.covariances
    elif name == "fbcsp":
        estimator = EAFilterBankCSP(n_components=2).fit(
            data.epochs[source_rows], data.labels[source_rows], groups=groups
        )
        features = data.epochs
    else:
        raise ValueError(f"unknown baseline {name!r}")

    estimator.calibrate(features[validation_cal])
    validation_calibration = estimator.predict_proba(features[validation_cal])
    validation_probabilities = estimator.predict_proba(features[validation_score])
    boundary_blend, blended_validation_balanced = _select_boundary_blend(
        data.labels[validation_score],
        validation_probabilities,
        validation_calibration,
    )
    temperature = _temperature_from_probabilities(
        data.labels[validation_score],
        _centered_probabilities(
            validation_probabilities,
            boundary_blend * _unlabeled_logit_center(validation_calibration),
        ),
    )

    estimator.calibrate(features[test_cal])
    calibration_probabilities = estimator.predict_proba(features[test_cal])
    live_probabilities = estimator.predict_proba(features[live_rows])
    elapsed = time.perf_counter() - started
    center = boundary_blend * _unlabeled_logit_center(calibration_probabilities)
    live_positive = np.clip(live_probabilities[:, 1], 1e-7, 1.0 - 1e-7)
    live_margins = np.log(live_positive) - np.log1p(-live_positive) - center
    calibrated_positive = 1.0 / (
        1.0 + np.exp(-np.clip(live_margins / temperature, -40.0, 40.0))
    )
    live_probabilities = np.column_stack(
        (1.0 - calibrated_positive, calibrated_positive)
    )
    task_mask = data.labels[live_rows] >= 0
    probabilities = live_probabilities[task_mask]
    task_labels = data.labels[live_rows][task_mask]
    metrics = classification_metrics(
        task_labels, probabilities,
        commit_confidence=config.commit_confidence,
    ).to_dict()
    rest_mask = ~task_mask
    metrics["rest_false_commit_rate"] = (
        float(np.mean(live_probabilities[rest_mask].max(axis=1) >= config.commit_confidence))
        if np.any(rest_mask)
        else None
    )
    metrics = {
        key: None if value is None else _finite_or_none(value)
        for key, value in metrics.items()
    }
    predictions: list[dict[str, Any]] = []
    for position, row in enumerate(live_rows):
        probability = live_probabilities[position]
        confidence = float(np.max(probability))
        predictions.append(
            {
                "row_index": int(row),
                "subject": int(data.subject_ids[row]),
                "run": int(data.run_ids[row]),
                "session_id": str(data.session_ids[row]),
                "event_sample": int(data.event_samples[row]),
                "event_onset_seconds": float(data.event_onsets[row]),
                "trial_id": int(data.trial_ids[row]),
                "annotation": str(data.annotations[row]),
                "label": int(data.labels[row]),
                "probability_left": float(probability[0]),
                "probability_right": float(probability[1]),
                "intent_probability": None,
                "rest_probability": None,
                "prediction": int(np.argmax(probability)),
                "confidence": confidence,
                "committed": bool(confidence >= config.commit_confidence),
                "margin": float(live_margins[position]),
                "center_before": float(center),
                "center_after": float(center),
                "covariance_updated": False,
                "boundary_updated": False,
            }
        )
    return {
        "model": name,
        "subject": subject,
        "seed": 0,
        "metrics": metrics,
        "predictions": predictions,
        "selection": {
            "temperature": temperature,
            "target_median_blend": boundary_blend,
            "blended_validation_balanced_accuracy": blended_validation_balanced,
            "training_sessions": sorted(np.unique(data.session_ids[source_rows]).tolist()),
            "validation_session": str(np.unique(data.session_ids[validation_rows])[0]),
        },
        "deployment": {
            "calibration_windows": len(test_cal),
            "scored_task_windows": len(test_score),
            "observed_post_calibration_windows": len(live_rows),
            "fit_and_inference_seconds": elapsed,
        },
    }


def _summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for model in sorted({str(row["model"]) for row in rows}):
        selected = [row for row in rows if row["model"] == model]
        subjects = sorted({int(row["subject"]) for row in selected})
        model_summary: dict[str, Any] = {
            "runs": len(selected),
            "participants": len(subjects),
        }
        for metric in (
            "accuracy", "balanced_accuracy", "kappa", "roc_auc", "brier",
            "ece", "coverage", "selective_accuracy",
            "rest_false_commit_rate",
        ):
            # Neural seeds are repeated fits, not additional participants.  First
            # average seeds within each person, then summarize across people.
            participant_values: list[float] = []
            for subject in subjects:
                values = np.asarray(
                    [
                        row["metrics"][metric]
                        for row in selected
                        if int(row["subject"]) == subject
                        and row["metrics"][metric] is not None
                    ],
                    dtype=np.float64,
                )
                finite_values = values[np.isfinite(values)]
                if len(finite_values):
                    participant_values.append(float(finite_values.mean()))
            finite = np.asarray(participant_values, dtype=np.float64)
            model_summary[metric] = {
                "mean": float(finite.mean()) if len(finite) else None,
                "std": float(finite.std(ddof=1)) if len(finite) > 1 else 0.0 if len(finite) else None,
            }
        result[model] = model_summary
    return result


def run_benchmark(
    *,
    subjects: Sequence[int],
    models: Sequence[str],
    seeds: Sequence[int],
    config: BenchmarkConfig,
    output: Path,
    resume: bool = True,
) -> dict[str, Any]:
    """Run and atomically checkpoint the chronological cross-session benchmark."""

    unknown = set(models) - set(DEFAULT_MODELS)
    if unknown:
        raise ValueError(f"unknown models: {sorted(unknown)}")
    if not subjects or not models or not seeds:
        raise ValueError("subjects, models, and seeds must be non-empty")
    invalid_subjects = set(subjects) - set(SUBJECT_RUNS)
    if invalid_subjects:
        raise ValueError(f"invalid subjects: {sorted(invalid_subjects)}")
    keys = [SessionKey(subject, run) for subject in subjects for run in SUBJECT_RUNS[subject]]
    data_config = replace(
        DEFAULT_DATA_CONFIG,
        window_name=config.window,
        include_rest=config.include_rest,
        covariance_shrinkage_method=config.covariance_shrinkage_method,
    )
    data = load_sessions(keys, data_config)
    input_contract = dataset_contract(data, data_config)
    payload: dict[str, Any]
    if resume and output.is_file():
        payload = json.loads(output.read_text(encoding="utf-8"))
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("cannot resume an incompatible result schema")
        if payload.get("benchmark_config") != asdict(config):
            raise ValueError("cannot resume with a different benchmark configuration")
        if payload.get("data_contract") != input_contract:
            raise ValueError("cannot resume with a different preprocessing/data contract")
        if payload.get("subjects") != list(subjects):
            raise ValueError("cannot resume with a different subject list")
        if payload.get("models") != list(models):
            raise ValueError("cannot resume with a different model list")
        if payload.get("seeds") != list(seeds):
            raise ValueError("cannot resume with a different seed list")
    else:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "protocol": "chronological_train-2_validate_calibrate-1_test-1",
            "benchmark_config": asdict(config),
            "data_contract": input_contract,
            "subjects": list(subjects),
            "models": list(models),
            "seeds": list(seeds),
            "environment": _environment(),
            "repository": _repository_state(),
            "command": list(sys.argv),
            "folds": [],
            "summary": {},
        }
    rows: list[dict[str, Any]] = payload["folds"]
    completed = {(row["model"], int(row["subject"]), int(row["seed"])) for row in rows}
    for subject in subjects:
        for model in models:
            model_seeds = seeds if model == "geoadapt" else (0,)
            for seed in model_seeds:
                key = (model, int(subject), int(seed))
                if key in completed:
                    print(f"skip complete {model} subject={subject} seed={seed}", flush=True)
                    continue
                print(f"run {model} subject={subject} seed={seed}", flush=True)
                row = (
                    _score_geoadapt(
                        data,
                        subject,
                        seed,
                        config,
                        input_contract=input_contract,
                        checkpoint_path=(
                            PROJECT_ROOT
                            / "deepnet"
                            / "checkpoints"
                            / output.stem
                            / f"subject-{subject:02d}_seed-{seed}.pt"
                        ),
                    )
                    if model == "geoadapt"
                    else _score_baseline(model, data, subject, seed, config)
                )
                rows.append(row)
                completed.add(key)
                payload["updated_at"] = datetime.now(timezone.utc).isoformat()
                payload["summary"] = _summary(rows)
                _atomic_json(output, payload)
                balanced = row["metrics"]["balanced_accuracy"]
                print(f"done {model} subject={subject}: balanced_accuracy={balanced:.3f}", flush=True)
    return payload


def _csv_ints(value: str, *, all_values: Iterable[int] | None = None) -> tuple[int, ...]:
    if value.strip().lower() == "all":
        if all_values is None:
            raise argparse.ArgumentTypeError("'all' is not valid here")
        return tuple(all_values)
    try:
        return tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error


def _csv_models(value: str) -> tuple[str, ...]:
    models = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    unknown = set(models) - set(DEFAULT_MODELS)
    if not models or unknown:
        raise argparse.ArgumentTypeError(f"models must be selected from {DEFAULT_MODELS}")
    return models


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="preprocess and cache the valid cohort")
    prepare.add_argument("--window", choices=tuple(EPOCH_WINDOWS), default="deployment")
    prepare.add_argument("--task-only", action="store_true")

    benchmark = subparsers.add_parser("benchmark", help="run chronological cross-session folds")
    benchmark.add_argument("--subjects", default="all")
    benchmark.add_argument("--models", type=_csv_models, default=DEFAULT_MODELS)
    benchmark.add_argument("--seeds", default="7")
    benchmark.add_argument("--window", choices=tuple(EPOCH_WINDOWS), default="deployment")
    benchmark.add_argument("--task-only", action="store_true")
    benchmark.add_argument("--calibration-windows", type=int, default=20)
    benchmark.add_argument("--max-epochs", type=int, default=180)
    benchmark.add_argument("--patience", type=int, default=25)
    benchmark.add_argument("--batch-size", type=int, default=64)
    benchmark.add_argument("--learning-rate", type=float, default=3e-4)
    benchmark.add_argument("--weight-decay", type=float, default=1e-3)
    benchmark.add_argument("--device", default="auto")
    benchmark.add_argument("--commit-confidence", type=float, default=0.85)
    # GeoAdaptNet accuracy knobs (defaults reproduce the locked baseline).
    benchmark.add_argument("--deep-supervision-weight", type=float, default=0.0)
    benchmark.add_argument("--residual-gate-init", type=float, default=0.02)
    benchmark.add_argument("--exclude-gate-weight-decay", action="store_true")
    benchmark.add_argument(
        "--select-metric", choices=("loss", "balanced_accuracy", "blend"), default="loss"
    )
    benchmark.add_argument("--label-smoothing", type=float, default=0.05)
    benchmark.add_argument("--csp-warm-start", action="store_true")
    benchmark.add_argument("--lr-swap-prob", type=float, default=0.0)
    benchmark.add_argument("--mixup-alpha", type=float, default=0.0)
    benchmark.add_argument(
        "--covariance-shrinkage-method", choices=("fixed", "oas"), default="fixed"
    )
    benchmark.add_argument(
        "--fast",
        action="store_true",
        help="faster iteration: non-deterministic kernels + TF32 matmul (accuracy ~unchanged)",
    )
    benchmark.add_argument(
        "--tta-mirror",
        action="store_true",
        help="test-time left/right mirror averaging at inference (2x inference cost)",
    )
    benchmark.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "deepnet" / "results" / "chronological.json",
    )
    benchmark.add_argument("--no-resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        config = replace(
            DEFAULT_DATA_CONFIG, window_name=args.window, include_rest=not args.task_only
        )
        keys = [SessionKey(subject, run) for subject, runs in SUBJECT_RUNS.items() for run in runs]
        print(f"preparing {len(keys)} sessions in {config.cache_dir}", flush=True)
        load_sessions(keys, config)
        return 0

    subjects = _csv_ints(args.subjects, all_values=SUBJECT_RUNS)
    seeds = _csv_ints(args.seeds)
    # The fast profile drops determinism and enables TF32.  It deliberately does
    # NOT change the batch size: at this data scale a larger batch measurably
    # hurts accuracy, so batch stays a separate, explicit knob.
    config = BenchmarkConfig(
        window=args.window,
        include_rest=not args.task_only,
        calibration_windows=args.calibration_windows,
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=args.device,
        commit_confidence=args.commit_confidence,
        deep_supervision_weight=args.deep_supervision_weight,
        residual_gate_init=args.residual_gate_init,
        exclude_gate_from_weight_decay=args.exclude_gate_weight_decay,
        select_metric=args.select_metric,
        label_smoothing=args.label_smoothing,
        csp_warm_start=args.csp_warm_start,
        lr_swap_prob=args.lr_swap_prob,
        mixup_alpha=args.mixup_alpha,
        covariance_shrinkage_method=args.covariance_shrinkage_method,
        deterministic=not args.fast,
        tta_mirror=args.tta_mirror,
    )
    payload = run_benchmark(
        subjects=subjects,
        models=args.models,
        seeds=seeds,
        config=config,
        output=args.output,
        resume=not args.no_resume,
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
