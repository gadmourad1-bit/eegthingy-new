"""Strict outer-LOSO evaluation for the two classical EEG baselines.

Every outer participant and protected inner-validation participant are absent
from estimator fitting.  The estimator is fit once on task epochs from the six
selection participants, with each source recording treated as a separate
alignment domain.  Temperature and target-median blend are selected on the four
inner-validation recordings and then locked before the outer recordings are
opened.  Each held-out recording is handled independently: every window before
the eleventh task event forms an unlabeled calibration prefix, the estimator's
target alignment is reset from that prefix, and only the subsequent
chronological stream is scored.

The baselines are binary left/right classifiers and do *not* have GeoAdaptNet's
intent/rest head.  Consequently a baseline commits whenever its maximum binary
class probability reaches ``commit_confidence``.  A high-confidence left/right
prediction during rest is therefore counted as a false commit without an intent
gate.  This is deliberately explicit in every result row and in the artifact's
limitations.

The prefix boundary uses task-versus-rest phase identity only to count exactly
ten task events.  Left/right labels in the prefix are never supplied to fitting,
alignment, boundary centering, or hyperparameter selection.  Classical
estimator hyperparameters and the temperature/blend candidate grids are fixed in
advance.  Only scored inner-validation labels select among those candidates;
outer labels never influence fitting, calibration, or selection.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .baselines import EAFilterBankCSP, ProbabilisticBaseline, RiemannianTangentLogistic
from .config import (
    DEFAULT_DATA_CONFIG,
    EPOCH_WINDOWS,
    PROJECT_ROOT,
    SUBJECT_RUNS,
    VALID_SUBJECTS,
)
from .data import SessionData, SessionKey, dataset_contract, load_sessions
from .experiment import (
    _atomic_json,
    _environment,
    _finite_or_none,
    _repository_state,
    _summary,
)
from .loso import (
    COHORT_LIMITATION,
    PROTOCOL_NAME,
    NestedLOSOFold,
    SessionCalibrationPartition,
    nested_loso_fold,
    session_calibration_partitions,
)
from .metrics import classification_metrics


SCHEMA_VERSION = 2
ARTIFACT_KIND = "strict_outer_loso_classical_baselines"
BASELINE_MODELS = ("riemann", "fbcsp")
CALIBRATION_TASK_EVENTS = 10
NO_INTENT_GATE_LIMITATION = (
    "Riemannian and EA-FBCSP baselines have no intent/rest head: task coverage "
    "and rest false commits use only maximum left/right probability at the fixed "
    "commit threshold, with no intent gate."
)
CALIBRATION_LIMITATION = (
    "Each target recording uses a separate unlabeled prefix ending immediately "
    "before its eleventh task event. Prefix direction labels are never used, and "
    "all prefix events are excluded from scoring."
)
BASELINE_NESTED_LIMITATION = (
    "One deterministic inner-validation participant is used per outer fold, "
    "rather than averaging calibration selection over a complete inner LOSO; "
    "its four recordings jointly select temperature and target-median blend "
    "while remaining excluded from estimator fitting."
)


@dataclass(frozen=True)
class BaselineLOSOConfig:
    """Predeclared settings for the deterministic strict-outer baselines."""

    window: str = "deployment"
    include_rest: bool = True
    calibration_task_events: int = CALIBRATION_TASK_EVENTS
    commit_confidence: float = 0.85
    fbcsp_components: int = 2
    temperature_grid: tuple[float, ...] = (0.5, 0.75, 1.0, 1.5, 2.0)
    target_median_blends: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "temperature_grid", tuple(self.temperature_grid))
        object.__setattr__(
            self, "target_median_blends", tuple(self.target_median_blends)
        )
        if self.window not in EPOCH_WINDOWS:
            raise ValueError(f"unknown window {self.window!r}")
        if self.calibration_task_events != CALIBRATION_TASK_EVENTS:
            raise ValueError(
                "strict LOSO baselines require exactly "
                f"{CALIBRATION_TASK_EVENTS} calibration task events per session"
            )
        if not 0.5 <= self.commit_confidence <= 1.0:
            raise ValueError("commit_confidence must be in [0.5, 1]")
        if self.fbcsp_components <= 0:
            raise ValueError("fbcsp_components must be positive")
        if not self.temperature_grid or any(
            value <= 0.0 for value in self.temperature_grid
        ):
            raise ValueError("temperature_grid must contain positive values")
        if len(set(self.temperature_grid)) != len(self.temperature_grid):
            raise ValueError("temperature_grid must not contain duplicates")
        if not self.target_median_blends or any(
            not 0.0 <= value <= 1.0 for value in self.target_median_blends
        ):
            raise ValueError("target_median_blends must lie in [0, 1]")
        if len(set(self.target_median_blends)) != len(self.target_median_blends):
            raise ValueError("target_median_blends must not contain duplicates")


def _json_config(config: BaselineLOSOConfig) -> dict[str, Any]:
    """Round-trip through JSON so resume comparisons use serialized types."""

    return json.loads(json.dumps(asdict(config), sort_keys=True))


def _new_estimator(name: str, config: BaselineLOSOConfig) -> ProbabilisticBaseline:
    if name == "riemann":
        return RiemannianTangentLogistic()
    if name == "fbcsp":
        return EAFilterBankCSP(n_components=config.fbcsp_components)
    raise ValueError(f"unknown baseline {name!r}")


def _features(name: str, data: SessionData | Any, rows: Sequence[int]) -> np.ndarray:
    indices = np.asarray(rows, dtype=np.int64)
    if name == "riemann":
        return np.asarray(data.covariances)[indices]
    if name == "fbcsp":
        return np.asarray(data.epochs)[indices]
    raise ValueError(f"unknown baseline {name!r}")


def _validate_binary_classes(estimator: ProbabilisticBaseline) -> None:
    classes = np.asarray(estimator.classes_)
    if not np.array_equal(classes, np.asarray([0, 1])):
        raise ValueError(f"baseline classes must be [0, 1], got {classes.tolist()}")


def _adapt_probabilities(
    probabilities: np.ndarray,
    *,
    center: float,
    temperature: float,
) -> np.ndarray:
    """Apply a locked boundary center and scalar temperature to binary logits."""

    positive = np.clip(
        np.asarray(probabilities, dtype=np.float64)[:, 1], 1e-7, 1.0 - 1e-7
    )
    logits = (np.log(positive) - np.log1p(-positive) - center) / temperature
    adapted_positive = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
    return np.column_stack((1.0 - adapted_positive, adapted_positive))


def _metrics(
    labels: Sequence[int],
    probabilities: np.ndarray,
    *,
    rest_windows: int,
    rest_false_commits: int,
    commit_confidence: float,
) -> dict[str, Any]:
    values = classification_metrics(
        np.asarray(labels, dtype=np.int64),
        np.asarray(probabilities, dtype=np.float64),
        commit_confidence=commit_confidence,
    ).to_dict()
    values["rest_false_commit_rate"] = (
        float(rest_false_commits / rest_windows) if rest_windows else None
    )
    return {
        key: None if value is None else _finite_or_none(value)
        for key, value in values.items()
    }


def evaluate_baseline_sessions(
    name: str,
    estimator: ProbabilisticBaseline,
    data: SessionData | Any,
    partitions: Sequence[SessionCalibrationPartition],
    *,
    temperature: float,
    target_median_blend: float,
    config: BaselineLOSOConfig,
) -> dict[str, Any]:
    """Score independent target recordings under fixed no-intent-gate semantics."""

    if name not in BASELINE_MODELS:
        raise ValueError(f"unknown baseline {name!r}")
    if not partitions:
        raise ValueError("at least one session partition is required")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if not 0.0 <= target_median_blend <= 1.0:
        raise ValueError("target_median_blend must be in [0, 1]")
    _validate_binary_classes(estimator)

    all_truth: list[int] = []
    all_probabilities: list[np.ndarray] = []
    total_rest = 0
    total_rest_false_commits = 0
    session_payloads: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []

    labels = np.asarray(data.labels, dtype=np.int64)
    for partition in partitions:
        calibration_features = _features(
            name, data, partition.calibration_indices
        )
        # ``calibrate`` replaces, rather than updates, the target whitener.  A
        # call inside this loop guarantees no target state crosses recordings.
        estimator.calibrate(calibration_features)
        calibration_probabilities = np.asarray(
            estimator.predict_proba(calibration_features), dtype=np.float64
        )
        positive = np.clip(calibration_probabilities[:, 1], 1e-7, 1.0 - 1e-7)
        prefix_center = float(np.median(np.log(positive) - np.log1p(-positive)))
        center = target_median_blend * prefix_center

        stream_rows = partition.stream_indices
        raw_probabilities = np.asarray(
            estimator.predict_proba(_features(name, data, stream_rows)),
            dtype=np.float64,
        )
        probabilities = _adapt_probabilities(
            raw_probabilities,
            center=center,
            temperature=temperature,
        )
        raw_positive = np.clip(raw_probabilities[:, 1], 1e-7, 1.0 - 1e-7)
        margins = np.log(raw_positive) - np.log1p(-raw_positive) - center
        predictions = probabilities.argmax(axis=1)
        confidences = probabilities.max(axis=1)
        committed = confidences >= config.commit_confidence
        session_predictions: list[dict[str, Any]] = []
        for position, row in enumerate(stream_rows):
            prediction_record = {
                "row_index": int(row),
                "subject": int(data.subject_ids[row]),
                "run": int(data.run_ids[row]),
                "session_id": str(data.session_ids[row]),
                "event_sample": int(data.event_samples[row]),
                "event_onset_seconds": float(data.event_onsets[row]),
                "trial_id": int(data.trial_ids[row]),
                "annotation": str(data.annotations[row]),
                "label": int(labels[row]),
                "raw_probability_left": float(raw_probabilities[position, 0]),
                "raw_probability_right": float(raw_probabilities[position, 1]),
                "probability_left": float(probabilities[position, 0]),
                "probability_right": float(probabilities[position, 1]),
                "intent_probability": None,
                "rest_probability": None,
                "prediction": int(predictions[position]),
                "confidence": float(confidences[position]),
                "committed": bool(committed[position]),
                "margin": float(margins[position]),
                "scaled_margin": float(margins[position] / temperature),
                "center_before": float(center),
                "center_after": float(center),
                "covariance_updated": False,
                "boundary_updated": False,
                "intent_gate_applied": False,
            }
            session_predictions.append(prediction_record)
        task_mask = labels[stream_rows] >= 0
        task_truth = labels[stream_rows][task_mask]
        task_probabilities = probabilities[task_mask]
        rest_committed = committed[~task_mask]
        rest_false_commits = int(np.sum(rest_committed))
        metrics = _metrics(
            task_truth,
            task_probabilities,
            rest_windows=len(rest_committed),
            rest_false_commits=rest_false_commits,
            commit_confidence=config.commit_confidence,
        )
        session_payloads.append(
            {
                "session_id": partition.session_id,
                "calibration_windows": len(partition.calibration_indices),
                "calibration_task_events": int(
                    np.sum(labels[partition.calibration_indices] >= 0)
                ),
                "observed_post_calibration_windows": len(stream_rows),
                "scored_task_windows": len(task_truth),
                "rest_windows": len(rest_committed),
                "rest_false_commits": rest_false_commits,
                "target_median_log_odds_center": _finite_or_none(center),
                "unscaled_prefix_median_log_odds": _finite_or_none(prefix_center),
                "temperature": _finite_or_none(temperature),
                "target_median_blend": _finite_or_none(target_median_blend),
                "intent_gate": False,
                "metrics": metrics,
                "predictions": session_predictions,
            }
        )
        all_truth.extend(task_truth.tolist())
        all_probabilities.extend(task_probabilities)
        total_rest += len(rest_committed)
        total_rest_false_commits += rest_false_commits
        all_predictions.extend(session_predictions)

    aggregate_probabilities = np.asarray(all_probabilities, dtype=np.float64)
    metrics = _metrics(
        all_truth,
        aggregate_probabilities,
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
            "calibration_task_events_per_session": CALIBRATION_TASK_EVENTS,
            "calibration_windows": int(
                sum(len(partition.calibration_indices) for partition in partitions)
            ),
            "scored_task_windows": len(all_truth),
            "rest_windows": total_rest,
            "rest_false_commits": total_rest_false_commits,
            "target_alignment_reset_per_session": True,
            "post_calibration_adaptation": False,
            "intent_gate": False,
            "commit_rule": "max_left_right_probability_at_threshold",
            "commit_confidence": config.commit_confidence,
            "temperature": _finite_or_none(temperature),
            "target_median_blend": _finite_or_none(target_median_blend),
            "boundary_center": "selected_blend_times_unlabeled_prefix_median_log_odds",
        },
    }


def _select_adaptation_parameters(
    name: str,
    estimator: ProbabilisticBaseline,
    data: SessionData | Any,
    partitions: Sequence[SessionCalibrationPartition],
    config: BaselineLOSOConfig,
) -> tuple[float, float, list[dict[str, Any]]]:
    """Select the matched temperature/blend grid on the protected inner subject."""

    trace: list[dict[str, Any]] = []
    best_key: tuple[float, float, float, float] | None = None
    best_values: tuple[float, float] | None = None
    for temperature in config.temperature_grid:
        for blend in config.target_median_blends:
            evaluated = evaluate_baseline_sessions(
                name,
                estimator,
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
            key = (balanced, -brier, -abs(math.log(temperature)), -blend)
            if best_key is None or key > best_key:
                best_key = key
                best_values = (float(temperature), float(blend))
    if best_values is None:
        raise RuntimeError("adaptation candidate grid produced no result")
    return best_values[0], best_values[1], trace


def score_baseline_fold(
    name: str,
    data: SessionData | Any,
    fold: NestedLOSOFold,
    config: BaselineLOSOConfig,
) -> dict[str, Any]:
    """Fit on six selection participants and score one protected outer subject."""

    if name not in BASELINE_MODELS:
        raise ValueError(f"unknown baseline {name!r}")
    source_indices = fold.selection_indices
    labels = np.asarray(data.labels, dtype=np.int64)
    source_rows = source_indices[labels[source_indices] >= 0]
    source_subjects = set(np.asarray(data.subject_ids)[source_rows].tolist())
    if fold.outer_subject in source_subjects:
        raise RuntimeError("outer subject leaked into baseline fitting rows")
    if fold.validation_subject in source_subjects:
        raise RuntimeError("inner-validation subject leaked into baseline fitting rows")
    if source_subjects != set(fold.selection_subjects):
        raise ValueError("baseline fitting rows do not contain all six selection subjects")

    estimator = _new_estimator(name, config)
    groups = np.asarray(data.session_ids)[source_rows]
    started = time.perf_counter()
    estimator.fit(
        _features(name, data, source_rows),
        labels[source_rows],
        groups=groups,
    )
    _validate_binary_classes(estimator)
    validation_partitions = session_calibration_partitions(
        data,
        fold.validation_indices,
        CALIBRATION_TASK_EVENTS,
    )
    temperature, blend, adaptation_trace = _select_adaptation_parameters(
        name, estimator, data, validation_partitions, config
    )
    outer_partitions = session_calibration_partitions(
        data,
        fold.outer_indices,
        CALIBRATION_TASK_EVENTS,
    )
    evaluated = evaluate_baseline_sessions(
        name,
        estimator,
        data,
        outer_partitions,
        temperature=temperature,
        target_median_blend=blend,
        config=config,
    )
    elapsed = time.perf_counter() - started
    return {
        "model": name,
        "subject": fold.outer_subject,
        "outer_subject": fold.outer_subject,
        "inner_validation_subject": fold.validation_subject,
        "training_subjects": list(fold.selection_subjects),
        "available_source_subjects": list(fold.available_source_subjects),
        "selection_subjects": list(fold.selection_subjects),
        "source_subjects": list(fold.selection_subjects),
        "seed": 0,
        "metrics": evaluated["metrics"],
        "predictions": evaluated["predictions"],
        "sessions": evaluated["sessions"],
        "selection": {
            "estimator_fit_subjects": list(fold.selection_subjects),
            "inner_validation_subject": fold.validation_subject,
            "inner_validation_used_for_estimator_fit": False,
            "inner_validation_sessions": [
                partition.session_id for partition in validation_partitions
            ],
            "temperature": temperature,
            "target_median_blend": blend,
            "candidate_trace": adaptation_trace,
        },
        "deployment": {
            **evaluated["deployment"],
            "source_task_windows": len(source_rows),
            "source_alignment_domains": len(np.unique(groups)),
            "fit_and_inference_seconds": _finite_or_none(elapsed),
        },
    }


def run_loso_baselines(
    *,
    subjects: Sequence[int],
    models: Sequence[str],
    config: BaselineLOSOConfig,
    output: Path,
    resume: bool = True,
) -> dict[str, Any]:
    """Run deterministic baseline folds with atomic per-fold resume checkpoints."""

    outer_subjects = tuple(int(subject) for subject in subjects)
    model_names = tuple(str(model).lower() for model in models)
    if not outer_subjects or len(set(outer_subjects)) != len(outer_subjects):
        raise ValueError("subjects must be non-empty and unique")
    if set(outer_subjects) - set(VALID_SUBJECTS):
        raise ValueError(f"outer subjects must be selected from {VALID_SUBJECTS}")
    if not model_names or len(set(model_names)) != len(model_names):
        raise ValueError("models must be non-empty and unique")
    unknown = set(model_names) - set(BASELINE_MODELS)
    if unknown:
        raise ValueError(f"unknown models: {sorted(unknown)}")

    keys = [
        SessionKey(subject, run)
        for subject in VALID_SUBJECTS
        for run in SUBJECT_RUNS[subject]
    ]
    data_config = replace(
        DEFAULT_DATA_CONFIG,
        window_name=config.window,
        include_rest=config.include_rest,
    )
    data = load_sessions(keys, data_config)
    input_contract = dataset_contract(data, data_config)
    normalized_config = _json_config(config)

    if resume and output.is_file():
        payload = json.loads(output.read_text(encoding="utf-8"))
        if (
            payload.get("schema_version") != SCHEMA_VERSION
            or payload.get("protocol") != PROTOCOL_NAME
            or payload.get("artifact_kind") != ARTIFACT_KIND
        ):
            raise ValueError("cannot resume an incompatible LOSO baseline result schema")
        if payload.get("baseline_config") != normalized_config:
            raise ValueError("cannot resume with a different LOSO baseline configuration")
        if payload.get("data_contract") != input_contract:
            raise ValueError("cannot resume with a different preprocessing/data contract")
        if payload.get("outer_subjects") != list(outer_subjects):
            raise ValueError("cannot resume with different outer subjects")
        if payload.get("models") != list(model_names):
            raise ValueError("cannot resume with different baseline models")
    else:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "protocol": PROTOCOL_NAME,
            "artifact_kind": ARTIFACT_KIND,
            "model": "classical_baselines",
            "models": list(model_names),
            "baseline_config": normalized_config,
            "loso_config": normalized_config,
            "data_contract": input_contract,
            "subject_order": list(VALID_SUBJECTS),
            "outer_subjects": list(outer_subjects),
            "seeds": [0],
            "environment": _environment(),
            "repository": _repository_state(),
            "command": list(sys.argv),
            "calibration_policy": {
                "task_events_per_session": CALIBRATION_TASK_EVENTS,
                "prefix_direction_labels_used": False,
                "inner_scored_labels_used_for_temperature_blend_selection": True,
                "outer_labels_used_for_selection": False,
                "prefix_excluded_from_scoring": True,
                "reset_per_session": True,
            },
            "limitations": [
                BASELINE_NESTED_LIMITATION,
                CALIBRATION_LIMITATION,
                NO_INTENT_GATE_LIMITATION,
                COHORT_LIMITATION,
            ],
            "folds": [],
            "summary": {},
        }

    rows: list[dict[str, Any]] = payload["folds"]
    completed = {
        (str(row["model"]), int(row["outer_subject"])) for row in rows
    }
    for outer_subject in outer_subjects:
        fold = nested_loso_fold(data, outer_subject)
        for name in model_names:
            key = (name, outer_subject)
            if key in completed:
                print(
                    f"skip complete strict LOSO {name} outer={outer_subject}",
                    flush=True,
                )
                continue
            print(f"run strict LOSO {name} outer={outer_subject}", flush=True)
            row = score_baseline_fold(name, data, fold, config)
            rows.append(row)
            completed.add(key)
            payload["updated_at"] = datetime.now(timezone.utc).isoformat()
            payload["summary"] = _summary(rows)
            _atomic_json(output, payload)
            balanced = row["metrics"]["balanced_accuracy"]
            print(
                f"done strict LOSO {name} outer={outer_subject}: "
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


def _csv_models(value: str) -> tuple[str, ...]:
    parsed = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    unknown = set(parsed) - set(BASELINE_MODELS)
    if not parsed or unknown:
        raise argparse.ArgumentTypeError(
            f"models must be selected from {BASELINE_MODELS}"
        )
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", default="all", help="outer subjects or 'all'")
    parser.add_argument(
        "--models", type=_csv_models, default=BASELINE_MODELS,
        help="comma-separated deterministic baselines",
    )
    parser.add_argument("--window", choices=tuple(EPOCH_WINDOWS), default="deployment")
    parser.add_argument("--task-only", action="store_true")
    parser.add_argument("--commit-confidence", type=float, default=0.85)
    parser.add_argument("--fbcsp-components", type=int, default=2)
    parser.add_argument(
        "--temperature-grid",
        type=_csv_floats,
        default=BaselineLOSOConfig.temperature_grid,
    )
    parser.add_argument(
        "--target-median-blends",
        type=_csv_floats,
        default=BaselineLOSOConfig.target_median_blends,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "deepnet" / "results" / "nested_loso_baselines.json",
    )
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    subjects = _csv_ints(args.subjects, all_values=VALID_SUBJECTS)
    config = BaselineLOSOConfig(
        window=args.window,
        include_rest=not args.task_only,
        commit_confidence=args.commit_confidence,
        fbcsp_components=args.fbcsp_components,
        temperature_grid=args.temperature_grid,
        target_median_blends=args.target_median_blends,
    )
    payload = run_loso_baselines(
        subjects=subjects,
        models=args.models,
        config=config,
        output=args.output,
        resume=not args.no_resume,
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ARTIFACT_KIND",
    "BASELINE_MODELS",
    "BaselineLOSOConfig",
    "CALIBRATION_TASK_EVENTS",
    "NO_INTENT_GATE_LIMITATION",
    "build_parser",
    "evaluate_baseline_sessions",
    "main",
    "run_loso_baselines",
    "score_baseline_fold",
]
