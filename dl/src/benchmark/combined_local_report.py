"""Audit and combine every current-protocol local Exp4 benchmark result.

This module deliberately does not run or import any benchmark runner.  It reads
the frozen all-neural final ranking, the current-cache geometric controls, and
the four exact outer-refit ``deepnet`` artifacts.  Scores are reconstructed at
the subject/seed level and are combined only after each input passes its own
schema-appropriate completeness checks.

The resulting winner is a post-selection local-development winner.  Recording
4 has been opened during earlier project development, so the report is never
confirmation evidence or support for a state-of-the-art/clinical claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "eeg-mi-combined-local-report-v1"
NEURAL_SCHEMA = "eeg-mi-local-model-tournament-result-v1"
NEURAL_ARTIFACT_SCHEMA = "eeg-mi-development-v4"
GEOMETRIC_SCHEMA = "eeg-mi-local-geometric-controls-v1"
DEEPNET_SCHEMA = "deepnet-local-outer-refit-v1"
DATASET = "local_exp4"
SUBJECTS: tuple[int, ...] = (1, 3, 4, 5, 6, 7, 8, 10)
SEEDS: tuple[int, ...] = (7, 17, 27, 37, 47)
FOLDS: tuple[int, ...] = (0,)
GEOMETRIC_MODELS: tuple[str, ...] = ("riemann", "tangent_anchor", "ea_fbcsp")
DEEPNET_MODELS: tuple[str, ...] = (
    "cameo",
    "hemiparity",
    "parity_fuse",
    "orbit_v3",
)

# Frozen independently here so this auditor cannot silently inherit a changed
# tournament roster from executable benchmark code.
NEURAL_CONFIGURATIONS: tuple[str, ...] = (
    "scope",
    "free_scope",
    "cardinal",
    "free_cardinal",
    "cardinal_dynamics",
    "cardinal_dynamics_compact",
    "cardinal_dynamics_extended",
    "cardinal_dynamics_sinc",
    "cardinal_dynamics_sinc_residual",
    "cardinal_dynamics_sinc_extended",
    "eegnet",
    "shallow",
    "deep4",
    "eegconformer",
    "eegconformer_compact",
    "atcnet",
    "atcnet_aggressive_pool",
    "fbcnet",
    "cardinal_fbc",
    "cardinal_fbc_extended",
    "cardinal_fbc_corr",
    "cardinal_fbc_corr_extended",
    "cardinal_fbc_compactdyn",
    "cardinal_fbc_compactdyn_extended",
    "cardinal_fbc_compactdyn_scale010",
    "cardinal_fbc_compactdyn_scale010_extended",
    "cardinal_fbc_compactdyn_scale025",
    "cardinal_fbc_compactdyn_scale025_extended",
    "cardinal_fbc_micro",
    "cardinal_fbc_micro_extended",
    "cardinal_fbc_physical",
    "cardinal_fbc_physical_extended",
    "eegtcnet",
    "fbmsnet",
    "cardinal_fbms",
    "cardinal_fbms_extended",
    "cardinal_mix",
    "cardinal_mix_drop",
    "ctnet",
    "ctnet_compact",
    "eegsym",
    "eegsym_wide",
    "tcformer",
)

CATEGORY_NEURAL = "common_recipe_neural_configuration"
CATEGORY_GEOMETRIC = "current_cache_geometric_control"
CATEGORY_PROCEDURE = "full_procedure_comparator"


@dataclass(frozen=True)
class Condition:
    name: str
    category: str
    subject_seed: dict[str, dict[str, float]]
    parameter_count: int | None
    score_origin: str
    source: str


class TestIdentityRegistry:
    """Require every available raw source to score the same R4 rows/labels."""

    def __init__(self) -> None:
        self._identity: dict[int, tuple[Any, ...]] = {}
        self.coverage: dict[str, set[int]] = {}

    def add(
        self,
        *,
        source: str,
        subject: int,
        rows: Sequence[int],
        labels: Sequence[int],
        sessions: Sequence[str] | None,
        runs: Sequence[str] | None,
    ) -> None:
        identity = (
            tuple(int(value) for value in rows),
            tuple(int(value) for value in labels),
            tuple(str(value) for value in sessions) if sessions is not None else None,
            tuple(str(value) for value in runs) if runs is not None else None,
        )
        previous = self._identity.setdefault(int(subject), identity)
        if previous != identity:
            raise ValueError(
                f"raw R4 row/label/session identity differs for subject {subject} "
                f"at source {source}"
            )
        self.coverage.setdefault(source, set()).add(int(subject))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def strict_load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(
            stream,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _finite_score(value: Any, *, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} is not numeric") from error
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{context} must be finite and lie in [0, 1]")
    return result


def _close(left: Any, right: Any) -> bool:
    try:
        return bool(
            np.isclose(float(left), float(right), rtol=1e-10, atol=1e-12)
        )
    except (TypeError, ValueError):
        return False


def _balanced_accuracy(labels: np.ndarray, probabilities: np.ndarray) -> float:
    prediction = probabilities.argmax(axis=1)
    recalls = [float(np.mean(prediction[labels == label] == label)) for label in (0, 1)]
    return float(np.mean(recalls))


def _validate_prediction(
    *,
    rows: Sequence[Any],
    labels: Sequence[Any],
    probabilities: Sequence[Any],
    sessions: Sequence[Any] | None,
    runs: Sequence[Any] | None,
    context: str,
) -> tuple[float, list[int], list[int], list[str] | None, list[str] | None]:
    row_values = [int(value) for value in rows]
    label_values = [int(value) for value in labels]
    probability = np.asarray(probabilities, dtype=np.float64)
    if len(row_values) != 60 or len(set(row_values)) != 60:
        raise ValueError(f"{context} must contain 60 unique R4 rows")
    if len(label_values) != 60 or np.bincount(label_values, minlength=2).tolist() != [30, 30]:
        raise ValueError(f"{context} must contain 30 labels from each class")
    if probability.shape != (60, 2):
        raise ValueError(f"{context} probabilities must have shape (60, 2)")
    if (
        not np.all(np.isfinite(probability))
        or np.any(probability < 0.0)
        or np.any(probability > 1.0)
        or not np.allclose(probability.sum(axis=1), 1.0, rtol=0.0, atol=1e-5)
    ):
        raise ValueError(f"{context} contains invalid probabilities")
    session_values = None if sessions is None else [str(value) for value in sessions]
    run_values = None if runs is None else [str(value) for value in runs]
    if session_values is not None and len(session_values) != 60:
        raise ValueError(f"{context} must contain 60 session identities")
    if run_values is not None and len(run_values) != 60:
        raise ValueError(f"{context} must contain 60 run identities")
    return (
        _balanced_accuracy(np.asarray(label_values, dtype=np.int64), probability),
        row_values,
        label_values,
        session_values,
        run_values,
    )


def _validate_matrix(
    value: Any, *, context: str
) -> dict[str, dict[str, float]]:
    if not isinstance(value, dict) or set(value) != {str(subject) for subject in SUBJECTS}:
        raise ValueError(f"{context} must contain exactly the eight formal subjects")
    result: dict[str, dict[str, float]] = {}
    for subject in SUBJECTS:
        raw_seeds = value[str(subject)]
        if not isinstance(raw_seeds, dict) or set(raw_seeds) != {
            str(seed) for seed in SEEDS
        }:
            raise ValueError(
                f"{context} subject {subject} must contain exactly the five formal seeds"
            )
        result[str(subject)] = {
            str(seed): _finite_score(
                raw_seeds[str(seed)], context=f"{context} S{subject} seed {seed} BA"
            )
            for seed in SEEDS
        }
    return result


def _subject_values(matrix: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    return {
        str(subject): float(
            np.mean([float(matrix[str(subject)][str(seed)]) for seed in SEEDS])
        )
        for subject in SUBJECTS
    }


def _condition_summary(condition: Condition) -> dict[str, Any]:
    subject_values = _subject_values(condition.subject_seed)
    values = np.asarray([subject_values[str(subject)] for subject in SUBJECTS])
    return {
        "name": condition.name,
        "category": condition.category,
        "balanced_accuracy_mean": float(values.mean()),
        "balanced_accuracy_standard_deviation": float(values.std(ddof=1)),
        "minimum_subject_balanced_accuracy": float(values.min()),
        "median_subject_balanced_accuracy": float(np.median(values)),
        "subject_balanced_accuracy": subject_values,
        "subject_seed_balanced_accuracy": condition.subject_seed,
        "parameter_count": condition.parameter_count,
        "score_origin": condition.score_origin,
        "source": condition.source,
    }


def _require_protocol_lists(payload: Mapping[str, Any], *, context: str) -> None:
    if payload.get("dataset") != DATASET:
        raise ValueError(f"{context} has the wrong dataset")
    if payload.get("subjects") != list(SUBJECTS):
        raise ValueError(f"{context} does not declare the eight formal subjects")
    if payload.get("seeds") != list(SEEDS):
        raise ValueError(f"{context} does not declare the five formal seeds")


def _neural_artifact_candidates(
    ranking_path: Path, rows: Mapping[str, Mapping[str, Any]]
) -> dict[str, Path]:
    result: dict[str, Path] = {}
    sibling_root = ranking_path.parent / "artifacts"
    for model, row in rows.items():
        declared = Path(str(row.get("artifact", "")))
        sibling = sibling_root / f"{model}.json"
        if declared.is_file():
            result[model] = declared.resolve()
        elif sibling.is_file():
            result[model] = sibling.resolve()
    if result and set(result) != set(NEURAL_CONFIGURATIONS):
        missing = sorted(set(NEURAL_CONFIGURATIONS) - set(result))
        raise ValueError(
            "only part of the neural raw-artifact set is available; "
            f"refusing a mixed audit, missing={missing}"
        )
    return result


def _validate_neural_artifact(
    path: Path,
    *,
    model: str,
    expected_matrix: Mapping[str, Mapping[str, float]],
    identities: TestIdentityRegistry,
) -> dict[str, dict[str, float]]:
    artifact = strict_load(path)
    if artifact.get("schema") != NEURAL_ARTIFACT_SCHEMA:
        raise ValueError(f"{path} has the wrong neural artifact schema")
    _require_protocol_lists(artifact, context=str(path))
    if artifact.get("models") != [model] or artifact.get("folds") != [0]:
        raise ValueError(f"{path} has incorrect model/fold dimensions")
    if (
        artifact.get("mode") != "development"
        or artifact.get("artifact_mode") != "formal"
        or artifact.get("evidence_scope")
        != "prespecified_formal_development_not_confirmation"
        or artifact.get("confirmation_evidence") is not False
    ):
        raise ValueError(f"{path} is not a formal-complete development artifact")
    records = artifact.get("records")
    expected_keys = set(product(SUBJECTS, FOLDS, SEEDS))
    if not isinstance(records, list) or len(records) != len(expected_keys):
        raise ValueError(f"{path} must contain exactly 40 neural records")
    seen: set[tuple[int, int, int]] = set()
    matrix: dict[str, dict[str, float]] = {str(subject): {} for subject in SUBJECTS}
    for record in records:
        key = (
            int(record.get("subject", -1)),
            int(record.get("fold", -1)),
            int(record.get("seed", -1)),
        )
        if key not in expected_keys or key in seen or record.get("model") != model:
            raise ValueError(f"{path} has an invalid/duplicate neural record {key}")
        seen.add(key)
        subject, _, seed = key
        split = record.get("split", {})
        counts = tuple(
            int(split.get(name, -1))
            for name in (
                "train_count",
                "validation_count",
                "refit_source_count",
                "test_count",
            )
        )
        if counts != (120, 60, 180, 60):
            raise ValueError(f"{path} record {key} has invalid split counts")
        test = record.get("test", {})
        score, rows, labels, sessions, runs = _validate_prediction(
            rows=test.get("rows", []),
            labels=test.get("labels", []),
            probabilities=test.get("probabilities", []),
            sessions=test.get("sessions"),
            runs=test.get("runs"),
            context=f"{path} record {key}",
        )
        if not _close(test.get("metrics", {}).get("balanced_accuracy"), score):
            raise ValueError(f"{path} record {key} has stale balanced accuracy")
        identities.add(
            source="neural_raw_artifacts",
            subject=subject,
            rows=rows,
            labels=labels,
            sessions=sessions,
            runs=runs,
        )
        matrix[str(subject)][str(seed)] = score
    if seen != expected_keys:
        raise ValueError(f"{path} neural record grid is incomplete")
    for subject in SUBJECTS:
        for seed in SEEDS:
            if not _close(
                matrix[str(subject)][str(seed)], expected_matrix[str(subject)][str(seed)]
            ):
                raise ValueError(
                    f"{path} raw BA differs from final ranking at S{subject}/seed{seed}"
                )
    return matrix


def _load_neural(
    path: Path, identities: TestIdentityRegistry
) -> tuple[list[Condition], dict[str, Any]]:
    payload = strict_load(path)
    if payload.get("schema") != NEURAL_SCHEMA:
        raise ValueError(f"{path} has the wrong neural final-ranking schema")
    if payload.get("evidence_scope") != "development_only_not_confirmation" or payload.get(
        "confirmation_evidence"
    ) is not False:
        raise ValueError(f"{path} incorrectly claims confirmation evidence")
    if payload.get("primary_metric") != (
        "mean_subject_balanced_accuracy_after_averaging_seeds_within_subject"
    ):
        raise ValueError(f"{path} has the wrong neural primary metric")
    if not _is_sha256(payload.get("plan_sha256")):
        raise ValueError(f"{path} has no valid frozen-plan digest")
    reference_analysis = payload.get("reference_analysis")
    if (
        not isinstance(reference_analysis, dict)
        or reference_analysis.get("evidence_scope")
        != "development_only_not_confirmation"
        or reference_analysis.get("confirmation_evidence") is not False
    ):
        raise ValueError(f"{path} reference analysis has an invalid evidence scope")
    ranking = payload.get("ranking")
    if not isinstance(ranking, list) or len(ranking) != len(NEURAL_CONFIGURATIONS):
        raise ValueError(f"{path} must contain exactly 43 neural configurations")
    rows: dict[str, Mapping[str, Any]] = {}
    conditions: dict[str, Condition] = {}
    for position, row in enumerate(ranking, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"{path} contains a non-object neural ranking row")
        model = str(row.get("model", ""))
        if model not in NEURAL_CONFIGURATIONS or model in rows:
            raise ValueError(f"{path} has an unknown/duplicate neural row {model!r}")
        if int(row.get("rank", -1)) != position:
            raise ValueError(f"{path} neural ranks are not contiguous")
        matrix = _validate_matrix(
            row.get("subject_seed_balanced_accuracy"),
            context=f"{path} {model}",
        )
        subject_values = _subject_values(matrix)
        stored_subjects = row.get("subject_balanced_accuracy")
        if not isinstance(stored_subjects, dict) or set(stored_subjects) != set(
            subject_values
        ):
            raise ValueError(f"{path} {model} has incomplete subject scores")
        for subject, expected in subject_values.items():
            if not _close(stored_subjects[subject], expected):
                raise ValueError(f"{path} {model} has a stale subject score")
        values = np.asarray(list(subject_values.values()), dtype=np.float64)
        checks = {
            "balanced_accuracy_mean": values.mean(),
            "balanced_accuracy_standard_deviation": values.std(ddof=1),
            "minimum_subject_balanced_accuracy": values.min(),
            "median_subject_balanced_accuracy": np.median(values),
        }
        for field, expected in checks.items():
            if not _close(row.get(field), expected):
                raise ValueError(f"{path} {model} has stale {field}")
        interval = row.get("bootstrap_95_ci")
        if (
            not isinstance(interval, list)
            or len(interval) != 2
            or any(not 0.0 <= _finite_score(value, context=f"{model} CI") <= 1.0 for value in interval)
            or float(interval[0]) > float(interval[1])
        ):
            raise ValueError(f"{path} {model} has an invalid bootstrap interval")
        if not _is_sha256(row.get("artifact_sha256")):
            raise ValueError(f"{path} {model} has no valid artifact digest")
        parameter_count = int(row.get("parameter_count", 0))
        if parameter_count <= 0:
            raise ValueError(f"{path} {model} has an invalid parameter count")
        rows[model] = row
        conditions[model] = Condition(
            name=model,
            category=CATEGORY_NEURAL,
            subject_seed=matrix,
            parameter_count=parameter_count,
            score_origin="final_ranking_subject_seed_matrix",
            source=str(path),
        )
    if set(rows) != set(NEURAL_CONFIGURATIONS):
        raise ValueError(f"{path} neural roster differs from the frozen 43")

    recomputed_order = sorted(
        NEURAL_CONFIGURATIONS,
        key=lambda model: (
            -float(np.mean(list(_subject_values(conditions[model].subject_seed).values()))),
            model,
        ),
    )
    if [str(row["model"]) for row in ranking] != recomputed_order:
        raise ValueError(f"{path} neural ranking order is stale")
    top_mean = float(
        np.mean(list(_subject_values(conditions[recomputed_order[0]].subject_seed).values()))
    )
    co_winners = [
        model
        for model in recomputed_order
        if abs(
            float(np.mean(list(_subject_values(conditions[model].subject_seed).values())))
            - top_mean
        )
        <= 1e-12
    ]
    runners = [model for model in recomputed_order if model not in co_winners]
    if not runners:
        raise ValueError(f"{path} has no distinct neural runner-up")
    if payload.get("winner") != co_winners[0] or payload.get("co_winners") != co_winners:
        raise ValueError(f"{path} has stale neural winner fields")
    if payload.get("runner_up") != runners[0] or not _close(
        payload.get("winner_balanced_accuracy"), top_mean
    ):
        raise ValueError(f"{path} has stale neural winner/runner score fields")

    raw_paths = _neural_artifact_candidates(path, rows)
    raw_record_count = 0
    if raw_paths:
        for model in NEURAL_CONFIGURATIONS:
            expected_digest = str(rows[model]["artifact_sha256"])
            if _file_sha256(raw_paths[model]) != expected_digest:
                raise ValueError(f"{raw_paths[model]} differs from its ranking digest")
            raw_matrix = _validate_neural_artifact(
                raw_paths[model],
                model=model,
                expected_matrix=conditions[model].subject_seed,
                identities=identities,
            )
            conditions[model] = Condition(
                **{
                    **conditions[model].__dict__,
                    "subject_seed": raw_matrix,
                    "score_origin": "raw_test_predictions",
                }
            )
            raw_record_count += len(SUBJECTS) * len(SEEDS)
        validation_level = "43 raw artifacts; 1,720 prediction records"
    else:
        validation_level = (
            "final-ranking subject/seed matrices only; trial predictions are not "
            "contained in this input schema"
        )
    return [conditions[model] for model in NEURAL_CONFIGURATIONS], {
        "path": str(path),
        "sha256": _file_sha256(path),
        "validation_level": validation_level,
        "raw_records_validated": raw_record_count,
    }


def _load_geometric(
    path: Path, identities: TestIdentityRegistry
) -> tuple[list[Condition], dict[str, Any]]:
    payload = strict_load(path)
    if payload.get("schema") != GEOMETRIC_SCHEMA:
        raise ValueError(f"{path} has the wrong geometric-control schema")
    _require_protocol_lists(payload, context=str(path))
    if payload.get("models") != list(GEOMETRIC_MODELS) or payload.get("folds") != [0]:
        raise ValueError(f"{path} has incorrect geometric dimensions")
    if (
        payload.get("evidence_scope") != "formal_development_not_confirmation"
        or payload.get("confirmation_evidence") is not False
        or payload.get("test_calibration") is not False
        or int(payload.get("canonical_fit_seed", -1)) != SEEDS[0]
        or payload.get("split_protocol")
        != "R1-2 selection fit; R3 selection; R1-3 refit; R4 prediction-only"
    ):
        raise ValueError(f"{path} is not a non-transductive development artifact")
    records = payload.get("records")
    expected_keys = set(product(GEOMETRIC_MODELS, SUBJECTS, SEEDS))
    if not isinstance(records, list) or len(records) != len(expected_keys):
        raise ValueError(f"{path} must contain exactly 120 geometric-control records")
    seen: set[tuple[str, int, int]] = set()
    matrices = {
        model: {str(subject): {} for subject in SUBJECTS}
        for model in GEOMETRIC_MODELS
    }
    canonical_test: dict[tuple[str, int], Any] = {}
    for record in records:
        key = (
            str(record.get("model", "")),
            int(record.get("subject", -1)),
            int(record.get("seed", -1)),
        )
        if key not in expected_keys or key in seen or int(record.get("fold", -1)) != 0:
            raise ValueError(f"{path} has an invalid/duplicate geometric record {key}")
        seen.add(key)
        model, subject, seed = key
        if record.get("deterministic_estimator") is not True:
            raise ValueError(f"{path} record {key} is not declared deterministic")
        expected_role = (
            "canonical_deterministic_fit" if seed == SEEDS[0] else "exact_deterministic_replication"
        )
        if record.get("seed_role") != expected_role:
            raise ValueError(f"{path} record {key} has an invalid deterministic seed role")
        split = record.get("split", {})
        counts = tuple(
            int(split.get(name, -1))
            for name in (
                "train_count",
                "validation_count",
                "refit_source_count",
                "test_count",
            )
        )
        if counts != (120, 60, 180, 60):
            raise ValueError(f"{path} record {key} has invalid split counts")
        test = record.get("test", {})
        score, rows, labels, sessions, runs = _validate_prediction(
            rows=test.get("rows", []),
            labels=test.get("labels", []),
            probabilities=test.get("probabilities", []),
            sessions=test.get("sessions"),
            runs=test.get("runs"),
            context=f"{path} record {key}",
        )
        if not _close(test.get("metrics", {}).get("balanced_accuracy"), score):
            raise ValueError(f"{path} record {key} has stale balanced accuracy")
        identities.add(
            source="geometric_controls",
            subject=subject,
            rows=rows,
            labels=labels,
            sessions=sessions,
            runs=runs,
        )
        canonical_key = (model, subject)
        canonical = canonical_test.setdefault(canonical_key, test)
        if _canonical_sha256(canonical) != _canonical_sha256(test):
            raise ValueError(
                f"{path} deterministic seed replicas differ for {canonical_key}"
            )
        matrices[model][str(subject)][str(seed)] = score
    if seen != expected_keys:
        raise ValueError(f"{path} geometric record grid is incomplete")
    conditions: list[Condition] = []
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise ValueError(f"{path} has no geometric summary")
    for model in GEOMETRIC_MODELS:
        matrix = _validate_matrix(matrices[model], context=f"{path} {model}")
        condition = Condition(
            name=model,
            category=CATEGORY_GEOMETRIC,
            subject_seed=matrix,
            parameter_count=None,
            score_origin="raw_test_predictions_deterministic_seed_replication",
            source=str(path),
        )
        recomputed = _condition_summary(condition)
        stored = summary.get(model, {})
        if not _close(stored.get("balanced_accuracy_mean"), recomputed["balanced_accuracy_mean"]):
            raise ValueError(f"{path} {model} has a stale summary mean")
        if not _close(
            stored.get("balanced_accuracy_std"),
            recomputed["balanced_accuracy_standard_deviation"],
        ):
            raise ValueError(f"{path} {model} has a stale summary standard deviation")
        if int(stored.get("n_subjects", -1)) != 8 or int(stored.get("n_records", -1)) != 40:
            raise ValueError(f"{path} {model} has stale summary counts")
        stored_subjects = stored.get("subject_balanced_accuracy")
        if not isinstance(stored_subjects, dict) or set(stored_subjects) != {
            str(subject) for subject in SUBJECTS
        }:
            raise ValueError(f"{path} {model} has incomplete summary subject scores")
        for subject in SUBJECTS:
            if not _close(
                stored_subjects[str(subject)],
                recomputed["subject_balanced_accuracy"][str(subject)],
            ):
                raise ValueError(f"{path} {model} has stale summary subject scores")
        conditions.append(condition)
    return conditions, {
        "path": str(path),
        "sha256": _file_sha256(path),
        "validation_level": "120 raw prediction records; deterministic seed replicas verified",
        "raw_records_validated": 120,
    }


def _load_deepnet(
    path: Path,
    *,
    expected_model: str,
    identities: TestIdentityRegistry,
    cache_hashes: dict[tuple[int, int], str],
) -> tuple[Condition, dict[str, Any], dict[str, Any]]:
    payload = strict_load(path)
    if payload.get("schema_version") != DEEPNET_SCHEMA:
        raise ValueError(f"{path} has the wrong deepnet schema")
    contract = payload.get("contract")
    if not isinstance(contract, dict):
        raise ValueError(f"{path} has no deepnet contract")
    _require_protocol_lists(contract, context=str(path))
    if contract.get("model") != expected_model or contract.get(
        "formal_complete_dimensions"
    ) is not True:
        raise ValueError(f"{path} has incorrect deepnet formal dimensions")
    if contract.get("evidence_scope") != (
        "post_selection_local_development_all_four_recordings_previously_opened"
    ):
        raise ValueError(f"{path} has an invalid evidence scope")
    outer_protocol = contract.get("outer_protocol")
    if (
        not isinstance(outer_protocol, dict)
        or int(outer_protocol.get("fold", -1)) != 0
        or outer_protocol.get("phase_a")
        != "recordings 1-2 fit; recording 3 selects duration and route only"
        or outer_protocol.get("phase_b")
        != "fresh reset; source preprocessing refit on recordings 1-3; fixed selected duration"
        or outer_protocol.get("test")
        != "recording 4 prediction only after phase B"
    ):
        raise ValueError(f"{path} has an invalid outer protocol")
    if payload.get("contract_sha256") != _canonical_sha256(contract):
        raise ValueError(f"{path} has a stale contract digest")
    completion = payload.get("completion", {})
    if (
        payload.get("status") != "complete"
        or completion.get("complete") is not True
        or int(completion.get("expected_records", -1)) != 40
        or int(completion.get("actual_records", -1)) != 40
    ):
        raise ValueError(f"{path} is not a complete 40-record deepnet artifact")
    records = payload.get("records")
    expected_keys = set(product(SUBJECTS, SEEDS))
    if not isinstance(records, list) or len(records) != 40:
        raise ValueError(f"{path} must contain exactly 40 deepnet records")
    seen: set[tuple[int, int]] = set()
    matrix = {str(subject): {} for subject in SUBJECTS}
    parameter_counts: set[int] = set()
    for record in records:
        key = (int(record.get("subject", -1)), int(record.get("seed", -1)))
        if key not in expected_keys or key in seen or record.get("model") != expected_model:
            raise ValueError(f"{path} has an invalid/duplicate deepnet record {key}")
        seen.add(key)
        subject, seed = key
        split = record.get("split", {})
        counts = tuple(
            int(split.get(part, {}).get("count", -1))
            for part in (
                "phase_a_train",
                "phase_a_validation",
                "phase_b_source",
                "prediction_only_test",
            )
        )
        if counts != (120, 60, 180, 60):
            raise ValueError(f"{path} record {key} has invalid split counts")
        phase_a = record.get("phase_a", {})
        phase_b = record.get("phase_b", {})
        if phase_b.get("reset_verified") is not True or int(
            phase_b.get("epoch_count", -2)
        ) != int(phase_a.get("selected_epoch_count", -1)):
            raise ValueError(f"{path} record {key} has an invalid reset/refit duration")
        parameter_count = int(phase_b.get("parameter_count", 0))
        if parameter_count <= 0:
            raise ValueError(f"{path} record {key} has an invalid parameter count")
        parameter_counts.add(parameter_count)
        prediction = record.get("prediction_only_test", {})
        trace = prediction.get("predictions")
        if not isinstance(trace, list) or len(trace) != 60:
            raise ValueError(f"{path} record {key} must contain 60 predictions")
        rows = [int(row["row"]) for row in trace]
        labels = [int(row["label"]) for row in trace]
        probabilities = [
            [float(row["probability_left"]), float(row["probability_right"])]
            for row in trace
        ]
        sessions = [str(row["session"]) for row in trace]
        runs = [str(row["run"]) for row in trace]
        score, rows, labels, sessions, runs = _validate_prediction(
            rows=rows,
            labels=labels,
            probabilities=probabilities,
            sessions=sessions,
            runs=runs,
            context=f"{path} record {key}",
        )
        if not _close(prediction.get("metrics", {}).get("balanced_accuracy"), score):
            raise ValueError(f"{path} record {key} has stale balanced accuracy")
        identities.add(
            source=f"deepnet:{expected_model}",
            subject=subject,
            rows=rows,
            labels=labels,
            sessions=sessions,
            runs=runs,
        )
        cache_digest = record.get("cache_identity_sha256")
        if not _is_sha256(cache_digest):
            raise ValueError(f"{path} record {key} has no valid cache digest")
        previous_cache = cache_hashes.setdefault(key, str(cache_digest))
        if previous_cache != cache_digest:
            raise ValueError(f"deepnet cache identity differs across methods at {key}")
        matrix[str(subject)][str(seed)] = score
    if seen != expected_keys or len(parameter_counts) != 1:
        raise ValueError(f"{path} has an incomplete grid or varying parameter count")
    condition = Condition(
        name=expected_model,
        category=CATEGORY_PROCEDURE,
        subject_seed=_validate_matrix(matrix, context=f"{path} {expected_model}"),
        parameter_count=parameter_counts.pop(),
        score_origin="raw_test_predictions",
        source=str(path),
    )
    recomputed = _condition_summary(condition)
    summary = payload.get("summary", {})
    if not _close(summary.get("balanced_accuracy_mean"), recomputed["balanced_accuracy_mean"]):
        raise ValueError(f"{path} has a stale deepnet summary mean")
    if not _close(
        summary.get("balanced_accuracy_std"),
        recomputed["balanced_accuracy_standard_deviation"],
    ):
        raise ValueError(f"{path} has a stale deepnet summary standard deviation")
    if int(summary.get("n_participants", -1)) != 8 or int(summary.get("n_records", -1)) != 40:
        raise ValueError(f"{path} has stale deepnet summary counts")
    if summary.get("aggregation") != (
        "mean seeds within participant, then equal-weight participant mean"
    ):
        raise ValueError(f"{path} has the wrong deepnet aggregation")
    stored_subjects = summary.get("subject_balanced_accuracy")
    if not isinstance(stored_subjects, dict) or set(stored_subjects) != {
        str(subject) for subject in SUBJECTS
    }:
        raise ValueError(f"{path} has incomplete deepnet summary subject scores")
    for subject in SUBJECTS:
        if not _close(
            stored_subjects[str(subject)],
            recomputed["subject_balanced_accuracy"][str(subject)],
        ):
            raise ValueError(f"{path} has stale deepnet summary subject scores")
    metadata = {
        "path": str(path),
        "sha256": _file_sha256(path),
        "validation_level": "40 raw prediction records",
        "raw_records_validated": 40,
    }
    compatibility = {
        "preprocessing": contract.get("preprocessing"),
        "covariance_view": contract.get("covariance_view"),
    }
    return condition, metadata, compatibility


def _bootstrap_interval(
    values: np.ndarray, *, generator: np.random.Generator, repetitions: int
) -> list[float]:
    indices = generator.integers(0, len(values), size=(repetitions, len(values)))
    samples = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(samples, (0.025, 0.975))]


def _exact_sign_flip(differences: np.ndarray) -> float:
    observed = abs(float(differences.mean()))
    values = [
        abs(float(np.mean(differences * np.asarray(bits, dtype=np.float64))))
        for bits in product((-1.0, 1.0), repeat=len(differences))
    ]
    return float(np.mean(np.asarray(values) >= observed - 1e-15))


def build_report(
    *,
    neural_ranking: Path,
    geometric_controls: Path,
    deepnet_paths: Mapping[str, Path],
    bootstrap_repetitions: int = 100_000,
    random_seed: int = 20260721,
) -> dict[str, Any]:
    """Validate all inputs and return a combined, subject-equal report."""

    if bootstrap_repetitions <= 0:
        raise ValueError("bootstrap repetitions must be positive")
    if set(deepnet_paths) != set(DEEPNET_MODELS):
        raise ValueError("deepnet_paths must contain cameo, hemiparity, parity_fuse, orbit_v3")
    identities = TestIdentityRegistry()
    conditions, neural_metadata = _load_neural(neural_ranking.resolve(), identities)
    geometric, geometric_metadata = _load_geometric(
        geometric_controls.resolve(), identities
    )
    conditions.extend(geometric)
    cache_hashes: dict[tuple[int, int], str] = {}
    deepnet_metadata: dict[str, Any] = {}
    compatibility: dict[str, Any] | None = None
    for model in DEEPNET_MODELS:
        condition, metadata, current_compatibility = _load_deepnet(
            deepnet_paths[model].resolve(),
            expected_model=model,
            identities=identities,
            cache_hashes=cache_hashes,
        )
        if compatibility is None:
            compatibility = current_compatibility
        elif compatibility != current_compatibility:
            raise ValueError("deepnet procedures use different preprocessing/covariance contracts")
        conditions.append(condition)
        deepnet_metadata[model] = metadata
    if len(conditions) != 50 or len({condition.name for condition in conditions}) != 50:
        raise RuntimeError("combined condition roster is not exactly 50 unique names")

    generator = np.random.default_rng(int(random_seed))
    summaries = {
        condition.name: _condition_summary(condition) for condition in conditions
    }
    for name in sorted(summaries):
        values = np.asarray(
            [summaries[name]["subject_balanced_accuracy"][str(subject)] for subject in SUBJECTS],
            dtype=np.float64,
        )
        summaries[name]["bootstrap_95_ci"] = _bootstrap_interval(
            values,
            generator=generator,
            repetitions=int(bootstrap_repetitions),
        )
    ranking_names = sorted(
        summaries,
        key=lambda name: (-float(summaries[name]["balanced_accuracy_mean"]), name),
    )
    top_mean = float(summaries[ranking_names[0]]["balanced_accuracy_mean"])
    co_winners = [
        name
        for name in ranking_names
        if abs(float(summaries[name]["balanced_accuracy_mean"]) - top_mean) <= 1e-12
    ]
    runner_candidates = [name for name in ranking_names if name not in co_winners]
    if not runner_candidates:
        raise ValueError("all 50 conditions are tied; no runner-up exists")
    winner = co_winners[0]
    runner_up = runner_candidates[0]
    differences = np.asarray(
        [
            summaries[winner]["subject_balanced_accuracy"][str(subject)]
            - summaries[runner_up]["subject_balanced_accuracy"][str(subject)]
            for subject in SUBJECTS
        ],
        dtype=np.float64,
    )
    paired_indices = generator.integers(
        0,
        len(differences),
        size=(int(bootstrap_repetitions), len(differences)),
    )
    paired_samples = differences[paired_indices].mean(axis=1)
    ranking: list[dict[str, Any]] = []
    for rank, name in enumerate(ranking_names, start=1):
        row = dict(summaries[name])
        row["rank"] = rank
        ranking.append(row)

    sources = {
        "neural_final_ranking": neural_metadata,
        "geometric_controls": geometric_metadata,
        "deepnet_outer_refit": deepnet_metadata,
    }
    raw_records = int(neural_metadata["raw_records_validated"]) + int(
        geometric_metadata["raw_records_validated"]
    ) + sum(
        int(metadata["raw_records_validated"])
        for metadata in deepnet_metadata.values()
    )
    coverage = {
        source: sorted(subjects) for source, subjects in sorted(identities.coverage.items())
    }
    return {
        "schema": SCHEMA,
        "created_at": _utc_now(),
        "dataset": DATASET,
        "evidence_scope": "post_selection_local_development_not_confirmation",
        "confirmation_evidence": False,
        "selection_warning": (
            "The winner was selected after comparing 50 conditions, and every R4 "
            "recording had been opened during prior project development. All intervals "
            "and winner-versus-runner statistics are descriptive, not confirmatory."
        ),
        "protocol": {
            "subjects": list(SUBJECTS),
            "seeds": list(SEEDS),
            "folds": list(FOLDS),
            "split": "R1-2 fit; R3 selects; fresh R1-3 refit; R4 prediction-only",
            "test_trials_per_subject_seed": 60,
            "test_class_counts": [30, 30],
            "aggregation": "average seeds within subject, then equal-weight eight subjects",
            "primary_metric": "balanced_accuracy",
            "inferential_unit": "subject (n=8)",
        },
        "condition_counts": {
            CATEGORY_NEURAL: 43,
            CATEGORY_GEOMETRIC: 3,
            CATEGORY_PROCEDURE: 4,
            "total": 50,
        },
        "winner": winner,
        "co_winners": co_winners,
        "runner_up": runner_up,
        "winner_balanced_accuracy": top_mean,
        "winner_vs_runner_up": {
            "mean_delta": float(differences.mean()),
            "median_delta": float(np.median(differences)),
            "wins_ties_losses": [
                int(np.sum(differences > 0.0)),
                int(np.sum(differences == 0.0)),
                int(np.sum(differences < 0.0)),
            ],
            "paired_subject_bootstrap_95_ci": [
                float(value) for value in np.quantile(paired_samples, (0.025, 0.975))
            ],
            "exact_two_sided_sign_flip_p_descriptive": _exact_sign_flip(differences),
            "n_subjects": 8,
        },
        "bootstrap_repetitions": int(bootstrap_repetitions),
        "random_seed": int(random_seed),
        "ranking": ranking,
        "tables": {
            CATEGORY_NEURAL: [row for row in ranking if row["category"] == CATEGORY_NEURAL],
            CATEGORY_GEOMETRIC: [
                row for row in ranking if row["category"] == CATEGORY_GEOMETRIC
            ],
            CATEGORY_PROCEDURE: [
                row for row in ranking if row["category"] == CATEGORY_PROCEDURE
            ],
        },
        "audit": {
            "inputs": sources,
            "raw_prediction_records_validated": raw_records,
            "raw_test_identity_coverage": coverage,
            "neural_trial_level_limit": (
                None
                if neural_metadata["raw_records_validated"]
                else "The neural final-ranking schema contains complete subject/seed BAs "
                "but no trial predictions; raw neural artifacts were unavailable."
            ),
            "deepnet_preprocessing_and_covariance_contracts_identical": True,
            "deepnet_cache_hashes_matched_across_four_methods": True,
        },
        "comparison_boundaries": {
            "neural_table": (
                "43 common-training-recipe configurations; variants are not 43 "
                "independent architecture families"
            ),
            "geometric_table": (
                "current-cache, non-transductive controls; five seed identities are "
                "exact deterministic replicas"
            ),
            "procedure_table": (
                "same outer rows, but a separate label-free covariance view and pinned "
                "historical procedure-specific configurations"
            ),
        },
        "claims_not_supported": [
            "independent confirmation or population-level superiority",
            "state of the art or top global performance",
            "generalization to other datasets, subjects, montages, or sessions",
            "asynchronous, closed-loop, or clinical efficacy",
            "performance in disabled or paralyzed users",
            "literature novelty based on accuracy alone",
        ],
    }


def _percent(value: float) -> str:
    return f"{100.0 * float(value):.3f}%"


def _markdown_table(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = [
        "| Overall rank | Condition | Mean BA | Subject SD | 95% subject-bootstrap CI | Median | Minimum | Params |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        low, high = row["bootstrap_95_ci"]
        parameters = (
            "—" if row["parameter_count"] is None else f"{int(row['parameter_count']):,}"
        )
        lines.append(
            f"| {row['rank']} | `{row['name']}` | {_percent(row['balanced_accuracy_mean'])} | "
            f"{_percent(row['balanced_accuracy_standard_deviation'])} | "
            f"[{_percent(low)}, {_percent(high)}] | "
            f"{_percent(row['median_subject_balanced_accuracy'])} | "
            f"{_percent(row['minimum_subject_balanced_accuracy'])} | {parameters} |"
        )
    return lines


def render_markdown(report: Mapping[str, Any]) -> str:
    comparison = report["winner_vs_runner_up"]
    low, high = comparison["paired_subject_bootstrap_95_ci"]
    wins, ties, losses = comparison["wins_ties_losses"]
    lines = [
        "# Combined local Exp4 benchmark audit",
        "",
        "> **Development-only, post-selection result.** Every R4 recording was opened during prior project development. This is not independent confirmation evidence.",
        "",
        f"**Numerical winner:** `{report['winner']}` — {_percent(report['winner_balanced_accuracy'])} mean balanced accuracy.",
        "",
        f"Runner-up: `{report['runner_up']}`. Paired subject margin: "
        f"{100.0 * comparison['mean_delta']:+.3f} points; 95% paired subject-bootstrap "
        f"interval [{100.0 * low:+.3f}, {100.0 * high:+.3f}] points; "
        f"wins/ties/losses {wins}/{ties}/{losses}; exact two-sided sign-flip "
        f"p={comparison['exact_two_sided_sign_flip_p_descriptive']:.6f} (descriptive).",
        "",
        "The inferential unit is the participant (n=8): five seeds are averaged within each participant, then the eight participants are weighted equally.",
        "",
        "## Common-recipe neural configurations (43)",
        "",
        "These are configurations/conditions, not 43 independent architecture families.",
        "",
        *_markdown_table(report["tables"][CATEGORY_NEURAL]),
        "",
        "## Current-cache geometric controls (3)",
        "",
        "These controls are non-transductive. Their five formal seed identities are exact deterministic replicas and are not five independent fits.",
        "",
        *_markdown_table(report["tables"][CATEGORY_GEOMETRIC]),
        "",
        "## Full-procedure comparators (4)",
        "",
        "These use the same outer rows but a separate label-free covariance view and pinned procedure-specific configurations; they are not raw-architecture apples-to-apples comparisons.",
        "",
        *_markdown_table(report["tables"][CATEGORY_PROCEDURE]),
        "",
        "## Protocol and audit boundary",
        "",
        "- Local participant-specific binary left/right MI: R1–2 fit, R3 selection, fresh R1–3 refit, and all 60 balanced R4 trials prediction-only.",
        "- Primary metric: subject-equal mean balanced accuracy after seed averaging.",
        f"- Raw prediction records validated: {report['audit']['raw_prediction_records_validated']:,}.",
        f"- Neural trial-level audit: {report['audit']['neural_trial_level_limit'] or 'all 1,720 raw neural prediction records validated.'}",
        "- Old R4-calibrated Riemann/EA-FBCSP numbers and post-hoc fusion scores are outside this ranking.",
        "",
        "## Claims this result does not support",
        "",
    ]
    lines.extend(f"- {claim}." for claim in report["claims_not_supported"])
    return "\n".join(lines) + "\n"


def write_report(
    report: Mapping[str, Any], *, output_json: Path, output_markdown: Path
) -> None:
    output_json = output_json.resolve()
    output_markdown = output_markdown.resolve()
    for path in (output_json, output_markdown):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing report: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    with output_markdown.open("x", encoding="utf-8") as stream:
        stream.write(render_markdown(report))
        stream.flush()
        os.fsync(stream.fileno())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neural-ranking", type=Path, required=True)
    parser.add_argument("--geometric-controls", type=Path, required=True)
    parser.add_argument("--cameo", type=Path, required=True)
    parser.add_argument("--hemiparity", type=Path, required=True)
    parser.add_argument("--parity-fuse", type=Path, required=True)
    parser.add_argument("--orbit-v3", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=100_000)
    parser.add_argument("--random-seed", type=int, default=20260721)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_report(
        neural_ranking=args.neural_ranking,
        geometric_controls=args.geometric_controls,
        deepnet_paths={
            "cameo": args.cameo,
            "hemiparity": args.hemiparity,
            "parity_fuse": args.parity_fuse,
            "orbit_v3": args.orbit_v3,
        },
        bootstrap_repetitions=int(args.bootstrap_repetitions),
        random_seed=int(args.random_seed),
    )
    write_report(
        report,
        output_json=args.output_json,
        output_markdown=args.output_markdown,
    )
    print(
        f"winner={report['winner']} "
        f"balanced_accuracy={report['winner_balanced_accuracy']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CATEGORY_GEOMETRIC",
    "CATEGORY_NEURAL",
    "CATEGORY_PROCEDURE",
    "DEEPNET_MODELS",
    "GEOMETRIC_MODELS",
    "NEURAL_CONFIGURATIONS",
    "SEEDS",
    "SUBJECTS",
    "TestIdentityRegistry",
    "_canonical_sha256",
    "_validate_neural_artifact",
    "build_report",
    "render_markdown",
    "strict_load",
    "write_report",
]
