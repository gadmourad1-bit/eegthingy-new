"""Aggregate-only analysis for the separate binary neural-procedure grid.

The producer in :mod:`benchmark.procedure_grid` never opens a test label for
scoring and never publishes a test metric.  This module first requires an exact
complete and quiescent prediction grid, then joins checksum-bound cache labels
to checksum-bound row vectors in memory.  Published artifacts contain scalar
aggregates only: no row indices, labels, class predictions, or probabilities.

These results remain a separately labelled procedure track.  They must not be
merged into the common raw-trial architecture table, interpreted as
confirmation evidence, or described as a global/SOTA comparison.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import math
import os
import re
import stat
import uuid
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
)

from . import procedure_grid

ANALYSIS_SCHEMA = "eeg-mi-binary-procedure-analysis-v1"
MANIFEST_SCHEMA = "eeg-mi-binary-procedure-analysis-manifest-v1"
TRACK = "binary_neural_procedures_separate_from_common_recipe"
DEFAULT_ECE_BINS = 15
METRIC_NAMES: tuple[str, ...] = (
    "accuracy",
    "balanced_accuracy",
    "chance_normalized_balanced_accuracy",
    "macro_f1",
    "cohen_kappa",
    "ovr_macro_auroc",
    "nll",
    "multiclass_brier",
    "ece",
)
TABLE_NAMES: tuple[str, ...] = (
    "job_metrics",
    "subject_seed_metrics",
    "subject_metrics",
    "dataset_summary",
    "overall_summary",
    "timing_summary",
)
PAYLOAD_FILENAMES: tuple[str, ...] = (
    "summary.json",
    *(f"{name}.csv" for name in TABLE_NAMES),
)
PUBLICATION_FILENAMES = frozenset((*PAYLOAD_FILENAMES, "manifest.json"))
EXPECTED_FORMAL_JOBS = procedure_grid.FORMAL_EXPECTED_JOBS
FORBIDDEN_OUTPUT_KEYS = frozenset(
    {
        "label",
        "labels",
        "prediction",
        "predictions",
        "probability",
        "probabilities",
        "row",
        "rows",
        "target",
        "targets",
        "test_rows",
        "y",
        "y_true",
    }
)
OUTCOME_ALIAS_TOKENS = frozenset(
    {
        "groundtruth",
        "label",
        "labels",
        "outcome",
        "outcomes",
        "prediction",
        "predictions",
        "probability",
        "probabilities",
        "target",
        "targets",
        "truth",
        "ytrue",
    }
)
LEGITIMATE_ALIAS_KEYS = frozenset({"table_row_counts"})
SUMMARY_KEYS = frozenset(
    {
        "schema",
        "track",
        "publication_mode",
        "formal_publication_eligible",
        "plan_sha256",
        "input_ledger_sha256",
        "evidence_scope",
        "confirmation_evidence",
        "common_recipe_result",
        "global_sota_claim",
        "interpretation",
        "procedure_ids",
        "blocked_procedure_ids",
        "datasets",
        "seeds",
        "expected_jobs",
        "aggregation_definition",
        "descriptive_ranking_only",
        "inferential_comparisons",
        "metric_names",
        "ece_bins",
        "audit_snapshot",
        "source_identity",
        "environment_identity_sha256",
        "table_row_counts",
    }
)
AUDIT_SNAPSHOT_KEYS = frozenset(
    {
        "schema",
        "plan_sha256",
        "publication_mode",
        "formal_publication_eligible",
        "expected_jobs",
        "complete_jobs",
        "missing_jobs",
        "corrupt_jobs",
        "complete",
        "missing_job_ids",
        "corrupt_job_ids",
        "extra_record_directories",
        "unexpected_record_paths",
        "residual_claim_paths",
        "residual_partial_paths",
        "unexpected_root_entries",
        "unsafe_paths",
    }
)
METRIC_MEAN_KEYS = frozenset(
    key
    for metric in METRIC_NAMES
    for key in (metric, f"{metric}_defined_count")
)
JOB_ROW_KEYS = frozenset(
    {
        "track",
        "dataset",
        "stable_id",
        "subject",
        "fold",
        "seed",
        "job_id",
        "test_count",
        *METRIC_NAMES,
        "selection_fit_seconds",
        "refit_fit_seconds",
        "test_inference_seconds",
        "job_total_seconds",
        "inference_ms_per_trial",
        "parameter_count",
    }
)
SUBJECT_SEED_ROW_KEYS = frozenset(
    {
        "track",
        "dataset",
        "stable_id",
        "subject",
        "seed",
        "folds_concatenated",
        "test_trials",
        *METRIC_NAMES,
    }
)
SUBJECT_ROW_KEYS = frozenset(
    {
        "track",
        "dataset",
        "stable_id",
        "subject",
        "seeds_averaged",
        "aggregation",
        *METRIC_MEAN_KEYS,
    }
)
DATASET_ROW_KEYS = frozenset(
    {
        "track",
        "dataset",
        "stable_id",
        "subjects_averaged",
        "aggregation",
        *METRIC_MEAN_KEYS,
    }
)
OVERALL_ROW_KEYS = frozenset(
    {
        "track",
        "stable_id",
        "datasets_equal_weighted",
        "aggregation",
        "descriptive_rank",
        *METRIC_MEAN_KEYS,
    }
)
TIMING_BASES = (
    "selection_fit_seconds",
    "refit_fit_seconds",
    "test_inference_seconds",
    "job_total_seconds",
    "inference_ms_per_trial",
    "parameter_count",
)
TIMING_ROW_KEYS = frozenset(
    {
        "track",
        "dataset",
        "stable_id",
        "jobs",
        *(
            f"{base}_{suffix}"
            for base in TIMING_BASES
            for suffix in ("mean", "median", "p95")
        ),
    }
)
TABLE_ROW_KEYS: Mapping[str, frozenset[str]] = {
    "job_metrics": JOB_ROW_KEYS,
    "subject_seed_metrics": SUBJECT_SEED_ROW_KEYS,
    "subject_metrics": SUBJECT_ROW_KEYS,
    "dataset_summary": DATASET_ROW_KEYS,
    "overall_summary": OVERALL_ROW_KEYS,
    "timing_summary": TIMING_ROW_KEYS,
}


class ProcedureAnalysisError(RuntimeError):
    """Raised when analysis cannot preserve its score-joining boundary."""


@dataclass
class AnalysisResult:
    summary: dict[str, Any]
    tables: dict[str, list[dict[str, Any]]]
    input_ledger: list[dict[str, str]] = field(repr=False)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(child) for child in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        result = float(value)
        return result if math.isfinite(result) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _json_ready(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def classification_metrics(
    labels: np.ndarray | Sequence[int],
    probabilities: np.ndarray,
    *,
    n_classes: int = 2,
    ece_bins: int = DEFAULT_ECE_BINS,
) -> dict[str, float | None]:
    y = np.asarray(labels, dtype=np.int64)
    values = np.asarray(probabilities, dtype=np.float64)
    if (
        y.ndim != 1
        or len(y) == 0
        or n_classes < 2
        or ece_bins < 2
        or values.shape != (len(y), n_classes)
        or np.any(y < 0)
        or np.any(y >= n_classes)
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(values > 1.0)
        or not np.allclose(values.sum(axis=1), 1.0, atol=1e-6)
    ):
        raise ProcedureAnalysisError("invalid arrays for classification metrics")
    predicted = np.argmax(values, axis=1)
    classes = np.arange(n_classes, dtype=np.int64)
    support = np.bincount(y, minlength=n_classes)
    correct = np.bincount(y[predicted == y], minlength=n_classes)
    present = support > 0
    balanced_accuracy = float(np.mean(correct[present] / support[present]))
    chance = 1.0 / n_classes
    normalized = (balanced_accuracy - chance) / (1.0 - chance)
    aucs: list[float] = []
    for class_index in range(n_classes):
        binary = (y == class_index).astype(np.int8)
        if binary.min() == binary.max():
            aucs = []
            break
        aucs.append(float(roc_auc_score(binary, values[:, class_index])))
    macro_auc = float(np.mean(aucs)) if len(aucs) == n_classes else None
    confidence = values[np.arange(len(y)), predicted]
    correctness = (predicted == y).astype(np.float64)
    bin_ids = np.minimum(np.floor(confidence * ece_bins).astype(np.int64), ece_bins - 1)
    ece = 0.0
    for bin_index in range(ece_bins):
        mask = bin_ids == bin_index
        if np.any(mask):
            ece += float(mask.mean()) * abs(
                float(correctness[mask].mean()) - float(confidence[mask].mean())
            )
    one_hot = np.eye(n_classes, dtype=np.float64)[y]
    true_probability = np.clip(values[np.arange(len(y)), y], 1e-15, 1.0)
    return {
        "accuracy": float(np.mean(predicted == y)),
        "balanced_accuracy": balanced_accuracy,
        "chance_normalized_balanced_accuracy": float(normalized),
        "macro_f1": float(
            f1_score(
                y,
                predicted,
                labels=classes.tolist(),
                average="macro",
                zero_division=0,
            )
        ),
        "cohen_kappa": _finite_float(
            cohen_kappa_score(y, predicted, labels=classes.tolist())
        ),
        "ovr_macro_auroc": macro_auc,
        "nll": float(-np.log(true_probability).mean()),
        "multiclass_brier": float(np.square(values - one_hot).sum(axis=1).mean()),
        "ece": float(ece),
    }


def _metric_means(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, float | int | None]:
    result: dict[str, float | int | None] = {}
    for metric in METRIC_NAMES:
        values = [float(row[metric]) for row in rows if row.get(metric) is not None]
        result[metric] = float(np.mean(values)) if values else None
        result[f"{metric}_defined_count"] = len(values)
    return result


def _stable_audit(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(child)
        for key, child in value.items()
        if key != "complete_job_ids"
    }


def audit_exact_grid(
    run_root: str | Path,
    *,
    require_formal: bool = True,
) -> tuple[dict[str, Any], tuple[procedure_grid.Job, ...], dict[str, Any]]:
    root = procedure_grid._safe_run_root(Path(run_root))
    plan = procedure_grid.load_plan(root)
    procedure_grid.validate_plan_semantics(plan)
    jobs = tuple(procedure_grid.iter_jobs(plan))
    if len(jobs) != int(plan.get("n_jobs", -1)):
        raise ProcedureAnalysisError("plan job count is internally inconsistent")
    if require_formal and (
        plan.get("publication_mode") != procedure_grid.FORMAL_PUBLICATION_MODE
        or plan.get("analysis_config") != {"ece_bins": DEFAULT_ECE_BINS}
        or tuple(plan.get("dataset_order", ())) != procedure_grid.BINARY_DATASETS
        or tuple(plan.get("procedure_ids", ()))
        != tuple(value.stable_id for value in procedure_grid.PROCEDURES)
        or tuple(plan.get("seeds", ())) != procedure_grid.FORMAL_SEEDS
        or len(jobs) != EXPECTED_FORMAL_JOBS
    ):
        raise ProcedureAnalysisError(
            "analysis requires the exact 8,780-job formal procedure grid"
        )
    audit = procedure_grid.audit_grid(root, plan)
    if (
        audit.get("plan_sha256") != plan.get("plan_sha256")
        or audit.get("complete") is not True
        or int(audit.get("complete_jobs", -1)) != len(jobs)
        or int(audit.get("missing_jobs", -1)) != 0
        or int(audit.get("corrupt_jobs", -1)) != 0
        or set(audit.get("complete_job_ids", ())) != {job.job_id for job in jobs}
        or audit.get("extra_record_directories") != []
        or audit.get("unexpected_record_paths") != []
        or audit.get("residual_claim_paths") != []
        or audit.get("residual_partial_paths") != []
        or audit.get("unexpected_root_entries") != []
        or audit.get("unsafe_paths") != []
        or (
            require_formal
            and audit.get("formal_publication_eligible") is not True
        )
    ):
        raise ProcedureAnalysisError(
            "refusing label access: grid is incomplete, active, stray, or corrupt"
        )
    return plan, jobs, _stable_audit(audit)


def _bound_labels(
    plan: Mapping[str, Any], cache_root: Path
) -> tuple[dict[tuple[str, int], np.ndarray], list[dict[str, str]]]:
    from .data import load_subject_cache, split_indices

    labels_by_subject: dict[tuple[str, int], np.ndarray] = {}
    ledger: list[dict[str, str]] = []
    for dataset in plan["dataset_order"]:
        contract = plan["datasets"][dataset]
        for raw_subject in contract["subjects"]:
            subject = int(raw_subject)
            key = procedure_grid._subject_key(str(dataset), subject)
            planned = plan["cache_identity"].get(key)
            if not isinstance(planned, Mapping):
                raise ProcedureAnalysisError(f"cache identity is absent for {key}")
            data = load_subject_cache(str(dataset), subject, cache_root=cache_root)
            observed_identity = procedure_grid._cache_identity_from_loaded(data)
            if observed_identity != planned:
                raise ProcedureAnalysisError(
                    f"cache identity differs from plan for {key}"
                )
            labels = np.asarray(data["y"])
            if (
                labels.dtype != np.int64
                or labels.ndim != 1
                or len(labels) != int(planned["trial_count"])
                or set(labels.tolist()) != {0, 1}
            ):
                raise ProcedureAnalysisError(f"invalid binary labels for {key}")
            seen_test: set[int] = set()
            split_payload: dict[str, Any] = {}
            for raw_fold in contract["folds"]:
                fold = int(raw_fold)
                train, validation, test = split_indices(
                    str(dataset),
                    labels,
                    data["sessions"],
                    data["runs"],
                    fold=fold,
                    subject=subject,
                )
                observed = procedure_grid._one_split_identity(
                    dataset=str(dataset),
                    subject=subject,
                    fold=fold,
                    cache_array_sha256=str(data["identity"]["array_sha256"]),
                    trial_count=len(labels),
                    train_rows=train,
                    validation_rows=validation,
                    test_rows=test,
                )
                split_key = procedure_grid._split_key(str(dataset), subject, fold)
                if observed != plan["split_identity"].get(split_key):
                    raise ProcedureAnalysisError(
                        f"split identity differs for {split_key}"
                    )
                test_set = set(np.asarray(test, dtype=np.int64).tolist())
                if seen_test.intersection(test_set):
                    raise ProcedureAnalysisError(f"test folds overlap for {key}")
                seen_test.update(test_set)
                split_payload[split_key] = observed
            labels_by_subject[(str(dataset), subject)] = labels.copy()
            ledger.append(
                {
                    "kind": "cache",
                    "identity": key,
                    "sha256_a": str(planned["array_sha256"]),
                    "sha256_b": _sha256_bytes(_canonical_bytes(observed_identity)),
                    "sha256_c": _sha256_bytes(_canonical_bytes(split_payload)),
                }
            )
    return labels_by_subject, ledger


def _job_ledger_row(
    job: procedure_grid.Job,
    payload: Mapping[str, Any],
) -> dict[str, str]:
    hashes = payload.get("file_sha256")
    if not isinstance(hashes, Mapping) or set(hashes) != {
        "record.json",
        "predictions.npz",
        "completion.json",
    }:
        raise ProcedureAnalysisError("validated completion has no exact hash ledger")
    return {
        "kind": "job",
        "identity": job.job_id,
        "sha256_a": str(hashes["record.json"]),
        "sha256_b": str(hashes["predictions.npz"]),
        "sha256_c": str(hashes["completion.json"]),
    }


def _plan_ledger_row(run_root: Path, plan: Mapping[str, Any]) -> dict[str, str]:
    del run_root
    plan_digest = str(plan["plan_sha256"])
    return {
        "kind": "plan",
        "identity": plan_digest,
        "sha256_a": _sha256_bytes(_canonical_bytes(plan) + b"\n"),
        "sha256_b": _sha256_bytes((plan_digest + "\n").encode("ascii")),
        "sha256_c": plan_digest,
    }


def _job_metric_row(
    job: procedure_grid.Job,
    payload: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    timing = payload["record"]["metadata"]["timing_seconds"]
    test_count = int(payload["record"]["test_count"])
    return {
        "track": TRACK,
        **job.identity(),
        "job_id": job.job_id,
        "test_count": test_count,
        **dict(metrics),
        "selection_fit_seconds": float(timing["selection_fit"]),
        "refit_fit_seconds": float(timing["refit_fit"]),
        "test_inference_seconds": float(timing["test_inference"]),
        "job_total_seconds": float(timing["job_total"]),
        "inference_ms_per_trial": (
            1000.0 * float(timing["test_inference"]) / test_count
        ),
        "parameter_count": int(payload["record"]["metadata"]["fit"]["parameter_count"]),
    }


def _timing_summary(
    job_rows: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    overall: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in job_rows:
        grouped[(str(row["dataset"]), str(row["stable_id"]))].append(row)
        overall[str(row["stable_id"])].append(row)
    bases = (
        "selection_fit_seconds",
        "refit_fit_seconds",
        "test_inference_seconds",
        "job_total_seconds",
        "inference_ms_per_trial",
        "parameter_count",
    )
    result: list[dict[str, Any]] = []
    for dataset in [*plan["dataset_order"], "ALL_DATASETS"]:
        for stable_id in plan["procedure_ids"]:
            rows = (
                overall[str(stable_id)]
                if dataset == "ALL_DATASETS"
                else grouped[(str(dataset), str(stable_id))]
            )
            row: dict[str, Any] = {
                "track": TRACK,
                "dataset": str(dataset),
                "stable_id": str(stable_id),
                "jobs": len(rows),
            }
            for base in bases:
                values = np.asarray(
                    [float(value[base]) for value in rows],
                    dtype=np.float64,
                )
                row[f"{base}_mean"] = float(values.mean())
                row[f"{base}_median"] = float(np.median(values))
                row[f"{base}_p95"] = float(np.quantile(values, 0.95))
            result.append(row)
    return result


def _reject_forbidden_output_keys(value: Any, *, path: str = "analysis") -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).strip().lower()
            compact = re.sub(r"[^a-z0-9]+", "", key)
            if key in FORBIDDEN_OUTPUT_KEYS or (
                key not in LEGITIMATE_ALIAS_KEYS
                and any(token in compact for token in OUTCOME_ALIAS_TOKENS)
            ):
                raise ProcedureAnalysisError(
                    f"analysis output contains forbidden key {path}.{raw_key}"
                )
            _reject_forbidden_output_keys(child, path=f"{path}.{raw_key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_forbidden_output_keys(child, path=f"{path}[{index}]")


def _same_scalar(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is right
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(
            float(left), float(right), rel_tol=1e-13, abs_tol=1e-15
        )
    return left == right


def _require_metric_means(
    row: Mapping[str, Any],
    children: Sequence[Mapping[str, Any]],
    *,
    path: str,
) -> None:
    expected = _metric_means(children)
    for key, value in expected.items():
        if not _same_scalar(row.get(key), value):
            raise ProcedureAnalysisError(f"{path}.{key} is not the defined mean")


def _validate_metric_values(row: Mapping[str, Any], *, path: str) -> None:
    bounded_zero_one = {
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "ovr_macro_auroc",
        "ece",
    }
    bounded_signed = {"chance_normalized_balanced_accuracy", "cohen_kappa"}
    for metric in METRIC_NAMES:
        value = row.get(metric)
        if value is None:
            if metric not in {"cohen_kappa", "ovr_macro_auroc"}:
                raise ProcedureAnalysisError(f"{path}.{metric} is unexpectedly null")
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ProcedureAnalysisError(f"{path}.{metric} is not numeric")
        number = float(value)
        if not math.isfinite(number):
            raise ProcedureAnalysisError(f"{path}.{metric} is non-finite")
        if metric in bounded_zero_one and not 0.0 <= number <= 1.0:
            raise ProcedureAnalysisError(f"{path}.{metric} is outside [0,1]")
        if metric in bounded_signed and not -1.0 <= number <= 1.0:
            raise ProcedureAnalysisError(f"{path}.{metric} is outside [-1,1]")
        if metric == "nll" and number < 0.0:
            raise ProcedureAnalysisError(f"{path}.nll is negative")
        if metric == "multiclass_brier" and not 0.0 <= number <= 2.0:
            raise ProcedureAnalysisError(f"{path}.multiclass_brier is invalid")
        count_key = f"{metric}_defined_count"
        if count_key in row and (
            isinstance(row[count_key], bool)
            or not isinstance(row[count_key], int)
            or row[count_key] < 0
        ):
            raise ProcedureAnalysisError(f"{path}.{count_key} is invalid")


def _validate_result(result: AnalysisResult) -> None:
    if set(result.tables) != set(TABLE_NAMES):
        raise ProcedureAnalysisError("analysis table roster is invalid")
    if not isinstance(result.summary, dict) or set(result.summary) != set(SUMMARY_KEYS):
        raise ProcedureAnalysisError("analysis summary exact schema is invalid")
    summary = result.summary
    if (
        summary["schema"] != ANALYSIS_SCHEMA
        or summary["track"] != TRACK
        or summary["publication_mode"]
        not in {
            procedure_grid.FORMAL_PUBLICATION_MODE,
            procedure_grid.TEST_PUBLICATION_MODE,
        }
        or summary["formal_publication_eligible"]
        is not (
            summary["publication_mode"] == procedure_grid.FORMAL_PUBLICATION_MODE
        )
        or summary["confirmation_evidence"] is not False
        or summary["common_recipe_result"] is not False
        or summary["global_sota_claim"] is not False
        or summary["descriptive_ranking_only"] is not True
        or summary["evidence_scope"]
        != "opened development cohorts only; no confirmation or SOTA claim"
        or summary["interpretation"]
        != (
            "opened-cohort development evidence for separately trained neural "
            "procedures; no global/SOTA or easy-publication claim"
        )
        or summary["aggregation_definition"]
        != (
            "concatenate disjoint held-out folds within subject/seed; compute "
            "metrics; mean five seeds within subject; equal-weight subjects "
            "within dataset; equal-weight four dataset summaries"
        )
        or summary["inferential_comparisons"]
        != (
            "not prespecified in this runner revision; no p-value or "
            "confirmatory superiority claim"
        )
        or summary["metric_names"] != list(METRIC_NAMES)
        or summary["ece_bins"] != DEFAULT_ECE_BINS
        or summary["blocked_procedure_ids"]
        != [str(value["stable_id"]) for value in procedure_grid.BLOCKED_ORBIT_PROVENANCE]
        or not isinstance(summary["plan_sha256"], str)
        or not procedure_grid.HEX_64_RE.fullmatch(summary["plan_sha256"])
        or not isinstance(summary["environment_identity_sha256"], str)
        or not procedure_grid.HEX_64_RE.fullmatch(
            summary["environment_identity_sha256"]
        )
        or not isinstance(summary["source_identity"], Mapping)
        or not summary["source_identity"]
        or any(
            not isinstance(name, str)
            or not name
            or not isinstance(digest, str)
            or not procedure_grid.HEX_64_RE.fullmatch(digest)
            for name, digest in summary["source_identity"].items()
        )
    ):
        raise ProcedureAnalysisError("analysis summary contract is invalid")
    if (
        not isinstance(summary["procedure_ids"], list)
        or not summary["procedure_ids"]
        or len(summary["procedure_ids"]) != len(set(summary["procedure_ids"]))
        or any(value not in procedure_grid.PROCEDURE_BY_ID for value in summary["procedure_ids"])
        or not isinstance(summary["datasets"], list)
        or not summary["datasets"]
        or len(summary["datasets"]) != len(set(summary["datasets"]))
        or not isinstance(summary["seeds"], list)
        or not summary["seeds"]
        or len(summary["seeds"]) != len(set(summary["seeds"]))
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in summary["seeds"]
        )
        or not isinstance(summary["expected_jobs"], int)
        or isinstance(summary["expected_jobs"], bool)
        or summary["expected_jobs"] <= 0
    ):
        raise ProcedureAnalysisError("analysis summary roster is invalid")
    if summary["publication_mode"] == procedure_grid.FORMAL_PUBLICATION_MODE and (
        summary["procedure_ids"]
        != [value.stable_id for value in procedure_grid.PROCEDURES]
        or summary["datasets"] != list(procedure_grid.BINARY_DATASETS)
        or summary["seeds"] != list(procedure_grid.FORMAL_SEEDS)
        or summary["expected_jobs"] != EXPECTED_FORMAL_JOBS
    ):
        raise ProcedureAnalysisError("formal analysis roster is not exact")
    if (
        not isinstance(summary["audit_snapshot"], Mapping)
        or set(summary["audit_snapshot"]) != set(AUDIT_SNAPSHOT_KEYS)
        or summary["audit_snapshot"]["complete"] is not True
        or summary["audit_snapshot"]["plan_sha256"] != summary["plan_sha256"]
        or summary["audit_snapshot"]["publication_mode"]
        != summary["publication_mode"]
        or summary["audit_snapshot"]["formal_publication_eligible"]
        is not summary["formal_publication_eligible"]
        or summary["audit_snapshot"]["expected_jobs"] != summary["expected_jobs"]
        or summary["audit_snapshot"]["complete_jobs"] != summary["expected_jobs"]
        or any(
            summary["audit_snapshot"][key]
            for key in (
                "missing_jobs",
                "corrupt_jobs",
                "missing_job_ids",
                "corrupt_job_ids",
                "extra_record_directories",
                "unexpected_record_paths",
                "residual_claim_paths",
                "residual_partial_paths",
                "unexpected_root_entries",
                "unsafe_paths",
            )
        )
    ):
        raise ProcedureAnalysisError("analysis audit snapshot is invalid")
    if result.summary.get("table_row_counts") != {
        name: len(rows) for name, rows in result.tables.items()
    }:
        raise ProcedureAnalysisError("analysis table counts are stale")
    _reject_forbidden_output_keys(result.summary)
    _reject_forbidden_output_keys(result.tables)
    for name, rows in result.tables.items():
        if not isinstance(rows, list) or not rows:
            raise ProcedureAnalysisError(f"table {name} is empty")
        for index, row in enumerate(rows):
            if not isinstance(row, dict) or set(row) != set(TABLE_ROW_KEYS[name]):
                raise ProcedureAnalysisError(
                    f"table {name} row {index} has an invalid exact schema"
                )
            if row["track"] != TRACK:
                raise ProcedureAnalysisError(f"table {name} escaped the track")
            for value in row.values():
                if value is not None and not isinstance(value, (str, int, float, bool)):
                    raise ProcedureAnalysisError(
                        f"table {name} contains a nonscalar value"
                    )
                if isinstance(value, float) and not math.isfinite(value):
                    raise ProcedureAnalysisError(
                        f"table {name} contains a non-finite value"
                    )
            if name != "timing_summary":
                _validate_metric_values(row, path=f"{name}[{index}]")
    jobs = result.tables["job_metrics"]
    if len(jobs) != summary["expected_jobs"]:
        raise ProcedureAnalysisError("job metric cardinality is invalid")
    job_keys: set[tuple[str, str, int, int, int]] = set()
    for row in jobs:
        if (
            not isinstance(row["dataset"], str)
            or not isinstance(row["stable_id"], str)
            or not isinstance(row["job_id"], str)
            or any(
                isinstance(row[name], bool) or not isinstance(row[name], int)
                for name in ("subject", "fold", "seed")
            )
        ):
            raise ProcedureAnalysisError("job metric scalar identity is invalid")
        key = (
            str(row["dataset"]),
            str(row["stable_id"]),
            int(row["subject"]),
            int(row["fold"]),
            int(row["seed"]),
        )
        if (
            row["dataset"] not in summary["datasets"]
            or row["stable_id"] not in summary["procedure_ids"]
            or row["seed"] not in summary["seeds"]
            or key in job_keys
            or row["job_id"]
            != procedure_grid.Job(
                dataset=key[0],
                stable_id=key[1],
                subject=key[2],
                fold=key[3],
                seed=key[4],
            ).job_id
            or isinstance(row["test_count"], bool)
            or not isinstance(row["test_count"], int)
            or row["test_count"] <= 0
            or isinstance(row["parameter_count"], bool)
            or not isinstance(row["parameter_count"], int)
            or row["parameter_count"] <= 0
            or any(
                isinstance(row[name], bool)
                or not isinstance(row[name], (int, float))
                or float(row[name]) < 0.0
                for name in (
                    "selection_fit_seconds",
                    "refit_fit_seconds",
                    "test_inference_seconds",
                    "job_total_seconds",
                    "inference_ms_per_trial",
                )
            )
            or not math.isclose(
                float(row["inference_ms_per_trial"]),
                1000.0
                * float(row["test_inference_seconds"])
                / int(row["test_count"]),
                rel_tol=1e-13,
                abs_tol=1e-15,
            )
            or float(row["job_total_seconds"])
            + 1e-12
            < (
                float(row["selection_fit_seconds"])
                + float(row["refit_fit_seconds"])
                + float(row["test_inference_seconds"])
            )
        ):
            raise ProcedureAnalysisError("job metric identity/runtime is invalid")
        job_keys.add(key)
    if summary["publication_mode"] == procedure_grid.FORMAL_PUBLICATION_MODE:
        contracts = procedure_grid._dataset_contracts()
        exact_job_keys = {
            (dataset, stable_id, int(subject), int(fold), int(seed))
            for stable_id in summary["procedure_ids"]
            for dataset in summary["datasets"]
            for subject in contracts[dataset]["subjects"]
            for fold in contracts[dataset]["folds"]
            for seed in summary["seeds"]
        }
        if job_keys != exact_job_keys or len(job_keys) != EXPECTED_FORMAL_JOBS:
            raise ProcedureAnalysisError("formal job table roster is not exact")

    subject_seed_rows = result.tables["subject_seed_metrics"]
    job_groups: dict[tuple[str, str, int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in jobs:
        job_groups[
            (
                str(row["dataset"]),
                str(row["stable_id"]),
                int(row["subject"]),
                int(row["seed"]),
            )
        ].append(row)
    subject_seed_by_key: dict[
        tuple[str, str, int, int], Mapping[str, Any]
    ] = {}
    for row in subject_seed_rows:
        if any(
            isinstance(row[name], bool) or not isinstance(row[name], int)
            for name in (
                "subject",
                "seed",
                "folds_concatenated",
                "test_trials",
            )
        ):
            raise ProcedureAnalysisError("subject/seed scalar identity is invalid")
        key = (
            str(row["dataset"]),
            str(row["stable_id"]),
            int(row["subject"]),
            int(row["seed"]),
        )
        children = job_groups.get(key, [])
        if (
            key in subject_seed_by_key
            or not children
            or row["folds_concatenated"] != len(children)
            or row["test_trials"] != sum(int(child["test_count"]) for child in children)
        ):
            raise ProcedureAnalysisError("subject/seed cardinality is invalid")
        subject_seed_by_key[key] = row
    if set(subject_seed_by_key) != set(job_groups):
        raise ProcedureAnalysisError("subject/seed roster is incomplete")

    subject_rows = result.tables["subject_metrics"]
    seed_groups: dict[tuple[str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for key, row in subject_seed_by_key.items():
        seed_groups[key[:3]].append(row)
    subject_by_key: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for row in subject_rows:
        if any(
            isinstance(row[name], bool) or not isinstance(row[name], int)
            for name in ("subject", "seeds_averaged")
        ):
            raise ProcedureAnalysisError("subject scalar identity is invalid")
        key = (str(row["dataset"]), str(row["stable_id"]), int(row["subject"]))
        children = seed_groups.get(key, [])
        if (
            key in subject_by_key
            or {child["seed"] for child in children} != set(summary["seeds"])
            or row["seeds_averaged"] != len(summary["seeds"])
            or row["aggregation"] != "concatenate folds then mean five seeds"
        ):
            raise ProcedureAnalysisError("subject aggregation roster is invalid")
        _require_metric_means(row, children, path=f"subject_metrics[{key}]")
        subject_by_key[key] = row
    if set(subject_by_key) != set(seed_groups):
        raise ProcedureAnalysisError("subject aggregation is incomplete")

    dataset_rows = result.tables["dataset_summary"]
    subject_groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for key, row in subject_by_key.items():
        subject_groups[key[:2]].append(row)
    dataset_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in dataset_rows:
        if (
            isinstance(row["subjects_averaged"], bool)
            or not isinstance(row["subjects_averaged"], int)
        ):
            raise ProcedureAnalysisError("dataset scalar identity is invalid")
        key = (str(row["dataset"]), str(row["stable_id"]))
        children = subject_groups.get(key, [])
        if (
            key in dataset_by_key
            or not children
            or row["subjects_averaged"] != len(children)
            or row["aggregation"]
            != "concatenate folds; mean seeds within subject; equal-weight subjects"
        ):
            raise ProcedureAnalysisError("dataset aggregation roster is invalid")
        _require_metric_means(row, children, path=f"dataset_summary[{key}]")
        dataset_by_key[key] = row
    expected_dataset_keys = {
        (dataset, stable_id)
        for dataset in summary["datasets"]
        for stable_id in summary["procedure_ids"]
    }
    if set(dataset_by_key) != expected_dataset_keys:
        raise ProcedureAnalysisError("dataset summary cardinality is invalid")

    overall_rows = result.tables["overall_summary"]
    overall_by_id: dict[str, Mapping[str, Any]] = {}
    for row in overall_rows:
        if (
            isinstance(row["datasets_equal_weighted"], bool)
            or not isinstance(row["datasets_equal_weighted"], int)
            or isinstance(row["descriptive_rank"], bool)
            or not isinstance(row["descriptive_rank"], int)
        ):
            raise ProcedureAnalysisError("overall scalar identity is invalid")
        stable_id = str(row["stable_id"])
        children = [
            dataset_by_key[(dataset, stable_id)] for dataset in summary["datasets"]
        ]
        if (
            stable_id in overall_by_id
            or row["datasets_equal_weighted"] != len(summary["datasets"])
            or row["aggregation"]
            != (
                "fold concatenation; seed mean; subject mean; "
                "equal four-dataset mean"
            )
        ):
            raise ProcedureAnalysisError("overall aggregation roster is invalid")
        _require_metric_means(row, children, path=f"overall_summary[{stable_id}]")
        overall_by_id[stable_id] = row
    if set(overall_by_id) != set(summary["procedure_ids"]):
        raise ProcedureAnalysisError("overall summary cardinality is invalid")
    ranked = sorted(
        overall_rows,
        key=lambda value: (-float(value["balanced_accuracy"]), str(value["stable_id"])),
    )
    if any(row["descriptive_rank"] != index for index, row in enumerate(ranked, 1)):
        raise ProcedureAnalysisError("descriptive ranks are inconsistent")

    for row in result.tables["timing_summary"]:
        if (
            not isinstance(row["dataset"], str)
            or not isinstance(row["stable_id"], str)
            or isinstance(row["jobs"], bool)
            or not isinstance(row["jobs"], int)
            or row["jobs"] <= 0
            or any(
                isinstance(row[key], bool)
                or not isinstance(row[key], (int, float))
                or not math.isfinite(float(row[key]))
                or float(row[key]) < 0.0
                for key in TIMING_ROW_KEYS
                if key not in {"track", "dataset", "stable_id", "jobs"}
            )
        ):
            raise ProcedureAnalysisError("timing summary scalar is invalid")
    timing_expected = _timing_summary(
        jobs,
        {
            "dataset_order": summary["datasets"],
            "procedure_ids": summary["procedure_ids"],
        },
    )
    if result.tables["timing_summary"] != timing_expected:
        raise ProcedureAnalysisError("timing summary is not derived from job rows")

    expected_counts = {
        "job_metrics": summary["expected_jobs"],
        "subject_seed_metrics": len(job_groups),
        "subject_metrics": len(seed_groups),
        "dataset_summary": len(summary["datasets"]) * len(summary["procedure_ids"]),
        "overall_summary": len(summary["procedure_ids"]),
        "timing_summary": (len(summary["datasets"]) + 1)
        * len(summary["procedure_ids"]),
    }
    if summary["table_row_counts"] != expected_counts:
        raise ProcedureAnalysisError("analysis table cardinalities are invalid")
    if result.summary.get("input_ledger_sha256") != _sha256_bytes(
        _canonical_bytes(result.input_ledger)
    ):
        raise ProcedureAnalysisError("input ledger digest is invalid")
    if not result.input_ledger:
        raise ProcedureAnalysisError("input ledger is empty")
    for index, row in enumerate(result.input_ledger):
        if (
            not isinstance(row, dict)
            or set(row) != {"kind", "identity", "sha256_a", "sha256_b", "sha256_c"}
            or row["kind"] not in {"plan", "cache", "job"}
            or not isinstance(row["identity"], str)
            or not row["identity"]
            or any(
                not isinstance(row[name], str)
                or len(row[name]) != 64
                or any(character not in "0123456789abcdef" for character in row[name])
                for name in ("sha256_a", "sha256_b", "sha256_c")
            )
        ):
            raise ProcedureAnalysisError(f"input ledger row {index} is invalid")
    subject_count = len({(row["dataset"], row["subject"]) for row in jobs})
    if (
        len(result.input_ledger) != 1 + subject_count + summary["expected_jobs"]
        or result.input_ledger[0]["kind"] != "plan"
        or [row["kind"] for row in result.input_ledger[1 : 1 + subject_count]]
        != ["cache"] * subject_count
        or [row["kind"] for row in result.input_ledger[1 + subject_count :]]
        != ["job"] * summary["expected_jobs"]
        or len(
            {(row["kind"], row["identity"]) for row in result.input_ledger}
        )
        != len(result.input_ledger)
    ):
        raise ProcedureAnalysisError("input ledger roster/order is invalid")


def compute_analysis(
    *,
    run_root: str | Path,
    cache_root: str | Path,
    ece_bins: int = DEFAULT_ECE_BINS,
    require_formal: bool = True,
    verify_runtime: bool = True,
) -> AnalysisResult:
    """Audit, join in memory, aggregate, re-audit, and seal scalar outputs."""

    if isinstance(ece_bins, bool) or ece_bins != DEFAULT_ECE_BINS:
        raise ValueError(f"procedure analysis freezes ECE at {DEFAULT_ECE_BINS} bins")
    if require_formal:
        procedure_grid._require_formal_release_runtime()
    root = procedure_grid._safe_run_root(Path(run_root))
    cache = procedure_grid._absolute_path(Path(cache_root))
    procedure_grid._require_no_symlink_ancestors(
        cache,
        allow_missing_tail=not cache.exists(),
    )
    if cache.exists():
        procedure_grid._require_real_directory(cache)
    plan, jobs, opening_audit = audit_exact_grid(root, require_formal=require_formal)
    if (
        plan["publication_mode"] == procedure_grid.FORMAL_PUBLICATION_MODE
        and not verify_runtime
    ):
        raise ProcedureAnalysisError(
            "formal analysis cannot bypass runtime identity verification"
        )
    if verify_runtime:
        procedure_grid.verify_runtime_identity(plan, cache_root=cache)
    labels_by_subject, cache_ledger = _bound_labels(plan, cache)

    job_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, int, int], dict[str, list[np.ndarray]]] = defaultdict(
        lambda: {"indices": [], "outputs": [], "outcomes": []}
    )
    job_ledger: list[dict[str, str]] = []
    for job in jobs:
        payload = procedure_grid.validate_completion(root, plan, job)
        indices = np.asarray(payload["rows"], dtype=np.int64)
        outputs = np.asarray(payload["probabilities"], dtype=np.float64)
        all_labels = labels_by_subject[(job.dataset, job.subject)]
        if (
            np.any(indices < 0)
            or np.any(indices >= len(all_labels))
            or len(np.unique(indices)) != len(indices)
        ):
            raise ProcedureAnalysisError(f"invalid prediction indices for {job.job_id}")
        outcomes = all_labels[indices]
        metrics = classification_metrics(
            outcomes, outputs, n_classes=2, ece_bins=ece_bins
        )
        job_rows.append(_job_metric_row(job, payload, metrics))
        key = (job.dataset, job.stable_id, job.subject, job.seed)
        grouped[key]["indices"].append(indices.copy())
        grouped[key]["outputs"].append(outputs.copy())
        grouped[key]["outcomes"].append(outcomes.copy())
        job_ledger.append(_job_ledger_row(job, payload))

    subject_seed_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    for dataset in plan["dataset_order"]:
        contract = plan["datasets"][dataset]
        for stable_id in plan["procedure_ids"]:
            selected_subjects: list[dict[str, Any]] = []
            for raw_subject in contract["subjects"]:
                subject = int(raw_subject)
                selected_seeds: list[dict[str, Any]] = []
                for raw_seed in plan["seeds"]:
                    seed = int(raw_seed)
                    group = grouped[(str(dataset), str(stable_id), subject, seed)]
                    indices = np.concatenate(group["indices"])
                    outputs = np.concatenate(group["outputs"])
                    outcomes = np.concatenate(group["outcomes"])
                    order = np.argsort(indices, kind="stable")
                    indices = indices[order]
                    outputs = outputs[order]
                    outcomes = outcomes[order]
                    if len(np.unique(indices)) != len(indices):
                        raise ProcedureAnalysisError(
                            f"fold test overlap for {dataset}/{stable_id}/"
                            f"S{subject}/seed{seed}"
                        )
                    row = {
                        "track": TRACK,
                        "dataset": str(dataset),
                        "stable_id": str(stable_id),
                        "subject": subject,
                        "seed": seed,
                        "folds_concatenated": len(contract["folds"]),
                        "test_trials": len(indices),
                        **classification_metrics(
                            outcomes,
                            outputs,
                            n_classes=2,
                            ece_bins=ece_bins,
                        ),
                    }
                    subject_seed_rows.append(row)
                    selected_seeds.append(row)
                subject_row = {
                    "track": TRACK,
                    "dataset": str(dataset),
                    "stable_id": str(stable_id),
                    "subject": subject,
                    "seeds_averaged": len(selected_seeds),
                    "aggregation": "concatenate folds then mean five seeds",
                    **_metric_means(selected_seeds),
                }
                subject_rows.append(subject_row)
                selected_subjects.append(subject_row)
            dataset_rows.append(
                {
                    "track": TRACK,
                    "dataset": str(dataset),
                    "stable_id": str(stable_id),
                    "subjects_averaged": len(selected_subjects),
                    "aggregation": (
                        "concatenate folds; mean seeds within subject; "
                        "equal-weight subjects"
                    ),
                    **_metric_means(selected_subjects),
                }
            )

    overall_rows: list[dict[str, Any]] = []
    for stable_id in plan["procedure_ids"]:
        selected = [row for row in dataset_rows if row["stable_id"] == str(stable_id)]
        if len(selected) != len(plan["dataset_order"]):
            raise ProcedureAnalysisError(
                f"dataset summaries are incomplete for {stable_id}"
            )
        overall_rows.append(
            {
                "track": TRACK,
                "stable_id": str(stable_id),
                "datasets_equal_weighted": len(selected),
                "aggregation": (
                    "fold concatenation; seed mean; subject mean; "
                    "equal four-dataset mean"
                ),
                **_metric_means(selected),
            }
        )
    descriptive_order = sorted(
        overall_rows,
        key=lambda value: (
            -float(value["balanced_accuracy"]),
            str(value["stable_id"]),
        ),
    )
    for rank, row in enumerate(descriptive_order, start=1):
        row["descriptive_rank"] = rank

    tables = {
        "job_metrics": job_rows,
        "subject_seed_metrics": subject_seed_rows,
        "subject_metrics": subject_rows,
        "dataset_summary": dataset_rows,
        "overall_summary": overall_rows,
        "timing_summary": _timing_summary(job_rows, plan),
    }
    ledger = [
        _plan_ledger_row(root, plan),
        *cache_ledger,
        *job_ledger,
    ]
    final_plan, final_jobs, final_audit = audit_exact_grid(
        root, require_formal=require_formal
    )
    if verify_runtime:
        procedure_grid.verify_runtime_identity(final_plan, cache_root=cache)
    if final_plan != plan or final_jobs != jobs or final_audit != opening_audit:
        raise ProcedureAnalysisError("plan or quiescent audit changed during analysis")
    summary = {
        "schema": ANALYSIS_SCHEMA,
        "track": TRACK,
        "publication_mode": plan["publication_mode"],
        "formal_publication_eligible": (
            plan["publication_mode"] == procedure_grid.FORMAL_PUBLICATION_MODE
        ),
        "plan_sha256": plan["plan_sha256"],
        "input_ledger_sha256": _sha256_bytes(_canonical_bytes(ledger)),
        "evidence_scope": plan["evidence_scope"],
        "confirmation_evidence": False,
        "common_recipe_result": False,
        "global_sota_claim": False,
        "interpretation": (
            "opened-cohort development evidence for separately trained neural "
            "procedures; no global/SOTA or easy-publication claim"
        ),
        "procedure_ids": list(plan["procedure_ids"]),
        "blocked_procedure_ids": list(plan["blocked_procedure_ids"]),
        "datasets": list(plan["dataset_order"]),
        "seeds": list(plan["seeds"]),
        "expected_jobs": len(jobs),
        "aggregation_definition": (
            "concatenate disjoint held-out folds within subject/seed; compute "
            "metrics; mean five seeds within subject; equal-weight subjects "
            "within dataset; equal-weight four dataset summaries"
        ),
        "descriptive_ranking_only": True,
        "inferential_comparisons": (
            "not prespecified in this runner revision; no p-value or "
            "confirmatory superiority claim"
        ),
        "metric_names": list(METRIC_NAMES),
        "ece_bins": int(ece_bins),
        "audit_snapshot": opening_audit,
        "source_identity": copy.deepcopy(plan["source_identity"]),
        "environment_identity_sha256": _sha256_bytes(
            _canonical_bytes(plan["environment_identity"])
        ),
        "table_row_counts": {name: len(rows) for name, rows in tables.items()},
    }
    result = AnalysisResult(summary=summary, tables=tables, input_ledger=ledger)
    _validate_result(result)
    return result


def _csv_bytes(path: Path, rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        raise ProcedureAnalysisError(f"refusing to write empty table {path}")
    fieldnames = list(rows[0])
    if any(list(row) != fieldnames for row in rows):
        raise ProcedureAnalysisError(f"table {path.name} has inconsistent column order")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(_json_ready(rows))
    return buffer.getvalue().encode("utf-8")


def _write_artifact_at(
    directory_descriptor: int,
    name: str,
    payload: bytes,
) -> str:
    """Write one immutable artifact beneath an already-bound directory."""

    if (
        not isinstance(name, str)
        or not name
        or name in {".", ".."}
        or "/" in name
        or not stat.S_ISDIR(os.fstat(directory_descriptor).st_mode)
    ):
        raise ProcedureAnalysisError("invalid descriptor-bound artifact path")
    descriptor = os.open(
        name,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_CLOEXEC", 0),
        0o600,
        dir_fd=directory_descriptor,
    )
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("artifact write made no progress")
            offset += written
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        descriptor_stat = os.fstat(descriptor)
        path_stat = os.stat(
            name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(descriptor_stat.st_mode)
            or descriptor_stat.st_nlink != 1
            or descriptor_stat.st_mode & 0o222
            or (descriptor_stat.st_dev, descriptor_stat.st_ino)
            != (path_stat.st_dev, path_stat.st_ino)
        ):
            raise ProcedureAnalysisError(
                "artifact inode changed during exclusive write"
            )
    finally:
        os.close(descriptor)
    os.fsync(directory_descriptor)
    return _sha256_bytes(payload)


def _read_artifact_at(directory_descriptor: int, name: str) -> bytes:
    """Read one exact immutable artifact snapshot through the bound directory."""

    if not name or name in {".", ".."} or "/" in name:
        raise ProcedureAnalysisError("invalid descriptor-bound artifact path")
    path_stat = os.stat(
        name,
        dir_fd=directory_descriptor,
        follow_symlinks=False,
    )
    descriptor = os.open(
        name,
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_CLOEXEC", 0),
        dir_fd=directory_descriptor,
    )
    try:
        descriptor_stat = os.fstat(descriptor)
        if (
            not stat.S_ISREG(path_stat.st_mode)
            or path_stat.st_nlink != 1
            or path_stat.st_mode & 0o222
            or not stat.S_ISREG(descriptor_stat.st_mode)
            or descriptor_stat.st_nlink != 1
            or descriptor_stat.st_mode & 0o222
            or (descriptor_stat.st_dev, descriptor_stat.st_ino)
            != (path_stat.st_dev, path_stat.st_ino)
        ):
            raise ProcedureAnalysisError(
                "publication artifact is aliased, mutable, or special"
            )
        payload = procedure_grid._read_stable_descriptor_bytes(
            descriptor,
            source=name,
        )
        final_path = os.stat(
            name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if (
            final_path.st_nlink != 1
            or final_path.st_mode & 0o222
            or (final_path.st_dev, final_path.st_ino)
            != (descriptor_stat.st_dev, descriptor_stat.st_ino)
        ):
            raise ProcedureAnalysisError(
                "publication artifact changed after descriptor read"
            )
        return payload
    finally:
        os.close(descriptor)


def _validate_publication_snapshot(
    directory_descriptor: int,
    *,
    manifest: Mapping[str, Any],
    intended_hashes: Mapping[str, str],
    require_sealed_directory: bool,
) -> dict[str, str]:
    """Validate the exact eight-file directory snapshot that will be published."""

    directory_stat = os.fstat(directory_descriptor)
    if (
        not stat.S_ISDIR(directory_stat.st_mode)
        or (
            require_sealed_directory
            and directory_stat.st_mode & 0o222
        )
        or set(os.listdir(directory_descriptor)) != PUBLICATION_FILENAMES
        or set(intended_hashes) != set(PAYLOAD_FILENAMES)
    ):
        raise ProcedureAnalysisError(
            "analysis publication directory roster/seal is invalid"
        )
    manifest_bytes = _read_artifact_at(
        directory_descriptor, "manifest.json"
    )
    parsed_manifest = procedure_grid._strict_json_bytes(
        manifest_bytes,
        source="manifest.json",
    )
    if (
        parsed_manifest != dict(manifest)
        or manifest_bytes != _canonical_bytes(manifest) + b"\n"
        or manifest.get("files") != dict(intended_hashes)
    ):
        raise ProcedureAnalysisError(
            "analysis publication manifest bytes are invalid"
        )
    observed_hashes: dict[str, str] = {}
    for name in PAYLOAD_FILENAMES:
        observed_hashes[name] = _sha256_bytes(
            _read_artifact_at(directory_descriptor, name)
        )
    if (
        observed_hashes != dict(intended_hashes)
        or observed_hashes != manifest["files"]
    ):
        raise ProcedureAnalysisError(
            "analysis publication bytes differ from their manifest"
        )
    return observed_hashes


def _move_bound_directory(
    descriptor: int,
    source: Path,
    destination: Path,
) -> tuple[int, int]:
    """Rename exactly the directory inode held by ``descriptor``."""

    before = procedure_grid._assert_directory_descriptor_path(
        descriptor, source
    )
    identity = (before.st_dev, before.st_ino)
    procedure_grid._atomic_rename_noreplace(source, destination)
    after = procedure_grid._assert_directory_descriptor_path(
        descriptor, destination
    )
    if (after.st_dev, after.st_ino) != identity:
        raise ProcedureAnalysisError(
            "renamed analysis directory differs from its bound inode"
        )
    procedure_grid._fsync_directory(destination.parent)
    return identity


def _verify_publication_inputs(
    result: AnalysisResult,
    *,
    run_root: Path,
    cache_root: Path,
    require_formal: bool,
    verify_runtime: bool,
) -> None:
    """Rebuild the complete checksum ledger immediately before publication."""

    plan, jobs, _audit = audit_exact_grid(run_root, require_formal=require_formal)
    if verify_runtime:
        procedure_grid.verify_runtime_identity(plan, cache_root=cache_root)
    _labels, cache_ledger = _bound_labels(plan, cache_root)
    observed = [
        _plan_ledger_row(run_root, plan),
        *cache_ledger,
        *[
            _job_ledger_row(
                job,
                procedure_grid.validate_completion(run_root, plan, job),
            )
            for job in jobs
        ],
    ]
    if (
        result.summary.get("plan_sha256") != plan.get("plan_sha256")
        or observed != result.input_ledger
        or result.summary.get("input_ledger_sha256")
        != _sha256_bytes(_canonical_bytes(observed))
    ):
        raise ProcedureAnalysisError(
            "analysis inputs changed between scoring and publication"
        )


def publish_analysis(
    result: AnalysisResult,
    *,
    run_root: str | Path,
    cache_root: str | Path,
    output_root: str | Path,
    require_formal: bool = True,
    verify_runtime: bool = True,
) -> Path:
    """Atomically publish aggregate-only JSON/CSV files outside the run root."""

    if require_formal:
        procedure_grid._require_formal_release_runtime()
    _validate_result(result)
    run = procedure_grid._safe_run_root(Path(run_root))
    cache = procedure_grid._absolute_path(Path(cache_root))
    procedure_grid._require_no_symlink_ancestors(
        cache,
        allow_missing_tail=not cache.exists(),
    )
    if cache.exists():
        procedure_grid._require_real_directory(cache)
    output = procedure_grid._absolute_path(Path(output_root))
    procedure_grid._require_no_symlink_ancestors(
        output,
        allow_missing_tail=True,
    )
    # This lexical check is sufficient only because every existing ancestor is
    # rejected if it is a symlink.  Perform it before creating any directory.
    for protected, label in ((run, "run"), (cache, "cache")):
        try:
            output.relative_to(protected)
        except ValueError:
            continue
        raise ProcedureAnalysisError(
            f"analysis output must be outside the immutable {label} root"
        )
    procedure_grid._secure_mkdir_absolute(output.parent)
    procedure_grid._require_real_directory(output.parent)
    procedure_grid._require_no_symlink_ancestors(output, allow_missing_tail=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite {output}")
    temporary = output.parent / f".{output.name}.{uuid.uuid4().hex}.partial"
    plan = procedure_grid.load_plan(run)
    if result.summary["plan_sha256"] != plan["plan_sha256"]:
        raise ProcedureAnalysisError("analysis plan changed before publication")
    formal_publication = (
        plan["publication_mode"] == procedure_grid.FORMAL_PUBLICATION_MODE
    )
    minimum_free_gib = float(
        plan["execution_config"]["minimum_free_gib"]
    )
    published = False
    published_identity: tuple[int, int] | None = None
    temporary_descriptor: int | None = None
    try:
        try:
            with procedure_grid.publication_fence(
                run,
                plan=plan,
                exclusive=True,
                blocking=True,
            ):
                fresh = compute_analysis(
                    run_root=run,
                    cache_root=cache,
                    ece_bins=DEFAULT_ECE_BINS,
                    require_formal=require_formal,
                    verify_runtime=verify_runtime,
                )
                _validate_result(fresh)
                supplied_payload = {
                    "summary": result.summary,
                    "tables": result.tables,
                    "input_ledger": result.input_ledger,
                }
                fresh_payload = {
                    "summary": fresh.summary,
                    "tables": fresh.tables,
                    "input_ledger": fresh.input_ledger,
                }
                if _canonical_bytes(supplied_payload) != _canonical_bytes(
                    fresh_payload
                ):
                    raise ProcedureAnalysisError(
                        "supplied analysis is not the fresh semantic recomputation"
                    )
                _verify_publication_inputs(
                    fresh,
                    run_root=run,
                    cache_root=cache,
                    require_formal=require_formal,
                    verify_runtime=verify_runtime,
                )
                if output.exists() or output.is_symlink():
                    raise FileExistsError(f"refusing to overwrite {output}")
                if formal_publication:
                    procedure_grid._require_hard_disk_floor(
                        output.parent, minimum_free_gib
                    )
                output_parent = procedure_grid._open_directory_absolute(
                    output.parent
                )
                try:
                    os.mkdir(temporary.name, 0o700, dir_fd=output_parent)
                    temporary_descriptor = os.open(
                        temporary.name,
                        os.O_RDONLY
                        | getattr(os, "O_DIRECTORY", 0)
                        | getattr(os, "O_NOFOLLOW", 0)
                        | getattr(os, "O_NONBLOCK", 0)
                        | getattr(os, "O_CLOEXEC", 0),
                        dir_fd=output_parent,
                    )
                    created_stat = os.stat(
                        temporary.name,
                        dir_fd=output_parent,
                        follow_symlinks=False,
                    )
                    bound_stat = os.fstat(temporary_descriptor)
                    if (
                        not stat.S_ISDIR(bound_stat.st_mode)
                        or (created_stat.st_dev, created_stat.st_ino)
                        != (bound_stat.st_dev, bound_stat.st_ino)
                    ):
                        raise ProcedureAnalysisError(
                            "new analysis directory was replaced while bound"
                        )
                    os.fsync(output_parent)
                finally:
                    os.close(output_parent)
                try:
                    payloads: dict[str, bytes] = {
                        "summary.json": _canonical_bytes(fresh.summary) + b"\n"
                    }
                    for name in TABLE_NAMES:
                        filename = f"{name}.csv"
                        payloads[filename] = _csv_bytes(
                            temporary / filename,
                            fresh.tables[name],
                        )
                    if set(payloads) != set(PAYLOAD_FILENAMES):
                        raise ProcedureAnalysisError(
                            "analysis payload roster is invalid"
                        )
                    intended_hashes: dict[str, str] = {}
                    for name in PAYLOAD_FILENAMES:
                        if formal_publication:
                            procedure_grid._require_hard_disk_floor(
                                output.parent, minimum_free_gib
                            )
                        intended_hashes[name] = _write_artifact_at(
                            temporary_descriptor,
                            name,
                            payloads[name],
                        )
                    manifest = {
                        "schema": MANIFEST_SCHEMA,
                        "plan_sha256": fresh.summary["plan_sha256"],
                        "track": TRACK,
                        "publication_mode": fresh.summary["publication_mode"],
                        "formal_publication": fresh.summary[
                            "formal_publication_eligible"
                        ],
                        "ece_bins": DEFAULT_ECE_BINS,
                        "files": intended_hashes,
                        "input_ledger_sha256": fresh.summary[
                            "input_ledger_sha256"
                        ],
                        "raw_labels_or_probabilities_published": False,
                    }
                    if set(manifest) != {
                        "schema",
                        "plan_sha256",
                        "track",
                        "publication_mode",
                        "formal_publication",
                        "ece_bins",
                        "files",
                        "input_ledger_sha256",
                        "raw_labels_or_probabilities_published",
                    }:
                        raise ProcedureAnalysisError(
                            "publication manifest schema is invalid"
                        )
                    if formal_publication:
                        procedure_grid._require_hard_disk_floor(
                            output.parent, minimum_free_gib
                        )
                    _write_artifact_at(
                        temporary_descriptor,
                        "manifest.json",
                        _canonical_bytes(manifest) + b"\n",
                    )
                    _verify_publication_inputs(
                        fresh,
                        run_root=run,
                        cache_root=cache,
                        require_formal=require_formal,
                        verify_runtime=verify_runtime,
                    )
                    procedure_grid._assert_directory_descriptor_path(
                        temporary_descriptor, temporary
                    )
                    _validate_publication_snapshot(
                        temporary_descriptor,
                        manifest=manifest,
                        intended_hashes=intended_hashes,
                        require_sealed_directory=False,
                    )
                    os.fsync(temporary_descriptor)
                    os.fchmod(temporary_descriptor, 0o555)
                    os.fsync(temporary_descriptor)
                    procedure_grid._assert_directory_descriptor_path(
                        temporary_descriptor, temporary
                    )
                    _validate_publication_snapshot(
                        temporary_descriptor,
                        manifest=manifest,
                        intended_hashes=intended_hashes,
                        require_sealed_directory=True,
                    )
                    if formal_publication:
                        procedure_grid._require_hard_disk_floor(
                            output.parent, minimum_free_gib
                        )
                    published_identity = _move_bound_directory(
                        temporary_descriptor,
                        temporary,
                        output,
                    )
                    published = True
                    procedure_grid._assert_directory_descriptor_path(
                        temporary_descriptor, output
                    )
                    _validate_publication_snapshot(
                        temporary_descriptor,
                        manifest=manifest,
                        intended_hashes=intended_hashes,
                        require_sealed_directory=True,
                    )
                    after_publish = os.fstat(temporary_descriptor)
                    if (
                        after_publish.st_dev,
                        after_publish.st_ino,
                    ) != published_identity:
                        raise ProcedureAnalysisError(
                            "published analysis directory inode was replaced"
                        )
                    os.fsync(temporary_descriptor)
                    procedure_grid._fsync_directory(output.parent)
                except Exception:
                    if temporary_descriptor is not None:
                        source = output if published else temporary
                        quarantine = (
                            output.parent
                            / (
                                f".{output.name}.failed-{uuid.uuid4().hex}"
                                if published
                                else (
                                    f"{temporary.name}.failed-"
                                    f"{uuid.uuid4().hex}"
                                )
                            )
                        )
                        _move_bound_directory(
                            temporary_descriptor,
                            source,
                            quarantine,
                        )
                        published = False
                    raise
        except procedure_grid.PublicationFenceLost as error:
            if published and temporary_descriptor is not None:
                quarantine = (
                    output.parent
                    / (
                        f".{output.name}.publication-fence-lost-"
                        f"{uuid.uuid4().hex}"
                    )
                )
                try:
                    _move_bound_directory(
                        temporary_descriptor,
                        output,
                        quarantine,
                    )
                    published = False
                except (
                    FileNotFoundError,
                    procedure_grid.ProcedureGridError,
                    ProcedureAnalysisError,
                ) as ownership_error:
                    raise ProcedureAnalysisError(
                        "publication fence and output inode ownership were both lost"
                    ) from ownership_error
            raise ProcedureAnalysisError(
                "publication fence ownership was lost; output quarantined"
            ) from error
    finally:
        if temporary_descriptor is not None:
            os.close(temporary_descriptor)
    return output


def analyze_and_publish(
    *,
    run_root: str | Path,
    cache_root: str | Path,
    output_root: str | Path,
    ece_bins: int = DEFAULT_ECE_BINS,
) -> Path:
    procedure_grid._require_formal_release_runtime()
    plan = procedure_grid.load_plan(Path(run_root))
    procedure_grid._install_thread_environment(
        int(plan["execution_config"]["cpu_threads_per_worker"])
    )
    os.environ.setdefault(
        "CUBLAS_WORKSPACE_CONFIG",
        procedure_grid.REQUIRED_CUBLAS_WORKSPACE_CONFIG,
    )
    if (
        os.environ["CUBLAS_WORKSPACE_CONFIG"]
        != procedure_grid.REQUIRED_CUBLAS_WORKSPACE_CONFIG
    ):
        raise ProcedureAnalysisError(
            "CUBLAS_WORKSPACE_CONFIG differs from the formal contract"
        )
    result = compute_analysis(
        run_root=run_root,
        cache_root=cache_root,
        ece_bins=ece_bins,
    )
    return publish_analysis(
        result,
        run_root=run_root,
        cache_root=cache_root,
        output_root=output_root,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--ece-bins", type=int, default=DEFAULT_ECE_BINS)
    args = parser.parse_args(argv)
    output = analyze_and_publish(
        run_root=args.run_root,
        cache_root=args.cache_root,
        output_root=args.output_root,
        ece_bins=args.ece_bins,
    )
    print(json.dumps({"output_root": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ANALYSIS_SCHEMA",
    "EXPECTED_FORMAL_JOBS",
    "METRIC_NAMES",
    "AnalysisResult",
    "ProcedureAnalysisError",
    "analyze_and_publish",
    "audit_exact_grid",
    "classification_metrics",
    "compute_analysis",
    "main",
    "publish_analysis",
]
