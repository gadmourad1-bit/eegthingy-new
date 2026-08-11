"""Fail-closed label-joining analysis for the author-recipe reference track.

No label is opened until :mod:`ieee_mi.author_recipe_grid` proves that the
complete score-blind Cartesian grid is checksum-valid and quiescent.  Published
artifacts contain scalar metrics only and retain the explicit
``author_recipe_adapted_reference_separate`` track label; they are never merged into
the common-recipe model table.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import io
import json
import math
import os
import re
import stat
import sys
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
)

from . import author_recipe_grid


ANALYSIS_SCHEMA = "ieee-mi-author-recipe-analysis-v1"
MANIFEST_SCHEMA = "ieee-mi-author-recipe-analysis-manifest-v1"
PUBLICATION_LOCK_SCHEMA = "ieee-mi-author-recipe-analysis-lock-v1"
TRACK = author_recipe_grid.TRACK
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
TABLE_NAMES = (
    "job_metrics",
    "subject_seed_metrics",
    "subject_summary",
    "dataset_summary",
    "overall_summary",
    "timing_summary",
)
SUBJECT_AGGREGATION = "concatenate_folds_then_mean_seeds"
DATASET_AGGREGATION = (
    "concatenate_folds_then_mean_seeds_then_mean_subjects"
)
OVERALL_AGGREGATION = (
    "fold_concat_then_seed_mean_then_subject_mean_then_equal_dataset_mean"
)
AGGREGATION_DEFINITION = (
    "concatenate disjoint folds within subject/seed; mean seeds; mean "
    "subjects within dataset; equally weight the five datasets"
)
INTERPRETATION = (
    "separately labelled author-recipe adapted reference results on opened "
    "development cohorts; not common-recipe rankings, confirmation evidence, "
    "or a global/SOTA claim"
)
INFERENTIAL_COMPARISONS = (
    "descriptive only; no outcome-selected ranking or hypothesis test"
)
METRIC_DEFINITIONS: dict[str, str] = {
    "accuracy": "fraction of correct top-probability classes",
    "balanced_accuracy": "unadjusted mean recall over observed true classes",
    "chance_normalized_balanced_accuracy": (
        "(balanced_accuracy - 1/K) / (1 - 1/K), using each dataset's "
        "planned class count K"
    ),
    "macro_f1": "macro F1 over the planned class set; zero for undefined F1",
    "cohen_kappa": "Cohen kappa over the planned class set",
    "ovr_macro_auroc": (
        "mean one-vs-rest AUROC only when every class has positive and "
        "negative examples"
    ),
    "nll": "mean negative log true-class probability clipped at 1e-15",
    "multiclass_brier": "mean sum of squared class-probability errors",
    "ece": "top-label ECE with 15 equal-width confidence bins",
}
_DIRECT_METRIC_COLUMNS = frozenset(METRIC_NAMES)
_AGGREGATE_METRIC_COLUMNS = frozenset(
    column
    for metric in METRIC_NAMES
    for column in (metric, f"{metric}_defined_count")
)
_TIMING_BASES = (
    "parameter_count",
    "source_fit_seconds",
    "test_inference_seconds",
    "job_total_seconds",
    "inference_ms_per_trial",
)
TABLE_SCHEMAS: dict[str, frozenset[str]] = {
    "job_metrics": frozenset(
        {
            "track",
            "dataset",
            "reference",
            "subject",
            "fold",
            "seed",
            "job_id",
            "test_count",
            "parameter_count",
            "source_fit_seconds",
            "test_inference_seconds",
            "job_total_seconds",
            "inference_ms_per_trial",
        }
    )
    | _DIRECT_METRIC_COLUMNS,
    "subject_seed_metrics": frozenset(
        {
            "track",
            "dataset",
            "reference",
            "subject",
            "seed",
            "folds_concatenated",
            "test_trials",
        }
    )
    | _DIRECT_METRIC_COLUMNS,
    "subject_summary": frozenset(
        {
            "track",
            "dataset",
            "reference",
            "subject",
            "seeds_averaged",
            "aggregation",
        }
    )
    | _AGGREGATE_METRIC_COLUMNS,
    "dataset_summary": frozenset(
        {
            "track",
            "dataset",
            "reference",
            "subjects_averaged",
            "aggregation",
        }
    )
    | _AGGREGATE_METRIC_COLUMNS,
    "overall_summary": frozenset(
        {
            "track",
            "reference",
            "datasets_equal_weighted",
            "aggregation",
        }
    )
    | _AGGREGATE_METRIC_COLUMNS,
    "timing_summary": frozenset(
        {"track", "dataset", "reference", "jobs"}
    )
    | frozenset(
        f"{base}_{suffix}"
        for base in _TIMING_BASES
        for suffix in ("mean", "median", "p95", "min", "max")
    ),
}
SUMMARY_KEYS = frozenset(
    {
        "schema",
        "plan_sha256",
        "input_ledger_sha256",
        "track",
        "track_description",
        "common_recipe_track",
        "identity_contract",
        "references",
        "datasets",
        "seeds",
        "expected_jobs",
        "evidence_scope",
        "confirmation_evidence",
        "interpretation",
        "aggregation_definition",
        "metric_definitions",
        "inferential_comparisons",
        "ece_bins",
        "audit_snapshot",
        "analysis_source_identity",
        "analysis_environment_sha256",
        "analysis_contract_sha256",
        "decision_contract_sha256",
        "source_identity_sha256",
        "environment_identity_sha256",
        "reference_provenance_sha256",
        "cache_identity_sha256",
        "split_identity_sha256",
        "table_row_counts",
    }
)
LEDGER_KEYS = frozenset(
    {"kind", "identity", "sha256_a", "sha256_b", "sha256_c"}
)
_RESULT_SEAL_KEY = os.urandom(32)
_FORBIDDEN_PUBLISHED_KEYS = frozenset(
    {
        "label",
        "labels",
        "target",
        "targets",
        "outcome",
        "outcomes",
        "truth",
        "ground_truth",
        "y",
        "y_true",
        "y_test",
        "row",
        "rows",
        "test_rows",
        "probability",
        "probabilities",
        "predicted_class",
        "predicted_classes",
    }
)


class AuthorRecipeAnalysisError(RuntimeError):
    """Raised when the label-joining boundary cannot be proven."""


@dataclass
class AnalysisResult:
    summary: dict[str, Any]
    tables: dict[str, list[dict[str, Any]]]
    input_ledger: list[dict[str, str]] = field(repr=False)
    _seal: str = field(default="", repr=False)


@dataclass
class _HeldPublishedDirectory:
    """An exact published directory whose directory and child FDs stay held."""

    destination: Path
    directory_descriptor: int = field(repr=False)
    child_descriptors: dict[str, int] = field(repr=False)
    directory_fingerprint: tuple[int, ...]
    child_fingerprints: dict[str, tuple[int, ...]] = field(repr=False)
    expected_payloads: Mapping[str, bytes] = field(repr=False)
    closed: bool = field(default=False, init=False, repr=False)

    def revalidate(self) -> None:
        if self.closed:
            raise AuthorRecipeAnalysisError(
                "published-directory snapshot was closed too early"
            )
        names = tuple(sorted(os.listdir(self.directory_descriptor)))
        if set(names) != set(self.expected_payloads):
            raise AuthorRecipeAnalysisError(
                "published analysis exact file set changed"
            )
        observed: list[tuple[str, tuple[int, ...]]] = []
        for name in names:
            fingerprint = author_recipe_grid._stat_fingerprint(
                os.stat(
                    name,
                    dir_fd=self.directory_descriptor,
                    follow_symlinks=False,
                )
            )
            observed.append((name, fingerprint))
            descriptor = self.child_descriptors.get(name)
            if (
                descriptor is None
                or fingerprint != self.child_fingerprints.get(name)
                or author_recipe_grid._stat_fingerprint(
                    os.fstat(descriptor)
                )
                != self.child_fingerprints.get(name)
            ):
                raise AuthorRecipeAnalysisError(
                    f"published analysis artifact changed: {name}"
                )
            expected = self.expected_payloads[name]
            if os.pread(descriptor, len(expected) + 1, 0) != expected:
                raise AuthorRecipeAnalysisError(
                    f"published analysis artifact bytes changed: {name}"
                )
        canonical = author_recipe_grid._anchored_lstat(
            self.destination
        )
        if (
            author_recipe_grid._stat_fingerprint(
                os.fstat(self.directory_descriptor)
            )
            != self.directory_fingerprint
            or author_recipe_grid._stat_fingerprint(canonical)
            != self.directory_fingerprint
            or tuple(observed)
            != tuple(sorted(self.child_fingerprints.items()))
        ):
            raise AuthorRecipeAnalysisError(
                "published analysis directory binding changed"
            )

    def close(self) -> None:
        if self.closed:
            return
        for descriptor in self.child_descriptors.values():
            os.close(descriptor)
        os.close(self.directory_descriptor)
        self.closed = True


def decision_contract() -> dict[str, Any]:
    """Return every frozen post-hoc metric, table, and aggregation decision."""

    return {
        "schema": "ieee-mi-author-recipe-analysis-decisions-v1",
        "track": TRACK,
        "track_description": author_recipe_grid.TRACK_DESCRIPTION,
        "ece_bins": DEFAULT_ECE_BINS,
        "metric_order": list(METRIC_NAMES),
        "metric_definitions": dict(METRIC_DEFINITIONS),
        "table_order": list(TABLE_NAMES),
        "table_schemas": {
            name: sorted(TABLE_SCHEMAS[name]) for name in TABLE_NAMES
        },
        "aggregation": {
            "subject": SUBJECT_AGGREGATION,
            "dataset": DATASET_AGGREGATION,
            "overall": OVERALL_AGGREGATION,
            "definition": AGGREGATION_DEFINITION,
        },
        "timing_bases": list(_TIMING_BASES),
        "timing_statistics": ["mean", "median", "p95", "min", "max"],
        "metric_computation": {
            "prediction": "numpy_argmax_first_maximum",
            "probability_dtype": "float64",
            "probability_bounds": "closed_interval_0_1",
            "probability_row_sum": {
                "rtol": 0.0,
                "atol": 1e-6,
            },
            "balanced_accuracy_classes": "observed_true_classes",
            "macro_f1_classes": "planned_class_set_zero_division_zero",
            "cohen_kappa_classes": "planned_class_set",
            "ovr_auroc": (
                "unweighted_class_mean_only_if_each_class_has_both_binary_states"
            ),
            "nll_probability_floor": 1e-15,
            "multiclass_brier": "mean_sum_squared_class_probability_error",
            "ece": (
                "top_label_absolute_calibration_gap_equal_width_confidence_bins"
            ),
            "undefined_aggregate_policy": (
                "mean_defined_values_only_and_publish_defined_count"
            ),
        },
        "row_combination": (
            "stable_sort_by_trial_row_after_disjoint_fold_concatenation"
        ),
        "timing_quantile": "numpy_quantile_q0.95_default_linear_method",
        "table_row_order": (
            "immutable_dataset_then_reference_then_subject_then_fold_then_seed_"
            "products_as_applicable"
        ),
        "inferential_comparisons": INFERENTIAL_COMPARISONS,
        "label_join": "only_after_exact_quiescent_score_blind_grid_audit",
        "published_material": "scalar_aggregates_only",
    }


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(child) for child in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return _finite_float(value)
    if isinstance(value, float):
        return _finite_float(value)
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


def _sha256_file(path: Path) -> str:
    return author_recipe_grid._sha256_file(path)


def _analysis_source_identity() -> dict[str, str]:
    return {
        "ieee_mi/author_recipe_analysis.py": _sha256_file(
            Path(__file__).resolve()
        )
    }


def _binding_sha256(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _result_seal(result: AnalysisResult) -> str:
    return hmac.new(
        _RESULT_SEAL_KEY,
        _canonical_bytes(
            {
                "summary": result.summary,
                "tables": result.tables,
                "ledger": result.input_ledger,
            }
        ),
        hashlib.sha256,
    ).hexdigest()


def classification_metrics(
    y_true: np.ndarray | Sequence[int],
    probabilities: np.ndarray,
    *,
    n_classes: int,
    ece_bins: int = DEFAULT_ECE_BINS,
) -> dict[str, float | None]:
    y = np.asarray(y_true, dtype=np.int64)
    values = np.asarray(probabilities, dtype=np.float64)
    if (
        y.ndim != 1
        or len(y) == 0
        or n_classes < 2
        or isinstance(ece_bins, bool)
        or not isinstance(ece_bins, int)
        or ece_bins != DEFAULT_ECE_BINS
        or values.shape != (len(y), n_classes)
        or np.any(y < 0)
        or np.any(y >= n_classes)
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(values > 1.0)
        or not np.allclose(
            values.sum(axis=1), 1.0, rtol=0.0, atol=1e-6
        )
    ):
        raise AuthorRecipeAnalysisError(
            "invalid arrays or non-frozen ECE bin count supplied to metrics"
        )
    predicted = values.argmax(axis=1)
    classes = np.arange(n_classes, dtype=np.int64)
    support = np.bincount(y, minlength=n_classes)
    correct = np.bincount(y[predicted == y], minlength=n_classes)
    present = support > 0
    balanced_accuracy = float(np.mean(correct[present] / support[present]))
    chance = 1.0 / n_classes
    normalized_ba = (balanced_accuracy - chance) / (1.0 - chance)
    aucs: list[float] = []
    for class_index in range(n_classes):
        binary = (y == class_index).astype(np.int8)
        if binary.min() == binary.max():
            aucs = []
            break
        aucs.append(float(roc_auc_score(binary, values[:, class_index])))
    clipped = np.clip(values[np.arange(len(y)), y], 1e-15, 1.0)
    one_hot = np.eye(n_classes, dtype=np.float64)[y]
    confidence = values[np.arange(len(y)), predicted]
    is_correct = (predicted == y).astype(np.float64)
    bins = np.minimum(
        np.floor(confidence * ece_bins).astype(np.int64), ece_bins - 1
    )
    ece = 0.0
    for bin_index in range(ece_bins):
        mask = bins == bin_index
        if np.any(mask):
            ece += float(mask.mean()) * abs(
                float(is_correct[mask].mean()) - float(confidence[mask].mean())
            )
    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": balanced_accuracy,
        "chance_normalized_balanced_accuracy": float(normalized_ba),
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
        "ovr_macro_auroc": (
            float(np.mean(aucs)) if len(aucs) == n_classes else None
        ),
        "nll": float(-np.log(clipped).mean()),
        "multiclass_brier": float(np.square(values - one_hot).sum(axis=1).mean()),
        "ece": float(ece),
    }


def _metric_means(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, float | int | None]:
    result: dict[str, float | int | None] = {}
    for metric in METRIC_NAMES:
        values = [
            float(row[metric])
            for row in rows
            if row.get(metric) is not None and math.isfinite(float(row[metric]))
        ]
        result[metric] = float(np.mean(values)) if values else None
        result[f"{metric}_defined_count"] = len(values)
    return result


def _require_readonly_regular(path: Path, context: str) -> None:
    try:
        author_recipe_grid._read_regular_bytes(
            path,
            readonly=True,
            context=context,
        )
    except (FileNotFoundError, author_recipe_grid.AuthorRecipeError) as error:
        raise AuthorRecipeAnalysisError(
            f"{context} is absent, aliased, non-regular, or writable: {path}"
        ) from error


def audit_exact_grid(
    run_root: str | Path,
    *,
    require_formal_grid: bool = True,
    publication_fence: author_recipe_grid.PublicationFence | None = None,
) -> tuple[dict[str, Any], tuple[author_recipe_grid.Job, ...], dict[str, Any]]:
    root = author_recipe_grid._absolute_path(run_root)
    try:
        author_recipe_grid._require_safe_ancestors(
            root,
            include_leaf=True,
            allow_missing_leaf=False,
            context="analysis run root",
        )
    except (FileNotFoundError, author_recipe_grid.AuthorRecipeError) as error:
        raise AuthorRecipeAnalysisError("analysis run root is unsafe") from error
    _require_readonly_regular(root / "plan.json", "immutable plan")
    _require_readonly_regular(root / "plan.sha256", "immutable plan digest")
    plan = author_recipe_grid.load_plan(root)
    if require_formal_grid and (
        tuple(plan.get("dataset_order", ()))
        != author_recipe_grid.OPENED_DATASETS
        or tuple(plan.get("references", ()))
        != author_recipe_grid.REFERENCES
        or tuple(plan.get("seeds", ())) != author_recipe_grid.FORMAL_SEEDS
        or plan.get("datasets") != author_recipe_grid._dataset_contracts()
        or plan.get("executor") != author_recipe_grid.DEFAULT_EXECUTOR
    ):
        raise AuthorRecipeAnalysisError(
            "analysis requires the exact formal reference/dataset/seed roster"
        )
    jobs = tuple(author_recipe_grid.iter_jobs(plan))
    if require_formal_grid and len(jobs) != author_recipe_grid.EXPECTED_JOB_COUNT:
        raise AuthorRecipeAnalysisError("formal grid must contain 4,480 jobs")
    audit = author_recipe_grid.audit_grid(
        root, plan, publication_fence=publication_fence
    )
    if (
        audit.get("plan_sha256") != plan.get("plan_sha256")
        or int(audit.get("expected_jobs", -1)) != len(jobs)
        or int(audit.get("complete_jobs", -1)) != len(jobs)
        or audit.get("exact_cartesian_complete") is not True
        or set(audit.get("complete_job_ids", ()))
        != {job.job_id for job in jobs}
    ):
        raise AuthorRecipeAnalysisError(
            "refusing label access until the score-blind grid is exact and "
            f"quiescent: {audit}"
        )
    return plan, jobs, audit


def _cache_ledger_row(
    plan: Mapping[str, Any],
    *,
    dataset: str,
    subject: int,
    cache: Mapping[str, Any],
) -> dict[str, str]:
    subject_key = author_recipe_grid._subject_key(dataset, subject)
    splits = {
        author_recipe_grid._split_key(dataset, subject, int(fold)): plan[
            "split_identity"
        ][author_recipe_grid._split_key(dataset, subject, int(fold))]
        for fold in plan["datasets"][dataset]["folds"]
    }
    return {
        "kind": "cache",
        "identity": subject_key,
        "sha256_a": str(plan["cache_identity"][subject_key]["array_sha256"]),
        "sha256_b": _sha256_bytes(_canonical_bytes(cache["identity"])),
        "sha256_c": _sha256_bytes(_canonical_bytes(splits)),
    }


def _bound_cache_labels(
    plan: Mapping[str, Any], cache_root: Path
) -> tuple[dict[tuple[str, int], np.ndarray], list[dict[str, str]]]:
    from .data import split_indices

    labels: dict[tuple[str, int], np.ndarray] = {}
    ledger: list[dict[str, str]] = []
    for dataset in plan["dataset_order"]:
        contract = plan["datasets"][dataset]
        for raw_subject in contract["subjects"]:
            subject = int(raw_subject)
            subject_key = author_recipe_grid._subject_key(dataset, subject)
            cache = author_recipe_grid._load_subject_cache_secure(
                dataset, subject, cache_root=cache_root
            )
            if cache["identity"] != plan["cache_identity"][subject_key]:
                raise AuthorRecipeAnalysisError(
                    f"cache identity differs from plan for {subject_key}"
                )
            y = np.asarray(cache["y"])
            if (
                y.dtype != np.int64
                or y.ndim != 1
                or np.any(y < 0)
                or np.any(y >= int(contract["n_classes"]))
            ):
                raise AuthorRecipeAnalysisError(
                    f"cache labels are invalid for {subject_key}"
                )
            seen_test_rows: set[int] = set()
            for raw_fold in contract["folds"]:
                fold = int(raw_fold)
                train, validation, test = split_indices(
                    dataset,
                    y,
                    cache["sessions"],
                    cache["runs"],
                    fold=fold,
                    subject=subject,
                )
                observed = author_recipe_grid.full_grid._one_split_identity(
                    dataset=dataset,
                    subject=subject,
                    fold=fold,
                    cache_array_sha256=str(cache["identity"]["array_sha256"]),
                    trial_count=len(y),
                    train_rows=train,
                    validation_rows=validation,
                    test_rows=test,
                )
                key = author_recipe_grid._split_key(dataset, subject, fold)
                if observed != plan["split_identity"][key]:
                    raise AuthorRecipeAnalysisError(
                        f"reconstructed split differs from plan for {key}"
                    )
                test_set = set(np.asarray(test, dtype=np.int64).tolist())
                if seen_test_rows.intersection(test_set):
                    raise AuthorRecipeAnalysisError(
                        f"test folds overlap for {subject_key}"
                    )
                seen_test_rows.update(test_set)
            labels[(dataset, subject)] = y.copy()
            ledger.append(
                _cache_ledger_row(
                    plan, dataset=dataset, subject=subject, cache=cache
                )
            )
    return labels, ledger


def _plan_ledger_row(
    run_root: Path, plan: Mapping[str, Any]
) -> dict[str, str]:
    return {
        "kind": "plan",
        "identity": str(plan["plan_sha256"]),
        "sha256_a": _sha256_file(run_root / "plan.json"),
        "sha256_b": _sha256_file(run_root / "plan.sha256"),
        "sha256_c": str(plan["plan_sha256"]),
    }


def _job_ledger_row(
    job: author_recipe_grid.Job,
    snapshot: author_recipe_grid.CompletionSnapshot,
) -> dict[str, str]:
    return {
        "kind": "job",
        "identity": job.job_id,
        "sha256_a": snapshot.artifact_sha256["completion.json"],
        "sha256_b": snapshot.artifact_sha256["record.json"],
        "sha256_c": snapshot.artifact_sha256["predictions.npz"],
    }


def _verify_ledger(
    *,
    run_root: Path,
    cache_root: Path,
    plan: Mapping[str, Any],
    jobs: Sequence[author_recipe_grid.Job],
    ledger: Sequence[Mapping[str, str]],
) -> None:
    _, cache_ledger = _bound_cache_labels(plan, cache_root)
    observed = [
        _plan_ledger_row(run_root, plan),
        *cache_ledger,
        *[
            _job_ledger_row(
                job,
                author_recipe_grid._capture_completion_snapshot(
                    run_root, plan, job
                ),
            )
            for job in jobs
        ],
    ]
    if observed != list(ledger):
        raise AuthorRecipeAnalysisError("analysis input ledger changed")


def _job_row(
    job: author_recipe_grid.Job,
    record: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    timing = record["metadata"]["timing_seconds"]
    test_count = int(record["test_count"])
    fit = record["metadata"]["fit"]
    return {
        "track": TRACK,
        **job.identity(),
        "job_id": job.job_id,
        "test_count": test_count,
        **metrics,
        "parameter_count": int(fit["parameter_count"]),
        "source_fit_seconds": float(timing["source_fit"]),
        "test_inference_seconds": float(timing["test_inference"]),
        "job_total_seconds": float(timing["job_total"]),
        "inference_ms_per_trial": (
            1000.0 * float(timing["test_inference"]) / test_count
        ),
    }


def compute_analysis(
    *,
    run_root: str | Path,
    cache_root: str | Path,
    ece_bins: int = DEFAULT_ECE_BINS,
    require_formal_grid: bool = True,
    _publication_fence: author_recipe_grid.PublicationFence | None = None,
) -> AnalysisResult:
    """Audit first, then join labels in memory and return aggregate-only state."""

    if (
        isinstance(ece_bins, bool)
        or not isinstance(ece_bins, int)
        or ece_bins != DEFAULT_ECE_BINS
    ):
        raise ValueError(
            f"ece_bins is frozen at {DEFAULT_ECE_BINS} by the immutable plan"
        )
    root = author_recipe_grid._absolute_path(run_root)
    cache = author_recipe_grid._absolute_path(cache_root)
    try:
        author_recipe_grid._require_safe_ancestors(
            cache,
            include_leaf=True,
            allow_missing_leaf=False,
            context="analysis cache root",
        )
    except (FileNotFoundError, author_recipe_grid.AuthorRecipeError) as error:
        raise AuthorRecipeAnalysisError("analysis cache root is unsafe") from error
    plan, jobs, opening_audit = audit_exact_grid(
        root,
        require_formal_grid=require_formal_grid,
        publication_fence=_publication_fence,
    )
    if require_formal_grid:
        try:
            author_recipe_grid.verify_runtime_identity(
                plan, cache_root=cache
            )
        except author_recipe_grid.AuthorRecipeError as error:
            raise AuthorRecipeAnalysisError(
                f"live runtime differs from plan: {error}"
            ) from error
    labels_by_subject, cache_ledger = _bound_cache_labels(plan, cache)
    job_rows: list[dict[str, Any]] = []
    grouped: dict[
        tuple[str, str, int, int], dict[str, list[np.ndarray]]
    ] = defaultdict(lambda: {"rows": [], "labels": [], "outputs": []})
    job_ledger: list[dict[str, str]] = []
    for job in jobs:
        snapshot = author_recipe_grid._capture_completion_snapshot(
            root, plan, job
        )
        rows = np.asarray(snapshot.rows, dtype=np.int64)
        probabilities = np.asarray(
            snapshot.probabilities, dtype=np.float64
        )
        subject_labels = labels_by_subject[(job.dataset, job.subject)]
        if np.any(rows < 0) or np.any(rows >= len(subject_labels)):
            raise AuthorRecipeAnalysisError(
                f"prediction rows are invalid for {job.job_id}"
            )
        outcomes = subject_labels[rows]
        metrics = classification_metrics(
            outcomes,
            probabilities,
            n_classes=int(plan["datasets"][job.dataset]["n_classes"]),
            ece_bins=ece_bins,
        )
        job_rows.append(_job_row(job, snapshot.record, metrics))
        key = (job.dataset, job.reference, job.subject, job.seed)
        grouped[key]["rows"].append(rows.copy())
        grouped[key]["labels"].append(outcomes.copy())
        grouped[key]["outputs"].append(probabilities.copy())
        job_ledger.append(_job_ledger_row(job, snapshot))
    if [row["job_id"] for row in job_rows] != [job.job_id for job in jobs]:
        raise AuthorRecipeAnalysisError("scored jobs differ from plan order")

    subject_seed_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    for dataset in plan["dataset_order"]:
        contract = plan["datasets"][dataset]
        n_classes = int(contract["n_classes"])
        for reference in plan["references"]:
            selected_subject_rows: list[dict[str, Any]] = []
            for raw_subject in contract["subjects"]:
                subject = int(raw_subject)
                selected_seed_rows: list[dict[str, Any]] = []
                for raw_seed in plan["seeds"]:
                    seed = int(raw_seed)
                    group = grouped[(dataset, reference, subject, seed)]
                    row_values = np.concatenate(group["rows"])
                    outcome_values = np.concatenate(group["labels"])
                    output_values = np.concatenate(group["outputs"])
                    order = np.argsort(row_values, kind="stable")
                    row_values = row_values[order]
                    outcome_values = outcome_values[order]
                    output_values = output_values[order]
                    if len(np.unique(row_values)) != len(row_values):
                        raise AuthorRecipeAnalysisError(
                            f"fold test rows overlap for {dataset}/{reference}/"
                            f"S{subject}/seed{seed}"
                        )
                    row = {
                        "track": TRACK,
                        "dataset": dataset,
                        "reference": reference,
                        "subject": subject,
                        "seed": seed,
                        "folds_concatenated": len(contract["folds"]),
                        "test_trials": len(row_values),
                        **classification_metrics(
                            outcome_values,
                            output_values,
                            n_classes=n_classes,
                            ece_bins=ece_bins,
                        ),
                    }
                    subject_seed_rows.append(row)
                    selected_seed_rows.append(row)
                subject_row = {
                    "track": TRACK,
                    "dataset": dataset,
                    "reference": reference,
                    "subject": subject,
                    "seeds_averaged": len(selected_seed_rows),
                    "aggregation": SUBJECT_AGGREGATION,
                    **_metric_means(selected_seed_rows),
                }
                subject_rows.append(subject_row)
                selected_subject_rows.append(subject_row)
            dataset_rows.append(
                {
                    "track": TRACK,
                    "dataset": dataset,
                    "reference": reference,
                    "subjects_averaged": len(selected_subject_rows),
                    "aggregation": DATASET_AGGREGATION,
                    **_metric_means(selected_subject_rows),
                }
            )
    overall_rows: list[dict[str, Any]] = []
    for reference in plan["references"]:
        selected = [
            row for row in dataset_rows if row["reference"] == reference
        ]
        if len(selected) != len(plan["dataset_order"]):
            raise AuthorRecipeAnalysisError(
                f"dataset aggregates are incomplete for {reference}"
            )
        overall_rows.append(
            {
                "track": TRACK,
                "reference": reference,
                "datasets_equal_weighted": len(selected),
                "aggregation": OVERALL_AGGREGATION,
                **_metric_means(selected),
            }
        )
    timing_rows: list[dict[str, Any]] = []
    timing_fields = _TIMING_BASES
    for dataset in (*plan["dataset_order"], "ALL_DATASETS"):
        for reference in plan["references"]:
            selected = [
                row
                for row in job_rows
                if row["reference"] == reference
                and (dataset == "ALL_DATASETS" or row["dataset"] == dataset)
            ]
            timing_row: dict[str, Any] = {
                "track": TRACK,
                "dataset": dataset,
                "reference": reference,
                "jobs": len(selected),
            }
            for field_name in timing_fields:
                values = np.asarray(
                    [float(row[field_name]) for row in selected],
                    dtype=np.float64,
                )
                for suffix, value in (
                    ("mean", values.mean()),
                    ("median", np.median(values)),
                    ("p95", np.quantile(values, 0.95)),
                    ("min", values.min()),
                    ("max", values.max()),
                ):
                    timing_row[f"{field_name}_{suffix}"] = float(value)
            timing_rows.append(timing_row)

    ledger = [
        _plan_ledger_row(root, plan),
        *cache_ledger,
        *job_ledger,
    ]
    _verify_ledger(
        run_root=root,
        cache_root=cache,
        plan=plan,
        jobs=jobs,
        ledger=ledger,
    )
    final_plan, final_jobs, final_audit = audit_exact_grid(
        root,
        require_formal_grid=require_formal_grid,
        publication_fence=_publication_fence,
    )
    if final_plan != plan or final_jobs != jobs or final_audit != opening_audit:
        raise AuthorRecipeAnalysisError("grid changed during analysis")
    tables = {
        "job_metrics": job_rows,
        "subject_seed_metrics": subject_seed_rows,
        "subject_summary": subject_rows,
        "dataset_summary": dataset_rows,
        "overall_summary": overall_rows,
        "timing_summary": timing_rows,
    }
    summary = {
        "schema": ANALYSIS_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "input_ledger_sha256": _sha256_bytes(_canonical_bytes(ledger)),
        "track": TRACK,
        "track_description": author_recipe_grid.TRACK_DESCRIPTION,
        "common_recipe_track": False,
        "identity_contract": _json_ready(plan["identity_contract"]),
        "references": list(plan["references"]),
        "datasets": list(plan["dataset_order"]),
        "seeds": list(plan["seeds"]),
        "expected_jobs": len(jobs),
        "evidence_scope": plan["evidence_scope"],
        "confirmation_evidence": False,
        "interpretation": INTERPRETATION,
        "aggregation_definition": AGGREGATION_DEFINITION,
        "metric_definitions": dict(METRIC_DEFINITIONS),
        "inferential_comparisons": INFERENTIAL_COMPARISONS,
        "ece_bins": int(ece_bins),
        "audit_snapshot": opening_audit,
        "analysis_source_identity": _analysis_source_identity(),
        "analysis_environment_sha256": _sha256_bytes(
            _canonical_bytes(
                author_recipe_grid._normalized_worker_environment(
                    plan["environment_identity"]
                )
            )
        ),
        "analysis_contract_sha256": _binding_sha256(
            plan.get("analysis_contract")
        ),
        "decision_contract_sha256": _binding_sha256(decision_contract()),
        "source_identity_sha256": _binding_sha256(plan["source_identity"]),
        "environment_identity_sha256": _binding_sha256(
            plan["environment_identity"]
        ),
        "reference_provenance_sha256": _binding_sha256(
            plan["reference_provenance"]
        ),
        "cache_identity_sha256": _binding_sha256(plan["cache_identity"]),
        "split_identity_sha256": _binding_sha256(plan["split_identity"]),
        "table_row_counts": {
            name: len(rows) for name, rows in tables.items()
        },
    }
    result = AnalysisResult(
        summary=summary, tables=tables, input_ledger=ledger
    )
    result._seal = _result_seal(result)
    _validate_result(result)
    _validate_against_plan(result, plan, jobs)
    return result


def _reject_published_trial_material(value: Any, path: str = "result") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().lower()
            if normalized in _FORBIDDEN_PUBLISHED_KEYS:
                raise AuthorRecipeAnalysisError(
                    f"published analysis contains forbidden key {path}.{key}"
                )
            _reject_published_trial_material(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_published_trial_material(child, f"{path}[{index}]")


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(
        author_recipe_grid.HEX_64_RE.fullmatch(value)
    )


def _validate_scalar_table(
    name: str, rows: Any
) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or not rows:
        raise AuthorRecipeAnalysisError(
            f"analysis table {name} must be a nonempty list"
        )
    expected = TABLE_SCHEMAS[name]
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != expected:
            raise AuthorRecipeAnalysisError(
                f"analysis table {name} row {index} violates its exact schema"
            )
        if row.get("track") != TRACK:
            raise AuthorRecipeAnalysisError(
                f"analysis table {name} row {index} escaped its track"
            )
        for key, value in row.items():
            if value is not None and not isinstance(value, (str, int, float)):
                raise AuthorRecipeAnalysisError(
                    f"analysis table {name} row {index} field {key} "
                    "is not scalar"
                )
            if isinstance(value, bool):
                raise AuthorRecipeAnalysisError(
                    f"analysis table {name} row {index} contains a boolean scalar"
                )
            if isinstance(value, float) and not math.isfinite(value):
                raise AuthorRecipeAnalysisError(
                    f"analysis table {name} row {index} contains non-finite data"
                )
    return rows


def _validate_metric_ranges(
    name: str, rows: Sequence[Mapping[str, Any]]
) -> None:
    unit_interval = {
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "ovr_macro_auroc",
        "ece",
    }
    signed_interval = {
        "cohen_kappa",
        "chance_normalized_balanced_accuracy",
    }
    nonnegative = {"nll", "multiclass_brier"}
    for row in rows:
        for metric in METRIC_NAMES:
            value = row.get(metric)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise AuthorRecipeAnalysisError(
                    f"{name} has non-numeric {metric}"
                )
            number = float(value)
            if metric in unit_interval and not 0.0 <= number <= 1.0:
                raise AuthorRecipeAnalysisError(
                    f"{name} has out-of-range {metric}"
                )
            if metric in signed_interval and not -1.0 <= number <= 1.0:
                raise AuthorRecipeAnalysisError(
                    f"{name} has out-of-range {metric}"
                )
            if metric in nonnegative and number < 0.0:
                raise AuthorRecipeAnalysisError(f"{name} has negative {metric}")
            defined_name = f"{metric}_defined_count"
            if defined_name in row:
                count = row[defined_name]
                if (
                    isinstance(count, bool)
                    or not isinstance(count, int)
                    or count < 0
                ):
                    raise AuthorRecipeAnalysisError(
                        f"{name} has invalid {defined_name}"
                    )


def _validate_result(result: AnalysisResult) -> None:
    """Reject fabricated schemas before any publication-time recomputation."""

    if not isinstance(result, AnalysisResult):
        raise AuthorRecipeAnalysisError(
            "publication requires an AnalysisResult instance"
        )
    if not hmac.compare_digest(result._seal, _result_seal(result)):
        raise AuthorRecipeAnalysisError("analysis result seal is invalid")
    if not isinstance(result.summary, dict) or set(result.summary) != SUMMARY_KEYS:
        raise AuthorRecipeAnalysisError("analysis summary violates its exact schema")
    if tuple(result.tables) != TABLE_NAMES:
        raise AuthorRecipeAnalysisError("analysis tables have invalid exact roster")
    validated = {
        name: _validate_scalar_table(name, result.tables[name])
        for name in TABLE_NAMES
    }
    summary = result.summary
    if (
        summary["schema"] != ANALYSIS_SCHEMA
        or summary["track"] != TRACK
        or summary["track_description"] != author_recipe_grid.TRACK_DESCRIPTION
        or summary["common_recipe_track"] is not False
        or summary["confirmation_evidence"] is not False
        or summary["interpretation"] != INTERPRETATION
        or summary["aggregation_definition"] != AGGREGATION_DEFINITION
        or summary["metric_definitions"] != METRIC_DEFINITIONS
        or summary["inferential_comparisons"] != INFERENTIAL_COMPARISONS
        or summary["ece_bins"] != DEFAULT_ECE_BINS
    ):
        raise AuthorRecipeAnalysisError(
            "analysis summary identity, decisions, or scope is invalid"
        )
    if summary["identity_contract"] != {
        "namespace": "registry_stable_id_author_recipe_adapted",
        "stable_ids": list(author_recipe_grid.REFERENCES),
        "bare_common_architecture_aliases_forbidden": [
            "tcformer",
            "fbcnet",
        ],
    }:
        raise AuthorRecipeAnalysisError(
            "analysis result lost author/common identity separation"
        )
    if summary["decision_contract_sha256"] != _binding_sha256(
        decision_contract()
    ):
        raise AuthorRecipeAnalysisError("analysis decision contract is invalid")
    digest_fields = (
        "plan_sha256",
        "input_ledger_sha256",
        "analysis_environment_sha256",
        "analysis_contract_sha256",
        "decision_contract_sha256",
        "source_identity_sha256",
        "environment_identity_sha256",
        "reference_provenance_sha256",
        "cache_identity_sha256",
        "split_identity_sha256",
    )
    if any(not _is_sha256(summary[name]) for name in digest_fields):
        raise AuthorRecipeAnalysisError("analysis summary has an invalid digest")
    if (
        not isinstance(summary["analysis_source_identity"], Mapping)
        or summary["analysis_source_identity"] != _analysis_source_identity()
        or not isinstance(summary["audit_snapshot"], Mapping)
        or not isinstance(summary["references"], list)
        or not isinstance(summary["datasets"], list)
        or not isinstance(summary["seeds"], list)
        or isinstance(summary["expected_jobs"], bool)
        or not isinstance(summary["expected_jobs"], int)
        or summary["expected_jobs"] <= 0
    ):
        raise AuthorRecipeAnalysisError(
            "analysis summary source, roster, or audit binding is invalid"
        )
    counts = {name: len(rows) for name, rows in validated.items()}
    if summary["table_row_counts"] != counts:
        raise AuthorRecipeAnalysisError(
            "analysis table counts differ from the summary"
        )
    if not isinstance(result.input_ledger, list) or not result.input_ledger:
        raise AuthorRecipeAnalysisError("analysis input ledger is absent")
    for index, row in enumerate(result.input_ledger):
        if not isinstance(row, dict) or set(row) != LEDGER_KEYS:
            raise AuthorRecipeAnalysisError(
                f"input ledger row {index} violates its exact schema"
            )
        if (
            row["kind"] not in {"plan", "cache", "job"}
            or not isinstance(row["identity"], str)
            or not row["identity"]
            or any(
                not _is_sha256(row[key])
                for key in ("sha256_a", "sha256_b", "sha256_c")
            )
        ):
            raise AuthorRecipeAnalysisError(
                f"input ledger row {index} is invalid"
            )
    if summary["input_ledger_sha256"] != _binding_sha256(
        result.input_ledger
    ):
        raise AuthorRecipeAnalysisError("analysis input ledger digest is invalid")
    for name, rows in validated.items():
        if name != "timing_summary":
            _validate_metric_ranges(name, rows)
    _reject_published_trial_material(
        {"summary": result.summary, "tables": result.tables}
    )


def _validate_aggregate_defined_counts(
    rows: Sequence[Mapping[str, Any]], *, maximum: int, context: str
) -> None:
    for row in rows:
        for metric in METRIC_NAMES:
            count = row[f"{metric}_defined_count"]
            value = row[metric]
            if count > maximum or (count == 0) != (value is None):
                raise AuthorRecipeAnalysisError(
                    f"{context} has inconsistent {metric} defined count"
                )


def _validate_against_plan(
    result: AnalysisResult,
    plan: Mapping[str, Any],
    jobs: Sequence[author_recipe_grid.Job],
) -> None:
    summary = result.summary
    expected_environment = author_recipe_grid._normalized_worker_environment(
        plan["environment_identity"]
    )
    expected_summary = {
        "plan_sha256": plan["plan_sha256"],
        "analysis_contract_sha256": _binding_sha256(
            plan.get("analysis_contract")
        ),
        "decision_contract_sha256": _binding_sha256(decision_contract()),
        "source_identity_sha256": _binding_sha256(plan["source_identity"]),
        "environment_identity_sha256": _binding_sha256(
            plan["environment_identity"]
        ),
        "reference_provenance_sha256": _binding_sha256(
            plan["reference_provenance"]
        ),
        "cache_identity_sha256": _binding_sha256(plan["cache_identity"]),
        "split_identity_sha256": _binding_sha256(plan["split_identity"]),
        "analysis_environment_sha256": _binding_sha256(expected_environment),
        "analysis_source_identity": _analysis_source_identity(),
        "identity_contract": _json_ready(plan["identity_contract"]),
        "references": list(plan["references"]),
        "datasets": list(plan["dataset_order"]),
        "seeds": list(plan["seeds"]),
        "expected_jobs": len(jobs),
        "evidence_scope": plan["evidence_scope"],
    }
    if any(summary.get(key) != value for key, value in expected_summary.items()):
        raise AuthorRecipeAnalysisError(
            "analysis summary differs from immutable plan bindings"
        )
    analysis_contract = plan.get("analysis_contract")
    if isinstance(analysis_contract, Mapping):
        if (
            analysis_contract.get("decision_contract") != decision_contract()
            or analysis_contract.get("decision_contract_sha256")
            != _binding_sha256(decision_contract())
            or analysis_contract.get("source_identity")
            != _analysis_source_identity()
        ):
            raise AuthorRecipeAnalysisError(
                "plan's frozen analysis decision contract has drifted"
            )
    job_rows = result.tables["job_metrics"]
    if len(job_rows) != len(jobs):
        raise AuthorRecipeAnalysisError("job table cardinality differs from plan")
    for job, row in zip(jobs, job_rows, strict=True):
        if (
            row["job_id"] != job.job_id
            or any(row[key] != value for key, value in job.identity().items())
        ):
            raise AuthorRecipeAnalysisError(
                f"job table identity differs for {job.job_id}"
            )
        expected_count = int(
            author_recipe_grid._planned_split(plan, job)["partitions"]["test"][
                "count"
            ]
        )
        if (
            row["test_count"] != expected_count
            or isinstance(row["parameter_count"], bool)
            or not isinstance(row["parameter_count"], int)
            or row["parameter_count"] <= 0
        ):
            raise AuthorRecipeAnalysisError(
                f"job table counters differ for {job.job_id}"
            )
        for field in (
            "source_fit_seconds",
            "test_inference_seconds",
            "job_total_seconds",
            "inference_ms_per_trial",
        ):
            if float(row[field]) < 0.0:
                raise AuthorRecipeAnalysisError(
                    f"job table timing is invalid for {job.job_id}"
                )
        if float(row["job_total_seconds"]) + 1e-9 < (
            float(row["source_fit_seconds"])
            + float(row["test_inference_seconds"])
        ) or not math.isclose(
            float(row["inference_ms_per_trial"]),
            1000.0 * float(row["test_inference_seconds"]) / expected_count,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise AuthorRecipeAnalysisError(
                f"job table timing identity differs for {job.job_id}"
            )

    expected_subject_seed = [
        (str(dataset), str(reference), int(subject), int(seed))
        for dataset in plan["dataset_order"]
        for reference in plan["references"]
        for subject in plan["datasets"][dataset]["subjects"]
        for seed in plan["seeds"]
    ]
    observed_subject_seed = [
        (
            str(row["dataset"]),
            str(row["reference"]),
            int(row["subject"]),
            int(row["seed"]),
        )
        for row in result.tables["subject_seed_metrics"]
    ]
    if observed_subject_seed != expected_subject_seed:
        raise AuthorRecipeAnalysisError("subject/seed table differs from plan order")
    for row in result.tables["subject_seed_metrics"]:
        dataset = str(row["dataset"])
        subject = int(row["subject"])
        expected_trials = sum(
            int(
                plan["split_identity"][
                    author_recipe_grid._split_key(dataset, subject, int(fold))
                ]["partitions"]["test"]["count"]
            )
            for fold in plan["datasets"][dataset]["folds"]
        )
        if (
            row["folds_concatenated"]
            != len(plan["datasets"][dataset]["folds"])
            or row["test_trials"] != expected_trials
        ):
            raise AuthorRecipeAnalysisError(
                "subject/seed fold concatenation differs from plan"
            )

    expected_subjects = [
        (str(dataset), str(reference), int(subject))
        for dataset in plan["dataset_order"]
        for reference in plan["references"]
        for subject in plan["datasets"][dataset]["subjects"]
    ]
    observed_subjects = [
        (
            str(row["dataset"]),
            str(row["reference"]),
            int(row["subject"]),
        )
        for row in result.tables["subject_summary"]
    ]
    if observed_subjects != expected_subjects:
        raise AuthorRecipeAnalysisError("subject summary differs from plan order")
    for row in result.tables["subject_summary"]:
        if (
            row["seeds_averaged"] != len(plan["seeds"])
            or row["aggregation"] != SUBJECT_AGGREGATION
        ):
            raise AuthorRecipeAnalysisError("subject aggregation contract drifted")
    _validate_aggregate_defined_counts(
        result.tables["subject_summary"],
        maximum=len(plan["seeds"]),
        context="subject summary",
    )

    expected_datasets = [
        (str(dataset), str(reference))
        for dataset in plan["dataset_order"]
        for reference in plan["references"]
    ]
    observed_datasets = [
        (str(row["dataset"]), str(row["reference"]))
        for row in result.tables["dataset_summary"]
    ]
    if observed_datasets != expected_datasets:
        raise AuthorRecipeAnalysisError("dataset summary differs from plan order")
    for row in result.tables["dataset_summary"]:
        if (
            row["subjects_averaged"]
            != len(plan["datasets"][str(row["dataset"])]["subjects"])
            or row["aggregation"] != DATASET_AGGREGATION
        ):
            raise AuthorRecipeAnalysisError("dataset aggregation contract drifted")
    for dataset in plan["dataset_order"]:
        _validate_aggregate_defined_counts(
            [
                row
                for row in result.tables["dataset_summary"]
                if row["dataset"] == dataset
            ],
            maximum=len(plan["datasets"][dataset]["subjects"]),
            context=f"dataset summary {dataset}",
        )

    if [row["reference"] for row in result.tables["overall_summary"]] != list(
        plan["references"]
    ):
        raise AuthorRecipeAnalysisError("overall summary differs from plan order")
    for row in result.tables["overall_summary"]:
        if (
            row["datasets_equal_weighted"] != len(plan["dataset_order"])
            or row["aggregation"] != OVERALL_AGGREGATION
        ):
            raise AuthorRecipeAnalysisError("overall aggregation contract drifted")
    _validate_aggregate_defined_counts(
        result.tables["overall_summary"],
        maximum=len(plan["dataset_order"]),
        context="overall summary",
    )

    expected_timing = [
        (str(dataset), str(reference))
        for dataset in (*plan["dataset_order"], "ALL_DATASETS")
        for reference in plan["references"]
    ]
    observed_timing = [
        (str(row["dataset"]), str(row["reference"]))
        for row in result.tables["timing_summary"]
    ]
    if observed_timing != expected_timing:
        raise AuthorRecipeAnalysisError("timing summary differs from plan order")
    for row in result.tables["timing_summary"]:
        dataset = str(row["dataset"])
        reference = str(row["reference"])
        expected_jobs = sum(
            1
            for job in jobs
            if job.reference == reference
            and (dataset == "ALL_DATASETS" or job.dataset == dataset)
        )
        if row["jobs"] != expected_jobs:
            raise AuthorRecipeAnalysisError("timing summary job count is invalid")
        for base in _TIMING_BASES:
            for suffix in ("mean", "median", "p95", "min", "max"):
                if float(row[f"{base}_{suffix}"]) < 0.0:
                    raise AuthorRecipeAnalysisError(
                        "timing summary contains a negative value"
                    )

    for table_name in (
        "job_metrics",
        "subject_seed_metrics",
        "subject_summary",
        "dataset_summary",
    ):
        for row in result.tables[table_name]:
            dataset = str(row["dataset"])
            ba = row["balanced_accuracy"]
            normalized = row["chance_normalized_balanced_accuracy"]
            if ba is not None and normalized is not None:
                chance = 1.0 / int(plan["datasets"][dataset]["n_classes"])
                expected = (float(ba) - chance) / (1.0 - chance)
                if not math.isclose(
                    float(normalized), expected, rel_tol=0.0, abs_tol=1e-12
                ):
                    raise AuthorRecipeAnalysisError(
                        f"{table_name} has inconsistent chance-normalized BA"
                    )

    expected_ledger_identities = [
        ("plan", str(plan["plan_sha256"])),
        *[
            (
                "cache",
                author_recipe_grid._subject_key(dataset, int(subject)),
            )
            for dataset in plan["dataset_order"]
            for subject in plan["datasets"][dataset]["subjects"]
        ],
        *[("job", job.job_id) for job in jobs],
    ]
    observed_ledger_identities = [
        (str(row["kind"]), str(row["identity"]))
        for row in result.input_ledger
    ]
    if observed_ledger_identities != expected_ledger_identities:
        raise AuthorRecipeAnalysisError(
            "input ledger ordering/cardinality differs from plan"
        )


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        raise AuthorRecipeAnalysisError("refusing an empty CSV")
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                columns.append(str(key))
                seen.add(str(key))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="raise")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                key: "" if row.get(key) is None else row.get(key)
                for key in columns
            }
        )
    return stream.getvalue().encode("utf-8")


def _format_percent(value: Any) -> str:
    number = _finite_float(value)
    return "N/A" if number is None else f"{100.0 * number:.2f}%"


def _markdown_report(result: AnalysisResult) -> str:
    lines = [
        "# Author-recipe neural reference results",
        "",
        "> Separate author-recipe adapted optimization track on opened development "
        "cohorts. These values must not be inserted into common-recipe model "
        "rows and are not confirmation or global/SOTA evidence.",
        "",
        f"- Plan SHA-256: `{result.summary['plan_sha256']}`",
        f"- Score-blind atomic records: {result.summary['expected_jobs']:,}",
        "- Aggregation: fold concatenation, seed mean, subject mean, then "
        "equal-dataset mean.",
        "",
        "## Equal-dataset descriptive results",
        "",
        "| Reference | Accuracy | Balanced accuracy | Chance-normalized BA | "
        "Macro F1 | Kappa | AUROC | NLL | Brier | ECE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result.tables["overall_summary"]:
        lines.append(
            f"| `{row['reference']}` | {_format_percent(row['accuracy'])} | "
            f"{_format_percent(row['balanced_accuracy'])} | "
            f"{_format_percent(row['chance_normalized_balanced_accuracy'])} | "
            f"{_format_percent(row['macro_f1'])} | "
            f"{_format_percent(row['cohen_kappa'])} | "
            f"{_format_percent(row['ovr_macro_auroc'])} | "
            f"{float(row['nll']):.4f} | "
            f"{float(row['multiclass_brier']):.4f} | "
            f"{_format_percent(row['ece'])} |"
        )
    lines.extend(
        [
            "",
            "Rows retain the prespecified reference order and are deliberately "
            "not sorted into a winner ranking.",
            "",
            "## Balanced accuracy by dataset",
            "",
            "| Dataset | Reference | Balanced accuracy | Chance-normalized BA |",
            "|---|---|---:|---:|",
        ]
    )
    for row in result.tables["dataset_summary"]:
        lines.append(
            f"| `{row['dataset']}` | `{row['reference']}` | "
            f"{_format_percent(row['balanced_accuracy'])} | "
            f"{_format_percent(row['chance_normalized_balanced_accuracy'])} |"
        )
    lines.extend(
        [
            "",
            "Published files contain scalar aggregates only. Test row vectors, "
            "outcomes, class decisions, and probability matrices remain absent.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_new_file(path: Path, payload: bytes, *, root: Path) -> None:
    author_recipe_grid._write_bytes_exclusive(
        path, payload, root=root
    )


def _close_publication_input_window(
    result: AnalysisResult,
    *,
    run_root: Path,
    cache_root: Path,
    require_formal_grid: bool,
    publication_fence: author_recipe_grid.PublicationFence,
) -> tuple[
    dict[str, Any],
    tuple[author_recipe_grid.Job, ...],
    dict[str, Any],
]:
    """Re-audit, re-identify, and rehash every input under the output lock."""

    plan, jobs, audit = audit_exact_grid(
        run_root,
        require_formal_grid=require_formal_grid,
        publication_fence=publication_fence,
    )
    author_recipe_grid.validate_publication_fence(
        publication_fence, plan
    )
    if require_formal_grid:
        try:
            author_recipe_grid.verify_runtime_identity(
                plan, cache_root=cache_root
            )
        except author_recipe_grid.AuthorRecipeError as error:
            raise AuthorRecipeAnalysisError(
                f"publication-time runtime identity differs from plan: {error}"
            ) from error
    _validate_result(result)
    _validate_against_plan(result, plan, jobs)
    if result.summary["audit_snapshot"] != audit:
        raise AuthorRecipeAnalysisError(
            "analysis result is not bound to the fresh exact audit"
        )
    _verify_ledger(
        run_root=run_root,
        cache_root=cache_root,
        plan=plan,
        jobs=jobs,
        ledger=result.input_ledger,
    )
    return plan, jobs, audit


def _validate_result_for_publication(
    result: AnalysisResult,
    *,
    run_root: Path,
    cache_root: Path,
    require_formal_grid: bool,
    publication_fence: author_recipe_grid.PublicationFence,
) -> None:
    """Recompute the full immutable-ledger result under the output lock."""

    _validate_result(result)
    _close_publication_input_window(
        result,
        run_root=run_root,
        cache_root=cache_root,
        require_formal_grid=require_formal_grid,
        publication_fence=publication_fence,
    )
    fresh = compute_analysis(
        run_root=run_root,
        cache_root=cache_root,
        ece_bins=DEFAULT_ECE_BINS,
        require_formal_grid=require_formal_grid,
        _publication_fence=publication_fence,
    )
    if (
        _canonical_bytes(result.summary) != _canonical_bytes(fresh.summary)
        or _canonical_bytes(result.tables) != _canonical_bytes(fresh.tables)
        or _canonical_bytes(result.input_ledger)
        != _canonical_bytes(fresh.input_ledger)
    ):
        raise AuthorRecipeAnalysisError(
            "analysis result differs from fresh publication-time recomputation"
        )


def _publication_artifacts(result: AnalysisResult) -> dict[str, bytes]:
    artifacts: dict[str, bytes] = {
        "analysis.json": _canonical_bytes(result.summary) + b"\n",
        "RESULTS.md": _markdown_report(result).encode("utf-8"),
    }
    for name in TABLE_NAMES:
        artifacts[f"{name}.csv"] = _csv_bytes(result.tables[name])
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "plan_sha256": result.summary["plan_sha256"],
        "input_ledger_sha256": result.summary["input_ledger_sha256"],
        "track": TRACK,
        "files": {
            name: _sha256_bytes(payload)
            for name, payload in sorted(artifacts.items())
        },
    }
    artifacts["manifest.json"] = _canonical_bytes(manifest) + b"\n"
    return artifacts


def _capture_published_directory(
    destination: Path, result: AnalysisResult
) -> _HeldPublishedDirectory:
    parent = destination.parent
    author_recipe_grid._require_safe_ancestors(
        destination,
        root=parent,
        include_leaf=True,
        allow_missing_leaf=False,
        context="published analysis directory",
    )
    expected = _publication_artifacts(result)
    directory_descriptor = author_recipe_grid._open_directory_anchored(
        destination
    )
    child_descriptors: dict[str, int] = {}
    child_fingerprints: dict[str, tuple[int, ...]] = {}
    try:
        directory_stat = os.fstat(directory_descriptor)
        directory_fingerprint = author_recipe_grid._stat_fingerprint(
            directory_stat
        )
        if (
            not stat.S_ISDIR(directory_stat.st_mode)
            or stat.S_IMODE(directory_stat.st_mode) != 0o555
        ):
            raise AuthorRecipeAnalysisError(
                "published analysis directory is not sealed 0555"
            )
        names = tuple(sorted(os.listdir(directory_descriptor)))
        if set(names) != set(expected):
            raise AuthorRecipeAnalysisError(
                "published analysis has an invalid exact file set"
            )
        for name in names:
            entry_stat = os.stat(
                name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            fingerprint = author_recipe_grid._stat_fingerprint(
                entry_stat
            )
            if (
                stat.S_ISLNK(entry_stat.st_mode)
                or not stat.S_ISREG(entry_stat.st_mode)
                or entry_stat.st_nlink != 1
                or entry_stat.st_mode & 0o222
            ):
                raise AuthorRecipeAnalysisError(
                    "published analysis contains an unsafe artifact"
                )
            descriptor = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=directory_descriptor,
            )
            child_descriptors[name] = descriptor
            if author_recipe_grid._stat_fingerprint(
                os.fstat(descriptor)
            ) != fingerprint:
                raise AuthorRecipeAnalysisError(
                    f"published analysis artifact changed: {name}"
                )
            payload = os.pread(descriptor, len(expected[name]) + 1, 0)
            if payload != expected[name]:
                raise AuthorRecipeAnalysisError(
                    f"published analysis artifact differs: {name}"
                )
            child_fingerprints[name] = fingerprint
        held = _HeldPublishedDirectory(
            destination=destination,
            directory_descriptor=directory_descriptor,
            child_descriptors=child_descriptors,
            directory_fingerprint=directory_fingerprint,
            child_fingerprints=child_fingerprints,
            expected_payloads=expected,
        )
        held.revalidate()
        return held
    except BaseException:
        for descriptor in child_descriptors.values():
            os.close(descriptor)
        os.close(directory_descriptor)
        raise


def _validate_published_directory(
    destination: Path, result: AnalysisResult
) -> None:
    held = _capture_published_directory(destination, result)
    try:
        held.revalidate()
    finally:
        held.close()


def _quarantine_output_path(
    path: Path,
    *,
    parent: Path,
    destination_name: str,
    reason: str,
    expected_identity: tuple[int, int],
) -> Path:
    quarantine = author_recipe_grid._safe_mkdir(
        parent / f".{destination_name}.author-analysis-quarantine",
        root=parent,
    )
    target = quarantine / (
        f"{path.name}.{reason}.{uuid.uuid4().hex}"
    )
    source_status = author_recipe_grid._anchored_lstat(path)
    if (
        int(source_status.st_dev),
        int(source_status.st_ino),
    ) != tuple(map(int, expected_identity)):
        raise author_recipe_grid.ClaimUnavailable(
            "analysis quarantine source identity changed"
        )
    source_is_directory = stat.S_ISDIR(source_status.st_mode)
    if source_is_directory:
        descriptor = author_recipe_grid._open_directory_anchored(path)
        try:
            opened = os.fstat(descriptor)
            if (
                int(opened.st_dev),
                int(opened.st_ino),
            ) != tuple(map(int, expected_identity)):
                raise author_recipe_grid.ClaimUnavailable(
                    "analysis quarantine directory changed"
                )
            os.fchmod(descriptor, 0o700)
            os.fsync(descriptor)
        except BaseException:
            try:
                os.fchmod(descriptor, 0o555)
                os.fsync(descriptor)
            finally:
                raise
        finally:
            os.close(descriptor)
        try:
            author_recipe_grid._fsync_directory(
                path.parent, root=parent
            )
        except BaseException:
            author_recipe_grid._seal_directory_read_only(
                path,
                root=parent,
                expected_identity=expected_identity,
            )
            raise
    try:
        author_recipe_grid._rename_leaf_nofollow(
            path,
            target,
            root=parent,
            expected_source_identity=expected_identity,
        )
    except BaseException:
        if not author_recipe_grid._path_binds_identity(
            target, expected_identity
        ):
            if (
                source_is_directory
                and author_recipe_grid._path_binds_identity(
                    path, expected_identity
                )
            ):
                author_recipe_grid._seal_directory_read_only(
                    path,
                    root=parent,
                    expected_identity=expected_identity,
                )
            raise
    if not author_recipe_grid._path_binds_identity(
        target, expected_identity
    ):
        raise AuthorRecipeAnalysisError(
            "quarantine destination differs from validated source"
        )
    if source_is_directory:
        author_recipe_grid._seal_directory_read_only(
            target,
            root=parent,
            expected_identity=expected_identity,
        )
    author_recipe_grid._fsync_directory(quarantine, root=parent)
    author_recipe_grid._fsync_directory(parent)
    return target


def _quarantine_owned_stage(
    stage: Path,
    *,
    parent: Path,
    destination: Path,
    identity: tuple[int, int] | None,
) -> None:
    if identity is None or not os.path.lexists(stage):
        return
    observed = os.lstat(stage)
    if (
        (observed.st_dev, observed.st_ino) != identity
        or stat.S_ISLNK(observed.st_mode)
        or not stat.S_ISDIR(observed.st_mode)
    ):
        return
    _quarantine_output_path(
        stage,
        parent=parent,
        destination_name=destination.name,
        reason="aborted-stage",
        expected_identity=identity,
    )


def _validate_output_lock(
    lock: Path,
    *,
    parent: Path,
    plan_sha256: str | None,
    destination: Path,
    expected_nonce: str | None = None,
    expected_owner: Mapping[str, Any] | None = None,
    expected_identity: tuple[int, int] | None = None,
    _bound_payload: tuple[bytes, tuple[int, ...]] | None = None,
) -> dict[str, Any]:
    if _bound_payload is None:
        payload, fingerprint = author_recipe_grid._read_regular_snapshot(
            lock,
            root=parent,
            readonly=True,
            context="analysis publication lock",
            expected_identity=expected_identity,
        )
    else:
        payload, fingerprint = _bound_payload
    value = author_recipe_grid._decode_json_object(
        payload, context=str(lock)
    )
    author_recipe_grid._require_exact_keys(
        value,
        {
            "schema",
            "created_at",
            "plan_sha256",
            "destination",
            "nonce",
            "owner",
        },
        "analysis publication lock",
    )
    author_recipe_grid._validate_timestamp(
        value["created_at"], "analysis publication lock"
    )
    author_recipe_grid._validate_owner_mapping(
        value["owner"], "analysis publication lock"
    )
    observed_identity = (fingerprint[0], fingerprint[1])
    if (
        value["schema"] != PUBLICATION_LOCK_SCHEMA
        or not isinstance(value["plan_sha256"], str)
        or not author_recipe_grid.HEX_64_RE.fullmatch(
            value["plan_sha256"]
        )
        or (
            plan_sha256 is not None
            and value["plan_sha256"] != plan_sha256
        )
        or value["destination"] != str(destination)
        or not isinstance(value["nonce"], str)
        or not re.fullmatch(r"[0-9a-f]{32}", value["nonce"])
        or (
            expected_nonce is not None
            and value["nonce"] != expected_nonce
        )
        or (
            expected_owner is not None
            and value["owner"] != dict(expected_owner)
        )
        or (
            expected_identity is not None
            and observed_identity != tuple(map(int, expected_identity))
        )
        or payload != _canonical_bytes(value) + b"\n"
    ):
        raise AuthorRecipeAnalysisError(
            "analysis publication lock is invalid or changed"
        )
    return value


def _acquire_output_lock(
    *,
    parent: Path,
    destination: Path,
    plan_sha256: str,
) -> tuple[Path, tuple[int, int], dict[str, Any]]:
    lock = parent / f".{destination.name}.author-analysis.lock"
    if os.path.lexists(lock):
        lock_stat = author_recipe_grid._anchored_lstat(lock)
        stale_identity = (
            int(lock_stat.st_dev),
            int(lock_stat.st_ino),
        )
        minimally_loaded: Mapping[str, Any] = {}
        try:
            bound_payload = author_recipe_grid._read_regular_snapshot(
                lock,
                root=parent,
                readonly=True,
                context="analysis publication-lock recovery",
                expected_identity=stale_identity,
            )
            minimally_loaded = author_recipe_grid._decode_json_object(
                bound_payload[0], context=str(lock)
            )
            stale = _validate_output_lock(
                lock,
                parent=parent,
                plan_sha256=None,
                destination=destination,
                expected_identity=stale_identity,
                _bound_payload=bound_payload,
            )
        except Exception:
            if author_recipe_grid._claim_is_live(minimally_loaded):
                raise AuthorRecipeAnalysisError(
                    "analysis destination has a live but invalid "
                    "publication lock"
                )
            reason = "invalid"
        else:
            if author_recipe_grid._claim_is_live(stale):
                raise AuthorRecipeAnalysisError(
                    "analysis destination has a live publication lock"
                )
            reason = "stale"
        try:
            _quarantine_output_path(
                lock,
                parent=parent,
                destination_name=destination.name,
                reason=reason,
                expected_identity=stale_identity,
            )
        except FileNotFoundError as error:
            raise AuthorRecipeAnalysisError(
                "lost stale publication-lock recovery race"
            ) from error
    owner = author_recipe_grid._process_identity()
    nonce = uuid.uuid4().hex
    payload = {
        "schema": PUBLICATION_LOCK_SCHEMA,
        "created_at": author_recipe_grid._utc_now(),
        "plan_sha256": plan_sha256,
        "destination": str(destination),
        "nonce": nonce,
        "owner": owner,
    }
    try:
        author_recipe_grid._write_json_exclusive(
            lock, payload, root=parent
        )
    except FileExistsError as error:
        raise AuthorRecipeAnalysisError(
            "analysis publication lock race lost"
        ) from error
    lock_stat = os.lstat(lock)
    identity = (int(lock_stat.st_dev), int(lock_stat.st_ino))
    _validate_output_lock(
        lock,
        parent=parent,
        plan_sha256=plan_sha256,
        destination=destination,
        expected_nonce=nonce,
        expected_owner=owner,
        expected_identity=identity,
    )
    return lock, identity, payload


def _release_output_lock(
    lock: Path,
    *,
    identity: tuple[int, int],
    payload: Mapping[str, Any],
    parent: Path,
    destination: Path,
) -> None:
    _validate_output_lock(
        lock,
        parent=parent,
        plan_sha256=str(payload["plan_sha256"]),
        destination=destination,
        expected_nonce=str(payload["nonce"]),
        expected_owner=payload["owner"],
        expected_identity=identity,
    )
    _quarantine_output_path(
        lock,
        parent=parent,
        destination_name=destination.name,
        reason="released",
        expected_identity=identity,
    )


def _recover_output_stages(parent: Path, destination: Path) -> None:
    prefix = f".{destination.name}.partial-"
    for entry in os.scandir(parent):
        if not entry.name.startswith(prefix):
            continue
        observed = entry.stat(follow_symlinks=False)
        try:
            _quarantine_output_path(
                Path(entry.path),
                parent=parent,
                destination_name=destination.name,
                reason="stale-stage",
                expected_identity=(
                    int(observed.st_dev),
                    int(observed.st_ino),
                ),
            )
        except FileNotFoundError:
            continue


def _rebind_analysis_publication_authority(
    result: AnalysisResult,
    *,
    run_root: Path,
    cache_root: Path,
    require_formal_grid: bool,
    publication_fence: author_recipe_grid.PublicationFence,
    lock: Path,
    lock_identity: tuple[int, int],
    lock_payload: Mapping[str, Any],
    destination: Path,
    expected_plan: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Rebind all result inputs plus both held publication authorities."""

    live_plan, _, _ = _close_publication_input_window(
        result,
        run_root=run_root,
        cache_root=cache_root,
        require_formal_grid=require_formal_grid,
        publication_fence=publication_fence,
    )
    if _canonical_bytes(live_plan) != _canonical_bytes(expected_plan):
        raise AuthorRecipeAnalysisError(
            "immutable plan changed at analysis publication boundary"
        )
    author_recipe_grid.validate_publication_fence(
        publication_fence, live_plan
    )
    _validate_output_lock(
        lock,
        parent=destination.parent,
        plan_sha256=str(live_plan["plan_sha256"]),
        destination=destination,
        expected_nonce=str(lock_payload["nonce"]),
        expected_owner=lock_payload["owner"],
        expected_identity=lock_identity,
    )
    return live_plan


def publish_analysis(
    result: AnalysisResult,
    *,
    run_root: str | Path,
    cache_root: str | Path,
    output_dir: str | Path,
    require_formal_grid: bool = True,
) -> Path:
    _validate_result(result)
    root = author_recipe_grid._absolute_path(run_root)
    cache = author_recipe_grid._absolute_path(cache_root)
    destination = author_recipe_grid._absolute_path(output_dir)
    try:
        author_recipe_grid._require_safe_ancestors(
            root,
            include_leaf=True,
            allow_missing_leaf=False,
            context="publication run root",
        )
        author_recipe_grid._require_safe_ancestors(
            cache,
            include_leaf=True,
            allow_missing_leaf=False,
            context="publication cache root",
        )
    except (FileNotFoundError, author_recipe_grid.AuthorRecipeError) as error:
        raise AuthorRecipeAnalysisError(
            "publication run/cache root is unsafe"
        ) from error
    if (
        destination == root
        or root in destination.parents
        or destination == cache
        or cache in destination.parents
    ):
        raise AuthorRecipeAnalysisError(
            "analysis output must be outside run_root and cache_root"
        )
    try:
        parent = author_recipe_grid._safe_mkdir(destination.parent)
        author_recipe_grid._require_safe_ancestors(
            destination,
            root=parent,
            context="analysis destination",
        )
    except author_recipe_grid.AuthorRecipeError as error:
        raise AuthorRecipeAnalysisError(
            "analysis destination has an unsafe ancestor"
        ) from error
    parent_stat = author_recipe_grid._anchored_lstat(parent)
    parent_identity = (
        int(parent_stat.st_dev),
        int(parent_stat.st_ino),
    )
    lock: Path | None = None
    lock_identity: tuple[int, int] | None = None
    lock_payload: dict[str, Any] | None = None
    fence: author_recipe_grid.PublicationFence | None = None
    stage = parent / f".{destination.name}.partial-{uuid.uuid4().hex}"
    stage_identity: tuple[int, int] | None = None
    published_identity: tuple[int, int] | None = None
    held_publication: _HeldPublishedDirectory | None = None
    plan = author_recipe_grid.load_plan(root)
    if plan.get("plan_sha256") != result.summary["plan_sha256"]:
        raise AuthorRecipeAnalysisError(
            "analysis result and run-root plan differ"
        )
    try:
        lock, lock_identity, lock_payload = _acquire_output_lock(
            parent=parent,
            destination=destination,
            plan_sha256=str(plan["plan_sha256"]),
        )
        _recover_output_stages(parent, destination)
        try:
            fence = author_recipe_grid.acquire_publication_fence(
                root, plan, destination=destination
            )
        except (
            author_recipe_grid.AuthorRecipeError,
            author_recipe_grid.ClaimUnavailable,
        ) as error:
            raise AuthorRecipeAnalysisError(
                f"cannot fence run root for publication: {error}"
            ) from error
        _validate_result_for_publication(
            result,
            run_root=root,
            cache_root=cache,
            require_formal_grid=require_formal_grid,
            publication_fence=fence,
        )
        if os.path.lexists(destination):
            held_publication = _capture_published_directory(
                destination, result
            )
            _rebind_analysis_publication_authority(
                result,
                run_root=root,
                cache_root=cache,
                require_formal_grid=require_formal_grid,
                publication_fence=fence,
                lock=lock,
                lock_identity=lock_identity,
                lock_payload=lock_payload,
                destination=destination,
                expected_plan=plan,
            )
            held_publication.revalidate()
            return destination
        author_recipe_grid._safe_mkdir(
            stage, root=parent, mode=0o700, exist_ok=False
        )
        stage_stat = author_recipe_grid._anchored_lstat(stage)
        stage_identity = (int(stage_stat.st_dev), int(stage_stat.st_ino))
        artifacts = _publication_artifacts(result)
        for name, payload in artifacts.items():
            _write_new_file(stage / name, payload, root=stage)
        author_recipe_grid._fsync_directory(stage, root=parent)
        # This is the final audit/recomputation window, still under the
        # run-root exclusive fence. No cooperative worker can acquire, retain,
        # or publish a claim until after the output rename is durable.
        _rebind_analysis_publication_authority(
            result,
            run_root=root,
            cache_root=cache,
            require_formal_grid=require_formal_grid,
            publication_fence=fence,
            lock=lock,
            lock_identity=lock_identity,
            lock_payload=lock_payload,
            destination=destination,
            expected_plan=plan,
        )
        current_parent = author_recipe_grid._anchored_lstat(parent)
        current_stage = author_recipe_grid._anchored_lstat(stage)
        if (
            (int(current_parent.st_dev), int(current_parent.st_ino))
            != parent_identity
            or stage_identity
            != (int(current_stage.st_dev), int(current_stage.st_ino))
            or stat.S_ISLNK(current_stage.st_mode)
            or not stat.S_ISDIR(current_stage.st_mode)
            or stat.S_IMODE(current_stage.st_mode) != 0o700
            or os.path.lexists(destination)
        ):
            raise AuthorRecipeAnalysisError(
                "publication parent/stage changed before atomic rename"
            )
        try:
            author_recipe_grid._rename_leaf_nofollow(
                stage,
                destination,
                root=parent,
                expected_source_identity=stage_identity,
            )
        except BaseException:
            if author_recipe_grid._path_binds_identity(
                destination, stage_identity
            ):
                published_identity = stage_identity
                author_recipe_grid._seal_directory_read_only(
                    destination,
                    root=parent,
                    expected_identity=stage_identity,
                )
                _rebind_analysis_publication_authority(
                    result,
                    run_root=root,
                    cache_root=cache,
                    require_formal_grid=require_formal_grid,
                    publication_fence=fence,
                    lock=lock,
                    lock_identity=lock_identity,
                    lock_payload=lock_payload,
                    destination=destination,
                    expected_plan=plan,
                )
            raise
        if not author_recipe_grid._path_binds_identity(
            destination, stage_identity
        ):
            raise AuthorRecipeAnalysisError(
                "published directory inode differs from the staged directory"
            )
        published_identity = stage_identity
        author_recipe_grid._seal_directory_read_only(
            destination,
            root=parent,
            expected_identity=stage_identity,
        )
        author_recipe_grid._fsync_directory(parent)
        held_publication = _capture_published_directory(
            destination, result
        )
        _rebind_analysis_publication_authority(
            result,
            run_root=root,
            cache_root=cache,
            require_formal_grid=require_formal_grid,
            publication_fence=fence,
            lock=lock,
            lock_identity=lock_identity,
            lock_payload=lock_payload,
            destination=destination,
            expected_plan=plan,
        )
        held_publication.revalidate()
        return destination
    except BaseException:
        if (
            published_identity is not None
            and os.path.lexists(destination)
        ):
            try:
                _quarantine_output_path(
                    destination,
                    parent=parent,
                    destination_name=destination.name,
                    reason="failed",
                    expected_identity=published_identity,
                )
            except (
                FileNotFoundError,
                author_recipe_grid.ClaimUnavailable,
            ):
                pass
            finally:
                published_identity = None
        if stage_identity is not None and os.path.lexists(stage):
            try:
                _quarantine_output_path(
                    stage,
                    parent=parent,
                    destination_name=destination.name,
                    reason="failed-stage",
                    expected_identity=stage_identity,
                )
            except (
                FileNotFoundError,
                author_recipe_grid.ClaimUnavailable,
            ):
                pass
        raise
    finally:
        release_error: BaseException | None = None
        if stage_identity is not None and os.path.lexists(stage):
            try:
                _quarantine_owned_stage(
                    stage,
                    parent=parent,
                    destination=destination,
                    identity=stage_identity,
                )
            except BaseException as error:
                release_error = error
        if fence is not None:
            try:
                author_recipe_grid.release_publication_fence(
                    fence, plan
                )
            except BaseException as error:
                if release_error is None:
                    release_error = error
        if (
            lock is not None
            and lock_identity is not None
            and lock_payload is not None
        ):
            try:
                _release_output_lock(
                    lock,
                    identity=lock_identity,
                    payload=lock_payload,
                    parent=parent,
                    destination=destination,
                )
            except BaseException as error:
                if release_error is None:
                    release_error = error
        if release_error is not None and published_identity is not None:
            if os.path.lexists(destination):
                try:
                    _quarantine_output_path(
                        destination,
                        parent=parent,
                        destination_name=destination.name,
                        reason="release-failed",
                        expected_identity=published_identity,
                    )
                except (
                    FileNotFoundError,
                    author_recipe_grid.ClaimUnavailable,
                ):
                    pass
            published_identity = None
        if held_publication is not None:
            try:
                if release_error is None:
                    held_publication.revalidate()
            except BaseException as error:
                if release_error is None:
                    release_error = error
            finally:
                held_publication.close()
        if release_error is not None and published_identity is not None:
            if os.path.lexists(destination):
                try:
                    _quarantine_output_path(
                        destination,
                        parent=parent,
                        destination_name=destination.name,
                        reason="final-validation-failed",
                        expected_identity=published_identity,
                    )
                except (
                    FileNotFoundError,
                    author_recipe_grid.ClaimUnavailable,
                ):
                    pass
            published_identity = None
        if release_error is None:
            published_identity = None
        if release_error is not None and sys.exc_info()[0] is None:
            raise release_error


def analyze_and_publish(
    *,
    run_root: str | Path,
    cache_root: str | Path,
    output_dir: str | Path,
    ece_bins: int = DEFAULT_ECE_BINS,
) -> tuple[AnalysisResult, Path]:
    result = compute_analysis(
        run_root=run_root,
        cache_root=cache_root,
        ece_bins=ece_bins,
    )
    destination = publish_analysis(
        result,
        run_root=run_root,
        cache_root=cache_root,
        output_dir=output_dir,
    )
    return result, destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--ece-bins",
        type=int,
        choices=(DEFAULT_ECE_BINS,),
        default=DEFAULT_ECE_BINS,
        help="frozen by plan; only 15 is accepted",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result, destination = analyze_and_publish(
        run_root=args.run_root,
        cache_root=args.cache_root,
        output_dir=args.output_dir,
        ece_bins=args.ece_bins,
    )
    print(
        json.dumps(
            {
                "plan_sha256": result.summary["plan_sha256"],
                "track": TRACK,
                "output_dir": str(destination),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
