"""Quiescent score-joining analysis for the GeoAdapt harmonized-v2 track.

The producer in :mod:`benchmark.geoadapt_v2` writes no outcomes or performance
values.  This module first requires the exact immutable prediction Cartesian
product to be complete and quiescent, then joins plan-bound cache outcomes to
checksum-bound held-out rows in memory.  Published artifacts contain aggregate
metrics only; trial outcomes and probability matrices are never exported.

Results remain explicitly labelled as a separate procedure track.  They must
not be inserted into the common raw-architecture table or treated as
BNCI2014-001 results.  Fixed GeoAdaptNet and GeoAdaptNet-FBSP x
BNCI2014-004 are also retained as explicit N/A cells because their frozen
eight-dimensional spatial paths are incompatible with the three supplied
bipolar channels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import stat
import time
import uuid
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
)

from . import geoadapt_v2


ANALYSIS_SCHEMA = "eeg-mi-geoadapt-v2-analysis-v2"
MANIFEST_SCHEMA = "eeg-mi-geoadapt-v2-analysis-manifest-v2"
DEFAULT_ECE_BINS = 15
COMMON_SUPPORT_DATASETS: tuple[str, ...] = (
    "local_exp4",
    "cho2017",
    "physionet_mi",
)
PRIMARY_INFERENCE_METRIC = "balanced_accuracy"
STATISTICAL_SEED = 20_260_729
BOOTSTRAP_RESAMPLES = 10_000
RANDOMIZATION_RESAMPLES = 20_000
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
TABLE_FIELDS: Mapping[str, tuple[str, ...]] = {
    "job_metrics": (
        "track",
        "stable_id",
        "dataset",
        "subject",
        "fold",
        "seed",
        "n_trials",
        *METRIC_NAMES,
    ),
    "subject_seed_metrics": (
        "track",
        "stable_id",
        "dataset",
        "subject",
        "seed",
        "n_folds",
        "n_trials",
        *METRIC_NAMES,
    ),
    "subject_metrics": (
        "track",
        "stable_id",
        "dataset",
        "subject",
        "n_seeds",
        "n_trials_per_seed",
        *METRIC_NAMES,
    ),
    "dataset_summary": (
        "track",
        "stable_id",
        "dataset",
        "n_subjects",
        "n_seeds",
        *METRIC_NAMES,
    ),
    "overall_summary": (
        "track",
        "stable_id",
        "n_datasets",
        "dataset_coverage",
        *METRIC_NAMES,
    ),
    "timing_summary": (
        "track",
        "stable_id",
        "dataset",
        "n_jobs",
        "mean_selection_fit_seconds",
        "mean_refit_fit_seconds",
        "mean_test_inference_seconds",
        "mean_job_total_seconds",
        "mean_parameter_count",
        "mean_cuda_peak_memory_bytes",
    ),
    "not_applicable": (
        "track",
        "stable_id",
        "dataset",
        "status",
        "reason",
    ),
    "common_support_summary": (
        "track",
        "stable_id",
        "metric",
        "n_datasets",
        "dataset_coverage",
        "n_subjects",
        "estimate",
        "bootstrap_ci95_low",
        "bootstrap_ci95_high",
        "bootstrap_resamples",
    ),
    "paired_common_support": (
        "track",
        "stable_id_a",
        "stable_id_b",
        "metric",
        "dataset_coverage",
        "n_paired_subjects",
        "mean_difference_a_minus_b",
        "bootstrap_ci95_low",
        "bootstrap_ci95_high",
        "randomization_p_two_sided",
        "holm_adjusted_p",
        "bootstrap_resamples",
        "randomization_resamples",
        "multiplicity_method",
    ),
}


class GeoAdaptV2AnalysisError(RuntimeError):
    """Raised when analysis cannot preserve its score-joining boundary."""


class PublicationContention(GeoAdaptV2AnalysisError):
    """A validated publication inode was replaced by a competing winner."""


@dataclass
class AnalysisResult:
    summary: dict[str, Any]
    tables: dict[str, list[dict[str, Any]]]
    input_ledger: list[dict[str, str]] = field(repr=False)


CompletionSnapshot = geoadapt_v2.CompletionSnapshot


@dataclass(frozen=True)
class PublicationLock:
    path: Path
    nonce: str
    owner: Mapping[str, Any]
    st_dev: int
    st_ino: int


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


def _sha256_file(path: Path) -> str:
    return geoadapt_v2._sha256_file(path)


def classification_metrics(
    y_true: np.ndarray | Sequence[int],
    probabilities: np.ndarray,
    *,
    n_classes: int = 2,
) -> dict[str, float | None]:
    y = np.asarray(y_true, dtype=np.int64)
    values = np.asarray(probabilities, dtype=np.float64)
    if (
        y.ndim != 1
        or len(y) == 0
        or values.shape != (len(y), n_classes)
        or n_classes < 2
        or np.any(y < 0)
        or np.any(y >= n_classes)
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(values > 1.0)
        or not np.allclose(values.sum(axis=1), 1.0, rtol=0.0, atol=1e-6)
    ):
        raise GeoAdaptV2AnalysisError("invalid arrays supplied to metric computation")
    predicted = np.argmax(values, axis=1)
    classes = np.arange(n_classes, dtype=np.int64)
    support = np.bincount(y, minlength=n_classes)
    correct = np.bincount(y[predicted == y], minlength=n_classes)
    present = support > 0
    balanced = float(np.mean(correct[present] / support[present]))
    chance = 1.0 / n_classes
    normalized = (balanced - chance) / (1.0 - chance)
    with np.errstate(all="ignore"):
        kappa = _finite_float(cohen_kappa_score(y, predicted, labels=classes.tolist()))
    aucs: list[float] = []
    for index in range(n_classes):
        binary = (y == index).astype(np.int8)
        if binary.min() == binary.max():
            continue
        aucs.append(float(roc_auc_score(binary, values[:, index])))
    clipped = np.clip(values[np.arange(len(y)), y], 1e-12, 1.0)
    one_hot = np.eye(n_classes, dtype=np.float64)[y]
    confidences = np.max(values, axis=1)
    correctness = (predicted == y).astype(np.float64)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, 16)
    for index in range(15):
        if index == 0:
            mask = (confidences >= edges[index]) & (confidences <= edges[index + 1])
        else:
            mask = (confidences > edges[index]) & (confidences <= edges[index + 1])
        if np.any(mask):
            ece += float(np.mean(mask)) * abs(
                float(np.mean(correctness[mask])) - float(np.mean(confidences[mask]))
            )
    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": balanced,
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
        "cohen_kappa": kappa,
        "ovr_macro_auroc": float(np.mean(aucs)) if aucs else None,
        "nll": float(-np.mean(np.log(clipped))),
        "multiclass_brier": float(np.mean(np.sum((values - one_hot) ** 2, axis=1))),
        "ece": float(ece),
    }


def _mean(values: Sequence[Any]) -> float | None:
    finite = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return float(np.mean(finite)) if finite else None


def _statistics_rng(label: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{STATISTICAL_SEED}:{label}".encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def _equal_dataset_estimate(
    values_by_dataset: Mapping[str, np.ndarray],
) -> float:
    if tuple(sorted(values_by_dataset)) != tuple(sorted(COMMON_SUPPORT_DATASETS)):
        raise GeoAdaptV2AnalysisError(
            "common-support inference requires exactly the three frozen datasets"
        )
    if any(
        values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values))
        for values in values_by_dataset.values()
    ):
        raise GeoAdaptV2AnalysisError("common-support values are invalid")
    return float(
        np.mean([np.mean(values_by_dataset[name]) for name in COMMON_SUPPORT_DATASETS])
    )


def _stratified_bootstrap_interval(
    values_by_dataset: Mapping[str, np.ndarray],
    *,
    label: str,
) -> tuple[float, float]:
    rng = _statistics_rng(f"bootstrap:{label}")
    replicates = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for index in range(BOOTSTRAP_RESAMPLES):
        dataset_means = []
        for dataset in COMMON_SUPPORT_DATASETS:
            values = values_by_dataset[dataset]
            sampled = values[rng.integers(0, len(values), size=len(values))]
            dataset_means.append(float(np.mean(sampled)))
        replicates[index] = float(np.mean(dataset_means))
    low, high = np.quantile(replicates, (0.025, 0.975))
    return float(low), float(high)


def _paired_randomization_p(
    differences_by_dataset: Mapping[str, np.ndarray],
    *,
    label: str,
) -> float:
    observed = abs(_equal_dataset_estimate(differences_by_dataset))
    rng = _statistics_rng(f"randomization:{label}")
    exceedances = 0
    remaining = RANDOMIZATION_RESAMPLES
    while remaining:
        batch = min(2_000, remaining)
        randomized_dataset_means: list[np.ndarray] = []
        for dataset in COMMON_SUPPORT_DATASETS:
            values = differences_by_dataset[dataset]
            signs = rng.integers(0, 2, size=(batch, len(values)), dtype=np.int8)
            signs = signs.astype(np.float64) * 2.0 - 1.0
            randomized_dataset_means.append(np.mean(signs * values, axis=1))
        statistics = np.mean(np.stack(randomized_dataset_means, axis=1), axis=1)
        exceedances += int(np.count_nonzero(np.abs(statistics) >= observed))
        remaining -= batch
    return float((exceedances + 1) / (RANDOMIZATION_RESAMPLES + 1))


def _holm_adjust(raw_p: Sequence[float]) -> list[float]:
    values = np.asarray(raw_p, dtype=np.float64)
    if (
        values.ndim != 1
        or len(values) == 0
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise GeoAdaptV2AnalysisError("Holm adjustment received invalid p-values")
    order = np.argsort(values, kind="stable")
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        running = max(running, float((count - rank) * values[index]))
        adjusted[index] = min(1.0, running)
    return [float(value) for value in adjusted]


def _common_support_tables(
    subject_rows: Sequence[Mapping[str, Any]],
    stable_ids: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    indexed: dict[tuple[str, str, int], float] = {}
    for row in subject_rows:
        dataset = str(row["dataset"])
        if dataset not in COMMON_SUPPORT_DATASETS:
            continue
        value = _finite_float(row[PRIMARY_INFERENCE_METRIC])
        if value is None:
            raise GeoAdaptV2AnalysisError("primary common-support metric is non-finite")
        key = (str(row["stable_id"]), dataset, int(row["subject"]))
        if key in indexed:
            raise GeoAdaptV2AnalysisError("duplicate common-support subject row")
        indexed[key] = value

    summary_rows: list[dict[str, Any]] = []
    subject_sets: dict[tuple[str, str], tuple[int, ...]] = {}
    for stable_id in stable_ids:
        values_by_dataset: dict[str, np.ndarray] = {}
        for dataset in COMMON_SUPPORT_DATASETS:
            subjects = tuple(
                sorted(
                    subject
                    for (model, name, subject) in indexed
                    if model == stable_id and name == dataset
                )
            )
            expected = tuple(
                sorted(int(value) for value in DATASET_SUBJECTS_FOR_STATS[dataset])
            )
            if subjects != expected:
                raise GeoAdaptV2AnalysisError(
                    f"common-support subject coverage drift for {stable_id} x {dataset}"
                )
            subject_sets[(stable_id, dataset)] = subjects
            values_by_dataset[dataset] = np.asarray(
                [indexed[(stable_id, dataset, subject)] for subject in subjects],
                dtype=np.float64,
            )
        estimate = _equal_dataset_estimate(values_by_dataset)
        low, high = _stratified_bootstrap_interval(
            values_by_dataset,
            label=f"model:{stable_id}",
        )
        summary_rows.append(
            {
                "track": geoadapt_v2.TRACK,
                "stable_id": stable_id,
                "metric": PRIMARY_INFERENCE_METRIC,
                "n_datasets": len(COMMON_SUPPORT_DATASETS),
                "dataset_coverage": ",".join(COMMON_SUPPORT_DATASETS),
                "n_subjects": sum(len(value) for value in values_by_dataset.values()),
                "estimate": estimate,
                "bootstrap_ci95_low": low,
                "bootstrap_ci95_high": high,
                "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            }
        )

    paired_rows: list[dict[str, Any]] = []
    for left_index, stable_id_a in enumerate(stable_ids):
        for stable_id_b in stable_ids[left_index + 1 :]:
            differences: dict[str, np.ndarray] = {}
            for dataset in COMMON_SUPPORT_DATASETS:
                subjects_a = subject_sets[(stable_id_a, dataset)]
                subjects_b = subject_sets[(stable_id_b, dataset)]
                if subjects_a != subjects_b:
                    raise GeoAdaptV2AnalysisError(
                        "paired common-support subjects do not match"
                    )
                differences[dataset] = np.asarray(
                    [
                        indexed[(stable_id_a, dataset, subject)]
                        - indexed[(stable_id_b, dataset, subject)]
                        for subject in subjects_a
                    ],
                    dtype=np.float64,
                )
            label = f"pair:{stable_id_a}:{stable_id_b}"
            estimate = _equal_dataset_estimate(differences)
            low, high = _stratified_bootstrap_interval(
                differences,
                label=label,
            )
            paired_rows.append(
                {
                    "track": geoadapt_v2.TRACK,
                    "stable_id_a": stable_id_a,
                    "stable_id_b": stable_id_b,
                    "metric": PRIMARY_INFERENCE_METRIC,
                    "dataset_coverage": ",".join(COMMON_SUPPORT_DATASETS),
                    "n_paired_subjects": sum(
                        len(value) for value in differences.values()
                    ),
                    "mean_difference_a_minus_b": estimate,
                    "bootstrap_ci95_low": low,
                    "bootstrap_ci95_high": high,
                    "randomization_p_two_sided": _paired_randomization_p(
                        differences,
                        label=label,
                    ),
                    "holm_adjusted_p": None,
                    "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                    "randomization_resamples": RANDOMIZATION_RESAMPLES,
                    "multiplicity_method": (
                        "Holm family-wise correction across three predeclared "
                        "pairwise model contrasts"
                    ),
                }
            )
    adjusted = _holm_adjust(
        [float(row["randomization_p_two_sided"]) for row in paired_rows]
    )
    for row, value in zip(paired_rows, adjusted, strict=True):
        row["holm_adjusted_p"] = value
    return summary_rows, paired_rows


# Kept local to the analyzer so the inferential unit and expected coverage are
# frozen independently of presentation ordering.
DATASET_SUBJECTS_FOR_STATS: Mapping[str, tuple[int, ...]] = {
    dataset: tuple(geoadapt_v2.DATASET_SUBJECTS[dataset])
    for dataset in COMMON_SUPPORT_DATASETS
}


def _expected_input_ledger_paths(plan: Mapping[str, Any]) -> list[str]:
    paths = ["plan.json", "plan.sha256"]
    for job in geoadapt_v2.iter_jobs(plan):
        directory = geoadapt_v2._record_directory(Path(), job)
        paths.extend(
            str(directory / name) for name in sorted(geoadapt_v2.FINAL_FILENAMES)
        )
    return paths


def _capture_plan_ledger(
    run_root: Path,
    plan: Mapping[str, Any],
) -> list[dict[str, str]]:
    directory_descriptor = geoadapt_v2._open_directory_absolute(run_root)
    try:
        directory_before = os.fstat(directory_descriptor)
        path_before = geoadapt_v2._anchored_lstat(run_root)
        expected_identity = (
            directory_before.st_dev,
            directory_before.st_ino,
            directory_before.st_mode,
        )
        if (
            not stat.S_ISDIR(directory_before.st_mode)
            or (path_before.st_dev, path_before.st_ino, path_before.st_mode)
            != expected_identity
        ):
            raise GeoAdaptV2AnalysisError("run-root inode changed before plan snapshot")
        payloads = {
            name: geoadapt_v2._safe_read_regular_at(
                directory_descriptor,
                run_root,
                name,
            )
            for name in ("plan.json", "plan.sha256")
        }
        directory_after = os.fstat(directory_descriptor)
        path_after = geoadapt_v2._anchored_lstat(run_root)
        if (
            directory_after.st_dev,
            directory_after.st_ino,
            directory_after.st_mode,
        ) != expected_identity or (
            path_after.st_dev,
            path_after.st_ino,
            path_after.st_mode,
        ) != expected_identity:
            raise GeoAdaptV2AnalysisError("run-root inode changed during plan snapshot")
    finally:
        os.close(directory_descriptor)
    if payloads["plan.json"] != geoadapt_v2._canonical_bytes(plan) + b"\n" or payloads[
        "plan.sha256"
    ] != f"{plan['plan_sha256']}\n".encode("ascii"):
        raise GeoAdaptV2AnalysisError(
            "captured plan files differ from the active immutable plan"
        )
    return [
        {
            "path": "plan.json",
            "sha256": hashlib.sha256(payloads["plan.json"]).hexdigest(),
        },
        {
            "path": "plan.sha256",
            "sha256": hashlib.sha256(payloads["plan.sha256"]).hexdigest(),
        },
    ]


def _capture_completion_snapshot(
    run_root: Path,
    plan: Mapping[str, Any],
    job: geoadapt_v2.Job,
) -> CompletionSnapshot:
    """Use the runner's sole descriptor-held validation/snapshot primitive."""

    try:
        return geoadapt_v2._capture_completion_snapshot(run_root, plan, job)
    except (geoadapt_v2.GeoAdaptV2Error, OSError, ValueError) as error:
        raise GeoAdaptV2AnalysisError(
            f"completion snapshot validation failed for {job.job_id}"
        ) from error


def _verify_captured_ledger_live(
    run_root: Path,
    plan: Mapping[str, Any],
    ledger: Sequence[Mapping[str, str]],
) -> None:
    """Verify live files against captured identities without rebuilding a ledger."""

    expected_paths = _expected_input_ledger_paths(plan)
    if (
        len(ledger) != len(expected_paths)
        or [entry.get("path") for entry in ledger] != expected_paths
        or any(not geoadapt_v2._is_sha256(entry.get("sha256")) for entry in ledger)
    ):
        raise GeoAdaptV2AnalysisError(
            "captured input ledger has an invalid exact path contract"
        )
    for entry in ledger:
        live_digest = _sha256_file(run_root / entry["path"])
        if live_digest != entry["sha256"]:
            raise GeoAdaptV2AnalysisError(
                f"captured input ledger changed at {entry['path']}"
            )


def _not_applicable_rows(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, cells in plan["eligibility"].items():
        for stable_id, cell in cells.items():
            if cell["status"] == "not_applicable":
                rows.append(
                    {
                        "track": geoadapt_v2.TRACK,
                        "stable_id": stable_id,
                        "dataset": dataset,
                        "status": "not_applicable",
                        "reason": cell["reason"],
                    }
                )
    return sorted(rows, key=lambda row: (row["dataset"], row["stable_id"]))


def _subject_expected_folds(plan: Mapping[str, Any], dataset: str) -> tuple[int, ...]:
    return tuple(int(value) for value in plan["datasets"][dataset]["folds"])


def analyze_grid(
    *,
    run_root: Path,
    cache_root: Path,
    gate: geoadapt_v2.AnalysisGate,
    plan: Mapping[str, Any] | None = None,
    cache_loader: Callable[..., Mapping[str, Any]] | None = None,
) -> AnalysisResult:
    """Join outcomes only after proving exact completion and quiescence."""

    run_root = geoadapt_v2._assert_safe_path(run_root, leaf_kind="directory")
    live_plan = geoadapt_v2.load_plan(run_root)
    if plan is not None and _canonical_bytes(plan) != _canonical_bytes(live_plan):
        raise GeoAdaptV2AnalysisError("caller plan differs from immutable live plan")
    active_plan = live_plan
    geoadapt_v2.validate_production_semantics(active_plan)
    geoadapt_v2.assert_analysis_gate(gate, run_root, active_plan)
    audit = geoadapt_v2.audit_grid(run_root, active_plan)
    if (
        not audit["exact_cartesian_complete"]
        or not audit["quiescent"]
        or audit["valid_records"] != audit["expected_jobs"]
    ):
        raise GeoAdaptV2AnalysisError(
            "analysis requires the exact complete and quiescent prediction grid"
        )
    # This is deliberately below the exact audit.  Even filesystem validation
    # of the cache root occurs only after the publication gate is held and the
    # score-blind grid has proved complete and quiescent.
    cache_root = geoadapt_v2._assert_safe_path(cache_root, leaf_kind="directory")
    from .data import load_subject_cache

    load = load_subject_cache if cache_loader is None else cache_loader
    outcome_cache: dict[tuple[str, int], np.ndarray] = {}
    job_rows: list[dict[str, Any]] = []
    raw_groups: dict[
        tuple[str, str, int, int],
        list[tuple[int, np.ndarray, np.ndarray]],
    ] = defaultdict(list)
    timing_groups: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    completion_times: list[str] = []
    ledger = _capture_plan_ledger(run_root, active_plan)
    captured_jobs: list[tuple[geoadapt_v2.Job, CompletionSnapshot]] = []
    for job in geoadapt_v2.iter_jobs(active_plan):
        snapshot = _capture_completion_snapshot(run_root, active_plan, job)
        captured_jobs.append((job, snapshot))
        directory = geoadapt_v2._record_directory(run_root, job)
        ledger.extend(
            {
                "path": str((directory / name).relative_to(run_root)),
                "sha256": snapshot.artifact_sha256[name],
            }
            for name in sorted(geoadapt_v2.FINAL_FILENAMES)
        )
    geoadapt_v2.assert_analysis_gate(gate, run_root, active_plan)
    precompute_audit = geoadapt_v2.audit_grid(run_root, active_plan)
    if (
        not precompute_audit["exact_cartesian_complete"]
        or not precompute_audit["quiescent"]
        or precompute_audit["valid_records"] != precompute_audit["expected_jobs"]
    ):
        raise GeoAdaptV2AnalysisError(
            "captured analysis inputs changed before score computation"
        )
    _verify_captured_ledger_live(run_root, active_plan, ledger)

    for job, artifact in captured_jobs:
        key = (job.dataset, job.subject)
        if key not in outcome_cache:
            data = load(
                job.dataset,
                job.subject,
                cache_root=cache_root,
                montage_profile=geoadapt_v2.MONTAGE_PROFILE,
            )
            planned_cache = active_plan["cache_identity"][
                geoadapt_v2._subject_key(job.dataset, job.subject)
            ]
            if dict(data["identity"]) != dict(planned_cache):
                raise GeoAdaptV2AnalysisError(
                    f"cache identity drift for {job.dataset} S{job.subject}"
                )
            outcome_cache[key] = np.ascontiguousarray(data["y"], dtype=np.int64)
        completion_times.append(str(artifact.completion["created_at"]))
        rows = np.asarray(artifact.rows, dtype=np.int64)
        probabilities = np.asarray(artifact.probabilities, dtype=np.float64)
        truth = outcome_cache[key][rows]
        metrics = classification_metrics(truth, probabilities, n_classes=2)
        job_rows.append(
            {
                "track": geoadapt_v2.TRACK,
                "stable_id": job.stable_id,
                "dataset": job.dataset,
                "subject": job.subject,
                "fold": job.fold,
                "seed": job.seed,
                "n_trials": len(rows),
                **metrics,
            }
        )
        raw_groups[(job.stable_id, job.dataset, job.subject, job.seed)].append(
            (job.fold, rows, probabilities)
        )
        record = artifact.record
        timings = record["metadata"]["timing_seconds"]
        fit = record["metadata"]["fit"]
        timing_groups[(job.stable_id, job.dataset)].append(
            {
                "selection_fit": float(timings["selection_fit"]),
                "refit_fit": float(timings["refit_fit"]),
                "test_inference": float(timings["test_inference"]),
                "job_total": float(timings["job_total"]),
                "parameter_count": float(fit["parameter_count"]),
                "cuda_peak_memory_bytes": float(fit["cuda_peak_memory_bytes"]),
            }
        )

    subject_seed_rows: list[dict[str, Any]] = []
    for (stable_id, dataset, subject, seed), folds in sorted(raw_groups.items()):
        expected_folds = _subject_expected_folds(active_plan, dataset)
        observed_folds = tuple(sorted(fold for fold, _, _ in folds))
        if observed_folds != expected_folds:
            raise GeoAdaptV2AnalysisError(
                f"subject-seed fold coverage drift for {dataset} S{subject}"
            )
        ordered = sorted(folds, key=lambda value: value[0])
        rows = np.concatenate([value[1] for value in ordered])
        if len(set(rows.tolist())) != len(rows):
            raise GeoAdaptV2AnalysisError(
                f"held-out fold rows overlap for {dataset} S{subject}"
            )
        if len(expected_folds) > 1 and not np.array_equal(
            np.sort(rows),
            np.arange(
                len(outcome_cache[(dataset, subject)]),
                dtype=np.int64,
            ),
        ):
            raise GeoAdaptV2AnalysisError(
                f"rotating held-out folds do not cover every trial for "
                f"{dataset} S{subject}"
            )
        probabilities = np.concatenate([value[2] for value in ordered], axis=0)
        truth = outcome_cache[(dataset, subject)][rows]
        metrics = classification_metrics(truth, probabilities, n_classes=2)
        subject_seed_rows.append(
            {
                "track": geoadapt_v2.TRACK,
                "stable_id": stable_id,
                "dataset": dataset,
                "subject": subject,
                "seed": seed,
                "n_folds": len(ordered),
                "n_trials": len(rows),
                **metrics,
            }
        )

    subject_groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in subject_seed_rows:
        subject_groups[(row["stable_id"], row["dataset"], int(row["subject"]))].append(
            row
        )
    subject_rows: list[dict[str, Any]] = []
    expected_seeds = tuple(int(value) for value in active_plan["seeds"])
    for (stable_id, dataset, subject), rows in sorted(subject_groups.items()):
        observed = tuple(sorted(int(row["seed"]) for row in rows))
        if observed != tuple(sorted(expected_seeds)):
            raise GeoAdaptV2AnalysisError(
                f"seed coverage drift for {dataset} S{subject}"
            )
        n_trials = {int(row["n_trials"]) for row in rows}
        if len(n_trials) != 1:
            raise GeoAdaptV2AnalysisError("seed trial cardinality drift")
        subject_rows.append(
            {
                "track": geoadapt_v2.TRACK,
                "stable_id": stable_id,
                "dataset": dataset,
                "subject": subject,
                "n_seeds": len(rows),
                "n_trials_per_seed": next(iter(n_trials)),
                **{
                    metric: _mean([row[metric] for row in rows])
                    for metric in METRIC_NAMES
                },
            }
        )

    dataset_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in subject_rows:
        dataset_groups[(row["stable_id"], row["dataset"])].append(row)
    dataset_rows: list[dict[str, Any]] = []
    for (stable_id, dataset), rows in sorted(dataset_groups.items()):
        expected_subjects = tuple(
            int(value) for value in active_plan["datasets"][dataset]["subjects"]
        )
        observed_subjects = tuple(sorted(int(row["subject"]) for row in rows))
        if observed_subjects != tuple(sorted(expected_subjects)):
            raise GeoAdaptV2AnalysisError(
                f"subject coverage drift for {stable_id} x {dataset}"
            )
        dataset_rows.append(
            {
                "track": geoadapt_v2.TRACK,
                "stable_id": stable_id,
                "dataset": dataset,
                "n_subjects": len(rows),
                "n_seeds": len(expected_seeds),
                **{
                    metric: _mean([row[metric] for row in rows])
                    for metric in METRIC_NAMES
                },
            }
        )

    overall_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in dataset_rows:
        overall_groups[row["stable_id"]].append(row)
    overall_rows: list[dict[str, Any]] = []
    for stable_id in active_plan["stable_id_order"]:
        rows = sorted(
            overall_groups.get(stable_id, []),
            key=lambda row: row["dataset"],
        )
        expected_datasets = sorted(
            dataset
            for dataset in active_plan["dataset_order"]
            if active_plan["eligibility"][dataset][stable_id]["status"] == "eligible"
        )
        observed_datasets = [row["dataset"] for row in rows]
        if observed_datasets != expected_datasets:
            raise GeoAdaptV2AnalysisError(f"dataset coverage drift for {stable_id}")
        overall_rows.append(
            {
                "track": geoadapt_v2.TRACK,
                "stable_id": stable_id,
                "n_datasets": len(rows),
                "dataset_coverage": ",".join(observed_datasets),
                **{
                    metric: _mean([row[metric] for row in rows])
                    for metric in METRIC_NAMES
                },
            }
        )

    timing_rows: list[dict[str, Any]] = []
    for (stable_id, dataset), rows in sorted(timing_groups.items()):
        timing_rows.append(
            {
                "track": geoadapt_v2.TRACK,
                "stable_id": stable_id,
                "dataset": dataset,
                "n_jobs": len(rows),
                "mean_selection_fit_seconds": _mean(
                    [row["selection_fit"] for row in rows]
                ),
                "mean_refit_fit_seconds": _mean([row["refit_fit"] for row in rows]),
                "mean_test_inference_seconds": _mean(
                    [row["test_inference"] for row in rows]
                ),
                "mean_job_total_seconds": _mean([row["job_total"] for row in rows]),
                "mean_parameter_count": _mean([row["parameter_count"] for row in rows]),
                "mean_cuda_peak_memory_bytes": _mean(
                    [row["cuda_peak_memory_bytes"] for row in rows]
                ),
            }
        )

    common_support_rows, paired_common_support_rows = _common_support_tables(
        subject_rows,
        active_plan["stable_id_order"],
    )
    tables = {
        "job_metrics": sorted(
            job_rows,
            key=lambda row: (
                row["dataset"],
                row["stable_id"],
                row["subject"],
                row["fold"],
                row["seed"],
            ),
        ),
        "subject_seed_metrics": sorted(
            subject_seed_rows,
            key=lambda row: (
                row["dataset"],
                row["stable_id"],
                row["subject"],
                row["seed"],
            ),
        ),
        "subject_metrics": sorted(
            subject_rows,
            key=lambda row: (
                row["dataset"],
                row["stable_id"],
                row["subject"],
            ),
        ),
        "dataset_summary": dataset_rows,
        "overall_summary": overall_rows,
        "timing_summary": timing_rows,
        "not_applicable": _not_applicable_rows(active_plan),
        "common_support_summary": common_support_rows,
        "paired_common_support": paired_common_support_rows,
    }
    for name, rows in tables.items():
        expected = set(TABLE_FIELDS[name])
        if any(set(row) != expected for row in rows):
            raise GeoAdaptV2AnalysisError(f"{name} row schema is invalid")
    summary = {
        "schema": ANALYSIS_SCHEMA,
        "created_at": max(completion_times),
        "plan_sha256": active_plan["plan_sha256"],
        "track": geoadapt_v2.TRACK,
        "evidence_scope": geoadapt_v2.EVIDENCE_SCOPE,
        "confirmation_evidence": False,
        "common_recipe_result": False,
        "score_joining": {
            "input": "immutable score-blind held-out row/probability artifacts",
            "outcomes": "plan-bound harmonized-v2 cache outcomes joined in memory",
            "trial_outcomes_published": False,
            "probability_matrices_published": False,
        },
        "aggregation": (
            "concatenate folds within subject-seed; mean seeds within subject; "
            "mean subjects within dataset; equal-weight eligible datasets"
        ),
        "primary_metric": "equal_dataset_macro_balanced_accuracy",
        "common_support_inference": {
            "datasets": list(COMMON_SUPPORT_DATASETS),
            "metric": PRIMARY_INFERENCE_METRIC,
            "estimand": (
                "equal-dataset mean of seed-averaged subject balanced accuracy"
            ),
            "uncertainty": (
                "deterministic dataset-stratified paired subject bootstrap "
                f"percentile 95% CI ({BOOTSTRAP_RESAMPLES} resamples)"
            ),
            "hypothesis_test": (
                "two-sided paired subject sign-randomization on the "
                f"equal-dataset contrast ({RANDOMIZATION_RESAMPLES} draws)"
            ),
            "multiplicity": (
                "Holm family-wise correction across the three predeclared "
                "pairwise GeoAdapt-family contrasts"
            ),
            "inference_unit": "subject after averaging five seeds",
            "statistical_seed": STATISTICAL_SEED,
            "secondary_metrics": "descriptive only; no unregistered p-values",
        },
        "ece_bins": 15,
        "audit": {
            key: audit[key]
            for key in (
                "expected_jobs",
                "valid_records",
                "missing_records",
                "corrupt_records",
                "unknown_records",
                "live_claims",
                "stale_claims",
                "unknown_claims",
                "partials",
                "quiescent",
                "exact_cartesian_complete",
            )
        },
        "table_row_counts": {name: len(rows) for name, rows in tables.items()},
        "stable_ids": list(active_plan["stable_id_order"]),
        "dataset_order": list(active_plan["dataset_order"]),
        "not_applicable_cells": len(tables["not_applicable"]),
    }
    geoadapt_v2.assert_analysis_gate(gate, run_root, active_plan)
    postcompute_audit = geoadapt_v2.audit_grid(run_root, active_plan)
    if (
        not postcompute_audit["exact_cartesian_complete"]
        or not postcompute_audit["quiescent"]
        or postcompute_audit["valid_records"] != postcompute_audit["expected_jobs"]
    ):
        raise GeoAdaptV2AnalysisError(
            "captured analysis inputs changed during score computation"
        )
    _verify_captured_ledger_live(run_root, active_plan, ledger)
    return AnalysisResult(summary=summary, tables=tables, input_ledger=ledger)


def _csv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(fields),
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {field: "" if row[field] is None else row[field] for field in fields}
        )
    return buffer.getvalue().encode("utf-8")


def _expected_table_cardinalities(plan: Mapping[str, Any]) -> dict[str, int]:
    seeds = len(plan["seeds"])
    subject_cells = 0
    eligible_cells = 0
    for dataset in plan["dataset_order"]:
        subjects = len(plan["datasets"][dataset]["subjects"])
        for stable_id in plan["stable_id_order"]:
            if plan["eligibility"][dataset][stable_id]["status"] == "eligible":
                subject_cells += subjects
                eligible_cells += 1
    not_applicable = sum(
        cell["status"] == "not_applicable"
        for cells in plan["eligibility"].values()
        for cell in cells.values()
    )
    common_ready = all(
        dataset in plan["datasets"]
        and all(
            plan["eligibility"][dataset][stable_id]["status"] == "eligible"
            for stable_id in plan["stable_id_order"]
        )
        for dataset in COMMON_SUPPORT_DATASETS
    )
    n_models = len(plan["stable_id_order"]) if common_ready else 0
    return {
        "job_metrics": int(plan["n_jobs"]),
        "subject_seed_metrics": subject_cells * seeds,
        "subject_metrics": subject_cells,
        "dataset_summary": eligible_cells,
        "overall_summary": len(plan["stable_id_order"]),
        "timing_summary": eligible_cells,
        "not_applicable": int(not_applicable),
        "common_support_summary": n_models,
        "paired_common_support": n_models * (n_models - 1) // 2,
    }


def _validate_table_value(field_name: str, value: Any) -> None:
    string_fields = {
        "track",
        "stable_id",
        "stable_id_a",
        "stable_id_b",
        "dataset",
        "dataset_coverage",
        "status",
        "reason",
        "metric",
        "multiplicity_method",
    }
    integer_fields = {
        "subject",
        "fold",
        "seed",
        "n_trials",
        "n_folds",
        "n_seeds",
        "n_trials_per_seed",
        "n_subjects",
        "n_datasets",
        "n_jobs",
        "n_paired_subjects",
        "bootstrap_resamples",
        "randomization_resamples",
    }
    if field_name in string_fields:
        if not isinstance(value, str) or not value:
            raise GeoAdaptV2AnalysisError(
                f"analysis table field {field_name} must be a nonempty string"
            )
        return
    if field_name in integer_fields:
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < (0 if field_name in {"fold", "seed"} else 1)
        ):
            raise GeoAdaptV2AnalysisError(
                f"analysis table field {field_name} must be an exact integer"
            )
        return
    if value is not None and (not isinstance(value, float) or not math.isfinite(value)):
        raise GeoAdaptV2AnalysisError(
            f"analysis table field {field_name} must be finite or null"
        )


def _validate_analysis_result_exact(
    result: AnalysisResult,
    *,
    plan: Mapping[str, Any],
) -> None:
    if not isinstance(result, AnalysisResult):
        raise GeoAdaptV2AnalysisError("analysis result has the wrong type")
    if not isinstance(result.tables, dict) or set(result.tables) != set(TABLE_FIELDS):
        raise GeoAdaptV2AnalysisError("analysis tables have an invalid exact schema")
    expected_counts = _expected_table_cardinalities(plan)
    primary_fields = {
        "job_metrics": ("stable_id", "dataset", "subject", "fold", "seed"),
        "subject_seed_metrics": ("stable_id", "dataset", "subject", "seed"),
        "subject_metrics": ("stable_id", "dataset", "subject"),
        "dataset_summary": ("stable_id", "dataset"),
        "overall_summary": ("stable_id",),
        "timing_summary": ("stable_id", "dataset"),
        "not_applicable": ("stable_id", "dataset"),
        "common_support_summary": ("stable_id",),
        "paired_common_support": ("stable_id_a", "stable_id_b"),
    }
    for name, fields in TABLE_FIELDS.items():
        rows = result.tables[name]
        if not isinstance(rows, list) or len(rows) != expected_counts[name]:
            raise GeoAdaptV2AnalysisError(
                f"{name} cardinality differs from the immutable plan"
            )
        keys: set[tuple[Any, ...]] = set()
        for row in rows:
            if not isinstance(row, dict) or tuple(row) != fields:
                raise GeoAdaptV2AnalysisError(
                    f"{name} row has an invalid exact ordered schema"
                )
            for field_name in fields:
                _validate_table_value(field_name, row[field_name])
            if row["track"] != geoadapt_v2.TRACK:
                raise GeoAdaptV2AnalysisError(f"{name} track identity drifted")
            key = tuple(row[field] for field in primary_fields[name])
            if key in keys:
                raise GeoAdaptV2AnalysisError(f"{name} contains duplicate rows")
            keys.add(key)
    if result.tables["not_applicable"] != _not_applicable_rows(plan):
        raise GeoAdaptV2AnalysisError("N/A table differs from the frozen matrix")

    summary = result.summary
    summary_fields = {
        "schema",
        "created_at",
        "plan_sha256",
        "track",
        "evidence_scope",
        "confirmation_evidence",
        "common_recipe_result",
        "score_joining",
        "aggregation",
        "primary_metric",
        "common_support_inference",
        "ece_bins",
        "audit",
        "table_row_counts",
        "stable_ids",
        "dataset_order",
        "not_applicable_cells",
    }
    if (
        not isinstance(summary, dict)
        or set(summary) != summary_fields
        or summary["schema"] != ANALYSIS_SCHEMA
        or not geoadapt_v2._is_utc_timestamp(summary["created_at"])
        or summary["plan_sha256"] != plan["plan_sha256"]
        or summary["track"] != geoadapt_v2.TRACK
        or summary["evidence_scope"] != geoadapt_v2.EVIDENCE_SCOPE
        or summary["confirmation_evidence"] is not False
        or summary["common_recipe_result"] is not False
        or summary["primary_metric"] != "equal_dataset_macro_balanced_accuracy"
        or summary["ece_bins"] != 15
        or not isinstance(summary["ece_bins"], int)
        or isinstance(summary["ece_bins"], bool)
        or not isinstance(summary["aggregation"], str)
        or not summary["aggregation"]
        or summary["stable_ids"] != list(plan["stable_id_order"])
        or summary["dataset_order"] != list(plan["dataset_order"])
        or summary["not_applicable_cells"] != expected_counts["not_applicable"]
        or not isinstance(summary["not_applicable_cells"], int)
        or isinstance(summary["not_applicable_cells"], bool)
        or not isinstance(summary["table_row_counts"], dict)
        or set(summary["table_row_counts"]) != set(TABLE_FIELDS)
        or any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in summary["table_row_counts"].values()
        )
        or summary["table_row_counts"] != expected_counts
    ):
        raise GeoAdaptV2AnalysisError("analysis summary has an invalid exact schema")
    joining = summary["score_joining"]
    if (
        not isinstance(joining, dict)
        or set(joining)
        != {
            "input",
            "outcomes",
            "trial_outcomes_published",
            "probability_matrices_published",
        }
        or joining["trial_outcomes_published"] is not False
        or joining["probability_matrices_published"] is not False
        or any(
            not isinstance(joining[key], str) or not joining[key]
            for key in ("input", "outcomes")
        )
    ):
        raise GeoAdaptV2AnalysisError("score-joining summary schema is invalid")
    inference = summary["common_support_inference"]
    if (
        not isinstance(inference, dict)
        or set(inference)
        != {
            "datasets",
            "metric",
            "estimand",
            "uncertainty",
            "hypothesis_test",
            "multiplicity",
            "inference_unit",
            "statistical_seed",
            "secondary_metrics",
        }
        or inference["datasets"] != list(COMMON_SUPPORT_DATASETS)
        or inference["metric"] != PRIMARY_INFERENCE_METRIC
        or inference["statistical_seed"] != STATISTICAL_SEED
        or isinstance(inference["statistical_seed"], bool)
        or any(
            not isinstance(inference[key], str) or not inference[key]
            for key in (
                "estimand",
                "uncertainty",
                "hypothesis_test",
                "multiplicity",
                "inference_unit",
                "secondary_metrics",
            )
        )
    ):
        raise GeoAdaptV2AnalysisError("common-support summary schema is invalid")
    audit = summary["audit"]
    audit_fields = {
        "expected_jobs",
        "valid_records",
        "missing_records",
        "corrupt_records",
        "unknown_records",
        "live_claims",
        "stale_claims",
        "unknown_claims",
        "partials",
        "quiescent",
        "exact_cartesian_complete",
    }
    if (
        not isinstance(audit, dict)
        or set(audit) != audit_fields
        or any(
            not isinstance(audit[key], int) or isinstance(audit[key], bool)
            for key in audit_fields - {"quiescent", "exact_cartesian_complete"}
        )
        or audit["expected_jobs"] != int(plan["n_jobs"])
        or audit["valid_records"] != int(plan["n_jobs"])
        or any(
            audit[key] != 0
            for key in (
                "missing_records",
                "corrupt_records",
                "unknown_records",
                "live_claims",
                "stale_claims",
                "unknown_claims",
                "partials",
            )
        )
        or audit["quiescent"] is not True
        or audit["exact_cartesian_complete"] is not True
    ):
        raise GeoAdaptV2AnalysisError("analysis audit summary is invalid")
    if not isinstance(result.input_ledger, list) or len(
        result.input_ledger
    ) != 2 + 3 * int(plan["n_jobs"]):
        raise GeoAdaptV2AnalysisError("analysis input-ledger cardinality is invalid")
    paths: set[str] = set()
    for entry in result.input_ledger:
        if (
            not isinstance(entry, dict)
            or set(entry) != {"path", "sha256"}
            or not isinstance(entry["path"], str)
            or not entry["path"]
            or Path(entry["path"]).is_absolute()
            or ".." in Path(entry["path"]).parts
            or not geoadapt_v2._is_sha256(entry["sha256"])
            or entry["path"] in paths
        ):
            raise GeoAdaptV2AnalysisError("analysis input ledger is invalid")
        paths.add(entry["path"])
    if [entry["path"] for entry in result.input_ledger] != (
        _expected_input_ledger_paths(plan)
    ):
        raise GeoAdaptV2AnalysisError("analysis input ledger path order is invalid")


def _write_file(path: Path, payload: bytes) -> None:
    geoadapt_v2._write_bytes_exclusive(path, payload)


def _ensure_separate_output(run_root: Path, output_root: Path) -> None:
    run = geoadapt_v2._assert_safe_path(run_root, leaf_kind="directory")
    output = geoadapt_v2._absolute_path(output_root)
    geoadapt_v2._assert_safe_path(output, leaf_kind="directory", allow_missing=True)
    if output == run or output.is_relative_to(run):
        raise GeoAdaptV2AnalysisError(
            "analysis output must be a sibling/outside the immutable run root"
        )


def _publication_lock_path(output_root: Path) -> Path:
    return output_root.parent / f".{output_root.name}.publish.lock"


def _validate_publication_lock_payload(
    value: Any,
    *,
    output_root: Path,
    plan: Mapping[str, Any],
) -> None:
    if (
        not isinstance(value, Mapping)
        or set(value)
        != {
            "schema",
            "created_at",
            "plan_sha256",
            "output_root",
            "nonce",
            "owner",
        }
        or value["schema"] != "eeg-mi-geoadapt-v2-analysis-lock-v2"
        or not geoadapt_v2._is_utc_timestamp(value["created_at"])
        or value["plan_sha256"] != plan["plan_sha256"]
        or value["output_root"] != str(output_root)
        or not geoadapt_v2._is_uuid4_hex(value["nonce"])
    ):
        raise GeoAdaptV2AnalysisError(
            "publication lock has an invalid exact schema or identity"
        )
    geoadapt_v2._validate_owner_mapping(value["owner"])


def _read_expected_publication_lock(
    path: Path,
    *,
    expected_identity: tuple[int, int],
) -> Mapping[str, Any]:
    """Read only the exact lock inode selected by recovery enumeration."""

    descriptor = geoadapt_v2._open_existing_regular(path)
    try:
        before = os.fstat(descriptor)
        if (int(before.st_dev), int(before.st_ino)) != expected_identity:
            raise PublicationContention(
                "publication lock inode changed before bound read"
            )
        fingerprint = geoadapt_v2._completion_stat_fingerprint(before)
        chunks: list[bytes] = []
        while True:
            try:
                chunk = os.read(descriptor, 1024 * 1024)
            except InterruptedError:
                continue
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        try:
            live = geoadapt_v2._anchored_lstat(path)
        except FileNotFoundError as error:
            raise PublicationContention(
                "publication lock disappeared during bound read"
            ) from error
        if (
            geoadapt_v2._completion_stat_fingerprint(after) != fingerprint
            or geoadapt_v2._completion_stat_fingerprint(live) != fingerprint
        ):
            raise PublicationContention(
                "publication lock inode changed during bound read"
            )
        payload = b"".join(chunks)
        if len(payload) != int(before.st_size):
            raise GeoAdaptV2AnalysisError(
                "publication lock bound read was incomplete"
            )
    finally:
        os.close(descriptor)
    observed = geoadapt_v2._strict_json_bytes(payload, source=str(path))
    if payload != geoadapt_v2._canonical_bytes(observed) + b"\n":
        raise GeoAdaptV2AnalysisError("publication lock JSON is not canonical")
    return observed


def _quarantine_publication_residue(
    path: Path,
    *,
    reason: str,
    expected_dev: int,
    expected_ino: int,
) -> Path:
    path = geoadapt_v2._assert_safe_path(path)
    source_identity = geoadapt_v2._anchored_lstat(path)
    if int(source_identity.st_dev) != expected_dev or int(
        source_identity.st_ino
    ) != expected_ino:
        raise PublicationContention(
            "publication residue inode changed before quarantine"
        )
    root = path.parent / ".geoadapt-v2-publication-quarantine"
    geoadapt_v2._safe_mkdir(root)
    destination = root / (f"{path.name}.{reason}.{time.time_ns()}.{uuid.uuid4().hex}")
    source_pair = (int(source_identity.st_dev), int(source_identity.st_ino))
    try:
        geoadapt_v2._rename_noreplace(
            path,
            destination,
            expected_source_dev=source_pair[0],
            expected_source_ino=source_pair[1],
        )
    except BaseException:
        if not geoadapt_v2._path_binds_identity(destination, source_pair):
            raise
    if not geoadapt_v2._path_binds_identity(destination, source_pair):
        raise GeoAdaptV2AnalysisError(
            "publication quarantine destination changed after move"
        )
    geoadapt_v2._fsync_directory(path.parent)
    geoadapt_v2._fsync_directory(root)
    return destination


def _acquire_publication_lock(
    output_root: Path,
    plan: Mapping[str, Any],
) -> PublicationLock:
    lock = _publication_lock_path(output_root)
    lock_root = output_root.parent / ".geoadapt-v2-authority-locks"
    with geoadapt_v2._authority_publication_lock(
        lock,
        lock_root=lock_root,
    ):
        for staged, staged_identity in geoadapt_v2._staged_publication_entries(lock):
            _quarantine_publication_residue(
                staged,
                reason="powercut",
                expected_dev=staged_identity[0],
                expected_ino=staged_identity[1],
            )
        if geoadapt_v2._path_exists(lock):
            lock_status = geoadapt_v2._anchored_lstat(lock)
            lock_identity = (
                int(lock_status.st_dev),
                int(lock_status.st_ino),
            )
            try:
                observed = _read_expected_publication_lock(
                    lock,
                    expected_identity=lock_identity,
                )
                _validate_publication_lock_payload(
                    observed,
                    output_root=output_root,
                    plan=plan,
                )
            except PublicationContention:
                raise
            except (
                GeoAdaptV2AnalysisError,
                geoadapt_v2.GeoAdaptV2Error,
                OSError,
                ValueError,
            ):
                _quarantine_publication_residue(
                    lock,
                    reason="malformed",
                    expected_dev=int(lock_status.st_dev),
                    expected_ino=int(lock_status.st_ino),
                )
            else:
                state = geoadapt_v2._owner_state(observed["owner"])
                if state != "dead_local":
                    raise GeoAdaptV2AnalysisError(f"publication lock owner is {state}")
                if not geoadapt_v2._path_binds_identity(lock, lock_identity):
                    raise PublicationContention(
                        "publication lock changed before residue recovery"
                    )
                nonce = str(observed.get("nonce", ""))
                partial = output_root.parent / (f".{output_root.name}.{nonce}.partial")
                if nonce and geoadapt_v2._path_exists(partial):
                    partial_status = geoadapt_v2._anchored_lstat(partial)
                    if not geoadapt_v2._path_binds_identity(lock, lock_identity):
                        raise PublicationContention(
                            "publication lock changed before partial recovery"
                        )
                    _quarantine_publication_residue(
                        partial,
                        reason="powercut",
                        expected_dev=int(partial_status.st_dev),
                        expected_ino=int(partial_status.st_ino),
                    )
                _quarantine_publication_residue(
                    lock,
                    reason="powercut",
                    expected_dev=lock_identity[0],
                    expected_ino=lock_identity[1],
                )
        nonce = uuid.uuid4().hex
        owner = geoadapt_v2._process_identity()
        geoadapt_v2._write_json_exclusive(
            lock,
            {
                "schema": "eeg-mi-geoadapt-v2-analysis-lock-v2",
                "created_at": geoadapt_v2._utc_now(),
                "plan_sha256": plan["plan_sha256"],
                "output_root": str(output_root),
                "nonce": nonce,
                "owner": owner,
            },
        )
        identity = geoadapt_v2._anchored_lstat(lock)
        return PublicationLock(
            path=lock,
            nonce=nonce,
            owner=owner,
            st_dev=int(identity.st_dev),
            st_ino=int(identity.st_ino),
        )


def _assert_publication_lock(
    lock: PublicationLock,
    *,
    output_root: Path,
    plan: Mapping[str, Any],
) -> None:
    observed = geoadapt_v2.strict_load(lock.path)
    identity = geoadapt_v2._anchored_lstat(lock.path)
    _validate_publication_lock_payload(
        observed,
        output_root=output_root,
        plan=plan,
    )
    if (
        observed["nonce"] != lock.nonce
        or observed["owner"] != dict(lock.owner)
        or (identity.st_dev, identity.st_ino) != (lock.st_dev, lock.st_ino)
    ):
        raise GeoAdaptV2AnalysisError("publication lock ownership changed")


def _release_publication_lock(
    lock: PublicationLock,
) -> None:
    observed = geoadapt_v2.strict_load(lock.path)
    identity = geoadapt_v2._anchored_lstat(lock.path)
    if (
        set(observed)
        != {
            "schema",
            "created_at",
            "plan_sha256",
            "output_root",
            "nonce",
            "owner",
        }
        or observed["schema"] != "eeg-mi-geoadapt-v2-analysis-lock-v2"
        or not geoadapt_v2._is_utc_timestamp(observed["created_at"])
        or not geoadapt_v2._is_sha256(observed["plan_sha256"])
        or not isinstance(observed["output_root"], str)
        or not observed["output_root"]
        or not geoadapt_v2._is_uuid4_hex(observed["nonce"])
    ):
        raise GeoAdaptV2AnalysisError("publication lock has an invalid exact schema")
    geoadapt_v2._validate_owner_mapping(observed["owner"])
    if (
        observed["nonce"] != lock.nonce
        or observed["owner"] != dict(lock.owner)
        or (identity.st_dev, identity.st_ino) != (lock.st_dev, lock.st_ino)
    ):
        raise GeoAdaptV2AnalysisError("refusing to release another publication lock")
    geoadapt_v2._unlink_owned_regular(
        lock.path,
        expected_dev=lock.st_dev,
        expected_ino=lock.st_ino,
    )


def _validate_existing_publication(
    output_root: Path,
    *,
    result: AnalysisResult,
    plan: Mapping[str, Any],
) -> None:
    _validate_analysis_result_exact(result, plan=plan)
    geoadapt_v2._assert_safe_path(output_root, leaf_kind="directory")
    expected_names = {
        "summary.json",
        "input_ledger.json",
        "manifest.json",
        *(f"{name}.csv" for name in TABLE_FIELDS),
    }
    entries = geoadapt_v2._anchored_directory_entries(output_root)
    if set(entries) != expected_names:
        raise GeoAdaptV2AnalysisError(
            "existing publication has an invalid exact file set"
        )
    for name, entry_status in entries.items():
        geoadapt_v2._require_single_link(entry_status, output_root / name)
        if not stat.S_ISREG(entry_status.st_mode):
            raise GeoAdaptV2AnalysisError(
                "existing publication has an invalid exact file set"
            )
    expected_payloads = {
        "summary.json": _canonical_bytes(result.summary) + b"\n",
        "input_ledger.json": _canonical_bytes(result.input_ledger) + b"\n",
        **{
            f"{name}.csv": _csv_bytes(rows, TABLE_FIELDS[name])
            for name, rows in result.tables.items()
        },
    }
    for name, payload in expected_payloads.items():
        if geoadapt_v2._safe_read_bytes(output_root / name) != payload:
            raise GeoAdaptV2AnalysisError(
                f"existing publication differs from freshly recomputed result: {name}"
            )
    manifest_bytes = geoadapt_v2._safe_read_bytes(output_root / "manifest.json")
    manifest = geoadapt_v2._strict_json_bytes(
        manifest_bytes,
        source=str(output_root / "manifest.json"),
    )
    expected_hashes = {
        name: hashlib.sha256(payload).hexdigest()
        for name, payload in expected_payloads.items()
    }
    if (
        manifest_bytes != _canonical_bytes(manifest) + b"\n"
        or set(manifest)
        != {
            "schema",
            "created_at",
            "plan_sha256",
            "track",
            "files",
            "input_ledger_sha256",
            "input_file_count",
        }
        or manifest["schema"] != MANIFEST_SCHEMA
        or manifest["created_at"] != result.summary["created_at"]
        or not geoadapt_v2._is_utc_timestamp(manifest["created_at"])
        or manifest["plan_sha256"] != plan["plan_sha256"]
        or manifest["track"] != geoadapt_v2.TRACK
        or not isinstance(manifest["files"], dict)
        or manifest["files"] != expected_hashes
        or not all(
            geoadapt_v2._is_sha256(value) for value in manifest["files"].values()
        )
        or manifest["input_ledger_sha256"]
        != hashlib.sha256(_canonical_bytes(result.input_ledger)).hexdigest()
        or not geoadapt_v2._is_sha256(manifest["input_ledger_sha256"])
        or manifest["input_file_count"] != len(result.input_ledger)
        or not isinstance(manifest["input_file_count"], int)
        or isinstance(manifest["input_file_count"], bool)
    ):
        raise GeoAdaptV2AnalysisError("existing publication manifest is invalid")


def _rebind_analysis_commit_authority(
    *,
    run_root: Path,
    cache_root: Path,
    output_root: Path,
    gate: geoadapt_v2.AnalysisGate,
    lock: PublicationLock,
    plan: Mapping[str, Any],
    result: AnalysisResult,
) -> Mapping[str, Any]:
    """Rebind all analysis inputs and held authorities at the commit edge."""

    live_plan = geoadapt_v2._rebind_authoritative_inputs(
        run_root=run_root,
        plan=plan,
        cache_root=cache_root,
    )
    geoadapt_v2.assert_analysis_gate(gate, run_root, live_plan)
    _assert_publication_lock(
        lock,
        output_root=output_root,
        plan=live_plan,
    )
    audit = geoadapt_v2.audit_grid(run_root, live_plan)
    if (
        not audit["exact_cartesian_complete"]
        or not audit["quiescent"]
        or audit["valid_records"] != audit["expected_jobs"]
    ):
        raise GeoAdaptV2AnalysisError(
            "analysis authority changed at publication boundary"
        )
    _verify_captured_ledger_live(run_root, live_plan, result.input_ledger)
    _validate_analysis_result_exact(result, plan=live_plan)
    return live_plan


def publish_analysis(
    *,
    run_root: Path,
    cache_root: Path,
    output_root: Path,
    gate: geoadapt_v2.AnalysisGate,
) -> Path:
    """Freshly recompute and atomically publish from live immutable inputs.

    This boundary accepts neither a caller-created result nor an injected cache
    loader. All validation and score joining occurs while both the worker gate
    and the publication lock are held.
    """

    run_root = geoadapt_v2._assert_safe_path(run_root, leaf_kind="directory")
    cache_root = geoadapt_v2._assert_safe_path(cache_root, leaf_kind="directory")
    output_root = geoadapt_v2._absolute_path(output_root)
    _ensure_separate_output(run_root, output_root)
    active_plan = geoadapt_v2.load_plan(run_root)
    geoadapt_v2.validate_production_semantics(active_plan)
    geoadapt_v2.assert_analysis_gate(gate, run_root, active_plan)

    geoadapt_v2._safe_mkdir(output_root.parent)
    lock = _acquire_publication_lock(output_root, active_plan)
    partial = output_root.parent / f".{output_root.name}.{lock.nonce}.partial"
    published_identity: tuple[int, int] | None = None
    partial_identity: os.stat_result | None = None
    try:
        # Do not trust the plan object used before lock acquisition. Reload and
        # validate every live identity under both synchronization boundaries.
        locked_plan = geoadapt_v2.load_plan(run_root)
        if _canonical_bytes(locked_plan) != _canonical_bytes(active_plan):
            raise GeoAdaptV2AnalysisError(
                "immutable plan changed during publication lock acquisition"
            )
        active_plan = locked_plan
        geoadapt_v2.validate_production_semantics(active_plan)
        geoadapt_v2.assert_analysis_gate(gate, run_root, active_plan)
        _assert_publication_lock(
            lock,
            output_root=output_root,
            plan=active_plan,
        )
        geoadapt_v2.verify_static_identity(active_plan)
        audit = geoadapt_v2.audit_grid(run_root, active_plan)
        if (
            not audit["exact_cartesian_complete"]
            or not audit["quiescent"]
            or audit["valid_records"] != audit["expected_jobs"]
        ):
            raise GeoAdaptV2AnalysisError(
                "publication requires an exact complete and quiescent grid"
            )
        geoadapt_v2.verify_cache_and_splits(active_plan, cache_root=cache_root)
        result = analyze_grid(
            run_root=run_root,
            cache_root=cache_root,
            gate=gate,
            plan=active_plan,
            cache_loader=None,
        )
        _validate_analysis_result_exact(result, plan=active_plan)

        # Fresh final checks bind every byte used by the aggregate to the live
        # source, registry, UV environment, cache, split, and prediction ledger.
        final_plan = geoadapt_v2.load_plan(run_root)
        if _canonical_bytes(final_plan) != _canonical_bytes(active_plan):
            raise GeoAdaptV2AnalysisError("live plan changed during score joining")
        geoadapt_v2.verify_static_identity(final_plan)
        geoadapt_v2.verify_cache_and_splits(final_plan, cache_root=cache_root)
        geoadapt_v2.assert_analysis_gate(gate, run_root, final_plan)
        _assert_publication_lock(
            lock,
            output_root=output_root,
            plan=final_plan,
        )
        final_audit = geoadapt_v2.audit_grid(run_root, final_plan)
        if not final_audit["exact_cartesian_complete"] or not final_audit["quiescent"]:
            raise GeoAdaptV2AnalysisError(
                "immutable publication inputs changed during score joining"
            )
        _verify_captured_ledger_live(
            run_root,
            final_plan,
            result.input_ledger,
        )
        _validate_analysis_result_exact(result, plan=final_plan)
        if geoadapt_v2._path_exists(output_root):
            _validate_existing_publication(
                output_root,
                result=result,
                plan=final_plan,
            )
            geoadapt_v2.assert_analysis_gate(gate, run_root, final_plan)
            _assert_publication_lock(
                lock,
                output_root=output_root,
                plan=final_plan,
            )
            _verify_captured_ledger_live(
                run_root,
                final_plan,
                result.input_ledger,
            )
            return output_root
        geoadapt_v2._safe_mkdir(partial, parents=False)
        partial_identity = geoadapt_v2._anchored_lstat(partial)
        files: dict[str, str] = {}
        summary_payload = _canonical_bytes(result.summary) + b"\n"
        try:
            geoadapt_v2._require_disk_floor(
                output_root.parent,
                plan=final_plan,
                phase="publication summary write",
            )
        except geoadapt_v2.DiskUnavailable as error:
            raise GeoAdaptV2AnalysisError(str(error)) from error
        _write_file(partial / "summary.json", summary_payload)
        files["summary.json"] = _sha256_file(partial / "summary.json")
        ledger_payload = _canonical_bytes(result.input_ledger) + b"\n"
        try:
            geoadapt_v2._require_disk_floor(
                output_root.parent,
                plan=final_plan,
                phase="publication input-ledger write",
            )
        except geoadapt_v2.DiskUnavailable as error:
            raise GeoAdaptV2AnalysisError(str(error)) from error
        _write_file(partial / "input_ledger.json", ledger_payload)
        files["input_ledger.json"] = _sha256_file(partial / "input_ledger.json")
        for name, rows in result.tables.items():
            filename = f"{name}.csv"
            try:
                geoadapt_v2._require_disk_floor(
                    output_root.parent,
                    plan=final_plan,
                    phase=f"publication {filename} write",
                )
            except geoadapt_v2.DiskUnavailable as error:
                raise GeoAdaptV2AnalysisError(str(error)) from error
            _write_file(
                partial / filename,
                _csv_bytes(rows, TABLE_FIELDS[name]),
            )
            files[filename] = _sha256_file(partial / filename)
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "created_at": result.summary["created_at"],
            "plan_sha256": final_plan["plan_sha256"],
            "track": geoadapt_v2.TRACK,
            "files": files,
            "input_ledger_sha256": hashlib.sha256(
                _canonical_bytes(result.input_ledger)
            ).hexdigest(),
            "input_file_count": len(result.input_ledger),
        }
        try:
            geoadapt_v2._require_disk_floor(
                output_root.parent,
                plan=final_plan,
                phase="publication manifest write",
            )
        except geoadapt_v2.DiskUnavailable as error:
            raise GeoAdaptV2AnalysisError(str(error)) from error
        _write_file(partial / "manifest.json", _canonical_bytes(manifest) + b"\n")
        geoadapt_v2._fsync_directory(partial)
        _validate_existing_publication(
            partial,
            result=result,
            plan=final_plan,
        )
        # The final mutable action is permitted only if the gate and exact
        # prediction ledger still match the just-computed publication.
        geoadapt_v2.assert_analysis_gate(gate, run_root, final_plan)
        _assert_publication_lock(
            lock,
            output_root=output_root,
            plan=final_plan,
        )
        precommit_audit = geoadapt_v2.audit_grid(run_root, final_plan)
        if (
            not precommit_audit["exact_cartesian_complete"]
            or not precommit_audit["quiescent"]
        ):
            raise GeoAdaptV2AnalysisError(
                "publication inputs changed before atomic rename"
            )
        _verify_captured_ledger_live(
            run_root,
            final_plan,
            result.input_ledger,
        )
        try:
            geoadapt_v2._require_disk_floor(
                output_root.parent,
                plan=final_plan,
                phase="publication commit",
            )
        except geoadapt_v2.DiskUnavailable as error:
            raise GeoAdaptV2AnalysisError(str(error)) from error
        staged_identity = (
            int(partial_identity.st_dev),
            int(partial_identity.st_ino),
        )
        _rebind_analysis_commit_authority(
            run_root=run_root,
            cache_root=cache_root,
            output_root=output_root,
            gate=gate,
            lock=lock,
            plan=final_plan,
            result=result,
        )
        try:
            geoadapt_v2._rename_noreplace(
                partial,
                output_root,
                expected_source_dev=staged_identity[0],
                expected_source_ino=staged_identity[1],
            )
        except BaseException:
            if geoadapt_v2._path_binds_identity(output_root, staged_identity):
                # The helper may move successfully and then raise. Publication
                # identity is determined from the canonical binding, never
                # from whether the helper returned.
                published_identity = staged_identity
                _rebind_analysis_commit_authority(
                    run_root=run_root,
                    cache_root=cache_root,
                    output_root=output_root,
                    gate=gate,
                    lock=lock,
                    plan=final_plan,
                    result=result,
                )
            raise
        published_identity = staged_identity
        _rebind_analysis_commit_authority(
            run_root=run_root,
            cache_root=cache_root,
            output_root=output_root,
            gate=gate,
            lock=lock,
            plan=final_plan,
            result=result,
        )
        geoadapt_v2._fsync_directory(output_root.parent)
        _validate_existing_publication(
            output_root,
            result=result,
            plan=final_plan,
        )
        geoadapt_v2.assert_analysis_gate(gate, run_root, final_plan)
        _assert_publication_lock(
            lock,
            output_root=output_root,
            plan=final_plan,
        )
        postcommit_audit = geoadapt_v2.audit_grid(run_root, final_plan)
        if (
            not postcommit_audit["exact_cartesian_complete"]
            or not postcommit_audit["quiescent"]
        ):
            raise GeoAdaptV2AnalysisError(
                "publication inputs changed after atomic rename"
            )
        _verify_captured_ledger_live(
            run_root,
            final_plan,
            result.input_ledger,
        )
    except BaseException:
        if published_identity is not None and geoadapt_v2._path_exists(output_root):
            try:
                _quarantine_publication_residue(
                    output_root,
                    reason="failed",
                    expected_dev=published_identity[0],
                    expected_ino=published_identity[1],
                )
            except (FileNotFoundError, PublicationContention):
                pass
            finally:
                published_identity = None
        if partial_identity is not None and geoadapt_v2._path_exists(partial):
            try:
                _quarantine_publication_residue(
                    partial,
                    reason="failed",
                    expected_dev=int(partial_identity.st_dev),
                    expected_ino=int(partial_identity.st_ino),
                )
            except (FileNotFoundError, PublicationContention):
                pass
        raise
    finally:
        # A missing or replaced owned lock is a contract failure, not an
        # implicit successful release.
        try:
            _release_publication_lock(lock)
        except BaseException:
            if published_identity is not None and geoadapt_v2._path_exists(output_root):
                try:
                    _quarantine_publication_residue(
                        output_root,
                        reason="releasefailed",
                        expected_dev=published_identity[0],
                        expected_ino=published_identity[1],
                    )
                except (FileNotFoundError, PublicationContention):
                    pass
                finally:
                    published_identity = None
            if partial_identity is not None and geoadapt_v2._path_exists(partial):
                try:
                    _quarantine_publication_residue(
                        partial,
                        reason="releasefailed",
                        expected_dev=int(partial_identity.st_dev),
                        expected_ino=int(partial_identity.st_ino),
                    )
                except (FileNotFoundError, PublicationContention):
                    pass
            raise
    return output_root


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    run_root = geoadapt_v2._absolute_path(args.run_root)
    cache_root = geoadapt_v2._absolute_path(args.cache_root)
    output_root = geoadapt_v2._absolute_path(args.output_root)
    plan = geoadapt_v2.load_plan(run_root)
    geoadapt_v2.verify_static_identity(plan)
    with geoadapt_v2.analysis_gate(run_root, plan) as gate:
        published = publish_analysis(
            run_root=run_root,
            cache_root=cache_root,
            output_root=output_root,
            gate=gate,
        )
    summary = geoadapt_v2.strict_load(published / "summary.json")
    print(
        json.dumps(
            {
                "schema": ANALYSIS_SCHEMA,
                "plan_sha256": plan["plan_sha256"],
                "track": geoadapt_v2.TRACK,
                "output_root": str(published),
                "table_row_counts": summary["table_row_counts"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ANALYSIS_SCHEMA",
    "AnalysisResult",
    "COMMON_SUPPORT_DATASETS",
    "GeoAdaptV2AnalysisError",
    "METRIC_NAMES",
    "TABLE_FIELDS",
    "analyze_grid",
    "classification_metrics",
    "main",
    "publish_analysis",
]
