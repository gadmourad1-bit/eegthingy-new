"""Fail-closed post-hoc analysis for :mod:`eeg_mi.full_grid`.

This module is deliberately separate from prediction production.  Its public
entry point first audits the complete immutable Cartesian grid without opening
any cache.  Only after every expected record and prediction checksum passes
does it load plan-bound caches, reconstruct the frozen splits, and join test
row identities to labels in memory.

The production bootstrap default is 100,000 deterministic resamples.  The
result is exploratory evidence on the five opened development cohorts; it is
neither a SOTA claim nor confirmation evidence.
"""

from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import hashlib
import hmac
import io
import json
import math
import os
import stat
import uuid
from collections import defaultdict
from contextlib import contextmanager
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

from . import full_grid


ANALYSIS_SCHEMA = "eeg-mi-common-grid-analysis-v4"
MANIFEST_SCHEMA = "eeg-mi-common-grid-analysis-manifest-v4"
PRODUCTION_BOOTSTRAP_RESAMPLES = 100_000
DEFAULT_BOOTSTRAP_SEED = 20_260_729
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
_RESULT_SEAL_KEY = os.urandom(32)


class GridAnalysisError(RuntimeError):
    """Raised when formal analysis cannot preserve its audit contract."""


@dataclass
class AnalysisResult:
    """Aggregate-only analysis state.

    ``tables`` contains scalar metric rows and resource summaries.  It never
    contains labels, prediction probabilities, or row-index vectors.
    """

    summary: dict[str, Any]
    tables: dict[str, list[dict[str, Any]]]
    input_ledger: list[dict[str, Any]] = field(repr=False)
    _seal: str = field(default="", repr=False)


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


def _sha256_file(path: Path) -> str:
    return full_grid._sha256_file(path)


def _analysis_result_seal(
    summary: Mapping[str, Any],
    tables: Mapping[str, Any],
    ledger: Sequence[Mapping[str, Any]],
) -> str:
    payload = _canonical_bytes(
        {
            "summary": summary,
            "tables": tables,
            "input_ledger": ledger,
        }
    )
    return hmac.new(_RESULT_SEAL_KEY, payload, hashlib.sha256).hexdigest()


def _strict_record_tree(
    run_root: Path,
    jobs: Sequence[full_grid.Job],
) -> None:
    """Reject unknown files, directories, and symlinks below ``records``."""

    records_root = run_root / "records"
    expected_directories = {
        full_grid._record_directory(run_root, job) for job in jobs
    }
    expected_files = {
        directory / filename
        for directory in expected_directories
        for filename in full_grid.FINAL_FILENAMES
    }
    allowed_directories = {records_root}
    for directory in expected_directories:
        current = directory
        while current != records_root:
            allowed_directories.add(current)
            current = current.parent
        allowed_directories.add(records_root)

    observed_files: set[Path] = set()
    observed_directories: set[Path] = {records_root}
    if not records_root.is_dir():
        raise GridAnalysisError(f"records directory is absent: {records_root}")
    for path in records_root.rglob("*"):
        if path.is_symlink():
            raise GridAnalysisError(f"records tree contains a symlink: {path}")
        if path.is_dir():
            observed_directories.add(path)
        elif path.is_file():
            observed_files.add(path)
        else:
            raise GridAnalysisError(f"records tree contains a special file: {path}")
    if observed_files != expected_files:
        missing = sorted(str(path) for path in expected_files - observed_files)
        extra = sorted(str(path) for path in observed_files - expected_files)
        raise GridAnalysisError(
            "records tree differs from the exact Cartesian file set; "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )
    extra_directories = observed_directories - allowed_directories
    if extra_directories:
        raise GridAnalysisError(
            "records tree contains unexpected directories: "
            f"{sorted(str(path) for path in extra_directories)[:3]}"
        )


def _require_quiescent_auxiliary_state(run_root: Path) -> None:
    """Reject any claim or unpublished partial, including unknown job IDs."""

    for name in (
        "claims",
        "partials",
        full_grid.CLAIM_TOMBSTONE_DIRECTORY,
    ):
        root = run_root / name
        if not root.exists() and not root.is_symlink():
            continue
        try:
            full_grid._require_real_directory(root)
        except (FileNotFoundError, full_grid.FullGridError) as error:
            raise GridAnalysisError(
                f"refusing analysis with unsafe {name} state"
            ) from error
        observed = [
            path
            for path in root.rglob("*")
            if not path.is_dir() or path.is_symlink()
        ]
        if observed:
            raise GridAnalysisError(
                f"refusing analysis while {name} state exists: "
                f"{[str(path) for path in observed[:3]]}"
            )


def audit_exact_grid(
    run_root: str | Path,
) -> tuple[dict[str, Any], tuple[full_grid.Job, ...], dict[str, Any]]:
    """Load and audit the exact plan without opening any data cache."""

    root = full_grid._safe_run_root(Path(run_root))
    plan = full_grid.load_plan(root)
    jobs = tuple(full_grid.iter_jobs(plan))
    if len(jobs) != int(plan.get("n_jobs", -1)):
        raise GridAnalysisError("plan job count differs from its Cartesian product")
    preflight_sha256: str | None = None
    if plan.get("publication_mode") == full_grid.FORMAL_PUBLICATION_MODE:
        try:
            preflight_sha256 = full_grid.load_preflight_attestation(
                root,
                plan,
            )["report_sha256"]
        except Exception as error:
            raise GridAnalysisError(
                "refusing analysis without the exact immutable preflight attestation"
            ) from error

    # This must remain the first potentially expensive operation after loading
    # the plan.  It validates completion identities plus record/prediction
    # checksums for every expected job.
    audit = full_grid.audit_grid(root, plan)
    audit_counter_names = (
        "expected",
        "complete",
        "missing",
        "corrupt",
        "extra",
        "live_claims",
        "stale_claims",
        "unknown_claims",
        "partials",
        "claim_tombstones",
        "unsafe_paths",
        "unexpected_root_entries",
        "failed_jobs",
        "quarantine_artifacts",
        "resolved_forensic_artifacts",
        "invalid_forensic_artifacts",
    )
    if any(
        type(audit.get(name)) is not int or int(audit[name]) < 0
        for name in audit_counter_names
    ):
        raise GridAnalysisError(
            "refusing analysis: audit counters are not exact nonnegative integers"
        )
    raw_forensic_ledger = (
        audit.get("details", {}).get("forensic_ledger")
        if isinstance(audit.get("details"), Mapping)
        else None
    )
    forensic_ledger_valid = isinstance(raw_forensic_ledger, list)
    if forensic_ledger_valid:
        for row in raw_forensic_ledger:
            if (
                not isinstance(row, dict)
                or set(row)
                != {
                    "category",
                    "reason",
                    "identity",
                    "path",
                    "artifact_sha256",
                }
                or not isinstance(row["category"], str)
                or row["category"] not in full_grid.QUARANTINE_REASONS
                or not isinstance(row["reason"], str)
                or row["reason"]
                not in full_grid.QUARANTINE_REASONS[row["category"]]
                or not isinstance(row["identity"], str)
                or not row["identity"]
                or not isinstance(row["path"], str)
                or not row["path"].startswith(
                    f"quarantine/{row['category']}/"
                )
                or not isinstance(row["artifact_sha256"], str)
                or full_grid.HEX_64_RE.fullmatch(
                    row["artifact_sha256"]
                )
                is None
            ):
                forensic_ledger_valid = False
                break
    if forensic_ledger_valid and raw_forensic_ledger != sorted(
        raw_forensic_ledger,
        key=lambda row: (
            row["category"],
            row["path"],
            row["artifact_sha256"],
        ),
    ):
        forensic_ledger_valid = False
    if (
        audit.get("plan_sha256") != plan.get("plan_sha256")
        or audit.get("expected") != len(jobs)
        or audit.get("complete") != len(jobs)
        or audit.get("exact_cartesian_complete") is not True
        or audit["missing"] != 0
        or audit["corrupt"] != 0
        or audit["extra"] != 0
        or audit["live_claims"] != 0
        or audit["stale_claims"] != 0
        or audit["unknown_claims"] != 0
        or audit["partials"] != 0
        or audit["claim_tombstones"] != 0
        or audit["unsafe_paths"] != 0
        or audit["unexpected_root_entries"] != 0
        or audit["invalid_forensic_artifacts"] != 0
        or audit["resolved_forensic_artifacts"]
        != audit["quarantine_artifacts"]
        or not forensic_ledger_valid
        or not isinstance(audit.get("forensic_ledger_sha256"), str)
        or full_grid.HEX_64_RE.fullmatch(
            audit["forensic_ledger_sha256"]
        )
        is None
        or not isinstance(audit.get("details"), Mapping)
        or not isinstance(audit["details"].get("forensic_ledger"), list)
        or len(audit["details"]["forensic_ledger"])
        != audit["resolved_forensic_artifacts"]
        or hashlib.sha256(
            full_grid._canonical_bytes(
                audit["details"]["forensic_ledger"]
            )
        ).hexdigest()
        != audit["forensic_ledger_sha256"]
        or (
            plan.get("publication_mode") == full_grid.FORMAL_PUBLICATION_MODE
            and (
                audit.get("preflight_attestation_valid") is not True
                or audit.get("preflight_report_sha256") != preflight_sha256
            )
        )
    ):
        raise GridAnalysisError(
            "refusing analysis: the score-blind Cartesian grid is not an "
            f"exact, quiescent, checksum-valid snapshot ({audit})"
        )
    _strict_record_tree(root, jobs)
    _require_quiescent_auxiliary_state(root)
    return plan, jobs, audit


def _analysis_contract_sha256(contract: Mapping[str, Any] | None) -> str | None:
    return (
        None
        if contract is None
        else hashlib.sha256(_canonical_bytes(contract)).hexdigest()
    )


def _verify_frozen_analysis_contract(
    plan: Mapping[str, Any],
    *,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    ece_bins: int,
    tcformer_name: str,
) -> dict[str, Any] | None:
    """Verify the pre-result analyzer source, environment, and decisions."""

    raw_contract = plan.get("analysis_contract")
    formal_grid = (
        plan.get("publication_mode") == full_grid.FORMAL_PUBLICATION_MODE
        and tuple(plan.get("dataset_order", ())) == full_grid.OPENED_DATASETS
        and tuple(plan.get("architectures", ()))
        == full_grid.COMMON_ARCHITECTURES
        and int(plan.get("n_jobs", -1)) == full_grid.FORMAL_EXPECTED_JOBS
    )
    if raw_contract is None:
        if formal_grid:
            raise GridAnalysisError(
                "formal full-grid plan has no pre-result analysis contract"
            )
        return None
    if not isinstance(raw_contract, Mapping):
        raise GridAnalysisError("analysis contract is not an object")
    contract = copy.deepcopy(dict(raw_contract))
    try:
        full_grid._validate_analysis_contract_shape(contract)
    except (TypeError, ValueError) as error:
        raise GridAnalysisError("analysis contract schema is invalid") from error
    expected = full_grid._formal_analysis_contract(
        plan["environment_identity"]
    )
    if contract != expected:
        raise GridAnalysisError(
            "analysis source or frozen decision contract differs from plan.json"
        )
    observed_environment = full_grid._environment_identity()
    if (
        full_grid._analysis_environment_sha256(observed_environment)
        != contract["environment_identity_sha256"]
    ):
        raise GridAnalysisError(
            "current analysis environment differs from the pre-result plan"
        )
    requested = {
        "bootstrap_resamples": int(bootstrap_resamples),
        "bootstrap_seed": int(bootstrap_seed),
        "ece_bins": int(ece_bins),
        "tcformer_comparator": str(tcformer_name),
    }
    frozen = {key: contract[key] for key in requested}
    if requested != frozen:
        raise GridAnalysisError(
            "requested analysis settings differ from the pre-result plan"
        )
    return contract


def _bound_cache_labels(
    plan: Mapping[str, Any],
    cache_root: Path,
) -> dict[tuple[str, int], np.ndarray]:
    """Load labels only after grid audit and revalidate every plan binding."""

    from .data import split_indices

    labels_by_subject: dict[tuple[str, int], np.ndarray] = {}
    for dataset in plan["dataset_order"]:
        contract = plan["datasets"][dataset]
        for raw_subject in contract["subjects"]:
            subject = int(raw_subject)
            subject_key = full_grid._subject_identity_key(dataset, subject)
            planned_cache = plan["cache_identity"].get(subject_key)
            if not isinstance(planned_cache, Mapping):
                raise GridAnalysisError(f"plan cache identity is absent for {subject_key}")
            cache = full_grid._load_subject_cache_safely(
                str(dataset),
                subject,
                cache_root=cache_root,
            )
            if cache.get("identity") != planned_cache:
                raise GridAnalysisError(
                    f"cache identity differs from the plan for {subject_key}"
                )
            labels = np.asarray(cache.get("y"))
            if (
                labels.dtype != np.int64
                or labels.ndim != 1
                or len(labels) == 0
                or np.any(labels < 0)
                or np.any(labels >= int(contract["n_classes"]))
            ):
                raise GridAnalysisError(f"cache labels are invalid for {subject_key}")

            seen_test_rows: set[int] = set()
            for raw_fold in contract["folds"]:
                fold = int(raw_fold)
                train_rows, validation_rows, test_rows = split_indices(
                    str(dataset),
                    labels,
                    cache["sessions"],
                    cache["runs"],
                    fold=fold,
                    subject=subject,
                )
                observed = full_grid._one_split_identity(
                    dataset=str(dataset),
                    subject=subject,
                    fold=fold,
                    cache_array_sha256=str(planned_cache["array_sha256"]),
                    trial_count=len(labels),
                    train_rows=train_rows,
                    validation_rows=validation_rows,
                    test_rows=test_rows,
                )
                split_key = full_grid._split_identity_key(dataset, subject, fold)
                if observed != plan["split_identity"].get(split_key):
                    raise GridAnalysisError(
                        f"reconstructed split differs from the plan for {split_key}"
                    )
                fold_rows = set(np.asarray(test_rows, dtype=np.int64).tolist())
                overlap = seen_test_rows.intersection(fold_rows)
                if overlap:
                    raise GridAnalysisError(
                        f"test folds overlap for {subject_key}: {sorted(overlap)[:5]}"
                    )
                seen_test_rows.update(fold_rows)
            labels_by_subject[(str(dataset), subject)] = labels.copy()
    return labels_by_subject


def classification_metrics(
    y_true: np.ndarray | Sequence[int],
    probabilities: np.ndarray,
    *,
    n_classes: int,
    ece_bins: int = DEFAULT_ECE_BINS,
) -> dict[str, float | None]:
    """Compute the preregistered multiclass metrics for one prediction set."""

    y = np.asarray(y_true, dtype=np.int64)
    values = np.asarray(probabilities, dtype=np.float64)
    if (
        y.ndim != 1
        or len(y) == 0
        or values.shape != (len(y), int(n_classes))
        or n_classes < 2
        or ece_bins < 2
        or np.any(y < 0)
        or np.any(y >= n_classes)
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(values > 1.0)
        or not np.allclose(values.sum(axis=1), 1.0, atol=1e-6)
    ):
        raise GridAnalysisError("invalid arrays supplied to metric computation")
    predicted = np.argmax(values, axis=1)
    all_classes = np.arange(n_classes, dtype=np.int64)
    with np.errstate(all="ignore"):
        kappa = _finite_float(
            cohen_kappa_score(y, predicted, labels=all_classes.tolist())
        )
    support = np.bincount(y, minlength=n_classes)
    correct_by_class = np.bincount(
        y[predicted == y],
        minlength=n_classes,
    )
    present = support > 0
    balanced_accuracy = float(
        np.mean(correct_by_class[present] / support[present])
    )
    chance_level = 1.0 / n_classes
    chance_normalized_balanced_accuracy = (
        balanced_accuracy - chance_level
    ) / (1.0 - chance_level)

    aucs: list[float] = []
    for class_index in range(n_classes):
        binary = (y == class_index).astype(np.int8)
        if binary.min() == binary.max():
            aucs = []
            break
        aucs.append(float(roc_auc_score(binary, values[:, class_index])))
    macro_auc = float(np.mean(aucs)) if len(aucs) == n_classes else None

    clipped = np.clip(values[np.arange(len(y)), y], 1e-15, 1.0)
    one_hot = np.eye(n_classes, dtype=np.float64)[y]
    confidence = values[np.arange(len(y)), predicted]
    correct = (predicted == y).astype(np.float64)
    bin_ids = np.minimum(
        np.floor(confidence * ece_bins).astype(np.int64),
        ece_bins - 1,
    )
    ece = 0.0
    for bin_index in range(ece_bins):
        mask = bin_ids == bin_index
        if np.any(mask):
            ece += (
                float(mask.mean())
                * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
            )
    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": balanced_accuracy,
        "chance_normalized_balanced_accuracy": float(
            chance_normalized_balanced_accuracy
        ),
        "macro_f1": float(
            f1_score(
                y,
                predicted,
                labels=all_classes.tolist(),
                average="macro",
                zero_division=0,
            )
        ),
        "cohen_kappa": kappa,
        "ovr_macro_auroc": macro_auc,
        "nll": float(-np.log(clipped).mean()),
        "multiclass_brier": float(np.square(values - one_hot).sum(axis=1).mean()),
        "ece": float(ece),
    }


def _metric_means(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for name in METRIC_NAMES:
        values = [
            float(row[name])
            for row in rows
            if row.get(name) is not None and math.isfinite(float(row[name]))
        ]
        result[name] = float(np.mean(values)) if values else None
        result[f"{name}_defined_count"] = len(values)
    return result


@dataclass
class _GlobalMicro:
    bins: int
    count: int = 0
    correct: int = 0
    nll_sum: float = 0.0
    brier_sum: float = 0.0
    bin_count: np.ndarray = field(init=False)
    bin_correct: np.ndarray = field(init=False)
    bin_confidence: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.bin_count = np.zeros(self.bins, dtype=np.int64)
        self.bin_correct = np.zeros(self.bins, dtype=np.float64)
        self.bin_confidence = np.zeros(self.bins, dtype=np.float64)

    def update(self, y: np.ndarray, probabilities: np.ndarray) -> None:
        predicted = np.argmax(probabilities, axis=1)
        confidence = probabilities[np.arange(len(y)), predicted]
        correct = (predicted == y).astype(np.float64)
        chosen = np.clip(probabilities[np.arange(len(y)), y], 1e-15, 1.0)
        one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[y]
        bin_ids = np.minimum(
            np.floor(confidence * self.bins).astype(np.int64),
            self.bins - 1,
        )
        self.count += len(y)
        self.correct += int(correct.sum())
        self.nll_sum += float(-np.log(chosen).sum())
        self.brier_sum += float(np.square(probabilities - one_hot).sum())
        for bin_index in range(self.bins):
            mask = bin_ids == bin_index
            self.bin_count[bin_index] += int(mask.sum())
            self.bin_correct[bin_index] += float(correct[mask].sum())
            self.bin_confidence[bin_index] += float(confidence[mask].sum())

    def metrics(self) -> dict[str, float | None]:
        if self.count <= 0:
            raise GridAnalysisError("empty global trial-micro accumulator")
        ece = 0.0
        for bin_index, count in enumerate(self.bin_count):
            if count:
                accuracy = self.bin_correct[bin_index] / count
                confidence = self.bin_confidence[bin_index] / count
                ece += (count / self.count) * abs(accuracy - confidence)
        return {
            "accuracy": self.correct / self.count,
            "balanced_accuracy": None,
            "chance_normalized_balanced_accuracy": None,
            "macro_f1": None,
            "cohen_kappa": None,
            "ovr_macro_auroc": None,
            "nll": self.nll_sum / self.count,
            "multiclass_brier": self.brier_sum / self.count,
            "ece": float(ece),
        }


def _bootstrap_summary(samples: np.ndarray, observed: float) -> dict[str, float]:
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "observed": float(observed),
        "ci95_low": float(low),
        "ci95_high": float(high),
    }


def bootstrap_subject_mean(
    differences: np.ndarray | Sequence[float],
    *,
    n_resamples: int = PRODUCTION_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, float | int]:
    """Bootstrap a dataset effect with subject as the indivisible cluster."""

    values = np.asarray(differences, dtype=np.float64)
    if (
        values.ndim != 1
        or len(values) == 0
        or not np.all(np.isfinite(values))
        or n_resamples <= 0
    ):
        raise ValueError("bootstrap differences/resample count are invalid")
    rng = np.random.default_rng(seed)
    samples = np.empty(n_resamples, dtype=np.float64)
    chunk_size = 4096
    for start in range(0, n_resamples, chunk_size):
        stop = min(n_resamples, start + chunk_size)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        samples[start:stop] = values[indices].mean(axis=1)
    result: dict[str, float | int] = _bootstrap_summary(samples, float(values.mean()))
    result.update({"resamples": int(n_resamples), "seed": int(seed)})
    return result


def fixed_suite_subject_bootstrap(
    dataset_subject_differences: Mapping[str, Sequence[float] | np.ndarray],
    *,
    n_resamples: int = PRODUCTION_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, float | int]:
    """Bootstrap the fixed dataset suite by resampling subjects within strata.

    Every replicate contains every planned dataset exactly once and gives each
    dataset equal weight. Only paired subjects are resampled within a dataset;
    folds and seeds remain inside the subject cluster.
    """

    names = tuple(sorted(str(name) for name in dataset_subject_differences))
    arrays = tuple(
        np.asarray(dataset_subject_differences[name], dtype=np.float64)
        for name in names
    )
    if (
        not names
        or n_resamples <= 0
        or any(
            values.ndim != 1
            or len(values) == 0
            or not np.all(np.isfinite(values))
            for values in arrays
        )
    ):
        raise ValueError("fixed-suite bootstrap inputs are invalid")
    observed = float(np.mean([values.mean() for values in arrays]))
    rng = np.random.default_rng(seed)
    samples = np.empty(n_resamples, dtype=np.float64)
    chunk_size = 4096
    for start in range(0, n_resamples, chunk_size):
        stop = min(n_resamples, start + chunk_size)
        size = stop - start
        chunk = np.zeros(size, dtype=np.float64)
        for values in arrays:
            indices = rng.integers(
                0,
                len(values),
                size=(size, len(values)),
            )
            chunk += values[indices].mean(axis=1)
        samples[start:stop] = chunk / len(arrays)
    result: dict[str, float | int] = _bootstrap_summary(samples, observed)
    result.update(
        {
            "resamples": int(n_resamples),
            "seed": int(seed),
            "datasets_fixed": len(arrays),
        }
    )
    return result


def hierarchical_subject_bootstrap(
    dataset_subject_differences: Mapping[str, Sequence[float] | np.ndarray],
    *,
    n_resamples: int = PRODUCTION_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, float | int]:
    """Dataset-superpopulation, then subject, paired hierarchical bootstrap.

    Datasets are sampled with replacement with equal probability.  Within each
    selected dataset occurrence, subjects are sampled with replacement; all
    folds and seeds have already been aggregated inside each subject cluster.
    This targets a dataset-superpopulation estimand and is not the fixed-suite
    headline interval.
    """

    names = tuple(sorted(str(name) for name in dataset_subject_differences))
    arrays = tuple(
        np.asarray(dataset_subject_differences[name], dtype=np.float64)
        for name in names
    )
    if (
        not names
        or n_resamples <= 0
        or any(
            value.ndim != 1
            or len(value) == 0
            or not np.all(np.isfinite(value))
            for value in arrays
        )
    ):
        raise ValueError("hierarchical bootstrap inputs are invalid")
    observed = float(np.mean([value.mean() for value in arrays]))
    rng = np.random.default_rng(seed)
    samples = np.empty(n_resamples, dtype=np.float64)
    n_datasets = len(arrays)
    chunk_size = 4096
    for start in range(0, n_resamples, chunk_size):
        stop = min(n_resamples, start + chunk_size)
        size = stop - start
        selected_datasets = rng.integers(
            0,
            n_datasets,
            size=(size, n_datasets),
        )
        chunk = np.zeros(size, dtype=np.float64)
        for hierarchy_position in range(n_datasets):
            choices = selected_datasets[:, hierarchy_position]
            for dataset_index, values in enumerate(arrays):
                mask = choices == dataset_index
                selected_count = int(mask.sum())
                if selected_count == 0:
                    continue
                subject_indices = rng.integers(
                    0,
                    len(values),
                    size=(selected_count, len(values)),
                )
                chunk[mask] += values[subject_indices].mean(axis=1)
        samples[start:stop] = chunk / n_datasets
    result: dict[str, float | int] = _bootstrap_summary(samples, observed)
    result.update({"resamples": int(n_resamples), "seed": int(seed)})
    return result


def _timing_summary(
    job_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    overall: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in job_rows:
        grouped[(str(row["dataset"]), str(row["model"]))].append(row)
        overall[str(row["model"])].append(row)

    result: list[dict[str, Any]] = []
    for (dataset, model), rows in sorted(grouped.items()):
        result.append(_one_timing_row(dataset, model, rows))
    for model, rows in sorted(overall.items()):
        result.append(_one_timing_row("ALL_DATASETS", model, rows))
    return result


def _one_timing_row(
    dataset: str,
    model: str,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "dataset": dataset,
        "model": model,
        "jobs": len(rows),
    }
    specifications = {
        "parameter_count": "parameter_count",
        "selected_epoch_index": "selected_epoch_index",
        "selected_epoch_count": "selected_epoch_count",
        "selection_fit_seconds": "selection_fit_seconds",
        "refit_fit_seconds": "refit_fit_seconds",
        "test_inference_seconds": "test_inference_seconds",
        "job_total_seconds": "job_total_seconds",
        "cuda_peak_memory_bytes": "cuda_peak_memory_bytes",
        "inference_ms_per_trial": "inference_ms_per_trial",
    }
    for output_name, input_name in specifications.items():
        values = np.asarray([float(row[input_name]) for row in rows], dtype=np.float64)
        result[f"{output_name}_mean"] = float(values.mean())
        result[f"{output_name}_median"] = float(np.median(values))
        result[f"{output_name}_p95"] = float(np.quantile(values, 0.95))
        result[f"{output_name}_min"] = float(values.min())
        result[f"{output_name}_max"] = float(values.max())
    return result


def _record_scalar_row(
    job: full_grid.Job,
    record: Mapping[str, Any],
    metrics: Mapping[str, float | None],
) -> dict[str, Any]:
    metadata = record["metadata"]
    fit = metadata["fit"]
    timing = metadata["timing_seconds"]
    test_count = int(record["test_count"])
    return {
        **job.identity(),
        "job_id": job.job_id,
        "test_count": test_count,
        **metrics,
        "selected_epoch_index": int(fit["source_selected_epoch"]),
        "selected_epoch_count": int(fit["source_selected_epoch"]) + 1,
        "selection_epochs_run": int(fit["selection_epochs_run"]),
        "refit_epochs_run": int(fit["refit_epochs_run"]),
        "parameter_count": int(fit["parameter_count"]),
        "selection_fit_seconds": float(timing["selection_fit"]),
        "refit_fit_seconds": float(timing["refit_fit"]),
        "test_inference_seconds": float(timing["test_inference"]),
        "job_total_seconds": float(timing["job_total"]),
        "cuda_peak_memory_bytes": int(metadata["cuda_peak_memory_bytes"]),
        "inference_ms_per_trial": (
            1000.0 * float(timing["test_inference"]) / test_count
        ),
    }


def _input_ledger_row(
    run_root: Path,
    job: full_grid.Job,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    del run_root
    artifact_sha256 = payload.get("artifact_sha256")
    if (
        not isinstance(artifact_sha256, Mapping)
        or set(artifact_sha256)
        != {
            "completion.json",
            "record.json",
            "predictions.npz",
        }
        or any(
            not isinstance(value, str)
            or full_grid.HEX_64_RE.fullmatch(value) is None
            for value in artifact_sha256.values()
        )
    ):
        raise GridAnalysisError(
            f"validated completion snapshot lacks exact digests for {job.job_id}"
        )
    return {
        "job_id": job.job_id,
        "completion_sha256": artifact_sha256["completion.json"],
        "record_sha256": artifact_sha256["record.json"],
        "predictions_sha256": artifact_sha256["predictions.npz"],
    }


def _verify_input_ledger(
    run_root: Path,
    plan: Mapping[str, Any],
    ledger: Sequence[Mapping[str, Any]],
) -> None:
    jobs = tuple(full_grid.iter_jobs(plan))
    if (
        len(ledger) != len(jobs)
        or [str(row.get("job_id")) for row in ledger]
        != [job.job_id for job in jobs]
    ):
        raise GridAnalysisError(
            "input checksum ledger differs from the exact plan job order"
        )
    preflight_report_sha256 = (
        full_grid.load_preflight_attestation(run_root, plan)["report_sha256"]
        if plan["publication_mode"] == full_grid.FORMAL_PUBLICATION_MODE
        else None
    )
    for job, row in zip(jobs, ledger, strict=True):
        payload = full_grid._validate_completion_bound(
            run_root,
            plan,
            job,
            expected_preflight_report_sha256=preflight_report_sha256,
        )
        artifact_sha256 = payload.get("artifact_sha256")
        if not isinstance(artifact_sha256, Mapping):
            raise GridAnalysisError(
                f"validated completion lacks a digest snapshot for {job.job_id}"
            )
        observed = {
            "completion_sha256": artifact_sha256.get("completion.json"),
            "record_sha256": artifact_sha256.get("record.json"),
            "predictions_sha256": artifact_sha256.get("predictions.npz"),
        }
        expected = {key: str(row[key]) for key in observed}
        if observed != expected:
            raise GridAnalysisError(
                f"record changed during analysis for {job.job_id}"
            )


def _subject_index(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
) -> dict[tuple[str, str, int], float]:
    result: dict[tuple[str, str, int], float] = {}
    for row in rows:
        value = row.get(metric)
        if value is None:
            raise GridAnalysisError(
                f"paired metric {metric} is undefined for a subject"
            )
        key = (str(row["dataset"]), str(row["model"]), int(row["subject"]))
        if key in result:
            raise GridAnalysisError(f"duplicate subject aggregate {key}")
        result[key] = float(value)
    return result


def _subject_seed_index(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
) -> dict[tuple[str, str, int, int], float]:
    result: dict[tuple[str, str, int, int], float] = {}
    for row in rows:
        value = row.get(metric)
        if value is None:
            raise GridAnalysisError(
                f"seed-sensitivity metric {metric} is undefined"
            )
        key = (
            str(row["dataset"]),
            str(row["model"]),
            int(row["subject"]),
            int(row["seed"]),
        )
        if key in result:
            raise GridAnalysisError(f"duplicate subject-seed aggregate {key}")
        result[key] = float(value)
    return result


def _paired_comparisons(
    *,
    plan: Mapping[str, Any],
    subject_rows: Sequence[Mapping[str, Any]],
    subject_seed_rows: Sequence[Mapping[str, Any]],
    overall_rows: Sequence[Mapping[str, Any]],
    bootstrap_resamples: int,
    bootstrap_seed: int,
    tcformer_name: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    architectures = [str(value) for value in plan["architectures"]]
    if tcformer_name not in architectures:
        raise GridAnalysisError(
            f"required TCFormer comparator {tcformer_name!r} is absent"
        )
    if int(plan["common_architecture_count"]) != len(architectures):
        raise GridAnalysisError("common-roster count differs from the plan")
    overall_by_model = {str(row["model"]): row for row in overall_rows}
    missing = [model for model in architectures if model not in overall_by_model]
    if missing:
        raise GridAnalysisError(f"overall aggregates missing models: {missing}")

    subject_index = _subject_index(subject_rows, metric="balanced_accuracy")
    seed_index = _subject_seed_index(
        subject_seed_rows,
        metric="balanced_accuracy",
    )
    dataset_effect_rows: list[dict[str, Any]] = []
    overall_effect_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    compared_models = [
        model for model in architectures if model != tcformer_name
    ]
    for comparison_index, model in enumerate(compared_models):
        differences_by_dataset: dict[str, np.ndarray] = {}
        for dataset in plan["dataset_order"]:
            subjects = [int(value) for value in plan["datasets"][dataset]["subjects"]]
            differences = np.asarray(
                [
                    subject_index[(str(dataset), model, subject)]
                    - subject_index[(str(dataset), tcformer_name, subject)]
                    for subject in subjects
                ],
                dtype=np.float64,
            )
            differences_by_dataset[str(dataset)] = differences
            bootstrap = bootstrap_subject_mean(
                differences,
                n_resamples=bootstrap_resamples,
                seed=(
                    bootstrap_seed
                    + comparison_index * 1000
                    + len(differences_by_dataset)
                ),
            )
            dataset_effect_rows.append(
                {
                    "model": model,
                    "comparator": tcformer_name,
                    "dataset": str(dataset),
                    "subjects": len(subjects),
                    "mean_balanced_accuracy_difference": float(differences.mean()),
                    "median_balanced_accuracy_difference": float(
                        np.median(differences)
                    ),
                    "descriptive_bootstrap_interval95_low": bootstrap["ci95_low"],
                    "descriptive_bootstrap_interval95_high": bootstrap["ci95_high"],
                    "bootstrap_resamples": bootstrap_resamples,
                    "bootstrap_seed": bootstrap["seed"],
                    "status": (
                        "descriptive development-cohort context only; no "
                        "hypothesis test, confirmation, or SOTA inference"
                    ),
                }
            )

        fixed_suite = fixed_suite_subject_bootstrap(
            differences_by_dataset,
            n_resamples=bootstrap_resamples,
            seed=bootstrap_seed + comparison_index * 10_000,
        )
        superpopulation = hierarchical_subject_bootstrap(
            differences_by_dataset,
            n_resamples=bootstrap_resamples,
            seed=bootstrap_seed + comparison_index * 10_000 + 500,
        )
        overall_effect_rows.append(
            {
                "model": model,
                "comparator": tcformer_name,
                "datasets": len(differences_by_dataset),
                "subjects_total": int(
                    sum(len(value) for value in differences_by_dataset.values())
                ),
                "equal_dataset_macro_balanced_accuracy_difference": fixed_suite[
                    "observed"
                ],
                "descriptive_fixed_suite_bootstrap_interval95_low": fixed_suite[
                    "ci95_low"
                ],
                "descriptive_fixed_suite_bootstrap_interval95_high": fixed_suite[
                    "ci95_high"
                ],
                "fixed_suite_bootstrap_resamples": bootstrap_resamples,
                "fixed_suite_bootstrap_seed": fixed_suite["seed"],
                "descriptive_dataset_superpopulation_interval95_low": (
                    superpopulation["ci95_low"]
                ),
                "descriptive_dataset_superpopulation_interval95_high": (
                    superpopulation["ci95_high"]
                ),
                "dataset_superpopulation_bootstrap_resamples": bootstrap_resamples,
                "dataset_superpopulation_bootstrap_seed": superpopulation["seed"],
                "status": (
                    "descriptive model-minus-prespecified-TCFormer context only; "
                    "intervals describe this opened development suite and are not "
                    "confirmatory confidence intervals"
                ),
            }
        )

        for seed in [int(value) for value in plan["seeds"]]:
            dataset_seed_effects: list[float] = []
            for dataset in plan["dataset_order"]:
                subjects = [
                    int(value) for value in plan["datasets"][dataset]["subjects"]
                ]
                differences = [
                    seed_index[(str(dataset), model, subject, seed)]
                    - seed_index[(str(dataset), tcformer_name, subject, seed)]
                    for subject in subjects
                ]
                dataset_effect = float(np.mean(differences))
                dataset_seed_effects.append(dataset_effect)
                sensitivity_rows.append(
                    {
                        "model": model,
                        "comparator": tcformer_name,
                        "seed": seed,
                        "dataset": str(dataset),
                        "subjects": len(subjects),
                        "balanced_accuracy_difference": dataset_effect,
                    }
                )
            sensitivity_rows.append(
                {
                    "model": model,
                    "comparator": tcformer_name,
                    "seed": seed,
                    "dataset": "EQUAL_DATASET_OVERALL",
                    "subjects": int(
                        sum(
                            len(plan["datasets"][dataset]["subjects"])
                            for dataset in plan["dataset_order"]
                        )
                    ),
                    "balanced_accuracy_difference": float(
                        np.mean(dataset_seed_effects)
                    ),
                }
            )
    return dataset_effect_rows, overall_effect_rows, sensitivity_rows


def compute_analysis(
    *,
    run_root: str | Path,
    cache_root: str | Path,
    bootstrap_resamples: int = PRODUCTION_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    ece_bins: int = DEFAULT_ECE_BINS,
    tcformer_name: str = "tcformer",
) -> AnalysisResult:
    """Audit, join, score, aggregate, and compare one complete formal grid."""

    if bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be positive")
    if ece_bins < 2:
        raise ValueError("ece_bins must be at least two")
    root = full_grid._safe_run_root(Path(run_root))
    cache = full_grid._absolute_path(Path(cache_root))

    # No cache-loading call may move above this fail-closed audit.
    plan, jobs, audit = audit_exact_grid(root)
    analysis_contract = _verify_frozen_analysis_contract(
        plan,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
        ece_bins=ece_bins,
        tcformer_name=tcformer_name,
    )
    preflight_report_sha256 = (
        audit["preflight_report_sha256"]
        if plan["publication_mode"] == full_grid.FORMAL_PUBLICATION_MODE
        else None
    )
    labels_by_subject = _bound_cache_labels(plan, cache)

    job_rows: list[dict[str, Any]] = []
    subject_seed_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    trial_micro_rows: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    global_micro = {
        str(model): _GlobalMicro(ece_bins) for model in plan["architectures"]
    }

    for dataset in plan["dataset_order"]:
        contract = plan["datasets"][dataset]
        n_classes = int(contract["n_classes"])
        folds = [int(value) for value in contract["folds"]]
        seeds = [int(value) for value in plan["seeds"]]
        for model in plan["architectures"]:
            model = str(model)
            dataset_subject_rows: list[dict[str, Any]] = []
            dataset_y: list[np.ndarray] = []
            dataset_probabilities: list[np.ndarray] = []
            for raw_subject in contract["subjects"]:
                subject = int(raw_subject)
                labels = labels_by_subject[(str(dataset), subject)]
                by_seed: dict[int, dict[str, list[np.ndarray]]] = {
                    seed: {"rows": [], "y": [], "probabilities": []}
                    for seed in seeds
                }
                for fold in folds:
                    for seed in seeds:
                        job = full_grid.Job(
                            dataset=str(dataset),
                            model=model,
                            subject=subject,
                            fold=fold,
                            seed=seed,
                        )
                        payload = full_grid._validate_completion_bound(
                            root,
                            plan,
                            job,
                            expected_preflight_report_sha256=(
                                preflight_report_sha256
                            ),
                        )
                        rows = np.asarray(payload["rows"], dtype=np.int64)
                        probabilities = np.asarray(
                            payload["probabilities"],
                            dtype=np.float64,
                        )
                        y = labels[rows]
                        metrics = classification_metrics(
                            y,
                            probabilities,
                            n_classes=n_classes,
                            ece_bins=ece_bins,
                        )
                        job_rows.append(
                            _record_scalar_row(job, payload["record"], metrics)
                        )
                        ledger.append(_input_ledger_row(root, job, payload))
                        by_seed[seed]["rows"].append(rows)
                        by_seed[seed]["y"].append(y)
                        by_seed[seed]["probabilities"].append(probabilities)
                        dataset_y.append(y)
                        dataset_probabilities.append(probabilities)
                        global_micro[model].update(y, probabilities)

                one_subject_seed_rows: list[dict[str, Any]] = []
                reference_rows: np.ndarray | None = None
                for seed in seeds:
                    row_values = np.concatenate(by_seed[seed]["rows"])
                    y_values = np.concatenate(by_seed[seed]["y"])
                    probability_values = np.concatenate(
                        by_seed[seed]["probabilities"]
                    )
                    order = np.argsort(row_values, kind="stable")
                    row_values = row_values[order]
                    y_values = y_values[order]
                    probability_values = probability_values[order]
                    if len(np.unique(row_values)) != len(row_values):
                        raise GridAnalysisError(
                            f"fold test rows overlap for {dataset}/{model}/"
                            f"S{subject}/seed{seed}"
                        )
                    if reference_rows is None:
                        reference_rows = row_values.copy()
                    elif not np.array_equal(row_values, reference_rows):
                        raise GridAnalysisError(
                            f"test rows differ across seeds for {dataset}/{model}/"
                            f"S{subject}"
                        )
                    metrics = classification_metrics(
                        y_values,
                        probability_values,
                        n_classes=n_classes,
                        ece_bins=ece_bins,
                    )
                    aggregate = {
                        "dataset": str(dataset),
                        "model": model,
                        "subject": subject,
                        "seed": seed,
                        "folds_concatenated": len(folds),
                        "test_trials": len(row_values),
                        **metrics,
                    }
                    subject_seed_rows.append(aggregate)
                    one_subject_seed_rows.append(aggregate)
                subject_aggregate = {
                    "dataset": str(dataset),
                    "model": model,
                    "subject": subject,
                    "seeds_averaged": len(seeds),
                    "test_trials_per_seed": int(len(reference_rows)),
                    **_metric_means(one_subject_seed_rows),
                }
                subject_rows.append(subject_aggregate)
                dataset_subject_rows.append(subject_aggregate)

            dataset_aggregate = {
                "dataset": str(dataset),
                "model": model,
                "subjects_averaged": len(dataset_subject_rows),
                "aggregation": "mean_seeds_within_subject_then_mean_subjects",
                **_metric_means(dataset_subject_rows),
            }
            dataset_rows.append(dataset_aggregate)

            pooled_y = np.concatenate(dataset_y)
            pooled_probabilities = np.concatenate(dataset_probabilities)
            trial_micro_rows.append(
                {
                    "level": "dataset",
                    "dataset": str(dataset),
                    "model": model,
                    "prediction_count_including_seeds": len(pooled_y),
                    **classification_metrics(
                        pooled_y,
                        pooled_probabilities,
                        n_classes=n_classes,
                        ece_bins=ece_bins,
                    ),
                }
            )

    expected_ids = [job.job_id for job in jobs]
    observed_ids = [str(row["job_id"]) for row in job_rows]
    if len(job_rows) != len(jobs) or set(observed_ids) != set(expected_ids):
        raise GridAnalysisError("scored jobs differ from the exact plan job set")

    overall_rows: list[dict[str, Any]] = []
    for model in [str(value) for value in plan["architectures"]]:
        selected = [row for row in dataset_rows if row["model"] == model]
        if len(selected) != len(plan["dataset_order"]):
            raise GridAnalysisError(f"dataset aggregates are incomplete for {model}")
        overall_rows.append(
            {
                "model": model,
                "datasets_equal_weighted": len(selected),
                "aggregation": (
                    "fold_concatenation_then_seed_mean_then_subject_mean_"
                    "then_equal_dataset_mean"
                ),
                **_metric_means(selected),
            }
        )
        trial_micro_rows.append(
            {
                "level": "all_datasets",
                "dataset": "ALL_DATASETS",
                "model": model,
                "prediction_count_including_seeds": global_micro[model].count,
                **global_micro[model].metrics(),
            }
        )

    (
        tcformer_dataset_rows,
        tcformer_overall_rows,
        tcformer_seed_rows,
    ) = _paired_comparisons(
        plan=plan,
        subject_rows=subject_rows,
        subject_seed_rows=subject_seed_rows,
        overall_rows=overall_rows,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
        tcformer_name=tcformer_name,
    )

    # Re-hash every input used and close the plan/tree window before any output
    # becomes visible. The opening complete audit and the scoring-time
    # validate_completion calls have each already parsed every prediction.
    _verify_input_ledger(root, plan, ledger)
    if full_grid.load_plan(root) != plan:
        raise GridAnalysisError("immutable plan changed during analysis")
    _strict_record_tree(root, jobs)
    _require_quiescent_auxiliary_state(root)

    timing_rows = _timing_summary(job_rows)
    ledger_digest = hashlib.sha256(_canonical_bytes(ledger)).hexdigest()
    overall_index = {str(row["model"]): row for row in overall_rows}
    architecture_order = {
        str(model): index for index, model in enumerate(plan["architectures"])
    }
    ranked_models = sorted(
        overall_index,
        key=lambda model: (
            -float(overall_index[model]["balanced_accuracy"]),
            architecture_order[model],
        ),
    )
    ranking_rows = [
        {
            "rank": rank,
            "model": model,
            "metric": "equal_dataset_macro_balanced_accuracy",
            "value": float(overall_index[model]["balanced_accuracy"]),
            "tie_break": "frozen_architecture_order",
            "status": "descriptive_opened_development_suite_only",
        }
        for rank, model in enumerate(ranked_models, start=1)
    ]
    calibration_rows = [
        {
            "level": "dataset",
            "dataset": str(row["dataset"]),
            "model": str(row["model"]),
            "nll": row["nll"],
            "multiclass_brier": row["multiclass_brier"],
            "ece": row["ece"],
            "ece_bins": ece_bins,
        }
        for row in dataset_rows
    ] + [
        {
            "level": "equal_dataset_macro",
            "dataset": "ALL_DATASETS",
            "model": str(row["model"]),
            "nll": row["nll"],
            "multiclass_brier": row["multiclass_brier"],
            "ece": row["ece"],
            "ece_bins": ece_bins,
        }
        for row in overall_rows
    ]
    timing_overall = {
        str(row["model"]): row
        for row in timing_rows
        if row["dataset"] == "ALL_DATASETS"
    }
    complexity_rows = [
        {
            "model": model,
            "parameter_count_median": timing_overall[model][
                "parameter_count_median"
            ],
            "parameter_count_min": timing_overall[model]["parameter_count_min"],
            "parameter_count_max": timing_overall[model]["parameter_count_max"],
            "inference_ms_per_trial_median": timing_overall[model][
                "inference_ms_per_trial_median"
            ],
            "cuda_peak_memory_bytes_median": timing_overall[model][
                "cuda_peak_memory_bytes_median"
            ],
            "job_total_seconds_median": timing_overall[model][
                "job_total_seconds_median"
            ],
        }
        for model in [str(value) for value in plan["architectures"]]
    ]
    descriptive_leader = ranked_models[0]
    summary = {
        "schema": ANALYSIS_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "analysis_contract": analysis_contract,
        "analysis_contract_sha256": _analysis_contract_sha256(
            analysis_contract
        ),
        "input_ledger_sha256": ledger_digest,
        "evidence_scope": plan.get("evidence_scope"),
        "confirmation_evidence": False,
        "interpretation": (
            "exploratory results on opened development cohorts; not confirmation "
            "evidence, not a global/SOTA claim"
        ),
        "audit": copy.deepcopy(audit),
        "tcformer_comparator": tcformer_name,
        "descriptive_leader": descriptive_leader,
        "ranking_warning": (
            f"the leader is outcome-ranked from all "
            f"{int(plan['common_architecture_count'])} common models on these "
            "opened development cohorts; it is not a confirmatory or SOTA result"
        ),
        "primary_metric": {
            "name": "equal-dataset macro balanced accuracy",
            "descriptive_leader": descriptive_leader,
            "descriptive_leader_value": overall_index[descriptive_leader][
                "balanced_accuracy"
            ],
            "definition": (
                "concatenate disjoint folds within subject/seed; average seeds "
                "within subject, subjects within dataset, then equally weight datasets"
            ),
        },
        "chance_normalized_sensitivity": {
            "name": "equal-dataset chance-normalized balanced accuracy",
            "descriptive_leader": descriptive_leader,
            "descriptive_leader_value": overall_index[descriptive_leader][
                "chance_normalized_balanced_accuracy"
            ],
            "definition": "(BA - 1/K) / (1 - 1/K) within each dataset",
        },
        "analysis_settings": {
            "bootstrap_resamples": bootstrap_resamples,
            "bootstrap_seed": bootstrap_seed,
            "ece_bins": ece_bins,
            "tcformer_comparator": tcformer_name,
        },
        "metric_definitions": {
            "balanced_accuracy": "unadjusted mean recall over observed true classes",
            "chance_normalized_balanced_accuracy": (
                "(balanced_accuracy - 1/K) / (1 - 1/K), using each "
                "dataset's planned class count K"
            ),
            "macro_f1": "macro F1 over the plan class set; zero for undefined class F1",
            "ovr_macro_auroc": (
                "mean one-vs-rest class AUROC only when every class has both "
                "positive and negative examples"
            ),
            "nll": "mean negative log probability of the true class; clipped at 1e-15",
            "multiclass_brier": "mean sum of squared class-probability errors",
            "ece": f"top-label ECE with {ece_bins} equal-width confidence bins",
        },
        "trial_micro_warning": (
            "trial-micro rows pool repeated seed predictions and weight subjects/"
            "datasets by trial count; cross-dataset BA/F1/kappa/AUROC are undefined"
        ),
        "bootstrap": {
            "resamples": bootstrap_resamples,
            "seed": bootstrap_seed,
            "production_default_resamples": PRODUCTION_BOOTSTRAP_RESAMPLES,
            "unit": (
                "paired subject cluster after fold concatenation and seed averaging"
            ),
            "headline_fixed_suite": (
                "all planned datasets retained and equally weighted in every "
                "replicate; subjects resampled within dataset"
            ),
            "separate_dataset_superpopulation_sensitivity": (
                "datasets then subjects resampled; different superpopulation estimand"
            ),
        },
        "inference_policy": (
            "no null-hypothesis tests or model-selection decisions are produced; all "
            "model-minus-TCFormer bootstrap intervals are explicitly descriptive"
        ),
        "aggregate_results": {
            "overall_equal_dataset": overall_rows,
            "by_dataset": dataset_rows,
            "trial_micro": trial_micro_rows,
            "model_ranking": ranking_rows,
            "calibration": calibration_rows,
            "complexity": complexity_rows,
            "tcformer_dataset_context": tcformer_dataset_rows,
            "tcformer_overall_context": tcformer_overall_rows,
            "tcformer_seed_context": tcformer_seed_rows,
        },
        "table_row_counts": {},
    }
    tables = {
        "job_metrics": job_rows,
        "subject_seed_metrics": subject_seed_rows,
        "subject_metrics": subject_rows,
        "dataset_summary": dataset_rows,
        "overall_summary": overall_rows,
        "trial_micro_summary": trial_micro_rows,
        "resource_timing_summary": timing_rows,
        "model_ranking": ranking_rows,
        "calibration_summary": calibration_rows,
        "complexity_summary": complexity_rows,
        "tcformer_dataset_context": tcformer_dataset_rows,
        "tcformer_overall_context": tcformer_overall_rows,
        "tcformer_seed_context": tcformer_seed_rows,
        "input_checksum_ledger": ledger,
    }
    summary["table_row_counts"] = {
        name: len(rows) for name, rows in tables.items()
    }
    result = AnalysisResult(summary=summary, tables=tables, input_ledger=ledger)
    result._seal = _analysis_result_seal(summary, tables, ledger)
    return result


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        raise GridAnalysisError("refusing to create an empty CSV artifact")
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


def _format_percent(value: Any, digits: int = 2) -> str:
    number = _finite_float(value)
    return "N/A" if number is None else f"{100.0 * number:.{digits}f}%"


def _markdown_report(result: AnalysisResult) -> str:
    summary = result.summary
    overall = result.tables["overall_summary"]
    datasets = result.tables["dataset_summary"]
    overall_by_model = {str(row["model"]): row for row in overall}
    ranked = [str(row["model"]) for row in result.tables["model_ranking"]]
    lines = [
        "# Exact 43-model common-grid development analysis",
        "",
        "> Exploratory opened-development-cohort evidence only. These results are "
        "**not confirmation evidence and not a global/SOTA claim**.",
        "",
        f"- Plan SHA-256: `{summary['plan_sha256']}`",
        f"- Pre-result analysis contract SHA-256: "
        f"`{summary['analysis_contract_sha256'] or 'synthetic/legacy-unbound'}`",
        f"- Descriptive same-grid leader: `{summary['descriptive_leader']}`",
        f"- Prespecified context comparator: `{summary['tcformer_comparator']}`",
        f"- Bootstrap: {summary['bootstrap']['resamples']:,} deterministic resamples "
        f"(seed {summary['bootstrap']['seed']})",
        "",
        "## Equal-dataset macro results",
        "",
        "| Rank | Model | Accuracy | Balanced accuracy | Chance-normalized BA | Macro F1 | Kappa | AUROC | NLL | Brier | ECE |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, model in enumerate(ranked, start=1):
        row = overall_by_model[model]
        lines.append(
            f"| {rank} | `{model}` | {_format_percent(row['accuracy'])} | "
            f"{_format_percent(row['balanced_accuracy'])} | "
            f"{_format_percent(row['chance_normalized_balanced_accuracy'])} | "
            f"{_format_percent(row['macro_f1'])} | "
            f"{_format_percent(row['cohen_kappa'])} | "
            f"{_format_percent(row['ovr_macro_auroc'])} | "
            f"{float(row['nll']):.4f} | {float(row['multiclass_brier']):.4f} | "
            f"{_format_percent(row['ece'])} |"
        )

    lines.extend(
        [
            "",
            "Primary aggregation: concatenate disjoint folds within each "
            "subject/seed, average seeds within subject, average subjects within "
            "dataset, then give each dataset equal weight.",
            "",
            "## Descriptive leader and TCFormer by dataset",
            "",
            "| Dataset | Descriptive leader | TCFormer |",
            "|---|---:|---:|",
        ]
    )
    dataset_index = {
        (str(row["dataset"]), str(row["model"])): row for row in datasets
    }
    tcformer = str(summary["tcformer_comparator"])
    leader = str(summary["descriptive_leader"])
    dataset_order = list(
        dict.fromkeys(str(row["dataset"]) for row in datasets)
    )
    for dataset in dataset_order:
        lines.append(
            f"| `{dataset}` | "
            f"{_format_percent(dataset_index[(dataset, leader)]['balanced_accuracy'])} | "
            f"{_format_percent(dataset_index[(dataset, tcformer)]['balanced_accuracy'])} |"
        )

    lines.extend(
        [
            "",
            "## TCFormer reference context",
            "",
            "| Model | Effect (BA points) | Descriptive fixed-suite 95% interval |",
            "|---|---:|---:|",
        ]
    )
    for row in result.tables["tcformer_overall_context"]:
        effect = 100.0 * float(
            row["equal_dataset_macro_balanced_accuracy_difference"]
        )
        low = 100.0 * float(
            row["descriptive_fixed_suite_bootstrap_interval95_low"]
        )
        high = 100.0 * float(
            row["descriptive_fixed_suite_bootstrap_interval95_high"]
        )
        lines.append(
            f"| `{row['model']}` | {effect:+.2f} | "
            f"[{low:+.2f}, {high:+.2f}] |"
        )
    lines.extend(
        [
            "",
            "No p-values, null-hypothesis decisions, or model-selection claims are "
            "computed. Every interval is descriptive context for these opened "
            "development cohorts.",
            "",
            "## Calibration, complexity, and trial-micro outputs",
            "",
            "`trial_micro_summary.csv` is intentionally separate: it pools repeated "
            "seed predictions and weights cohorts by trial count. Cross-dataset "
            "balanced accuracy, macro F1, kappa, and AUROC are left undefined because "
            "class semantics differ. `calibration_summary.csv` freezes 15-bin ECE, "
            "NLL, and multiclass Brier outputs. `complexity_summary.csv` and "
            "`resource_timing_summary.csv` report parameters, selected epochs, "
            "fit/inference time, inference time per trial, and peak CUDA allocation.",
            "",
            "## Limitations",
            "",
            "- Only previously opened development cohorts are analyzed.",
            "- Architecture ranking occurred in the "
            "development evidence stream; independent sealed confirmation remains required.",
            "- The hierarchical bootstrap samples datasets and then paired subject "
            "clusters. It does not treat trials, folds, or random seeds as independent.",
            "- Runtime comparisons inherit shared-workstation load variation recorded "
            "during each job.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_new_file(path: Path, payload: bytes) -> None:
    full_grid._write_bytes_exclusive(path, payload)


def _write_new_file_at(descriptor: int, name: str, payload: bytes) -> None:
    full_grid._write_bytes_exclusive_at(descriptor, name, payload)


_TABLE_NAMES = frozenset(
    {
        "job_metrics",
        "subject_seed_metrics",
        "subject_metrics",
        "dataset_summary",
        "overall_summary",
        "trial_micro_summary",
        "resource_timing_summary",
        "model_ranking",
        "calibration_summary",
        "complexity_summary",
        "tcformer_dataset_context",
        "tcformer_overall_context",
        "tcformer_seed_context",
        "input_checksum_ledger",
    }
)
ANALYSIS_FILENAMES = frozenset(
    {
        "analysis.json",
        "RESULTS.md",
        "manifest.json",
        *(f"{name}.csv" for name in _TABLE_NAMES),
    }
)
_DIRECT_METRIC_COLUMNS = frozenset(METRIC_NAMES)
_AGGREGATE_METRIC_COLUMNS = frozenset(
    {
        column
        for metric in METRIC_NAMES
        for column in (metric, f"{metric}_defined_count")
    }
)
_TIMING_BASES = (
    "parameter_count",
    "selected_epoch_index",
    "selected_epoch_count",
    "selection_fit_seconds",
    "refit_fit_seconds",
    "test_inference_seconds",
    "job_total_seconds",
    "cuda_peak_memory_bytes",
    "inference_ms_per_trial",
)
_TABLE_SCHEMAS: dict[str, frozenset[str]] = {
    "job_metrics": frozenset(
        {
            "dataset",
            "model",
            "subject",
            "fold",
            "seed",
            "job_id",
            "test_count",
            "selected_epoch_index",
            "selected_epoch_count",
            "selection_epochs_run",
            "refit_epochs_run",
            "parameter_count",
            "selection_fit_seconds",
            "refit_fit_seconds",
            "test_inference_seconds",
            "job_total_seconds",
            "cuda_peak_memory_bytes",
            "inference_ms_per_trial",
        }
    )
    | _DIRECT_METRIC_COLUMNS,
    "subject_seed_metrics": frozenset(
        {
            "dataset",
            "model",
            "subject",
            "seed",
            "folds_concatenated",
            "test_trials",
        }
    )
    | _DIRECT_METRIC_COLUMNS,
    "subject_metrics": frozenset(
        {
            "dataset",
            "model",
            "subject",
            "seeds_averaged",
            "test_trials_per_seed",
        }
    )
    | _AGGREGATE_METRIC_COLUMNS,
    "dataset_summary": frozenset(
        {"dataset", "model", "subjects_averaged", "aggregation"}
    )
    | _AGGREGATE_METRIC_COLUMNS,
    "overall_summary": frozenset(
        {"model", "datasets_equal_weighted", "aggregation"}
    )
    | _AGGREGATE_METRIC_COLUMNS,
    "trial_micro_summary": frozenset(
        {"level", "dataset", "model", "prediction_count_including_seeds"}
    )
    | _DIRECT_METRIC_COLUMNS,
    "resource_timing_summary": frozenset({"dataset", "model", "jobs"})
    | frozenset(
        f"{base}_{suffix}"
        for base in _TIMING_BASES
        for suffix in ("mean", "median", "p95", "min", "max")
    ),
    "model_ranking": frozenset(
        {"rank", "model", "metric", "value", "tie_break", "status"}
    ),
    "calibration_summary": frozenset(
        {
            "level",
            "dataset",
            "model",
            "nll",
            "multiclass_brier",
            "ece",
            "ece_bins",
        }
    ),
    "complexity_summary": frozenset(
        {
            "model",
            "parameter_count_median",
            "parameter_count_min",
            "parameter_count_max",
            "inference_ms_per_trial_median",
            "cuda_peak_memory_bytes_median",
            "job_total_seconds_median",
        }
    ),
    "tcformer_dataset_context": frozenset(
        {
            "model",
            "comparator",
            "dataset",
            "subjects",
            "mean_balanced_accuracy_difference",
            "median_balanced_accuracy_difference",
            "descriptive_bootstrap_interval95_low",
            "descriptive_bootstrap_interval95_high",
            "bootstrap_resamples",
            "bootstrap_seed",
            "status",
        }
    ),
    "tcformer_overall_context": frozenset(
        {
            "model",
            "comparator",
            "datasets",
            "subjects_total",
            "equal_dataset_macro_balanced_accuracy_difference",
            "descriptive_fixed_suite_bootstrap_interval95_low",
            "descriptive_fixed_suite_bootstrap_interval95_high",
            "fixed_suite_bootstrap_resamples",
            "fixed_suite_bootstrap_seed",
            "descriptive_dataset_superpopulation_interval95_low",
            "descriptive_dataset_superpopulation_interval95_high",
            "dataset_superpopulation_bootstrap_resamples",
            "dataset_superpopulation_bootstrap_seed",
            "status",
        }
    ),
    "tcformer_seed_context": frozenset(
        {
            "model",
            "comparator",
            "seed",
            "dataset",
            "subjects",
            "balanced_accuracy_difference",
        }
    ),
    "input_checksum_ledger": frozenset(
        {
            "job_id",
            "completion_sha256",
            "record_sha256",
            "predictions_sha256",
        }
    ),
}
_STRING_TABLE_FIELDS = frozenset(
    {
        "dataset",
        "model",
        "job_id",
        "aggregation",
        "level",
        "metric",
        "tie_break",
        "status",
        "comparator",
        "completion_sha256",
        "record_sha256",
        "predictions_sha256",
    }
)
_SHA256_TABLE_FIELDS = frozenset(
    {
        "completion_sha256",
        "record_sha256",
        "predictions_sha256",
    }
)
_NONNEGATIVE_INTEGER_TABLE_FIELDS = frozenset(
    {
        "fold",
        "seed",
        "selected_epoch_index",
        "cuda_peak_memory_bytes",
        "bootstrap_seed",
        "fixed_suite_bootstrap_seed",
        "dataset_superpopulation_bootstrap_seed",
    }
)
_POSITIVE_INTEGER_TABLE_FIELDS = frozenset(
    {
        "subject",
        "test_count",
        "selected_epoch_count",
        "selection_epochs_run",
        "refit_epochs_run",
        "parameter_count",
        "folds_concatenated",
        "test_trials",
        "seeds_averaged",
        "test_trials_per_seed",
        "subjects_averaged",
        "datasets_equal_weighted",
        "prediction_count_including_seeds",
        "jobs",
        "rank",
        "ece_bins",
        "subjects",
        "bootstrap_resamples",
        "datasets",
        "subjects_total",
        "fixed_suite_bootstrap_resamples",
        "dataset_superpopulation_bootstrap_resamples",
    }
)
_DEFINED_COUNT_TABLE_FIELDS = frozenset(
    f"{metric}_defined_count" for metric in METRIC_NAMES
)
_OPTIONAL_FLOAT_TABLE_FIELDS = frozenset(METRIC_NAMES)
_NONNEGATIVE_FLOAT_TABLE_FIELDS = frozenset(
    {
        "selection_fit_seconds",
        "refit_fit_seconds",
        "test_inference_seconds",
        "job_total_seconds",
        "inference_ms_per_trial",
        "nll",
        "multiclass_brier",
        "ece",
        "parameter_count_median",
        "parameter_count_min",
        "parameter_count_max",
        "inference_ms_per_trial_median",
        "cuda_peak_memory_bytes_median",
        "job_total_seconds_median",
    }
    | {
        f"{base}_{suffix}"
        for base in _TIMING_BASES
        for suffix in ("mean", "median", "p95", "min", "max")
    }
)
_UNIT_INTERVAL_FLOAT_TABLE_FIELDS = frozenset(
    {
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "ovr_macro_auroc",
        "ece",
        "value",
    }
)
_SIGNED_UNIT_FLOAT_TABLE_FIELDS = frozenset(
    {
        "chance_normalized_balanced_accuracy",
        "cohen_kappa",
        "mean_balanced_accuracy_difference",
        "median_balanced_accuracy_difference",
        "descriptive_bootstrap_interval95_low",
        "descriptive_bootstrap_interval95_high",
        "equal_dataset_macro_balanced_accuracy_difference",
        "descriptive_fixed_suite_bootstrap_interval95_low",
        "descriptive_fixed_suite_bootstrap_interval95_high",
        "descriptive_dataset_superpopulation_interval95_low",
        "descriptive_dataset_superpopulation_interval95_high",
        "balanced_accuracy_difference",
    }
)
_FLOAT_TABLE_FIELDS = (
    _NONNEGATIVE_FLOAT_TABLE_FIELDS
    | _UNIT_INTERVAL_FLOAT_TABLE_FIELDS
    | _SIGNED_UNIT_FLOAT_TABLE_FIELDS
)
_INTEGER_TABLE_FIELDS = (
    _NONNEGATIVE_INTEGER_TABLE_FIELDS
    | _POSITIVE_INTEGER_TABLE_FIELDS
    | _DEFINED_COUNT_TABLE_FIELDS
)
_KNOWN_TABLE_FIELDS = (
    _STRING_TABLE_FIELDS
    | _INTEGER_TABLE_FIELDS
    | _OPTIONAL_FLOAT_TABLE_FIELDS
    | _FLOAT_TABLE_FIELDS
)
if set().union(*_TABLE_SCHEMAS.values()) != _KNOWN_TABLE_FIELDS:
    raise RuntimeError("analysis table field contracts are incomplete")
_SUMMARY_KEYS = frozenset(
    {
        "schema",
        "plan_sha256",
        "analysis_contract",
        "analysis_contract_sha256",
        "input_ledger_sha256",
        "evidence_scope",
        "confirmation_evidence",
        "interpretation",
        "audit",
        "tcformer_comparator",
        "descriptive_leader",
        "ranking_warning",
        "primary_metric",
        "chance_normalized_sensitivity",
        "analysis_settings",
        "metric_definitions",
        "trial_micro_warning",
        "bootstrap",
        "inference_policy",
        "aggregate_results",
        "table_row_counts",
    }
)
_FORBIDDEN_ARTIFACT_KEYS = frozenset(
    {
        "label",
        "labels",
        "raw_label",
        "raw_labels",
        "target",
        "targets",
        "y",
        "y_true",
        "probability",
        "probabilities",
        "row",
        "rows",
        "test_rows",
    }
)


def _reject_forbidden_artifact_keys(value: Any, *, path: str = "result") -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).lower()
            if key in _FORBIDDEN_ARTIFACT_KEYS:
                raise GridAnalysisError(
                    f"analysis artifact contains forbidden key {path}.{raw_key}"
                )
            _reject_forbidden_artifact_keys(child, path=f"{path}.{raw_key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_forbidden_artifact_keys(child, path=f"{path}[{index}]")


def _validate_scalar_table(
    name: str,
    rows: Any,
) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or not rows:
        raise GridAnalysisError(f"analysis table {name} must be a nonempty list")
    expected_schema = _TABLE_SCHEMAS[name]
    for index, raw_row in enumerate(rows):
        if not isinstance(raw_row, dict) or set(raw_row) != expected_schema:
            raise GridAnalysisError(
                f"analysis table {name} row {index} violates its exact schema"
            )
        for key, value in raw_row.items():
            field = f"analysis table {name} row {index} field {key}"
            if isinstance(value, bool):
                raise GridAnalysisError(f"{field} must not be boolean")
            if key in _STRING_TABLE_FIELDS:
                if type(value) is not str or not value or "\x00" in value:
                    raise GridAnalysisError(f"{field} is not a valid string")
                if (
                    key in _SHA256_TABLE_FIELDS
                    and full_grid.HEX_64_RE.fullmatch(value) is None
                ):
                    raise GridAnalysisError(f"{field} is not a SHA-256 digest")
                continue
            if key in _INTEGER_TABLE_FIELDS:
                if type(value) is not int:
                    raise GridAnalysisError(f"{field} is not an exact integer")
                if (
                    key in _POSITIVE_INTEGER_TABLE_FIELDS
                    and value <= 0
                ) or (
                    key
                    in (
                        _NONNEGATIVE_INTEGER_TABLE_FIELDS
                        | _DEFINED_COUNT_TABLE_FIELDS
                    )
                    and value < 0
                ):
                    raise GridAnalysisError(f"{field} is out of range")
                if key == "ece_bins" and value < 2:
                    raise GridAnalysisError(f"{field} is out of range")
                continue
            if key in _OPTIONAL_FLOAT_TABLE_FIELDS and value is None:
                continue
            if key in _FLOAT_TABLE_FIELDS or key in _OPTIONAL_FLOAT_TABLE_FIELDS:
                if type(value) is not float or not math.isfinite(value):
                    raise GridAnalysisError(f"{field} is not a finite exact float")
                if (
                    key in _NONNEGATIVE_FLOAT_TABLE_FIELDS
                    and value < 0.0
                ) or (
                    key in _UNIT_INTERVAL_FLOAT_TABLE_FIELDS
                    and not 0.0 <= value <= 1.0
                ) or (
                    key in _SIGNED_UNIT_FLOAT_TABLE_FIELDS
                    and not -1.0 <= value <= 1.0
                ) or (
                    key == "multiclass_brier"
                    and value > 2.0
                ):
                    raise GridAnalysisError(f"{field} is out of range")
                continue
            if value is not None:
                raise GridAnalysisError(
                    f"{field} has no exact publication contract"
                )
        denominator = {
            "subject_metrics": raw_row.get("seeds_averaged"),
            "dataset_summary": raw_row.get("subjects_averaged"),
            "overall_summary": raw_row.get("datasets_equal_weighted"),
        }.get(name)
        if denominator is not None and any(
            raw_row[field] > denominator
            for field in _DEFINED_COUNT_TABLE_FIELDS
        ):
            raise GridAnalysisError(
                f"analysis table {name} row {index} has an impossible defined count"
            )
    return rows


def _validate_result_for_publication(
    result: AnalysisResult,
    *,
    run_root: Path,
) -> tuple[dict[str, Any], tuple[full_grid.Job, ...], dict[str, Any]]:
    """Re-audit the run and reject fabricated, mutated, or leaky artifacts."""

    if not isinstance(result, AnalysisResult):
        raise GridAnalysisError("publish_analysis requires an AnalysisResult")
    expected_seal = _analysis_result_seal(
        result.summary,
        result.tables,
        result.input_ledger,
    )
    if not hmac.compare_digest(result._seal, expected_seal):
        raise GridAnalysisError("analysis result seal is absent or invalid")
    if set(result.tables) != _TABLE_NAMES:
        raise GridAnalysisError("analysis tables differ from the exact whitelist")
    validated_tables = {
        name: _validate_scalar_table(name, result.tables[name])
        for name in sorted(_TABLE_NAMES)
    }
    if not isinstance(result.summary, dict) or set(result.summary) != _SUMMARY_KEYS:
        raise GridAnalysisError("analysis summary violates its exact schema")
    _reject_forbidden_artifact_keys(result.summary)
    _reject_forbidden_artifact_keys(validated_tables)

    # Fresh complete audit is intentionally inside the publication boundary.
    plan, jobs, audit = audit_exact_grid(run_root)
    if (
        plan.get("publication_mode") != full_grid.FORMAL_PUBLICATION_MODE
        or tuple(plan.get("architectures", ()))
        != full_grid.COMMON_ARCHITECTURES
        or int(plan.get("n_jobs", -1)) != full_grid.FORMAL_EXPECTED_JOBS
    ):
        raise GridAnalysisError("test-only or non-common plans cannot be published")
    settings = result.summary["analysis_settings"]
    if not isinstance(settings, Mapping):
        raise GridAnalysisError("analysis settings are invalid")
    try:
        frozen_contract = _verify_frozen_analysis_contract(
            plan,
            bootstrap_resamples=int(settings["bootstrap_resamples"]),
            bootstrap_seed=int(settings["bootstrap_seed"]),
            ece_bins=int(settings["ece_bins"]),
            tcformer_name=str(settings["tcformer_comparator"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise GridAnalysisError("analysis settings are invalid") from error
    if (
        result.summary["schema"] != ANALYSIS_SCHEMA
        or result.summary["plan_sha256"] != plan["plan_sha256"]
        or result.summary["audit"] != audit
        or result.summary["confirmation_evidence"] is not False
    ):
        raise GridAnalysisError("analysis summary is not bound to the fresh run audit")
    if (
        result.summary["analysis_contract"] != frozen_contract
        or result.summary["analysis_contract_sha256"]
        != _analysis_contract_sha256(frozen_contract)
    ):
        raise GridAnalysisError(
            "analysis summary differs from the pre-result analysis contract"
        )
    expected_job_ids = [job.job_id for job in jobs]
    job_rows = validated_tables["job_metrics"]
    ledger_rows = validated_tables["input_checksum_ledger"]
    if (
        [row["job_id"] for row in job_rows] != expected_job_ids
        or [row["job_id"] for row in ledger_rows] != expected_job_ids
        or result.input_ledger != ledger_rows
    ):
        raise GridAnalysisError("job metrics/checksum ledger differ from plan order")
    for job, row in zip(jobs, job_rows, strict=True):
        expected_identity = job.identity()
        if any(row[key] != value for key, value in expected_identity.items()):
            raise GridAnalysisError(f"job metric identity differs for {job.job_id}")
        split = full_grid._planned_split_identity(plan, job)
        if int(row["test_count"]) != int(split["partitions"]["test"]["count"]):
            raise GridAnalysisError(f"job test count differs for {job.job_id}")
        if int(row["selected_epoch_count"]) != int(row["selected_epoch_index"]) + 1:
            raise GridAnalysisError(f"selected epoch fields differ for {job.job_id}")

    architectures = [str(value) for value in plan["architectures"]]
    seeds = [int(value) for value in plan["seeds"]]
    expected_subject_seed = [
        (str(dataset), model, int(subject), seed)
        for dataset in plan["dataset_order"]
        for model in architectures
        for subject in plan["datasets"][dataset]["subjects"]
        for seed in seeds
    ]
    observed_subject_seed = [
        (
            str(row["dataset"]),
            str(row["model"]),
            int(row["subject"]),
            int(row["seed"]),
        )
        for row in validated_tables["subject_seed_metrics"]
    ]
    if observed_subject_seed != expected_subject_seed:
        raise GridAnalysisError("subject-seed table differs from the plan product")

    expected_subject = [
        (str(dataset), model, int(subject))
        for dataset in plan["dataset_order"]
        for model in architectures
        for subject in plan["datasets"][dataset]["subjects"]
    ]
    observed_subject = [
        (str(row["dataset"]), str(row["model"]), int(row["subject"]))
        for row in validated_tables["subject_metrics"]
    ]
    if observed_subject != expected_subject:
        raise GridAnalysisError("subject table differs from the plan product")

    expected_dataset = [
        (str(dataset), model)
        for dataset in plan["dataset_order"]
        for model in architectures
    ]
    observed_dataset = [
        (str(row["dataset"]), str(row["model"]))
        for row in validated_tables["dataset_summary"]
    ]
    if observed_dataset != expected_dataset:
        raise GridAnalysisError("dataset table differs from the plan product")
    if [str(row["model"]) for row in validated_tables["overall_summary"]] != architectures:
        raise GridAnalysisError("overall table differs from the architecture roster")
    ranking = validated_tables["model_ranking"]
    if (
        [int(row["rank"]) for row in ranking] != list(range(1, len(architectures) + 1))
        or {str(row["model"]) for row in ranking} != set(architectures)
        or any(
            row["metric"] != "equal_dataset_macro_balanced_accuracy"
            or row["tie_break"] != "frozen_architecture_order"
            or row["status"] != "descriptive_opened_development_suite_only"
            for row in ranking
        )
    ):
        raise GridAnalysisError("descriptive ranking table is invalid")
    if [str(row["model"]) for row in validated_tables["complexity_summary"]] != architectures:
        raise GridAnalysisError("complexity table differs from the roster")
    expected_trial_micro = [
        ("dataset", str(dataset), model)
        for dataset in plan["dataset_order"]
        for model in architectures
    ] + [("all_datasets", "ALL_DATASETS", model) for model in architectures]
    observed_trial_micro = [
        (str(row["level"]), str(row["dataset"]), str(row["model"]))
        for row in validated_tables["trial_micro_summary"]
    ]
    if observed_trial_micro != expected_trial_micro:
        raise GridAnalysisError("trial-micro table differs from the plan product")
    expected_calibration = [
        ("dataset", str(dataset), model)
        for dataset in plan["dataset_order"]
        for model in architectures
    ] + [
        ("equal_dataset_macro", "ALL_DATASETS", model)
        for model in architectures
    ]
    observed_calibration = [
        (str(row["level"]), str(row["dataset"]), str(row["model"]))
        for row in validated_tables["calibration_summary"]
    ]
    if (
        observed_calibration != expected_calibration
        or any(int(row["ece_bins"]) != full_grid.FORMAL_ANALYSIS_ECE_BINS for row in validated_tables["calibration_summary"])
    ):
        raise GridAnalysisError("calibration table differs from the frozen contract")
    compared = [model for model in architectures if model != "tcformer"]
    expected_tc_dataset = [
        (model, str(dataset))
        for model in compared
        for dataset in plan["dataset_order"]
    ]
    observed_tc_dataset = [
        (str(row["model"]), str(row["dataset"]))
        for row in validated_tables["tcformer_dataset_context"]
    ]
    if observed_tc_dataset != expected_tc_dataset:
        raise GridAnalysisError("TCFormer dataset context differs from the plan")
    if [
        str(row["model"]) for row in validated_tables["tcformer_overall_context"]
    ] != compared:
        raise GridAnalysisError("TCFormer overall context differs from the plan")
    expected_tc_seed = [
        (model, seed, str(dataset))
        for model in compared
        for seed in seeds
        for dataset in (*plan["dataset_order"], "EQUAL_DATASET_OVERALL")
    ]
    observed_tc_seed = [
        (str(row["model"]), int(row["seed"]), str(row["dataset"]))
        for row in validated_tables["tcformer_seed_context"]
    ]
    if observed_tc_seed != expected_tc_seed:
        raise GridAnalysisError("TCFormer seed context differs from the plan")

    for table_name in (
        "job_metrics",
        "subject_seed_metrics",
        "subject_metrics",
        "dataset_summary",
        "overall_summary",
        "trial_micro_summary",
    ):
        for row in validated_tables[table_name]:
            for metric in METRIC_NAMES:
                value = row.get(metric)
                if value is None:
                    continue
                number = float(value)
                if metric in {
                    "accuracy",
                    "balanced_accuracy",
                    "macro_f1",
                    "ovr_macro_auroc",
                    "ece",
                } and not 0.0 <= number <= 1.0:
                    raise GridAnalysisError(
                        f"{table_name} contains out-of-range {metric}"
                    )
                if metric in {"cohen_kappa", "chance_normalized_balanced_accuracy"}:
                    if not -1.0 <= number <= 1.0:
                        raise GridAnalysisError(
                            f"{table_name} contains out-of-range {metric}"
                        )
                if metric in {"nll", "multiclass_brier"} and number < 0.0:
                    raise GridAnalysisError(
                        f"{table_name} contains negative {metric}"
                    )
            dataset = str(row.get("dataset", "ALL_DATASETS"))
            if dataset in plan["datasets"]:
                ba = row.get("balanced_accuracy")
                normalized = row.get("chance_normalized_balanced_accuracy")
                if ba is not None and normalized is not None:
                    classes = int(plan["datasets"][dataset]["n_classes"])
                    chance = 1.0 / classes
                    expected_normalized = (float(ba) - chance) / (1.0 - chance)
                    if not math.isclose(
                        float(normalized),
                        expected_normalized,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    ):
                        raise GridAnalysisError(
                            f"{table_name} has inconsistent chance-normalized BA"
                        )

    if result.summary["input_ledger_sha256"] != hashlib.sha256(
        _canonical_bytes(ledger_rows)
    ).hexdigest():
        raise GridAnalysisError("analysis checksum-ledger digest is invalid")
    _verify_input_ledger(run_root, plan, ledger_rows)

    expected_counts = {
        name: len(rows) for name, rows in validated_tables.items()
    }
    if result.summary["table_row_counts"] != expected_counts:
        raise GridAnalysisError("analysis table counts are invalid")
    aggregate = result.summary.get("aggregate_results")
    expected_aggregate = {
        "overall_equal_dataset": validated_tables["overall_summary"],
        "by_dataset": validated_tables["dataset_summary"],
        "trial_micro": validated_tables["trial_micro_summary"],
        "model_ranking": validated_tables["model_ranking"],
        "calibration": validated_tables["calibration_summary"],
        "complexity": validated_tables["complexity_summary"],
        "tcformer_dataset_context": validated_tables["tcformer_dataset_context"],
        "tcformer_overall_context": validated_tables["tcformer_overall_context"],
        "tcformer_seed_context": validated_tables["tcformer_seed_context"],
    }
    if aggregate != expected_aggregate:
        raise GridAnalysisError("summary aggregate results differ from CSV tables")
    overall_by_model = {
        str(row["model"]): row for row in validated_tables["overall_summary"]
    }
    leader = str(result.summary["descriptive_leader"])
    if (
        leader != str(validated_tables["model_ranking"][0]["model"])
        or result.summary["primary_metric"]["descriptive_leader"] != leader
        or result.summary["primary_metric"]["descriptive_leader_value"]
        != overall_by_model[leader]["balanced_accuracy"]
        or result.summary["chance_normalized_sensitivity"]["descriptive_leader"]
        != leader
        or result.summary["chance_normalized_sensitivity"][
            "descriptive_leader_value"
        ]
        != overall_by_model[leader]["chance_normalized_balanced_accuracy"]
    ):
        raise GridAnalysisError("headline values differ from the overall table")
    return plan, jobs, audit


def _safe_output_destination(
    output_dir: str | Path,
    *,
    run_root: Path,
) -> Path:
    destination = full_grid._absolute_path(Path(output_dir))
    if destination == run_root or run_root in destination.parents:
        raise GridAnalysisError("analysis output must be outside the run root")
    if destination.name in {"", ".", ".."}:
        raise GridAnalysisError("analysis output name is invalid")
    parent = destination.parent
    try:
        descriptor = full_grid._open_absolute_directory(parent)
    except (FileNotFoundError, OSError, full_grid.FullGridError) as error:
        raise GridAnalysisError(
            "analysis output parent must already be a real directory"
        ) from error
    try:
        try:
            os.stat(
                destination.name,
                dir_fd=descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(destination)
        full_grid._assert_directory_descriptor_path(parent, descriptor)
    finally:
        os.close(descriptor)
    return destination


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    try:
        full_grid._atomic_rename_noreplace(source, destination)
    except full_grid.FullGridError as error:
        raise GridAnalysisError(str(error)) from error


@contextmanager
def _output_publication_lock(destination: Path) -> Iterator[tuple[int, int]]:
    parent_descriptor = full_grid._open_absolute_directory(destination.parent)
    fence_name = f".{destination.name}.analysis-publish.fence"
    descriptor: int | None = None
    try:
        try:
            full_grid._write_bytes_exclusive_at(
                parent_descriptor,
                fence_name,
                b"eeg-mi-common-analysis-output-fence-v1\n",
            )
        except FileExistsError:
            pass
        descriptor = os.open(
                fence_name,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0),
                dir_fd=parent_descriptor,
            )
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        before = os.fstat(descriptor)
        after = os.stat(
            fence_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_mode & 0o222
            or before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
        ):
            raise GridAnalysisError("output publication fence changed")
        full_grid._assert_directory_descriptor_path(
            destination.parent,
            parent_descriptor,
        )
        yield descriptor, parent_descriptor
        final = os.stat(
            fence_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (final.st_dev, final.st_ino) != (before.st_dev, before.st_ino):
            raise GridAnalysisError("output publication fence was replaced")
        full_grid._assert_directory_descriptor_path(
            destination.parent,
            parent_descriptor,
        )
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_descriptor)


def _read_exact_readonly_file_at(
    directory_descriptor: int,
    name: str,
) -> bytes:
    """Read one immutable leaf while binding its descriptor to its directory name."""

    if (
        not name
        or name in {".", ".."}
        or os.sep in name
        or (os.altsep is not None and os.altsep in name)
    ):
        raise GridAnalysisError("analysis artifact leaf name is invalid")
    try:
        path_stat = os.stat(
            name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
            dir_fd=directory_descriptor,
        )
    except OSError as error:
        raise GridAnalysisError(f"analysis artifact {name} is unsafe") from error
    try:
        before = os.fstat(descriptor)
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_mode & 0o222
            or any(
                getattr(before, field) != getattr(path_stat, field)
                for field in stable_fields
            )
        ):
            raise GridAnalysisError(
                f"analysis artifact {name} is not one unique read-only file"
            )
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
        final_path = os.stat(
            name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if any(
            getattr(before, field) != getattr(observed, field)
            for observed in (after, final_path)
            for field in stable_fields
        ):
            raise GridAnalysisError(f"analysis artifact {name} changed during read")
        payload = b"".join(chunks)
        if len(payload) != before.st_size:
            raise GridAnalysisError(f"analysis artifact {name} read was incomplete")
        return payload
    finally:
        os.close(descriptor)


def _validate_published_analysis_tree_at(
    *,
    output_parent: int,
    destination: Path,
    staged_stat: os.stat_result,
    expected_artifacts: Mapping[str, bytes],
    expected_manifest: Mapping[str, Any],
) -> None:
    """Re-open and byte-verify the exact final tree after its atomic rename."""

    try:
        path_stat = os.stat(
            destination.name,
            dir_fd=output_parent,
            follow_symlinks=False,
        )
        descriptor = os.open(
            destination.name,
            full_grid._directory_open_flags(),
            dir_fd=output_parent,
        )
    except OSError as error:
        raise GridAnalysisError("published analysis directory is unsafe") from error
    try:
        final_stat = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(final_stat.st_mode)
            or final_stat.st_mode & 0o222
            or (final_stat.st_dev, final_stat.st_ino)
            != (staged_stat.st_dev, staged_stat.st_ino)
            or (path_stat.st_dev, path_stat.st_ino)
            != (final_stat.st_dev, final_stat.st_ino)
        ):
            raise GridAnalysisError(
                "published analysis directory differs from the sealed stage"
            )
        observed_names = set(os.listdir(descriptor))
        if (
            observed_names != set(expected_artifacts)
            or observed_names != set(ANALYSIS_FILENAMES)
        ):
            raise GridAnalysisError(
                "published analysis directory has a non-exact artifact set"
            )
        for name, expected in sorted(expected_artifacts.items()):
            observed = _read_exact_readonly_file_at(descriptor, name)
            if not hmac.compare_digest(observed, expected):
                raise GridAnalysisError(
                    f"published analysis artifact {name} differs from staged bytes"
                )
        manifest_payload = expected_artifacts["manifest.json"]
        try:
            parsed_manifest = json.loads(manifest_payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise GridAnalysisError("analysis manifest is not canonical JSON") from error
        if (
            parsed_manifest != expected_manifest
            or _canonical_bytes(parsed_manifest) + b"\n" != manifest_payload
        ):
            raise GridAnalysisError("analysis manifest is not exact canonical JSON")
        try:
            full_grid._assert_directory_descriptor_path(
                destination,
                descriptor,
            )
        except (OSError, full_grid.FullGridError) as error:
            raise GridAnalysisError(
                "published analysis directory path changed"
            ) from error
    finally:
        os.close(descriptor)


def _invalidate_published_analysis_at(
    *,
    output_parent: int,
    destination: Path,
    published_identity: tuple[int, int],
) -> str:
    """Atomically remove one known failed publication from its canonical name."""

    try:
        descriptor = os.open(
            destination.name,
            full_grid._directory_open_flags(),
            dir_fd=output_parent,
        )
    except OSError as error:
        raise GridAnalysisError(
            "failed analysis publication could not be opened for invalidation"
        ) from error
    invalid_name = (
        f".{destination.name}.analysis-invalid-{uuid.uuid4().hex}"
    )
    try:
        observed = os.fstat(descriptor)
        path_stat = os.stat(
            destination.name,
            dir_fd=output_parent,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(observed.st_mode)
            or (observed.st_dev, observed.st_ino) != published_identity
            or (path_stat.st_dev, path_stat.st_ino) != published_identity
        ):
            raise GridAnalysisError(
                "failed analysis publication could not be identified "
                "for invalidation"
            )
        if stat.S_IMODE(observed.st_mode) == 0o700:
            if frozenset(os.listdir(descriptor)) != ANALYSIS_FILENAMES:
                raise GridAnalysisError(
                    "writable failed analysis has a non-exact artifact set"
                )
            for child_name in sorted(ANALYSIS_FILENAMES):
                child = os.stat(
                    child_name,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
                if (
                    not stat.S_ISREG(child.st_mode)
                    or child.st_nlink != 1
                    or stat.S_IMODE(child.st_mode) != 0o444
                ):
                    raise GridAnalysisError(
                        "writable failed analysis has an unsafe artifact"
                    )
            os.fchmod(descriptor, 0o555)
            os.fsync(descriptor)
        elif stat.S_IMODE(observed.st_mode) != 0o555:
            raise GridAnalysisError(
                "failed analysis publication has an unknown directory mode"
            )
        try:
            full_grid._publish_sealed_directory_noreplace_at(
                output_parent,
                destination.name,
                descriptor,
                output_parent,
                invalid_name,
                expected_names=ANALYSIS_FILENAMES,
            )
        except BaseException:
            try:
                os.stat(
                    destination.name,
                    dir_fd=output_parent,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                invalid = os.stat(
                    invalid_name,
                    dir_fd=output_parent,
                    follow_symlinks=False,
                )
                held = os.fstat(descriptor)
                if (
                    (invalid.st_dev, invalid.st_ino)
                    == (held.st_dev, held.st_ino)
                    == published_identity
                    and stat.S_IMODE(held.st_mode) == 0o555
                ):
                    os.fsync(output_parent)
                    return invalid_name
            raise
        os.fsync(output_parent)
        return invalid_name
    finally:
        os.close(descriptor)


def _compute_and_publish_analysis(
    *,
    run_root: str | Path,
    cache_root: str | Path,
    output_dir: str | Path,
) -> tuple[AnalysisResult, Path]:
    """Compute only under the exclusive run fence, then publish atomically."""

    root = full_grid._safe_run_root(Path(run_root))
    cache = full_grid._absolute_path(Path(cache_root))
    destination = _safe_output_destination(output_dir, run_root=root)
    stage_name = f".{destination.name}.partial-{uuid.uuid4().hex}"
    recovery_parent = full_grid._open_absolute_directory(destination.parent)
    published_identity: tuple[int, int] | None = None
    completed: tuple[AnalysisResult, Path] | None = None
    try:
        with _output_publication_lock(destination) as (
            output_lock,
            output_parent,
        ):
            recovery_stat = os.fstat(recovery_parent)
            output_parent_stat = os.fstat(output_parent)
            if (
                recovery_stat.st_dev != output_parent_stat.st_dev
                or recovery_stat.st_ino != output_parent_stat.st_ino
            ):
                raise GridAnalysisError("analysis output parent changed")
            try:
                os.stat(
                    destination.name,
                    dir_fd=output_parent,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                pass
            else:
                raise FileExistsError(destination)
            with full_grid.publication_fence(
                root,
                exclusive=True,
                blocking=True,
            ) as run_fence:
                plan = full_grid.load_plan(root)
                full_grid.verify_runtime_identity(plan, cache_root=cache)
                fresh = compute_analysis(
                    run_root=root,
                    cache_root=cache,
                    bootstrap_resamples=PRODUCTION_BOOTSTRAP_RESAMPLES,
                    bootstrap_seed=DEFAULT_BOOTSTRAP_SEED,
                    ece_bins=DEFAULT_ECE_BINS,
                    tcformer_name=full_grid.FORMAL_ANALYSIS_TCFORMER_COMPARATOR,
                )
                validated_plan, validated_jobs, _ = _validate_result_for_publication(
                    fresh,
                    run_root=root,
                )
                os.mkdir(stage_name, 0o700, dir_fd=output_parent)
                os.fsync(output_parent)
                stage_descriptor = os.open(
                    stage_name,
                    full_grid._directory_open_flags(),
                    dir_fd=output_parent,
                )
                initial_stage_stat = os.fstat(stage_descriptor)
                published = False
                written_identities: dict[str, tuple[int, int]] = {}
                try:
                    artifacts: dict[str, bytes] = {
                        "analysis.json": _canonical_bytes(fresh.summary) + b"\n",
                        "RESULTS.md": _markdown_report(fresh).encode("utf-8"),
                    }
                    for name, rows in fresh.tables.items():
                        artifacts[f"{name}.csv"] = _csv_bytes(rows)
                    for name, payload in sorted(artifacts.items()):
                        _write_new_file_at(stage_descriptor, name, payload)
                        written = os.stat(
                            name,
                            dir_fd=stage_descriptor,
                            follow_symlinks=False,
                        )
                        written_identities[name] = (
                            written.st_dev,
                            written.st_ino,
                        )
                    manifest = {
                        "schema": MANIFEST_SCHEMA,
                        "plan_sha256": fresh.summary["plan_sha256"],
                        "analysis_contract_sha256": fresh.summary[
                            "analysis_contract_sha256"
                        ],
                        "analysis_source_identity": validated_plan[
                            "analysis_contract"
                        ]["source_identity"],
                        "analysis_environment_identity_sha256": validated_plan[
                            "analysis_contract"
                        ]["environment_identity_sha256"],
                        "input_ledger_sha256": fresh.summary[
                            "input_ledger_sha256"
                        ],
                        "forensic_ledger_sha256": fresh.summary["audit"][
                            "forensic_ledger_sha256"
                        ],
                        "resolved_forensic_artifacts": fresh.summary["audit"][
                            "resolved_forensic_artifacts"
                        ],
                        "invalid_forensic_artifacts": fresh.summary["audit"][
                            "invalid_forensic_artifacts"
                        ],
                        "files": {
                            name: hashlib.sha256(payload).hexdigest()
                            for name, payload in sorted(artifacts.items())
                        },
                    }
                    manifest_payload = _canonical_bytes(manifest) + b"\n"
                    _write_new_file_at(
                        stage_descriptor,
                        "manifest.json",
                        manifest_payload,
                    )
                    written_manifest = os.stat(
                        "manifest.json",
                        dir_fd=stage_descriptor,
                        follow_symlinks=False,
                    )
                    written_identities["manifest.json"] = (
                        written_manifest.st_dev,
                        written_manifest.st_ino,
                    )
                    expected_artifacts = {
                        **artifacts,
                        "manifest.json": manifest_payload,
                    }
                    if frozenset(expected_artifacts) != ANALYSIS_FILENAMES:
                        raise GridAnalysisError(
                            "analysis stage differs from the exact artifact set"
                        )
                    os.fsync(stage_descriptor)
                    os.fchmod(stage_descriptor, 0o555)
                    os.fsync(stage_descriptor)
                    staged_stat = os.fstat(stage_descriptor)
                    if staged_stat.st_mode & 0o222:
                        raise GridAnalysisError("analysis stage was not sealed")
                    full_grid._assert_publication_fence_descriptor(
                        root,
                        run_fence,
                    )
                    output_stat = os.fstat(output_lock)
                    output_path = os.stat(
                        f".{destination.name}.analysis-publish.fence",
                        dir_fd=output_parent,
                        follow_symlinks=False,
                    )
                    if (
                        output_stat.st_dev != output_path.st_dev
                        or output_stat.st_ino != output_path.st_ino
                    ):
                        raise GridAnalysisError(
                            "output publication fence was replaced"
                        )
                    try:
                        full_grid._publish_sealed_directory_noreplace_at(
                            output_parent,
                            stage_name,
                            stage_descriptor,
                            output_parent,
                            destination.name,
                            expected_names=ANALYSIS_FILENAMES,
                        )
                    except BaseException:
                        try:
                            os.stat(
                                stage_name,
                                dir_fd=output_parent,
                                follow_symlinks=False,
                            )
                        except FileNotFoundError:
                            # The helper may move successfully and then fail
                            # while re-sealing, fsyncing, or final-verifying.
                            # Mark that exact held inode for outer invalidation.
                            published = True
                            published_identity = (
                                staged_stat.st_dev,
                                staged_stat.st_ino,
                            )
                        raise
                    published = True
                    published_identity = (
                        staged_stat.st_dev,
                        staged_stat.st_ino,
                    )
                    os.fsync(output_parent)
                    _validate_published_analysis_tree_at(
                        output_parent=output_parent,
                        destination=destination,
                        staged_stat=staged_stat,
                        expected_artifacts=expected_artifacts,
                        expected_manifest=manifest,
                    )
                    _verify_input_ledger(
                        root,
                        validated_plan,
                        fresh.input_ledger,
                    )
                    if full_grid.load_plan(root) != validated_plan:
                        raise GridAnalysisError(
                            "immutable plan changed after analysis publication"
                        )
                    _strict_record_tree(
                        root,
                        validated_jobs,
                    )
                    _require_quiescent_auxiliary_state(root)
                    full_grid._assert_publication_fence_descriptor(
                        root,
                        run_fence,
                    )
                    final_output_lock = os.fstat(output_lock)
                    final_output_path = os.stat(
                        f".{destination.name}.analysis-publish.fence",
                        dir_fd=output_parent,
                        follow_symlinks=False,
                    )
                    if (
                        final_output_lock.st_dev != final_output_path.st_dev
                        or final_output_lock.st_ino != final_output_path.st_ino
                    ):
                        raise GridAnalysisError(
                            "output publication fence was replaced"
                        )
                    full_grid._assert_directory_descriptor_path(
                        destination.parent,
                        output_parent,
                    )
                    completed = (fresh, destination)
                finally:
                    if not published:
                        os.fchmod(stage_descriptor, 0o700)
                        for child_name, expected_identity in sorted(
                            written_identities.items()
                        ):
                            try:
                                child = os.stat(
                                    child_name,
                                    dir_fd=stage_descriptor,
                                    follow_symlinks=False,
                                )
                            except FileNotFoundError:
                                continue
                            if (
                                stat.S_ISREG(child.st_mode)
                                and child.st_nlink == 1
                                and (child.st_dev, child.st_ino)
                                == expected_identity
                            ):
                                os.unlink(
                                    child_name,
                                    dir_fd=stage_descriptor,
                                )
                        os.fsync(stage_descriptor)
                    os.close(stage_descriptor)
                    if not published:
                        try:
                            stage_path_stat = os.stat(
                                stage_name,
                                dir_fd=output_parent,
                                follow_symlinks=False,
                            )
                            stage_path_descriptor = os.open(
                                stage_name,
                                full_grid._directory_open_flags(),
                                dir_fd=output_parent,
                            )
                            try:
                                stage_path_descriptor_stat = os.fstat(
                                    stage_path_descriptor
                                )
                                stage_is_exact_empty = (
                                    stat.S_ISDIR(stage_path_stat.st_mode)
                                    and (
                                        stage_path_stat.st_dev,
                                        stage_path_stat.st_ino,
                                    )
                                    == (
                                        initial_stage_stat.st_dev,
                                        initial_stage_stat.st_ino,
                                    )
                                    == (
                                        stage_path_descriptor_stat.st_dev,
                                        stage_path_descriptor_stat.st_ino,
                                    )
                                    and not os.listdir(stage_path_descriptor)
                                )
                            finally:
                                os.close(stage_path_descriptor)
                            if stage_is_exact_empty:
                                os.rmdir(stage_name, dir_fd=output_parent)
                                os.fsync(output_parent)
                        except OSError:
                            # Preserve unexpected content as forensic evidence.
                            pass
        if completed is None:
            raise GridAnalysisError("analysis publication did not complete")
        return completed
    except BaseException as publication_error:
        if published_identity is not None:
            try:
                _invalidate_published_analysis_at(
                    output_parent=recovery_parent,
                    destination=destination,
                    published_identity=published_identity,
                )
            except BaseException as invalidation_error:
                raise GridAnalysisError(
                    "analysis publication failed and its canonical destination "
                    "could not be invalidated"
                ) from invalidation_error
        raise
    finally:
        os.close(recovery_parent)


def publish_analysis(
    *,
    run_root: str | Path,
    cache_root: str | Path,
    output_dir: str | Path,
) -> Path:
    """Publish one fresh plan-frozen analysis; no caller result is accepted."""

    _, destination = _compute_and_publish_analysis(
        run_root=run_root,
        cache_root=cache_root,
        output_dir=output_dir,
    )
    return destination


def analyze_and_publish(
    *,
    run_root: str | Path,
    cache_root: str | Path,
    output_dir: str | Path,
) -> tuple[AnalysisResult, Path]:
    return _compute_and_publish_analysis(
        run_root=run_root,
        cache_root=cache_root,
        output_dir=output_dir,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fail-closed post-hoc analysis of one exact score-blind full grid"
        )
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    full_grid._require_virtual_environment()
    arguments = _parser().parse_args(argv)
    result, destination = analyze_and_publish(
        run_root=arguments.run_root,
        cache_root=arguments.cache_root,
        output_dir=arguments.output_dir,
    )
    print(
        json.dumps(
            {
                "output_dir": str(destination),
                "plan_sha256": result.summary["plan_sha256"],
                "descriptive_leader": result.summary["descriptive_leader"],
                "primary_value": result.summary["primary_metric"][
                    "descriptive_leader_value"
                ],
                "bootstrap_resamples": result.summary["bootstrap"]["resamples"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
