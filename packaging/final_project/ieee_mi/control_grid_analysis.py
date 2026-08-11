"""Fail-closed aggregate analysis for the deterministic classical-control grid.

The producer in :mod:`ieee_mi.control_grid` deliberately publishes no outcome
or score.  This module is the only score-joining layer: it first proves that
the exact 1,344-record grid is complete and quiescent, then joins plan-bound
cache labels to checksum-bound row indices in memory.  Trial labels, row
vectors, and probability matrices are never placed in an ``AnalysisResult``
or a published artifact.

Classical controls remain a separately labelled, seedless track.  No control
is ranked as a neural common-recipe fit and no inferential candidate contrast
is fabricated.
"""

from __future__ import annotations

import argparse
import contextlib
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
from collections.abc import Mapping, Sequence
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

from . import control_grid


ANALYSIS_SCHEMA = "ieee-mi-classical-control-analysis-v1"
MANIFEST_SCHEMA = "ieee-mi-classical-control-analysis-manifest-v1"
TRACK = "classical_controls_separate_non_neural"
DEFAULT_ECE_BINS = 15
ANALYSIS_SOURCE_FILES: tuple[str, ...] = tuple(
    sorted(
        {
            *control_grid.SOURCE_FILES,
            "ieee_mi/control_grid_analysis.py",
            "ieee_mi/full_grid_analysis.py",
            "pyproject.toml",
            "uv.lock",
        }
    )
)
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
TABLE_NAMES = frozenset(
    {
        "job_metrics",
        "fold_metrics",
        "subject_metrics",
        "dataset_summary",
        "overall_summary",
        "timing_summary",
    }
)
_RESULT_SEAL_KEY = os.urandom(32)


class ControlGridAnalysisError(RuntimeError):
    """Raised when aggregate analysis cannot preserve its audit boundary."""


@dataclass
class AnalysisResult:
    """Aggregate-only analysis state.

    ``input_ledger`` is retained in memory for publication-time revalidation.
    It contains file/cache digests only and is intentionally not published.
    """

    summary: dict[str, Any]
    tables: dict[str, list[dict[str, Any]]]
    input_ledger: list[dict[str, str]] = field(repr=False)
    _seal: str = field(default="", repr=False)


@dataclass
class _PublicationSnapshot:
    destination: Path
    parent_descriptor: int
    parent_fingerprint: tuple[int, ...]
    directory_descriptor: int
    directory_fingerprint: tuple[int, ...]
    member_descriptors: dict[str, int]
    member_fingerprints: dict[str, tuple[int, ...]]

    def close(self) -> None:
        for descriptor in self.member_descriptors.values():
            os.close(descriptor)
        self.member_descriptors.clear()
        if self.directory_descriptor >= 0:
            os.close(self.directory_descriptor)
            self.directory_descriptor = -1
        if self.parent_descriptor >= 0:
            os.close(self.parent_descriptor)
            self.parent_descriptor = -1


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
    return control_grid._sha256_file(path)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


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


def classification_metrics(
    y_true: np.ndarray | Sequence[int],
    probabilities: np.ndarray,
    *,
    n_classes: int,
    ece_bins: int = DEFAULT_ECE_BINS,
) -> dict[str, float | None]:
    """Use the exact metric definitions frozen by ``full_grid_analysis``."""

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
        raise ControlGridAnalysisError(
            "invalid arrays supplied to metric computation"
        )
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
    normalized_ba = (
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
                * abs(
                    float(correct[mask].mean())
                    - float(confidence[mask].mean())
                )
            )
    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": balanced_accuracy,
        "chance_normalized_balanced_accuracy": float(normalized_ba),
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
        "multiclass_brier": float(
            np.square(values - one_hot).sum(axis=1).mean()
        ),
        "ece": float(ece),
    }


def _metric_means(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, float | None | int]:
    result: dict[str, float | None | int] = {}
    for name in METRIC_NAMES:
        values = [
            float(row[name])
            for row in rows
            if row.get(name) is not None
            and math.isfinite(float(row[name]))
        ]
        result[name] = float(np.mean(values)) if values else None
        result[f"{name}_defined_count"] = len(values)
    return result


def _stable_audit_snapshot(audit: Mapping[str, Any]) -> dict[str, Any]:
    """Remove wall-clock and per-job lists while retaining the exact verdict."""

    keys = {
        "schema",
        "plan_sha256",
        "score_blind",
        "expected_jobs",
        "complete_jobs",
        "missing_jobs",
        "corrupt_jobs",
        "extra_directories",
        "unexpected_record_paths",
        "residual_partial_paths",
        "residual_claim_paths",
        "residual_cpu_slot_paths",
        "unexpected_root_entries",
        "expected_per_control",
        "complete_per_control",
        "complete_per_dataset",
        "complete",
    }
    if not keys.issubset(audit):
        missing = sorted(keys - set(audit))
        raise ControlGridAnalysisError(
            f"control audit is missing required fields: {missing}"
        )
    return {key: _json_ready(audit[key]) for key in sorted(keys)}


def _require_regular_readonly(path: Path, context: str) -> None:
    try:
        observed = control_grid._anchored_lstat(path)
        control_grid._assert_unique_regular(
            observed, source=str(path), readonly=True
        )
    except (OSError, control_grid.ControlGridError) as error:
        raise ControlGridAnalysisError(
            f"{context} is absent, aliased, non-regular, or writable: {path}"
        ) from error


def audit_exact_grid(
    run_root: str | Path,
    *,
    require_formal_grid: bool = True,
    _fenced: bool = False,
) -> tuple[dict[str, Any], tuple[control_grid.Job, ...], dict[str, Any]]:
    """Require a complete, quiescent, checksum-valid control snapshot."""

    root = control_grid._absolute(Path(run_root))
    if not _fenced:
        with control_grid.analysis_fence(root):
            return audit_exact_grid(
                root,
                require_formal_grid=require_formal_grid,
                _fenced=True,
            )
    _require_regular_readonly(root / "plan.json", "immutable plan")
    _require_regular_readonly(root / "plan.sha256", "immutable plan digest")
    plan = control_grid.load_plan(root)
    if require_formal_grid:
        if (
            tuple(plan.get("controls", ())) != control_grid.CONTROLS
            or tuple(plan.get("dataset_order", ()))
            != control_grid.OPENED_DATASETS
        ):
            raise ControlGridAnalysisError(
                "analysis requires the exact formal control and dataset roster"
            )
        # Cardinality is not identity. This exact comparison binds every
        # subject, fold, class count, protocol, and preprocessing field before
        # any completion audit can lead to a cache/label read.
        expected_contracts = control_grid._dataset_contracts()
        if plan.get("datasets") != expected_contracts:
            raise ControlGridAnalysisError(
                "formal dataset contracts differ from the live sealed roster"
            )
    jobs = tuple(control_grid.iter_jobs(plan))
    if len(jobs) != int(plan.get("n_jobs", -1)):
        raise ControlGridAnalysisError(
            "plan job count differs from its Cartesian product"
        )
    if require_formal_grid and len(jobs) != control_grid.EXPECTED_JOB_COUNT:
        raise ControlGridAnalysisError(
            "analysis requires the exact 1,344-job formal classical-control grid"
        )

    audit = control_grid.audit_grid(root, plan, _fenced=True)
    expected_ids = {job.job_id for job in jobs}
    complete_ids = set(audit.get("complete_job_ids", ()))
    if (
        audit.get("plan_sha256") != plan.get("plan_sha256")
        or int(audit.get("expected_jobs", -1)) != len(jobs)
        or int(audit.get("complete_jobs", -1)) != len(jobs)
        or int(audit.get("missing_jobs", -1)) != 0
        or int(audit.get("corrupt_jobs", -1)) != 0
        or audit.get("complete") is not True
        or complete_ids != expected_ids
        or audit.get("missing_job_ids") != []
        or audit.get("corrupt_job_ids") != {}
        or audit.get("extra_directories") != []
        or audit.get("unexpected_record_paths") != []
        or audit.get("residual_partial_paths") != []
        or audit.get("residual_claim_paths") != []
        or audit.get("residual_cpu_slot_paths") != []
        or audit.get("unexpected_root_entries") != []
    ):
        raise ControlGridAnalysisError(
            "refusing analysis: control grid is incomplete, active, stray, "
            f"or corrupt ({_stable_audit_snapshot(audit)})"
        )
    return plan, jobs, _stable_audit_snapshot(audit)


def _verify_live_runtime_identity(
    plan: Mapping[str, Any],
    cache_root: Path,
) -> None:
    """Bind a score-blind audited plan to the exact live execution closure."""

    try:
        control_grid.verify_runtime_identity(
            plan,
            cache_root=cache_root.resolve(),
        )
    except control_grid.ControlGridError as error:
        raise ControlGridAnalysisError(
            f"live control runtime identity differs from plan: {error}"
        ) from error


def _cache_ledger_row(
    *,
    plan: Mapping[str, Any],
    dataset: str,
    subject: int,
    cache: Mapping[str, Any],
) -> dict[str, str]:
    subject_key = control_grid._subject_key(dataset, subject)
    split_payload = {
        control_grid._split_key(dataset, subject, int(fold)): plan[
            "split_identity"
        ][control_grid._split_key(dataset, subject, int(fold))]
        for fold in plan["datasets"][dataset]["folds"]
    }
    return {
        "kind": "cache",
        "identity": subject_key,
        "sha256_a": str(plan["cache_identity"][subject_key]["array_sha256"]),
        "sha256_b": _sha256_bytes(_canonical_bytes(cache["identity"])),
        "sha256_c": _sha256_bytes(_canonical_bytes(split_payload)),
    }


def _bound_cache_labels(
    plan: Mapping[str, Any],
    cache_root: Path,
) -> tuple[
    dict[tuple[str, int], np.ndarray],
    list[dict[str, str]],
]:
    """Load labels only after audit and rebuild every planned split identity."""

    from .data import load_subject_cache, split_indices

    labels_by_subject: dict[tuple[str, int], np.ndarray] = {}
    ledger: list[dict[str, str]] = []
    for dataset in plan["dataset_order"]:
        contract = plan["datasets"][dataset]
        for raw_subject in contract["subjects"]:
            subject = int(raw_subject)
            subject_key = control_grid._subject_key(str(dataset), subject)
            planned_cache = plan["cache_identity"].get(subject_key)
            if not isinstance(planned_cache, Mapping):
                raise ControlGridAnalysisError(
                    f"plan cache identity is absent for {subject_key}"
                )
            cache = load_subject_cache(
                str(dataset),
                subject,
                cache_root=cache_root,
            )
            planned_base = {
                key: value
                for key, value in planned_cache.items()
                if key not in {"trial_count", "n_channels"}
            }
            if cache.get("identity") != planned_base:
                raise ControlGridAnalysisError(
                    f"cache identity differs from plan for {subject_key}"
                )
            labels = np.asarray(cache.get("y"))
            if (
                labels.dtype != np.int64
                or labels.ndim != 1
                or len(labels) != int(planned_cache["trial_count"])
                or int(cache["x"].shape[1])
                != int(planned_cache["n_channels"])
                or np.any(labels < 0)
                or np.any(labels >= int(contract["n_classes"]))
                or set(labels.tolist())
                != set(range(int(contract["n_classes"])))
            ):
                raise ControlGridAnalysisError(
                    f"cache arrays/classes are invalid for {subject_key}"
                )

            seen_test_rows: set[int] = set()
            for raw_fold in contract["folds"]:
                fold = int(raw_fold)
                train, validation, test = split_indices(
                    str(dataset),
                    labels,
                    cache["sessions"],
                    cache["runs"],
                    fold=fold,
                    subject=subject,
                )
                observed = control_grid._one_split_identity(
                    dataset=str(dataset),
                    subject=subject,
                    fold=fold,
                    cache_array_sha256=str(
                        planned_cache["array_sha256"]
                    ),
                    trial_count=len(labels),
                    train_rows=train,
                    validation_rows=validation,
                    test_rows=test,
                )
                split_key = control_grid._split_key(
                    str(dataset), subject, fold
                )
                if observed != plan["split_identity"].get(split_key):
                    raise ControlGridAnalysisError(
                        f"reconstructed split differs from plan for {split_key}"
                    )
                test_set = set(np.asarray(test, dtype=np.int64).tolist())
                overlap = seen_test_rows.intersection(test_set)
                if overlap:
                    raise ControlGridAnalysisError(
                        f"test folds overlap for {subject_key}: "
                        f"{sorted(overlap)[:5]}"
                    )
                seen_test_rows.update(test_set)
            labels_by_subject[(str(dataset), subject)] = labels.copy()
            ledger.append(
                _cache_ledger_row(
                    plan=plan,
                    dataset=str(dataset),
                    subject=subject,
                    cache=cache,
                )
            )
    return labels_by_subject, ledger


def _plan_ledger_row(
    run_root: Path, plan: Mapping[str, Any]
) -> dict[str, str]:
    payloads, _, _ = control_grid._snapshot_regular_files(
        run_root,
        ("plan.json", "plan.sha256"),
        child_mode=0o444,
    )
    return {
        "kind": "plan",
        "identity": str(plan["plan_sha256"]),
        "sha256_a": _sha256_bytes(payloads["plan.json"]),
        "sha256_b": _sha256_bytes(payloads["plan.sha256"]),
        "sha256_c": str(plan["plan_sha256"]),
    }


def _job_ledger_row(
    run_root: Path,
    job: control_grid.Job,
    *,
    payload: Mapping[str, Any] | None = None,
    plan: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    if payload is None:
        if plan is None:
            raise ControlGridAnalysisError(
                "job ledger recomputation requires the immutable plan"
            )
        payload = control_grid.validate_completion(
            run_root,
            plan,
            job,
            _stable_parent=True,
        )
    files = payload.get("file_sha256")
    if not isinstance(files, Mapping):
        raise ControlGridAnalysisError(
            f"job snapshot digest ledger is absent for {job.job_id}"
        )
    return {
        "kind": "job",
        "identity": job.job_id,
        "sha256_a": str(files["completion.json"]),
        "sha256_b": str(files["record.json"]),
        "sha256_c": str(files["predictions.npz"]),
    }


def _source_identity() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    paths = {name: root / name for name in ANALYSIS_SOURCE_FILES}
    try:
        return {
            name: _sha256_file(path)
            for name, path in sorted(paths.items())
        }
    except (OSError, control_grid.ControlGridError) as error:
        raise ControlGridAnalysisError(
            "analysis source closure contains an absent or unsafe file"
        ) from error


def _analysis_environment() -> dict[str, Any]:
    """Record the full UV runtime plus explicit critical-package provenance."""

    runtime = control_grid._environment_identity()
    if runtime.get("uv_version") is None:
        raise ControlGridAnalysisError(
            "analysis runtime cannot record the required UV version"
        )
    packages = {
        str(name): str(version)
        for name, version in runtime.get("packages", ())
    }
    required_versions = {
        name: packages.get(name)
        for name in sorted(control_grid.REQUIRED_DIRECT_DEPENDENCIES)
    }
    missing = [
        name for name, version in required_versions.items() if version is None
    ]
    if missing:
        raise ControlGridAnalysisError(
            "analysis runtime lacks required locked packages: "
            f"{missing}"
        )
    return {
        "full_runtime_identity": runtime,
        "required_package_versions": required_versions,
    }


def _job_scalar_row(
    job: control_grid.Job,
    record: Mapping[str, Any],
    metrics: Mapping[str, float | None],
) -> dict[str, Any]:
    metadata = record["metadata"]
    fit = metadata["fit"]
    timing = metadata["timing_seconds"]
    test_count = int(record["test_count"])
    return {
        "track": TRACK,
        **job.identity(),
        "job_id": job.job_id,
        "test_count": test_count,
        **metrics,
        "candidate_count": int(fit["selection_candidate_count"]),
        "estimator_fit_calls": int(fit["estimator_fit_calls"]),
        "parameter_count": int(fit["parameter_count"]),
        "selection_fit_seconds": float(timing["selection_fit"]),
        "refit_fit_seconds": float(timing["refit_fit"]),
        "test_inference_seconds": float(timing["test_inference"]),
        "job_total_seconds": float(timing["job_total"]),
        "inference_ms_per_trial": (
            1000.0 * float(timing["test_inference"]) / test_count
        ),
    }


def _fold_scalar_row(
    job: control_grid.Job,
    *,
    test_count: int,
    metrics: Mapping[str, float | None],
) -> dict[str, Any]:
    return {
        "track": TRACK,
        **job.identity(),
        "test_count": int(test_count),
        **metrics,
    }


_TIMING_BASES = (
    "candidate_count",
    "estimator_fit_calls",
    "parameter_count",
    "selection_fit_seconds",
    "refit_fit_seconds",
    "test_inference_seconds",
    "job_total_seconds",
    "inference_ms_per_trial",
)


def _one_timing_row(
    dataset: str,
    control: str,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "track": TRACK,
        "dataset": dataset,
        "control": control,
        "jobs": len(rows),
    }
    for base in _TIMING_BASES:
        values = np.asarray(
            [float(row[base]) for row in rows],
            dtype=np.float64,
        )
        result[f"{base}_mean"] = float(values.mean())
        result[f"{base}_median"] = float(np.median(values))
        result[f"{base}_p95"] = float(np.quantile(values, 0.95))
        result[f"{base}_min"] = float(values.min())
        result[f"{base}_max"] = float(values.max())
    return result


def _timing_summary(
    plan: Mapping[str, Any],
    job_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    overall: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in job_rows:
        grouped[(str(row["dataset"]), str(row["control"]))].append(row)
        overall[str(row["control"])].append(row)
    result: list[dict[str, Any]] = []
    for dataset in plan["dataset_order"]:
        for control in plan["controls"]:
            result.append(
                _one_timing_row(
                    str(dataset),
                    str(control),
                    grouped[(str(dataset), str(control))],
                )
            )
    for control in plan["controls"]:
        result.append(
            _one_timing_row(
                "ALL_DATASETS",
                str(control),
                overall[str(control)],
            )
        )
    return result


def _verify_input_ledger(
    *,
    run_root: Path,
    cache_root: Path,
    plan: Mapping[str, Any],
    jobs: Sequence[control_grid.Job],
    ledger: Sequence[Mapping[str, str]],
) -> None:
    _, cache_rows = _bound_cache_labels(plan, cache_root)
    observed = [
        _plan_ledger_row(run_root, plan),
        *cache_rows,
        *[
            _job_ledger_row(
                run_root, job, plan=plan
            )
            for job in jobs
        ],
    ]
    if observed != list(ledger):
        raise ControlGridAnalysisError(
            "input checksum ledger changed or differs from the exact plan"
        )


def compute_analysis(
    *,
    run_root: str | Path,
    cache_root: str | Path,
    ece_bins: int = DEFAULT_ECE_BINS,
    _fenced: bool = False,
) -> AnalysisResult:
    """Audit, join in memory, score, aggregate, and seal one formal grid."""

    if ece_bins < 2:
        raise ValueError("ece_bins must be at least two")
    control_grid._require_uv_virtual_environment()
    root = control_grid._absolute(Path(run_root))
    cache = control_grid._absolute(Path(cache_root))
    if not _fenced:
        with control_grid.analysis_fence(root):
            return compute_analysis(
                run_root=root,
                cache_root=cache,
                ece_bins=ece_bins,
                _fenced=True,
            )

    # No cache is opened before this exact score-blind audit succeeds.
    plan, jobs, opening_audit = audit_exact_grid(
        root, _fenced=True
    )
    _verify_live_runtime_identity(plan, cache)
    labels_by_subject, cache_ledger = _bound_cache_labels(plan, cache)

    job_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    grouped: dict[
        tuple[str, str, int],
        dict[str, list[np.ndarray]],
    ] = defaultdict(lambda: {"rows": [], "labels": [], "outputs": []})
    job_ledger: list[dict[str, str]] = []

    for job in jobs:
        payload = control_grid.validate_completion(
            root, plan, job, _stable_parent=True
        )
        rows = np.asarray(payload["rows"], dtype=np.int64)
        probabilities = np.asarray(
            payload["probabilities"], dtype=np.float64
        )
        labels = labels_by_subject[(job.dataset, job.subject)]
        if (
            np.any(rows < 0)
            or np.any(rows >= len(labels))
            or len(np.unique(rows)) != len(rows)
        ):
            raise ControlGridAnalysisError(
                f"prediction rows are invalid for {job.job_id}"
            )
        outcomes = labels[rows]
        metrics = classification_metrics(
            outcomes,
            probabilities,
            n_classes=int(plan["datasets"][job.dataset]["n_classes"]),
            ece_bins=ece_bins,
        )
        job_rows.append(
            _job_scalar_row(job, payload["record"], metrics)
        )
        fold_rows.append(
            _fold_scalar_row(
                job,
                test_count=len(rows),
                metrics=metrics,
            )
        )
        key = (job.dataset, job.control, job.subject)
        grouped[key]["rows"].append(rows.copy())
        grouped[key]["labels"].append(outcomes.copy())
        grouped[key]["outputs"].append(probabilities.copy())
        job_ledger.append(
            _job_ledger_row(root, job, payload=payload)
        )

    if [str(row["job_id"]) for row in job_rows] != [
        job.job_id for job in jobs
    ]:
        raise ControlGridAnalysisError(
            "scored job rows differ from the immutable plan order"
        )

    subject_rows: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    for dataset in plan["dataset_order"]:
        contract = plan["datasets"][dataset]
        n_classes = int(contract["n_classes"])
        for control in plan["controls"]:
            selected_subjects: list[dict[str, Any]] = []
            for raw_subject in contract["subjects"]:
                subject = int(raw_subject)
                group = grouped[(str(dataset), str(control), subject)]
                row_values = np.concatenate(group["rows"])
                outcome_values = np.concatenate(group["labels"])
                output_values = np.concatenate(group["outputs"])
                order = np.argsort(row_values, kind="stable")
                row_values = row_values[order]
                outcome_values = outcome_values[order]
                output_values = output_values[order]
                if len(np.unique(row_values)) != len(row_values):
                    raise ControlGridAnalysisError(
                        f"fold test rows overlap for {dataset}/{control}/"
                        f"S{subject}"
                    )
                metrics = classification_metrics(
                    outcome_values,
                    output_values,
                    n_classes=n_classes,
                    ece_bins=ece_bins,
                )
                aggregate = {
                    "track": TRACK,
                    "dataset": str(dataset),
                    "control": str(control),
                    "subject": subject,
                    "folds_concatenated": len(contract["folds"]),
                    "test_trials": len(row_values),
                    **metrics,
                }
                subject_rows.append(aggregate)
                selected_subjects.append(aggregate)
            dataset_rows.append(
                {
                    "track": TRACK,
                    "dataset": str(dataset),
                    "control": str(control),
                    "subjects_averaged": len(selected_subjects),
                    "aggregation": (
                        "concatenate_disjoint_folds_within_subject_then_"
                        "mean_subjects"
                    ),
                    **_metric_means(selected_subjects),
                }
            )

    overall_rows: list[dict[str, Any]] = []
    for control in plan["controls"]:
        selected = [
            row
            for row in dataset_rows
            if row["control"] == str(control)
        ]
        if len(selected) != len(plan["dataset_order"]):
            raise ControlGridAnalysisError(
                f"dataset aggregates are incomplete for {control}"
            )
        overall_rows.append(
            {
                "track": TRACK,
                "control": str(control),
                "datasets_equal_weighted": len(selected),
                "aggregation": (
                    "fold_concatenation_then_subject_mean_then_"
                    "equal_dataset_mean"
                ),
                **_metric_means(selected),
            }
        )

    timing_rows = _timing_summary(plan, job_rows)
    ledger = [
        _plan_ledger_row(root, plan),
        *cache_ledger,
        *job_ledger,
    ]
    _verify_input_ledger(
        run_root=root,
        cache_root=cache,
        plan=plan,
        jobs=jobs,
        ledger=ledger,
    )
    final_plan, final_jobs, final_audit = audit_exact_grid(
        root, _fenced=True
    )
    _verify_live_runtime_identity(final_plan, cache)
    if (
        final_plan != plan
        or final_jobs != jobs
        or final_audit != opening_audit
    ):
        raise ControlGridAnalysisError(
            "plan or exact audit snapshot changed during analysis"
        )

    tables = {
        "job_metrics": job_rows,
        "fold_metrics": fold_rows,
        "subject_metrics": subject_rows,
        "dataset_summary": dataset_rows,
        "overall_summary": overall_rows,
        "timing_summary": timing_rows,
    }
    ledger_digest = _sha256_bytes(_canonical_bytes(ledger))
    summary = {
        "schema": ANALYSIS_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "input_ledger_sha256": ledger_digest,
        "track": TRACK,
        "evidence_scope": plan.get("evidence_scope"),
        "confirmation_evidence": False,
        "interpretation": (
            "exploratory classical-control results on opened development "
            "cohorts; separate from neural common-recipe rankings and not "
            "confirmation or a global/SOTA claim"
        ),
        "controls": [str(value) for value in plan["controls"]],
        "datasets": [str(value) for value in plan["dataset_order"]],
        "expected_jobs": len(jobs),
        "ece_bins": int(ece_bins),
        "aggregation_definition": (
            "concatenate disjoint folds within subject; compute subject "
            "metrics; mean subjects within dataset; equally weight datasets"
        ),
        "metric_definitions": {
            "accuracy": "fraction of correct top-probability classes",
            "balanced_accuracy": (
                "unadjusted mean recall over observed true classes"
            ),
            "chance_normalized_balanced_accuracy": (
                "(balanced_accuracy - 1/K) / (1 - 1/K), using each "
                "dataset's planned class count K"
            ),
            "macro_f1": (
                "macro F1 over the planned class set; zero for undefined F1"
            ),
            "cohen_kappa": "Cohen kappa over the planned class set",
            "ovr_macro_auroc": (
                "mean one-vs-rest AUROC only when every class has positive "
                "and negative examples"
            ),
            "nll": (
                "mean negative log true-class probability clipped at 1e-15"
            ),
            "multiclass_brier": (
                "mean sum of squared class-probability errors"
            ),
            "ece": (
                f"top-label ECE with {ece_bins} equal-width confidence bins"
            ),
        },
        "inferential_comparisons": (
            "omitted: there is no prespecified candidate-control contrast; "
            "no bootstrap, p-value, ranking, or fabricated variance"
        ),
        "audit_snapshot": opening_audit,
        "analysis_source_identity": _source_identity(),
        "analysis_environment": _analysis_environment(),
        "table_row_counts": {
            name: len(rows) for name, rows in tables.items()
        },
    }
    result = AnalysisResult(
        summary=summary,
        tables=tables,
        input_ledger=ledger,
    )
    result._seal = _analysis_result_seal(summary, tables, ledger)
    _validate_result_structure(result)
    return result


_DIRECT_METRIC_COLUMNS = frozenset(METRIC_NAMES)
_AGGREGATE_METRIC_COLUMNS = frozenset(
    column
    for metric in METRIC_NAMES
    for column in (metric, f"{metric}_defined_count")
)
_TABLE_SCHEMAS: dict[str, frozenset[str]] = {
    "job_metrics": frozenset(
        {
            "track",
            "dataset",
            "control",
            "subject",
            "fold",
            "job_id",
            "test_count",
            "candidate_count",
            "estimator_fit_calls",
            "parameter_count",
            "selection_fit_seconds",
            "refit_fit_seconds",
            "test_inference_seconds",
            "job_total_seconds",
            "inference_ms_per_trial",
        }
    )
    | _DIRECT_METRIC_COLUMNS,
    "fold_metrics": frozenset(
        {
            "track",
            "dataset",
            "control",
            "subject",
            "fold",
            "test_count",
        }
    )
    | _DIRECT_METRIC_COLUMNS,
    "subject_metrics": frozenset(
        {
            "track",
            "dataset",
            "control",
            "subject",
            "folds_concatenated",
            "test_trials",
        }
    )
    | _DIRECT_METRIC_COLUMNS,
    "dataset_summary": frozenset(
        {
            "track",
            "dataset",
            "control",
            "subjects_averaged",
            "aggregation",
        }
    )
    | _AGGREGATE_METRIC_COLUMNS,
    "overall_summary": frozenset(
        {
            "track",
            "control",
            "datasets_equal_weighted",
            "aggregation",
        }
    )
    | _AGGREGATE_METRIC_COLUMNS,
    "timing_summary": frozenset(
        {"track", "dataset", "control", "jobs"}
    )
    | frozenset(
        f"{base}_{suffix}"
        for base in _TIMING_BASES
        for suffix in ("mean", "median", "p95", "min", "max")
    ),
}
_SUMMARY_KEYS = frozenset(
    {
        "schema",
        "plan_sha256",
        "input_ledger_sha256",
        "track",
        "evidence_scope",
        "confirmation_evidence",
        "interpretation",
        "controls",
        "datasets",
        "expected_jobs",
        "ece_bins",
        "aggregation_definition",
        "metric_definitions",
        "inferential_comparisons",
        "audit_snapshot",
        "analysis_source_identity",
        "analysis_environment",
        "table_row_counts",
    }
)
_LEDGER_KEYS = frozenset(
    {"kind", "identity", "sha256_a", "sha256_b", "sha256_c"}
)
_FORBIDDEN_ARTIFACT_KEYS = frozenset(
    {
        "label",
        "labels",
        "raw_label",
        "raw_labels",
        "target",
        "targets",
        "truth",
        "ground_truth",
        "y",
        "y_true",
        "y_test",
        "probability",
        "probabilities",
        "row",
        "rows",
        "test_rows",
        "predicted_class",
        "predicted_classes",
    }
)


def _reject_forbidden_artifact_keys(
    value: Any,
    *,
    path: str = "result",
) -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).strip().lower()
            if key in _FORBIDDEN_ARTIFACT_KEYS:
                raise ControlGridAnalysisError(
                    f"analysis artifact contains forbidden key {path}.{raw_key}"
                )
            if "seed" in key:
                raise ControlGridAnalysisError(
                    f"seed field is forbidden in deterministic control output: "
                    f"{path}.{raw_key}"
                )
            _reject_forbidden_artifact_keys(
                child, path=f"{path}.{raw_key}"
            )
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_forbidden_artifact_keys(
                child, path=f"{path}[{index}]"
            )


def _validate_scalar_table(
    name: str,
    rows: Any,
) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or not rows:
        raise ControlGridAnalysisError(
            f"analysis table {name} must be a nonempty list"
        )
    expected = _TABLE_SCHEMAS[name]
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != expected:
            raise ControlGridAnalysisError(
                f"analysis table {name} row {index} violates its exact schema"
            )
        for key, value in row.items():
            if isinstance(value, bool):
                continue
            if value is not None and not isinstance(
                value, (str, int, float)
            ):
                raise ControlGridAnalysisError(
                    f"analysis table {name} row {index} field {key} "
                    "is not scalar"
                )
            if isinstance(value, float) and not math.isfinite(value):
                raise ControlGridAnalysisError(
                    f"analysis table {name} row {index} has non-finite data"
                )
    return rows


def _validate_metric_ranges(
    table_name: str,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    for row in rows:
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
                raise ControlGridAnalysisError(
                    f"{table_name} has out-of-range {metric}"
                )
            if metric in {
                "cohen_kappa",
                "chance_normalized_balanced_accuracy",
            } and not -1.0 <= number <= 1.0:
                raise ControlGridAnalysisError(
                    f"{table_name} has out-of-range {metric}"
                )
            if metric in {"nll", "multiclass_brier"} and number < 0.0:
                raise ControlGridAnalysisError(
                    f"{table_name} has negative {metric}"
                )


def _validate_result_structure(result: AnalysisResult) -> None:
    """Reject mutated, fabricated, non-scalar, leaky, or seeded results."""

    if not isinstance(result, AnalysisResult):
        raise ControlGridAnalysisError(
            "publication requires a sealed AnalysisResult"
        )
    expected_seal = _analysis_result_seal(
        result.summary,
        result.tables,
        result.input_ledger,
    )
    if not hmac.compare_digest(result._seal, expected_seal):
        raise ControlGridAnalysisError(
            "analysis result seal is absent or invalid"
        )
    if set(result.tables) != TABLE_NAMES:
        raise ControlGridAnalysisError(
            "analysis tables differ from the exact aggregate whitelist"
        )
    validated = {
        name: _validate_scalar_table(name, result.tables[name])
        for name in sorted(TABLE_NAMES)
    }
    if not isinstance(result.summary, dict) or set(result.summary) != _SUMMARY_KEYS:
        raise ControlGridAnalysisError(
            "analysis summary violates its exact schema"
        )
    _reject_forbidden_artifact_keys(result.summary)
    _reject_forbidden_artifact_keys(validated)
    if (
        result.summary["schema"] != ANALYSIS_SCHEMA
        or result.summary["track"] != TRACK
        or result.summary["confirmation_evidence"] is not False
        or not isinstance(result.summary["ece_bins"], int)
        or int(result.summary["ece_bins"]) < 2
    ):
        raise ControlGridAnalysisError(
            "analysis summary identity/scope is invalid"
        )
    counts = {name: len(rows) for name, rows in validated.items()}
    if result.summary["table_row_counts"] != counts:
        raise ControlGridAnalysisError(
            "analysis table counts differ from summary"
        )
    if not isinstance(result.input_ledger, list) or not result.input_ledger:
        raise ControlGridAnalysisError("input checksum ledger is absent")
    for index, row in enumerate(result.input_ledger):
        if not isinstance(row, dict) or set(row) != _LEDGER_KEYS:
            raise ControlGridAnalysisError(
                f"input ledger row {index} violates its exact schema"
            )
        if row["kind"] not in {"plan", "cache", "job"}:
            raise ControlGridAnalysisError(
                f"input ledger row {index} has an unknown kind"
            )
        if not isinstance(row["identity"], str) or not row["identity"]:
            raise ControlGridAnalysisError(
                f"input ledger row {index} has an invalid identity"
            )
        if not all(
            _is_sha256(row[key])
            for key in ("sha256_a", "sha256_b", "sha256_c")
        ):
            raise ControlGridAnalysisError(
                f"input ledger row {index} has an invalid digest"
            )
    if result.summary["input_ledger_sha256"] != _sha256_bytes(
        _canonical_bytes(result.input_ledger)
    ):
        raise ControlGridAnalysisError(
            "input checksum-ledger digest is invalid"
        )
    for name, rows in validated.items():
        if name != "timing_summary":
            _validate_metric_ranges(name, rows)
        if any(row.get("track") != TRACK for row in rows):
            raise ControlGridAnalysisError(
                f"analysis table {name} escaped the classical track"
            )


def _validate_against_plan(
    result: AnalysisResult,
    plan: Mapping[str, Any],
    jobs: Sequence[control_grid.Job],
) -> None:
    if (
        result.summary["plan_sha256"] != plan["plan_sha256"]
        or result.summary["controls"]
        != [str(value) for value in plan["controls"]]
        or result.summary["datasets"]
        != [str(value) for value in plan["dataset_order"]]
        or int(result.summary["expected_jobs"]) != len(jobs)
        or result.summary["analysis_source_identity"] != _source_identity()
        or result.summary["analysis_environment"] != _analysis_environment()
    ):
        raise ControlGridAnalysisError(
            "analysis summary differs from plan/source/environment"
        )
    job_rows = result.tables["job_metrics"]
    fold_rows = result.tables["fold_metrics"]
    if len(job_rows) != len(jobs) or len(fold_rows) != len(jobs):
        raise ControlGridAnalysisError(
            "job/fold table cardinality differs from plan"
        )
    for job, job_row, fold_row in zip(
        jobs, job_rows, fold_rows, strict=True
    ):
        identity = job.identity()
        if (
            str(job_row["job_id"]) != job.job_id
            or any(job_row[key] != value for key, value in identity.items())
            or any(fold_row[key] != value for key, value in identity.items())
            or "seed" in job_row
            or "seed" in fold_row
        ):
            raise ControlGridAnalysisError(
                f"job/fold aggregate identity differs for {job.job_id}"
            )
        expected_count = int(
            control_grid._planned_split(plan, job)["partitions"]["test"][
                "count"
            ]
        )
        if (
            int(job_row["test_count"]) != expected_count
            or int(fold_row["test_count"]) != expected_count
        ):
            raise ControlGridAnalysisError(
                f"job/fold test count differs for {job.job_id}"
            )

    expected_subjects = [
        (str(dataset), str(control), int(subject))
        for dataset in plan["dataset_order"]
        for control in plan["controls"]
        for subject in plan["datasets"][dataset]["subjects"]
    ]
    observed_subjects = [
        (
            str(row["dataset"]),
            str(row["control"]),
            int(row["subject"]),
        )
        for row in result.tables["subject_metrics"]
    ]
    if observed_subjects != expected_subjects:
        raise ControlGridAnalysisError(
            "subject table differs from plan product"
        )
    expected_datasets = [
        (str(dataset), str(control))
        for dataset in plan["dataset_order"]
        for control in plan["controls"]
    ]
    observed_datasets = [
        (str(row["dataset"]), str(row["control"]))
        for row in result.tables["dataset_summary"]
    ]
    if observed_datasets != expected_datasets:
        raise ControlGridAnalysisError(
            "dataset table differs from plan product"
        )
    if [str(row["control"]) for row in result.tables["overall_summary"]] != [
        str(value) for value in plan["controls"]
    ]:
        raise ControlGridAnalysisError(
            "overall table differs from control roster"
        )
    expected_timing = expected_datasets + [
        ("ALL_DATASETS", str(control))
        for control in plan["controls"]
    ]
    observed_timing = [
        (str(row["dataset"]), str(row["control"]))
        for row in result.tables["timing_summary"]
    ]
    if observed_timing != expected_timing:
        raise ControlGridAnalysisError(
            "timing table differs from plan product"
        )

    for table_name in (
        "job_metrics",
        "fold_metrics",
        "subject_metrics",
        "dataset_summary",
    ):
        for row in result.tables[table_name]:
            dataset = str(row["dataset"])
            ba = row.get("balanced_accuracy")
            normalized = row.get(
                "chance_normalized_balanced_accuracy"
            )
            if ba is not None and normalized is not None:
                classes = int(plan["datasets"][dataset]["n_classes"])
                chance = 1.0 / classes
                expected = (float(ba) - chance) / (1.0 - chance)
                if not math.isclose(
                    float(normalized),
                    expected,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ControlGridAnalysisError(
                        f"{table_name} has inconsistent chance-normalized BA"
                    )


def _validate_result_for_publication(
    result: AnalysisResult,
    *,
    run_root: Path,
    cache_root: Path,
) -> None:
    """Freshly recompute the aggregate result under the owned output lock."""

    _validate_result_structure(result)
    _close_publication_input_window(
        result,
        run_root=run_root,
        cache_root=cache_root,
    )

    # This is intentionally stronger than a mutable Python object seal:
    # re-open every cache and prediction, recompute every aggregate, and demand
    # byte-identical scalar output while the publisher owns its lock.
    fresh = compute_analysis(
        run_root=run_root,
        cache_root=cache_root,
        ece_bins=int(result.summary["ece_bins"]),
    )
    if (
        _canonical_bytes(result.summary)
        != _canonical_bytes(fresh.summary)
        or _canonical_bytes(result.tables)
        != _canonical_bytes(fresh.tables)
        or _canonical_bytes(result.input_ledger)
        != _canonical_bytes(fresh.input_ledger)
    ):
        raise ControlGridAnalysisError(
            "analysis result differs from fresh publication-time recomputation"
        )


def _close_publication_input_window(
    result: AnalysisResult,
    *,
    run_root: Path,
    cache_root: Path,
    _fenced: bool = False,
) -> None:
    """Re-audit and rehash all inputs without publishing trial material."""

    if not _fenced:
        with control_grid.analysis_fence(run_root):
            _close_publication_input_window(
                result,
                run_root=run_root,
                cache_root=cache_root,
                _fenced=True,
            )
        return
    plan, jobs, audit = audit_exact_grid(
        run_root, _fenced=True
    )
    _verify_live_runtime_identity(plan, cache_root)
    _validate_against_plan(result, plan, jobs)
    if result.summary["audit_snapshot"] != audit:
        raise ControlGridAnalysisError(
            "analysis summary is not bound to the fresh exact audit"
        )
    _verify_input_ledger(
        run_root=run_root,
        cache_root=cache_root,
        plan=plan,
        jobs=jobs,
        ledger=result.input_ledger,
    )


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        raise ControlGridAnalysisError(
            "refusing to create an empty CSV artifact"
        )
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                columns.append(str(key))
                seen.add(str(key))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=columns, extrasaction="raise"
    )
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
    overall = result.tables["overall_summary"]
    datasets = result.tables["dataset_summary"]
    lines = [
        "# Deterministic classical-control analysis",
        "",
        "> Separate non-neural classical-control track on opened development "
        "cohorts. These are not neural common-recipe rankings, confirmation "
        "evidence, or a global/SOTA claim.",
        "",
        f"- Plan SHA-256: `{result.summary['plan_sha256']}`",
        f"- Atomic seedless records: {result.summary['expected_jobs']:,}",
        "- Inferential comparisons: omitted because no candidate-control "
        "contrast was prespecified.",
        "",
        "## Equal-dataset macro results",
        "",
        "| Control | Accuracy | Balanced accuracy | Chance-normalized BA | "
        "Macro F1 | Kappa | AUROC | NLL | Brier | ECE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in overall:
        lines.append(
            f"| `{row['control']}` | {_format_percent(row['accuracy'])} | "
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
            "Rows retain the immutable registry order; they are deliberately "
            "not sorted into a winner ranking.",
            "",
            "## Balanced accuracy by dataset",
            "",
            "| Dataset | Control | Balanced accuracy | Chance-normalized BA |",
            "|---|---|---:|---:|",
        ]
    )
    for row in datasets:
        lines.append(
            f"| `{row['dataset']}` | `{row['control']}` | "
            f"{_format_percent(row['balanced_accuracy'])} | "
            f"{_format_percent(row['chance_normalized_balanced_accuracy'])} |"
        )
    lines.extend(
        [
            "",
            "Aggregation concatenates disjoint held-out folds within each "
            "subject, averages subjects within each dataset, and gives the "
            "five datasets equal weight. Deterministic controls have no "
            "optimization-seed replication or seed-variance estimate.",
            "",
            "Published CSVs contain scalar job, fold, subject, dataset, "
            "overall, and timing aggregates only. Per-trial row indices, "
            "outcomes, class decisions, and probability matrices are absent.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_new_file(path: Path, payload: bytes) -> None:
    parent = control_grid._open_directory_absolute(path.parent)
    try:
        control_grid._write_exclusive_at(
            parent, path.name, payload, mode=0o444
        )
        os.fsync(parent)
    finally:
        os.close(parent)


def _publication_names() -> frozenset[str]:
    return frozenset(
        {
            "analysis.json",
            "RESULTS.md",
            "manifest.json",
            *{f"{name}.csv" for name in TABLE_NAMES},
        }
    )


def _assert_publication_snapshot(
    snapshot: _PublicationSnapshot,
) -> None:
    if (
        control_grid._stat_fingerprint(
            os.fstat(snapshot.parent_descriptor)
        )
        != snapshot.parent_fingerprint
    ):
        raise ControlGridAnalysisError(
            "analysis publication parent changed during verification"
        )
    held_directory = os.fstat(snapshot.directory_descriptor)
    current_directory = os.stat(
        snapshot.destination.name,
        dir_fd=snapshot.parent_descriptor,
        follow_symlinks=False,
    )
    if (
        control_grid._stat_fingerprint(held_directory)
        != snapshot.directory_fingerprint
        or control_grid._stat_fingerprint(current_directory)
        != snapshot.directory_fingerprint
        or stat.S_IMODE(held_directory.st_mode) != 0o555
        or frozenset(os.listdir(snapshot.directory_descriptor))
        != _publication_names()
    ):
        raise ControlGridAnalysisError(
            "analysis publication directory changed during verification"
        )
    for name, descriptor in snapshot.member_descriptors.items():
        held = os.fstat(descriptor)
        current = os.stat(
            name,
            dir_fd=snapshot.directory_descriptor,
            follow_symlinks=False,
        )
        expected = snapshot.member_fingerprints[name]
        if (
            control_grid._stat_fingerprint(held) != expected
            or control_grid._stat_fingerprint(current) != expected
            or not stat.S_ISREG(held.st_mode)
            or held.st_nlink != 1
            or stat.S_IMODE(held.st_mode) != 0o444
        ):
            raise ControlGridAnalysisError(
                f"analysis publication member changed: {name}"
            )
    control_grid._assert_directory_path(
        snapshot.destination.parent,
        snapshot.parent_descriptor,
    )


def _open_publication_snapshot(
    destination: Path,
) -> _PublicationSnapshot:
    absolute = control_grid._absolute(destination)
    parent = control_grid._open_directory_absolute(absolute.parent)
    directory = -1
    members: dict[str, int] = {}
    try:
        parent_fingerprint = control_grid._stat_fingerprint(
            os.fstat(parent)
        )
        directory = os.open(
            absolute.name,
            control_grid._directory_flags(),
            dir_fd=parent,
        )
        directory_status = os.fstat(directory)
        directory_fingerprint = control_grid._stat_fingerprint(
            directory_status
        )
        current = os.stat(
            absolute.name,
            dir_fd=parent,
            follow_symlinks=False,
        )
        if (
            control_grid._stat_fingerprint(current)
            != directory_fingerprint
            or stat.S_IMODE(directory_status.st_mode) != 0o555
            or frozenset(os.listdir(directory)) != _publication_names()
        ):
            raise ControlGridAnalysisError(
                "published analysis directory is not exactly sealed"
            )
        member_fingerprints: dict[str, tuple[int, ...]] = {}
        for name in sorted(_publication_names()):
            before = os.stat(
                name, dir_fd=directory, follow_symlinks=False
            )
            control_grid._assert_unique_regular(
                before,
                source=str(absolute / name),
                readonly=True,
            )
            if stat.S_IMODE(before.st_mode) != 0o444:
                raise ControlGridAnalysisError(
                    f"published member is not mode 0444: {name}"
                )
            descriptor = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0),
                dir_fd=directory,
            )
            opened = os.fstat(descriptor)
            if (
                control_grid._stat_fingerprint(opened)
                != control_grid._stat_fingerprint(before)
            ):
                os.close(descriptor)
                raise ControlGridAnalysisError(
                    f"published member changed while opening: {name}"
                )
            members[name] = descriptor
            member_fingerprints[name] = (
                control_grid._stat_fingerprint(opened)
            )
        snapshot = _PublicationSnapshot(
            destination=absolute,
            parent_descriptor=parent,
            parent_fingerprint=parent_fingerprint,
            directory_descriptor=directory,
            directory_fingerprint=directory_fingerprint,
            member_descriptors=members,
            member_fingerprints=member_fingerprints,
        )
        parent = -1
        directory = -1
        members = {}
        _assert_publication_snapshot(snapshot)
        return snapshot
    finally:
        for descriptor in members.values():
            os.close(descriptor)
        if directory >= 0:
            os.close(directory)
        if parent >= 0:
            os.close(parent)


def _verify_publication(
    snapshot: _PublicationSnapshot,
    result: AnalysisResult,
) -> None:
    payloads = {
        name: control_grid._read_descriptor_bytes(
            descriptor,
            source=str(snapshot.destination / name),
        )
        for name, descriptor in snapshot.member_descriptors.items()
    }
    _assert_publication_snapshot(snapshot)
    manifest = control_grid._strict_json_bytes(
        payloads["manifest.json"],
        source=str(snapshot.destination / "manifest.json"),
    )
    expected_artifacts = _publication_names() - {"manifest.json"}
    expected_manifest = {
        "schema": MANIFEST_SCHEMA,
        "plan_sha256": result.summary["plan_sha256"],
        "input_ledger_sha256": result.summary[
            "input_ledger_sha256"
        ],
        "track": TRACK,
        "files": {
            name: _sha256_bytes(payloads[name])
            for name in sorted(expected_artifacts)
        },
    }
    if manifest != expected_manifest:
        raise ControlGridAnalysisError(
            "published analysis manifest differs from held member bytes"
        )
    summary = control_grid._strict_json_bytes(
        payloads["analysis.json"],
        source=str(snapshot.destination / "analysis.json"),
    )
    if summary != result.summary:
        raise ControlGridAnalysisError(
            "published analysis summary differs from the sealed result"
        )
    _assert_publication_snapshot(snapshot)


def _hide_exact_analysis_directory(
    path: Path,
    *,
    expected_identity: tuple[int, int],
) -> Path | None:
    absolute = control_grid._absolute(path)
    parent = control_grid._open_directory_absolute(absolute.parent)
    hidden_name = f".{absolute.name}.failed-{uuid.uuid4().hex}"
    try:
        try:
            observed = os.stat(
                absolute.name,
                dir_fd=parent,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return None
        if (int(observed.st_dev), int(observed.st_ino)) != expected_identity:
            return None
        try:
            control_grid._rename_noreplace_at(
                parent,
                absolute.name,
                parent,
                hidden_name,
            )
        except Exception:
            try:
                current = os.stat(
                    absolute.name,
                    dir_fd=parent,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                return absolute.parent / hidden_name
            if (int(current.st_dev), int(current.st_ino)) == expected_identity:
                raise ControlGridAnalysisError(
                    "failed analysis publication left its exact inode visible"
                )
            return None
        os.fsync(parent)
        return absolute.parent / hidden_name
    finally:
        os.close(parent)


def _release_output_lock(
    *,
    path: Path,
    descriptor: int,
    identity: tuple[int, int],
    payload: bytes,
) -> None:
    try:
        control_grid._assert_locked_authority(
            path,
            descriptor,
            expected_identity=identity,
            expected_payload=payload,
        )
        parent = control_grid._open_directory_absolute(path.parent)
        try:
            current = os.stat(
                path.name,
                dir_fd=parent,
                follow_symlinks=False,
            )
            if (int(current.st_dev), int(current.st_ino)) != identity:
                raise ControlGridAnalysisError(
                    "refusing to release a replaced analysis lock"
                )
            os.unlink(path.name, dir_fd=parent)
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _validated_output_destination(
    *,
    run_root: Path,
    cache_root: Path,
    output_dir: Path,
) -> Path:
    root = run_root.resolve()
    cache = cache_root.resolve()
    destination = output_dir.expanduser().resolve()
    if (
        destination == root
        or root in destination.parents
        or destination == cache
        or cache in destination.parents
    ):
        raise ControlGridAnalysisError(
            "analysis output must be outside run_root and cache_root"
        )
    return destination


def publish_analysis(
    result: AnalysisResult,
    *,
    run_root: str | Path,
    cache_root: str | Path,
    output_dir: str | Path,
) -> Path:
    """Atomically publish aggregate-only artifacts after locked reanalysis."""

    root = control_grid._absolute(Path(run_root))
    cache = control_grid._absolute(Path(cache_root))
    destination = _validated_output_destination(
        run_root=root,
        cache_root=cache,
        output_dir=Path(output_dir),
    )
    control_grid._safe_mkdir(destination.parent)
    if control_grid._path_exists(destination):
        raise FileExistsError(
            f"analysis destination exists; refusing overwrite: {destination}"
        )

    lock_root = control_grid._safe_mkdir(
        destination.parent / ".control-analysis-locks"
    )
    lock_path = lock_root / (
        f"{destination.name}.control-analysis-publish.lock"
    )
    lock_descriptor = -1
    lock_identity: tuple[int, int] | None = None
    lock_payload = b""
    lock_token = uuid.uuid4().hex
    stage = destination.parent / (
        f".{destination.name}.partial-{uuid.uuid4().hex}"
    )
    stage_identity: tuple[int, int] | None = None
    published_identity: tuple[int, int] | None = None
    publication_snapshot: _PublicationSnapshot | None = None
    active_error: BaseException | None = None
    try:
        (
            lock_descriptor,
            lock_identity,
            lock_payload,
        ) = control_grid._publish_locked_json(
            root,
            lock_path,
            {
                "schema": "ieee-mi-control-analysis-output-lock-v1",
                "token": lock_token,
            },
            failure_category="analysis_output_lock_failure",
        )
        if control_grid._path_exists(destination):
            raise FileExistsError(destination)

        # A complete deterministic recomputation occurs while the exact output
        # lock is held. It acquires its own worker-respected input fence.
        _validate_result_for_publication(
            result,
            run_root=root,
            cache_root=cache,
        )
        with control_grid.analysis_fence(root):
            _close_publication_input_window(
                result,
                run_root=root,
                cache_root=cache,
                _fenced=True,
            )
            parent = control_grid._open_directory_absolute(
                destination.parent
            )
            stage_descriptor = -1
            try:
                os.mkdir(
                    stage.name,
                    mode=0o700,
                    dir_fd=parent,
                )
                os.fsync(parent)
                stage_descriptor = os.open(
                    stage.name,
                    control_grid._directory_flags(),
                    dir_fd=parent,
                )
                staged = os.fstat(stage_descriptor)
                stage_identity = (
                    int(staged.st_dev),
                    int(staged.st_ino),
                )
                artifacts: dict[str, bytes] = {
                    "analysis.json": (
                        _canonical_bytes(result.summary) + b"\n"
                    ),
                    "RESULTS.md": _markdown_report(result).encode("utf-8"),
                }
                for name in sorted(TABLE_NAMES):
                    artifacts[f"{name}.csv"] = _csv_bytes(
                        result.tables[name]
                    )
                for name, payload in sorted(artifacts.items()):
                    control_grid._write_exclusive_at(
                        stage_descriptor,
                        name,
                        payload,
                        mode=0o444,
                    )
                manifest = {
                    "schema": MANIFEST_SCHEMA,
                    "plan_sha256": result.summary["plan_sha256"],
                    "input_ledger_sha256": result.summary[
                        "input_ledger_sha256"
                    ],
                    "track": TRACK,
                    "files": {
                        name: _sha256_bytes(payload)
                        for name, payload in sorted(artifacts.items())
                    },
                }
                control_grid._write_exclusive_at(
                    stage_descriptor,
                    "manifest.json",
                    _canonical_bytes(manifest) + b"\n",
                    mode=0o444,
                )
                os.fsync(stage_descriptor)
                os.fchmod(stage_descriptor, 0o555)
                os.fsync(stage_descriptor)
                control_grid._snapshot_regular_files(
                    stage,
                    tuple(sorted(_publication_names())),
                    exact_roster=_publication_names(),
                    directory_mode=0o555,
                    child_mode=0o444,
                    stable_parent=True,
                )
                _close_publication_input_window(
                    result,
                    run_root=root,
                    cache_root=cache,
                    _fenced=True,
                )
                control_grid._rename_noreplace_at(
                    parent,
                    stage.name,
                    parent,
                    destination.name,
                )
                os.fsync(parent)
                current = os.stat(
                    destination.name,
                    dir_fd=parent,
                    follow_symlinks=False,
                )
                if (
                    int(current.st_dev),
                    int(current.st_ino),
                ) != stage_identity:
                    raise ControlGridAnalysisError(
                        "published analysis differs from its staged inode"
                    )
                published_identity = stage_identity
                publication_snapshot = _open_publication_snapshot(
                    destination
                )
                _verify_publication(publication_snapshot, result)
                _close_publication_input_window(
                    result,
                    run_root=root,
                    cache_root=cache,
                    _fenced=True,
                )
                _assert_publication_snapshot(publication_snapshot)
            finally:
                if stage_descriptor >= 0:
                    os.close(stage_descriptor)
                os.close(parent)
        if publication_snapshot is None:
            raise ControlGridAnalysisError(
                "analysis publication produced no retained snapshot"
            )
        _verify_publication(publication_snapshot, result)
        _assert_publication_snapshot(publication_snapshot)
        if lock_identity is None:
            raise ControlGridAnalysisError(
                "analysis output lock identity was lost"
            )
        _release_output_lock(
            path=lock_path,
            descriptor=lock_descriptor,
            identity=lock_identity,
            payload=lock_payload,
        )
        lock_descriptor = -1
        lock_identity = None
        _assert_publication_snapshot(publication_snapshot)
        return destination
    except BaseException as error:
        active_error = error
        expected = (
            published_identity
            if published_identity is not None
            else stage_identity
        )
        if expected is not None:
            _hide_exact_analysis_directory(
                destination, expected_identity=expected
            )
            _hide_exact_analysis_directory(
                stage, expected_identity=expected
            )
        raise
    finally:
        if publication_snapshot is not None:
            publication_snapshot.close()
        if lock_descriptor >= 0 and lock_identity is not None:
            try:
                _release_output_lock(
                    path=lock_path,
                    descriptor=lock_descriptor,
                    identity=lock_identity,
                    payload=lock_payload,
                )
            except BaseException:
                if active_error is None:
                    raise


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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fail-closed aggregate analysis of the deterministic control grid"
        )
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--ece-bins", type=int, default=DEFAULT_ECE_BINS
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result, destination = analyze_and_publish(
        run_root=arguments.run_root,
        cache_root=arguments.cache_root,
        output_dir=arguments.output_dir,
        ece_bins=arguments.ece_bins,
    )
    print(
        json.dumps(
            {
                "output_dir": str(destination),
                "plan_sha256": result.summary["plan_sha256"],
                "track": TRACK,
                "records": result.summary["expected_jobs"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ANALYSIS_SCHEMA",
    "AnalysisResult",
    "ControlGridAnalysisError",
    "DEFAULT_ECE_BINS",
    "METRIC_NAMES",
    "TRACK",
    "analyze_and_publish",
    "audit_exact_grid",
    "classification_metrics",
    "compute_analysis",
    "main",
    "publish_analysis",
]
