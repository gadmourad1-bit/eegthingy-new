"""Audited, score-blind, resumable execution of the 43-model MI common grid.

This module intentionally separates *prediction production* from statistical
analysis.  Each atomic job writes only its identity, source-only fitting
metadata, test-row identities, and compressed class probabilities.  It never
writes test labels or computes a test score.  A later, separately invoked
analysis stage may join the frozen predictions to the cache labels.

The public ``run`` command is both the initial-run and resume interface::

    CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run python -m ieee_mi.full_grid run \
      --run-root results/full_grid \
      --cache-root data_cache

It launches at most three independent single-GPU workers (no DDP).  Work is
assigned dynamically through exclusive filesystem claims.  Before every new
claim, the worker asks ``nvidia-smi`` about its assigned physical GPU.  By
default a GPU is considered available only when it has no foreign compute PID,
utilization is at most 10 percent, and non-worker memory use is at most
1024 MiB.  There is no busy-GPU bypass.  A job that is already running is
allowed to finish if another user starts using the GPU, but publication
requires a second clean resource probe.  Every claim and commit checks a hard
50 GiB filesystem floor; it cannot be lowered or bypassed.  Both resource
snapshots are retained in the completed record.  Checkpoints and verbose
training traces are not stored; only state hashes and compressed predictions
are persisted.

The runner requires an activated virtual environment by default.  Its worker
subprocesses use ``sys.executable``, so a ``uv run``/``uv venv`` environment is
preserved and the system Python installation is never modified.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import errno
import fcntl
import hashlib
import importlib
import importlib.metadata
import inspect
import io
import json
import math
import os
import platform
import re
import socket
import stat
import subprocess
import sys
import sysconfig
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np

from . import project_gpu_leases
from .config import CHANNEL_SCALING, dataset_spec, preprocessing_for_dataset


PLAN_SCHEMA = "ieee-mi-score-blind-common-grid-plan-v5"
RECORD_SCHEMA = "ieee-mi-score-blind-common-prediction-record-v6"
COMPLETION_SCHEMA = "ieee-mi-score-blind-common-completion-v3"
CLAIM_SCHEMA = "ieee-mi-common-filesystem-claim-v4"
FAILURE_SCHEMA = "ieee-mi-common-grid-failure-v2"
AUDIT_SCHEMA = "ieee-mi-score-blind-common-grid-audit-v3"
PREFLIGHT_SCHEMA = "ieee-mi-common-track-cuda-preflight-v3"
PREFLIGHT_RECEIPT_SCHEMA = "ieee-mi-common-track-cuda-preflight-receipt-v1"
ANALYSIS_CONTRACT_SCHEMA = "ieee-mi-common-grid-analysis-contract-v2"
FORMAL_PUBLICATION_MODE = "formal_publishable"
TEST_PUBLICATION_MODE = "test_nonpublishable"
TRACK_SCOPE = "common_recipe_43_only"
FORMAL_EXPECTED_JOBS = 96_320
PUBLICATION_FENCE_FILENAME = ".common-grid-publication.fence"
CLAIM_TOMBSTONE_DIRECTORY = "claim_tombstones"
PREFLIGHT_DIRECTORY = "preflight"
PREFLIGHT_REPORT_FILENAME = "report.json"
PREFLIGHT_RECEIPT_FILENAME = "receipt.json"
PREFLIGHT_STAGE_RE = re.compile(
    r"^\.(?:report|receipt)\.json\.stage-[0-9a-f]{32}$"
)
QUARANTINE_REASONS: Mapping[str, frozenset[str]] = {
    "claims": frozenset(
        {"stale", "completed-stale", "lease-postcondition"}
    ),
    "partials": frozenset({"stale", "duplicate", "failed"}),
    "records": frozenset(
        {"corrupt", "lost-claim-after-rename", "lease-postcondition"}
    ),
    "preflight": frozenset(
        {
            "lease-postcondition-full",
            "lease-postcondition-report",
            "lease-postcondition-receipt",
        }
    ),
}

OPENED_DATASETS: tuple[str, ...] = (
    "local_exp4",
    "bnci2014_001",
    "bnci2014_004",
    "cho2017",
    "physionet_mi",
)
DATASET_FOLDS: Mapping[str, tuple[int, ...]] = {
    "local_exp4": (0,),
    "bnci2014_001": (0,),
    "bnci2014_004": (0,),
    "cho2017": (0, 1, 2, 3, 4),
    "physionet_mi": (0, 1, 2),
}
FORMAL_SEEDS: tuple[int, ...] = (7, 17, 27, 37, 47)
MAX_GPU_WORKERS = 3
FORMAL_GPU_INDICES: tuple[str, ...] = ("0", "1", "2")
REQUIRED_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
DEFAULT_MIN_FREE_GIB = 50.0
DEFAULT_MAX_IDLE_UTILIZATION_PERCENT = 10.0
DEFAULT_MAX_FOREIGN_MEMORY_MIB = 1024.0
PUBLICATION_GPU_COOLDOWN_SECONDS = 30.0
PUBLICATION_GPU_COOLDOWN_POLL_SECONDS = 0.25
DEFAULT_EPOCHS = 200
DEFAULT_PATIENCE = 35
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 8e-4
DEFAULT_WEIGHT_DECAY = 5e-4
FORMAL_TRAIN_CONFIG: Mapping[str, Any] = {
    "epochs": 200,
    "batch_size": 64,
    "learning_rate": 8e-4,
    "weight_decay": 5e-4,
    "patience": 35,
    "min_delta": 1e-4,
    "label_smoothing": 0.05,
    "segment_probability": 0.5,
    "segment_count": 8,
    "time_shift_samples": 8,
    "noise_std": 0.01,
    "reflection_weight": 0.0,
    "reflection_probability": 0.0,
    "gradient_clip": 5.0,
    "seed": 7,
    "device": "cuda",
}
FORMAL_ANALYSIS_BOOTSTRAP_RESAMPLES = 100_000
FORMAL_ANALYSIS_BOOTSTRAP_SEED = 20_260_729
FORMAL_ANALYSIS_ECE_BINS = 15
FORMAL_ANALYSIS_TCFORMER_COMPARATOR = "tcformer"
THREAD_ENVIRONMENT_VARIABLES: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

# The exact 43 common-recipe architectures from the completed local
# tournament.  Width/dropout aliases that were not distinct architectures are
# deliberately absent.  No experimental architecture can be appended to a
# formal common-track plan.
COMMON_ARCHITECTURES: tuple[str, ...] = (
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

SOURCE_FILES: tuple[str, ...] = (
    "full_grid.py",
    "project_gpu_leases.py",
    "benchmark.py",
    "baselines.py",
    "tcformer_source.py",
    "models.py",
    "training.py",
    "data.py",
    "config.py",
)
ANALYSIS_SOURCE_FILES: tuple[str, ...] = ("full_grid_analysis.py",)
MODEL_PREFLIGHT_CONTRACTS: tuple[dict[str, Any], ...] = (
    {
        "name": "local_exp4_15ch_256t_binary",
        "dataset": "local_exp4",
        "subject": 1,
        "n_channels": 15,
        "n_times": 256,
        "n_outputs": 2,
    },
    {
        "name": "bnci2014_004_3ch_320t_binary",
        "dataset": "bnci2014_004",
        "subject": 1,
        "n_channels": 3,
        "n_times": 320,
        "n_outputs": 2,
    },
    {
        "name": "cho2017_21ch_320t_binary",
        "dataset": "cho2017",
        "subject": 1,
        "n_channels": 21,
        "n_times": 320,
        "n_outputs": 2,
    },
    {
        "name": "bnci2014_001_21ch_320t_four_class",
        "dataset": "bnci2014_001",
        "subject": 1,
        "n_channels": 21,
        "n_times": 320,
        "n_outputs": 4,
    },
)
DEFAULT_EXECUTOR = "ieee_mi.full_grid:execute_benchmark_job"
MODEL_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")
HEX_64_RE = re.compile(r"^[0-9a-f]{64}$")
GPU_UUID_RE = re.compile(
    r"^GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
_LOCAL_BOOT_FALLBACK = uuid.uuid4().hex
_LOCAL_PROCESS_START_FALLBACK = uuid.uuid4().hex
FINAL_FILENAMES = frozenset({"record.json", "predictions.npz", "completion.json"})
FORBIDDEN_SCORE_BLIND_KEYS = frozenset(
    {
        "y",
        "target",
        "targets",
        "label",
        "labels",
        "test_label",
        "test_labels",
        "predicted_label",
        "predicted_labels",
        "metric",
        "metrics",
        "score",
        "scores",
        "accuracy",
        "balanced_accuracy",
        "cohen_kappa",
        "roc_auc",
        "negative_log_likelihood",
    }
)
OUTCOME_KEY_TOKENS = frozenset(
    {
        "accuracy",
        "auc",
        "balancedaccuracy",
        "brier",
        "calibration",
        "confusion",
        "ece",
        "f1",
        "kappa",
        "label",
        "metric",
        "nll",
        "predictedclass",
        "score",
        "target",
        "ytrue",
    }
)
LEGITIMATE_OUTCOME_TOKEN_KEYS = frozenset(
    {
        "gpu_lease_receipt",
        "score_blind",
        "test_performance_computed",
    }
)
REQUIRED_UV_RUNTIME_PACKAGES = frozenset(
    {
        "braindecode",
        "mne",
        "moabb",
        "numpy",
        "scikit-learn",
        "scipy",
        "torch",
    }
)
REQUIRED_UV_TEST_PACKAGES = frozenset({"pytest"})


class FullGridError(RuntimeError):
    """Base error for runner contract violations."""


class ClaimUnavailable(FullGridError):
    """Raised when another live process owns a job claim."""


class GPUUnavailable(FullGridError):
    """Raised when the cooperative GPU guard refuses a new claim."""


class DiskUnavailable(FullGridError):
    """Raised when the shared output filesystem is below its low-water mark."""


@dataclass(frozen=True, order=True)
class Job:
    dataset: str
    model: str
    subject: int
    fold: int
    seed: int

    def identity(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "model": self.model,
            "subject": self.subject,
            "fold": self.fold,
            "seed": self.seed,
        }

    @property
    def job_id(self) -> str:
        digest = hashlib.sha256(_canonical_bytes(self.identity())).hexdigest()
        return f"job-{digest[:24]}"


@dataclass(frozen=True)
class Claim:
    job: Job
    path: Path
    nonce: str
    owner: Mapping[str, Any]
    resource_guard: Mapping[str, Any]
    gpu_lease_receipt: Mapping[str, Any] | None
    preflight_report_sha256: str | None
    st_dev: int
    st_ino: int
    value: Mapping[str, Any]
    fence_fd: int


@dataclass(frozen=True)
class GPUStatus:
    safe: bool
    gpu: str
    utilization_percent: float | None
    memory_used_mib: float | None
    own_compute_memory_mib: float
    foreign_processes: tuple[dict[str, Any], ...]
    reason: str
    gpu_uuid: str | None = None
    pci_bus_id: str | None = None
    name: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GPUIdentity:
    index: str
    uuid: str
    pci_bus_id: str
    name: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class DiskStatus:
    safe: bool
    path: str
    free_bytes: int | None
    total_bytes: int | None
    free_gib: float | None
    minimum_free_gib: float
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResourceStatus:
    safe: bool
    gpu: Mapping[str, Any]
    disk: Mapping[str, Any]
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "safe": self.safe,
            "gpu": dict(self.gpu),
            "disk": dict(self.disk),
            "reason": self.reason,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _directory_open_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _open_absolute_directory(
    path: Path,
    *,
    create_leaf: bool = False,
    mode: int = 0o700,
) -> int:
    """Open an absolute directory by an anchored no-follow component walk."""

    absolute = _absolute_path(path)
    descriptor = os.open(absolute.anchor, _directory_open_flags())
    try:
        for index, component in enumerate(absolute.parts[1:]):
            try:
                child = os.open(
                    component,
                    _directory_open_flags(),
                    dir_fd=descriptor,
                )
            except FileNotFoundError:
                if not create_leaf or index != len(absolute.parts[1:]) - 1:
                    raise
                os.mkdir(component, mode, dir_fd=descriptor)
                os.fsync(descriptor)
                child = os.open(
                    component,
                    _directory_open_flags(),
                    dir_fd=descriptor,
                )
            observed = os.fstat(child)
            if not stat.S_ISDIR(observed.st_mode):
                os.close(child)
                raise FullGridError(f"{absolute} has a non-directory ancestor")
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _assert_directory_descriptor_path(path: Path, descriptor: int) -> None:
    """Require the absolute path to resolve to the held directory descriptor."""

    replacement = _open_absolute_directory(path)
    try:
        held = os.fstat(descriptor)
        current = os.fstat(replacement)
        if (
            held.st_dev != current.st_dev
            or held.st_ino != current.st_ino
            or not stat.S_ISDIR(held.st_mode)
        ):
            raise FullGridError(f"directory path changed during transaction: {path}")
    finally:
        os.close(replacement)


def _open_relative_directory(
    root_descriptor: int,
    relative: str | Path,
    *,
    create: bool,
    mode: int = 0o700,
) -> int:
    """Open a directory below one held root without resolving another pathname."""

    descriptor = os.dup(root_descriptor)
    try:
        for component in Path(relative).parts:
            if component in {"", ".", ".."} or os.sep in component:
                raise FullGridError(f"invalid relative directory {relative}")
            try:
                child = os.open(
                    component,
                    _directory_open_flags(),
                    dir_fd=descriptor,
                )
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(component, mode, dir_fd=descriptor)
                os.fsync(descriptor)
                child = os.open(
                    component,
                    _directory_open_flags(),
                    dir_fd=descriptor,
                )
            observed = os.fstat(child)
            if not stat.S_ISDIR(observed.st_mode):
                os.close(child)
                raise FullGridError(f"unsafe relative directory component: {component}")
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _safe_run_root(run_root: Path, *, create: bool = False) -> Path:
    root = _absolute_path(run_root)
    try:
        descriptor = _open_absolute_directory(root, create_leaf=create)
    except FileNotFoundError:
        if create:
            raise
        return root
    except OSError as error:
        raise FullGridError(
            "run root has a symlinked/non-directory ancestor"
        ) from error
    try:
        _assert_directory_descriptor_path(root, descriptor)
    finally:
        os.close(descriptor)
    return root


def _require_real_directory(path: Path) -> os.stat_result:
    try:
        descriptor = _open_absolute_directory(path)
    except OSError as error:
        raise FullGridError(f"{path} is not a real directory") from error
    try:
        observed = os.fstat(descriptor)
        _assert_directory_descriptor_path(path, descriptor)
        return observed
    finally:
        os.close(descriptor)


def _ensure_real_directory(root: Path, relative: str | Path) -> Path:
    base = _safe_run_root(root, create=True)
    root_descriptor = _open_absolute_directory(base)
    try:
        descriptor = _open_relative_directory(
            root_descriptor,
            relative,
            create=True,
        )
        try:
            _assert_directory_descriptor_path(base, root_descriptor)
        finally:
            os.close(descriptor)
    finally:
        os.close(root_descriptor)
    return base / relative


def _require_real_path_components(root: Path, path: Path) -> None:
    base = _safe_run_root(root)
    try:
        relative = _absolute_path(path).relative_to(base)
    except ValueError as error:
        raise FullGridError(f"path escapes run root: {path}") from error
    current = base
    for part in relative.parts:
        current /= part
        _require_real_directory(current)


def _require_unique_regular(path: Path, *, read_only: bool = False) -> os.stat_result:
    absolute = _absolute_path(path)
    parent_descriptor = _open_absolute_directory(absolute.parent)
    try:
        try:
            observed = os.stat(
                absolute.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except OSError as error:
            raise FullGridError(f"{path} is not a regular file") from error
        if (
            stat.S_ISLNK(observed.st_mode)
            or not stat.S_ISREG(observed.st_mode)
            or observed.st_nlink != 1
            or (read_only and observed.st_mode & 0o222)
        ):
            qualifier = "read-only " if read_only else ""
            raise FullGridError(f"{path} is not a unique {qualifier}regular file")
        _assert_directory_descriptor_path(absolute.parent, parent_descriptor)
        return observed
    finally:
        os.close(parent_descriptor)


def _read_unique_regular(path: Path) -> bytes:
    absolute = _absolute_path(path)
    parent_descriptor = _open_absolute_directory(absolute.parent)
    try:
        path_stat = os.stat(
            absolute.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        descriptor = os.open(
            absolute.name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
            dir_fd=parent_descriptor,
        )
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
            stat.S_ISLNK(path_stat.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or any(
                getattr(before, name) != getattr(path_stat, name)
                for name in stable_fields
            )
        ):
            raise FullGridError(f"{path} is not a unique regular file")
        try:
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
            final_path_stat = os.stat(
                absolute.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if any(
                getattr(before, name) != getattr(observed, name)
                for observed in (after, final_path_stat)
                for name in stable_fields
            ):
                raise FullGridError(f"{path} changed during its verified read")
            payload = b"".join(chunks)
            if len(payload) != before.st_size:
                raise FullGridError(f"{path} read was incomplete")
            _assert_directory_descriptor_path(absolute.parent, parent_descriptor)
            return payload
        finally:
            os.close(descriptor)
    finally:
        os.close(parent_descriptor)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(_read_unique_regular(path)).hexdigest()


def _rows_sha256(values: np.ndarray | Sequence[int]) -> str:
    """Hash one row-index vector in a platform-independent representation."""

    rows = np.asarray(values, dtype="<i8")
    if rows.ndim != 1:
        raise ValueError("row indices must be one-dimensional")
    rows = np.ascontiguousarray(rows)
    digest = hashlib.sha256()
    digest.update(b"ieee-mi-row-indices-v1\0")
    digest.update(np.asarray(rows.shape, dtype="<i8").tobytes())
    digest.update(rows.tobytes())
    return digest.hexdigest()


def _subject_identity_key(dataset: str, subject: int) -> str:
    return f"{dataset}:s{int(subject):03d}"


def _split_identity_key(dataset: str, subject: int, fold: int) -> str:
    return f"{dataset}:s{int(subject):03d}:f{int(fold):02d}"


def _partition_identity(rows: np.ndarray) -> dict[str, Any]:
    values = np.asarray(rows)
    if values.dtype != np.dtype(np.int64) or values.ndim != 1:
        raise FullGridError("split rows must be one-dimensional int64 arrays")
    if len(values) == 0 or np.any(values < 0) or len(np.unique(values)) != len(values):
        raise FullGridError("split rows must be nonempty, nonnegative, and unique")
    return {
        "count": len(values),
        "rows_sha256": _rows_sha256(values),
    }


def _one_split_identity(
    *,
    dataset: str,
    subject: int,
    fold: int,
    cache_array_sha256: str,
    trial_count: int,
    train_rows: np.ndarray,
    validation_rows: np.ndarray,
    test_rows: np.ndarray,
) -> dict[str, Any]:
    source_rows = np.sort(np.concatenate((train_rows, validation_rows))).astype(
        np.int64,
        copy=False,
    )
    return {
        "dataset": dataset,
        "subject": int(subject),
        "fold": int(fold),
        "cache_array_sha256": cache_array_sha256,
        "trial_count": int(trial_count),
        "partitions": {
            "train": _partition_identity(train_rows),
            "validation": _partition_identity(validation_rows),
            "source": _partition_identity(source_rows),
            "test": _partition_identity(test_rows),
        },
    }


def _validate_split_identity_exact(
    identity: Any,
    *,
    dataset: str,
    subject: int,
    fold: int,
    cache_array_sha256: str,
) -> None:
    _exact_keys(
        identity,
        {
            "dataset",
            "subject",
            "fold",
            "cache_array_sha256",
            "trial_count",
            "partitions",
        },
        path=f"split_identity.{dataset}.S{subject}.F{fold}",
    )
    if (
        identity["dataset"] != dataset
        or type(identity["subject"]) is not int
        or identity["subject"] != subject
        or type(identity["fold"]) is not int
        or identity["fold"] != fold
        or identity["cache_array_sha256"] != cache_array_sha256
        or type(identity["trial_count"]) is not int
        or identity["trial_count"] <= 0
    ):
        raise FullGridError("split identity fields differ from their grid key")
    partitions = identity["partitions"]
    _exact_keys(
        partitions,
        {"train", "validation", "source", "test"},
        path=f"split_identity.{dataset}.partitions",
    )
    for name, partition in partitions.items():
        _exact_keys(
            partition,
            {"count", "rows_sha256"},
            path=f"split_identity.{dataset}.partitions.{name}",
        )
        if (
            type(partition["count"]) is not int
            or partition["count"] <= 0
            or not isinstance(partition["rows_sha256"], str)
            or HEX_64_RE.fullmatch(partition["rows_sha256"]) is None
        ):
            raise FullGridError(f"split partition identity is invalid: {name}")
    counts = {name: partitions[name]["count"] for name in partitions}
    if (
        counts["source"] != counts["train"] + counts["validation"]
        or counts["source"] + counts["test"] != identity["trial_count"]
    ):
        raise FullGridError("split identity counts do not cover the cache")


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
    try:
        payload = _read_unique_regular(path).decode("utf-8")
    except UnicodeDecodeError as error:
        raise FullGridError(f"{path} is not UTF-8 JSON") from error
    return _strict_load_text(payload, source=str(path))


def _strict_load_text(payload: str, *, source: str) -> dict[str, Any]:
    value = json.loads(
        payload,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{source} does not contain a JSON object")
    return value


def _strict_load_bytes(payload: bytes, *, source: str) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise FullGridError(f"{source} is not UTF-8 JSON") from error
    return _strict_load_text(text, source=source)


def _strict_load_descriptor(descriptor: int, *, source: str) -> dict[str, Any]:
    observed = os.fstat(descriptor)
    if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
        raise FullGridError(f"{source} is not a unique regular file")
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    try:
        value = json.loads(
            b"".join(chunks).decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except UnicodeDecodeError as error:
        raise FullGridError(f"{source} is not UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise FullGridError(f"{source} is not a JSON object")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = _open_absolute_directory(path)
    try:
        os.fsync(descriptor)
        _assert_directory_descriptor_path(path, descriptor)
    finally:
        os.close(descriptor)


def _atomic_rename_noreplace_at(
    source_descriptor: int,
    source_name: str,
    destination_descriptor: int,
    destination_name: str,
) -> None:
    """Atomically rename relative to held directories without replacement."""

    if (
        source_name in {"", ".", ".."}
        or destination_name in {"", ".", ".."}
        or os.sep in source_name
        or os.sep in destination_name
    ):
        raise FullGridError("atomic rename requires single safe path components")
    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source_name)
    destination_bytes = os.fsencode(destination_name)
    if hasattr(libc, "renameat2"):
        function = libc.renameat2
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        result = function(
            source_descriptor,
            source_bytes,
            destination_descriptor,
            destination_bytes,
            1,
        )
    elif hasattr(libc, "renameatx_np"):
        function = libc.renameatx_np
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        result = function(
            source_descriptor,
            source_bytes,
            destination_descriptor,
            destination_bytes,
            0x00000004,
        )
    else:
        raise FullGridError(
            "platform has no dirfd-relative atomic no-replace rename primitive"
        )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(destination_name)
        raise OSError(error_number, os.strerror(error_number), destination_name)


_SEALED_TREE_STABLE_FIELDS: tuple[str, ...] = (
    "st_dev",
    "st_ino",
    "st_mode",
    "st_nlink",
    "st_size",
    "st_mtime_ns",
    "st_ctime_ns",
)


def _sealed_tree_fingerprint(observed: os.stat_result) -> tuple[int, ...]:
    return tuple(
        int(getattr(observed, field))
        for field in _SEALED_TREE_STABLE_FIELDS
    )


def _assert_exact_sealed_directory_at(
    parent_descriptor: int,
    name: str,
    directory_descriptor: int,
    *,
    expected_identity: tuple[int, int] | None,
    expected_names: frozenset[str],
    require_readonly_regular_children: bool,
    expected_child_identities: Mapping[str, tuple[int, ...]] | None = None,
) -> tuple[os.stat_result, dict[str, tuple[int, ...]]]:
    """Bind one exact flat sealed tree to its held directory descriptor."""

    held = os.fstat(directory_descriptor)
    path_stat = os.stat(
        name,
        dir_fd=parent_descriptor,
        follow_symlinks=False,
    )
    identity = (held.st_dev, held.st_ino)
    if (
        not stat.S_ISDIR(held.st_mode)
        or stat.S_IMODE(held.st_mode) != 0o555
        or identity != (path_stat.st_dev, path_stat.st_ino)
        or (
            expected_identity is not None
            and identity != expected_identity
        )
    ):
        raise FullGridError("staged publication directory is not exactly sealed")
    observed_names = frozenset(os.listdir(directory_descriptor))
    if observed_names != expected_names:
        raise FullGridError("staged publication directory has a non-exact tree")
    child_identities: dict[str, tuple[int, ...]] = {}
    for child_name in sorted(expected_names):
        child = os.stat(
            child_name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if require_readonly_regular_children:
            if (
                not stat.S_ISREG(child.st_mode)
                or child.st_nlink != 1
                or stat.S_IMODE(child.st_mode) != 0o444
            ):
                raise FullGridError(
                    "staged publication tree contains an unsafe artifact"
                )
        elif (
            stat.S_ISLNK(child.st_mode)
            or not (
                stat.S_ISREG(child.st_mode)
                or stat.S_ISDIR(child.st_mode)
            )
            or (stat.S_ISREG(child.st_mode) and child.st_nlink != 1)
        ):
            raise FullGridError(
                "staged publication tree contains an unsafe node"
            )
        child_identities[child_name] = _sealed_tree_fingerprint(child)
    if (
        expected_child_identities is not None
        and child_identities != dict(expected_child_identities)
    ):
        raise FullGridError("staged publication tree identities changed")
    final_held = os.fstat(directory_descriptor)
    final_path = os.stat(
        name,
        dir_fd=parent_descriptor,
        follow_symlinks=False,
    )
    final_child_identities = {
        child_name: _sealed_tree_fingerprint(
            os.stat(
                child_name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        )
        for child_name in sorted(expected_names)
    }
    if (
        _sealed_tree_fingerprint(final_held)
        != _sealed_tree_fingerprint(held)
        or _sealed_tree_fingerprint(final_path)
        != _sealed_tree_fingerprint(held)
        or frozenset(os.listdir(directory_descriptor)) != expected_names
        or final_child_identities != child_identities
    ):
        raise FullGridError("staged publication directory changed during validation")
    return held, child_identities


def _publish_sealed_directory_noreplace_at(
    source_descriptor: int,
    source_name: str,
    sealed_directory_descriptor: int,
    destination_descriptor: int,
    destination_name: str,
    *,
    expected_names: frozenset[str] = FINAL_FILENAMES,
    require_readonly_regular_children: bool = True,
) -> None:
    """Atomically move one exact pre-sealed tree and retain its held inode.

    Darwin needs the historical pre-rename compatibility transition.  Some
    Linux/NFS mounts report ``EACCES``/``EPERM`` for the same sealed-directory
    rename.  Linux first attempts the continuously sealed rename and uses the
    transition only for those two permission errors, after a second exact-tree
    and inode check.  Every attempted move re-seals and fsyncs the held inode,
    including failure paths.
    """

    frozen_names = frozenset(expected_names)
    staged, staged_child_identities = _assert_exact_sealed_directory_at(
        source_descriptor,
        source_name,
        sealed_directory_descriptor,
        expected_identity=None,
        expected_names=frozen_names,
        require_readonly_regular_children=require_readonly_regular_children,
    )
    staged_identity = (staged.st_dev, staged.st_ino)
    temporarily_unsealed = sys.platform == "darwin"

    def transition_to_temporary_writable(*, context: str) -> None:
        os.fchmod(sealed_directory_descriptor, 0o700)
        os.fsync(sealed_directory_descriptor)
        transition_path = os.stat(
            source_name,
            dir_fd=source_descriptor,
            follow_symlinks=False,
        )
        transition_held = os.fstat(sealed_directory_descriptor)
        transition_child_identities = {
            child_name: _sealed_tree_fingerprint(
                os.stat(
                    child_name,
                    dir_fd=sealed_directory_descriptor,
                    follow_symlinks=False,
                )
            )
            for child_name in sorted(frozen_names)
        }
        if (
            (transition_path.st_dev, transition_path.st_ino)
            != staged_identity
            or (transition_held.st_dev, transition_held.st_ino)
            != staged_identity
            or stat.S_IMODE(transition_held.st_mode) != 0o700
            or frozenset(os.listdir(sealed_directory_descriptor))
            != frozen_names
            or transition_child_identities != staged_child_identities
        ):
            raise FullGridError(
                f"staged publication directory changed before {context}"
            )

    moved = False
    try:
        if temporarily_unsealed:
            _assert_exact_sealed_directory_at(
                source_descriptor,
                source_name,
                sealed_directory_descriptor,
                expected_identity=staged_identity,
                expected_names=frozen_names,
                require_readonly_regular_children=(
                    require_readonly_regular_children
                ),
                expected_child_identities=staged_child_identities,
            )
            transition_to_temporary_writable(context="Darwin rename")
        try:
            _atomic_rename_noreplace_at(
                source_descriptor,
                source_name,
                destination_descriptor,
                destination_name,
            )
        except OSError as error:
            if (
                temporarily_unsealed
                or error.errno not in {errno.EACCES, errno.EPERM}
            ):
                raise
            _assert_exact_sealed_directory_at(
                source_descriptor,
                source_name,
                sealed_directory_descriptor,
                expected_identity=staged_identity,
                expected_names=frozen_names,
                require_readonly_regular_children=(
                    require_readonly_regular_children
                ),
                expected_child_identities=staged_child_identities,
            )
            transition_to_temporary_writable(context="NFS retry")
            temporarily_unsealed = True
            _atomic_rename_noreplace_at(
                source_descriptor,
                source_name,
                destination_descriptor,
                destination_name,
            )
        moved = True
    finally:
        try:
            os.fchmod(sealed_directory_descriptor, 0o555)
        finally:
            os.fsync(sealed_directory_descriptor)
    if moved:
        _assert_exact_sealed_directory_at(
            destination_descriptor,
            destination_name,
            sealed_directory_descriptor,
            expected_identity=staged_identity,
            expected_names=frozen_names,
            require_readonly_regular_children=require_readonly_regular_children,
            expected_child_identities=staged_child_identities,
        )


def _atomic_rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically rename using stable source/destination parent descriptors."""

    source_absolute = _absolute_path(source)
    destination_absolute = _absolute_path(destination)
    source_parent = _open_absolute_directory(source_absolute.parent)
    try:
        destination_parent = _open_absolute_directory(destination_absolute.parent)
        try:
            _atomic_rename_noreplace_at(
                source_parent,
                source_absolute.name,
                destination_parent,
                destination_absolute.name,
            )
            os.fsync(source_parent)
            source_parent_stat = os.fstat(source_parent)
            destination_parent_stat = os.fstat(destination_parent)
            if (
                source_parent_stat.st_dev,
                source_parent_stat.st_ino,
            ) != (
                destination_parent_stat.st_dev,
                destination_parent_stat.st_ino,
            ):
                os.fsync(destination_parent)
            _assert_directory_descriptor_path(
                source_absolute.parent,
                source_parent,
            )
            _assert_directory_descriptor_path(
                destination_absolute.parent,
                destination_parent,
            )
        finally:
            os.close(destination_parent)
    finally:
        os.close(source_parent)


def _write_bytes_exclusive_at(
    parent_descriptor: int,
    leaf_name: str,
    payload: bytes,
    *,
    mode: int = 0o444,
    on_publish: Callable[[os.stat_result], None] | None = None,
) -> os.stat_result:
    """Crash-atomically publish one immutable leaf below a held parent FD."""

    if leaf_name in {"", ".", ".."} or os.sep in leaf_name:
        raise FullGridError("exclusive write has no safe leaf name")
    stage_name = f".{leaf_name}.stage-{uuid.uuid4().hex}"
    descriptor: int | None = None
    published = False
    parent_locked = False
    try:
        # Serialize stage creation/publication/cleanup for one parent.  Without
        # this lock, a concurrent writer can remove another writer's live
        # stage while the latter is still validating its descriptor.
        fcntl.flock(parent_descriptor, fcntl.LOCK_EX)
        parent_locked = True
        descriptor = os.open(
            stage_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
            0o600,
            dir_fd=parent_descriptor,
        )
        view = memoryview(payload)
        written = 0
        while written < len(view):
            try:
                count = os.write(descriptor, view[written:])
            except InterruptedError:
                continue
            if count <= 0:
                raise OSError("short write while staging immutable artifact")
            written += count
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
        staged_stat = os.fstat(descriptor)
        if (
            not stat.S_ISREG(staged_stat.st_mode)
            or staged_stat.st_nlink != 1
            or staged_stat.st_size != len(payload)
            or staged_stat.st_mode & 0o222
        ):
            raise FullGridError("staged immutable artifact failed descriptor checks")
        os.close(descriptor)
        descriptor = None
        _atomic_rename_noreplace_at(
            parent_descriptor,
            stage_name,
            parent_descriptor,
            leaf_name,
        )
        published = True
        if on_publish is not None:
            on_publish(staged_stat)
        os.fsync(parent_descriptor)
        final_stat = os.stat(
            leaf_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(final_stat.st_mode)
            or final_stat.st_nlink != 1
            or final_stat.st_size != len(payload)
            or final_stat.st_mode & 0o222
            or (final_stat.st_dev, final_stat.st_ino)
            != (staged_stat.st_dev, staged_stat.st_ino)
        ):
            raise FullGridError("published immutable artifact failed identity checks")
        stage_pattern = re.compile(
            rf"^\.{re.escape(leaf_name)}\.stage-[0-9a-f]{{32}}$"
        )
        for sibling in os.listdir(parent_descriptor):
            if stage_pattern.fullmatch(sibling):
                try:
                    os.unlink(sibling, dir_fd=parent_descriptor)
                except FileNotFoundError:
                    pass
        os.fsync(parent_descriptor)
        return final_stat
    finally:
        try:
            if descriptor is not None:
                os.close(descriptor)
            if not published:
                try:
                    os.unlink(stage_name, dir_fd=parent_descriptor)
                    os.fsync(parent_descriptor)
                except FileNotFoundError:
                    pass
        finally:
            if parent_locked:
                fcntl.flock(parent_descriptor, fcntl.LOCK_UN)


def _write_bytes_exclusive(
    path: Path,
    payload: bytes,
    *,
    mode: int = 0o444,
    on_publish: Callable[[os.stat_result], None] | None = None,
) -> os.stat_result:
    absolute = _absolute_path(path)
    parent_descriptor = _open_absolute_directory(absolute.parent)
    try:
        published_stat = _write_bytes_exclusive_at(
            parent_descriptor,
            absolute.name,
            payload,
            mode=mode,
            on_publish=on_publish,
        )
        _assert_directory_descriptor_path(absolute.parent, parent_descriptor)
        return published_stat
    finally:
        os.close(parent_descriptor)


def _read_unique_regular_at(parent_descriptor: int, leaf_name: str) -> bytes:
    if leaf_name in {"", ".", ".."} or os.sep in leaf_name:
        raise FullGridError("descriptor read requires one safe leaf")
    descriptor = os.open(
        leaf_name,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0),
        dir_fd=parent_descriptor,
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise FullGridError("descriptor leaf is not a unique regular file")
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
        path_stat = os.stat(
            leaf_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(
            getattr(before, name) != getattr(observed, name)
            for observed in (after, path_stat)
            for name in stable_fields
        ):
            raise FullGridError("descriptor leaf changed during read")
        payload = b"".join(chunks)
        if len(payload) != before.st_size:
            raise FullGridError("descriptor leaf read was incomplete")
        return payload
    finally:
        os.close(descriptor)


def _write_json_exclusive(
    path: Path,
    value: Mapping[str, Any],
    *,
    on_publish: Callable[[os.stat_result], None] | None = None,
) -> os.stat_result:
    return _write_bytes_exclusive(
        path,
        _canonical_bytes(dict(value)) + b"\n",
        on_publish=on_publish,
    )


def _exact_keys(value: Any, expected: set[str] | frozenset[str], *, path: str) -> None:
    if not isinstance(value, Mapping) or set(value) != set(expected):
        observed = (
            sorted(str(key) for key in value)
            if isinstance(value, Mapping)
            else []
        )
        raise FullGridError(
            f"{path} schema is invalid; expected {sorted(expected)}, got {observed}"
        )


def _recursive_outcome_aliases(value: Any, path: str = "record") -> list[str]:
    violations: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).strip().lower()
            compact = re.sub(r"[^a-z0-9]+", "", key)
            public_filterbank_f1 = (
                raw_key == "F1"
                and path == "record.metadata.fit.architecture.scalar_attributes"
                and child in {8, 32}
            )
            if (
                not public_filterbank_f1
                and key not in LEGITIMATE_OUTCOME_TOKEN_KEYS
                and any(
                token in compact for token in OUTCOME_KEY_TOKENS
                )
            ):
                violations.append(f"{path}.{raw_key}")
            violations.extend(
                _recursive_outcome_aliases(child, f"{path}.{raw_key}")
            )
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            violations.extend(
                _recursive_outcome_aliases(child, f"{path}[{index}]")
            )
    return violations


def _resolve_callable(reference: str) -> Callable[..., Any]:
    module_name, separator, attribute = reference.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("callable references must use module.path:attribute")
    value: Any = importlib.import_module(module_name)
    for token in attribute.split("."):
        value = getattr(value, token)
    if not callable(value):
        raise TypeError(f"{reference!r} is not callable")
    return value


def _callable_file(reference: str) -> Path:
    value = _resolve_callable(reference)
    source = inspect.getsourcefile(value) or inspect.getfile(value)
    return _absolute_path(Path(source))


def validate_uv_dependency_closure(
    pyproject_text: str,
    uv_lock_text: str,
) -> None:
    """Validate the final runtime/test lock closure without installing anything."""

    import tomllib

    try:
        project = tomllib.loads(pyproject_text)
        lock = tomllib.loads(uv_lock_text)
    except (tomllib.TOMLDecodeError, TypeError) as error:
        raise FullGridError("UV dependency manifests are invalid TOML") from error

    def names(values: Any) -> set[str]:
        if not isinstance(values, list):
            return set()
        result: set[str] = set()
        for value in values:
            if not isinstance(value, str):
                continue
            match = re.match(r"^[A-Za-z0-9_.-]+", value.strip())
            if match:
                result.add(match.group(0).lower().replace("_", "-"))
        return result

    runtime = names(project.get("project", {}).get("dependencies", []))
    dependency_groups = project.get("dependency-groups", {})
    optional = project.get("project", {}).get("optional-dependencies", {})
    tests: set[str] = set()
    for container in (dependency_groups, optional):
        if isinstance(container, Mapping):
            for group in ("test", "dev"):
                tests.update(names(container.get(group, [])))
    locked = {
        str(value.get("name", "")).lower().replace("_", "-")
        for value in lock.get("package", [])
        if isinstance(value, Mapping)
    }
    uv_settings = project.get("tool", {}).get("uv", {})
    if (
        not REQUIRED_UV_RUNTIME_PACKAGES.issubset(runtime)
        or not REQUIRED_UV_TEST_PACKAGES.issubset(tests)
        or not (
            REQUIRED_UV_RUNTIME_PACKAGES | REQUIRED_UV_TEST_PACKAGES
        ).issubset(locked)
        or not isinstance(uv_settings, Mapping)
        or uv_settings.get("default-groups") != []
    ):
        raise FullGridError(
            "UV closure must lock runtime EEG/Torch dependencies, keep pytest "
            "test-only, and set [tool.uv] default-groups=[]"
        )


def _source_identity(*, executor: str = DEFAULT_EXECUTOR) -> dict[str, str]:
    package_root = _absolute_path(Path(__file__)).parent
    project_root = package_root.parent
    paths: dict[str, Path] = {
        f"ieee_mi/{name}": package_root / name for name in SOURCE_FILES
    }
    for name in ("pyproject.toml", "uv.lock"):
        path = project_root / name
        if not path.is_file():
            raise FullGridError(
                f"formal source identity requires project-local {name}; "
                "create and freeze the dedicated UV project before planning"
            )
        paths[name] = path
    validate_uv_dependency_closure(
        _read_unique_regular(paths["pyproject.toml"]).decode("utf-8"),
        _read_unique_regular(paths["uv.lock"]).decode("utf-8"),
    )
    for label, reference in (("executor", executor),):
        path = _callable_file(reference)
        try:
            relative = path.relative_to(project_root)
            key = str(relative)
        except ValueError:
            key = f"external:{label}:{reference}:{path.name}"
        paths[key] = path
    return {name: _sha256_file(path) for name, path in sorted(paths.items())}


def _validate_tcformer_source_identity(value: Any) -> None:
    from .tcformer_source import (
        TCFORMER_COMMIT,
        TCFORMER_LICENSE,
        TCFORMER_PINNED_FILES,
        TCFORMER_REPOSITORY,
        canonical_manifest_bytes,
    )

    _exact_keys(
        value,
        {
            "source_mode",
            "repository",
            "source_root",
            "commit",
            "license",
            "manifest_sha256",
            "file_identity",
        },
        path="tcformer_source_identity",
    )
    if (
        value["source_mode"] != "vendored_runtime_snapshot"
        or value["repository"] != TCFORMER_REPOSITORY
        or value["commit"] != TCFORMER_COMMIT
        or value["license"] != TCFORMER_LICENSE
        or not isinstance(value["source_root"], str)
        or not os.path.isabs(value["source_root"])
        or value["manifest_sha256"]
        != hashlib.sha256(canonical_manifest_bytes()).hexdigest()
        or value["file_identity"]
        != {
            name: dict(contract)
            for name, contract in sorted(TCFORMER_PINNED_FILES.items())
        }
    ):
        raise FullGridError("TCFormer source identity differs from the pinned source")


def _current_tcformer_source_identity() -> dict[str, Any]:
    from .tcformer_source import verify_tcformer_source

    identity = verify_tcformer_source()
    _validate_tcformer_source_identity(identity)
    return identity


def _nvidia_driver_versions() -> list[str] | None:
    """Return the sorted physical-driver inventory, or fail-closed identity."""

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return sorted(
        {
            line.strip()
            for line in result.stdout.splitlines()
            if line.strip()
        }
    )


def _nvidia_gpu_inventory() -> list[dict[str, str]] | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,pci.bus_id,name,driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    inventory: list[dict[str, str]] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        fields = [value.strip() for value in line.split(",")]
        if (
            len(fields) != 5
            or not fields[0].isdigit()
            or not GPU_UUID_RE.fullmatch(fields[1])
            or not all(fields[2:])
        ):
            raise FullGridError(f"invalid nvidia-smi inventory row: {line!r}")
        inventory.append(
            {
                "index": fields[0],
                "uuid": fields[1],
                "pci_bus_id": fields[2],
                "name": fields[3],
                "driver_version": fields[4],
            }
        )
    if not inventory or len({row["uuid"] for row in inventory}) != len(inventory):
        raise FullGridError("nvidia-smi returned an invalid physical GPU inventory")
    return sorted(inventory, key=lambda row: int(row["index"]))


def _environment_distribution_paths() -> tuple[str, ...]:
    """Return the fixed installation roots for this Python environment."""

    paths = tuple(
        sorted(
            {
                str(Path(value).resolve())
                for key in ("purelib", "platlib")
                if (value := sysconfig.get_path(key))
            }
        )
    )
    if not paths or any(not Path(value).is_dir() for value in paths):
        raise FullGridError("Python environment distribution roots are invalid")
    return paths


def _environment_package_inventory() -> list[tuple[str, str]]:
    """Inventory installed distributions without consulting mutable sys.path."""

    return sorted(
        (
            str(distribution.metadata.get("Name", "unknown")).lower(),
            str(distribution.version),
        )
        for distribution in importlib.metadata.distributions(
            path=_environment_distribution_paths()
        )
    )


def _environment_identity() -> dict[str, Any]:
    packages = _environment_package_inventory()
    try:
        uv_version = subprocess.run(
            ["uv", "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        uv_version = None
    torch_identity: dict[str, Any]
    try:
        import torch

        torch_identity = {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
        }
    except ImportError:
        torch_identity = {"version": None}
    return {
        "python_version": sys.version,
        "python_executable": str(Path(sys.executable).resolve()),
        "python_prefix": str(Path(sys.prefix).resolve()),
        "base_prefix": str(Path(sys.base_prefix).resolve()),
        "platform": platform.platform(),
        "uv_version": uv_version,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "packages": [[name, version] for name, version in packages],
        "packages_sha256": _sha256_bytes(_canonical_bytes(packages)),
        "nvidia_driver_versions": _nvidia_driver_versions(),
        "nvidia_gpu_inventory": _nvidia_gpu_inventory(),
        "torch": torch_identity,
    }


def _formal_gpu_uuid_roster(
    environment_identity: Mapping[str, Any],
) -> list[str]:
    """Bind formal execution to physical nvidia-smi indexes 0, 1, and 2."""

    inventory = environment_identity.get("nvidia_gpu_inventory")
    if not isinstance(inventory, list):
        return []
    by_index: dict[str, str] = {}
    for raw_row in inventory:
        if not isinstance(raw_row, Mapping):
            raise FullGridError("environment GPU inventory row is invalid")
        index = raw_row.get("index")
        gpu_uuid = raw_row.get("uuid")
        if (
            not isinstance(index, str)
            or not index.isdigit()
            or not isinstance(gpu_uuid, str)
            or not GPU_UUID_RE.fullmatch(gpu_uuid)
            or index in by_index
        ):
            raise FullGridError("environment GPU inventory identity is invalid")
        by_index[index] = gpu_uuid
    result = [
        by_index[index] for index in FORMAL_GPU_INDICES if index in by_index
    ]
    if len(result) != len(set(result)):
        raise FullGridError("formal GPU roster contains duplicate physical UUIDs")
    return result


def _normalized_worker_environment_identity(
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Remove only CUDA fields changed by one-device worker visibility."""

    result = copy.deepcopy(dict(identity))
    torch_identity = result.get("torch")
    if isinstance(torch_identity, dict):
        torch_identity.pop("cuda_device_count", None)
        torch_identity.pop("cuda_devices", None)
    return result


def _analysis_source_identity() -> dict[str, str]:
    package_root = _absolute_path(Path(__file__)).parent
    return {
        f"ieee_mi/{name}": _sha256_file(package_root / name)
        for name in ANALYSIS_SOURCE_FILES
    }


def _analysis_environment_sha256(identity: Mapping[str, Any]) -> str:
    """Hash the locked environment without CUDA visibility-only fields."""

    normalized = _normalized_worker_environment_identity(identity)
    return _sha256_bytes(_canonical_bytes(normalized))


def _formal_analysis_contract(
    environment_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze post-hoc decision code and settings before any grid result exists."""

    return {
        "schema": ANALYSIS_CONTRACT_SCHEMA,
        "source_identity": _analysis_source_identity(),
        "environment_identity_sha256": _analysis_environment_sha256(
            environment_identity
        ),
        "bootstrap_resamples": FORMAL_ANALYSIS_BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": FORMAL_ANALYSIS_BOOTSTRAP_SEED,
        "ece_bins": FORMAL_ANALYSIS_ECE_BINS,
        "tcformer_comparator": FORMAL_ANALYSIS_TCFORMER_COMPARATOR,
        "primary_metric": "equal_dataset_macro_balanced_accuracy",
        "aggregation": (
            "concatenate_folds_within_subject_seed_then_mean_seeds_then_"
            "mean_subjects_then_equal_weight_datasets"
        ),
        "comparison_context": (
            "descriptive_model_minus_prespecified_tcformer_only"
        ),
        "evidence_scope": "opened_development_datasets_only_not_confirmation",
        "track_scope": TRACK_SCOPE,
    }


def _validate_analysis_contract_shape(contract: Mapping[str, Any]) -> None:
    expected_keys = {
        "schema",
        "source_identity",
        "environment_identity_sha256",
        "bootstrap_resamples",
        "bootstrap_seed",
        "ece_bins",
        "tcformer_comparator",
        "primary_metric",
        "aggregation",
        "comparison_context",
        "evidence_scope",
        "track_scope",
    }
    if set(contract) != expected_keys:
        raise ValueError("analysis contract has an invalid exact schema")
    sources = contract.get("source_identity")
    try:
        bootstrap_resamples = int(contract.get("bootstrap_resamples", 0))
        bootstrap_seed = int(contract["bootstrap_seed"])
        ece_bins = int(contract.get("ece_bins", 0))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("analysis contract has invalid numeric decisions") from error
    if (
        contract.get("schema") != ANALYSIS_CONTRACT_SCHEMA
        or not isinstance(sources, Mapping)
        or set(sources) != {f"ieee_mi/{name}" for name in ANALYSIS_SOURCE_FILES}
        or any(
            not isinstance(value, str) or not HEX_64_RE.fullmatch(value)
            for value in sources.values()
        )
        or not isinstance(contract.get("environment_identity_sha256"), str)
        or not HEX_64_RE.fullmatch(
            str(contract.get("environment_identity_sha256"))
        )
        or bootstrap_resamples <= 0
        or bootstrap_seed < 0
        or ece_bins < 2
        or contract.get("tcformer_comparator")
        != FORMAL_ANALYSIS_TCFORMER_COMPARATOR
        or contract.get("primary_metric")
        != "equal_dataset_macro_balanced_accuracy"
        or contract.get("aggregation")
        != (
            "concatenate_folds_within_subject_seed_then_mean_seeds_then_"
            "mean_subjects_then_equal_weight_datasets"
        )
        or contract.get("comparison_context")
        != "descriptive_model_minus_prespecified_tcformer_only"
        or contract.get("evidence_scope")
        != "opened_development_datasets_only_not_confirmation"
        or contract.get("track_scope") != TRACK_SCOPE
    ):
        raise ValueError("analysis contract is invalid")


def _dataset_contracts() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in OPENED_DATASETS:
        spec = dataset_spec(key)
        if not spec.development_subjects:
            raise FullGridError(f"{key} has no development subjects")
        result[key] = {
            "dataset_spec": asdict(spec),
            "subjects": list(spec.development_subjects),
            "folds": list(DATASET_FOLDS[key]),
            "n_classes": spec.n_classes,
            "protocol": spec.protocol,
            "preprocessing": preprocessing_for_dataset(key),
        }
    # ``DatasetSpec`` contains tuples.  Normalize the complete contract through
    # JSON before it enters the immutable plan so a write/load round trip cannot
    # change tuple fields into lists and make every production resume fail.
    return {
        dataset: json.loads(_canonical_bytes(contract))
        for dataset, contract in result.items()
    }


def _safe_cache_path(cache_root: Path, dataset: str, subject: int) -> Path:
    from .config import DEFAULT_MONTAGE_PROFILE
    from .data import _cache_path

    root = _absolute_path(cache_root)
    _require_real_directory(root)
    current_parent = root.parent
    while current_parent != current_parent.parent:
        observed_parent = current_parent.lstat()
        if current_parent.is_symlink() or not stat.S_ISDIR(observed_parent.st_mode):
            raise FullGridError("cache root has a symlinked/non-directory ancestor")
        current_parent = current_parent.parent
    path = _absolute_path(
        _cache_path(root, dataset, subject, DEFAULT_MONTAGE_PROFILE)
    )
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise FullGridError("cache path escapes the declared cache root") from error
    current = root
    for part in relative.parts[:-1]:
        current /= part
        _require_real_directory(current)
    return path


def _validate_cache_identity_exact(
    identity: Any,
    *,
    dataset: str,
    subject: int,
    channel_names: tuple[str, ...] | None = None,
    observed_shape: Sequence[int] | None = None,
) -> tuple[str, ...]:
    """Validate the complete label-free cache identity used by formal plans."""

    from .config import (
        DEFAULT_MONTAGE_PROFILE,
        channels_for_dataset,
        coordinate_contract_for_dataset,
    )

    expected_keys = {
        "array_sha256",
        "channels",
        "coordinates",
        "dataset",
        "montage_profile",
        "preprocessing",
        "shape",
        "subject",
    }
    if dataset == "local_exp4":
        expected_keys.add("source_manifest")
    _exact_keys(identity, expected_keys, path=f"cache_identity.{dataset}.S{subject}")
    aliases = _recursive_outcome_aliases(
        identity,
        f"cache_identity.{dataset}.S{subject}",
    )
    expected_preprocessing = preprocessing_for_dataset(dataset)
    observed_preprocessing = identity.get("preprocessing")
    label_provenance_path = (
        f"cache_identity.{dataset}.S{subject}.preprocessing.labels"
    )
    label_provenance_value = "ordered exactly as DatasetSpec.events"
    if (
        isinstance(observed_preprocessing, Mapping)
        and set(observed_preprocessing) == set(expected_preprocessing)
        and type(observed_preprocessing.get("labels")) is str
        and observed_preprocessing["labels"] == label_provenance_value
        and expected_preprocessing.get("labels") == label_provenance_value
    ):
        # This one schema- and value-bound field describes the ordering
        # provenance of the cache's class vocabulary; it is not a per-trial
        # outcome.  Keep the recursive guard intact for every other path.
        aliases = [
            path for path in aliases if path != label_provenance_path
        ]
    if aliases:
        raise FullGridError(f"cache identity contains outcome aliases: {aliases[:5]}")

    declared_channels = identity["channels"]
    shape = identity["shape"]
    if (
        not isinstance(identity["array_sha256"], str)
        or HEX_64_RE.fullmatch(identity["array_sha256"]) is None
        or type(identity["subject"]) is not int
        or identity["subject"] != subject
        or not isinstance(declared_channels, list)
        or not declared_channels
        or any(not isinstance(value, str) or not value for value in declared_channels)
        or len(declared_channels) != len(set(declared_channels))
        or not isinstance(shape, list)
        or len(shape) != 3
        or any(type(value) is not int or value <= 0 for value in shape)
    ):
        raise FullGridError(f"cache identity types are invalid for {dataset} S{subject}")
    channels = tuple(declared_channels)
    if channel_names is not None and channels != channel_names:
        raise FullGridError(f"cache channel identity differs for {dataset} S{subject}")
    requested = channels_for_dataset(dataset, DEFAULT_MONTAGE_PROFILE)
    if requested is not None and tuple(requested) != channels:
        raise FullGridError(f"cache channels are invalid for {dataset} S{subject}")
    if observed_shape is not None and list(observed_shape) != shape:
        raise FullGridError(f"cache shape identity differs for {dataset} S{subject}")
    expected_dataset = json.loads(_canonical_bytes(asdict(dataset_spec(dataset))))
    if (
        identity["dataset"] != expected_dataset
        or identity["montage_profile"] != DEFAULT_MONTAGE_PROFILE
        or identity["preprocessing"] != preprocessing_for_dataset(dataset)
        or identity["coordinates"]
        != coordinate_contract_for_dataset(
            dataset,
            DEFAULT_MONTAGE_PROFILE,
            channels,
        )
    ):
        raise FullGridError(f"cache identity contract is invalid for {dataset} S{subject}")

    if dataset == "local_exp4":
        from .data import LOCAL_EXP4_SUBJECT_RUNS

        expected_runs = LOCAL_EXP4_SUBJECT_RUNS[subject]
        manifest = identity["source_manifest"]
        if not isinstance(manifest, list) or len(manifest) != len(expected_runs):
            raise FullGridError(f"cache source manifest is invalid for {dataset} S{subject}")
        for source, expected_run in zip(manifest, expected_runs, strict=True):
            _exact_keys(
                source,
                {"filename", "run", "sha256", "size_bytes", "subject"},
                path=f"cache_identity.{dataset}.source_manifest",
            )
            if (
                type(source["run"]) is not int
                or source["run"] != expected_run
                or type(source["subject"]) is not int
                or source["subject"] != subject
                or type(source["size_bytes"]) is not int
                or source["size_bytes"] <= 0
                or source["filename"]
                != f"exp4_subject{subject}_training_{expected_run}_mi_raw.fif"
                or not isinstance(source["sha256"], str)
                or HEX_64_RE.fullmatch(source["sha256"]) is None
            ):
                raise FullGridError(
                    f"cache source manifest values are invalid for {dataset} S{subject}"
                )
    return channels


def _load_subject_cache_safely(
    dataset: str,
    subject: int,
    *,
    cache_root: Path,
) -> dict[str, Any]:
    """Load one cache through a verified descriptor, never through an alias."""

    from .config import (
        DEFAULT_MONTAGE_PROFILE,
        channels_for_dataset,
        coordinate_contract_for_dataset,
    )
    from .data import (
        SUBJECT_CACHE_NPZ_MEMBERS,
        _atlas_unit_positions,
        _read_unique_regular_bytes,
        _validate_npz_member_contract,
    )

    path = _safe_cache_path(cache_root, dataset, subject)
    try:
        payload = _read_unique_regular_bytes(path)
    except RuntimeError as error:
        raise FullGridError(str(error)) from error
    try:
        expected_members = _validate_npz_member_contract(
            payload,
            expected_members=SUBJECT_CACHE_NPZ_MEMBERS,
            source=str(path),
        )
    except RuntimeError as error:
        raise FullGridError(str(error)) from error
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        if tuple(archive.files) != expected_members:
            raise FullGridError(f"cache {path} fields differ from the frozen schema")
        result = {key: archive[key].copy() for key in expected_members}

    try:
        result["identity"] = json.loads(
            str(result["identity"].item()),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except Exception as error:
        raise FullGridError(f"cache identity is invalid: {path}") from error
    identity = result["identity"]
    if not isinstance(identity, Mapping):
        raise FullGridError(f"cache identity is not an object: {path}")
    profile = DEFAULT_MONTAGE_PROFILE
    requested_channels = channels_for_dataset(dataset, profile)
    expected_preprocessing = preprocessing_for_dataset(dataset)
    channel_names = tuple(str(value) for value in result["channel_names"].tolist())
    _validate_cache_identity_exact(
        identity,
        dataset=dataset,
        subject=subject,
        channel_names=channel_names,
        observed_shape=result["x"].shape,
    )
    expected_coordinates = coordinate_contract_for_dataset(
        dataset,
        profile,
        channel_names,
    )
    expected_dataset = json.loads(_canonical_bytes(asdict(dataset_spec(dataset))))
    if (
        not channel_names
        or len(set(channel_names)) != len(channel_names)
        or (
            requested_channels is not None
            and tuple(requested_channels) != channel_names
        )
        or identity.get("channels") != list(channel_names)
        or identity.get("coordinates") != expected_coordinates
        or list(result["x"].shape) != identity.get("shape")
        or result["x"].dtype != np.float32
        or result["x"].ndim != 3
        or result["x"].shape[1:]
        != (len(channel_names), int(expected_preprocessing["n_times"]))
        or result["y"].dtype != np.int64
        or result["y"].ndim != 1
        or result["positions"].dtype != np.float32
        or result["positions"].shape != (len(channel_names), 3)
    ):
        raise FullGridError(f"cache contract is invalid: {path}")
    expected_positions = _atlas_unit_positions(channel_names)
    trial_count = int(result["x"].shape[0])
    if (
        trial_count <= 0
        or not all(
            len(result[key]) == trial_count
            for key in ("y", "sessions", "runs")
        )
        or set(result["y"].tolist())
        != set(range(dataset_spec(dataset).n_classes))
        or not np.all(np.isfinite(result["x"]))
        or not np.all(np.isfinite(result["positions"]))
        or not np.allclose(
            result["positions"], expected_positions, rtol=0.0, atol=1e-7
        )
        or not np.allclose(
            np.linalg.norm(result["positions"], axis=1),
            1.0,
            rtol=0.0,
            atol=1e-6,
        )
    ):
        raise FullGridError(f"cache arrays are invalid: {path}")
    digest = hashlib.sha256()
    for key in ("x", "y", "positions", "sessions", "runs"):
        value = result[key].astype("U") if key in {"sessions", "runs"} else result[key]
        digest.update(np.ascontiguousarray(value).tobytes())
    if (
        not isinstance(identity.get("array_sha256"), str)
        or not HEX_64_RE.fullmatch(str(identity["array_sha256"]))
        or digest.hexdigest() != identity["array_sha256"]
    ):
        raise FullGridError(f"cache array digest failed: {path}")
    return result


def _cache_and_split_identity(
    cache_root: Path,
    contracts: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read each cache once and freeze cache plus label-free split identities."""

    from .data import split_indices

    cache_result: dict[str, Any] = {}
    split_result: dict[str, Any] = {}
    for dataset in OPENED_DATASETS:
        for raw_subject in contracts[dataset]["subjects"]:
            subject = int(raw_subject)
            cache = _load_subject_cache_safely(
                dataset,
                subject,
                cache_root=cache_root,
            )
            identity = copy.deepcopy(cache["identity"])
            digest = identity.get("array_sha256")
            if not isinstance(digest, str) or not HEX_64_RE.fullmatch(digest):
                raise FullGridError(
                    f"cache {dataset} S{subject} has no valid array digest"
                )
            subject_key = _subject_identity_key(dataset, subject)
            cache_result[subject_key] = identity
            for raw_fold in contracts[dataset]["folds"]:
                fold = int(raw_fold)
                train_rows, validation_rows, test_rows = split_indices(
                    dataset,
                    cache["y"],
                    cache["sessions"],
                    cache["runs"],
                    fold=fold,
                    subject=subject,
                )
                split_result[_split_identity_key(dataset, subject, fold)] = (
                    _one_split_identity(
                        dataset=dataset,
                        subject=subject,
                        fold=fold,
                        cache_array_sha256=digest,
                        trial_count=len(cache["y"]),
                        train_rows=train_rows,
                        validation_rows=validation_rows,
                        test_rows=test_rows,
                    )
                )
    return cache_result, split_result


def _cache_identity(
    cache_root: Path,
    contracts: Mapping[str, Any],
) -> dict[str, Any]:
    """Compatibility helper returning only the cache half of frozen identity."""

    return _cache_and_split_identity(cache_root, contracts)[0]


def assemble_plan(
    *,
    dataset_contracts: Mapping[str, Any],
    architectures: Sequence[str],
    seeds: Sequence[int],
    train_config: Mapping[str, Any],
    cache_identity: Mapping[str, Any],
    split_identity: Mapping[str, Any],
    source_identity: Mapping[str, str],
    environment_identity: Mapping[str, Any],
    tcformer_source_identity: Mapping[str, Any] | None = None,
    executor: str = DEFAULT_EXECUTOR,
    worker_cpu_threads: int = 4,
    analysis_contract: Mapping[str, Any] | None = None,
    publishable: bool = False,
) -> dict[str, Any]:
    """Build a canonical plan from explicit components.

    This pure constructor is also useful for tiny synthetic runner tests.  The
    production :func:`build_plan` wrapper supplies the exact five real dataset
    contracts and their validated cache identities.
    """

    architecture_values = tuple(str(value) for value in architectures)
    seed_values = tuple(int(value) for value in seeds)
    if not architecture_values or len(set(architecture_values)) != len(
        architecture_values
    ):
        raise ValueError("architectures must be nonempty and unique")
    if not seed_values or len(set(seed_values)) != len(seed_values):
        raise ValueError("seeds must be nonempty and unique")
    if any(not MODEL_NAME_RE.fullmatch(value) for value in architecture_values):
        raise ValueError("architecture names must contain lowercase letters/digits/_")
    if publishable and architecture_values != COMMON_ARCHITECTURES:
        raise ValueError("formal plans require the exact 43-model common roster")
    if publishable and executor != DEFAULT_EXECUTOR:
        raise ValueError("formal plans require DEFAULT_EXECUTOR")
    worker_cpu_threads = _validate_cpu_threads(worker_cpu_threads)
    normalized_analysis_contract: dict[str, Any] | None = None
    if analysis_contract is not None:
        normalized_analysis_contract = copy.deepcopy(dict(analysis_contract))
        _validate_analysis_contract_shape(normalized_analysis_contract)
    if publishable and normalized_analysis_contract is None:
        raise ValueError("formal plans require a frozen analysis contract")
    normalized_tcformer_identity = (
        None
        if tcformer_source_identity is None
        else copy.deepcopy(dict(tcformer_source_identity))
    )
    if publishable:
        if normalized_tcformer_identity is None:
            raise ValueError("formal plans require pinned TCFormer source identity")
        try:
            _validate_tcformer_source_identity(normalized_tcformer_identity)
        except FullGridError as error:
            raise ValueError("formal TCFormer source identity is invalid") from error
    elif normalized_tcformer_identity is not None:
        raise ValueError("test-only plans cannot claim TCFormer source provenance")

    normalized_contracts: dict[str, Any] = {}
    subject_fold_count = 0
    for dataset, raw_contract in dataset_contracts.items():
        contract = copy.deepcopy(dict(raw_contract))
        subjects = tuple(int(value) for value in contract["subjects"])
        folds = tuple(int(value) for value in contract["folds"])
        if (
            not subjects
            or not folds
            or len(subjects) != len(set(subjects))
            or len(folds) != len(set(folds))
        ):
            raise ValueError(f"{dataset} subjects/folds must be nonempty and unique")
        contract["subjects"] = list(subjects)
        contract["folds"] = list(folds)
        contract["n_classes"] = int(contract["n_classes"])
        normalized_contracts[str(dataset)] = contract
        subject_fold_count += len(subjects) * len(folds)
    if not normalized_contracts:
        raise ValueError("at least one dataset contract is required")

    normalized_cache_identity = copy.deepcopy(dict(cache_identity))
    normalized_split_identity = copy.deepcopy(dict(split_identity))
    expected_subject_keys = {
        _subject_identity_key(dataset, subject)
        for dataset, contract in normalized_contracts.items()
        for subject in contract["subjects"]
    }
    expected_split_keys = {
        _split_identity_key(dataset, subject, fold)
        for dataset, contract in normalized_contracts.items()
        for subject in contract["subjects"]
        for fold in contract["folds"]
    }
    if set(normalized_cache_identity) != expected_subject_keys:
        raise ValueError("cache identity keys differ from the dataset subjects")
    if set(normalized_split_identity) != expected_split_keys:
        raise ValueError("split identity keys differ from the subject/fold grid")
    for key, identity in normalized_cache_identity.items():
        digest = identity.get("array_sha256") if isinstance(identity, Mapping) else None
        if not isinstance(digest, str) or not HEX_64_RE.fullmatch(digest):
            raise ValueError(f"cache identity {key} has no valid array digest")
        if publishable:
            try:
                raw_subject = identity.get("subject")
                raw_dataset = identity.get("dataset")
                dataset_key = (
                    str(raw_dataset.get("key"))
                    if isinstance(raw_dataset, Mapping)
                    else ""
                )
                if type(raw_subject) is not int:
                    raise FullGridError("formal cache subject has an invalid type")
                if key != _subject_identity_key(dataset_key, raw_subject):
                    raise FullGridError("formal cache key differs from its identity")
                _validate_cache_identity_exact(
                    identity,
                    dataset=dataset_key,
                    subject=raw_subject,
                )
            except (KeyError, TypeError, ValueError, FullGridError) as error:
                raise ValueError(f"cache identity {key} is not exact") from error
    for key, raw_identity in normalized_split_identity.items():
        if not isinstance(raw_identity, Mapping):
            raise ValueError(f"split identity {key} is not an object")
        identity = dict(raw_identity)
        try:
            raw_dataset = identity.get("dataset")
            raw_subject = identity.get("subject")
            raw_fold = identity.get("fold")
            if (
                not isinstance(raw_dataset, str)
                or type(raw_subject) is not int
                or type(raw_fold) is not int
            ):
                raise FullGridError("split identity key fields have invalid types")
            subject_key = _subject_identity_key(raw_dataset, raw_subject)
            cache = normalized_cache_identity.get(subject_key)
            if not isinstance(cache, Mapping):
                raise FullGridError("split identity has no cache")
            _validate_split_identity_exact(
                identity,
                dataset=raw_dataset,
                subject=raw_subject,
                fold=raw_fold,
                cache_array_sha256=str(cache.get("array_sha256")),
            )
            if key != _split_identity_key(raw_dataset, raw_subject, raw_fold):
                raise FullGridError("split identity differs from its key")
        except (KeyError, TypeError, ValueError, FullGridError) as error:
            raise ValueError(f"split identity {key} is not exact") from error
        partitions = identity.get("partitions")
        if (
            not isinstance(partitions, Mapping)
            or set(partitions) != {"train", "validation", "source", "test"}
            or not isinstance(identity.get("trial_count"), int)
            or int(identity["trial_count"]) <= 0
        ):
            raise ValueError(f"split identity {key} has an invalid partition contract")
        subject_key = _subject_identity_key(
            str(identity.get("dataset")),
            int(identity.get("subject", -1)),
        )
        cache = normalized_cache_identity.get(subject_key)
        if (
            cache is None
            or identity.get("cache_array_sha256") != cache.get("array_sha256")
            or key
            != _split_identity_key(
                str(identity.get("dataset")),
                int(identity.get("subject", -1)),
                int(identity.get("fold", -1)),
            )
        ):
            raise ValueError(f"split identity {key} does not match its cache/grid key")
        for partition_name, raw_partition in partitions.items():
            if (
                not isinstance(raw_partition, Mapping)
                or not isinstance(raw_partition.get("count"), int)
                or int(raw_partition["count"]) <= 0
                or not isinstance(raw_partition.get("rows_sha256"), str)
                or not HEX_64_RE.fullmatch(str(raw_partition["rows_sha256"]))
            ):
                raise ValueError(
                    f"split identity {key} has invalid {partition_name} identity"
                )
        counts = {
            name: int(partitions[name]["count"])
            for name in ("train", "validation", "source", "test")
        }
        if (
            counts["source"] != counts["train"] + counts["validation"]
            or counts["source"] + counts["test"] != int(identity["trial_count"])
        ):
            raise ValueError(f"split identity {key} counts do not cover the cache")

    payload: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "purpose": "exact_43_model_common_recipe_cross_dataset_prediction_grid",
        "publication_mode": (
            FORMAL_PUBLICATION_MODE if publishable else TEST_PUBLICATION_MODE
        ),
        "track_scope": TRACK_SCOPE,
        "evidence_scope": "opened_development_datasets_only_not_confirmation",
        "confirmation_evidence": False,
        "score_blind": True,
        "dataset_order": list(normalized_contracts),
        "datasets": normalized_contracts,
        "architectures": list(architecture_values),
        "common_architecture_count": len(architecture_values),
        "executor": executor,
        "seeds": list(seed_values),
        "train_config": copy.deepcopy(dict(train_config)),
        "channel_scaling": copy.deepcopy(CHANNEL_SCALING),
        "cache_identity": normalized_cache_identity,
        "split_identity": normalized_split_identity,
        "source_identity": copy.deepcopy(dict(source_identity)),
        "tcformer_source_identity": normalized_tcformer_identity,
        "environment_identity": copy.deepcopy(dict(environment_identity)),
        "analysis_contract": normalized_analysis_contract,
        "execution_config": {
            "worker_cpu_threads": worker_cpu_threads,
            "maximum_concurrent_gpu_workers": MAX_GPU_WORKERS,
            "minimum_free_gib": DEFAULT_MIN_FREE_GIB,
            "device": "cuda:0" if publishable else "cpu",
            "physical_gpu_uuid_roster": (
                _formal_gpu_uuid_roster(environment_identity)
                if publishable
                else []
            ),
        },
        "job_order": ["dataset", "model", "subject", "fold", "seed"],
        "n_jobs": (
            subject_fold_count * len(architecture_values) * len(seed_values)
        ),
        "output_contract": {
            "test_labels_present": False,
            "test_scores_present": False,
            "prediction_arrays": ["rows", "probabilities"],
            "atomic_unit": "dataset/model/subject/fold/seed",
            "selection": "train_only_optimization_validation_only_epoch_selection",
            "final_refit": (
                "seeded_initialization_reset_then_train_plus_validation_for_"
                "selected_epoch_count"
            ),
        },
    }
    if publishable and payload["n_jobs"] != FORMAL_EXPECTED_JOBS:
        raise ValueError(
            f"formal common plan must contain exactly {FORMAL_EXPECTED_JOBS} jobs"
        )
    payload["plan_sha256"] = plan_sha256(payload)
    return payload


def plan_sha256(plan: Mapping[str, Any]) -> str:
    payload = copy.deepcopy(dict(plan))
    payload.pop("plan_sha256", None)
    return _sha256_bytes(_canonical_bytes(payload))


def validate_plan_semantics(
    plan: Mapping[str, Any],
    *,
    require_runtime_identity: bool = False,
) -> None:
    """Reject forged formal semantics independently of the plan checksum."""

    _exact_keys(
        plan,
        {
            "schema",
            "purpose",
            "publication_mode",
            "track_scope",
            "evidence_scope",
            "confirmation_evidence",
            "score_blind",
            "dataset_order",
            "datasets",
            "architectures",
            "common_architecture_count",
            "executor",
            "seeds",
            "train_config",
            "channel_scaling",
            "cache_identity",
            "split_identity",
            "source_identity",
            "tcformer_source_identity",
            "environment_identity",
            "analysis_contract",
            "execution_config",
            "job_order",
            "n_jobs",
            "output_contract",
            "plan_sha256",
        },
        path="plan",
    )
    if (
        plan["schema"] != PLAN_SCHEMA
        or plan["purpose"]
        != "exact_43_model_common_recipe_cross_dataset_prediction_grid"
        or plan["track_scope"] != TRACK_SCOPE
        or plan["evidence_scope"]
        != "opened_development_datasets_only_not_confirmation"
        or plan["confirmation_evidence"] is not False
        or plan["score_blind"] is not True
        or plan_sha256(plan) != plan["plan_sha256"]
    ):
        raise FullGridError("plan identity or checksum is invalid")
    mode = plan["publication_mode"]
    if mode not in {FORMAL_PUBLICATION_MODE, TEST_PUBLICATION_MODE}:
        raise FullGridError("plan publication mode is invalid")
    formal = mode == FORMAL_PUBLICATION_MODE
    dataset_order = plan["dataset_order"]
    datasets = plan["datasets"]
    architectures = plan["architectures"]
    seeds = plan["seeds"]
    if (
        not isinstance(dataset_order, list)
        or not dataset_order
        or any(type(value) is not str or not value for value in dataset_order)
        or len(dataset_order) != len(set(dataset_order))
        or not isinstance(datasets, Mapping)
        # Canonical JSON sorts mapping keys, so their decoded insertion order
        # cannot encode the scientific dataset traversal order.  That order is
        # carried exclusively by dataset_order; the mapping must have exactly
        # the same membership.
        or set(datasets) != set(dataset_order)
        or not isinstance(architectures, list)
        or not architectures
        or len(architectures) != len(set(architectures))
        or any(
            not isinstance(value, str) or not MODEL_NAME_RE.fullmatch(value)
            for value in architectures
        )
        or not isinstance(seeds, list)
        or not seeds
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in seeds
        )
        or len(seeds) != len(set(seeds))
        or plan["common_architecture_count"] != len(architectures)
        or not isinstance(plan["executor"], str)
        or not isinstance(plan["n_jobs"], int)
        or isinstance(plan["n_jobs"], bool)
    ):
        raise FullGridError("plan roster/grid structure is invalid")
    expected_jobs = 0
    expected_cache_keys: set[str] = set()
    expected_split_keys: set[str] = set()
    for dataset in dataset_order:
        contract = datasets[dataset]
        if not isinstance(contract, Mapping):
            raise FullGridError(f"dataset contract is invalid: {dataset}")
        subjects = contract.get("subjects")
        folds = contract.get("folds")
        if (
            not isinstance(subjects, list)
            or not subjects
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in subjects
            )
            or len(subjects) != len(set(subjects))
            or not isinstance(folds, list)
            or not folds
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in folds
            )
            or len(folds) != len(set(folds))
            or isinstance(contract.get("n_classes"), bool)
            or not isinstance(contract.get("n_classes"), int)
            or int(contract["n_classes"]) < 2
        ):
            raise FullGridError(f"dataset subject/fold structure is invalid: {dataset}")
        expected_jobs += (
            len(subjects) * len(folds) * len(architectures) * len(seeds)
        )
        for subject in subjects:
            expected_cache_keys.add(_subject_identity_key(dataset, subject))
            for fold in folds:
                expected_split_keys.add(
                    _split_identity_key(dataset, subject, fold)
                )
    if plan["n_jobs"] != expected_jobs:
        raise FullGridError("plan n_jobs differs from its Cartesian product")
    if (
        not isinstance(plan["cache_identity"], Mapping)
        or set(plan["cache_identity"]) != expected_cache_keys
        or not isinstance(plan["split_identity"], Mapping)
        or set(plan["split_identity"]) != expected_split_keys
    ):
        raise FullGridError("plan cache/split key set differs from its grid")
    if formal and tuple(dataset_order) != OPENED_DATASETS:
        raise FullGridError("formal plan differs from the frozen 43-model grid")
    for dataset in dataset_order:
        for subject in datasets[dataset]["subjects"]:
            cache_key = _subject_identity_key(dataset, subject)
            cache_identity = plan["cache_identity"][cache_key]
            if formal:
                _validate_cache_identity_exact(
                    cache_identity,
                    dataset=dataset,
                    subject=subject,
                )
            elif (
                not isinstance(cache_identity, Mapping)
                or set(cache_identity) != {"array_sha256"}
                or not isinstance(cache_identity["array_sha256"], str)
                or HEX_64_RE.fullmatch(cache_identity["array_sha256"]) is None
            ):
                raise FullGridError("test-only cache identity is invalid")
            for fold in datasets[dataset]["folds"]:
                _validate_split_identity_exact(
                    plan["split_identity"][
                        _split_identity_key(dataset, subject, fold)
                    ],
                    dataset=dataset,
                    subject=subject,
                    fold=fold,
                    cache_array_sha256=cache_identity["array_sha256"],
                )
    _exact_keys(
        plan["execution_config"],
        {
            "worker_cpu_threads",
            "maximum_concurrent_gpu_workers",
            "minimum_free_gib",
            "device",
            "physical_gpu_uuid_roster",
        },
        path="plan.execution_config",
    )
    execution = plan["execution_config"]
    if (
        project_gpu_leases.MAX_ACTIVE_LEASES != MAX_GPU_WORKERS
        or
        isinstance(execution["worker_cpu_threads"], bool)
        or not isinstance(execution["worker_cpu_threads"], int)
        or not 1 <= execution["worker_cpu_threads"] <= 16
        or execution["maximum_concurrent_gpu_workers"] != MAX_GPU_WORKERS
        or not math.isclose(
            float(execution["minimum_free_gib"]),
            DEFAULT_MIN_FREE_GIB,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or not isinstance(execution["physical_gpu_uuid_roster"], list)
        or len(execution["physical_gpu_uuid_roster"])
        != len(set(execution["physical_gpu_uuid_roster"]))
        or any(
            not isinstance(value, str) or not GPU_UUID_RE.fullmatch(value)
            for value in execution["physical_gpu_uuid_roster"]
        )
    ):
        raise FullGridError("plan execution contract is invalid")
    expected_output = {
        "test_labels_present": False,
        "test_scores_present": False,
        "prediction_arrays": ["rows", "probabilities"],
        "atomic_unit": "dataset/model/subject/fold/seed",
        "selection": "train_only_optimization_validation_only_epoch_selection",
        "final_refit": (
            "seeded_initialization_reset_then_train_plus_validation_for_"
            "selected_epoch_count"
        ),
    }
    if (
        plan["job_order"]
        != ["dataset", "model", "subject", "fold", "seed"]
        or plan["output_contract"] != expected_output
        or plan["channel_scaling"] != CHANNEL_SCALING
    ):
        raise FullGridError("plan protocol/output contract is invalid")
    if formal:
        _validate_tcformer_source_identity(plan["tcformer_source_identity"])
        if (
            tuple(dataset_order) != OPENED_DATASETS
            or datasets != _dataset_contracts()
            or tuple(architectures) != COMMON_ARCHITECTURES
            or tuple(seeds) != FORMAL_SEEDS
            or plan["n_jobs"] != FORMAL_EXPECTED_JOBS
            or plan["executor"] != DEFAULT_EXECUTOR
            or plan["train_config"] != FORMAL_TRAIN_CONFIG
            or execution["device"] != "cuda:0"
            or len(execution["physical_gpu_uuid_roster"]) < 1
            or execution["physical_gpu_uuid_roster"]
            != _formal_gpu_uuid_roster(plan["environment_identity"])
        ):
            raise FullGridError("formal plan differs from the frozen 43-model grid")
        try:
            _validate_analysis_contract_shape(plan["analysis_contract"])
        except (TypeError, ValueError) as error:
            raise FullGridError("formal analysis contract is invalid") from error
        expected_analysis = _formal_analysis_contract(
            plan["environment_identity"]
        )
        if plan["analysis_contract"] != expected_analysis:
            raise FullGridError("formal analysis contract differs from source")
        if require_runtime_identity:
            if (
                plan["source_identity"]
                != _source_identity(executor=DEFAULT_EXECUTOR)
                or plan["tcformer_source_identity"]
                != _current_tcformer_source_identity()
                or plan["environment_identity"] != _environment_identity()
            ):
                raise FullGridError(
                    "formal plan source/environment differs from this runtime"
                )
    else:
        if (
            execution["device"] != "cpu"
            or execution["physical_gpu_uuid_roster"]
            or plan["analysis_contract"] is not None
            or plan["tcformer_source_identity"] is not None
        ):
            raise FullGridError(
                "test-only plans cannot claim formal CUDA/analysis provenance"
            )


def build_plan(
    *,
    cache_root: Path,
    worker_cpu_threads: int = 4,
) -> dict[str, Any]:
    from .training import TrainConfig

    train_config = TrainConfig()
    contracts = _dataset_contracts()
    cache_identity, split_identity = _cache_and_split_identity(cache_root, contracts)
    environment_identity = _environment_identity()
    plan = assemble_plan(
        dataset_contracts=contracts,
        architectures=COMMON_ARCHITECTURES,
        seeds=FORMAL_SEEDS,
        train_config=asdict(train_config),
        cache_identity=cache_identity,
        split_identity=split_identity,
        source_identity=_source_identity(executor=DEFAULT_EXECUTOR),
        tcformer_source_identity=_current_tcformer_source_identity(),
        environment_identity=environment_identity,
        executor=DEFAULT_EXECUTOR,
        worker_cpu_threads=worker_cpu_threads,
        analysis_contract=_formal_analysis_contract(environment_identity),
        publishable=True,
    )
    validate_plan_semantics(plan, require_runtime_identity=True)
    return plan


def write_or_validate_plan(run_root: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    root = _safe_run_root(run_root, create=True)
    path = root / "plan.json"
    digest_path = root / "plan.sha256"
    expected_value = copy.deepcopy(dict(expected))
    validate_plan_semantics(
        expected_value,
        require_runtime_identity=(
            expected_value.get("publication_mode") == FORMAL_PUBLICATION_MODE
        ),
    )
    expected_digest = plan_sha256(expected_value)
    if expected_value.get("plan_sha256") != expected_digest:
        raise FullGridError("expected plan carries an invalid plan_sha256")
    if path.exists() and digest_path.exists():
        observed = load_plan(run_root)
        if observed != expected_value:
            raise FullGridError(
                "existing immutable plan differs from the requested plan"
            )
        return observed
    if path.exists():
        # A power loss may occur after the canonical plan is durable but before
        # its sidecar digest is created.  Repair only when the existing payload
        # is exactly the requested, internally checksummed plan.
        observed = strict_load(path)
        if (
            observed != expected_value
            or observed.get("schema") != PLAN_SCHEMA
            or observed.get("plan_sha256") != expected_digest
            or _canonical_bytes(observed) + b"\n" != _read_unique_regular(path)
        ):
            raise FullGridError(
                "orphaned plan.json differs from the requested immutable plan"
            )
        try:
            _write_bytes_exclusive(
                digest_path,
                f"{expected_digest}\n".encode("ascii"),
            )
        except FileExistsError:
            pass
        return load_plan(run_root)
    if digest_path.exists():
        # Symmetric recovery is possible because the caller has rebuilt the
        # exact expected plan.  Never replace or accept an unrelated digest.
        if _read_unique_regular(digest_path).decode("ascii").strip() != expected_digest:
            raise FullGridError(
                "orphaned plan.sha256 differs from the requested immutable plan"
            )
        try:
            _write_json_exclusive(path, expected_value)
        except FileExistsError:
            pass
        return load_plan(run_root)
    try:
        _write_json_exclusive(path, expected_value)
        _write_bytes_exclusive(digest_path, f"{expected_digest}\n".encode("ascii"))
    except FileExistsError:
        observed = load_plan(run_root)
        if observed != expected_value:
            raise FullGridError(
                "concurrently created immutable plan differs from requested plan"
            )
        return observed
    return expected_value


def load_plan(run_root: Path) -> dict[str, Any]:
    root = _safe_run_root(run_root)
    path = root / "plan.json"
    digest_path = root / "plan.sha256"
    if not path.exists() or not digest_path.exists():
        raise FileNotFoundError(f"plan.json/plan.sha256 absent under {run_root}")
    plan = strict_load(path)
    _require_unique_regular(path, read_only=True)
    _require_unique_regular(digest_path, read_only=True)
    observed_digest = _read_unique_regular(digest_path).decode("ascii").strip()
    computed_digest = plan_sha256(plan)
    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("plan_sha256") != computed_digest
        or observed_digest != computed_digest
    ):
        raise FullGridError("immutable plan checksum or schema is invalid")
    if _canonical_bytes(plan) + b"\n" != _read_unique_regular(path):
        raise FullGridError("plan.json is not in canonical serialized form")
    validate_plan_semantics(plan)
    return plan


def load_or_repair_plan(run_root: Path) -> dict[str, Any]:
    """Load a plan, repairing the write-order power-cut state when possible."""

    root = _safe_run_root(run_root)
    path = root / "plan.json"
    digest_path = root / "plan.sha256"
    if digest_path.exists():
        return load_plan(run_root)
    if not path.exists():
        raise FileNotFoundError(f"plan.json/plan.sha256 absent under {run_root}")
    observed = strict_load(path)
    computed_digest = plan_sha256(observed)
    if (
        observed.get("schema") != PLAN_SCHEMA
        or observed.get("plan_sha256") != computed_digest
        or _canonical_bytes(observed) + b"\n" != _read_unique_regular(path)
    ):
        raise FullGridError("orphaned plan.json checksum or schema is invalid")
    validate_plan_semantics(observed)
    return write_or_validate_plan(root, observed)


def verify_runtime_identity(
    plan: Mapping[str, Any],
    *,
    cache_root: Path,
    worker_cuda_visibility: bool = False,
) -> None:
    """Reject source, cache, environment, or protocol drift before resume."""

    validate_plan_semantics(plan)
    if worker_cuda_visibility:
        # Freeze every lazily imported in-repository execution module in this
        # worker before its source digest is checked. This prevents a worker
        # waiting on a busy GPU from importing a later on-disk edit hours later.
        for module_name in (
            "benchmark",
            "baselines",
            "models",
            "training",
            "data",
        ):
            importlib.import_module(f"{__package__}.{module_name}")
    if tuple(plan.get("dataset_order", ())) != OPENED_DATASETS:
        raise FullGridError("plan does not contain the exact five opened datasets")
    if plan.get("datasets") != _dataset_contracts():
        raise FullGridError("current dataset contracts differ from plan.json")
    if tuple(plan.get("architectures", ())) != COMMON_ARCHITECTURES:
        raise FullGridError("current 43-model architecture roster differs from plan.json")
    cache_identity, split_identity = _cache_and_split_identity(
        cache_root,
        plan["datasets"],
    )
    if cache_identity != plan.get("cache_identity"):
        raise FullGridError("current cache identities differ from plan.json")
    if split_identity != plan.get("split_identity"):
        raise FullGridError("current split identities differ from plan.json")
    source_identity = _source_identity(executor=str(plan["executor"]))
    if source_identity != plan.get("source_identity"):
        raise FullGridError("current source identities differ from plan.json")
    if _current_tcformer_source_identity() != plan.get("tcformer_source_identity"):
        raise FullGridError("current TCFormer source identity differs from plan.json")
    observed_environment = _environment_identity()
    planned_environment = plan.get("environment_identity")
    if worker_cuda_visibility:
        observed_environment = _normalized_worker_environment_identity(
            observed_environment
        )
        planned_environment = _normalized_worker_environment_identity(
            planned_environment
            if isinstance(planned_environment, Mapping)
            else {}
        )
    if observed_environment != planned_environment:
        raise FullGridError("current execution environment differs from plan.json")
    planned_analysis = plan.get("analysis_contract")
    if tuple(plan.get("dataset_order", ())) == OPENED_DATASETS:
        if not isinstance(planned_analysis, Mapping):
            raise FullGridError(
                "formal plan has no frozen post-hoc analysis contract"
            )
        try:
            _validate_analysis_contract_shape(planned_analysis)
        except (TypeError, ValueError) as error:
            raise FullGridError("formal analysis contract is invalid") from error
        expected_analysis = _formal_analysis_contract(
            plan["environment_identity"]
        )
        if dict(planned_analysis) != expected_analysis:
            raise FullGridError(
                "current analysis source/decisions differ from plan.json"
            )


def _verify_job_runtime_identity(
    plan: Mapping[str, Any],
    job: Job,
    *,
    cache_root: Path,
    worker_cuda_visibility: bool,
) -> None:
    """Freshly bind one impending publication to source/env/cache/split."""

    validate_plan_semantics(plan)
    if plan["publication_mode"] != FORMAL_PUBLICATION_MODE:
        return
    if plan["executor"] != DEFAULT_EXECUTOR:
        raise FullGridError("formal job has a non-default executor")
    if _source_identity(executor=DEFAULT_EXECUTOR) != plan["source_identity"]:
        raise FullGridError("source identity changed before job publication")
    if job.model == "tcformer" and (
        _current_tcformer_source_identity()
        != plan["tcformer_source_identity"]
    ):
        raise FullGridError("TCFormer source identity changed before publication")
    observed_environment: Mapping[str, Any] = _environment_identity()
    planned_environment: Mapping[str, Any] = plan["environment_identity"]
    if worker_cuda_visibility:
        observed_environment = _normalized_worker_environment_identity(
            observed_environment
        )
        planned_environment = _normalized_worker_environment_identity(
            planned_environment
        )
    if observed_environment != planned_environment:
        raise FullGridError("environment identity changed before job publication")
    from .data import split_indices

    cache = _load_subject_cache_safely(
        job.dataset,
        job.subject,
        cache_root=cache_root,
    )
    if cache["identity"] != _planned_cache_identity(plan, job):
        raise FullGridError("cache identity changed before job publication")
    train_rows, validation_rows, test_rows = split_indices(
        job.dataset,
        cache["y"],
        cache["sessions"],
        cache["runs"],
        fold=job.fold,
        subject=job.subject,
    )
    observed_split = _one_split_identity(
        dataset=job.dataset,
        subject=job.subject,
        fold=job.fold,
        cache_array_sha256=str(cache["identity"]["array_sha256"]),
        trial_count=len(cache["y"]),
        train_rows=train_rows,
        validation_rows=validation_rows,
        test_rows=test_rows,
    )
    if observed_split != _planned_split_identity(plan, job):
        raise FullGridError("split identity changed before job publication")


def iter_jobs(plan: Mapping[str, Any]) -> Iterator[Job]:
    validate_plan_semantics(plan)
    architectures = tuple(str(value) for value in plan["architectures"])
    seeds = tuple(int(value) for value in plan["seeds"])
    count = 0
    seen_ids: set[str] = set()
    for dataset in plan["dataset_order"]:
        contract = plan["datasets"][dataset]
        for model, subject, fold, seed in product(
            architectures,
            contract["subjects"],
            contract["folds"],
            seeds,
        ):
            job = Job(
                dataset=str(dataset),
                model=str(model),
                subject=int(subject),
                fold=int(fold),
                seed=int(seed),
            )
            if job.job_id in seen_ids:
                raise FullGridError(f"job-id collision for {job.identity()}")
            seen_ids.add(job.job_id)
            count += 1
            yield job
    if count != int(plan["n_jobs"]):
        raise FullGridError(
            f"plan declares {plan['n_jobs']} jobs but Cartesian product has {count}"
        )


def _record_directory(run_root: Path, job: Job) -> Path:
    name = (
        f"s{job.subject:03d}_f{job.fold:02d}_seed{job.seed}_"
        f"{job.job_id.removeprefix('job-')[:10]}"
    )
    return _safe_run_root(run_root) / "records" / job.dataset / job.model / name


def _claim_path(run_root: Path, job: Job) -> Path:
    return (
        _safe_run_root(run_root)
        / "claims"
        / job.job_id[-2:]
        / f"{job.job_id}.json"
    )


def _failure_root(run_root: Path, job: Job) -> Path:
    return _safe_run_root(run_root) / "failures" / job.job_id[-2:]


def _quarantine(
    run_root: Path,
    path: Path,
    *,
    category: str,
    reason: str,
    expected_identity: tuple[int, int] | None = None,
) -> Path:
    root = _safe_run_root(run_root)
    root_descriptor = _open_absolute_directory(root)
    try:
        source = _absolute_path(path)
        try:
            relative = source.relative_to(root)
        except ValueError as error:
            raise FullGridError("quarantine source escapes the run root") from error
        return _quarantine_at(
            root_descriptor,
            root,
            relative,
            category=category,
            reason=reason,
            expected_identity=expected_identity,
        )
    finally:
        os.close(root_descriptor)


def _seal_forensic_node_at(parent_descriptor: int, name: str) -> None:
    """Recursively make one quarantined node immutable via held dirfds."""

    observed = os.stat(
        name,
        dir_fd=parent_descriptor,
        follow_symlinks=False,
    )
    if stat.S_ISREG(observed.st_mode):
        if observed.st_nlink != 1:
            raise FullGridError("forensic file is hardlinked")
        descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
            dir_fd=parent_descriptor,
        )
        try:
            current = os.fstat(descriptor)
            if (
                not stat.S_ISREG(current.st_mode)
                or current.st_nlink != 1
                or (current.st_dev, current.st_ino)
                != (observed.st_dev, observed.st_ino)
            ):
                raise FullGridError("forensic file changed while sealing")
            os.fchmod(descriptor, 0o444)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return
    if not stat.S_ISDIR(observed.st_mode):
        raise FullGridError("forensic node is special or symlinked")
    descriptor = os.open(
        name,
        _directory_open_flags(),
        dir_fd=parent_descriptor,
    )
    try:
        current = os.fstat(descriptor)
        if (current.st_dev, current.st_ino) != (
            observed.st_dev,
            observed.st_ino,
        ):
            raise FullGridError("forensic directory changed while sealing")
        for child_name in sorted(os.listdir(descriptor)):
            _seal_forensic_node_at(descriptor, child_name)
        os.fchmod(descriptor, 0o555)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _quarantine_at(
    root_descriptor: int,
    root: Path,
    relative_source: Path,
    *,
    category: str,
    reason: str,
    expected_identity: tuple[int, int] | None = None,
) -> Path:
    """Quarantine one node relative to a held, path-bound run-root descriptor."""

    if (
        category not in QUARANTINE_REASONS
        or reason not in QUARANTINE_REASONS[category]
    ):
        raise FullGridError("unknown quarantine category/reason")
    relative = Path(relative_source)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} or os.sep in part for part in relative.parts)
    ):
        raise FullGridError("quarantine source is not a safe run-relative path")
    source_parent = _open_relative_directory(
        root_descriptor,
        relative.parent,
        create=False,
    )
    destination_relative = Path("quarantine") / category
    destination_parent = _open_relative_directory(
        root_descriptor,
        destination_relative,
        create=True,
    )
    try:
        observed = os.stat(
            relative.name,
            dir_fd=source_parent,
            follow_symlinks=False,
        )
        source_identity = (observed.st_dev, observed.st_ino)
        if (
            expected_identity is not None
            and source_identity != expected_identity
        ):
            raise ClaimUnavailable(
                "quarantine source identity changed before invalidation"
            )
        if stat.S_ISLNK(observed.st_mode) or not (
        stat.S_ISREG(observed.st_mode) or stat.S_ISDIR(observed.st_mode)
        ) or (stat.S_ISREG(observed.st_mode) and observed.st_nlink != 1):
            raise FullGridError(
                f"refusing to quarantine special/aliased node {root / relative}"
            )
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        destination_name = (
            f"{relative.name}.{reason}.{stamp}.{uuid.uuid4().hex[:10]}"
        )
        source_directory_descriptor: int | None = None
        sealed_directory_names: frozenset[str] | None = None
        if stat.S_ISDIR(observed.st_mode):
            source_directory_descriptor = os.open(
                relative.name,
                _directory_open_flags(),
                dir_fd=source_parent,
            )
            current = os.fstat(source_directory_descriptor)
            if (current.st_dev, current.st_ino) != (
                observed.st_dev,
                observed.st_ino,
            ):
                os.close(source_directory_descriptor)
                raise FullGridError("quarantine source directory changed")
            if stat.S_IMODE(current.st_mode) == 0o555:
                sealed_directory_names = frozenset(
                    os.listdir(source_directory_descriptor)
                )
        try:
            try:
                if (
                    source_directory_descriptor is not None
                    and sealed_directory_names is not None
                ):
                    _publish_sealed_directory_noreplace_at(
                        source_parent,
                        relative.name,
                        source_directory_descriptor,
                        destination_parent,
                        destination_name,
                        expected_names=sealed_directory_names,
                        require_readonly_regular_children=False,
                    )
                else:
                    _atomic_rename_noreplace_at(
                        source_parent,
                        relative.name,
                        destination_parent,
                        destination_name,
                    )
            except BaseException:
                try:
                    os.stat(
                        relative.name,
                        dir_fd=source_parent,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    recovered_destination = os.stat(
                        destination_name,
                        dir_fd=destination_parent,
                        follow_symlinks=False,
                    )
                    if (
                        recovered_destination.st_dev,
                        recovered_destination.st_ino,
                    ) == source_identity:
                        # A move-then-reseal/fsync failure is recoverable here:
                        # the exact inode is already outside its canonical
                        # name and is recursively sealed below.
                        pass
                    else:
                        raise
                else:
                    raise
        finally:
            if source_directory_descriptor is not None:
                os.close(source_directory_descriptor)
        moved = os.stat(
            destination_name,
            dir_fd=destination_parent,
            follow_symlinks=False,
        )
        if (
            (moved.st_dev, moved.st_ino) != source_identity
            or stat.S_IFMT(moved.st_mode) != stat.S_IFMT(observed.st_mode)
        ):
            raise FullGridError("quarantine destination is not the exact source inode")
        os.fsync(source_parent)
        _seal_forensic_node_at(destination_parent, destination_name)
        os.fsync(destination_parent)
        sealed = os.stat(
            destination_name,
            dir_fd=destination_parent,
            follow_symlinks=False,
        )
        if (
            (sealed.st_dev, sealed.st_ino) != source_identity
            or sealed.st_mode & 0o222
        ):
            raise FullGridError("quarantine source was not exactly and durably sealed")
        _assert_directory_descriptor_path(root, root_descriptor)
        return root / destination_relative / destination_name
    finally:
        os.close(destination_parent)
        os.close(source_parent)


def _quarantine_or_lose_claim_race(
    run_root: Path,
    path: Path,
    *,
    category: str,
    reason: str,
    expected_identity: tuple[int, int],
) -> Path:
    """Quarantine one path, converting concurrent repair into normal contention."""

    try:
        return _quarantine(
            run_root,
            path,
            category=category,
            reason=reason,
            expected_identity=expected_identity,
        )
    except (FileNotFoundError, ClaimUnavailable) as error:
        raise ClaimUnavailable(
            f"lost concurrent {category} recovery race for {path.name}"
        ) from error


def _boot_id() -> str | None:
    path = Path("/proc/sys/kernel/random/boot_id")
    if path.exists():
        value = path.read_text(encoding="utf-8").strip()
        return value or None
    # Non-Linux unit tests cannot query a kernel boot UUID without privileged
    # platform tools.  A process-local non-null token keeps ownership exact for
    # the current test process; unknown other processes remain fail-live below.
    return _LOCAL_BOOT_FALLBACK


def _process_start_ticks(pid: int) -> str | None:
    path = Path(f"/proc/{pid}/stat")
    if path.exists():
        text = path.read_text(encoding="utf-8")
        close = text.rfind(")")
        if close < 0:
            return None
        fields_after_command = text[close + 2 :].split()
        # The first token is field 3 (state); starttime is field 22.
        return fields_after_command[19] if len(fields_after_command) > 19 else None
    return _LOCAL_PROCESS_START_FALLBACK if int(pid) == os.getpid() else None


def _process_identity(pid: int | None = None) -> dict[str, Any]:
    current_pid = os.getpid() if pid is None else int(pid)
    value = {
        "host": socket.gethostname(),
        "pid": current_pid,
        "boot_id": _boot_id(),
        "start_ticks": _process_start_ticks(current_pid),
    }
    if not value["boot_id"] or not value["start_ticks"]:
        raise FullGridError("cannot establish exact boot/process-start identity")
    return value


def _validate_resource_guard(
    value: Any,
    *,
    plan: Mapping[str, Any],
) -> None:
    _exact_keys(value, {"safe", "gpu", "disk", "reason"}, path="resource_guard")
    _exact_keys(
        value["gpu"],
        {
            "safe",
            "gpu",
            "utilization_percent",
            "memory_used_mib",
            "own_compute_memory_mib",
            "foreign_processes",
            "reason",
            "gpu_uuid",
            "pci_bus_id",
            "name",
        },
        path="resource_guard.gpu",
    )
    _exact_keys(
        value["disk"],
        {
            "safe",
            "path",
            "free_bytes",
            "total_bytes",
            "free_gib",
            "minimum_free_gib",
            "reason",
        },
        path="resource_guard.disk",
    )
    disk = value["disk"]
    gpu = value["gpu"]
    if (
        value["safe"] is not True
        or gpu["safe"] is not True
        or disk["safe"] is not True
        or value["reason"] != "resources available"
        or not isinstance(gpu["foreign_processes"], (list, tuple))
        or len(gpu["foreign_processes"]) != 0
        or isinstance(disk["minimum_free_gib"], bool)
        or not isinstance(disk["minimum_free_gib"], (int, float))
        or not math.isclose(
            float(disk["minimum_free_gib"]),
            DEFAULT_MIN_FREE_GIB,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or isinstance(disk["free_gib"], bool)
        or not isinstance(disk["free_gib"], (int, float))
        or not math.isfinite(float(disk["free_gib"]))
        or float(disk["free_gib"]) < DEFAULT_MIN_FREE_GIB
        or isinstance(disk["free_bytes"], bool)
        or not isinstance(disk["free_bytes"], int)
        or disk["free_bytes"] < int(DEFAULT_MIN_FREE_GIB * 1024**3)
        or isinstance(disk["total_bytes"], bool)
        or not isinstance(disk["total_bytes"], int)
        or disk["total_bytes"] <= 0
        or disk["total_bytes"] < disk["free_bytes"]
        or not isinstance(disk["path"], str)
        or not os.path.isabs(disk["path"])
        or not isinstance(disk["reason"], str)
        or not disk["reason"]
        or not isinstance(gpu["reason"], str)
        or not gpu["reason"]
    ):
        raise FullGridError("resource guard violates the hard safety floor")
    for name in (
        "utilization_percent",
        "memory_used_mib",
        "own_compute_memory_mib",
    ):
        raw = gpu[name]
        if (
            isinstance(raw, bool)
            or not isinstance(raw, (int, float))
            or not math.isfinite(float(raw))
            or float(raw) < 0.0
        ):
            raise FullGridError("resource guard has invalid GPU counters")
    utilization = float(gpu["utilization_percent"])
    memory_used = float(gpu["memory_used_mib"])
    own_memory = float(gpu["own_compute_memory_mib"])
    foreign_memory = memory_used - own_memory
    if (
        utilization > DEFAULT_MAX_IDLE_UTILIZATION_PERCENT
        or own_memory > memory_used
        or foreign_memory < 0.0
        or foreign_memory > DEFAULT_MAX_FOREIGN_MEMORY_MIB
        or not math.isclose(
            float(disk["free_gib"]),
            disk["free_bytes"] / float(1024**3),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise FullGridError("resource guard counters contradict its safe state")
    formal = plan["publication_mode"] == FORMAL_PUBLICATION_MODE
    if formal:
        inventory = plan["environment_identity"].get("nvidia_gpu_inventory")
        inventory_by_uuid = {
            row["uuid"]: row
            for row in inventory
            if isinstance(row, Mapping) and isinstance(row.get("uuid"), str)
        } if isinstance(inventory, list) else {}
        planned_gpu = inventory_by_uuid.get(gpu["gpu_uuid"])
        if (
            gpu["gpu_uuid"]
            not in plan["execution_config"]["physical_gpu_uuid_roster"]
            or gpu["gpu"] != gpu["gpu_uuid"]
            or planned_gpu is None
            or gpu["pci_bus_id"] != planned_gpu.get("pci_bus_id")
            or gpu["name"] != planned_gpu.get("name")
        ):
            raise FullGridError("resource guard is not bound to a planned GPU")
    elif (
        gpu["gpu_uuid"] is not None
        or gpu["pci_bus_id"] is not None
        or gpu["name"] is not None
    ):
        raise FullGridError("test-only resource guard cannot claim a physical GPU")


def _validate_persisted_gpu_lease_receipt(
    receipt: Any,
    *,
    plan: Mapping[str, Any],
    run_root: Path | None,
    expected_gpu_uuid: str | None = None,
) -> None:
    formal = plan["publication_mode"] == FORMAL_PUBLICATION_MODE
    if not formal:
        if receipt is not None:
            raise FullGridError(
                "test-only artifacts cannot claim project GPU-lease authority"
            )
        return
    if run_root is None:
        raise FullGridError("formal GPU-lease receipt validation requires run_root")
    try:
        lease_value = receipt["lease"]
        gpu_uuid = lease_value["gpu_uuid"]
    except (KeyError, TypeError) as error:
        raise FullGridError("formal artifact has no valid GPU-lease receipt") from error
    if expected_gpu_uuid is not None and gpu_uuid != expected_gpu_uuid:
        raise FullGridError("GPU-lease receipt differs from resource provenance")
    if gpu_uuid not in plan["execution_config"]["physical_gpu_uuid_roster"]:
        raise FullGridError("GPU-lease receipt is outside the immutable plan")
    try:
        project_gpu_leases.validate_gpu_lease_receipt(
            receipt,
            project_root=_verified_project_root(),
            run_root=_safe_run_root(run_root),
            plan_sha256=plan["plan_sha256"],
            gpu_uuid=gpu_uuid,
            track_scope=TRACK_SCOPE,
        )
    except project_gpu_leases.ProjectGPULeaseError as error:
        raise FullGridError(f"GPU-lease receipt is invalid: {error}") from error


@contextmanager
def _gpu_lease_publication_authority(
    *,
    plan: Mapping[str, Any],
    run_root: Path,
    gpu_lease: project_gpu_leases.GPULease | None,
    expected_receipt: Mapping[str, Any] | None = None,
) -> Iterator[Mapping[str, Any] | None]:
    """Hold the project registry lock across one authoritative publication."""

    formal = plan["publication_mode"] == FORMAL_PUBLICATION_MODE
    if not formal:
        if gpu_lease is not None or expected_receipt is not None:
            raise FullGridError("test-only publication cannot use a formal GPU lease")
        yield None
        return
    if gpu_lease is None:
        raise FullGridError("formal publication requires an active project GPU lease")
    try:
        project_gpu_leases.assert_gpu_lease(gpu_lease)
        with project_gpu_leases.guard_gpu_lease(gpu_lease) as receipt:
            _validate_persisted_gpu_lease_receipt(
                receipt,
                plan=plan,
                run_root=run_root,
                expected_gpu_uuid=str(gpu_lease.value["gpu_uuid"]),
            )
            if expected_receipt is not None and receipt != dict(expected_receipt):
                raise FullGridError(
                    "claim and commit GPU-lease receipts are not identical"
                )
            yield copy.deepcopy(receipt)
    except project_gpu_leases.ProjectGPULeaseError as error:
        raise FullGridError(f"project GPU-lease authority was lost: {error}") from error


def _validate_claim_value(
    value: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    job: Job,
    run_root: Path | None = None,
    expected_preflight_report_sha256: str | None = None,
    expected_nonce: str | None = None,
    expected_owner: Mapping[str, Any] | None = None,
) -> None:
    _exact_keys(
        value,
        {
            "schema",
            "created_at",
            "plan_sha256",
            "job",
            "job_id",
            "nonce",
            "owner",
            "resource_guard",
            "gpu_lease_receipt",
            "preflight_report_sha256",
        },
        path="claim",
    )
    _exact_keys(
        value["owner"],
        {"host", "pid", "boot_id", "start_ticks"},
        path="claim.owner",
    )
    try:
        created = datetime.fromisoformat(str(value["created_at"]))
    except (TypeError, ValueError) as error:
        raise FullGridError("claim timestamp is invalid") from error
    owner = value["owner"]
    if (
        created.tzinfo is None
        or value["schema"] != CLAIM_SCHEMA
        or value["plan_sha256"] != plan["plan_sha256"]
        or value["job"] != job.identity()
        or value["job_id"] != job.job_id
        or not isinstance(value["nonce"], str)
        or not re.fullmatch(r"[0-9a-f]{32}", value["nonce"])
        or (expected_nonce is not None and value["nonce"] != expected_nonce)
        or (
            expected_owner is not None
            and owner != dict(expected_owner)
        )
        or not isinstance(owner["host"], str)
        or not owner["host"]
        or isinstance(owner["pid"], bool)
        or not isinstance(owner["pid"], int)
        or owner["pid"] <= 0
        or not isinstance(owner["boot_id"], str)
        or not owner["boot_id"]
        or not isinstance(owner["start_ticks"], str)
        or not owner["start_ticks"]
    ):
        raise FullGridError("claim identity is invalid")
    _validate_resource_guard(value["resource_guard"], plan=plan)
    _validate_persisted_gpu_lease_receipt(
        value["gpu_lease_receipt"],
        plan=plan,
        run_root=run_root,
        expected_gpu_uuid=value["resource_guard"]["gpu"]["gpu_uuid"],
    )
    if plan["publication_mode"] == FORMAL_PUBLICATION_MODE:
        if run_root is None:
            raise FullGridError("formal claim requires preflight run-root binding")
        expected_preflight = expected_preflight_report_sha256
        if expected_preflight is None:
            expected_preflight = load_preflight_attestation(
                run_root,
                plan,
            )["report_sha256"]
        elif HEX_64_RE.fullmatch(expected_preflight) is None:
            raise FullGridError("expected preflight digest is invalid")
        if value["preflight_report_sha256"] != expected_preflight:
            raise FullGridError("claim differs from the immutable preflight digest")
    elif (
        value["preflight_report_sha256"] is not None
        or expected_preflight_report_sha256 is not None
    ):
        raise FullGridError("test-only claim cannot bind a formal preflight")


def _publication_fence_path(run_root: Path) -> Path:
    return _safe_run_root(run_root) / PUBLICATION_FENCE_FILENAME


def _acquire_publication_fence(
    run_root: Path,
    *,
    exclusive: bool,
    blocking: bool,
) -> int:
    root = _safe_run_root(run_root, create=True)
    root_descriptor = _open_absolute_directory(root)
    try:
        try:
            _write_bytes_exclusive_at(
                root_descriptor,
                PUBLICATION_FENCE_FILENAME,
                b"",
            )
        except FileExistsError:
            pass
        descriptor = os.open(
            PUBLICATION_FENCE_FILENAME,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
            dir_fd=root_descriptor,
        )
        try:
            observed = os.fstat(descriptor)
            if (
                not stat.S_ISREG(observed.st_mode)
                or observed.st_nlink != 1
                or observed.st_mode & 0o222
            ):
                raise FullGridError(
                    "publication fence is not a unique read-only regular file"
                )
            mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
            if not blocking:
                mode |= fcntl.LOCK_NB
            try:
                fcntl.flock(descriptor, mode)
            except BlockingIOError as error:
                raise ClaimUnavailable("analysis publication is active") from error
            path_stat = os.stat(
                PUBLICATION_FENCE_FILENAME,
                dir_fd=root_descriptor,
                follow_symlinks=False,
            )
            if (
                observed.st_dev != path_stat.st_dev
                or observed.st_ino != path_stat.st_ino
            ):
                raise FullGridError("publication fence path changed")
            _assert_directory_descriptor_path(root, root_descriptor)
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise
    finally:
        os.close(root_descriptor)


def _assert_publication_fence_descriptor(
    run_root: Path,
    descriptor: int,
) -> None:
    root = _safe_run_root(run_root)
    root_descriptor = _open_absolute_directory(root)
    try:
        observed = os.fstat(descriptor)
        path_stat = os.stat(
            PUBLICATION_FENCE_FILENAME,
            dir_fd=root_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(observed.st_mode)
            or observed.st_nlink != 1
            or observed.st_mode & 0o222
            or observed.st_dev != path_stat.st_dev
            or observed.st_ino != path_stat.st_ino
        ):
            raise FullGridError("publication fence ownership was lost")
        _assert_directory_descriptor_path(root, root_descriptor)
    finally:
        os.close(root_descriptor)


@contextmanager
def publication_fence(
    run_root: Path,
    *,
    exclusive: bool,
    blocking: bool = True,
) -> Iterator[int]:
    descriptor = _acquire_publication_fence(
        run_root,
        exclusive=exclusive,
        blocking=blocking,
    )
    try:
        yield descriptor
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _claim_is_live(
    claim: Mapping[str, Any],
    *,
    foreign_claim_timeout_seconds: float = 0.0,
) -> bool:
    if (
        not math.isfinite(float(foreign_claim_timeout_seconds))
        or float(foreign_claim_timeout_seconds) != 0.0
    ):
        raise FullGridError("foreign claim age-stealing is permanently disabled")
    try:
        pid = int(claim["owner"]["pid"])
        owner_host = str(claim["owner"]["host"])
    except (KeyError, TypeError, ValueError):
        return False
    if owner_host != socket.gethostname():
        # A shared filesystem cannot prove a foreign process died.  Wall-clock
        # age is never authority to steal another host's claim.
        return True
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, ValueError):
        return False
    except PermissionError:
        return True
    try:
        current = _process_identity(pid)
    except FullGridError:
        return True
    owner = claim["owner"]
    for field in ("boot_id", "start_ticks"):
        if current.get(field) != owner.get(field):
            return False
    return True


def _quarantine_partials(run_root: Path, job: Job) -> tuple[Path, ...]:
    root = _safe_run_root(run_root)
    partial_root = root / "partials"
    if not partial_root.exists():
        return ()
    _require_real_directory(partial_root)
    recovered: list[Path] = []
    for path in partial_root.glob(f"{job.job_id}.*.partial"):
        observed = _require_real_directory(path)
        expected_identity = (int(observed.st_dev), int(observed.st_ino))
        try:
            recovered.append(
                _quarantine(
                    root,
                    path,
                    category="partials",
                    reason="stale",
                    expected_identity=expected_identity,
                )
            )
        except (FileNotFoundError, ClaimUnavailable):
            # Another recovery worker won this individual partial. The job-level
            # claim race remains protected by O_EXCL below.
            continue
    return tuple(recovered)


def _quarantine_stale_claim_atomically(
    run_root: Path,
    path: Path,
    *,
    plan: Mapping[str, Any],
    job: Job,
    expected_preflight_report_sha256: str | None = None,
    reason: str = "stale",
    live_is_noop: bool = False,
) -> Path | None:
    """Move exactly one validated dead local claim under its inode lock."""

    if reason not in {"stale", "completed-stale"}:
        raise FullGridError("unknown stale-claim quarantine reason")
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
        )
    except FileNotFoundError as error:
        raise ClaimUnavailable("another worker won claim recovery") from error
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ClaimUnavailable("claim owner still holds the inode") from error
        descriptor_stat = os.fstat(descriptor)
        if not stat.S_ISREG(descriptor_stat.st_mode) or descriptor_stat.st_nlink != 1:
            raise FullGridError("claim is not a unique regular file")
        path_stat = path.lstat()
        if (
            descriptor_stat.st_dev != path_stat.st_dev
            or descriptor_stat.st_ino != path_stat.st_ino
        ):
            raise ClaimUnavailable("claim path changed during recovery")
        value = _strict_load_descriptor(descriptor, source=str(path))
        _validate_claim_value(
            value,
            plan=plan,
            job=job,
            run_root=run_root,
            expected_preflight_report_sha256=(
                expected_preflight_report_sha256
            ),
        )
        if _claim_is_live(value):
            if live_is_noop:
                return None
            raise ClaimUnavailable("claim became live during recovery")
        path_stat = path.lstat()
        if (
            descriptor_stat.st_dev != path_stat.st_dev
            or descriptor_stat.st_ino != path_stat.st_ino
        ):
            raise ClaimUnavailable("claim path changed before quarantine")
        return _quarantine(
            run_root,
            path,
            category="claims",
            reason=reason,
            expected_identity=(
                int(descriptor_stat.st_dev),
                int(descriptor_stat.st_ino),
            ),
        )
    finally:
        os.close(descriptor)


def recover_claim_tombstones(
    run_root: Path,
    plan: Mapping[str, Any],
    *,
    expected_preflight_report_sha256: str | None = None,
) -> tuple[Path, ...]:
    """Remove only exact dead-owner release tombstones after a power cut."""

    root = _safe_run_root(run_root)
    jobs = {job.job_id: job for job in iter_jobs(plan)}
    recovered: list[Path] = []
    pattern = re.compile(r"^(job-[0-9a-f]{24})\.([0-9a-f]{32})\.released$")
    root_descriptor = _open_absolute_directory(root)
    try:
        try:
            tombstone_descriptor = _open_relative_directory(
                root_descriptor,
                CLAIM_TOMBSTONE_DIRECTORY,
                create=False,
            )
        except FileNotFoundError:
            return ()
        try:
            for name in sorted(os.listdir(tombstone_descriptor)):
                path = root / CLAIM_TOMBSTONE_DIRECTORY / name
                match = pattern.fullmatch(name)
                if match is None:
                    raise FullGridError(f"unknown claim tombstone: {path}")
                job = jobs.get(match.group(1))
                if job is None:
                    raise FullGridError(
                        f"claim tombstone is outside the plan: {path}"
                    )
                descriptor = os.open(
                    name,
                    os.O_RDONLY
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_NONBLOCK", 0),
                    dir_fd=tombstone_descriptor,
                )
                try:
                    before = os.fstat(descriptor)
                    path_before = os.stat(
                        name,
                        dir_fd=tombstone_descriptor,
                        follow_symlinks=False,
                    )
                    if (
                        not stat.S_ISREG(before.st_mode)
                        or before.st_nlink != 1
                        or before.st_mode & 0o222
                        or (
                            before.st_dev,
                            before.st_ino,
                        )
                        != (
                            path_before.st_dev,
                            path_before.st_ino,
                        )
                    ):
                        raise FullGridError(
                            f"claim tombstone is not an immutable unique "
                            f"file: {path}"
                        )
                    value = _strict_load_descriptor(
                        descriptor,
                        source=str(path),
                    )
                    _validate_claim_value(
                        value,
                        plan=plan,
                        job=job,
                        run_root=root,
                        expected_preflight_report_sha256=(
                            expected_preflight_report_sha256
                        ),
                        expected_nonce=match.group(2),
                    )
                    if _claim_is_live(value):
                        continue
                    path_after = os.stat(
                        name,
                        dir_fd=tombstone_descriptor,
                        follow_symlinks=False,
                    )
                    if (
                        path_after.st_dev,
                        path_after.st_ino,
                        path_after.st_mode,
                    ) != (
                        before.st_dev,
                        before.st_ino,
                        before.st_mode,
                    ):
                        raise FullGridError(
                            f"claim tombstone changed before recovery: {path}"
                        )
                    os.unlink(name, dir_fd=tombstone_descriptor)
                    os.fsync(tombstone_descriptor)
                    _assert_directory_descriptor_path(root, root_descriptor)
                    recovered.append(path)
                finally:
                    os.close(descriptor)
        finally:
            os.close(tombstone_descriptor)
    finally:
        os.close(root_descriptor)
    return tuple(recovered)


def recover_completed_job_auxiliary_state(
    run_root: Path,
    plan: Mapping[str, Any],
    job: Job,
    *,
    recover_stale: bool = True,
    expected_preflight_report_sha256: str | None = None,
) -> tuple[Path, ...]:
    """Recover only stale claim/partial state for a durable valid completion.

    A power cut can occur after the atomic record-directory rename and before
    ``release_claim``.  The completion must be checksum-valid before this
    function considers auxiliary recovery.  A live owner is never disturbed.
    """

    _validate_completion_bound(
        run_root,
        plan,
        job,
        expected_preflight_report_sha256=expected_preflight_report_sha256,
    )
    recovered: list[Path] = []
    claim_path = _claim_path(run_root, job)
    if claim_path.exists() or claim_path.is_symlink():
        if not recover_stale:
            observed = strict_load(claim_path)
            _validate_claim_value(
                observed,
                plan=plan,
                job=job,
                run_root=run_root,
                expected_preflight_report_sha256=(
                    expected_preflight_report_sha256
                ),
            )
            qualifier = "live" if _claim_is_live(observed) else "stale"
            raise ClaimUnavailable(
                f"{job.job_id} has a {qualifier} completed claim"
            )
        try:
            moved = _quarantine_stale_claim_atomically(
                run_root,
                claim_path,
                plan=plan,
                job=job,
                expected_preflight_report_sha256=(
                    expected_preflight_report_sha256
                ),
                reason="completed-stale",
                live_is_noop=True,
            )
            if moved is None:
                return ()
            recovered.append(moved)
        except ClaimUnavailable:
            # A competing recovery may have moved the same stale inode.  With
            # a valid durable completion no new legitimate claim can be
            # acquired, so a missing path is successful contention.
            if claim_path.exists() or claim_path.is_symlink():
                raise
    recovered.extend(_quarantine_partials(run_root, job))
    return tuple(recovered)


def acquire_claim(
    run_root: Path,
    plan: Mapping[str, Any],
    job: Job,
    *,
    recover_stale: bool = True,
    before_claim: Callable[[], GPUStatus | ResourceStatus] | None = None,
    gpu_lease: project_gpu_leases.GPULease | None = None,
) -> Claim:
    """Exclusively claim one job or raise without modifying live work."""

    validate_plan_semantics(plan)
    root = _safe_run_root(run_root)
    preflight_report_sha256 = (
        load_preflight_attestation(root, plan)["report_sha256"]
        if plan["publication_mode"] == FORMAL_PUBLICATION_MODE
        else None
    )
    fence_fd = _acquire_publication_fence(
        root,
        exclusive=False,
        blocking=False,
    )
    try:
        output = _record_directory(root, job)
        if output.exists() or output.is_symlink():
            output_before = os.lstat(output)
            output_identity = (
                int(output_before.st_dev),
                int(output_before.st_ino),
            )
            try:
                _validate_completion_bound(
                    root,
                    plan,
                    job,
                    expected_preflight_report_sha256=preflight_report_sha256,
                )
            except Exception:
                if not recover_stale:
                    raise
                _quarantine_or_lose_claim_race(
                    root,
                    output,
                    category="records",
                    reason="corrupt",
                    expected_identity=output_identity,
                )
            else:
                raise ClaimUnavailable(f"{job.job_id} is already complete")

        path = _claim_path(root, job)
        _ensure_real_directory(root, Path("claims") / job.job_id[-2:])
        if path.exists() or path.is_symlink():
            observed = strict_load(path)
            _validate_claim_value(
                observed,
                plan=plan,
                job=job,
                run_root=root,
                expected_preflight_report_sha256=preflight_report_sha256,
            )
            if _claim_is_live(observed):
                raise ClaimUnavailable(f"{job.job_id} has a live claim")
            if not recover_stale:
                raise ClaimUnavailable(f"{job.job_id} has a stale claim")
            _quarantine_stale_claim_atomically(
                root,
                path,
                plan=plan,
                job=job,
                expected_preflight_report_sha256=preflight_report_sha256,
            )
            _quarantine_partials(root, job)

        if before_claim is None:
            raise FullGridError("claim requires a fresh resource guard")
        resources = before_claim()
        resource_guard = resources.as_dict()
        if not resources.safe:
            if isinstance(resources, ResourceStatus):
                if not bool(resources.disk.get("safe", False)):
                    raise DiskUnavailable(resources.reason)
                if not bool(resources.gpu.get("safe", False)):
                    raise GPUUnavailable(resources.reason)
            raise GPUUnavailable(resources.reason)
        _validate_resource_guard(resource_guard, plan=plan)

        nonce = uuid.uuid4().hex
        owner = _process_identity()
        claim: Claim | None = None
        published_claim_identity: tuple[int, int] | None = None
        try:
            with _gpu_lease_publication_authority(
                plan=plan,
                run_root=root,
                gpu_lease=gpu_lease,
            ) as gpu_lease_receipt:
                value = {
                    "schema": CLAIM_SCHEMA,
                    "created_at": _utc_now(),
                    "plan_sha256": plan["plan_sha256"],
                    "job": job.identity(),
                    "job_id": job.job_id,
                    "nonce": nonce,
                    "owner": owner,
                    "resource_guard": resource_guard,
                    "gpu_lease_receipt": copy.deepcopy(gpu_lease_receipt),
                    "preflight_report_sha256": preflight_report_sha256,
                }
                _validate_claim_value(
                    value,
                    plan=plan,
                    job=job,
                    run_root=root,
                    expected_preflight_report_sha256=preflight_report_sha256,
                    expected_nonce=nonce,
                    expected_owner=owner,
                )

                def remember_published_claim(
                    published_stat: os.stat_result,
                ) -> None:
                    nonlocal published_claim_identity
                    published_claim_identity = (
                        int(published_stat.st_dev),
                        int(published_stat.st_ino),
                    )

                try:
                    published_stat = _write_json_exclusive(
                        path,
                        value,
                        on_publish=remember_published_claim,
                    )
                except FileExistsError as error:
                    raise ClaimUnavailable(
                        f"{job.job_id} claim race lost"
                    ) from error
                except Exception:
                    if published_claim_identity is None:
                        try:
                            candidate_payload = _read_unique_regular(path)
                            candidate_stat = _require_unique_regular(
                                path,
                                read_only=True,
                            )
                        except (FileNotFoundError, OSError, FullGridError):
                            pass
                        else:
                            if candidate_payload == _canonical_bytes(value) + b"\n":
                                published_claim_identity = (
                                    int(candidate_stat.st_dev),
                                    int(candidate_stat.st_ino),
                                )
                    raise
                if (
                    int(published_stat.st_dev),
                    int(published_stat.st_ino),
                ) != published_claim_identity:
                    raise FullGridError(
                        "exclusive claim writer returned a different inode"
                    )
                observed = strict_load(path)
                _validate_claim_value(
                    observed,
                    plan=plan,
                    job=job,
                    run_root=root,
                    expected_preflight_report_sha256=(
                        preflight_report_sha256
                    ),
                    expected_nonce=nonce,
                    expected_owner=owner,
                )
                observed_stat = _require_unique_regular(path, read_only=True)
                if (
                    observed_stat.st_dev,
                    observed_stat.st_ino,
                ) != published_claim_identity:
                    raise FullGridError(
                        "new claim path changed before ownership construction"
                    )
                _assert_publication_fence_descriptor(root, fence_fd)
                claim = Claim(
                    job=job,
                    path=path,
                    nonce=nonce,
                    owner=owner,
                    resource_guard=copy.deepcopy(observed["resource_guard"]),
                    gpu_lease_receipt=copy.deepcopy(
                        observed["gpu_lease_receipt"]
                    ),
                    preflight_report_sha256=observed[
                        "preflight_report_sha256"
                    ],
                    st_dev=int(observed_stat.st_dev),
                    st_ino=int(observed_stat.st_ino),
                    value=copy.deepcopy(observed),
                    fence_fd=fence_fd,
                )
        except BaseException:
            if published_claim_identity is not None:
                try:
                    _quarantine(
                        root,
                        path,
                        category="claims",
                        reason="lease-postcondition",
                        expected_identity=published_claim_identity,
                    )
                except (FileNotFoundError, ClaimUnavailable):
                    # The exact newly-published claim is already absent.  A
                    # replacement at the canonical name is never moved.
                    pass
            raise
        assert claim is not None
        return claim
    except BaseException:
        os.close(fence_fd)
        raise


def release_claim(claim: Claim) -> None:
    root = claim.path.parents[2]
    root_descriptor: int | None = None
    claim_parent_descriptor: int | None = None
    tombstone_descriptor: int | None = None
    try:
        _assert_publication_fence_descriptor(
            root,
            claim.fence_fd,
        )
        root_descriptor = _open_absolute_directory(root)
        relative_parent = claim.path.parent.relative_to(root)
        claim_parent_descriptor = _open_relative_directory(
            root_descriptor,
            relative_parent,
            create=False,
        )
        try:
            descriptor = os.open(
                claim.path.name,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0),
                dir_fd=claim_parent_descriptor,
            )
        except FileNotFoundError as error:
            raise FullGridError("owned claim disappeared before release") from error
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            descriptor_stat = os.fstat(descriptor)
            path_stat = os.stat(
                claim.path.name,
                dir_fd=claim_parent_descriptor,
                follow_symlinks=False,
            )
            observed = _strict_load_descriptor(
                descriptor,
                source=str(claim.path),
            )
            if (
                descriptor_stat.st_dev != claim.st_dev
                or descriptor_stat.st_ino != claim.st_ino
                or path_stat.st_dev != claim.st_dev
                or path_stat.st_ino != claim.st_ino
                or observed != dict(claim.value)
                or observed.get("nonce") != claim.nonce
                or observed.get("owner") != dict(claim.owner)
            ):
                raise FullGridError("refusing to release a claim owned elsewhere")
            tombstone_descriptor = _open_relative_directory(
                root_descriptor,
                CLAIM_TOMBSTONE_DIRECTORY,
                create=True,
            )
            tombstone_name = (
                f"{claim.job.job_id}.{claim.nonce}.released"
            )
            _atomic_rename_noreplace_at(
                claim_parent_descriptor,
                claim.path.name,
                tombstone_descriptor,
                tombstone_name,
            )
            os.fsync(claim_parent_descriptor)
            os.fsync(tombstone_descriptor)
            _assert_directory_descriptor_path(root, root_descriptor)
            os.unlink(tombstone_name, dir_fd=tombstone_descriptor)
            os.fsync(tombstone_descriptor)
            _assert_directory_descriptor_path(root, root_descriptor)
        finally:
            os.close(descriptor)
    finally:
        if tombstone_descriptor is not None:
            os.close(tombstone_descriptor)
        if claim_parent_descriptor is not None:
            os.close(claim_parent_descriptor)
        if root_descriptor is not None:
            os.close(root_descriptor)
        try:
            fcntl.flock(claim.fence_fd, fcntl.LOCK_UN)
        finally:
            os.close(claim.fence_fd)


def _recursive_forbidden_keys(value: Any, path: str = "record") -> list[str]:
    violations: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower()
            if normalized in FORBIDDEN_SCORE_BLIND_KEYS:
                violations.append(f"{path}.{key}")
            violations.extend(_recursive_forbidden_keys(child, f"{path}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            violations.extend(_recursive_forbidden_keys(child, f"{path}[{index}]"))
    return violations


def _planned_cache_identity(
    plan: Mapping[str, Any],
    job: Job,
) -> Mapping[str, Any]:
    key = _subject_identity_key(job.dataset, job.subject)
    try:
        identity = plan["cache_identity"][key]
    except (KeyError, TypeError) as error:
        raise FullGridError(f"plan has no cache identity for {key}") from error
    if not isinstance(identity, Mapping):
        raise FullGridError(f"plan cache identity {key} is not an object")
    return identity


def _planned_split_identity(
    plan: Mapping[str, Any],
    job: Job,
) -> Mapping[str, Any]:
    key = _split_identity_key(job.dataset, job.subject, job.fold)
    try:
        identity = plan["split_identity"][key]
    except (KeyError, TypeError) as error:
        raise FullGridError(f"plan has no split identity for {key}") from error
    if not isinstance(identity, Mapping):
        raise FullGridError(f"plan split identity {key} is not an object")
    return identity


def _split_metadata(identity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "trial_count": int(identity["trial_count"]),
        "partitions": copy.deepcopy(dict(identity["partitions"])),
    }


def _validate_job_metadata(
    plan: Mapping[str, Any],
    job: Job,
    metadata: Mapping[str, Any],
) -> None:
    """Require the source-only fitting and provenance contract for every record."""

    _exact_keys(
        metadata,
        {
            "cache_array_sha256",
            "channels",
            "split",
            "scalers",
            "fit",
            "protocol",
            "timing_seconds",
            "cuda_peak_memory_bytes",
            "runtime",
        },
        path="record.metadata",
    )
    expected_cache = _planned_cache_identity(plan, job)
    expected_split = _planned_split_identity(plan, job)
    if metadata.get("cache_array_sha256") != expected_cache.get("array_sha256"):
        raise FullGridError("executor metadata cache identity differs from the plan")
    if metadata.get("split") != _split_metadata(expected_split):
        raise FullGridError("executor metadata split identity differs from the plan")

    channels = metadata.get("channels")
    if (
        not isinstance(channels, list)
        or not channels
        or any(not isinstance(value, str) or not value for value in channels)
        or len(set(channels)) != len(channels)
    ):
        raise FullGridError("executor metadata has an invalid channel contract")

    scalers = metadata.get("scalers")
    required_scaler_hashes = {
        "selection_mean_sha256",
        "selection_std_sha256",
        "refit_mean_sha256",
        "refit_std_sha256",
    }
    if (
        not isinstance(scalers, Mapping)
        or set(scalers) != required_scaler_hashes
        or any(
            not isinstance(value, str) or not HEX_64_RE.fullmatch(value)
            for value in scalers.values()
        )
    ):
        raise FullGridError("executor metadata has an invalid scaler provenance")

    fit = metadata.get("fit")
    if not isinstance(fit, Mapping):
        raise FullGridError("executor metadata has no fit contract")
    _exact_keys(
        fit,
        {
            "seed_installed_before_construction",
            "initial_state_sha256",
            "selection_state_sha256",
            "reset_state_sha256",
            "reset_verified",
            "source_selected_epoch",
            "selection_epochs_run",
            "refit_epochs_run",
            "refit_state_sha256",
            "parameter_count",
            "architecture",
        },
        path="record.metadata.fit",
    )
    if fit.get("seed_installed_before_construction") is not True:
        raise FullGridError(
            "executor metadata does not prove seeding before model construction"
        )
    initial_hash = fit.get("initial_state_sha256")
    reset_hash = fit.get("reset_state_sha256")
    state_hash_fields = (
        initial_hash,
        fit.get("selection_state_sha256"),
        reset_hash,
        fit.get("refit_state_sha256"),
    )
    if any(
        not isinstance(value, str) or not HEX_64_RE.fullmatch(value)
        for value in state_hash_fields
    ):
        raise FullGridError("executor metadata contains an invalid model-state hash")
    if fit.get("reset_verified") is not True or initial_hash != reset_hash:
        raise FullGridError("executor metadata does not prove initialization reset")
    counter_names = (
        "source_selected_epoch",
        "selection_epochs_run",
        "refit_epochs_run",
        "parameter_count",
    )
    if any(type(fit.get(name)) is not int for name in counter_names):
        raise FullGridError("executor metadata has invalid fit counters")
    selected_epoch = fit["source_selected_epoch"]
    selection_epochs = fit["selection_epochs_run"]
    refit_epochs = fit["refit_epochs_run"]
    parameter_count = fit["parameter_count"]
    maximum_epochs = int(plan["train_config"]["epochs"])
    if (
        selected_epoch < 0
        or selection_epochs < selected_epoch + 1
        or selection_epochs > maximum_epochs
        or refit_epochs != selected_epoch + 1
        or parameter_count <= 0
    ):
        raise FullGridError("executor metadata violates selection/reset/refit semantics")
    architecture = fit.get("architecture")
    if (
        not isinstance(architecture, Mapping)
        or architecture.get("requested_name") != job.model
        or not {
            "requested_name",
            "class",
            "wrapped_class",
            "scalar_attributes",
        }.issubset(architecture)
        or not set(architecture).issubset(
            {
                "requested_name",
                "class",
                "wrapped_class",
                "config",
                "scalar_attributes",
                "third_party_provenance",
            }
        )
        or not isinstance(architecture.get("class"), str)
        or not isinstance(architecture.get("wrapped_class"), str)
        or not isinstance(architecture.get("scalar_attributes"), Mapping)
    ):
        raise FullGridError("executor metadata architecture differs from the job")
    scalar_attributes = architecture["scalar_attributes"]
    if any(
        not isinstance(key, str)
        or not isinstance(value, (str, bool, int, float, list, type(None)))
        or (
            isinstance(value, list)
            and any(
                not isinstance(item, (str, bool, int, float, type(None)))
                for item in value
            )
        )
        or (
            isinstance(value, float)
            and not math.isfinite(value)
        )
        for key, value in scalar_attributes.items()
    ):
        raise FullGridError("executor architecture scalar attributes are invalid")
    if "F1" in scalar_attributes:
        expected_f1 = {"eegnet": 8, "tcformer": 32}.get(job.model)
        if expected_f1 is None or scalar_attributes["F1"] != expected_f1:
            raise FullGridError("architecture F1 is not a documented source parameter")
    if "config" in architecture and not isinstance(architecture["config"], Mapping):
        raise FullGridError("executor architecture config is invalid")
    if (
        job.model == "tcformer"
        and plan["publication_mode"] == FORMAL_PUBLICATION_MODE
    ):
        if (
            set(architecture)
            != {
                "requested_name",
                "class",
                "wrapped_class",
                "scalar_attributes",
                "third_party_provenance",
            }
            or architecture["class"]
            != "_ieee_tcformer.models.tcformer.TCFormerModule"
            or architecture["wrapped_class"]
            != "_ieee_tcformer.models.tcformer.TCFormerModule"
            or architecture["third_party_provenance"]
            != plan["tcformer_source_identity"]
        ):
            raise FullGridError("TCFormer record provenance differs from the plan")
    elif (
        plan["publication_mode"] == FORMAL_PUBLICATION_MODE
        and "third_party_provenance" in architecture
    ):
        raise FullGridError("non-TCFormer record contains third-party provenance")

    protocol = metadata.get("protocol")
    expected_protocol = {
        "selection_scaler_rows": "train_only",
        "selection_optimization_rows": "train_only",
        "epoch_selection_rows": "validation_only",
        "refit_scaler_rows": "train_plus_validation",
        "refit_optimization_rows": "train_plus_validation",
        "test_use": "prediction_only",
        "test_performance_computed": False,
    }
    if protocol != expected_protocol:
        raise FullGridError("executor metadata has an invalid source/test protocol")

    timing = metadata.get("timing_seconds")
    if not isinstance(timing, Mapping) or set(timing) != {
        "selection_fit",
        "refit_fit",
        "test_inference",
        "job_total",
    }:
        raise FullGridError("executor metadata has an invalid timing contract")
    for name, raw_value in timing.items():
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise FullGridError(f"executor timing {name} is invalid")
        value = float(raw_value)
        if not math.isfinite(value) or value < 0.0:
            raise FullGridError(f"executor timing {name} is invalid")
    peak_memory = metadata.get("cuda_peak_memory_bytes")
    if (
        isinstance(peak_memory, bool)
        or not isinstance(peak_memory, int)
        or peak_memory < 0
    ):
        raise FullGridError("executor metadata has invalid CUDA peak memory")

    expected_threads = int(plan["execution_config"]["worker_cpu_threads"])
    runtime = metadata.get("runtime")
    expected_thread_environment = {
        name: str(expected_threads) for name in THREAD_ENVIRONMENT_VARIABLES
    }
    expected_driver_versions = plan.get("environment_identity", {}).get(
        "nvidia_driver_versions"
    )
    if (
        not isinstance(runtime, Mapping)
        or set(runtime)
        != {
            "worker_cpu_threads",
            "worker_interop_threads",
            "thread_environment",
            "nvidia_driver_versions",
            "physical_gpu_uuid",
            "cuda_visible_devices",
        }
        or runtime.get("worker_cpu_threads") != expected_threads
        or runtime.get("worker_interop_threads") != 1
        or runtime.get("thread_environment") != expected_thread_environment
        or runtime.get("nvidia_driver_versions") != expected_driver_versions
    ):
        raise FullGridError(
            "executor metadata has an invalid worker runtime provenance"
        )
    if plan["publication_mode"] == FORMAL_PUBLICATION_MODE:
        if (
            runtime.get("physical_gpu_uuid")
            not in plan["execution_config"]["physical_gpu_uuid_roster"]
            or runtime.get("cuda_visible_devices")
            != runtime.get("physical_gpu_uuid")
        ):
            raise FullGridError("executor runtime is not bound to a planned GPU")
    elif (
        runtime.get("physical_gpu_uuid") is not None
        or runtime.get("cuda_visible_devices") is not None
    ):
        raise FullGridError("test metadata cannot claim a CUDA binding")


@contextmanager
def _owned_claim_lock(
    claim: Claim,
    *,
    plan: Mapping[str, Any],
    expected_preflight_report_sha256: str | None = None,
) -> Iterator[int]:
    root = claim.path.parents[2]
    root_descriptor = _open_absolute_directory(root)
    try:
        claim_parent_descriptor = _open_relative_directory(
            root_descriptor,
            claim.path.parent.relative_to(root),
            create=False,
        )
    except BaseException:
        os.close(root_descriptor)
        raise
    try:
        descriptor = os.open(
            claim.path.name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
            dir_fd=claim_parent_descriptor,
        )
    except BaseException as error:
        os.close(claim_parent_descriptor)
        os.close(root_descriptor)
        if not isinstance(error, FileNotFoundError):
            raise
        raise FullGridError("owned claim is absent") from error
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        descriptor_stat = os.fstat(descriptor)
        path_stat = os.stat(
            claim.path.name,
            dir_fd=claim_parent_descriptor,
            follow_symlinks=False,
        )
        observed = _strict_load_descriptor(descriptor, source=str(claim.path))
        _validate_claim_value(
            observed,
            plan=plan,
            job=claim.job,
            run_root=root,
            expected_preflight_report_sha256=(
                expected_preflight_report_sha256
            ),
            expected_nonce=claim.nonce,
            expected_owner=claim.owner,
        )
        if (
            descriptor_stat.st_dev != claim.st_dev
            or descriptor_stat.st_ino != claim.st_ino
            or path_stat.st_dev != claim.st_dev
            or path_stat.st_ino != claim.st_ino
            or observed != dict(claim.value)
        ):
            raise FullGridError("claim token/inode ownership was lost")
        _assert_directory_descriptor_path(root, root_descriptor)
        yield descriptor
        _assert_directory_descriptor_path(root, root_descriptor)
    finally:
        os.close(descriptor)
        os.close(claim_parent_descriptor)
        os.close(root_descriptor)


def _recheck_owned_claim_path(claim: Claim, descriptor: int) -> None:
    descriptor_stat = os.fstat(descriptor)
    root = claim.path.parents[2]
    root_descriptor = _open_absolute_directory(root)
    try:
        claim_parent_descriptor = _open_relative_directory(
            root_descriptor,
            claim.path.parent.relative_to(root),
            create=False,
        )
        try:
            try:
                path_stat = os.stat(
                    claim.path.name,
                    dir_fd=claim_parent_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError as error:
                raise FullGridError(
                    "owned claim disappeared before publish"
                ) from error
            if (
                descriptor_stat.st_dev != claim.st_dev
                or descriptor_stat.st_ino != claim.st_ino
                or path_stat.st_dev != claim.st_dev
                or path_stat.st_ino != claim.st_ino
            ):
                raise FullGridError("owned claim path changed before publish")
            _assert_directory_descriptor_path(root, root_descriptor)
        finally:
            os.close(claim_parent_descriptor)
    finally:
        os.close(root_descriptor)


def _validate_record_object(
    record: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    job: Job,
    run_root: Path | None = None,
    expected_preflight_report_sha256: str | None = None,
) -> None:
    _exact_keys(
        record,
        {
            "schema",
            "created_at",
            "plan_sha256",
            "job_id",
            "job",
            "score_blind",
            "claim_resource_guard",
            "commit_resource_guard",
            "gpu_lease_receipt",
            "preflight_report_sha256",
            "test_count",
            "test_rows_sha256",
            "metadata",
        },
        path="record",
    )
    aliases = _recursive_outcome_aliases(record)
    if aliases:
        raise FullGridError(
            f"score-blind record contains outcome aliases: {aliases[:5]}"
        )
    try:
        created = datetime.fromisoformat(str(record["created_at"]))
    except (TypeError, ValueError) as error:
        raise FullGridError("record timestamp is invalid") from error
    if (
        created.tzinfo is None
        or record["schema"] != RECORD_SCHEMA
        or record["plan_sha256"] != plan["plan_sha256"]
        or record["job_id"] != job.job_id
        or record["job"] != job.identity()
        or record["score_blind"] is not True
        or isinstance(record["test_count"], bool)
        or not isinstance(record["test_count"], int)
        or record["test_count"] <= 0
        or not isinstance(record["test_rows_sha256"], str)
        or not HEX_64_RE.fullmatch(record["test_rows_sha256"])
    ):
        raise FullGridError("record identity is invalid")
    _validate_resource_guard(record["claim_resource_guard"], plan=plan)
    _validate_resource_guard(record["commit_resource_guard"], plan=plan)
    _validate_persisted_gpu_lease_receipt(
        record["gpu_lease_receipt"],
        plan=plan,
        run_root=run_root,
        expected_gpu_uuid=record["commit_resource_guard"]["gpu"]["gpu_uuid"],
    )
    if plan["publication_mode"] == FORMAL_PUBLICATION_MODE:
        if run_root is None:
            raise FullGridError("formal record requires preflight run-root binding")
        expected_preflight = expected_preflight_report_sha256
        if expected_preflight is None:
            expected_preflight = load_preflight_attestation(
                run_root,
                plan,
            )["report_sha256"]
        elif HEX_64_RE.fullmatch(expected_preflight) is None:
            raise FullGridError("expected preflight digest is invalid")
        if record["preflight_report_sha256"] != expected_preflight:
            raise FullGridError("record differs from the immutable preflight digest")
    elif (
        record["preflight_report_sha256"] is not None
        or expected_preflight_report_sha256 is not None
    ):
        raise FullGridError("test-only record cannot bind a formal preflight")
    _validate_job_metadata(plan, job, record["metadata"])
    if (
        plan["publication_mode"] == FORMAL_PUBLICATION_MODE
        and (
            record["claim_resource_guard"]["gpu"]["gpu_uuid"]
            != record["commit_resource_guard"]["gpu"]["gpu_uuid"]
            or record["metadata"]["runtime"]["physical_gpu_uuid"]
            != record["commit_resource_guard"]["gpu"]["gpu_uuid"]
            or record["gpu_lease_receipt"]["lease"]["gpu_uuid"]
            != record["claim_resource_guard"]["gpu"]["gpu_uuid"]
        )
    ):
        raise FullGridError("GPU binding changed between claim and publication")


def _validate_completion_object(
    completion: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    job: Job,
) -> None:
    _exact_keys(
        completion,
        {"schema", "created_at", "plan_sha256", "job_id", "job", "files"},
        path="completion",
    )
    _exact_keys(
        completion["files"],
        {"record.json", "predictions.npz"},
        path="completion.files",
    )
    aliases = _recursive_outcome_aliases(completion, "completion")
    if aliases:
        raise FullGridError(
            f"completion contains outcome aliases: {aliases[:5]}"
        )
    try:
        created = datetime.fromisoformat(str(completion["created_at"]))
    except (TypeError, ValueError) as error:
        raise FullGridError("completion timestamp is invalid") from error
    if (
        created.tzinfo is None
        or completion["schema"] != COMPLETION_SCHEMA
        or completion["plan_sha256"] != plan["plan_sha256"]
        or completion["job_id"] != job.job_id
        or completion["job"] != job.identity()
        or any(
            not isinstance(value, str) or not HEX_64_RE.fullmatch(value)
            for value in completion["files"].values()
        )
    ):
        raise FullGridError("completion receipt identity is invalid")


def commit_job_output(
    run_root: Path,
    plan: Mapping[str, Any],
    claim: Claim,
    *,
    metadata: Mapping[str, Any],
    test_rows: np.ndarray,
    probabilities: np.ndarray,
    cache_root: Path | None = None,
    before_publish: Callable[[], GPUStatus | ResourceStatus] | None = None,
    gpu_lease: project_gpu_leases.GPULease | None = None,
) -> Path:
    """Write and atomically publish one immutable score-blind job directory."""

    job = claim.job
    rows = np.asarray(test_rows)
    values = np.asarray(probabilities)
    n_classes = int(plan["datasets"][job.dataset]["n_classes"])
    expected_split = _planned_split_identity(plan, job)
    expected_test = expected_split["partitions"]["test"]
    if (
        rows.dtype != np.dtype(np.int64)
        or rows.ndim != 1
        or len(rows) == 0
        or np.any(rows < 0)
        or np.any(rows >= int(expected_split["trial_count"]))
        or len(set(rows.tolist())) != len(rows)
        or len(rows) != int(expected_test["count"])
        or _rows_sha256(rows) != expected_test["rows_sha256"]
        or values.dtype != np.dtype(np.float64)
        or values.shape != (len(rows), n_classes)
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(values > 1.0)
        or not np.allclose(values.sum(axis=1), 1.0, atol=1e-6)
    ):
        raise FullGridError("executor returned invalid score-blind prediction arrays")
    rows = np.ascontiguousarray(rows)
    values = np.ascontiguousarray(values)
    _validate_job_metadata(plan, job, metadata)
    validate_plan_semantics(plan)
    root = _safe_run_root(run_root)
    preflight_report_sha256: str | None = None
    if plan["publication_mode"] == FORMAL_PUBLICATION_MODE:
        preflight_report_sha256 = load_preflight_attestation(
            root,
            plan,
        )["report_sha256"]
        if claim.preflight_report_sha256 != preflight_report_sha256:
            raise FullGridError(
                "claim differs from the immutable preflight attestation"
            )
    if plan["publication_mode"] == FORMAL_PUBLICATION_MODE and cache_root is None:
        raise FullGridError("formal commit requires cache-root identity recheck")
    if plan["publication_mode"] == FORMAL_PUBLICATION_MODE:
        if gpu_lease is None:
            raise FullGridError("formal commit requires an active project GPU lease")
        try:
            project_gpu_leases.assert_gpu_lease(gpu_lease)
        except project_gpu_leases.ProjectGPULeaseError as error:
            raise FullGridError(
                f"project GPU lease was lost before commit: {error}"
            ) from error
        if before_publish is None:
            raise FullGridError("formal commit requires a pre-write resource probe")
        prewrite_resources = before_publish()
        if not prewrite_resources.safe:
            if isinstance(prewrite_resources, ResourceStatus) and not bool(
                prewrite_resources.disk.get("safe", False)
            ):
                raise DiskUnavailable(prewrite_resources.reason)
            raise GPUUnavailable(prewrite_resources.reason)
        _validate_resource_guard(prewrite_resources.as_dict(), plan=plan)
    _assert_publication_fence_descriptor(root, claim.fence_fd)
    partial_root = _ensure_real_directory(root, "partials")
    partial = partial_root / f"{job.job_id}.{claim.nonce}.partial"
    root_descriptor = _open_absolute_directory(root)
    partial_root_descriptor = _open_relative_directory(
        root_descriptor,
        "partials",
        create=True,
    )
    partial_name = partial.name
    os.mkdir(partial_name, 0o700, dir_fd=partial_root_descriptor)
    os.fsync(partial_root_descriptor)
    partial_descriptor = os.open(
        partial_name,
        _directory_open_flags(),
        dir_fd=partial_root_descriptor,
    )
    partial_stat = os.fstat(partial_descriptor)
    partial_identity = (
        int(partial_stat.st_dev),
        int(partial_stat.st_ino),
    )
    destination_parent_descriptor: int | None = None
    destination: Path | None = None
    record_was_renamed = False
    try:
        descriptor = os.open(
            "predictions.npz",
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
            0o600,
            dir_fd=partial_descriptor,
        )
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as handle:
                np.savez_compressed(handle, rows=rows, probabilities=values)
                handle.flush()
                os.fchmod(handle.fileno(), 0o444)
                os.fsync(handle.fileno())
        finally:
            os.close(descriptor)
        with _owned_claim_lock(
            claim,
            plan=plan,
            expected_preflight_report_sha256=preflight_report_sha256,
        ) as claim_descriptor:
            if cache_root is not None:
                _verify_job_runtime_identity(
                    plan,
                    job,
                    cache_root=cache_root,
                    worker_cuda_visibility=True,
                )
            if before_publish is None:
                if plan["publication_mode"] == FORMAL_PUBLICATION_MODE:
                    raise FullGridError(
                        "formal commit requires a fresh resource probe"
                    )
                commit_resource = copy.deepcopy(dict(claim.resource_guard))
            else:
                resources = before_publish()
                if not resources.safe:
                    if isinstance(resources, ResourceStatus) and not bool(
                        resources.disk.get("safe", False)
                    ):
                        raise DiskUnavailable(resources.reason)
                    raise GPUUnavailable(resources.reason)
                commit_resource = resources.as_dict()
            _validate_resource_guard(commit_resource, plan=plan)
            record = {
                "schema": RECORD_SCHEMA,
                "created_at": _utc_now(),
                "plan_sha256": plan["plan_sha256"],
                "job_id": job.job_id,
                "job": job.identity(),
                "score_blind": True,
                "claim_resource_guard": copy.deepcopy(dict(claim.resource_guard)),
                "commit_resource_guard": copy.deepcopy(dict(commit_resource)),
                "gpu_lease_receipt": copy.deepcopy(
                    claim.gpu_lease_receipt
                ),
                "preflight_report_sha256": claim.preflight_report_sha256,
                "test_count": len(rows),
                "test_rows_sha256": _rows_sha256(rows),
                "metadata": copy.deepcopy(dict(metadata)),
            }
            _validate_record_object(
                record,
                plan=plan,
                job=job,
                run_root=root,
                expected_preflight_report_sha256=preflight_report_sha256,
            )
            record_payload = _canonical_bytes(record) + b"\n"
            _write_bytes_exclusive_at(
                partial_descriptor,
                "record.json",
                record_payload,
            )
            completion = {
                "schema": COMPLETION_SCHEMA,
                "created_at": _utc_now(),
                "plan_sha256": plan["plan_sha256"],
                "job_id": job.job_id,
                "job": job.identity(),
                "files": {
                    "record.json": hashlib.sha256(record_payload).hexdigest(),
                    "predictions.npz": hashlib.sha256(
                        _read_unique_regular_at(
                            partial_descriptor,
                            "predictions.npz",
                        )
                    ).hexdigest(),
                },
            }
            _validate_completion_object(completion, plan=plan, job=job)
            _write_bytes_exclusive_at(
                partial_descriptor,
                "completion.json",
                _canonical_bytes(completion) + b"\n",
            )
            os.fsync(partial_descriptor)
            os.fchmod(partial_descriptor, 0o555)
            os.fsync(partial_descriptor)
            staged_directory_stat = os.fstat(partial_descriptor)
            if (
                not stat.S_ISDIR(staged_directory_stat.st_mode)
                or staged_directory_stat.st_mode & 0o222
            ):
                raise FullGridError(
                    "staged record directory is not immutable before publish"
                )

            destination = _record_directory(root, job)
            relative_parent = destination.parent.relative_to(root)
            destination_parent_descriptor = _open_relative_directory(
                root_descriptor,
                relative_parent,
                create=True,
            )

            def publish_staged_directory() -> Path:
                nonlocal destination_parent_descriptor, record_was_renamed
                assert destination is not None
                _recheck_owned_claim_path(claim, claim_descriptor)
                _assert_publication_fence_descriptor(root, claim.fence_fd)
                try:
                    _assert_directory_descriptor_path(root, root_descriptor)
                    _publish_sealed_directory_noreplace_at(
                        partial_root_descriptor,
                        partial_name,
                        partial_descriptor,
                        destination_parent_descriptor,
                        destination.name,
                        expected_names=FINAL_FILENAMES,
                    )
                    record_was_renamed = True
                    published_directory_stat = os.stat(
                        destination.name,
                        dir_fd=destination_parent_descriptor,
                        follow_symlinks=False,
                    )
                    if (
                        not stat.S_ISDIR(published_directory_stat.st_mode)
                        or published_directory_stat.st_mode & 0o222
                        or (
                            published_directory_stat.st_dev,
                            published_directory_stat.st_ino,
                        )
                        != (
                            staged_directory_stat.st_dev,
                            staged_directory_stat.st_ino,
                        )
                    ):
                        raise FullGridError(
                            "published record directory differs from its "
                            "immutable staged inode"
                        )
                except FileExistsError:
                    _assert_directory_descriptor_path(root, root_descriptor)
                    _validate_completion_bound(
                        root,
                        plan,
                        job,
                        expected_preflight_report_sha256=(
                            preflight_report_sha256
                        ),
                    )
                    _assert_directory_descriptor_path(root, root_descriptor)
                    _quarantine_at(
                        root_descriptor,
                        root,
                        partial.relative_to(root),
                        category="partials",
                        reason="duplicate",
                        expected_identity=partial_identity,
                    )
                    os.close(destination_parent_descriptor)
                    destination_parent_descriptor = None
                    return destination
                except BaseException:
                    try:
                        os.stat(
                            partial_name,
                            dir_fd=partial_root_descriptor,
                            follow_symlinks=False,
                        )
                    except FileNotFoundError:
                        # The stage left its anchored source name. Any
                        # canonical destination is therefore part of this
                        # failed publication and must be invalidated by the
                        # outer transaction before the error escapes.
                        record_was_renamed = True
                    raise
                try:
                    _recheck_owned_claim_path(claim, claim_descriptor)
                    _assert_publication_fence_descriptor(root, claim.fence_fd)
                    _assert_directory_descriptor_path(root, root_descriptor)
                except Exception:
                    _quarantine_at(
                        root_descriptor,
                        root,
                        destination.relative_to(root),
                        category="records",
                        reason="lost-claim-after-rename",
                        expected_identity=partial_identity,
                    )
                    record_was_renamed = False
                    os.close(destination_parent_descriptor)
                    destination_parent_descriptor = None
                    raise
                os.fsync(destination_parent_descriptor)
                _assert_directory_descriptor_path(root, root_descriptor)
                _validate_completion_bound(
                    root,
                    plan,
                    job,
                    expected_preflight_report_sha256=preflight_report_sha256,
                )
                _assert_directory_descriptor_path(root, root_descriptor)
                os.close(destination_parent_descriptor)
                destination_parent_descriptor = None
                return destination

            with _gpu_lease_publication_authority(
                plan=plan,
                run_root=root,
                gpu_lease=gpu_lease,
                expected_receipt=claim.gpu_lease_receipt,
            ) as publication_receipt:
                if publication_receipt != record["gpu_lease_receipt"]:
                    raise FullGridError(
                        "persisted record differs from guarded GPU-lease receipt"
                    )
                published = publish_staged_directory()
            return published
    except Exception:
        if record_was_renamed and destination is not None:
            try:
                _quarantine_at(
                    root_descriptor,
                    root,
                    destination.relative_to(root),
                    category="records",
                    reason="lease-postcondition",
                    expected_identity=partial_identity,
                )
            except (FileNotFoundError, ClaimUnavailable):
                # A disappeared canonical name is already unusable, and an
                # inode mismatch is a replacement winner that must survive.
                pass
            record_was_renamed = False
        try:
            os.stat(
                partial_name,
                dir_fd=partial_root_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            try:
                _quarantine_at(
                    root_descriptor,
                    root,
                    partial.relative_to(root),
                    category="partials",
                    reason="failed",
                    expected_identity=partial_identity,
                )
            except (FileNotFoundError, ClaimUnavailable):
                # A replacement partial belongs to a competing recovery and
                # is never invalidated by this failed commit.
                pass
        raise
    finally:
        if destination_parent_descriptor is not None:
            os.close(destination_parent_descriptor)
        os.close(partial_descriptor)
        os.close(partial_root_descriptor)
        os.close(root_descriptor)


def _validate_completion_bound(
    run_root: Path,
    plan: Mapping[str, Any],
    job: Job,
    *,
    expected_preflight_report_sha256: str | None = None,
) -> dict[str, Any]:
    validate_plan_semantics(plan)
    root = _safe_run_root(run_root)
    directory = _record_directory(root, job)
    root_descriptor = _open_absolute_directory(root)
    try:
        try:
            directory_descriptor = _open_relative_directory(
                root_descriptor,
                directory.relative_to(root),
                create=False,
            )
        except FileNotFoundError:
            raise FileNotFoundError(directory) from None
        try:
            directory_before = os.fstat(directory_descriptor)
            if (
                not stat.S_ISDIR(directory_before.st_mode)
                or directory_before.st_mode & 0o222
            ):
                raise FullGridError(
                    f"{directory} is not an immutable record directory"
                )
            directory_fingerprint = _sealed_tree_fingerprint(directory_before)
            if frozenset(os.listdir(directory_descriptor)) != FINAL_FILENAMES:
                raise FullGridError(f"{directory} has unexpected entries")
            descriptors: dict[str, int] = {}
            fingerprints: dict[str, tuple[int, ...]] = {}
            payloads: dict[str, bytes] = {}
            try:
                # Open every member before reading any member. Keeping all
                # three descriptors live makes this one coherent package
                # snapshot instead of three independently swappable reads.
                for name in sorted(FINAL_FILENAMES):
                    descriptor = os.open(
                        name,
                        os.O_RDONLY
                        | getattr(os, "O_CLOEXEC", 0)
                        | getattr(os, "O_NOFOLLOW", 0)
                        | getattr(os, "O_NONBLOCK", 0),
                        dir_fd=directory_descriptor,
                    )
                    descriptors[name] = descriptor
                    held = os.fstat(descriptor)
                    path_stat = os.stat(
                        name,
                        dir_fd=directory_descriptor,
                        follow_symlinks=False,
                    )
                    fingerprint = _sealed_tree_fingerprint(held)
                    if (
                        not stat.S_ISREG(held.st_mode)
                        or held.st_nlink != 1
                        or held.st_mode & 0o222
                        or _sealed_tree_fingerprint(path_stat) != fingerprint
                    ):
                        raise FullGridError(
                            f"{directory / name} is not a unique read-only "
                            "regular file"
                        )
                    fingerprints[name] = fingerprint

                if (
                    frozenset(os.listdir(directory_descriptor))
                    != FINAL_FILENAMES
                    or _sealed_tree_fingerprint(
                        os.fstat(directory_descriptor)
                    )
                    != directory_fingerprint
                ):
                    raise FullGridError(
                        f"{directory} changed while its package was opened"
                    )

                for name in sorted(FINAL_FILENAMES):
                    descriptor = descriptors[name]
                    chunks: list[bytes] = []
                    while True:
                        try:
                            chunk = os.read(descriptor, 1024 * 1024)
                        except InterruptedError:
                            continue
                        if not chunk:
                            break
                        chunks.append(chunk)
                    payload = b"".join(chunks)
                    if len(payload) != fingerprints[name][
                        _SEALED_TREE_STABLE_FIELDS.index("st_size")
                    ]:
                        raise FullGridError(
                            f"{directory / name} read was incomplete"
                        )
                    payloads[name] = payload

                # Rebind every still-open descriptor to its exact canonical
                # name only after all reads finish. This rejects both leaf
                # toggles and a coherent held-A/live-B package swap.
                for name in sorted(FINAL_FILENAMES):
                    held_after = os.fstat(descriptors[name])
                    path_after = os.stat(
                        name,
                        dir_fd=directory_descriptor,
                        follow_symlinks=False,
                    )
                    if (
                        _sealed_tree_fingerprint(held_after)
                        != fingerprints[name]
                        or _sealed_tree_fingerprint(path_after)
                        != fingerprints[name]
                    ):
                        raise FullGridError(
                            f"{directory / name} changed during package read"
                        )
                directory_after = os.fstat(directory_descriptor)
                if (
                    _sealed_tree_fingerprint(directory_after)
                    != directory_fingerprint
                    or frozenset(os.listdir(directory_descriptor))
                    != FINAL_FILENAMES
                ):
                    raise FullGridError(
                        f"{directory} changed while it was validated"
                    )
                _assert_directory_descriptor_path(root, root_descriptor)
                _assert_directory_descriptor_path(
                    directory,
                    directory_descriptor,
                )
            finally:
                for descriptor in descriptors.values():
                    os.close(descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        os.close(root_descriptor)

    completion = _strict_load_bytes(
        payloads["completion.json"],
        source=str(directory / "completion.json"),
    )
    _validate_completion_object(completion, plan=plan, job=job)
    if completion.get("files") != {
        "record.json": hashlib.sha256(payloads["record.json"]).hexdigest(),
        "predictions.npz": hashlib.sha256(
            payloads["predictions.npz"]
        ).hexdigest(),
    } or payloads["completion.json"] != _canonical_bytes(completion) + b"\n":
        raise FullGridError(f"{directory} completion checksums are invalid")

    record = _strict_load_bytes(
        payloads["record.json"],
        source=str(directory / "record.json"),
    )
    _validate_record_object(
        record,
        plan=plan,
        job=job,
        run_root=root,
        expected_preflight_report_sha256=(
            expected_preflight_report_sha256
        ),
    )
    if payloads["record.json"] != _canonical_bytes(record) + b"\n":
        raise FullGridError(f"{directory} record is noncanonical")
    from .data import _validate_npz_member_contract

    try:
        prediction_members = _validate_npz_member_contract(
            payloads["predictions.npz"],
            expected_members=("rows", "probabilities"),
            source=str(directory / "predictions.npz"),
        )
    except RuntimeError as error:
        raise FullGridError(str(error)) from error
    try:
        with np.load(
            io.BytesIO(payloads["predictions.npz"]),
            allow_pickle=False,
        ) as archive:
            if tuple(archive.files) != prediction_members:
                raise FullGridError("prediction archive has unexpected arrays")
            rows = archive["rows"].copy()
            probabilities = archive["probabilities"].copy()
    except (OSError, ValueError) as error:
        raise FullGridError(f"prediction archive cannot be loaded: {error}") from error
    n_classes = int(plan["datasets"][job.dataset]["n_classes"])
    expected_split = _planned_split_identity(plan, job)
    expected_test = expected_split["partitions"]["test"]
    if (
        rows.dtype != np.int64
        or rows.ndim != 1
        or len(rows) != int(record["test_count"])
        or len(rows) != int(expected_test["count"])
        or np.any(rows < 0)
        or np.any(rows >= int(expected_split["trial_count"]))
        or len(set(rows.tolist())) != len(rows)
        or _rows_sha256(rows) != record["test_rows_sha256"]
        or _rows_sha256(rows) != expected_test["rows_sha256"]
        or probabilities.dtype != np.float64
        or probabilities.shape != (len(rows), n_classes)
        or not np.all(np.isfinite(probabilities))
        or np.any(probabilities < 0.0)
        or np.any(probabilities > 1.0)
        or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    ):
        raise FullGridError(f"{directory} prediction arrays are invalid")
    return {
        "record": record,
        "completion": completion,
        "rows": rows,
        "probabilities": probabilities,
        "artifact_sha256": {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in sorted(payloads.items())
        },
    }


def validate_completion(
    run_root: Path,
    plan: Mapping[str, Any],
    job: Job,
) -> dict[str, Any]:
    """Validate one completion against the currently loaded attestation."""

    return _validate_completion_bound(
        run_root,
        plan,
        job,
        expected_preflight_report_sha256=None,
    )


def _state_sha256(model: Any) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _make_model(
    plan: Mapping[str, Any],
    job: Job,
    *,
    n_channels: int,
    n_outputs: int,
    n_times: int,
    sfreq: float,
    channel_names: tuple[str, ...],
    positions: np.ndarray,
) -> Any:
    import torch

    if job.model not in COMMON_ARCHITECTURES:
        raise FullGridError(f"model is outside the exact common roster: {job.model}")

    from . import benchmark
    from .baselines import make_model
    from .models import CardinalFieldConfig, ScopeConfig

    model_name, variant = benchmark._model_definition(job.model)
    scope_config = (
        benchmark._scope_config(variant)
        if model_name in {"scope", "free_scope"}
        else ScopeConfig()
    )
    cardinal_config = (
        benchmark._cardinal_config(variant)
        if model_name in {"cardinal", "free_cardinal"}
        else CardinalFieldConfig()
    )
    return make_model(
        model_name,
        n_channels=n_channels,
        n_outputs=n_outputs,
        n_times=n_times,
        sfreq=sfreq,
        scope_config=scope_config,
        cardinal_config=cardinal_config,
        channel_names=channel_names,
        channel_positions=torch.as_tensor(positions, dtype=torch.float32),
    )


def execute_benchmark_job(
    *,
    job: Job,
    plan: Mapping[str, Any],
    cache_root: Path,
    device: str,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Execute reset/refit training and return score-blind prediction material."""

    import torch

    from . import benchmark
    from .data import (
        apply_channel_scaler,
        fit_channel_scaler,
        reflection_index,
        split_indices,
    )
    from .models import parameter_count
    from .training import (
        TrainConfig,
        configure_determinism,
        fit_model,
        predict_probabilities,
        refit_model,
    )

    job_started = time.perf_counter()
    data = _load_subject_cache_safely(
        job.dataset,
        job.subject,
        cache_root=cache_root,
    )
    planned_cache = _planned_cache_identity(plan, job)
    if data["identity"] != planned_cache:
        raise FullGridError(
            f"loaded cache identity differs from the frozen plan for "
            f"{job.dataset} S{job.subject}"
        )
    train_rows, validation_rows, test_rows = split_indices(
        job.dataset,
        data["y"],
        data["sessions"],
        data["runs"],
        fold=job.fold,
        subject=job.subject,
    )
    observed_split = _one_split_identity(
        dataset=job.dataset,
        subject=job.subject,
        fold=job.fold,
        cache_array_sha256=str(data["identity"]["array_sha256"]),
        trial_count=len(data["y"]),
        train_rows=train_rows,
        validation_rows=validation_rows,
        test_rows=test_rows,
    )
    if observed_split != _planned_split_identity(plan, job):
        raise FullGridError(
            f"runtime split differs from the frozen plan for {job.identity()}"
        )
    channel_names = tuple(str(value) for value in data["channel_names"].tolist())
    selection_mean, selection_std = fit_channel_scaler(
        data["x"][train_rows], channel_names
    )
    x_train = apply_channel_scaler(
        data["x"][train_rows], selection_mean, selection_std
    )
    x_validation = apply_channel_scaler(
        data["x"][validation_rows], selection_mean, selection_std
    )
    source_rows = np.sort(np.concatenate((train_rows, validation_rows)))
    refit_mean, refit_std = fit_channel_scaler(
        data["x"][source_rows], channel_names
    )
    x_source = apply_channel_scaler(data["x"][source_rows], refit_mean, refit_std)
    x_test = apply_channel_scaler(data["x"][test_rows], refit_mean, refit_std)

    config = TrainConfig(**dict(plan["train_config"]))
    config = replace(config, seed=job.seed, device=device)
    configure_determinism(job.seed)
    model = _make_model(
        plan,
        job,
        n_channels=x_train.shape[1],
        n_outputs=int(plan["datasets"][job.dataset]["n_classes"]),
        n_times=x_train.shape[2],
        sfreq=float(plan["datasets"][job.dataset]["preprocessing"]["sfreq_hz"]),
        channel_names=channel_names,
        positions=data["positions"],
    )
    target_device = torch.device(device)
    if target_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target_device)
    initial_state = copy.deepcopy(model.state_dict())
    initial_hash = _state_sha256(model)
    mirror = reflection_index(channel_names)
    effective_config = benchmark._train_config_for_model(config, job.model)
    fit = fit_model(
        model,
        x_train,
        data["y"][train_rows],
        x_validation,
        data["y"][validation_rows],
        data["positions"],
        config=effective_config,
        mirror_index=mirror,
    )
    selection_hash = _state_sha256(fit["model"])
    fit["model"].load_state_dict(initial_state)
    reset_hash = _state_sha256(fit["model"])
    if reset_hash != initial_hash:
        raise FullGridError("model did not reset to its seeded initialization")
    selected_epochs = int(fit["best_epoch"]) + 1
    refit = refit_model(
        fit["model"],
        x_source,
        data["y"][source_rows],
        data["positions"],
        epochs=selected_epochs,
        config=effective_config,
        mirror_index=mirror,
    )
    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)
    inference_started = time.perf_counter()
    probabilities = predict_probabilities(
        refit["model"],
        x_test,
        data["positions"],
        device=device,
    )
    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)
    inference_seconds = time.perf_counter() - inference_started
    peak_memory_bytes = (
        int(torch.cuda.max_memory_allocated(target_device))
        if target_device.type == "cuda"
        else 0
    )
    metadata = {
        "cache_array_sha256": str(data["identity"]["array_sha256"]),
        "channels": list(channel_names),
        "split": _split_metadata(observed_split),
        "scalers": {
            "selection_mean_sha256": _array_sha256(selection_mean),
            "selection_std_sha256": _array_sha256(selection_std),
            "refit_mean_sha256": _array_sha256(refit_mean),
            "refit_std_sha256": _array_sha256(refit_std),
        },
        "fit": {
            "seed_installed_before_construction": True,
            "initial_state_sha256": initial_hash,
            "selection_state_sha256": selection_hash,
            "reset_state_sha256": reset_hash,
            "reset_verified": True,
            "source_selected_epoch": int(fit["best_epoch"]),
            "selection_epochs_run": int(fit["epochs_run"]),
            "refit_epochs_run": int(refit["epochs_run"]),
            "refit_state_sha256": _state_sha256(refit["model"]),
            "parameter_count": parameter_count(refit["model"]),
            "architecture": benchmark._architecture_identity(
                refit["model"], job.model
            ),
        },
        "protocol": {
            "selection_scaler_rows": "train_only",
            "selection_optimization_rows": "train_only",
            "epoch_selection_rows": "validation_only",
            "refit_scaler_rows": "train_plus_validation",
            "refit_optimization_rows": "train_plus_validation",
            "test_use": "prediction_only",
            "test_performance_computed": False,
        },
        "timing_seconds": {
            "selection_fit": float(fit["fit_seconds"]),
            "refit_fit": float(refit["fit_seconds"]),
            "test_inference": inference_seconds,
            "job_total": time.perf_counter() - job_started,
        },
        "cuda_peak_memory_bytes": peak_memory_bytes,
        "runtime": {
            "worker_cpu_threads": int(torch.get_num_threads()),
            "worker_interop_threads": int(torch.get_num_interop_threads()),
            "thread_environment": {
                name: os.environ.get(name)
                for name in THREAD_ENVIRONMENT_VARIABLES
            },
            "nvidia_driver_versions": _nvidia_driver_versions(),
            "physical_gpu_uuid": os.environ.get("FULL_GRID_GPU_UUID"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }
    del fit, refit, model, initial_state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metadata, np.asarray(test_rows, dtype=np.int64), probabilities


def _load_preflight_cache_metadata(
    *,
    cache_root: Path,
    dataset: str,
    subject: int,
) -> dict[str, Any]:
    """Read only non-trial NPZ members needed for the model shape gate."""

    from .config import (
        DEFAULT_MONTAGE_PROFILE,
        channels_for_dataset,
        coordinate_contract_for_dataset,
    )
    from .data import (
        SUBJECT_CACHE_NPZ_MEMBERS,
        _atlas_unit_positions,
        _read_unique_regular_bytes,
        _validate_npz_member_contract,
    )

    path = _safe_cache_path(cache_root, dataset, subject)
    try:
        payload = _read_unique_regular_bytes(path)
    except RuntimeError as error:
        raise FullGridError(str(error)) from error
    try:
        expected_members = _validate_npz_member_contract(
            payload,
            expected_members=SUBJECT_CACHE_NPZ_MEMBERS,
            source=str(path),
        )
    except RuntimeError as error:
        raise FullGridError(str(error)) from error
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        if tuple(archive.files) != expected_members:
            raise FullGridError(
                f"preflight cache {path} fields differ from the frozen schema"
            )
        # Deliberately do not index x, y, sessions, or runs. NumPy decompresses
        # an NPZ member only when it is indexed.
        identity = json.loads(
            str(archive["identity"].item()),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
        positions = np.asarray(
            archive["positions"],
            dtype=np.float32,
        ).copy()
        channel_names = tuple(
            str(value) for value in archive["channel_names"].tolist()
        )

    if not isinstance(identity, Mapping):
        raise FullGridError(
            f"preflight cache identity is not an object for {dataset} S{subject}"
        )
    if type(identity.get("subject")) is not int:
        raise FullGridError(
            f"preflight cache subject identity is invalid for {dataset} S{subject}"
        )
    expected_preprocessing = preprocessing_for_dataset(dataset)
    expected_channels = channels_for_dataset(
        dataset,
        DEFAULT_MONTAGE_PROFILE,
    )
    shape = identity.get("shape")
    try:
        observed_subject = int(identity.get("subject", -1))
    except (TypeError, ValueError) as error:
        raise FullGridError(
            f"preflight cache subject identity is invalid for "
            f"{dataset} S{subject}"
        ) from error
    expected_dataset_identity = json.loads(
        _canonical_bytes(asdict(dataset_spec(dataset)))
    )
    _validate_cache_identity_exact(
        identity,
        dataset=dataset,
        subject=subject,
        channel_names=channel_names,
    )
    if (
        identity.get("dataset") != expected_dataset_identity
        or observed_subject != subject
        or identity.get("montage_profile") != DEFAULT_MONTAGE_PROFILE
        or identity.get("preprocessing") != expected_preprocessing
        or not isinstance(shape, list)
        or len(shape) != 3
        or any(not isinstance(value, int) or value <= 0 for value in shape)
        or shape[1:] != [len(channel_names), int(expected_preprocessing["n_times"])]
        or identity.get("channels") != list(channel_names)
        or (
            expected_channels is not None
            and tuple(expected_channels) != channel_names
        )
        or identity.get("coordinates")
        != coordinate_contract_for_dataset(
            dataset,
            DEFAULT_MONTAGE_PROFILE,
            channel_names,
        )
        or not isinstance(identity.get("array_sha256"), str)
        or not HEX_64_RE.fullmatch(str(identity.get("array_sha256")))
    ):
        raise FullGridError(
            f"preflight cache metadata identity is invalid for "
            f"{dataset} S{subject}"
        )
    expected_positions = _atlas_unit_positions(channel_names)
    if (
        not channel_names
        or len(set(channel_names)) != len(channel_names)
        or positions.shape != (len(channel_names), 3)
        or not np.all(np.isfinite(positions))
        or not np.allclose(
            positions,
            expected_positions,
            rtol=0.0,
            atol=1e-7,
        )
        or not np.allclose(
            np.linalg.norm(positions, axis=1),
            1.0,
            rtol=0.0,
            atol=1e-6,
        )
    ):
        raise FullGridError(
            f"preflight cache montage metadata is invalid for "
            f"{dataset} S{subject}"
        )
    return {
        "path": str(_absolute_path(path)),
        "identity_shape": list(shape),
        "declared_cache_array_sha256": str(identity["array_sha256"]),
        "channel_names": channel_names,
        "positions": positions,
        "members_opened": ["identity", "positions", "channel_names"],
    }


def _synthetic_preflight_input(
    contract: Mapping[str, Any],
    *,
    batch_size: int,
    seed: int,
) -> np.ndarray:
    """Create deterministic channel-standardized input without opening EEG."""

    generator = np.random.default_rng(seed)
    values = generator.standard_normal(
        (
            batch_size,
            int(contract["n_channels"]),
            int(contract["n_times"]),
        ),
        dtype=np.float32,
    )
    mean = values.mean(axis=(0, 2), keepdims=True, dtype=np.float64)
    std = values.std(axis=(0, 2), keepdims=True, dtype=np.float64)
    standardized = (
        (values.astype(np.float64) - mean) / np.maximum(std, 1e-6)
    ).astype(np.float32)
    if (
        not np.all(np.isfinite(standardized))
        or not np.allclose(
            standardized.mean(axis=(0, 2)),
            0.0,
            rtol=0.0,
            atol=1e-5,
        )
        or not np.allclose(
            standardized.std(axis=(0, 2)),
            1.0,
            rtol=0.0,
            atol=1e-5,
        )
    ):
        raise FullGridError("synthetic preflight standardization failed")
    return np.ascontiguousarray(standardized)


def _preflight_runtime_identity(
    plan: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, Any]]:
    """Verify score-blind source/environment identity without opening trials."""

    validate_plan_semantics(plan)
    if plan["publication_mode"] != FORMAL_PUBLICATION_MODE:
        raise FullGridError("publishable CUDA preflight requires a formal plan")
    source_identity = _source_identity(executor=DEFAULT_EXECUTOR)
    if source_identity != plan["source_identity"]:
        raise FullGridError("preflight source identity differs from the plan")
    if _current_tcformer_source_identity() != plan["tcformer_source_identity"]:
        raise FullGridError("preflight TCFormer source differs from the plan")
    environment = _environment_identity()
    if _normalized_worker_environment_identity(
        environment
    ) != _normalized_worker_environment_identity(plan["environment_identity"]):
        raise FullGridError("preflight environment identity differs from the plan")
    return source_identity, environment


def _validate_preflight_report(
    report: Mapping[str, Any],
    *,
    plan: Mapping[str, Any] | None,
    run_root: Path | None,
) -> None:
    expected_keys = {
        "schema",
        "created_at",
        "score_blind",
        "input_source",
        "cache_members_opened",
        "trial_feature_members_opened",
        "label_or_split_members_opened",
        "test_splits_opened",
        "accuracy_computed",
        "architectures",
        "track_scope",
        "contracts",
        "cache_metadata",
        "expected_checks",
        "completed_checks",
        "source_identity",
        "environment_identity_sha256",
        "execution_backend",
        "cuda_device_name",
        "plan_sha256",
        "physical_gpu_uuid",
        "gpu_lease_receipt",
        "results",
    }
    _exact_keys(report, expected_keys, path="preflight_report")
    try:
        created = datetime.fromisoformat(str(report["created_at"]))
    except (TypeError, ValueError) as error:
        raise FullGridError("preflight timestamp is invalid") from error
    expected_checks = len(COMMON_ARCHITECTURES) * len(MODEL_PREFLIGHT_CONTRACTS)
    if (
        created.tzinfo is None
        or report["schema"] != PREFLIGHT_SCHEMA
        or report["score_blind"] is not True
        or report["input_source"]
        != "deterministic_synthetic_standardized_tensors"
        or report["cache_members_opened"]
        != ["identity", "positions", "channel_names"]
        or report["trial_feature_members_opened"] is not False
        or report["label_or_split_members_opened"] is not False
        or report["test_splits_opened"] is not False
        or report["accuracy_computed"] is not False
        or report["architectures"] != list(COMMON_ARCHITECTURES)
        or report["track_scope"] != TRACK_SCOPE
        or report["contracts"]
        != [copy.deepcopy(value) for value in MODEL_PREFLIGHT_CONTRACTS]
        or type(report["expected_checks"]) is not int
        or report["expected_checks"] != expected_checks
        or type(report["completed_checks"]) is not int
        or report["completed_checks"] != expected_checks
        or not isinstance(report["cache_metadata"], list)
        or len(report["cache_metadata"]) != len(MODEL_PREFLIGHT_CONTRACTS)
        or not isinstance(report["source_identity"], Mapping)
        or any(
            not isinstance(value, str) or HEX_64_RE.fullmatch(value) is None
            for value in report["source_identity"].values()
        )
        or not isinstance(report["environment_identity_sha256"], str)
        or HEX_64_RE.fullmatch(report["environment_identity_sha256"]) is None
        or not isinstance(report["results"], list)
        or len(report["results"]) != expected_checks
    ):
        raise FullGridError("preflight report violates its frozen contract")
    for index, metadata in enumerate(report["cache_metadata"]):
        _exact_keys(
            metadata,
            {
                "path",
                "identity_shape",
                "declared_cache_array_sha256",
                "members_opened",
            },
            path=f"preflight_report.cache_metadata[{index}]",
        )
        if (
            not isinstance(metadata["path"], str)
            or not os.path.isabs(metadata["path"])
            or not isinstance(metadata["identity_shape"], list)
            or len(metadata["identity_shape"]) != 3
            or any(type(value) is not int or value <= 0 for value in metadata["identity_shape"])
            or not isinstance(metadata["declared_cache_array_sha256"], str)
            or HEX_64_RE.fullmatch(
                metadata["declared_cache_array_sha256"]
            ) is None
            or metadata["members_opened"]
            != ["identity", "positions", "channel_names"]
        ):
            raise FullGridError("preflight cache metadata is invalid")
    result_index = 0
    for model_name in COMMON_ARCHITECTURES:
        for contract in MODEL_PREFLIGHT_CONTRACTS:
            result = report["results"][result_index]
            result_index += 1
            _exact_keys(
                result,
                {
                    "model",
                    "contract",
                    "dataset",
                    "n_channels",
                    "n_times",
                    "n_outputs",
                    "parameter_count",
                    "forward_backward_optimizer_step",
                },
                path=f"preflight_report.results[{result_index - 1}]",
            )
            if (
                result["model"] != model_name
                or result["contract"] != contract["name"]
                or result["dataset"] != contract["dataset"]
                or type(result["n_channels"]) is not int
                or result["n_channels"] != contract["n_channels"]
                or type(result["n_times"]) is not int
                or result["n_times"] != contract["n_times"]
                or type(result["n_outputs"]) is not int
                or result["n_outputs"] != contract["n_outputs"]
                or type(result["parameter_count"]) is not int
                or result["parameter_count"] <= 0
            ):
                raise FullGridError("preflight result identity is invalid")
    if plan is None:
        if (
            run_root is not None
            or report["plan_sha256"] is not None
            or report["physical_gpu_uuid"] is not None
            or report["gpu_lease_receipt"] is not None
            or report["execution_backend"] != "injected_mock_cpu_test"
            or report["cuda_device_name"] is not None
            or any(
                row["forward_backward_optimizer_step"] != "mocked_cpu_test"
                for row in report["results"]
            )
        ):
            raise FullGridError("injected preflight report claims formal authority")
        return
    if run_root is None:
        raise FullGridError("formal preflight report requires run_root")
    for metadata, contract in zip(
        report["cache_metadata"],
        MODEL_PREFLIGHT_CONTRACTS,
        strict=True,
    ):
        planned_cache = plan["cache_identity"].get(
            _subject_identity_key(
                str(contract["dataset"]),
                int(contract["subject"]),
            )
        )
        if (
            not isinstance(planned_cache, Mapping)
            or metadata["declared_cache_array_sha256"]
            != planned_cache.get("array_sha256")
        ):
            raise FullGridError(
                "formal preflight cache metadata differs from the plan"
            )
    gpu_uuid = report["physical_gpu_uuid"]
    inventory = plan["environment_identity"].get("nvidia_gpu_inventory")
    planned_gpu = next(
        (
            row
            for row in inventory
            if isinstance(row, Mapping) and row.get("uuid") == gpu_uuid
        ),
        None,
    ) if isinstance(inventory, list) else None
    if (
        report["plan_sha256"] != plan["plan_sha256"]
        or report["source_identity"] != plan["source_identity"]
        or report["environment_identity_sha256"]
        != plan["analysis_contract"]["environment_identity_sha256"]
        or report["execution_backend"] != "cuda_forward_backward_adamw"
        or not isinstance(report["cuda_device_name"], str)
        or not report["cuda_device_name"]
        or not isinstance(planned_gpu, Mapping)
        or report["cuda_device_name"] != planned_gpu.get("name")
        or not isinstance(gpu_uuid, str)
        or gpu_uuid not in plan["execution_config"]["physical_gpu_uuid_roster"]
        or any(
            row["forward_backward_optimizer_step"] != "passed"
            for row in report["results"]
        )
    ):
        raise FullGridError("formal preflight report differs from its plan")
    _validate_persisted_gpu_lease_receipt(
        report["gpu_lease_receipt"],
        plan=plan,
        run_root=run_root,
        expected_gpu_uuid=gpu_uuid,
    )


def _assert_preflight_directory_descriptor_path(
    root: Path,
    preflight_descriptor: int,
) -> None:
    try:
        _assert_directory_descriptor_path(
            root / PREFLIGHT_DIRECTORY,
            preflight_descriptor,
        )
    except (OSError, FullGridError) as error:
        raise FullGridError(
            "preflight directory path changed during transaction"
        ) from error


def _preflight_authoritative_names(
    run_root: Path,
    *,
    allow_stages: bool,
) -> tuple[Path, set[str]]:
    root = _safe_run_root(run_root)
    root_descriptor = _open_absolute_directory(root)
    try:
        try:
            preflight_descriptor = _open_relative_directory(
                root_descriptor,
                PREFLIGHT_DIRECTORY,
                create=False,
            )
        except FileNotFoundError:
            return root, set()
        try:
            names = set(os.listdir(preflight_descriptor))
            authoritative = {
                PREFLIGHT_REPORT_FILENAME,
                PREFLIGHT_RECEIPT_FILENAME,
            }
            extras = names - authoritative
            if allow_stages:
                extras = {
                    name
                    for name in extras
                    if PREFLIGHT_STAGE_RE.fullmatch(name) is None
                }
            if extras:
                raise FullGridError(
                    f"preflight directory has unexpected entries: {sorted(extras)}"
                )
            _assert_directory_descriptor_path(root, root_descriptor)
            _assert_preflight_directory_descriptor_path(
                root,
                preflight_descriptor,
            )
            return root, names & authoritative
        finally:
            os.close(preflight_descriptor)
    finally:
        os.close(root_descriptor)


def _preflight_canonical_snapshot(
    run_root: Path,
) -> dict[str, Any]:
    """Bind the current canonical preflight leaves to one held directory."""

    root = _safe_run_root(run_root)
    root_descriptor = _open_absolute_directory(root)
    try:
        try:
            preflight_descriptor = _open_relative_directory(
                root_descriptor,
                PREFLIGHT_DIRECTORY,
                create=False,
            )
        except FileNotFoundError:
            return {"directory_identity": None, "artifacts": {}}
        try:
            directory_before = os.fstat(preflight_descriptor)
            directory_fingerprint = _sealed_tree_fingerprint(directory_before)
            names = set(os.listdir(preflight_descriptor))
            authoritative = {
                PREFLIGHT_REPORT_FILENAME,
                PREFLIGHT_RECEIPT_FILENAME,
            }
            unknown = {
                name
                for name in names - authoritative
                if PREFLIGHT_STAGE_RE.fullmatch(name) is None
            }
            if unknown:
                raise FullGridError(
                    "preflight directory has unexpected entries: "
                    f"{sorted(unknown)}"
                )
            artifacts: dict[str, dict[str, Any]] = {}
            for name in sorted(names & authoritative):
                payload = _read_unique_regular_at(
                    preflight_descriptor,
                    name,
                )
                observed = os.stat(
                    name,
                    dir_fd=preflight_descriptor,
                    follow_symlinks=False,
                )
                if (
                    not stat.S_ISREG(observed.st_mode)
                    or observed.st_nlink != 1
                    or observed.st_mode & 0o222
                ):
                    raise FullGridError(
                        f"preflight {name} is not a unique read-only file"
                    )
                artifacts[name] = {
                    "identity": (
                        int(observed.st_dev),
                        int(observed.st_ino),
                    ),
                    "payload": payload,
                }
            if (
                _sealed_tree_fingerprint(os.fstat(preflight_descriptor))
                != directory_fingerprint
                or set(os.listdir(preflight_descriptor)) != names
            ):
                raise FullGridError(
                    "preflight directory changed during canonical snapshot"
                )
            _assert_directory_descriptor_path(root, root_descriptor)
            _assert_preflight_directory_descriptor_path(
                root,
                preflight_descriptor,
            )
            return {
                "directory_identity": (
                    int(directory_before.st_dev),
                    int(directory_before.st_ino),
                ),
                "artifacts": artifacts,
            }
        finally:
            os.close(preflight_descriptor)
    finally:
        os.close(root_descriptor)


def _preflight_stage_names(run_root: Path) -> tuple[str, ...]:
    """Return only schema-recognized stages; reject every unknown entry."""

    root = _safe_run_root(run_root)
    root_descriptor = _open_absolute_directory(root)
    try:
        try:
            preflight_descriptor = _open_relative_directory(
                root_descriptor,
                PREFLIGHT_DIRECTORY,
                create=False,
            )
        except FileNotFoundError:
            return ()
        try:
            names = set(os.listdir(preflight_descriptor))
            authoritative = {
                PREFLIGHT_REPORT_FILENAME,
                PREFLIGHT_RECEIPT_FILENAME,
            }
            extras = names - authoritative
            unknown = {
                name
                for name in extras
                if PREFLIGHT_STAGE_RE.fullmatch(name) is None
            }
            if unknown:
                raise FullGridError(
                    f"preflight directory has unexpected entries: "
                    f"{sorted(unknown)}"
                )
            _assert_directory_descriptor_path(root, root_descriptor)
            _assert_preflight_directory_descriptor_path(
                root,
                preflight_descriptor,
            )
            return tuple(sorted(extras))
        finally:
            os.close(preflight_descriptor)
    finally:
        os.close(root_descriptor)


def _unlink_exact_preflight_stage_at(
    preflight_descriptor: int,
    name: str,
) -> None:
    """Unlink only one exact, descriptor-bound preflight stage leaf."""

    if PREFLIGHT_STAGE_RE.fullmatch(name) is None:
        raise FullGridError(f"unknown preflight stage name: {name}")
    descriptor = os.open(
        name,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0),
        dir_fd=preflight_descriptor,
    )
    try:
        before = os.fstat(descriptor)
        path_stat = os.stat(
            name,
            dir_fd=preflight_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or (before.st_dev, before.st_ino)
            != (path_stat.st_dev, path_stat.st_ino)
        ):
            raise FullGridError(
                f"preflight stage is not a unique regular file: {name}"
            )
        os.unlink(name, dir_fd=preflight_descriptor)
        after = os.fstat(descriptor)
        if (
            (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino)
            or after.st_nlink != 0
        ):
            raise FullGridError(
                f"preflight stage unlink was not exact: {name}"
            )
    finally:
        os.close(descriptor)


def _cleanup_preflight_stages(run_root: Path) -> tuple[str, ...]:
    """Delete only exact stale stage leaves below held run/preflight FDs."""

    root = _safe_run_root(run_root)
    root_descriptor = _open_absolute_directory(root)
    removed: list[str] = []
    try:
        try:
            preflight_descriptor = _open_relative_directory(
                root_descriptor,
                PREFLIGHT_DIRECTORY,
                create=False,
            )
        except FileNotFoundError:
            return ()
        try:
            names = set(os.listdir(preflight_descriptor))
            authoritative = {
                PREFLIGHT_REPORT_FILENAME,
                PREFLIGHT_RECEIPT_FILENAME,
            }
            extras = names - authoritative
            unknown = {
                name
                for name in extras
                if PREFLIGHT_STAGE_RE.fullmatch(name) is None
            }
            if unknown:
                raise FullGridError(
                    "refusing stage recovery with unknown preflight entries: "
                    f"{sorted(unknown)}"
                )
            for name in sorted(extras):
                _unlink_exact_preflight_stage_at(
                    preflight_descriptor,
                    name,
                )
                removed.append(name)
            if removed:
                os.fsync(preflight_descriptor)
            _assert_directory_descriptor_path(root, root_descriptor)
            _assert_preflight_directory_descriptor_path(
                root,
                preflight_descriptor,
            )
            return tuple(removed)
        finally:
            os.close(preflight_descriptor)
    finally:
        os.close(root_descriptor)


def _read_preflight_report(
    run_root: Path,
    plan: Mapping[str, Any],
    *,
    require_exact_directory: bool,
) -> tuple[dict[str, Any], bytes]:
    root = _safe_run_root(run_root)
    root_descriptor = _open_absolute_directory(root)
    try:
        preflight_descriptor = _open_relative_directory(
            root_descriptor,
            PREFLIGHT_DIRECTORY,
            create=False,
        )
        try:
            names = set(os.listdir(preflight_descriptor))
            required = {PREFLIGHT_REPORT_FILENAME}
            allowed = required | {PREFLIGHT_RECEIPT_FILENAME}
            if (
                not required.issubset(names)
                or (
                    require_exact_directory
                    and names != allowed
                )
                or (
                    not require_exact_directory
                    and names - allowed
                    and any(
                        PREFLIGHT_STAGE_RE.fullmatch(name) is None
                        for name in names - allowed
                    )
                )
            ):
                raise FullGridError("preflight report directory is not exact")
            observed = os.stat(
                PREFLIGHT_REPORT_FILENAME,
                dir_fd=preflight_descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISREG(observed.st_mode)
                or observed.st_nlink != 1
                or observed.st_mode & 0o222
            ):
                raise FullGridError(
                    "preflight report is not a unique read-only regular file"
                )
            payload = _read_unique_regular_at(
                preflight_descriptor,
                PREFLIGHT_REPORT_FILENAME,
            )
            report = _strict_load_bytes(
                payload,
                source=str(root / PREFLIGHT_DIRECTORY / PREFLIGHT_REPORT_FILENAME),
            )
            if payload != _canonical_bytes(report) + b"\n":
                raise FullGridError("preflight report is not canonical JSON")
            _validate_preflight_report(report, plan=plan, run_root=root)
            _assert_directory_descriptor_path(root, root_descriptor)
            _assert_preflight_directory_descriptor_path(
                root,
                preflight_descriptor,
            )
            return report, payload
        finally:
            os.close(preflight_descriptor)
    finally:
        os.close(root_descriptor)


def _validate_preflight_receipt(
    receipt: Mapping[str, Any],
    *,
    report: Mapping[str, Any],
    report_payload: bytes,
    plan: Mapping[str, Any],
    run_root: Path,
) -> None:
    _exact_keys(
        receipt,
        {
            "schema",
            "created_at",
            "plan_sha256",
            "report_schema",
            "report_sha256",
            "expected_checks",
            "completed_checks",
            "publication_gpu_lease_receipt",
        },
        path="preflight_receipt",
    )
    try:
        created = datetime.fromisoformat(str(receipt["created_at"]))
        gpu_uuid = receipt["publication_gpu_lease_receipt"]["lease"]["gpu_uuid"]
    except (KeyError, TypeError, ValueError) as error:
        raise FullGridError("preflight receipt identity is invalid") from error
    if (
        created.tzinfo is None
        or receipt["schema"] != PREFLIGHT_RECEIPT_SCHEMA
        or receipt["plan_sha256"] != plan["plan_sha256"]
        or receipt["report_schema"] != PREFLIGHT_SCHEMA
        or receipt["report_sha256"] != hashlib.sha256(report_payload).hexdigest()
        or receipt["expected_checks"] != report["expected_checks"]
        or receipt["completed_checks"] != report["completed_checks"]
        or type(receipt["expected_checks"]) is not int
        or type(receipt["completed_checks"]) is not int
        or gpu_uuid not in plan["execution_config"]["physical_gpu_uuid_roster"]
    ):
        raise FullGridError("preflight receipt differs from its report or plan")
    _validate_persisted_gpu_lease_receipt(
        receipt["publication_gpu_lease_receipt"],
        plan=plan,
        run_root=run_root,
        expected_gpu_uuid=gpu_uuid,
    )


def load_preflight_attestation(
    run_root: Path,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Load the mandatory immutable formal preflight report/digest receipt."""

    if plan["publication_mode"] != FORMAL_PUBLICATION_MODE:
        raise FullGridError("test-only plans have no formal preflight attestation")
    root = _safe_run_root(run_root)
    root_descriptor = _open_absolute_directory(root)
    try:
        try:
            preflight_descriptor = _open_relative_directory(
                root_descriptor,
                PREFLIGHT_DIRECTORY,
                create=False,
            )
        except FileNotFoundError as error:
            raise FullGridError(
                "formal execution requires the exact completed "
                "preflight attestation"
            ) from error
        try:
            if set(os.listdir(preflight_descriptor)) != {
                PREFLIGHT_REPORT_FILENAME,
                PREFLIGHT_RECEIPT_FILENAME,
            }:
                raise FullGridError(
                    "formal execution requires the exact completed "
                    "preflight attestation"
                )

            def read_immutable(leaf_name: str) -> bytes:
                observed = os.stat(
                    leaf_name,
                    dir_fd=preflight_descriptor,
                    follow_symlinks=False,
                )
                if (
                    not stat.S_ISREG(observed.st_mode)
                    or observed.st_nlink != 1
                    or observed.st_mode & 0o222
                ):
                    raise FullGridError(
                        f"preflight {leaf_name} is not a unique read-only "
                        "regular file"
                    )
                return _read_unique_regular_at(
                    preflight_descriptor,
                    leaf_name,
                )

            report_payload = read_immutable(PREFLIGHT_REPORT_FILENAME)
            report = _strict_load_bytes(
                report_payload,
                source=str(
                    root
                    / PREFLIGHT_DIRECTORY
                    / PREFLIGHT_REPORT_FILENAME
                ),
            )
            if report_payload != _canonical_bytes(report) + b"\n":
                raise FullGridError("preflight report is not canonical JSON")
            _validate_preflight_report(report, plan=plan, run_root=root)

            receipt_payload = read_immutable(PREFLIGHT_RECEIPT_FILENAME)
            receipt = _strict_load_bytes(
                receipt_payload,
                source=str(
                    root
                    / PREFLIGHT_DIRECTORY
                    / PREFLIGHT_RECEIPT_FILENAME
                ),
            )
            if receipt_payload != _canonical_bytes(receipt) + b"\n":
                raise FullGridError("preflight receipt is not canonical JSON")
            _validate_preflight_receipt(
                receipt,
                report=report,
                report_payload=report_payload,
                plan=plan,
                run_root=root,
            )
            _assert_directory_descriptor_path(root, root_descriptor)
            _assert_preflight_directory_descriptor_path(
                root,
                preflight_descriptor,
            )
            return {
                "report": report,
                "receipt": receipt,
                "report_sha256": hashlib.sha256(report_payload).hexdigest(),
            }
        finally:
            os.close(preflight_descriptor)
    finally:
        os.close(root_descriptor)


def _publish_preflight_attestation(
    *,
    run_root: Path,
    plan: Mapping[str, Any],
    report: Mapping[str, Any],
    publication_gpu_lease_receipt: Mapping[str, Any],
    _publication_tracker: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Publish/repair the fixed report-first attestation under held authority."""

    root = _safe_run_root(run_root)
    _validate_preflight_report(report, plan=plan, run_root=root)
    report_payload = _canonical_bytes(report) + b"\n"
    receipt = {
        "schema": PREFLIGHT_RECEIPT_SCHEMA,
        "created_at": _utc_now(),
        "plan_sha256": plan["plan_sha256"],
        "report_schema": PREFLIGHT_SCHEMA,
        "report_sha256": hashlib.sha256(report_payload).hexdigest(),
        "expected_checks": report["expected_checks"],
        "completed_checks": report["completed_checks"],
        "publication_gpu_lease_receipt": copy.deepcopy(
            publication_gpu_lease_receipt
        ),
    }
    _validate_preflight_receipt(
        receipt,
        report=report,
        report_payload=report_payload,
        plan=plan,
        run_root=root,
    )
    root_descriptor = _open_absolute_directory(root)
    try:
        preflight_descriptor = _open_relative_directory(
            root_descriptor,
            PREFLIGHT_DIRECTORY,
            create=True,
        )
        try:
            if _publication_tracker is not None:
                preflight_stat = os.fstat(preflight_descriptor)
                _publication_tracker["directory_identity"] = (
                    int(preflight_stat.st_dev),
                    int(preflight_stat.st_ino),
                )
            names = set(os.listdir(preflight_descriptor))
            authoritative = {
                PREFLIGHT_REPORT_FILENAME,
                PREFLIGHT_RECEIPT_FILENAME,
            }
            extras = names - authoritative
            unknown = {
                name
                for name in extras
                if PREFLIGHT_STAGE_RE.fullmatch(name) is None
            }
            if unknown:
                raise FullGridError(
                    "immutable preflight publication found unknown entries: "
                    f"{sorted(unknown)}"
                )
            for stage_name in sorted(extras):
                _unlink_exact_preflight_stage_at(
                    preflight_descriptor,
                    stage_name,
                )
            if extras:
                os.fsync(preflight_descriptor)
            names -= extras
            if PREFLIGHT_REPORT_FILENAME in names:
                observed_report = _read_unique_regular_at(
                    preflight_descriptor,
                    PREFLIGHT_REPORT_FILENAME,
                )
                if observed_report != report_payload:
                    raise FullGridError(
                        "immutable preflight report already differs"
                    )
            else:
                report_keywords: dict[str, Any] = {}
                if _publication_tracker is not None:
                    def remember_report(published: os.stat_result) -> None:
                        _publication_tracker["artifacts"][
                            PREFLIGHT_REPORT_FILENAME
                        ] = {
                            "identity": (
                                int(published.st_dev),
                                int(published.st_ino),
                            ),
                            "payload": report_payload,
                        }

                    report_keywords["on_publish"] = remember_report
                _write_bytes_exclusive_at(
                    preflight_descriptor,
                    PREFLIGHT_REPORT_FILENAME,
                    report_payload,
                    **report_keywords,
                )
            if PREFLIGHT_RECEIPT_FILENAME in names:
                raise FullGridError(
                    "immutable preflight receipt already exists"
                )
            receipt_payload = _canonical_bytes(receipt) + b"\n"
            receipt_keywords: dict[str, Any] = {}
            if _publication_tracker is not None:
                def remember_receipt(published: os.stat_result) -> None:
                    _publication_tracker["artifacts"][
                        PREFLIGHT_RECEIPT_FILENAME
                    ] = {
                        "identity": (
                            int(published.st_dev),
                            int(published.st_ino),
                        ),
                        "payload": receipt_payload,
                    }

                receipt_keywords["on_publish"] = remember_receipt
            _write_bytes_exclusive_at(
                preflight_descriptor,
                PREFLIGHT_RECEIPT_FILENAME,
                receipt_payload,
                **receipt_keywords,
            )
            os.fsync(preflight_descriptor)
            _assert_directory_descriptor_path(root, root_descriptor)
            _assert_preflight_directory_descriptor_path(
                root,
                preflight_descriptor,
            )
        finally:
            os.close(preflight_descriptor)
    finally:
        os.close(root_descriptor)
    return load_preflight_attestation(root, plan)


def _formal_preflight_cuda_binding(
    *,
    plan: Mapping[str, Any],
    gpu_lease: project_gpu_leases.GPULease,
    device: str,
) -> str:
    """Bind a direct formal call to its one planned CUDA-visible device."""

    gpu_uuid = str(gpu_lease.value.get("gpu_uuid", ""))
    if (
        device != "cuda:0"
        or os.environ.get("CUDA_VISIBLE_DEVICES") != gpu_uuid
        or os.environ.get("FULL_GRID_GPU_UUID") != gpu_uuid
    ):
        raise GPUUnavailable(
            "formal preflight CUDA environment differs from its leased GPU"
        )
    _verify_visible_cuda_device(gpu_uuid)
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise GPUUnavailable(
            "formal preflight requires exactly one available CUDA device"
        )
    observed_name = str(torch.cuda.get_device_name(torch.device("cuda:0")))
    inventory = plan["environment_identity"].get("nvidia_gpu_inventory")
    planned = next(
        (
            row
            for row in inventory
            if isinstance(row, Mapping) and row.get("uuid") == gpu_uuid
        ),
        None,
    ) if isinstance(inventory, list) else None
    if (
        not isinstance(planned, Mapping)
        or not isinstance(planned.get("name"), str)
        or not planned["name"]
        or observed_name != planned["name"]
    ):
        raise GPUUnavailable(
            "formal preflight CUDA device name differs from planned inventory"
        )
    return observed_name


def _invalidate_preflight_publication(
    run_root: Path,
    *,
    keep_report: bool,
    plan: Mapping[str, Any],
    publication_tracker: Mapping[str, Any],
) -> tuple[Path, ...]:
    """Invalidate only exact leaves demonstrably published by this call."""

    root = _safe_run_root(run_root)
    tracked_raw = publication_tracker.get("artifacts")
    if not isinstance(tracked_raw, Mapping) or not tracked_raw:
        return ()
    current = _preflight_canonical_snapshot(root)
    current_artifacts = current["artifacts"]
    complete = {
        PREFLIGHT_REPORT_FILENAME,
        PREFLIGHT_RECEIPT_FILENAME,
    }

    # A later guarded winner can legitimately reuse an orphaned report while
    # publishing a new receipt. If a complete, valid pair contains any
    # different inode, it is not this failed invocation's pair and survives.
    if set(current_artifacts) == complete:
        tracked_differs = any(
            current_artifacts[name]["identity"]
            != tracked_raw[name]["identity"]
            for name in complete
            if name in tracked_raw
        )
        initial_raw = publication_tracker.get("initial_artifacts")
        initial_artifacts = (
            initial_raw if isinstance(initial_raw, Mapping) else {}
        )
        newly_published_untracked = any(
            name not in tracked_raw
            and (
                name not in initial_artifacts
                or current_artifacts[name]["identity"]
                != initial_artifacts[name]["identity"]
            )
            for name in complete
        )
        if tracked_differs or newly_published_untracked:
            try:
                load_preflight_attestation(root, plan)
            except (OSError, FullGridError):
                pass
            else:
                return ()

    targets = [PREFLIGHT_RECEIPT_FILENAME]
    if not keep_report:
        targets.append(PREFLIGHT_REPORT_FILENAME)
    moved: list[Path] = []
    for name in targets:
        expected = tracked_raw.get(name)
        if not isinstance(expected, Mapping):
            continue
        expected_identity = expected.get("identity")
        expected_payload = expected.get("payload")
        if (
            not isinstance(expected_identity, tuple)
            or len(expected_identity) != 2
            or not isinstance(expected_payload, bytes)
        ):
            raise FullGridError("preflight publication tracker is invalid")
        observed = _preflight_canonical_snapshot(root)["artifacts"].get(name)
        if (
            not isinstance(observed, Mapping)
            or observed.get("identity") != expected_identity
            or observed.get("payload") != expected_payload
        ):
            # Missing or replaced canonical state is ordinary contention.
            continue
        try:
            moved.append(
                _quarantine(
                    root,
                    root / PREFLIGHT_DIRECTORY / name,
                    category="preflight",
                    reason=(
                        "lease-postcondition-report"
                        if name == PREFLIGHT_REPORT_FILENAME
                        else "lease-postcondition-receipt"
                    ),
                    expected_identity=expected_identity,
                )
            )
        except (FileNotFoundError, ClaimUnavailable):
            continue

    # Preserve the historical empty-directory postcondition only when this
    # invocation created that exact directory from an absent initial state.
    if (
        publication_tracker.get("initial_directory_identity") is None
        and publication_tracker.get("directory_identity") is not None
    ):
        root_descriptor = _open_absolute_directory(root)
        try:
            try:
                observed_directory = os.stat(
                    PREFLIGHT_DIRECTORY,
                    dir_fd=root_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                pass
            else:
                expected_directory_identity = publication_tracker[
                    "directory_identity"
                ]
                if (
                    (
                        int(observed_directory.st_dev),
                        int(observed_directory.st_ino),
                    )
                    == expected_directory_identity
                ):
                    preflight_descriptor = _open_relative_directory(
                        root_descriptor,
                        PREFLIGHT_DIRECTORY,
                        create=False,
                    )
                    try:
                        held = os.fstat(preflight_descriptor)
                        if (
                            (int(held.st_dev), int(held.st_ino))
                            == expected_directory_identity
                            and not os.listdir(preflight_descriptor)
                        ):
                            os.rmdir(
                                PREFLIGHT_DIRECTORY,
                                dir_fd=root_descriptor,
                            )
                            os.fsync(root_descriptor)
                    finally:
                        os.close(preflight_descriptor)
        finally:
            os.close(root_descriptor)
    return tuple(moved)


def _publish_preflight_under_lease(
    *,
    run_root: Path,
    plan: Mapping[str, Any],
    gpu_lease: project_gpu_leases.GPULease,
    report_builder: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    keep_report_on_postcondition_failure: bool,
) -> dict[str, Any]:
    """Publish one attestation and invalidate it if lease-guard exit fails."""

    complete_names = {
        PREFLIGHT_REPORT_FILENAME,
        PREFLIGHT_RECEIPT_FILENAME,
    }
    attestation: dict[str, Any] | None = None
    publication_tracker: dict[str, Any] = {
        "initial_directory_identity": None,
        "initial_artifacts": {},
        "directory_identity": None,
        "artifacts": {},
    }
    try:
        with _gpu_lease_publication_authority(
            plan=plan,
            run_root=run_root,
            gpu_lease=gpu_lease,
        ) as receipt:
            assert receipt is not None
            # This is the first canonical snapshot. It must be taken only
            # after the project-wide publication registry guard is held.
            initial_snapshot = _preflight_canonical_snapshot(run_root)
            publication_tracker["initial_directory_identity"] = (
                initial_snapshot["directory_identity"]
            )
            publication_tracker["initial_artifacts"] = copy.deepcopy(
                initial_snapshot["artifacts"]
            )
            initial_names = set(initial_snapshot["artifacts"])
            if initial_names == complete_names:
                # A serialized loser safely accepts the already-valid winner.
                attestation = load_preflight_attestation(run_root, plan)
            else:
                report = copy.deepcopy(dict(report_builder(receipt)))
                attestation = _publish_preflight_attestation(
                    run_root=run_root,
                    plan=plan,
                    report=report,
                    publication_gpu_lease_receipt=receipt,
                    _publication_tracker=publication_tracker,
                )
    except Exception:
        if publication_tracker["artifacts"]:
            # Reacquire the same persistent project-wide registry fence
            # without asserting the now-failed lease. This prevents a
            # legitimate next publisher from racing exact-inode cleanup.
            with project_gpu_leases._registry_lock(
                _verified_project_root()
            ):
                _invalidate_preflight_publication(
                    run_root,
                    keep_report=keep_report_on_postcondition_failure,
                    plan=plan,
                    publication_tracker=publication_tracker,
                )
        raise
    assert attestation is not None
    return attestation


def run_model_cuda_preflight(
    *,
    cache_root: Path,
    device: str = "cuda:0",
    batch_size: int = 2,
    _model_check: Callable[..., Mapping[str, Any]] | None = None,
    plan: Mapping[str, Any] | None = None,
    run_root: Path | None = None,
    gpu_lease: project_gpu_leases.GPULease | None = None,
) -> dict[str, Any]:
    """Construct and differentiate every model in the exact common roster.

    This is a no-score engineering gate. It opens only cache identity/shape and
    montage metadata for four already developed contracts, creates synthetic
    standardized inputs, never reads trial features or labels, never
    reconstructs a split, and never computes an accuracy. ``_model_check`` is
    an internal dependency-injection seam used only by CPU enumeration tests.
    """

    formal = plan is not None
    if formal != (run_root is not None) or formal != (gpu_lease is not None):
        raise FullGridError(
            "formal preflight requires plan, run_root, and GPU lease together"
        )
    if formal and _model_check is not None:
        raise FullGridError("formal preflight prohibits injected model checks")
    if not formal and _model_check is None:
        raise FullGridError("CUDA preflight must be bound to an immutable plan")
    if batch_size < 2:
        raise ValueError("preflight batch_size must be at least two")
    formal_plan: Mapping[str, Any] | None = plan
    formal_root: Path | None = (
        _safe_run_root(Path(run_root)) if run_root is not None else None
    )
    physical_gpu_uuid: str | None = None
    bound_cuda_device_name: str | None = None
    if formal:
        assert formal_plan is not None
        assert formal_root is not None
        assert gpu_lease is not None
        if formal_plan["publication_mode"] != FORMAL_PUBLICATION_MODE:
            raise FullGridError("CUDA preflight plan is not formally publishable")
        physical_gpu_uuid = str(gpu_lease.value.get("gpu_uuid", ""))
        if physical_gpu_uuid not in formal_plan["execution_config"][
            "physical_gpu_uuid_roster"
        ]:
            raise FullGridError("preflight GPU lease is outside the immutable plan")
        try:
            project_gpu_leases.assert_gpu_lease(gpu_lease)
        except project_gpu_leases.ProjectGPULeaseError as error:
            raise FullGridError(
                f"project GPU lease was lost before preflight: {error}"
            ) from error
        bound_cuda_device_name = _formal_preflight_cuda_binding(
            plan=formal_plan,
            gpu_lease=gpu_lease,
            device=device,
        )
        # A completed attestation is reusable only when the current code,
        # vendored comparator, and worker environment still match the plan.
        # This gate deliberately does not open trial arrays, labels, or splits.
        _preflight_runtime_identity(formal_plan)
        if _preflight_stage_names(formal_root):
            with _gpu_lease_publication_authority(
                plan=formal_plan,
                run_root=formal_root,
                gpu_lease=gpu_lease,
            ):
                _cleanup_preflight_stages(formal_root)
        _root, authoritative_names = _preflight_authoritative_names(
            formal_root,
            allow_stages=True,
        )
        if authoritative_names == {
            PREFLIGHT_REPORT_FILENAME,
            PREFLIGHT_RECEIPT_FILENAME,
        }:
            return load_preflight_attestation(formal_root, formal_plan)["report"]
        if authoritative_names == {PREFLIGHT_RECEIPT_FILENAME}:
            raise FullGridError("preflight receipt exists without its report")
        if authoritative_names == {PREFLIGHT_REPORT_FILENAME}:
            orphan_report, _payload = _read_preflight_report(
                formal_root,
                formal_plan,
                require_exact_directory=False,
            )
            recovered = _publish_preflight_under_lease(
                run_root=formal_root,
                plan=formal_plan,
                gpu_lease=gpu_lease,
                report_builder=lambda unused: orphan_report,
                keep_report_on_postcondition_failure=True,
            )
            return recovered["report"]

    torch: Any | None = None
    target_device: Any = device
    parameter_count: Callable[[Any], int] | None = None
    forward: Callable[..., Any] | None = None
    configure_determinism: Callable[[int], None] | None = None
    if _model_check is None:
        import torch as torch_module

        from .models import parameter_count as count_parameters
        from .training import _forward as training_forward
        from .training import configure_determinism as configure

        torch = torch_module
        parameter_count = count_parameters
        forward = training_forward
        configure_determinism = configure
        target_device = torch.device(device)
        if target_device.type != "cuda" or not torch.cuda.is_available():
            raise GPUUnavailable("CUDA preflight requires an available CUDA device")
    architectures = COMMON_ARCHITECTURES
    plan_stub: dict[str, Any] = {}
    contract_data: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for contract_index, raw_contract in enumerate(MODEL_PREFLIGHT_CONTRACTS):
        contract = copy.deepcopy(raw_contract)
        dataset = str(contract["dataset"])
        subject = int(contract["subject"])
        metadata = _load_preflight_cache_metadata(
            cache_root=cache_root,
            dataset=dataset,
            subject=subject,
        )
        if formal_plan is not None:
            planned_cache = formal_plan["cache_identity"].get(
                _subject_identity_key(dataset, subject)
            )
            if (
                not isinstance(planned_cache, Mapping)
                or metadata["declared_cache_array_sha256"]
                != planned_cache.get("array_sha256")
            ):
                raise FullGridError(
                    f"preflight cache {dataset} S{subject} differs from the plan"
                )
        expected_shape = (
            int(contract["n_channels"]),
            int(contract["n_times"]),
        )
        identity_shape = tuple(int(value) for value in metadata["identity_shape"])
        if identity_shape[1:] != expected_shape:
            raise FullGridError(
                f"preflight cache {dataset} S{subject} has shape "
                f"{identity_shape}, expected (*, {expected_shape[0]}, "
                f"{expected_shape[1]})"
            )
        channel_names = tuple(metadata["channel_names"])
        positions = np.asarray(metadata["positions"], dtype=np.float32)
        if (
            len(channel_names) != expected_shape[0]
            or positions.shape != (expected_shape[0], 3)
            or not np.all(np.isfinite(positions))
        ):
            raise FullGridError(
                f"preflight cache {dataset} S{subject} has invalid montage metadata"
            )
        examples = _synthetic_preflight_input(
            contract,
            batch_size=batch_size,
            seed=71_000 + contract_index,
        )
        contract_data.append(
            (
                contract,
                {
                    "x": examples.astype(np.float32, copy=False),
                    "channel_names": channel_names,
                    "positions": positions,
                    "cache_metadata": {
                        "path": metadata["path"],
                        "identity_shape": metadata["identity_shape"],
                        "declared_cache_array_sha256": metadata[
                            "declared_cache_array_sha256"
                        ],
                        "members_opened": metadata["members_opened"],
                    },
                },
            )
        )

    results: list[dict[str, Any]] = []
    for model_index, model_name in enumerate(architectures):
        for contract_index, (contract, data) in enumerate(contract_data):
            if gpu_lease is not None:
                try:
                    project_gpu_leases.assert_gpu_lease(gpu_lease)
                except project_gpu_leases.ProjectGPULeaseError as error:
                    raise FullGridError(
                        "project GPU lease was lost between preflight checks"
                    ) from error
            seed = 91_000 + model_index * 101 + contract_index
            if _model_check is not None:
                check = dict(
                    _model_check(
                        model_name=model_name,
                        contract=copy.deepcopy(contract),
                        synthetic_input=np.asarray(data["x"]).copy(),
                        channel_names=tuple(data["channel_names"]),
                        positions=np.asarray(data["positions"]).copy(),
                        seed=seed,
                    )
                )
                try:
                    raw_parameter_count = check["parameter_count"]
                except KeyError as error:
                    raise FullGridError(
                        "injected preflight check returned invalid provenance"
                    ) from error
                if (
                    type(raw_parameter_count) is not int
                    or raw_parameter_count <= 0
                ):
                    raise FullGridError(
                        "injected preflight check returned invalid parameter count"
                    )
                observed_parameter_count = raw_parameter_count
                results.append(
                    {
                        "model": model_name,
                        "contract": str(contract["name"]),
                        "dataset": str(contract["dataset"]),
                        "n_channels": int(contract["n_channels"]),
                        "n_times": int(contract["n_times"]),
                        "n_outputs": int(contract["n_outputs"]),
                        "parameter_count": observed_parameter_count,
                        "forward_backward_optimizer_step": "mocked_cpu_test",
                    }
                )
                continue
            assert torch is not None
            assert parameter_count is not None
            assert forward is not None
            assert configure_determinism is not None
            configure_determinism(seed)
            job = Job(
                dataset=str(contract["dataset"]),
                model=model_name,
                subject=int(contract["subject"]),
                fold=0,
                seed=seed,
            )
            model: Any | None = None
            optimizer: Any | None = None
            logits: Any | None = None
            loss: Any | None = None
            post_step: Any | None = None
            try:
                model = _make_model(
                    plan_stub,
                    job,
                    n_channels=int(contract["n_channels"]),
                    n_outputs=int(contract["n_outputs"]),
                    n_times=int(contract["n_times"]),
                    sfreq=float(
                        preprocessing_for_dataset(str(contract["dataset"]))[
                            "sfreq_hz"
                        ]
                    ),
                    channel_names=data["channel_names"],
                    positions=data["positions"],
                ).to(target_device)
                model.train()
                values = torch.as_tensor(
                    data["x"],
                    dtype=torch.float32,
                    device=target_device,
                )
                positions = torch.as_tensor(
                    data["positions"],
                    dtype=torch.float32,
                    device=target_device,
                )
                targets = (
                    torch.arange(batch_size, device=target_device)
                    % int(contract["n_outputs"])
                ).long()
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=1e-4,
                    weight_decay=0.0,
                )
                optimizer.zero_grad(set_to_none=True)
                logits = forward(model, values, positions)
                expected_logits = (batch_size, int(contract["n_outputs"]))
                if tuple(logits.shape) != expected_logits or not torch.isfinite(
                    logits
                ).all():
                    raise FullGridError(
                        f"invalid logits {tuple(logits.shape)} for "
                        f"{model_name}/{contract['name']}"
                    )
                loss = torch.nn.functional.cross_entropy(logits, targets)
                if not torch.isfinite(loss):
                    raise FullGridError("preflight loss is not finite")
                loss.backward()
                gradients = [
                    parameter.grad
                    for parameter in model.parameters()
                    if parameter.grad is not None
                ]
                if (
                    not gradients
                    or not all(torch.isfinite(value).all() for value in gradients)
                    or not any(torch.count_nonzero(value) for value in gradients)
                ):
                    raise FullGridError("preflight gradients are absent/non-finite")
                optimizer.step()
                model.eval()
                with torch.no_grad():
                    post_step = forward(model, values, positions)
                if (
                    tuple(post_step.shape) != expected_logits
                    or not torch.isfinite(post_step).all()
                ):
                    raise FullGridError("post-step preflight logits are invalid")
                results.append(
                    {
                        "model": model_name,
                        "contract": str(contract["name"]),
                        "dataset": str(contract["dataset"]),
                        "n_channels": int(contract["n_channels"]),
                        "n_times": int(contract["n_times"]),
                        "n_outputs": int(contract["n_outputs"]),
                        "parameter_count": int(parameter_count(model)),
                        "forward_backward_optimizer_step": "passed",
                    }
                )
            except Exception as error:
                raise FullGridError(
                    f"CUDA preflight failed for {model_name}/"
                    f"{contract['name']}: {error}"
                ) from error
            finally:
                del model, optimizer, logits, loss, post_step
                torch.cuda.empty_cache()
    expected_count = len(architectures) * len(MODEL_PREFLIGHT_CONTRACTS)
    if len(results) != expected_count:
        raise FullGridError("CUDA preflight result cardinality is incomplete")

    def build_report(
        *,
        source_identity: Mapping[str, str],
        environment: Mapping[str, Any],
        gpu_lease_receipt: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            "schema": PREFLIGHT_SCHEMA,
            "created_at": _utc_now(),
            "score_blind": True,
            "input_source": "deterministic_synthetic_standardized_tensors",
            "cache_members_opened": ["identity", "positions", "channel_names"],
            "trial_feature_members_opened": False,
            "label_or_split_members_opened": False,
            "test_splits_opened": False,
            "accuracy_computed": False,
            "architectures": list(architectures),
            "track_scope": TRACK_SCOPE,
            "contracts": [
                copy.deepcopy(value) for value in MODEL_PREFLIGHT_CONTRACTS
            ],
            "cache_metadata": [
                copy.deepcopy(data["cache_metadata"])
                for _, data in contract_data
            ],
            "expected_checks": expected_count,
            "completed_checks": len(results),
            "source_identity": copy.deepcopy(dict(source_identity)),
            "environment_identity_sha256": _analysis_environment_sha256(
                environment
            ),
            "execution_backend": (
                "cuda_forward_backward_adamw"
                if _model_check is None
                else "injected_mock_cpu_test"
            ),
            "cuda_device_name": (
                bound_cuda_device_name
                if formal_plan is not None
                else None
            ),
            "plan_sha256": (
                formal_plan["plan_sha256"]
                if formal_plan is not None
                else None
            ),
            "physical_gpu_uuid": physical_gpu_uuid,
            "gpu_lease_receipt": copy.deepcopy(gpu_lease_receipt),
            "results": copy.deepcopy(results),
        }

    if formal_plan is None:
        report = build_report(
            source_identity=_source_identity(executor=DEFAULT_EXECUTOR),
            environment=_environment_identity(),
            gpu_lease_receipt=None,
        )
        _validate_preflight_report(report, plan=None, run_root=None)
        return report

    assert formal_root is not None
    assert gpu_lease is not None

    def build_formal_report(
        receipt: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        source_identity, environment = _preflight_runtime_identity(
            formal_plan
        )
        report = build_report(
            source_identity=source_identity,
            environment=environment,
            gpu_lease_receipt=receipt,
        )
        _validate_preflight_report(
            report,
            plan=formal_plan,
            run_root=formal_root,
        )
        return report

    attestation = _publish_preflight_under_lease(
        run_root=formal_root,
        plan=formal_plan,
        gpu_lease=gpu_lease,
        report_builder=build_formal_report,
        keep_report_on_postcondition_failure=False,
    )
    return attestation["report"]


def _parse_csv_row(line: str) -> list[str]:
    return [token.strip() for token in line.split(",")]


def probe_gpu(
    gpu: str,
    *,
    allowed_pids: Iterable[int] = (),
    max_utilization_percent: float = DEFAULT_MAX_IDLE_UTILIZATION_PERCENT,
    max_foreign_memory_mib: float = DEFAULT_MAX_FOREIGN_MEMORY_MIB,
) -> GPUStatus:
    """Return the cooperative availability state of one physical NVIDIA GPU."""

    allowed = {int(value) for value in allowed_pids}
    allowed.add(os.getpid())
    try:
        gpu_result = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(gpu),
                "--query-gpu=uuid,pci.bus_id,name,utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        gpu_rows = [
            _parse_csv_row(line)
            for line in gpu_result.stdout.splitlines()
            if line.strip()
        ]
        if len(gpu_rows) != 1 or len(gpu_rows[0]) != 5:
            raise ValueError("unexpected GPU query shape")
        gpu_uuid, pci_bus_id, gpu_name, utilization_text, memory_text = gpu_rows[0]
        if str(gpu).startswith("GPU-") and gpu_uuid != str(gpu):
            raise ValueError(
                f"nvidia-smi resolved {gpu!r} to unexpected UUID {gpu_uuid!r}"
            )
        utilization = float(utilization_text)
        memory_used = float(memory_text)
        if (
            GPU_UUID_RE.fullmatch(gpu_uuid) is None
            or not pci_bus_id
            or not gpu_name
            or not math.isfinite(utilization)
            or not 0.0 <= utilization <= 100.0
            or not math.isfinite(memory_used)
            or memory_used < 0.0
        ):
            raise ValueError("GPU query contains invalid identity/counters")
        process_result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        processes: list[dict[str, Any]] = []
        own_memory = 0.0
        for line in process_result.stdout.splitlines():
            if not line.strip():
                continue
            fields = _parse_csv_row(line)
            if (
                len(fields) != 4
                or not fields[0].isdigit()
                or GPU_UUID_RE.fullmatch(fields[1]) is None
                or not fields[2]
            ):
                raise ValueError(f"invalid compute-process query row: {line!r}")
            try:
                used_memory = float(fields[3])
            except ValueError as error:
                raise ValueError(
                    f"invalid compute-process memory value: {line!r}"
                ) from error
            if not math.isfinite(used_memory) or used_memory < 0.0:
                raise ValueError(f"invalid compute-process memory value: {line!r}")
            if fields[1] != gpu_uuid:
                continue
            pid = int(fields[0])
            process = {
                "pid": pid,
                "process_name": fields[2],
                "used_gpu_memory_mib": used_memory,
            }
            if pid in allowed:
                own_memory += used_memory
            else:
                processes.append(process)
        foreign_memory = max(0.0, memory_used - own_memory)
        reasons: list[str] = []
        if processes:
            reasons.append(f"{len(processes)} foreign compute process(es)")
        if utilization > max_utilization_percent:
            reasons.append(
                f"utilization {utilization:.1f}% > {max_utilization_percent:.1f}%"
            )
        if foreign_memory > max_foreign_memory_mib:
            reasons.append(
                f"foreign memory {foreign_memory:.1f} MiB > "
                f"{max_foreign_memory_mib:.1f} MiB"
            )
        safe = not reasons
        return GPUStatus(
            safe=safe,
            gpu=str(gpu),
            utilization_percent=utilization,
            memory_used_mib=memory_used,
            own_compute_memory_mib=own_memory,
            foreign_processes=tuple(processes),
            reason="idle" if not reasons else "; ".join(reasons),
            gpu_uuid=gpu_uuid,
            pci_bus_id=pci_bus_id,
            name=gpu_name,
        )
    except (FileNotFoundError, subprocess.SubprocessError, ValueError) as error:
        return GPUStatus(
            safe=False,
            gpu=str(gpu),
            utilization_percent=None,
            memory_used_mib=None,
            own_compute_memory_mib=0.0,
            foreign_processes=(),
            reason=f"nvidia-smi probe failed closed: {error}",
            gpu_uuid=str(gpu) if str(gpu).startswith("GPU-") else None,
        )


def probe_disk(
    path: Path,
    *,
    minimum_free_gib: float = DEFAULT_MIN_FREE_GIB,
) -> DiskStatus:
    """Fail closed when the shared filesystem is below its low-water mark."""

    if (
        not math.isfinite(minimum_free_gib)
        or minimum_free_gib < DEFAULT_MIN_FREE_GIB
    ):
        raise FullGridError(
            f"disk floor cannot be below {DEFAULT_MIN_FREE_GIB:.0f} GiB"
        )
    resolved = _absolute_path(path)
    try:
        stat = os.statvfs(resolved)
        free_bytes = int(stat.f_bavail) * int(stat.f_frsize)
        total_bytes = int(stat.f_blocks) * int(stat.f_frsize)
        free_gib = free_bytes / float(1024**3)
        below = free_gib < minimum_free_gib
        safe = not below
        if below:
            reason = (
                f"{free_gib:.2f} GiB free < {minimum_free_gib:.2f} GiB "
                "low-water mark"
            )
        else:
            reason = (
                f"{free_gib:.2f} GiB free >= {minimum_free_gib:.2f} GiB "
                "low-water mark"
            )
        return DiskStatus(
            safe=safe,
            path=str(resolved),
            free_bytes=free_bytes,
            total_bytes=total_bytes,
            free_gib=free_gib,
            minimum_free_gib=minimum_free_gib,
            reason=reason,
        )
    except OSError as error:
        return DiskStatus(
            safe=False,
            path=str(resolved),
            free_bytes=None,
            total_bytes=None,
            free_gib=None,
            minimum_free_gib=minimum_free_gib,
            reason=f"statvfs probe failed closed: {error}",
        )


def combine_resource_status(gpu: GPUStatus, disk: DiskStatus) -> ResourceStatus:
    reasons: list[str] = []
    if not gpu.safe:
        reasons.append(f"GPU: {gpu.reason}")
    if not disk.safe:
        reasons.append(f"disk: {disk.reason}")
    return ResourceStatus(
        safe=gpu.safe and disk.safe,
        gpu=gpu.as_dict(),
        disk=disk.as_dict(),
        reason="resources available" if not reasons else "; ".join(reasons),
    )


def _own_gpu_utilization_is_the_only_blocker(
    gpu: GPUStatus,
    *,
    max_utilization_percent: float,
    max_foreign_memory_mib: float,
) -> bool:
    """Identify a worker's sampled post-kernel utilization without weakening guards."""

    utilization = gpu.utilization_percent
    memory_used = gpu.memory_used_mib
    own_memory = gpu.own_compute_memory_mib
    if (
        gpu.safe
        or gpu.foreign_processes
        or isinstance(utilization, bool)
        or not isinstance(utilization, (int, float))
        or not math.isfinite(float(utilization))
        or float(utilization) <= max_utilization_percent
        or isinstance(memory_used, bool)
        or not isinstance(memory_used, (int, float))
        or not math.isfinite(float(memory_used))
        or float(memory_used) < 0.0
        or isinstance(own_memory, bool)
        or not isinstance(own_memory, (int, float))
        or not math.isfinite(float(own_memory))
        or float(own_memory) <= 0.0
        or max(0.0, float(memory_used) - float(own_memory))
        > max_foreign_memory_mib
    ):
        return False
    return True


def _wait_for_publication_resources(
    *,
    gpu_probe: Callable[[], GPUStatus],
    disk_probe: Callable[[], DiskStatus],
    max_utilization_percent: float = DEFAULT_MAX_IDLE_UTILIZATION_PERCENT,
    max_foreign_memory_mib: float = DEFAULT_MAX_FOREIGN_MEMORY_MIB,
    max_wait_seconds: float = PUBLICATION_GPU_COOLDOWN_SECONDS,
    poll_seconds: float = PUBLICATION_GPU_COOLDOWN_POLL_SECONDS,
    monotonic: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> ResourceStatus:
    """Wait only for this worker's residual utilization sample to become idle."""

    if (
        not math.isfinite(max_wait_seconds)
        or max_wait_seconds < 0.0
        or not math.isfinite(poll_seconds)
        or poll_seconds <= 0.0
    ):
        raise FullGridError("publication GPU cooldown settings are invalid")
    deadline = monotonic() + max_wait_seconds
    while True:
        gpu = gpu_probe()
        disk = disk_probe()
        resources = combine_resource_status(gpu, disk)
        if resources.safe or not disk.safe:
            return resources
        if not _own_gpu_utilization_is_the_only_blocker(
            gpu,
            max_utilization_percent=max_utilization_percent,
            max_foreign_memory_mib=max_foreign_memory_mib,
        ):
            return resources
        remaining = deadline - monotonic()
        if remaining <= 0.0:
            return resources
        sleeper(min(poll_seconds, remaining))


def _failure_files(run_root: Path, job: Job) -> tuple[Path, ...]:
    root = _failure_root(run_root, job)
    if not root.exists():
        return ()
    return tuple(sorted(root.glob(f"{job.job_id}.attempt-*.json")))


def _record_failure(
    run_root: Path,
    plan: Mapping[str, Any],
    claim: Claim,
    error: BaseException,
) -> Path:
    run = _safe_run_root(run_root)
    root = _ensure_real_directory(
        run,
        Path("failures") / claim.job.job_id[-2:],
    )
    existing = _failure_files(run_root, claim.job)
    attempt = len(existing) + 1
    path = root / f"{claim.job.job_id}.attempt-{attempt:04d}.json"
    value = {
        "schema": FAILURE_SCHEMA,
        "created_at": _utc_now(),
        "plan_sha256": plan["plan_sha256"],
        "job_id": claim.job.job_id,
        "job": claim.job.identity(),
        "owner": dict(claim.owner),
        "error_type": type(error).__name__,
        "error": str(error),
    }
    _validate_failure_receipt(value, plan=plan, job=claim.job)
    _write_json_exclusive(path, value)
    return path


def _validate_failure_receipt(
    value: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    job: Job,
) -> None:
    _exact_keys(
        value,
        {
            "schema",
            "created_at",
            "plan_sha256",
            "job_id",
            "job",
            "owner",
            "error_type",
            "error",
        },
        path="failure",
    )
    _exact_keys(
        value["owner"],
        {"host", "pid", "boot_id", "start_ticks"},
        path="failure.owner",
    )
    try:
        created = datetime.fromisoformat(str(value["created_at"]))
    except (TypeError, ValueError) as error:
        raise FullGridError("failure timestamp is invalid") from error
    owner = value["owner"]
    if (
        created.tzinfo is None
        or value["schema"] != FAILURE_SCHEMA
        or value["plan_sha256"] != plan["plan_sha256"]
        or value["job_id"] != job.job_id
        or value["job"] != job.identity()
        or not isinstance(value["error_type"], str)
        or not value["error_type"]
        or not isinstance(value["error"], str)
        or isinstance(owner["pid"], bool)
        or not isinstance(owner["pid"], int)
        or owner["pid"] <= 0
        or not all(
            isinstance(owner[name], str) and bool(owner[name])
            for name in ("host", "boot_id", "start_ticks")
        )
    ):
        raise FullGridError("failure receipt identity is invalid")


_FORENSIC_STABLE_FIELDS: tuple[str, ...] = (
    "st_dev",
    "st_ino",
    "st_mode",
    "st_nlink",
    "st_size",
    "st_mtime_ns",
    "st_ctime_ns",
)


def _forensic_stat_fingerprint(observed: os.stat_result) -> tuple[int, ...]:
    return tuple(
        int(getattr(observed, field)) for field in _FORENSIC_STABLE_FIELDS
    )


def _forensic_tree_digest_at(
    parent_descriptor: int,
    name: str,
    *,
    display_name: str | None = None,
) -> tuple[str, tuple[int, ...]]:
    """Hash one sealed tree while retaining and re-binding every opened inode."""

    rows: list[dict[str, Any]] = []

    def visit(
        node_parent: int,
        node_name: str,
        relative: str,
    ) -> tuple[int, ...]:
        initial_path = os.stat(
            node_name,
            dir_fd=node_parent,
            follow_symlinks=False,
        )
        mode = stat.S_IMODE(initial_path.st_mode)
        if stat.S_ISREG(initial_path.st_mode):
            if initial_path.st_nlink != 1 or mode & 0o222:
                raise FullGridError("forensic tree contains aliased/writable file")
            flags = (
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0)
            )
        elif stat.S_ISDIR(initial_path.st_mode):
            if mode & 0o222:
                raise FullGridError(
                    "forensic tree contains special/writable directory"
                )
            flags = _directory_open_flags()
        else:
            raise FullGridError("forensic tree contains a special node")
        descriptor = os.open(node_name, flags, dir_fd=node_parent)
        try:
            initial_descriptor = os.fstat(descriptor)
            initial_fingerprint = _forensic_stat_fingerprint(initial_path)
            if (
                _forensic_stat_fingerprint(initial_descriptor)
                != initial_fingerprint
            ):
                raise FullGridError("forensic node identity changed while opening")
            if stat.S_ISREG(initial_descriptor.st_mode):
                chunks: list[bytes] = []
                while True:
                    try:
                        chunk = os.read(descriptor, 1024 * 1024)
                    except InterruptedError:
                        continue
                    if not chunk:
                        break
                    chunks.append(chunk)
                payload = b"".join(chunks)
                if len(payload) != initial_descriptor.st_size:
                    raise FullGridError("forensic file read was incomplete")
                rows.append(
                    {
                        "path": relative,
                        "type": "file",
                        "mode": mode,
                        "size_bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                )
            else:
                children = sorted(os.listdir(descriptor))
                rows.append(
                    {
                        "path": relative,
                        "type": "directory",
                        "mode": mode,
                        "entries": children,
                    }
                )
                for child_name in children:
                    visit(
                        descriptor,
                        child_name,
                        f"{relative}/{child_name}",
                    )
                if sorted(os.listdir(descriptor)) != children:
                    raise FullGridError(
                        "forensic directory entries changed during traversal"
                    )
            final_descriptor = os.fstat(descriptor)
            final_path = os.stat(
                node_name,
                dir_fd=node_parent,
                follow_symlinks=False,
            )
            if any(
                _forensic_stat_fingerprint(observed) != initial_fingerprint
                for observed in (final_descriptor, final_path)
            ):
                raise FullGridError(
                    "forensic node name or inode changed during traversal"
                )
            return initial_fingerprint
        finally:
            os.close(descriptor)

    root_fingerprint = visit(
        parent_descriptor,
        name,
        display_name if display_name is not None else name,
    )
    return hashlib.sha256(_canonical_bytes(rows)).hexdigest(), root_fingerprint


def _forensic_tree_digest(path: Path) -> str:
    """Hash one sealed forensic tree through an anchored parent descriptor."""

    absolute = _absolute_path(path)
    parent_descriptor = _open_absolute_directory(absolute.parent)
    try:
        digest, _ = _forensic_tree_digest_at(
            parent_descriptor,
            absolute.name,
            display_name=absolute.name,
        )
        _assert_directory_descriptor_path(absolute.parent, parent_descriptor)
        return digest
    finally:
        os.close(parent_descriptor)


def audit_grid(run_root: Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    validate_plan_semantics(plan)
    root = _safe_run_root(run_root)
    preflight_attestation_valid = (
        plan["publication_mode"] != FORMAL_PUBLICATION_MODE
    )
    preflight_report_sha256: str | None = None
    if plan["publication_mode"] == FORMAL_PUBLICATION_MODE:
        try:
            preflight = load_preflight_attestation(root, plan)
        except Exception as error:
            preflight_error = str(error)
        else:
            preflight_error = None
            preflight_attestation_valid = True
            preflight_report_sha256 = preflight["report_sha256"]
    else:
        preflight_error = None
    validation_preflight_sha256 = (
        preflight_report_sha256
        if preflight_report_sha256 is not None
        else ("0" * 64 if plan["publication_mode"] == FORMAL_PUBLICATION_MODE else None)
    )
    jobs = tuple(iter_jobs(plan))
    jobs_by_id = {job.job_id: job for job in jobs}
    expected_directories = {_record_directory(root, job): job for job in jobs}
    complete: list[str] = []
    missing: list[str] = []
    corrupt: dict[str, str] = {}
    for directory, job in expected_directories.items():
        if not directory.exists():
            missing.append(job.job_id)
            continue
        try:
            _validate_completion_bound(
                root,
                plan,
                job,
                expected_preflight_report_sha256=validation_preflight_sha256,
            )
        except Exception as error:
            corrupt[job.job_id] = str(error)
        else:
            complete.append(job.job_id)

    unsafe_paths: list[str] = []
    if preflight_error is not None:
        unsafe_paths.append(f"{PREFLIGHT_DIRECTORY}: {preflight_error}")
    unexpected_record_paths: list[str] = []
    observed_directories: set[Path] = set()
    records_root = root / "records"
    if records_root.exists():
        try:
            _require_real_directory(records_root)
        except FullGridError as error:
            unsafe_paths.append(f"records: {error}")
        if not records_root.is_symlink():
            expected_files = {
                directory / name
                for directory in expected_directories
                for name in FINAL_FILENAMES
            }
            observed_files: set[Path] = set()
            allowed_directories = {records_root}
            for directory in expected_directories:
                current = directory
                while current != root:
                    allowed_directories.add(current)
                    if current == records_root:
                        break
                    current = current.parent
            for path in records_root.rglob("*"):
                observed = path.lstat()
                if path.is_symlink():
                    unsafe_paths.append(f"{path.relative_to(root)}: symlink")
                elif stat.S_ISREG(observed.st_mode):
                    observed_files.add(path)
                    if observed.st_nlink != 1:
                        unsafe_paths.append(
                            f"{path.relative_to(root)}: hardlinked file"
                        )
                elif stat.S_ISDIR(observed.st_mode):
                    if path not in allowed_directories:
                        unexpected_record_paths.append(str(path.relative_to(root)))
                else:
                    unsafe_paths.append(f"{path.relative_to(root)}: special node")
            observed_directories = {path.parent for path in observed_files}
            unexpected_record_paths.extend(
                sorted(
                    str(path.relative_to(root))
                    for path in observed_files - expected_files
                )
            )
    extras = sorted(
        str(path.relative_to(root))
        for path in observed_directories - set(expected_directories)
    )
    live_claims: list[str] = []
    stale_claims: list[str] = []
    unknown_claim_paths: list[str] = []
    expected_claims = {_claim_path(root, job): job for job in jobs}
    observed_claims: set[Path] = set()
    claims_root = root / "claims"
    if claims_root.exists() or claims_root.is_symlink():
        try:
            _require_real_directory(claims_root)
        except (FileNotFoundError, FullGridError) as error:
            unsafe_paths.append(f"claims: {error}")
        else:
            observed_claims = {
                path
                for path in claims_root.rglob("*")
                if not path.is_dir() or path.is_symlink()
            }
    for path in sorted(observed_claims):
        job = expected_claims.get(path)
        if job is None:
            unknown_claim_paths.append(str(path.relative_to(root)))
            continue
        try:
            value = strict_load(path)
            _validate_claim_value(
                value,
                plan=plan,
                job=job,
                run_root=root,
                expected_preflight_report_sha256=validation_preflight_sha256,
            )
        except Exception as error:
            unsafe_paths.append(f"{path.relative_to(root)}: {error}")
            stale_claims.append(job.job_id)
            continue
        target = live_claims if _claim_is_live(value) else stale_claims
        target.append(job.job_id)
    partial_paths: list[str] = []
    partial_root = root / "partials"
    if partial_root.exists() or partial_root.is_symlink():
        try:
            _require_real_directory(partial_root)
        except (FileNotFoundError, FullGridError) as error:
            unsafe_paths.append(f"partials: {error}")
        else:
            partial_paths = sorted(
                str(path.relative_to(root))
                for path in partial_root.iterdir()
            )
    residual_tombstones: list[str] = []
    tombstone_root = root / CLAIM_TOMBSTONE_DIRECTORY
    if tombstone_root.exists() or tombstone_root.is_symlink():
        try:
            _require_real_directory(tombstone_root)
        except FullGridError as error:
            unsafe_paths.append(f"{CLAIM_TOMBSTONE_DIRECTORY}: {error}")
        else:
            for path in tombstone_root.iterdir():
                residual_tombstones.append(str(path.relative_to(root)))
                try:
                    _require_unique_regular(path, read_only=True)
                    match = re.fullmatch(
                        r"(job-[0-9a-f]{24})\.([0-9a-f]{32})\.released",
                        path.name,
                    )
                    job = jobs_by_id.get(match.group(1)) if match else None
                    if job is None:
                        raise FullGridError("invalid claim tombstone filename")
                    _validate_claim_value(
                        strict_load(path),
                        plan=plan,
                        job=job,
                        run_root=root,
                        expected_preflight_report_sha256=(
                            validation_preflight_sha256
                        ),
                        expected_nonce=match.group(2),
                    )
                except FullGridError as error:
                    unsafe_paths.append(f"{path.relative_to(root)}: {error}")

    failed: dict[str, int] = {}
    failures_root = root / "failures"
    if failures_root.exists() or failures_root.is_symlink():
        _require_real_directory(failures_root)
        for receipt in failures_root.rglob("*"):
            observed = receipt.lstat()
            if stat.S_ISDIR(observed.st_mode) and not receipt.is_symlink():
                continue
            if receipt.is_symlink() or not stat.S_ISREG(observed.st_mode):
                unsafe_paths.append(f"{receipt.relative_to(root)}: unsafe failure node")
                continue
            if observed.st_nlink != 1 or observed.st_mode & 0o222:
                unsafe_paths.append(f"{receipt.relative_to(root)}: aliased/writable failure")
                continue
            match = re.fullmatch(
                r"(job-[0-9a-f]{24})\.attempt-([0-9]{4})\.json",
                receipt.name,
            )
            job = jobs_by_id.get(match.group(1)) if match else None
            if (
                job is None
                or receipt.parent.name != job.job_id[-2:]
            ):
                unsafe_paths.append(f"{receipt.relative_to(root)}: unknown failure leaf")
                continue
            try:
                value = strict_load(receipt)
                _validate_failure_receipt(value, plan=plan, job=job)
            except Exception as error:
                unsafe_paths.append(f"{receipt.relative_to(root)}: {error}")
                continue
            failed[job.job_id] = failed.get(job.job_id, 0) + 1

    quarantine_root = root / "quarantine"
    allowed_quarantine = set(QUARANTINE_REASONS)
    quarantine_artifacts: list[str] = []
    forensic_ledger: list[dict[str, Any]] = []
    forensic_candidates: list[
        tuple[dict[str, Any], str, str, tuple[int, ...]]
    ] = []
    forensic_category_bindings: dict[
        str, tuple[int, tuple[int, ...], frozenset[str]]
    ] = {}
    forensic_root_descriptor: int | None = None
    quarantine_descriptor: int | None = None
    quarantine_fingerprint: tuple[int, ...] | None = None
    quarantine_names: frozenset[str] = frozenset()
    invalid_forensic_artifacts = 0
    if quarantine_root.exists() or quarantine_root.is_symlink():
        try:
            forensic_root_descriptor = _open_absolute_directory(root)
            quarantine_descriptor = _open_relative_directory(
                forensic_root_descriptor,
                "quarantine",
                create=False,
            )
            quarantine_stat = os.fstat(quarantine_descriptor)
            quarantine_fingerprint = _forensic_stat_fingerprint(
                quarantine_stat
            )
            quarantine_names = frozenset(os.listdir(quarantine_descriptor))
        except (FileNotFoundError, OSError, FullGridError) as error:
            unsafe_paths.append(f"quarantine: {error}")
            if quarantine_descriptor is not None:
                os.close(quarantine_descriptor)
                quarantine_descriptor = None
            if forensic_root_descriptor is not None:
                os.close(forensic_root_descriptor)
                forensic_root_descriptor = None
        for category_name in sorted(quarantine_names):
            assert quarantine_descriptor is not None
            category = quarantine_root / category_name
            try:
                category_path_stat = os.stat(
                    category_name,
                    dir_fd=quarantine_descriptor,
                    follow_symlinks=False,
                )
            except OSError as error:
                quarantine_artifacts.append(
                    str(category.relative_to(root))
                )
                invalid_forensic_artifacts += 1
                unsafe_paths.append(
                    f"{category.relative_to(root)}: {error}"
                )
                continue
            if (
                category_name not in allowed_quarantine
                or not stat.S_ISDIR(category_path_stat.st_mode)
            ):
                quarantine_artifacts.append(str(category.relative_to(root)))
                invalid_forensic_artifacts += 1
                unsafe_paths.append(
                    f"{category.relative_to(root)}: unknown quarantine category"
                )
                continue
            category_descriptor: int | None = None
            try:
                category_descriptor = os.open(
                    category_name,
                    _directory_open_flags(),
                    dir_fd=quarantine_descriptor,
                )
                category_descriptor_stat = os.fstat(category_descriptor)
                if (
                    _forensic_stat_fingerprint(category_descriptor_stat)
                    != _forensic_stat_fingerprint(category_path_stat)
                ):
                    raise FullGridError(
                        "quarantine category changed while opening"
                    )
                artifact_names = frozenset(os.listdir(category_descriptor))
            except (OSError, FullGridError) as error:
                if category_descriptor is not None:
                    os.close(category_descriptor)
                quarantine_artifacts.append(str(category.relative_to(root)))
                invalid_forensic_artifacts += 1
                unsafe_paths.append(f"{category.relative_to(root)}: {error}")
                continue
            assert category_descriptor is not None
            forensic_category_bindings[category_name] = (
                category_descriptor,
                _forensic_stat_fingerprint(category_descriptor_stat),
                artifact_names,
            )
            for artifact_name in sorted(artifact_names):
                artifact = category / artifact_name
                relative_artifact = artifact.relative_to(root)
                quarantine_artifacts.append(str(relative_artifact))
                try:
                    observed = os.stat(
                        artifact_name,
                        dir_fd=category_descriptor,
                        follow_symlinks=False,
                    )
                    suffix = (
                        r"\.(?P<reason>[a-z-]+)\."
                        r"(?P<stamp>[0-9]{8}T[0-9]{6})\."
                        r"(?P<token>[0-9a-f]{10})"
                    )
                    if category_name == "claims":
                        match = re.fullmatch(
                            rf"(?P<job>job-[0-9a-f]{{24}})\.json{suffix}",
                            artifact_name,
                        )
                        if (
                            match is None
                            or match.group("reason")
                            not in QUARANTINE_REASONS["claims"]
                            or match.group("job") not in jobs_by_id
                            or not stat.S_ISREG(observed.st_mode)
                            or observed.st_nlink != 1
                            or observed.st_mode & 0o222
                        ):
                            raise FullGridError(
                                "quarantined claim violates its exact schema"
                        )
                        job = jobs_by_id[match.group("job")]
                        _validate_claim_value(
                            _strict_load_bytes(
                                _read_unique_regular_at(
                                    category_descriptor,
                                    artifact_name,
                                ),
                                source=str(artifact),
                            ),
                            plan=plan,
                            job=job,
                            run_root=root,
                            expected_preflight_report_sha256=(
                                validation_preflight_sha256
                            ),
                        )
                        forensic_identity = match.group("job")
                    elif category_name == "partials":
                        match = re.fullmatch(
                            rf"(?P<job>job-[0-9a-f]{{24}})\."
                            rf"(?P<nonce>[0-9a-f]{{32}})\.partial{suffix}",
                            artifact_name,
                        )
                        if (
                            match is None
                            or match.group("reason")
                            not in QUARANTINE_REASONS["partials"]
                            or match.group("job") not in jobs_by_id
                            or not stat.S_ISDIR(observed.st_mode)
                        ):
                            raise FullGridError(
                                "quarantined partial violates its exact schema"
                        )
                        forensic_identity = match.group("job")
                    elif category_name == "records":
                        match = re.fullmatch(
                            rf"(?P<record>s[0-9]{{3}}_f[0-9]{{2}}_seed[0-9]+_"
                            rf"[0-9a-f]{{10}}){suffix}",
                            artifact_name,
                        )
                        if (
                            match is None
                            or match.group("reason")
                            not in QUARANTINE_REASONS["records"]
                            or match.group("record")
                            not in {
                                directory.name
                                for directory in expected_directories
                            }
                            or not stat.S_ISDIR(observed.st_mode)
                        ):
                            raise FullGridError(
                                "quarantined record violates its exact schema"
                            )
                        forensic_identity = match.group("record")
                    else:
                        full_match = re.fullmatch(
                            rf"preflight{suffix}",
                            artifact_name,
                        )
                        receipt_match = re.fullmatch(
                            rf"receipt\.json{suffix}",
                            artifact_name,
                        )
                        report_match = re.fullmatch(
                            rf"report\.json{suffix}",
                            artifact_name,
                        )
                        match = full_match or receipt_match or report_match
                        if (
                            match is None
                            or match.group("reason")
                            not in QUARANTINE_REASONS["preflight"]
                            or (
                                full_match is not None
                                and (
                                    match.group("reason")
                                    != "lease-postcondition-full"
                                    or not stat.S_ISDIR(observed.st_mode)
                                )
                            )
                            or (
                                receipt_match is not None
                                and (
                                    match.group("reason")
                                    != "lease-postcondition-receipt"
                                    or not stat.S_ISREG(observed.st_mode)
                                    or observed.st_nlink != 1
                                )
                            )
                            or (
                                report_match is not None
                                and (
                                    match.group("reason")
                                    != "lease-postcondition-report"
                                    or not stat.S_ISREG(observed.st_mode)
                                    or observed.st_nlink != 1
                                )
                            )
                        ):
                            raise FullGridError(
                                "quarantined preflight evidence violates "
                                "its exact schema"
                            )
                        forensic_identity = plan["plan_sha256"]
                    artifact_digest, artifact_fingerprint = (
                        _forensic_tree_digest_at(
                            category_descriptor,
                            artifact_name,
                            display_name=artifact_name,
                        )
                    )
                    row = {
                        "category": category_name,
                        "reason": match.group("reason"),
                        "identity": forensic_identity,
                        "path": str(relative_artifact),
                        "artifact_sha256": artifact_digest,
                    }
                    forensic_candidates.append(
                        (
                            row,
                            category_name,
                            artifact_name,
                            artifact_fingerprint,
                        )
                    )
                except (FileNotFoundError, FullGridError) as error:
                    invalid_forensic_artifacts += 1
                    unsafe_paths.append(f"{relative_artifact}: {error}")

    worker_logs = root / "worker_logs"
    if worker_logs.exists() or worker_logs.is_symlink():
        try:
            _require_real_directory(worker_logs)
        except FullGridError as error:
            unsafe_paths.append(f"worker_logs: {error}")
        else:
            for path in worker_logs.rglob("*"):
                observed = path.lstat()
                if path.is_symlink():
                    unsafe_paths.append(f"{path.relative_to(root)}: symlink")
                elif stat.S_ISREG(observed.st_mode):
                    if observed.st_nlink != 1:
                        unsafe_paths.append(f"{path.relative_to(root)}: hardlinked")
                elif not stat.S_ISDIR(observed.st_mode):
                    unsafe_paths.append(f"{path.relative_to(root)}: special node")

    expected_root_entries = {
        "plan.json",
        "plan.sha256",
        "records",
        "claims",
        "partials",
        "failures",
        "quarantine",
        CLAIM_TOMBSTONE_DIRECTORY,
        "worker_logs",
        PREFLIGHT_DIRECTORY,
        PUBLICATION_FENCE_FILENAME,
    }
    unexpected_root = sorted(
        path.name for path in root.iterdir() if path.name not in expected_root_entries
    )
    for name in (PUBLICATION_FENCE_FILENAME,):
        path = root / name
        if path.exists() or path.is_symlink():
            try:
                _require_unique_regular(path, read_only=True)
            except FullGridError as error:
                unsafe_paths.append(f"{name}: {error}")
    if (
        plan["publication_mode"] == FORMAL_PUBLICATION_MODE
        and preflight_attestation_valid
    ):
        try:
            final_preflight = load_preflight_attestation(root, plan)
        except Exception as error:
            unsafe_paths.append(
                f"{PREFLIGHT_DIRECTORY}: changed during audit: {error}"
            )
            preflight_attestation_valid = False
        else:
            if final_preflight["report_sha256"] != preflight_report_sha256:
                unsafe_paths.append(
                    f"{PREFLIGHT_DIRECTORY}: digest changed during audit"
                )
                preflight_attestation_valid = False

    forensic_global_error: str | None = None
    forensic_category_errors: dict[str, str] = {}
    try:
        if quarantine_descriptor is not None:
            assert forensic_root_descriptor is not None
            assert quarantine_fingerprint is not None
            final_quarantine = os.fstat(quarantine_descriptor)
            final_quarantine_path = os.stat(
                "quarantine",
                dir_fd=forensic_root_descriptor,
                follow_symlinks=False,
            )
            if (
                _forensic_stat_fingerprint(final_quarantine)
                != quarantine_fingerprint
                or _forensic_stat_fingerprint(final_quarantine_path)
                != quarantine_fingerprint
                or frozenset(os.listdir(quarantine_descriptor))
                != quarantine_names
            ):
                raise FullGridError(
                    "quarantine root changed during the complete audit"
                )
            _assert_directory_descriptor_path(root, forensic_root_descriptor)
            _assert_directory_descriptor_path(
                quarantine_root,
                quarantine_descriptor,
            )
        for category_name, (
            category_descriptor,
            category_fingerprint,
            artifact_names,
        ) in forensic_category_bindings.items():
            assert quarantine_descriptor is not None
            try:
                final_category = os.fstat(category_descriptor)
                final_category_path = os.stat(
                    category_name,
                    dir_fd=quarantine_descriptor,
                    follow_symlinks=False,
                )
                if (
                    _forensic_stat_fingerprint(final_category)
                    != category_fingerprint
                    or _forensic_stat_fingerprint(final_category_path)
                    != category_fingerprint
                    or frozenset(os.listdir(category_descriptor))
                    != artifact_names
                ):
                    raise FullGridError(
                        "quarantine category changed during the complete audit"
                    )
            except (FileNotFoundError, OSError, FullGridError) as error:
                forensic_category_errors[category_name] = str(error)
                unsafe_paths.append(
                    f"quarantine/{category_name}: changed during audit: {error}"
                )
        for (
            row,
            category_name,
            artifact_name,
            artifact_fingerprint,
        ) in forensic_candidates:
            category_descriptor = forensic_category_bindings[category_name][0]
            relative_artifact = row["path"]
            category_error = forensic_category_errors.get(category_name)
            if forensic_global_error is not None or category_error is not None:
                invalid_forensic_artifacts += 1
                unsafe_paths.append(
                    f"{relative_artifact}: "
                    f"{forensic_global_error or category_error}"
                )
                continue
            try:
                final_digest, final_fingerprint = _forensic_tree_digest_at(
                    category_descriptor,
                    artifact_name,
                    display_name=artifact_name,
                )
                if (
                    final_fingerprint != artifact_fingerprint
                    or final_digest != row["artifact_sha256"]
                ):
                    raise FullGridError(
                        "forensic artifact changed during the complete audit"
                    )
            except (FileNotFoundError, OSError, FullGridError) as error:
                invalid_forensic_artifacts += 1
                unsafe_paths.append(f"{relative_artifact}: {error}")
            else:
                forensic_ledger.append(row)
    except (FileNotFoundError, OSError, FullGridError) as error:
        forensic_global_error = str(error)
        unsafe_paths.append(f"quarantine: changed during audit: {error}")
        invalid_forensic_artifacts += len(forensic_candidates)
    finally:
        for category_descriptor, _, _ in forensic_category_bindings.values():
            os.close(category_descriptor)
        if quarantine_descriptor is not None:
            os.close(quarantine_descriptor)
        if forensic_root_descriptor is not None:
            os.close(forensic_root_descriptor)

    if forensic_global_error is not None:
        forensic_ledger = []
    forensic_ledger.sort(
        key=lambda row: (
            row["category"],
            row["path"],
            row["artifact_sha256"],
        )
    )
    forensic_ledger_sha256 = hashlib.sha256(
        _canonical_bytes(forensic_ledger)
    ).hexdigest()
    exact = (
        len(complete) == len(jobs)
        and not missing
        and not corrupt
        and not extras
        and not live_claims
        and not stale_claims
        and not unknown_claim_paths
        and not partial_paths
        and not residual_tombstones
        and not unexpected_record_paths
        and not unsafe_paths
        and not unexpected_root
        and preflight_attestation_valid
    )
    return {
        "schema": AUDIT_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "expected": len(jobs),
        "complete": len(complete),
        "missing": len(missing),
        "corrupt": len(corrupt),
        "extra": len(extras),
        "live_claims": len(live_claims),
        "stale_claims": len(stale_claims),
        "unknown_claims": len(unknown_claim_paths),
        "partials": len(partial_paths),
        "claim_tombstones": len(residual_tombstones),
        "unsafe_paths": len(unsafe_paths),
        "unexpected_root_entries": len(unexpected_root),
        "failed_jobs": len(failed),
        "quarantine_artifacts": len(quarantine_artifacts),
        "resolved_forensic_artifacts": len(forensic_ledger),
        "invalid_forensic_artifacts": invalid_forensic_artifacts,
        "forensic_ledger_sha256": forensic_ledger_sha256,
        "preflight_attestation_valid": preflight_attestation_valid,
        "preflight_report_sha256": preflight_report_sha256,
        "exact_cartesian_complete": exact,
        "details": {
            "missing_job_ids": missing,
            "corrupt": corrupt,
            "extra_directories": extras,
            "live_claim_job_ids": live_claims,
            "stale_claim_job_ids": stale_claims,
            "unknown_claim_paths": unknown_claim_paths,
            "partial_paths": partial_paths,
            "claim_tombstone_paths": residual_tombstones,
            "unsafe_paths": sorted(set(unsafe_paths)),
            "unexpected_root_entries": unexpected_root,
            "unexpected_record_paths": sorted(set(unexpected_record_paths)),
            "failure_attempts": failed,
            "quarantine_artifact_paths": sorted(quarantine_artifacts),
            "forensic_ledger": forensic_ledger,
        },
    }


def _rotated_indices(length: int, start: int) -> Iterator[int]:
    for offset in range(length):
        yield (start + offset) % length


def _verified_project_root() -> Path:
    """Return the physical project root bound into the formal source identity."""

    project_root = _absolute_path(Path(__file__)).parent.parent
    _require_real_directory(project_root)
    current = project_root
    while current != current.parent:
        observed = current.lstat()
        if current.is_symlink() or not stat.S_ISDIR(observed.st_mode):
            raise FullGridError("project root has a symlinked/non-directory ancestor")
        current = current.parent
    for name in ("pyproject.toml", "uv.lock"):
        _read_unique_regular(project_root / name)
    return project_root


def _worker_loop_with_gpu_lease_held(
    *,
    run_root: Path,
    cache_root: Path,
    gpu: str,
    gpu_lease: project_gpu_leases.GPULease | None,
    worker_index: int,
    recover_stale: bool,
    retry_failed: bool,
    max_attempts_per_job: int,
    poll_seconds: float,
    max_utilization_percent: float,
    max_foreign_memory_mib: float,
    executor: Callable[..., tuple[dict[str, Any], np.ndarray, np.ndarray]]
    | None = None,
    gpu_probe: Callable[[], GPUStatus] | None = None,
    disk_probe: Callable[[], DiskStatus] | None = None,
    space_check_path: Path | None = None,
    minimum_free_gib: float = DEFAULT_MIN_FREE_GIB,
) -> int:
    """Run one independent dynamic worker until the grid completes or blocks."""

    plan = load_plan(run_root)
    preflight_report_sha256: str | None = None
    if plan["publication_mode"] == FORMAL_PUBLICATION_MODE:
        preflight_report_sha256 = load_preflight_attestation(
            run_root,
            plan,
        )["report_sha256"]
    if not math.isclose(
        float(minimum_free_gib),
        DEFAULT_MIN_FREE_GIB,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise FullGridError("worker disk floor must remain exactly 50 GiB")
    if (
        not math.isclose(
            float(max_utilization_percent),
            DEFAULT_MAX_IDLE_UTILIZATION_PERCENT,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or not math.isclose(
            float(max_foreign_memory_mib),
            DEFAULT_MAX_FOREIGN_MEMORY_MIB,
            rel_tol=0.0,
            abs_tol=0.0,
        )
    ):
        raise FullGridError("formal GPU-idle thresholds have no override")
    recover_claim_tombstones(
        run_root,
        plan,
        expected_preflight_report_sha256=preflight_report_sha256,
    )
    jobs = tuple(iter_jobs(plan))
    if not jobs:
        return 0
    if (
        plan["publication_mode"] == FORMAL_PUBLICATION_MODE
        and executor is not None
    ):
        raise FullGridError("formal workers prohibit executor injection")
    if plan["publication_mode"] == FORMAL_PUBLICATION_MODE and (
        gpu_probe is not None or disk_probe is not None
    ):
        raise FullGridError("formal workers prohibit resource-probe injection")
    if plan["executor"] != DEFAULT_EXECUTOR and (
        plan["publication_mode"] == FORMAL_PUBLICATION_MODE
    ):
        raise FullGridError("formal worker plan has a non-default executor")
    selected_executor = (
        _resolve_callable(str(plan["executor"])) if executor is None else executor
    )
    selected_gpu_probe = gpu_probe or (
        lambda: probe_gpu(
            gpu,
            allowed_pids=(os.getpid(),),
            max_utilization_percent=max_utilization_percent,
            max_foreign_memory_mib=max_foreign_memory_mib,
        )
    )
    checked_path = None if space_check_path is None else _absolute_path(space_check_path)

    def default_disk_probe() -> DiskStatus:
        primary = probe_disk(
            run_root,
            minimum_free_gib=minimum_free_gib,
        )
        if not primary.safe:
            return primary
        if checked_path is not None and checked_path != _absolute_path(run_root):
            secondary = probe_disk(
                checked_path,
                minimum_free_gib=minimum_free_gib,
            )
            if not secondary.safe:
                return secondary
        return primary

    selected_disk_probe = disk_probe or default_disk_probe

    def selected_probe() -> ResourceStatus:
        return combine_resource_status(
            selected_gpu_probe(),
            selected_disk_probe(),
        )

    def publication_probe() -> ResourceStatus:
        return _wait_for_publication_resources(
            gpu_probe=selected_gpu_probe,
            disk_probe=selected_disk_probe,
            max_utilization_percent=max_utilization_percent,
            max_foreign_memory_mib=max_foreign_memory_mib,
        )

    cursor = (worker_index * math.ceil(len(jobs) / MAX_GPU_WORKERS)) % len(jobs)
    invocation_failures: dict[str, int] = {}
    while True:
        claimed_or_ran = False
        gpu_was_busy = False
        unblocked_missing = False
        for index in _rotated_indices(len(jobs), cursor):
            job = jobs[index]
            output = _record_directory(run_root, job)
            if output.exists():
                try:
                    _validate_completion_bound(
                        run_root,
                        plan,
                        job,
                        expected_preflight_report_sha256=(
                            preflight_report_sha256
                        ),
                    )
                except Exception:
                    if not recover_stale:
                        continue
                else:
                    if recover_stale:
                        try:
                            recover_completed_job_auxiliary_state(
                                run_root,
                                plan,
                                job,
                                recover_stale=recover_stale,
                                expected_preflight_report_sha256=(
                                    preflight_report_sha256
                                ),
                            )
                        except ClaimUnavailable:
                            # Another worker won the stale-recovery race.
                            pass
                    continue
            historical_failures = len(_failure_files(run_root, job))
            if (
                not retry_failed
                and historical_failures >= max_attempts_per_job
            ) or invocation_failures.get(job.job_id, 0) >= max_attempts_per_job:
                continue
            unblocked_missing = True
            try:
                claim = acquire_claim(
                    run_root,
                    plan,
                    job,
                    recover_stale=recover_stale,
                    before_claim=selected_probe,
                    gpu_lease=gpu_lease,
                )
            except GPUUnavailable:
                gpu_was_busy = True
                break
            except DiskUnavailable as error:
                print(
                    f"[{_utc_now()}] worker={worker_index} stop: {error}",
                    file=sys.stderr,
                    flush=True,
                )
                return 2
            except ClaimUnavailable:
                continue
            try:
                print(
                    f"[{_utc_now()}] worker={worker_index} gpu={gpu} "
                    f"start {job.identity()}",
                    flush=True,
                )
                metadata, rows, probabilities = selected_executor(
                    job=job,
                    plan=plan,
                    cache_root=cache_root,
                    device="cuda:0",
                )
                destination = commit_job_output(
                    run_root,
                    plan,
                    claim,
                    metadata=metadata,
                    test_rows=rows,
                    probabilities=probabilities,
                    cache_root=cache_root,
                    before_publish=publication_probe,
                    gpu_lease=gpu_lease,
                )
                print(
                    f"[{_utc_now()}] worker={worker_index} complete "
                    f"{job.job_id} -> {destination}",
                    flush=True,
                )
            except Exception as error:
                invocation_failures[job.job_id] = (
                    invocation_failures.get(job.job_id, 0) + 1
                )
                failure = _record_failure(run_root, plan, claim, error)
                print(
                    f"[{_utc_now()}] worker={worker_index} failed "
                    f"{job.job_id}: {error}; {failure}",
                    file=sys.stderr,
                    flush=True,
                )
            finally:
                release_claim(claim)
            cursor = (index + 1) % len(jobs)
            claimed_or_ran = True
            break

        if claimed_or_ran:
            continue
        if gpu_was_busy:
            time.sleep(max(poll_seconds, 0.1))
            continue
        audit = audit_grid(run_root, plan)
        if audit["exact_cartesian_complete"]:
            return 0
        if audit["live_claims"]:
            time.sleep(max(poll_seconds, 0.1))
            continue
        if recover_stale and audit["stale_claims"]:
            jobs_by_id = {job.job_id: job for job in jobs}
            recovered_any = False
            for job_id in audit["details"]["stale_claim_job_ids"]:
                job = jobs_by_id[job_id]
                try:
                    recovered_any = bool(
                        recover_completed_job_auxiliary_state(
                            run_root,
                            plan,
                            job,
                            recover_stale=recover_stale,
                        )
                    ) or recovered_any
                except (ClaimUnavailable, FileNotFoundError):
                    continue
            if recovered_any:
                continue
        if not unblocked_missing or audit["failed_jobs"]:
            return 1
        time.sleep(max(poll_seconds, 0.1))


def worker_loop(
    *,
    run_root: Path,
    cache_root: Path,
    gpu: str,
    worker_index: int,
    recover_stale: bool,
    retry_failed: bool,
    max_attempts_per_job: int,
    poll_seconds: float,
    max_utilization_percent: float,
    max_foreign_memory_mib: float,
    executor: Callable[..., tuple[dict[str, Any], np.ndarray, np.ndarray]]
    | None = None,
    gpu_probe: Callable[[], GPUStatus] | None = None,
    disk_probe: Callable[[], DiskStatus] | None = None,
    space_check_path: Path | None = None,
    minimum_free_gib: float = DEFAULT_MIN_FREE_GIB,
) -> int:
    """Hold one distinct physical-GPU lease for the complete worker lifetime."""

    root = _safe_run_root(run_root)
    plan = load_plan(root)
    lease: project_gpu_leases.GPULease | None = None
    if plan["publication_mode"] == FORMAL_PUBLICATION_MODE:
        if not GPU_UUID_RE.fullmatch(gpu):
            raise GPUUnavailable("formal worker requires one exact physical GPU UUID")
        if gpu not in plan["execution_config"]["physical_gpu_uuid_roster"]:
            raise GPUUnavailable("GPU UUID is outside the immutable plan")
        try:
            lease = project_gpu_leases.acquire_gpu_lease(
                project_root=_verified_project_root(),
                run_root=root,
                plan_sha256=plan["plan_sha256"],
                gpu_uuid=gpu,
                track_scope=TRACK_SCOPE,
            )
        except project_gpu_leases.ProjectGPULeaseError as error:
            raise GPUUnavailable(str(error)) from error
    try:
        return _worker_loop_with_gpu_lease_held(
            run_root=root,
            cache_root=cache_root,
            gpu=gpu,
            gpu_lease=lease,
            worker_index=worker_index,
            recover_stale=recover_stale,
            retry_failed=retry_failed,
            max_attempts_per_job=max_attempts_per_job,
            poll_seconds=poll_seconds,
            max_utilization_percent=max_utilization_percent,
            max_foreign_memory_mib=max_foreign_memory_mib,
            executor=executor,
            gpu_probe=gpu_probe,
            disk_probe=disk_probe,
            space_check_path=space_check_path,
            minimum_free_gib=minimum_free_gib,
        )
    finally:
        if lease is not None:
            project_gpu_leases.release_gpu_lease(lease)


def _require_virtual_environment() -> None:
    if Path(sys.prefix).resolve() == Path(sys.base_prefix).resolve():
        raise FullGridError(
            "runner requires a virtual environment; use `uv venv` and "
            "`uv run python -m ieee_mi.full_grid ...`"
        )
    try:
        completed = subprocess.run(
            ["uv", "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as error:
        raise FullGridError("runner requires UV and an isolated UV venv") from error
    if not completed.stdout.strip().startswith("uv "):
        raise FullGridError("unexpected UV identity")


def _require_cublas_determinism_environment() -> None:
    observed = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if observed != REQUIRED_CUBLAS_WORKSPACE_CONFIG:
        raise FullGridError(
            "deterministic CUDA execution requires "
            f"CUBLAS_WORKSPACE_CONFIG={REQUIRED_CUBLAS_WORKSPACE_CONFIG!r} "
            "to be exported before Python starts"
        )


def _validate_disk_threshold(
    minimum_free_gib: float,
    *,
    allow_low_disk: bool = False,
) -> None:
    if allow_low_disk:
        raise FullGridError("the common-grid 50 GiB floor has no bypass")
    if (
        not math.isfinite(minimum_free_gib)
        or not math.isclose(
            minimum_free_gib,
            DEFAULT_MIN_FREE_GIB,
            rel_tol=0.0,
            abs_tol=0.0,
        )
    ):
        raise FullGridError(
            f"formal workers require the exact {DEFAULT_MIN_FREE_GIB:.0f} GiB floor"
        )


def _validate_worker_safety_values(args: argparse.Namespace) -> None:
    _validate_cpu_threads(args.cpu_threads)
    if args.max_attempts_per_job <= 0:
        raise ValueError("--max-attempts-per-job must be positive")
    if not math.isfinite(args.poll_seconds) or args.poll_seconds < 0.0:
        raise ValueError("--poll-seconds must be finite and nonnegative")
    if (
        not math.isclose(
            float(args.max_idle_utilization),
            DEFAULT_MAX_IDLE_UTILIZATION_PERCENT,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or not math.isclose(
            float(args.max_idle_foreign_memory_mib),
            DEFAULT_MAX_FOREIGN_MEMORY_MIB,
            rel_tol=0.0,
            abs_tol=0.0,
        )
    ):
        raise ValueError("formal GPU-idle thresholds have no override")
    _validate_disk_threshold(args.min_free_gib)


def _parse_gpus(value: str) -> tuple[str, ...]:
    result = tuple(token.strip() for token in value.split(",") if token.strip())
    if (
        not result
        or len(result) != len(set(result))
        or len(result) > MAX_GPU_WORKERS
    ):
        raise ValueError("GPUs must be 1-3 unique comma-separated identifiers")
    if any(
        not (token.isdigit() or GPU_UUID_RE.fullmatch(token))
        for token in result
    ):
        raise ValueError("GPU identifiers must be exact indices or full GPU UUIDs")
    return result


def _discover_gpu_identities() -> tuple[GPUIdentity, ...]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,pci.bus_id,name",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as error:
        raise GPUUnavailable(f"cannot enumerate physical GPUs: {error}") from error
    identities: list[GPUIdentity] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        fields = _parse_csv_row(line)
        if (
            len(fields) != 4
            or not fields[0].isdigit()
            or not GPU_UUID_RE.fullmatch(fields[1])
        ):
            raise GPUUnavailable(f"invalid nvidia-smi GPU inventory row: {line!r}")
        identities.append(
            GPUIdentity(
                index=fields[0],
                uuid=fields[1],
                pci_bus_id=fields[2],
                name=fields[3],
            )
        )
    if not identities:
        raise GPUUnavailable("nvidia-smi returned no physical GPUs")
    if len({identity.uuid for identity in identities}) != len(identities):
        raise GPUUnavailable("nvidia-smi returned duplicate physical GPU UUIDs")
    return tuple(identities)


def _resolve_gpus(
    value: str,
    *,
    identities: Sequence[GPUIdentity] | None = None,
) -> tuple[GPUIdentity, ...]:
    tokens = _parse_gpus(value)
    inventory = tuple(_discover_gpu_identities() if identities is None else identities)
    eligible = tuple(
        identity for identity in inventory if identity.index in FORMAL_GPU_INDICES
    )
    by_index = {identity.index: identity for identity in eligible}
    by_uuid = {identity.uuid: identity for identity in eligible}
    resolved: list[GPUIdentity] = []
    for token in tokens:
        identity = by_index.get(token) if token.isdigit() else by_uuid.get(token)
        if identity is None:
            raise GPUUnavailable(
                f"requested GPU {token!r} is outside formal indexes 0,1,2"
            )
        resolved.append(identity)
    if len({identity.uuid for identity in resolved}) != len(resolved):
        raise GPUUnavailable(
            "requested GPU identifiers contain aliases of the same physical GPU"
        )
    return tuple(resolved)


def _visible_cuda_uuid() -> str:
    import torch

    if torch.cuda.device_count() != 1:
        raise GPUUnavailable(
            f"worker requires exactly one CUDA-visible GPU; found "
            f"{torch.cuda.device_count()}"
        )
    raw_observed = getattr(torch.cuda.get_device_properties(0), "uuid", None)
    if raw_observed is None:
        raise GPUUnavailable("PyTorch did not expose the CUDA-visible GPU UUID")
    # PyTorch 2.6 exposes ``torch._C._CUuuid`` rather than ``str``.  Its stable
    # string representation is the bare UUID reported by nvidia-smi.
    observed = str(raw_observed).strip()
    if not observed:
        raise GPUUnavailable("PyTorch exposed an empty CUDA-visible GPU UUID")
    return observed if observed.startswith("GPU-") else f"GPU-{observed}"


def _verify_visible_cuda_device(
    expected_uuid: str,
    *,
    uuid_probe: Callable[[], str] = _visible_cuda_uuid,
) -> str:
    if not GPU_UUID_RE.fullmatch(expected_uuid):
        raise GPUUnavailable(f"worker received invalid GPU UUID {expected_uuid!r}")
    observed_uuid = uuid_probe()
    if observed_uuid != expected_uuid:
        raise GPUUnavailable(
            f"CUDA-visible device 0 is {observed_uuid}, expected {expected_uuid}"
        )
    return observed_uuid


def _validate_cpu_threads(value: int) -> int:
    threads = int(value)
    if not 1 <= threads <= 16:
        raise ValueError("--cpu-threads must be in [1, 16] on the shared workstation")
    return threads


def _configure_torch_cpu_threads(value: int) -> None:
    import torch

    threads = _validate_cpu_threads(value)
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # A direct test invocation may already have initialized the inter-op
        # pool. Spawned formal workers call this before any training operation.
        pass


def _spawn_workers(args: argparse.Namespace, plan: Mapping[str, Any]) -> int:
    if plan["publication_mode"] == FORMAL_PUBLICATION_MODE:
        load_preflight_attestation(Path(args.run_root), plan)
    gpus = _resolve_gpus(args.gpus)
    cpu_threads = _validate_cpu_threads(
        int(plan["execution_config"]["worker_cpu_threads"])
    )
    root = _safe_run_root(Path(args.run_root))
    root_descriptor = _open_absolute_directory(root)
    try:
        log_root_descriptor = _open_relative_directory(
            root_descriptor,
            "worker_logs",
            create=True,
        )
    except BaseException:
        os.close(root_descriptor)
        raise
    processes: list[tuple[subprocess.Popen[bytes], Any]] = []
    return_code = 0
    try:
        for index, gpu_identity in enumerate(gpus):
            gpu_uuid = gpu_identity.uuid
            command = [
                sys.executable,
                "-m",
                "ieee_mi.full_grid",
                "worker",
                "--run-root",
                str(root),
                "--cache-root",
                str(_absolute_path(Path(args.cache_root))),
                "--gpu",
                gpu_uuid,
                "--worker-index",
                str(index),
                "--max-attempts-per-job",
                str(args.max_attempts_per_job),
                "--poll-seconds",
                str(args.poll_seconds),
                "--space-check-path",
                str(
                    _absolute_path(Path(args.space_check_path))
                    if args.space_check_path
                    else root
                ),
                "--cpu-threads",
                str(cpu_threads),
            ]
            if args.retry_failed:
                command.append("--retry-failed")
            if not args.recover_stale:
                command.append("--no-recover-stale")
            log_name = (
                f"worker-{index:02d}-gpu-{gpu_identity.index}-"
                f"{gpu_uuid.removeprefix('GPU-')[:8]}.log"
            )
            log_descriptor = os.open(
                log_name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_APPEND
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0),
                0o600,
                dir_fd=log_root_descriptor,
            )
            log_stat = os.fstat(log_descriptor)
            log_path_stat = os.stat(
                log_name,
                dir_fd=log_root_descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISREG(log_stat.st_mode)
                or log_stat.st_nlink != 1
                or (
                    log_stat.st_dev,
                    log_stat.st_ino,
                )
                != (
                    log_path_stat.st_dev,
                    log_path_stat.st_ino,
                )
            ):
                os.close(log_descriptor)
                raise FullGridError(
                    f"worker log is not a unique regular file: {log_name}"
                )
            _assert_directory_descriptor_path(
                root / "worker_logs",
                log_root_descriptor,
            )
            _assert_directory_descriptor_path(root, root_descriptor)
            log_handle = os.fdopen(log_descriptor, "ab", buffering=0)
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu_uuid
            environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            environment["CUBLAS_WORKSPACE_CONFIG"] = REQUIRED_CUBLAS_WORKSPACE_CONFIG
            environment["FULL_GRID_PLAN_SHA256"] = str(plan["plan_sha256"])
            environment["FULL_GRID_GPU_UUID"] = gpu_uuid
            for variable in THREAD_ENVIRONMENT_VARIABLES:
                environment[variable] = str(cpu_threads)
            try:
                process = subprocess.Popen(
                    command,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    env=environment,
                )
            except BaseException:
                log_handle.close()
                raise
            processes.append((process, log_handle))
        for process, _ in processes:
            return_code = max(return_code, process.wait())
    except BaseException:
        # This also covers a failure while opening/spawning a later worker.
        # Never leave already-started GPU processes running unattended.
        for process, _ in processes:
            if process.poll() is None:
                process.terminate()
        for process, _ in processes:
            process.wait()
        raise
    finally:
        for _, handle in processes:
            handle.close()
        os.close(log_root_descriptor)
        os.close(root_descriptor)
    audit = audit_grid(root, plan)
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)
    return 0 if audit["exact_cartesian_complete"] else max(return_code, 1)


def _spawn_preflight_worker(
    args: argparse.Namespace,
    plan: Mapping[str, Any],
) -> int:
    if (
        not math.isclose(
            float(args.max_idle_utilization),
            DEFAULT_MAX_IDLE_UTILIZATION_PERCENT,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or not math.isclose(
            float(args.max_idle_foreign_memory_mib),
            DEFAULT_MAX_FOREIGN_MEMORY_MIB,
            rel_tol=0.0,
            abs_tol=0.0,
        )
    ):
        raise ValueError("preflight GPU-idle thresholds have no override")
    resolved = _resolve_gpus(args.gpu)
    if len(resolved) != 1:
        raise GPUUnavailable("CUDA preflight requires exactly one physical GPU")
    identity = resolved[0]
    if identity.uuid not in plan["execution_config"]["physical_gpu_uuid_roster"]:
        raise GPUUnavailable("preflight GPU is outside the immutable plan")
    status = probe_gpu(
        identity.uuid,
        max_utilization_percent=args.max_idle_utilization,
        max_foreign_memory_mib=args.max_idle_foreign_memory_mib,
    )
    if not status.safe:
        raise GPUUnavailable(status.reason)
    threads = _validate_cpu_threads(args.cpu_threads)
    command = [
        sys.executable,
        "-m",
        "ieee_mi.full_grid",
        "preflight-worker",
        "--run-root",
        str(_absolute_path(Path(args.run_root))),
        "--cache-root",
        str(_absolute_path(Path(args.cache_root))),
        "--gpu",
        identity.uuid,
        "--cpu-threads",
        str(threads),
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = identity.uuid
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment["CUBLAS_WORKSPACE_CONFIG"] = REQUIRED_CUBLAS_WORKSPACE_CONFIG
    environment["FULL_GRID_GPU_UUID"] = identity.uuid
    environment["FULL_GRID_PLAN_SHA256"] = str(plan["plan_sha256"])
    for variable in THREAD_ENVIRONMENT_VARIABLES:
        environment[variable] = str(threads)
    return subprocess.run(command, env=environment, check=False).returncode


def _add_worker_safety_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--recover-stale",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="quarantine and recover dead claims/corrupt records (default: on)",
    )
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--max-attempts-per-job", type=int, default=3)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument(
        "--space-check-path",
        help="filesystem path checked with statvfs (default: run root)",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=4,
        help="CPU threads per GPU worker on the shared workstation (default: 4)",
    )
    parser.set_defaults(
        min_free_gib=DEFAULT_MIN_FREE_GIB,
        max_idle_utilization=DEFAULT_MAX_IDLE_UTILIZATION_PERCENT,
        max_idle_foreign_memory_mib=DEFAULT_MAX_FOREIGN_MEMORY_MIB,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="initialize or resume the three-GPU grid")
    run.add_argument("--run-root", required=True)
    run.add_argument("--cache-root", required=True)
    run.add_argument("--gpus", default="0,1,2")
    _add_worker_safety_options(run)

    worker = subparsers.add_parser("worker", help=argparse.SUPPRESS)
    worker.add_argument("--run-root", required=True)
    worker.add_argument("--cache-root", required=True)
    worker.add_argument("--gpu", required=True)
    worker.add_argument("--worker-index", type=int, required=True)
    _add_worker_safety_options(worker)

    status = subparsers.add_parser("status", help="show validated grid progress")
    status.add_argument("--run-root", required=True)

    audit = subparsers.add_parser("audit", help="require the exact Cartesian grid")
    audit.add_argument("--run-root", required=True)
    audit.add_argument("--output")

    preflight = subparsers.add_parser(
        "preflight",
        help=(
            "run a no-score CUDA construction/backprop gate for the exact "
            "43-model common roster"
        ),
    )
    preflight.add_argument("--cache-root", required=True)
    preflight.add_argument("--run-root", required=True)
    preflight.add_argument("--gpu", default="0")
    preflight.set_defaults(
        max_idle_utilization=DEFAULT_MAX_IDLE_UTILIZATION_PERCENT,
        max_idle_foreign_memory_mib=DEFAULT_MAX_FOREIGN_MEMORY_MIB,
    )
    preflight.add_argument("--cpu-threads", type=int, default=4)

    preflight_worker = subparsers.add_parser(
        "preflight-worker",
        help=argparse.SUPPRESS,
    )
    preflight_worker.add_argument("--cache-root", required=True)
    preflight_worker.add_argument("--run-root", required=True)
    preflight_worker.add_argument("--gpu", required=True)
    preflight_worker.add_argument("--cpu-threads", type=int, default=4)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "preflight":
        _require_virtual_environment()
        _require_cublas_determinism_environment()
        _validate_cpu_threads(args.cpu_threads)
        if (
            not math.isclose(
                float(args.max_idle_utilization),
                DEFAULT_MAX_IDLE_UTILIZATION_PERCENT,
                rel_tol=0.0,
                abs_tol=0.0,
            )
            or not math.isclose(
                float(args.max_idle_foreign_memory_mib),
                DEFAULT_MAX_FOREIGN_MEMORY_MIB,
                rel_tol=0.0,
                abs_tol=0.0,
            )
        ):
            raise ValueError("preflight GPU-idle thresholds have no override")
        preflight_root = _absolute_path(Path(args.run_root))
        preflight_cache = _absolute_path(Path(args.cache_root))
        if (preflight_root / "plan.json").exists():
            plan = load_or_repair_plan(preflight_root)
            if (
                args.cpu_threads
                != plan.get("execution_config", {}).get("worker_cpu_threads")
            ):
                raise FullGridError("--cpu-threads differs from immutable plan")
            verify_runtime_identity(plan, cache_root=preflight_cache)
        else:
            plan = build_plan(
                cache_root=preflight_cache,
                worker_cpu_threads=args.cpu_threads,
            )
            plan = write_or_validate_plan(preflight_root, plan)
        raise SystemExit(_spawn_preflight_worker(args, plan))

    if args.command == "preflight-worker":
        _require_virtual_environment()
        _require_cublas_determinism_environment()
        _configure_torch_cpu_threads(args.cpu_threads)
        _verify_visible_cuda_device(args.gpu)
        preflight_root = _absolute_path(Path(args.run_root))
        plan = load_plan(preflight_root)
        if os.environ.get("FULL_GRID_PLAN_SHA256") != plan["plan_sha256"]:
            raise FullGridError("preflight worker plan binding is absent or stale")
        if args.gpu not in plan["execution_config"]["physical_gpu_uuid_roster"]:
            raise GPUUnavailable("preflight GPU is outside the immutable plan")
        status = probe_gpu(
            args.gpu,
            allowed_pids=(os.getpid(),),
        )
        if not status.safe:
            raise GPUUnavailable(status.reason)
        try:
            lease = project_gpu_leases.acquire_gpu_lease(
                project_root=_verified_project_root(),
                run_root=preflight_root,
                plan_sha256=plan["plan_sha256"],
                gpu_uuid=args.gpu,
                track_scope=TRACK_SCOPE,
            )
        except project_gpu_leases.ProjectGPULeaseError as error:
            raise GPUUnavailable(str(error)) from error
        try:
            report = run_model_cuda_preflight(
                cache_root=_absolute_path(Path(args.cache_root)),
                plan=plan,
                run_root=preflight_root,
                gpu_lease=lease,
            )
        finally:
            project_gpu_leases.release_gpu_lease(lease)
        payload = _canonical_bytes(report) + b"\n"
        print(payload.decode("utf-8"), end="")
        return

    run_root = _absolute_path(Path(args.run_root))
    if args.command == "run":
        _require_virtual_environment()
        _require_cublas_determinism_environment()
        _validate_worker_safety_values(args)
        _validate_disk_threshold(args.min_free_gib)
        cache_root = _absolute_path(Path(args.cache_root))
        if (run_root / "plan.json").exists():
            plan = load_or_repair_plan(run_root)
            if (
                args.cpu_threads
                != plan.get("execution_config", {}).get("worker_cpu_threads")
            ):
                raise FullGridError("--cpu-threads differs from immutable plan")
            verify_runtime_identity(plan, cache_root=cache_root)
        else:
            plan = build_plan(
                cache_root=cache_root,
                worker_cpu_threads=args.cpu_threads,
            )
            plan = write_or_validate_plan(run_root, plan)
        load_preflight_attestation(run_root, plan)
        raise SystemExit(_spawn_workers(args, plan))

    if args.command == "worker":
        _require_virtual_environment()
        _require_cublas_determinism_environment()
        _validate_worker_safety_values(args)
        _validate_disk_threshold(args.min_free_gib)
        plan = load_plan(run_root)
        if (
            args.cpu_threads
            != plan.get("execution_config", {}).get("worker_cpu_threads")
        ):
            raise FullGridError("worker CPU thread count differs from immutable plan")
        _configure_torch_cpu_threads(args.cpu_threads)
        if plan.get("executor") != DEFAULT_EXECUTOR:
            raise FullGridError(
                "formal workers refuse plans that do not use DEFAULT_EXECUTOR"
            )
        expected_hash = os.environ.get("FULL_GRID_PLAN_SHA256")
        if expected_hash is None:
            raise FullGridError(
                "worker must be launched by the audited orchestrator with "
                "FULL_GRID_PLAN_SHA256"
            )
        if expected_hash != plan["plan_sha256"]:
            raise FullGridError("worker plan hash differs from orchestrator")
        _verify_visible_cuda_device(args.gpu)
        cache_root = _absolute_path(Path(args.cache_root))
        verify_runtime_identity(
            plan,
            cache_root=cache_root,
            worker_cuda_visibility=True,
        )
        raise SystemExit(
            worker_loop(
                run_root=run_root,
                cache_root=cache_root,
                gpu=args.gpu,
                worker_index=args.worker_index,
                recover_stale=args.recover_stale,
                retry_failed=args.retry_failed,
                max_attempts_per_job=args.max_attempts_per_job,
                poll_seconds=args.poll_seconds,
                max_utilization_percent=args.max_idle_utilization,
                max_foreign_memory_mib=args.max_idle_foreign_memory_mib,
                space_check_path=(
                    _absolute_path(Path(args.space_check_path))
                    if args.space_check_path
                    else run_root
                ),
                minimum_free_gib=args.min_free_gib,
            )
        )

    plan = load_plan(run_root)
    report = audit_grid(run_root, plan)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.command == "audit" and args.output:
        output = _absolute_path(Path(args.output))
        if output.exists():
            raise FileExistsError(output)
        _write_bytes_exclusive(output, payload.encode("utf-8"))
    print(payload, end="")
    if args.command == "audit" and not report["exact_cartesian_complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
