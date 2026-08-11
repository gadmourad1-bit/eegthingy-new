"""Atomic harmonized-v2 procedure runner for the GeoAdaptNet family.

This module is intentionally separate from the common raw-epoch architecture
grid.  It executes the three registry identities in their own procedure track:

* ``architecture.geoadaptnet`` consumes a deterministic four-band SPD view;
* ``architecture.geoadaptnet_fb`` learns four temporal filters and keeps the
  full channel covariance (``reduced_dim=None``);
* ``architecture.geoadaptnet_fbsp`` learns the same temporal filters and uses
  the frozen eight-dimensional spatial BiMap (``reduced_dim=8``).

Every atomic job is one ``(stable_id, dataset, subject, fold, seed)`` fit.  A
validation partition selects only the epoch count.  All model, optimizer,
normalization, and running-statistic state is then reset to the seeded initial
state and refit on train plus validation for that fixed count.  The held-out
partition is exposed to exactly one probability-prediction call and its labels
are not passed to the fitting backend.

The producer is score blind: immutable records contain held-out row indices and
probabilities, never held-out labels or performance values.  The companion
``ieee_mi.geoadapt_v2_analysis`` module is the only score-joining boundary.

Production planning requires a UV-managed virtual environment and a lock whose
direct project dependencies include the numerical stack.  This module never
installs packages, mutates a system interpreter, or opens a confirmation cache.
"""

from __future__ import annotations

import argparse
import csv
import copy
import contextlib
import ctypes
import errno
import fcntl
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import random
import re
import socket
import stat
import subprocess
import sys
import time
import tomllib
import uuid
import zipfile
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from . import project_gpu_leases
from .config import (
    CHANNEL_SCALING,
    channels_for_dataset,
    coordinate_contract_for_dataset,
    dataset_spec,
    preprocessing_for_dataset,
)
from .model_registry import (
    GEOADAPT_BNCI001_NA_REASON as BNCI001_NA_REASON,
    GEOADAPT_BNCI004_NA_REASON,
    GEOADAPT_FBSP_BNCI004_NA_REASON as FBSP_BNCI004_NA_REASON,
)
from .project_gpu_leases import (
    acquire_gpu_lease,
    guard_gpu_lease,
    release_gpu_lease,
)
from .project_gpu_leases import assert_gpu_lease, gpu_lease_receipt


PLAN_SCHEMA = "ieee-mi-geoadapt-v2-plan-v2"
RECORD_SCHEMA = "ieee-mi-geoadapt-v2-score-blind-record-v1"
COMPLETION_SCHEMA = "ieee-mi-geoadapt-v2-completion-v1"
CLAIM_SCHEMA = "ieee-mi-geoadapt-v2-claim-v1"
FAILURE_SCHEMA = "ieee-mi-geoadapt-v2-failure-v1"
QUARANTINE_SCHEMA = "ieee-mi-geoadapt-v2-quarantine-v1"
AUDIT_SCHEMA = "ieee-mi-geoadapt-v2-audit-v1"
ANALYSIS_GATE_SCHEMA = "ieee-mi-geoadapt-v2-analysis-gate-v1"
GPU_UUID_RE = re.compile(
    r"^GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)

TRACK = "separate_procedure_geoadapt_harmonized_v2"
GPU_LEASE_TRACK_SCOPE = "geoadapt-v2"
EVIDENCE_SCOPE = "opened_development_datasets_only_not_confirmation"
MONTAGE_PROFILE = "harmonized"

STABLE_IDS: tuple[str, ...] = (
    "architecture.geoadaptnet",
    "architecture.geoadaptnet_fb",
    "architecture.geoadaptnet_fbsp",
)
DATASET_ORDER: tuple[str, ...] = (
    "local_exp4",
    "bnci2014_004",
    "cho2017",
    "physionet_mi",
)
PUBLIC_NA_DATASET = "bnci2014_001"
SEEDS: tuple[int, ...] = (7, 17, 27, 37, 47)

DATASET_SUBJECTS: Mapping[str, tuple[int, ...]] = {
    "local_exp4": (1, 3, 4, 5, 6, 7, 8, 10),
    "bnci2014_004": tuple(range(1, 10)),
    "cho2017": tuple(range(1, 53)),
    "physionet_mi": tuple(range(1, 55)),
}
DATASET_FOLDS: Mapping[str, tuple[int, ...]] = {
    "local_exp4": (0,),
    "bnci2014_004": (0,),
    "cho2017": (0, 1, 2, 3, 4),
    "physionet_mi": (0, 1, 2),
}

# These are complete stable-ID configurations, not runtime defaults.  A plan
# serializes them byte-for-byte and refuses a source/config drift on resume.
STABLE_CONFIGS: Mapping[str, Mapping[str, Any]] = {
    "architecture.geoadaptnet": {
        "runtime_key": "geoadapt",
        "implementation": "deepnet.model.GeoAdaptNet",
        "input_representation": "deterministic_four_band_spd",
        "model": {
            "bands": 4,
            "reduced_dim": 8,
            "num_classes": 2,
            "band_width": 24,
            "fusion_width": 32,
            "dropout": 0.25,
            "eps": 1e-5,
            "reference_momentum": 0.05,
            "residual_gate_init": 0.02,
            "auxiliary_intent": False,
        },
        "trainer": {
            "epochs": 180,
            "batch_size": 64,
            "learning_rate": 3e-4,
            "weight_decay": 1e-3,
            "patience": 25,
            "min_delta": 1e-4,
            "label_smoothing": 0.05,
            "intent_loss_weight": 0.0,
            "gradient_clip": 5.0,
            "num_workers": 0,
            "deep_supervision_weight": 0.0,
            "exclude_gate_from_weight_decay": False,
            "select_metric": "loss",
            "lr_swap_prob": 0.0,
            "mixup_alpha": 0.0,
            "augment_eps": 1e-5,
            "deterministic": True,
        },
    },
    "architecture.geoadaptnet_fb": {
        "runtime_key": "geoadapt_fb",
        "implementation": "deepnet.filterbank_net.FilterBankSPDNet",
        "input_representation": "harmonized_broadband_raw",
        "model": {
            "n_bands": 4,
            "kernel_size": 65,
            "dropout": 0.25,
            "eps": 1e-5,
            "reduced_dim": None,
            "warm_start_bands": True,
        },
        "trainer": {
            "epochs": 250,
            "batch_size": 64,
            "learning_rate": 1e-3,
            "weight_decay": 1e-3,
            "patience": 40,
            "min_delta": 1e-4,
            "label_smoothing": 0.05,
            "gradient_clip": 5.0,
            "num_workers": 0,
            "deterministic": True,
        },
    },
    "architecture.geoadaptnet_fbsp": {
        "runtime_key": "geoadapt_fbsp",
        "implementation": "deepnet.filterbank_net.FilterBankSPDNet",
        "input_representation": "harmonized_broadband_raw",
        "model": {
            "n_bands": 4,
            "kernel_size": 65,
            "dropout": 0.25,
            "eps": 1e-5,
            "reduced_dim": 8,
            "warm_start_bands": True,
        },
        "trainer": {
            "epochs": 250,
            "batch_size": 64,
            "learning_rate": 1e-3,
            "weight_decay": 1e-3,
            "patience": 40,
            "min_delta": 1e-4,
            "label_smoothing": 0.05,
            "gradient_clip": 5.0,
            "num_workers": 0,
            "deterministic": True,
        },
    },
}

FILTER_BANDS_HZ: tuple[tuple[float, float], ...] = (
    (8.0, 12.0),
    (11.0, 15.0),
    (14.0, 20.0),
    (20.0, 30.0),
)
FIR_TAPS = 65
FIR_WINDOW = "hamming"
COVARIANCE_SHRINKAGE = 1e-3

SOURCE_FILES: tuple[str, ...] = (
    "ieee_mi/__init__.py",
    "ieee_mi/geoadapt_v2.py",
    "ieee_mi/geoadapt_v2_analysis.py",
    "ieee_mi/config.py",
    "ieee_mi/data.py",
    "ieee_mi/model_registry.py",
    "ieee_mi/project_gpu_leases.py",
    "deepnet/model.py",
    "deepnet/__init__.py",
    "deepnet/filterbank_net.py",
    "deepnet/engine.py",
    "deepnet/spd.py",
    "deepnet/augment.py",
    "deepnet/csp_init.py",
    "deepnet/config.py",
    "pyproject.toml",
    "uv.lock",
)
REQUIRED_RUNTIME_DIRECT_DEPENDENCIES = frozenset(
    {"numpy", "scipy", "scikit-learn", "torch"}
)
REQUIRED_TEST_GROUP_DEPENDENCIES = frozenset({"pytest"})
REQUIRED_DOCS_GROUP_DEPENDENCIES = frozenset({"pdfplumber", "pypdf", "reportlab"})
REQUIRED_DEPENDENCY_GROUPS: Mapping[str, frozenset[str]] = {
    "docs": REQUIRED_DOCS_GROUP_DEPENDENCIES,
    "test": REQUIRED_TEST_GROUP_DEPENDENCIES,
}
# Backward-compatible internal name: this is intentionally the *runtime* set.
# Test tooling is frozen separately and is never a production import requirement.
REQUIRED_DIRECT_DEPENDENCIES = REQUIRED_RUNTIME_DIRECT_DEPENDENCIES
THREAD_ENVIRONMENT_VARIABLES: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "TBB_NUM_THREADS",
    "NUMBA_NUM_THREADS",
)
DEFAULT_MIN_FREE_GIB = 50.0
DEFAULT_WORKER_CPU_THREADS = 4
_FALLBACK_BOOT_MARKER = hashlib.sha256(
    (f"{socket.gethostname()}:{int((time.time() - time.monotonic()) // 60)}").encode(
        "utf-8"
    )
).hexdigest()
_FALLBACK_PROCESS_START_MARKER = hashlib.sha256(
    f"{os.getpid()}:{time.time_ns()}".encode("ascii")
).hexdigest()
DETERMINISTIC_TORCH_CONTRACT: Mapping[str, Any] = {
    "deterministic_algorithms": True,
    "warn_only": False,
    "cudnn_deterministic": True,
    "cudnn_benchmark": False,
    "cuda_matmul_tf32": False,
    "cudnn_tf32": False,
    "float32_matmul_precision": "highest",
    "cublas_workspace_config_on_cuda": ":4096:8",
}
PATH_COMPONENT_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
FINAL_FILENAMES = frozenset({"record.json", "predictions.npz", "completion.json"})
_COMPLETION_STABLE_FIELDS: tuple[str, ...] = (
    "st_dev",
    "st_ino",
    "st_mode",
    "st_nlink",
    "st_size",
    "st_mtime_ns",
    "st_ctime_ns",
)
FORBIDDEN_SCORE_BLIND_KEY_ALIASES = frozenset(
    {
        "y",
        "ytrue",
        "ytest",
        "target",
        "targets",
        "label",
        "labels",
        "testlabel",
        "testlabels",
        "classlabel",
        "classlabels",
        "predictedlabel",
        "predictedlabels",
        "predictionlabel",
        "predictionlabels",
        "metric",
        "metrics",
        "score",
        "scores",
        "accuracy",
        "balancedaccuracy",
        "cohenkappa",
        "rocauc",
        "groundtruth",
        "truth",
        "actual",
        "actuals",
        "actualoutcome",
        "actualoutcomes",
        "outcome",
        "outcomes",
        "testoutcome",
        "testoutcomes",
        "validationloss",
        "validationbalancedaccuracy",
    }
)


class GeoAdaptV2Error(RuntimeError):
    """Base error for a fail-closed procedure contract violation."""


class NotApplicableError(GeoAdaptV2Error):
    """The requested registry cell is explicitly not applicable."""


class ClaimUnavailable(GeoAdaptV2Error):
    """Another live worker owns a job or the job is already complete."""


class DiskUnavailable(GeoAdaptV2Error):
    """The output filesystem is below the frozen free-space floor."""


class GPUWorkerUnavailable(GeoAdaptV2Error):
    """The selected physical GPU or project-wide worker capacity is busy."""


@dataclass(frozen=True, order=True)
class Job:
    dataset: str
    stable_id: str
    subject: int
    fold: int
    seed: int

    def identity(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "stable_id": self.stable_id,
            "subject": int(self.subject),
            "fold": int(self.fold),
            "seed": int(self.seed),
        }

    @property
    def job_id(self) -> str:
        digest = hashlib.sha256(_canonical_bytes(self.identity())).hexdigest()
        return f"geoadapt-v2-{digest[:24]}"


@dataclass
class ClaimPublicationState:
    """Process-local exact record ownership carried from commit to release."""

    record_identity: tuple[int, int] | None = None


@dataclass(frozen=True)
class Claim:
    job: Job
    path: Path
    nonce: str
    owner: Mapping[str, Any]
    resource_guard: Mapping[str, Any]
    gpu_lease_receipt: Mapping[str, Any] | None
    st_dev: int
    st_ino: int
    publication_state: ClaimPublicationState = field(
        default_factory=ClaimPublicationState,
        compare=False,
        repr=False,
    )


@dataclass(frozen=True)
class CompletionSnapshot:
    """One coherent, descriptor-held completion package and its decoded data."""

    record: Mapping[str, Any]
    completion: Mapping[str, Any]
    rows: np.ndarray = field(repr=False)
    probabilities: np.ndarray = field(repr=False)
    artifact_bytes: Mapping[str, bytes] = field(repr=False)
    artifact_sha256: Mapping[str, str]
    directory_fingerprint: tuple[int, ...]
    entry_fingerprints: tuple[tuple[str, tuple[int, ...]], ...]


@dataclass(frozen=True)
class AnalysisGate:
    run_root: Path
    marker_path: Path
    nonce: str
    owner: Mapping[str, Any]
    marker_st_dev: int
    marker_st_ino: int
    lock_descriptor: int = field(repr=False)


@dataclass(frozen=True)
class PreparedJob:
    """Leakage-safe model inputs; deliberately contains no held-out labels."""

    channel_names: tuple[str, ...]
    positions: np.ndarray
    train_rows: np.ndarray
    validation_rows: np.ndarray
    source_rows: np.ndarray
    test_rows: np.ndarray
    y_train: np.ndarray
    y_validation: np.ndarray
    y_source: np.ndarray
    x_train: np.ndarray
    x_validation: np.ndarray
    x_source: np.ndarray
    x_test: np.ndarray
    cov_train: np.ndarray | None
    cov_validation: np.ndarray | None
    cov_source: np.ndarray | None
    cov_test: np.ndarray | None
    selection_mean: np.ndarray
    selection_std: np.ndarray
    refit_mean: np.ndarray
    refit_std: np.ndarray
    cache_array_sha256: str
    split_identity: Mapping[str, Any]


class ProcedureBackend(Protocol):
    def __call__(
        self,
        *,
        job: Job,
        plan: Mapping[str, Any],
        prepared: PreparedJob,
        device: str,
    ) -> tuple[Mapping[str, Any], np.ndarray]:
        """Return fit metadata and held-out probabilities."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


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


def _absolute_path(path: Path | str) -> Path:
    """Return an absolute lexical path without resolving any symbolic link."""

    return Path(os.path.abspath(os.fspath(path)))


def _open_directory_absolute(path: Path | str) -> int:
    """Open an absolute directory through anchored no-follow descriptors."""

    absolute = _absolute_path(path)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = os.open(absolute.anchor, flags)
    try:
        for component in absolute.parts[1:]:
            child = os.open(component, flags, dir_fd=descriptor)
            observed = os.fstat(child)
            if not stat.S_ISDIR(observed.st_mode):
                os.close(child)
                raise GeoAdaptV2Error(f"path component is not a directory: {absolute}")
            os.close(descriptor)
            descriptor = child
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _anchored_lstat(path: Path | str) -> os.stat_result:
    absolute = _absolute_path(path)
    parent = _open_directory_absolute(absolute.parent)
    try:
        return os.stat(
            absolute.name,
            dir_fd=parent,
            follow_symlinks=False,
        )
    finally:
        os.close(parent)


def _anchored_directory_entries(path: Path | str) -> dict[str, os.stat_result]:
    descriptor = _open_directory_absolute(path)
    try:
        return {
            name: os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            for name in os.listdir(descriptor)
        }
    finally:
        os.close(descriptor)


def _path_exists(path: Path | str) -> bool:
    try:
        _anchored_lstat(path)
    except FileNotFoundError:
        return False
    return True


def _path_binds_identity(
    path: Path | str,
    expected_identity: tuple[int, int],
) -> bool:
    try:
        observed = _anchored_lstat(path)
    except FileNotFoundError:
        return False
    return (int(observed.st_dev), int(observed.st_ino)) == tuple(
        map(int, expected_identity)
    )


def _require_single_link(status: os.stat_result, path: Path | str) -> None:
    """Reject regular-file aliases before reading, mutating, or unlinking them."""

    if stat.S_ISREG(status.st_mode) and int(status.st_nlink) != 1:
        raise GeoAdaptV2Error(
            f"regular files must have exactly one hard link: {_absolute_path(path)}"
        )


def _completion_stat_fingerprint(status: os.stat_result) -> tuple[int, ...]:
    return tuple(
        int(getattr(status, field)) for field in _COMPLETION_STABLE_FIELDS
    )


def _assert_safe_path(
    path: Path | str,
    *,
    leaf_kind: str = "any",
    allow_missing: bool = False,
) -> Path:
    """Reject symlinks and special nodes in every existing path component."""

    if leaf_kind not in {"any", "file", "directory"}:
        raise ValueError(f"unsupported leaf kind {leaf_kind!r}")
    absolute = _absolute_path(path)
    current = Path(absolute.anchor)
    missing = False
    parts = absolute.parts[1:] if absolute.is_absolute() else absolute.parts
    for index, component in enumerate(parts):
        current /= component
        is_leaf = index == len(parts) - 1
        if missing:
            continue
        try:
            status = os.lstat(current)
        except FileNotFoundError:
            if not allow_missing:
                raise
            missing = True
            continue
        mode = status.st_mode
        if stat.S_ISLNK(mode):
            raise GeoAdaptV2Error(f"symbolic links are forbidden: {current}")
        _require_single_link(status, current)
        if not is_leaf:
            if not stat.S_ISDIR(mode):
                raise GeoAdaptV2Error(f"path ancestor is not a directory: {current}")
            continue
        if leaf_kind == "file" and not stat.S_ISREG(mode):
            raise GeoAdaptV2Error(f"path is not a regular file: {current}")
        if leaf_kind == "directory" and not stat.S_ISDIR(mode):
            raise GeoAdaptV2Error(f"path is not a directory: {current}")
        if leaf_kind == "any" and not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise GeoAdaptV2Error(f"special filesystem nodes are forbidden: {current}")
    return absolute


def _safe_mkdir(path: Path | str, *, parents: bool = True) -> Path:
    """Create a directory tree while refusing aliases and special nodes."""

    absolute = _absolute_path(path)
    if not parents:
        parent = _open_directory_absolute(absolute.parent)
        try:
            os.mkdir(absolute.name, 0o700, dir_fd=parent)
            os.fsync(parent)
            child = os.open(
                absolute.name,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=parent,
            )
            try:
                if not stat.S_ISDIR(os.fstat(child).st_mode):
                    raise GeoAdaptV2Error(
                        f"created path is not a directory: {absolute}"
                    )
            finally:
                os.close(child)
        finally:
            os.close(parent)
        return absolute
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = os.open(absolute.anchor, flags)
    try:
        for component in absolute.parts[1:]:
            try:
                try:
                    child = os.open(component, flags, dir_fd=descriptor)
                except FileNotFoundError:
                    try:
                        os.mkdir(component, 0o700, dir_fd=descriptor)
                        os.fsync(descriptor)
                    except FileExistsError:
                        pass
                    child = os.open(component, flags, dir_fd=descriptor)
            except OSError as error:
                raise GeoAdaptV2Error(
                    f"directory path contains an alias or special node: {absolute}"
                ) from error
            observed = os.fstat(child)
            if not stat.S_ISDIR(observed.st_mode):
                os.close(child)
                raise GeoAdaptV2Error(
                    f"directory path contains a special node: {absolute}"
                )
            os.close(descriptor)
            descriptor = child
    finally:
        os.close(descriptor)
    return absolute


def _walk_safe_tree(root: Path | str) -> Iterator[Path]:
    """Yield descendants without following aliases; reject every special node."""

    absolute = _absolute_path(root)
    if not _path_exists(absolute):
        return
    root_descriptor = _open_directory_absolute(absolute)

    def walk(directory: Path, descriptor: int) -> Iterator[Path]:
        for name in sorted(os.listdir(descriptor)):
            status = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            child = directory / name
            if stat.S_ISDIR(status.st_mode):
                child_descriptor = os.open(
                    name,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_NONBLOCK", 0)
                    | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=descriptor,
                )
                try:
                    yield child
                    yield from walk(child, child_descriptor)
                finally:
                    os.close(child_descriptor)
            elif stat.S_ISREG(status.st_mode):
                _require_single_link(status, child)
                yield child
            else:
                raise GeoAdaptV2Error(
                    f"tree contains an aliased or special node: {child}"
                )

    try:
        yield from walk(absolute, root_descriptor)
    finally:
        os.close(root_descriptor)


def _safe_tree_sha256(path: Path | str) -> str:
    """Hash an exact regular file or alias-free directory tree."""

    absolute = _assert_safe_path(path)
    digest = hashlib.sha256()
    root_status = _anchored_lstat(absolute)
    if stat.S_ISREG(root_status.st_mode):
        _require_single_link(root_status, absolute)
        digest.update(b"file\0")
        digest.update(_safe_read_bytes(absolute))
        return digest.hexdigest()
    if not stat.S_ISDIR(root_status.st_mode):
        raise GeoAdaptV2Error(
            f"tree hash root is not a regular file or directory: {absolute}"
        )
    digest.update(b"directory\0")
    for child in _walk_safe_tree(absolute):
        relative = str(child.relative_to(absolute)).encode("utf-8")
        child_status = _anchored_lstat(child)
        if stat.S_ISDIR(child_status.st_mode):
            digest.update(b"d\0" + relative + b"\0")
        else:
            _require_single_link(child_status, child)
            digest.update(b"f\0" + relative + b"\0")
            digest.update(_safe_read_bytes(child))
    return digest.hexdigest()


def _nofollow_flags(flags: int) -> int:
    return flags | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)


def _open_existing_regular(path: Path | str, flags: int = os.O_RDONLY) -> int:
    """Open a regular leaf through no-follow directory descriptors.

    ``O_NONBLOCK`` prevents a raced FIFO replacement from hanging before the
    post-open type check. Walking with ``openat``-style ``dir_fd`` handles also
    prevents an ancestor from being swapped to a symlink between validation and
    leaf open.
    """

    absolute = _absolute_path(path)
    if absolute == Path(absolute.anchor):
        raise GeoAdaptV2Error("a filesystem root cannot be opened as a file")
    directory_flags = _nofollow_flags(
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NONBLOCK", 0)
    )
    directory_descriptor = os.open(absolute.anchor, directory_flags)
    try:
        for component in absolute.parts[1:-1]:
            try:
                next_descriptor = os.open(
                    component,
                    directory_flags,
                    dir_fd=directory_descriptor,
                )
            except FileNotFoundError:
                raise
            except OSError as error:
                raise GeoAdaptV2Error(
                    f"path ancestor is an alias or special node: {absolute}"
                ) from error
            next_status = os.fstat(next_descriptor)
            if not stat.S_ISDIR(next_status.st_mode):
                os.close(next_descriptor)
                raise GeoAdaptV2Error(f"path ancestor is not a directory: {absolute}")
            os.close(directory_descriptor)
            directory_descriptor = next_descriptor
        path_status = os.stat(
            absolute.name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if stat.S_ISLNK(path_status.st_mode):
            raise GeoAdaptV2Error(f"symbolic links are forbidden: {absolute}")
        if not stat.S_ISREG(path_status.st_mode):
            raise GeoAdaptV2Error(f"path is not a regular file: {absolute}")
        _require_single_link(path_status, absolute)
        try:
            descriptor = os.open(
                absolute.name,
                _nofollow_flags(flags | getattr(os, "O_NONBLOCK", 0)),
                dir_fd=directory_descriptor,
            )
        except FileNotFoundError:
            raise
        except OSError as error:
            raise GeoAdaptV2Error(
                f"path changed to an alias or special node: {absolute}"
            ) from error
    finally:
        os.close(directory_descriptor)
    descriptor_status = os.fstat(descriptor)
    if not stat.S_ISREG(descriptor_status.st_mode):
        os.close(descriptor)
        raise GeoAdaptV2Error(f"path is not a regular file: {absolute}")
    try:
        _require_single_link(descriptor_status, absolute)
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
            getattr(descriptor_status, field) != getattr(path_status, field)
            for field in stable_fields
        ):
            raise GeoAdaptV2Error(f"path changed while opening: {absolute}")
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def _assert_descriptor_matches_path(
    descriptor: int,
    path: Path | str,
    *,
    kind: str,
) -> None:
    absolute = _absolute_path(path)
    descriptor_status = os.fstat(descriptor)
    path_status = _anchored_lstat(absolute)
    _require_single_link(descriptor_status, absolute)
    _require_single_link(path_status, absolute)
    if (
        not stat.S_ISREG(descriptor_status.st_mode)
        or not stat.S_ISREG(path_status.st_mode)
        or (descriptor_status.st_dev, descriptor_status.st_ino)
        != (path_status.st_dev, path_status.st_ino)
    ):
        raise GeoAdaptV2Error(f"{kind} inode changed while held: {absolute}")


def _safe_read_bytes(path: Path | str) -> bytes:
    absolute = _absolute_path(path)
    descriptor = _open_existing_regular(absolute)
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
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
            getattr(before, field) != getattr(after, field) for field in stable_fields
        ):
            raise GeoAdaptV2Error(f"file changed while being read: {absolute}")
        _assert_descriptor_matches_path(
            descriptor,
            absolute,
            kind="read target",
        )
        payload = b"".join(chunks)
        if len(payload) != before.st_size:
            raise GeoAdaptV2Error(f"file read was incomplete: {absolute}")
        return payload
    finally:
        os.close(descriptor)


def _safe_read_text(path: Path | str, *, encoding: str) -> str:
    return _safe_read_bytes(path).decode(encoding)


def _safe_read_regular_at(
    directory_descriptor: int,
    directory: Path,
    name: str,
) -> bytes:
    """Read one immutable regular child through a held directory inode."""

    path = directory / name
    path_before = os.stat(
        name,
        dir_fd=directory_descriptor,
        follow_symlinks=False,
    )
    if not stat.S_ISREG(path_before.st_mode):
        raise GeoAdaptV2Error(f"record child is not a regular file: {path}")
    _require_single_link(path_before, path)
    descriptor = os.open(
        name,
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_CLOEXEC", 0),
        dir_fd=directory_descriptor,
    )
    try:
        descriptor_before = os.fstat(descriptor)
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
            not stat.S_ISREG(descriptor_before.st_mode)
            or descriptor_before.st_nlink != 1
            or any(
                getattr(descriptor_before, field) != getattr(path_before, field)
                for field in stable_fields
            )
        ):
            raise GeoAdaptV2Error(f"record child changed while opening: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        descriptor_after = os.fstat(descriptor)
        path_after = os.stat(
            name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if any(
            getattr(descriptor_before, field) != getattr(descriptor_after, field)
            or getattr(descriptor_before, field) != getattr(path_after, field)
            for field in stable_fields
        ):
            raise GeoAdaptV2Error(f"record child changed while being read: {path}")
        payload = b"".join(chunks)
        if len(payload) != descriptor_before.st_size:
            raise GeoAdaptV2Error(f"record child read was incomplete: {path}")
        return payload
    finally:
        os.close(descriptor)


def _read_procfs_regular_bounded(
    path: Path | str,
    *,
    maximum_bytes: int,
) -> bytes:
    """Read a procfs pseudo-file without trusting its reported size."""

    if (
        not isinstance(maximum_bytes, int)
        or isinstance(maximum_bytes, bool)
        or maximum_bytes < 1
    ):
        raise ValueError("maximum_bytes must be a positive integer")
    absolute = _absolute_path(path)
    parent = _open_directory_absolute(absolute.parent)
    descriptor = -1
    try:
        path_before = os.stat(
            absolute.name,
            dir_fd=parent,
            follow_symlinks=False,
        )
        _require_single_link(path_before, absolute)
        if not stat.S_ISREG(path_before.st_mode):
            raise GeoAdaptV2Error(f"procfs target is not a regular file: {absolute}")
        descriptor = os.open(
            absolute.name,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent,
        )
        descriptor_before = os.fstat(descriptor)
        _require_single_link(descriptor_before, absolute)
        if not stat.S_ISREG(descriptor_before.st_mode):
            raise GeoAdaptV2Error(f"procfs target is not a regular file: {absolute}")

        identity_before = (
            path_before.st_dev,
            path_before.st_ino,
            path_before.st_mode,
            path_before.st_nlink,
        )
        descriptor_identity_before = (
            descriptor_before.st_dev,
            descriptor_before.st_ino,
            descriptor_before.st_mode,
            descriptor_before.st_nlink,
        )
        if descriptor_identity_before != identity_before:
            raise GeoAdaptV2Error(
                f"procfs target inode changed before reading: {absolute}"
            )

        payload = bytearray()
        while True:
            remaining = maximum_bytes + 1 - len(payload)
            if remaining <= 0:
                raise GeoAdaptV2Error(
                    f"procfs target exceeds {maximum_bytes} bytes: {absolute}"
                )
            try:
                chunk = os.read(descriptor, min(4096, remaining))
            except InterruptedError:
                continue
            if not chunk:
                break
            payload.extend(chunk)
            if len(payload) > maximum_bytes:
                raise GeoAdaptV2Error(
                    f"procfs target exceeds {maximum_bytes} bytes: {absolute}"
                )

        descriptor_after = os.fstat(descriptor)
        path_after = os.stat(
            absolute.name,
            dir_fd=parent,
            follow_symlinks=False,
        )
        _require_single_link(descriptor_after, absolute)
        _require_single_link(path_after, absolute)
        if not stat.S_ISREG(descriptor_after.st_mode) or not stat.S_ISREG(
            path_after.st_mode
        ):
            raise GeoAdaptV2Error(f"procfs target changed type: {absolute}")
        identities_after = (
            (
                descriptor_after.st_dev,
                descriptor_after.st_ino,
                descriptor_after.st_mode,
                descriptor_after.st_nlink,
            ),
            (
                path_after.st_dev,
                path_after.st_ino,
                path_after.st_mode,
                path_after.st_nlink,
            ),
        )
        if any(identity != identity_before for identity in identities_after):
            raise GeoAdaptV2Error(
                f"procfs target inode changed while being read: {absolute}"
            )
        return bytes(payload)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(_safe_read_bytes(path))
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _rows_sha256(value: np.ndarray | Sequence[int]) -> str:
    rows = np.asarray(value)
    if rows.dtype != np.dtype(np.int64) or rows.ndim != 1:
        raise ValueError("split rows must be an int64 vector")
    return _array_sha256(rows)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_exact_int(value: Any, *, minimum: int | None = None) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and (minimum is None or value >= minimum)
    )


def _is_exact_float(value: Any, *, minimum: float | None = None) -> bool:
    return (
        isinstance(value, float)
        and math.isfinite(value)
        and (minimum is None or value >= minimum)
    )


def _require_exact_scalar_types(
    observed: Any,
    expected: Any,
    *,
    path: str,
) -> None:
    """Reject JSON scalar coercions even when Python equality accepts them."""

    if isinstance(expected, Mapping):
        if not isinstance(observed, Mapping) or set(observed) != set(expected):
            raise GeoAdaptV2Error(f"{path} has an invalid exact mapping schema")
        for key, expected_child in expected.items():
            _require_exact_scalar_types(
                observed[key],
                expected_child,
                path=f"{path}.{key}",
            )
        return
    if isinstance(expected, list):
        if not isinstance(observed, list) or len(observed) != len(expected):
            raise GeoAdaptV2Error(f"{path} has an invalid exact list schema")
        for index, (observed_child, expected_child) in enumerate(
            zip(observed, expected, strict=True)
        ):
            _require_exact_scalar_types(
                observed_child,
                expected_child,
                path=f"{path}[{index}]",
            )
        return
    if isinstance(expected, tuple):
        if not isinstance(observed, tuple) or len(observed) != len(expected):
            raise GeoAdaptV2Error(f"{path} has an invalid exact tuple schema")
        for index, (observed_child, expected_child) in enumerate(
            zip(observed, expected, strict=True)
        ):
            _require_exact_scalar_types(
                observed_child,
                expected_child,
                path=f"{path}[{index}]",
            )
        return
    if expected is None:
        if observed is not None:
            raise GeoAdaptV2Error(f"{path} must be null")
        return
    if type(observed) is not type(expected):
        raise GeoAdaptV2Error(
            f"{path} has scalar type {type(observed).__name__}; "
            f"expected {type(expected).__name__}"
        )


def _exact_scalar_types_match(observed: Any, expected: Any) -> bool:
    try:
        _require_exact_scalar_types(observed, expected, path="value")
    except GeoAdaptV2Error:
        return False
    return True


def _is_finite_number(value: Any, *, minimum: float | None = None) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and (minimum is None or float(value) >= minimum)
    )


def _is_utc_timestamp(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    return (
        parsed.tzinfo is not None
        and parsed.utcoffset() is not None
        and parsed.utcoffset().total_seconds() == 0.0
    )


def _is_uuid4_hex(value: Any) -> bool:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{32}", value) is None:
        return False
    try:
        parsed = uuid.UUID(hex=value)
    except ValueError:
        return False
    return parsed.version == 4 and parsed.variant == uuid.RFC_4122


def _validate_job_mapping(value: Any, *, expected: Job | None = None) -> None:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"dataset", "stable_id", "subject", "fold", "seed"}
        or not isinstance(value["dataset"], str)
        or not value["dataset"]
        or not isinstance(value["stable_id"], str)
        or not value["stable_id"]
        or not _is_exact_int(value["subject"], minimum=1)
        or not _is_exact_int(value["fold"], minimum=0)
        or not _is_exact_int(value["seed"], minimum=0)
        or (expected is not None and dict(value) != expected.identity())
    ):
        raise GeoAdaptV2Error("job object has an invalid exact schema or type")


def _validate_owner_mapping(value: Any) -> None:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"host", "pid", "start_marker", "boot_marker"}
        or not isinstance(value["host"], str)
        or not value["host"]
        or not _is_exact_int(value["pid"], minimum=1)
        or not isinstance(value["start_marker"], str)
        or not value["start_marker"]
        or not isinstance(value["boot_marker"], str)
        or not value["boot_marker"]
    ):
        raise GeoAdaptV2Error("owner object has an invalid exact schema or type")


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _strict_json_bytes(payload: bytes, *, source: str) -> dict[str, Any]:
    value = json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{source} does not contain a JSON object")
    return value


def strict_load(path: Path) -> dict[str, Any]:
    return _strict_json_bytes(_safe_read_bytes(path), source=str(path))


def _fsync_directory(path: Path) -> None:
    absolute = _absolute_path(path)
    descriptor = _open_directory_absolute(absolute)
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise GeoAdaptV2Error(f"path is not a directory: {absolute}")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _seal_directory_read_only(
    path: Path,
    *,
    expected_dev: int,
    expected_ino: int,
) -> None:
    """Seal and fsync the exact held directory inode before publication."""

    absolute = _absolute_path(path)
    descriptor = _open_directory_absolute(absolute)
    try:
        observed = os.fstat(descriptor)
        if not stat.S_ISDIR(observed.st_mode) or (observed.st_dev, observed.st_ino) != (
            expected_dev,
            expected_ino,
        ):
            raise GeoAdaptV2Error(
                f"directory identity changed before sealing: {absolute}"
            )
        os.fchmod(descriptor, 0o555)
        os.fsync(descriptor)
        sealed = os.fstat(descriptor)
        path_status = _anchored_lstat(absolute)
        if (
            stat.S_IMODE(sealed.st_mode) != 0o555
            or (sealed.st_dev, sealed.st_ino) != (expected_dev, expected_ino)
            or (path_status.st_dev, path_status.st_ino, path_status.st_mode)
            != (sealed.st_dev, sealed.st_ino, sealed.st_mode)
        ):
            raise GeoAdaptV2Error(f"directory seal did not persist: {absolute}")
    finally:
        os.close(descriptor)
    _fsync_directory(absolute.parent)


def _write_bytes_exclusive(path: Path, payload: bytes) -> None:
    """Durably stage bytes and atomically publish without replacing a target."""

    path = _absolute_path(path)
    _safe_mkdir(path.parent)
    staged = path.parent / f".{path.name}.publish-{uuid.uuid4().hex}.staged"
    parent_descriptor = _open_directory_absolute(path.parent)
    descriptor: int | None = None
    staged_identity: os.stat_result | None = None
    try:
        descriptor = os.open(
            staged.name,
            _nofollow_flags(
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NONBLOCK", 0)
            ),
            0o600,
            dir_fd=parent_descriptor,
        )
        try:
            descriptor_status = os.fstat(descriptor)
            if not stat.S_ISREG(descriptor_status.st_mode):
                raise GeoAdaptV2Error(
                    f"created staged path is not a regular file: {staged}"
                )
            _require_single_link(descriptor_status, staged)
            offset = 0
            while offset < len(payload):
                written = os.write(descriptor, payload[offset:])
                if written <= 0:
                    raise OSError("staged publication write made no progress")
                offset += written
            os.fchmod(descriptor, 0o444)
            os.fsync(descriptor)
            staged_identity = os.fstat(descriptor)
        finally:
            os.close(descriptor)
            descriptor = None
        os.fsync(parent_descriptor)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_descriptor)
    if staged_identity is None:
        raise GeoAdaptV2Error(f"staged publication has no inode identity: {staged}")
    _rename_noreplace(
        staged,
        path,
        expected_source_dev=int(staged_identity.st_dev),
        expected_source_ino=int(staged_identity.st_ino),
    )


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    _write_bytes_exclusive(path, _canonical_bytes(payload) + b"\n")


def _staged_publication_entries(
    path: Path | str,
) -> tuple[tuple[Path, tuple[int, int]], ...]:
    """Return validated stage paths with their exact enumerated identities."""

    target = _absolute_path(path)
    if not _path_exists(target.parent):
        return ()
    prefix = f".{target.name}.publish-"
    suffix = ".staged"
    stages: list[tuple[Path, tuple[int, int]]] = []
    for name, observed in sorted(_anchored_directory_entries(target.parent).items()):
        if not name.startswith(prefix) or not name.endswith(suffix):
            continue
        nonce = name[len(prefix) : -len(suffix)]
        candidate = target.parent / name
        if re.fullmatch(r"[0-9a-f]{32}", nonce) is None:
            raise GeoAdaptV2Error(
                f"staged publication has an invalid identity: {candidate}"
            )
        if not stat.S_ISREG(observed.st_mode):
            raise GeoAdaptV2Error(
                f"staged publication is not a regular file: {candidate}"
            )
        _require_single_link(observed, candidate)
        stages.append(
            (
                candidate,
                (int(observed.st_dev), int(observed.st_ino)),
            )
        )
    return tuple(stages)


def _staged_publication_paths(path: Path | str) -> tuple[Path, ...]:
    """Return only uniquely linked regular stages created for one target."""

    return tuple(candidate for candidate, _identity in _staged_publication_entries(path))


@contextlib.contextmanager
def _authority_publication_lock(
    path: Path | str,
    *,
    lock_root: Path | str | None = None,
) -> Iterator[None]:
    """Serialize recovery and publication for one final authority path."""

    target = _absolute_path(path)
    _safe_mkdir(target.parent)
    lock_directory = (
        target.parent if lock_root is None else _safe_mkdir(_absolute_path(lock_root))
    )
    lock_identity = hashlib.sha256(os.fsencode(target)).hexdigest()
    lock = lock_directory / f".{lock_identity}.publication.flock"
    parent = _open_directory_absolute(lock_directory)
    descriptor: int | None = None
    try:
        descriptor = os.open(
            lock.name,
            _nofollow_flags(os.O_RDWR | os.O_CREAT | getattr(os, "O_NONBLOCK", 0)),
            0o600,
            dir_fd=parent,
        )
        observed = os.fstat(descriptor)
        if not stat.S_ISREG(observed.st_mode):
            raise GeoAdaptV2Error(f"authority publication lock is not regular: {lock}")
        _require_single_link(observed, lock)
        os.fsync(parent)
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
            descriptor = None
        raise
    finally:
        os.close(parent)
    try:
        assert descriptor is not None
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        _assert_descriptor_matches_path(
            descriptor,
            lock,
            kind="authority publication lock",
        )
        try:
            yield
        finally:
            _assert_descriptor_matches_path(
                descriptor,
                lock,
                kind="authority publication lock",
            )
    finally:
        if descriptor is not None:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)


def _unlink_owned_regular(
    path: Path | str,
    *,
    expected_dev: int | None = None,
    expected_ino: int | None = None,
) -> None:
    """Remove only the exact unique regular inode opened beneath its parent."""

    absolute = _absolute_path(path)
    parent = _open_directory_absolute(absolute.parent)
    descriptor: int | None = None
    try:
        descriptor = os.open(
            absolute.name,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent,
        )
        descriptor_stat = os.fstat(descriptor)
        path_stat = os.stat(
            absolute.name,
            dir_fd=parent,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(descriptor_stat.st_mode)
            or descriptor_stat.st_nlink != 1
            or (descriptor_stat.st_dev, descriptor_stat.st_ino)
            != (path_stat.st_dev, path_stat.st_ino)
            or (expected_dev is not None and descriptor_stat.st_dev != expected_dev)
            or (expected_ino is not None and descriptor_stat.st_ino != expected_ino)
        ):
            raise GeoAdaptV2Error(
                f"refusing to remove a replaced or aliased inode: {absolute}"
            )
        os.fchmod(descriptor, 0o600)
        os.unlink(absolute.name, dir_fd=parent)
        os.fsync(parent)
        if os.fstat(descriptor).st_nlink != 0:
            raise GeoAdaptV2Error(f"removed inode retained a link: {absolute}")
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent)


def _remove_owned_empty_directory(
    path: Path | str,
    *,
    expected_dev: int | None = None,
    expected_ino: int | None = None,
) -> None:
    """Remove only the exact empty directory inode opened beneath its parent."""

    absolute = _absolute_path(path)
    parent = _open_directory_absolute(absolute.parent)
    descriptor: int | None = None
    try:
        descriptor = os.open(
            absolute.name,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent,
        )
        descriptor_stat = os.fstat(descriptor)
        path_stat = os.stat(
            absolute.name,
            dir_fd=parent,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(descriptor_stat.st_mode)
            or (descriptor_stat.st_dev, descriptor_stat.st_ino)
            != (path_stat.st_dev, path_stat.st_ino)
            or (expected_dev is not None and descriptor_stat.st_dev != expected_dev)
            or (expected_ino is not None and descriptor_stat.st_ino != expected_ino)
            or os.listdir(descriptor)
        ):
            raise GeoAdaptV2Error(
                f"refusing to remove a replaced or nonempty directory: {absolute}"
            )
        os.rmdir(absolute.name, dir_fd=parent)
        os.fsync(parent)
        try:
            os.stat(
                absolute.name,
                dir_fd=parent,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            raise GeoAdaptV2Error(f"removed directory path still exists: {absolute}")
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = _absolute_path(path)
    _safe_mkdir(path.parent)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    parent_descriptor = _open_directory_absolute(path.parent)
    descriptor: int | None = None
    try:
        try:
            existing = os.stat(
                path.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            existing = None
        if existing is not None:
            if not stat.S_ISREG(existing.st_mode):
                raise GeoAdaptV2Error(f"atomic JSON target is not regular: {path}")
            _require_single_link(existing, path)
        descriptor = os.open(
            temporary.name,
            _nofollow_flags(
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NONBLOCK", 0)
            ),
            0o600,
            dir_fd=parent_descriptor,
        )
        descriptor_status = os.fstat(descriptor)
        if not stat.S_ISREG(descriptor_status.st_mode):
            raise GeoAdaptV2Error(
                f"created temporary path is not a regular file: {temporary}"
            )
        _require_single_link(descriptor_status, temporary)
        encoded = _canonical_bytes(payload) + b"\n"
        offset = 0
        while offset < len(encoded):
            offset += os.write(descriptor, encoded[offset:])
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        os.replace(
            temporary.name,
            path.name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
        )
        os.fsync(parent_descriptor)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            status = os.stat(
                temporary.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            if not stat.S_ISREG(status.st_mode):
                raise GeoAdaptV2Error(
                    f"temporary path became a special node: {temporary}"
                )
            os.unlink(temporary.name, dir_fd=parent_descriptor)
            os.fsync(parent_descriptor)
        os.close(parent_descriptor)


def _rename_noreplace(
    source: Path | str,
    destination: Path | str,
    *,
    expected_source_dev: int | None = None,
    expected_source_ino: int | None = None,
) -> None:
    """Atomically rename while refusing to replace an existing destination."""

    source_path = _absolute_path(source)
    destination_path = _absolute_path(destination)
    libc = ctypes.CDLL(None, use_errno=True)
    source_parent = _open_directory_absolute(source_path.parent)
    try:
        destination_parent = _open_directory_absolute(destination_path.parent)
    except Exception:
        os.close(source_parent)
        raise
    try:
        source_status = os.stat(
            source_path.name,
            dir_fd=source_parent,
            follow_symlinks=False,
        )
        if not (
            stat.S_ISREG(source_status.st_mode) or stat.S_ISDIR(source_status.st_mode)
        ):
            raise GeoAdaptV2Error(
                f"rename source is an alias or special node: {source_path}"
            )
        _require_single_link(source_status, source_path)
        if (
            expected_source_dev is not None
            and source_status.st_dev != expected_source_dev
        ) or (
            expected_source_ino is not None
            and source_status.st_ino != expected_source_ino
        ):
            raise GeoAdaptV2Error(
                f"rename source inode changed before commit: {source_path}"
            )
        source_bytes = os.fsencode(source_path.name)
        destination_bytes = os.fsencode(destination_path.name)
        if platform.system() == "Darwin" and hasattr(libc, "renameatx_np"):
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
                source_parent,
                source_bytes,
                destination_parent,
                destination_bytes,
                0x00000004,
            )
        elif platform.system() == "Linux" and hasattr(libc, "renameat2"):
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
                source_parent,
                source_bytes,
                destination_parent,
                destination_bytes,
                1,
            )
        else:  # fail closed instead of emulating a racy check-then-rename
            raise GeoAdaptV2Error(
                "this platform lacks an anchored atomic no-replace rename primitive"
            )
        if result == 0:
            destination_status = os.stat(
                destination_path.name,
                dir_fd=destination_parent,
                follow_symlinks=False,
            )
            if (
                destination_status.st_dev,
                destination_status.st_ino,
            ) != (source_status.st_dev, source_status.st_ino):
                raise GeoAdaptV2Error(
                    "atomic rename destination does not retain the source inode"
                )
            os.fsync(source_parent)
            if (
                source_parent != destination_parent
                or source_path.parent != destination_path.parent
            ):
                os.fsync(destination_parent)
    finally:
        os.close(source_parent)
        os.close(destination_parent)
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(
                error_number, os.strerror(error_number), destination_path
            )
        raise OSError(error_number, os.strerror(error_number), destination_path)


def _subject_key(dataset: str, subject: int) -> str:
    return f"{dataset}:s{int(subject):03d}"


def _split_key(dataset: str, subject: int, fold: int) -> str:
    return f"{dataset}:s{int(subject):03d}:f{int(fold):02d}"


def _partition_identity(rows: np.ndarray) -> dict[str, Any]:
    values = np.asarray(rows)
    if (
        values.dtype != np.dtype(np.int64)
        or values.ndim != 1
        or len(values) == 0
        or np.any(values < 0)
        or len(set(values.tolist())) != len(values)
    ):
        raise ValueError("split rows must be unique, nonnegative, nonempty int64")
    return {"count": len(values), "rows_sha256": _rows_sha256(values)}


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
        np.int64, copy=False
    )
    partitions = {
        "train": _partition_identity(train_rows),
        "validation": _partition_identity(validation_rows),
        "source": _partition_identity(source_rows),
        "test": _partition_identity(test_rows),
    }
    return {
        "dataset": dataset,
        "subject": int(subject),
        "fold": int(fold),
        "cache_array_sha256": cache_array_sha256,
        "trial_count": int(trial_count),
        "partitions": partitions,
    }


def _cell_status(stable_id: str, dataset: str) -> dict[str, Any]:
    if stable_id not in STABLE_IDS:
        raise KeyError(stable_id)
    if dataset == PUBLIC_NA_DATASET:
        return {"status": "not_applicable", "reason": BNCI001_NA_REASON}
    if stable_id == "architecture.geoadaptnet_fbsp" and dataset == "bnci2014_004":
        return {"status": "not_applicable", "reason": FBSP_BNCI004_NA_REASON}
    if stable_id == "architecture.geoadaptnet" and dataset == "bnci2014_004":
        return {
            "status": "not_applicable",
            "reason": GEOADAPT_BNCI004_NA_REASON,
        }
    if dataset in DATASET_ORDER:
        return {"status": "eligible", "reason": None}
    raise KeyError(dataset)


def _eligibility_matrix(
    *,
    stable_ids: Sequence[str] = STABLE_IDS,
    dataset_order: Sequence[str] = DATASET_ORDER,
    include_public_na: bool = True,
) -> dict[str, dict[str, dict[str, Any]]]:
    datasets = list(dataset_order)
    if include_public_na and PUBLIC_NA_DATASET not in datasets:
        datasets.append(PUBLIC_NA_DATASET)
    return {
        dataset: {
            stable_id: _cell_status(stable_id, dataset) for stable_id in stable_ids
        }
        for dataset in datasets
    }


def _dataset_contracts() -> dict[str, Any]:
    contracts: dict[str, Any] = {}
    for dataset in DATASET_ORDER:
        spec = dataset_spec(dataset)
        subjects = DATASET_SUBJECTS[dataset]
        if tuple(spec.development_subjects) != subjects:
            raise GeoAdaptV2Error(f"{dataset} development subject contract drifted")
        contracts[dataset] = json.loads(
            _canonical_bytes(
                {
                    "dataset_spec": asdict(spec),
                    "subjects": list(subjects),
                    "folds": list(DATASET_FOLDS[dataset]),
                    "n_classes": int(spec.n_classes),
                    "protocol": spec.protocol,
                    "preprocessing": preprocessing_for_dataset(dataset),
                    "montage_profile": MONTAGE_PROFILE,
                }
            )
        )
    return contracts


def _registry_snapshot(stable_ids: Sequence[str]) -> dict[str, Any]:
    from .model_registry import record_by_id, validate_registry

    validate_registry()
    result: dict[str, Any] = {}
    for stable_id in stable_ids:
        record = record_by_id(str(stable_id))
        if (
            record.stable_id != stable_id
            or record.track != "procedure"
            or record.class_support != "binary"
        ):
            raise GeoAdaptV2Error(f"registry contract drift for {stable_id}")
        result[stable_id] = json.loads(_canonical_bytes(asdict(record)))
    return result


def _canonical_requirement_name(requirement: str) -> str:
    name = re.split(r"[\s<>=!~;\[]", requirement, maxsplit=1)[0]
    return re.sub(r"[-_.]+", "-", name.strip().lower())


def _exact_requirement_pin(requirement: str) -> tuple[str, str] | None:
    match = re.fullmatch(
        r"\s*([A-Za-z0-9_.-]+)(?:\[[A-Za-z0-9_,.-]+\])?=="
        r"([A-Za-z0-9_.+!-]+)\s*",
        requirement,
    )
    if match is None:
        return None
    return _canonical_requirement_name(match.group(1)), match.group(2)


def _validate_uv_project_contract(root: Path) -> dict[str, Any]:
    root = _assert_safe_path(root, leaf_kind="directory")
    pyproject = _assert_safe_path(root / "pyproject.toml", leaf_kind="file")
    lock = _assert_safe_path(root / "uv.lock", leaf_kind="file")
    try:
        project_document = tomllib.loads(_safe_read_text(pyproject, encoding="utf-8"))
        lock_document = tomllib.loads(_safe_read_text(lock, encoding="utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise GeoAdaptV2Error("UV project metadata is not valid TOML") from error
    project = project_document.get("project")
    if not isinstance(project, Mapping):
        raise GeoAdaptV2Error("pyproject.toml has no [project] table")
    project_name = _canonical_requirement_name(str(project.get("name", "")))
    project_version = str(project.get("version", ""))
    requires_python = project.get("requires-python")
    requirements = project.get("dependencies")
    if (
        not project_name
        or not project_version
        or not isinstance(requires_python, str)
        or not requires_python
        or not isinstance(requirements, list)
    ):
        raise GeoAdaptV2Error("pyproject project identity is incomplete")
    runtime_pins: dict[str, str] = {}
    for raw_requirement in requirements:
        parsed = _exact_requirement_pin(str(raw_requirement))
        if parsed is None:
            raise GeoAdaptV2Error(
                f"every runtime direct dependency must be exactly pinned: "
                f"{raw_requirement!r}"
            )
        name, version = parsed
        if name in runtime_pins:
            raise GeoAdaptV2Error(f"duplicate direct dependency pin: {name}")
        runtime_pins[name] = version
    missing = REQUIRED_RUNTIME_DIRECT_DEPENDENCIES - set(runtime_pins)
    if missing:
        raise GeoAdaptV2Error(
            "pyproject direct dependencies do not freeze the procedure stack: "
            f"{sorted(missing)}"
        )
    dependency_groups = project_document.get("dependency-groups")
    if not isinstance(dependency_groups, Mapping):
        raise GeoAdaptV2Error("pyproject.toml must freeze every dependency group")
    if not set(REQUIRED_DEPENDENCY_GROUPS).issubset(dependency_groups):
        raise GeoAdaptV2Error("pyproject omits the required test/docs groups")
    dependency_group_pins: dict[str, dict[str, str]] = {}
    for raw_group, group_requirements in dependency_groups.items():
        group = str(raw_group)
        if (
            not group
            or PATH_COMPONENT_RE.fullmatch(group) is None
            or not isinstance(group_requirements, list)
        ):
            raise GeoAdaptV2Error("pyproject dependency-group schema is invalid")
        pins: dict[str, str] = {}
        for raw_requirement in group_requirements:
            parsed = _exact_requirement_pin(str(raw_requirement))
            if parsed is None:
                raise GeoAdaptV2Error(
                    f"every {group} dependency must be exactly pinned: "
                    f"{raw_requirement!r}"
                )
            name, version = parsed
            if name in pins:
                raise GeoAdaptV2Error(f"duplicate {group} dependency pin: {name}")
            pins[name] = version
        dependency_group_pins[group] = pins
    for group, required in REQUIRED_DEPENDENCY_GROUPS.items():
        missing_group = required - set(dependency_group_pins[group])
        if missing_group:
            raise GeoAdaptV2Error(
                f"pyproject {group} dependency group is incomplete: "
                f"{sorted(missing_group)}"
            )
    test_pins = dependency_group_pins["test"]
    docs_pins = dependency_group_pins["docs"]
    if (
        not isinstance(lock_document.get("version"), int)
        or not isinstance(lock_document.get("revision"), int)
        or lock_document.get("requires-python") != requires_python
        or not isinstance(lock_document.get("package"), list)
    ):
        raise GeoAdaptV2Error("uv.lock header is incomplete or inconsistent")
    packages = lock_document["package"]
    locked_versions: dict[str, set[str]] = {}
    project_package: Mapping[str, Any] | None = None
    for package in packages:
        if not isinstance(package, Mapping):
            raise GeoAdaptV2Error("uv.lock contains a malformed package")
        name = _canonical_requirement_name(str(package.get("name", "")))
        version = str(package.get("version", ""))
        if not name or not version:
            raise GeoAdaptV2Error("uv.lock package identity is incomplete")
        locked_versions.setdefault(name, set()).add(version)
        source = package.get("source")
        if (
            name == project_name
            and version == project_version
            and isinstance(source, Mapping)
            and source.get("virtual") == "."
        ):
            if project_package is not None:
                raise GeoAdaptV2Error("uv.lock contains duplicate project packages")
            project_package = package
    if project_package is None:
        raise GeoAdaptV2Error("uv.lock does not contain the local project package")
    root_dependencies = {
        _canonical_requirement_name(str(value.get("name", "")))
        for value in project_package.get("dependencies", ())
        if isinstance(value, Mapping)
    }
    if root_dependencies != set(runtime_pins):
        raise GeoAdaptV2Error(
            "uv.lock project runtime dependency closure differs from pyproject"
        )
    root_dev_dependencies = project_package.get("dev-dependencies")
    if not isinstance(root_dev_dependencies, Mapping) or set(
        root_dev_dependencies
    ) != set(dependency_group_pins):
        raise GeoAdaptV2Error("uv.lock project dependency groups differ from pyproject")
    for group, pins in dependency_group_pins.items():
        locked_group = root_dev_dependencies[group]
        if not isinstance(locked_group, list):
            raise GeoAdaptV2Error(f"uv.lock {group} dependency group is malformed")
        locked_names = {
            _canonical_requirement_name(str(value.get("name", "")))
            for value in locked_group
            if isinstance(value, Mapping)
        }
        if locked_names != set(pins) or len(locked_group) != len(pins):
            raise GeoAdaptV2Error(
                f"uv.lock {group} dependency closure differs from pyproject"
            )
    project_metadata = project_package.get("metadata")
    requires_dist = (
        project_metadata.get("requires-dist")
        if isinstance(project_metadata, Mapping)
        else None
    )
    if not isinstance(requires_dist, list):
        raise GeoAdaptV2Error("uv.lock project metadata has no requires-dist ledger")
    locked_direct_pins = {
        _canonical_requirement_name(str(value.get("name", ""))): str(
            value.get("specifier", "")
        )
        for value in requires_dist
        if isinstance(value, Mapping)
    }
    if set(locked_direct_pins) != set(runtime_pins):
        raise GeoAdaptV2Error(
            "uv.lock requires-dist closure differs from pyproject dependencies"
        )
    for name in runtime_pins:
        if locked_direct_pins.get(name) != f"=={runtime_pins[name]}":
            raise GeoAdaptV2Error(
                f"uv.lock direct requirement metadata drifted for {name}"
            )
        if locked_versions.get(name) != {runtime_pins[name]}:
            raise GeoAdaptV2Error(
                f"uv.lock pin for {name} does not exactly match pyproject"
            )
    requires_dev = (
        project_metadata.get("requires-dev")
        if isinstance(project_metadata, Mapping)
        else None
    )
    if not isinstance(requires_dev, Mapping) or set(requires_dev) != set(
        dependency_group_pins
    ):
        raise GeoAdaptV2Error("uv.lock requires-dev groups differ from pyproject")
    for group, pins in dependency_group_pins.items():
        locked_metadata = requires_dev[group]
        if not isinstance(locked_metadata, list):
            raise GeoAdaptV2Error(
                f"uv.lock project metadata has no requires-dev.{group} ledger"
            )
        locked_pins = {
            _canonical_requirement_name(str(value.get("name", ""))): str(
                value.get("specifier", "")
            )
            for value in locked_metadata
            if isinstance(value, Mapping)
        }
        if set(locked_pins) != set(pins) or len(locked_metadata) != len(pins):
            raise GeoAdaptV2Error(
                f"uv.lock requires-dev.{group} differs from pyproject"
            )
        for name, version in pins.items():
            if locked_pins.get(name) != f"=={version}":
                raise GeoAdaptV2Error(
                    f"uv.lock {group} requirement metadata drifted for {name}"
                )
            if locked_versions.get(name) != {version}:
                raise GeoAdaptV2Error(
                    f"uv.lock {group} pin for {name} differs from pyproject"
                )
    available_names = set(locked_versions)
    dependency_graph: dict[str, set[str]] = {}
    runtime_dependency_graph: dict[str, set[str]] = {}
    for package in packages:
        package_name = _canonical_requirement_name(str(package["name"]))
        children = {
            _canonical_requirement_name(str(value.get("name", "")))
            for value in package.get("dependencies", ())
            if isinstance(value, Mapping)
        }
        runtime_dependency_graph.setdefault(package_name, set()).update(children)
        optional = package.get("optional-dependencies")
        if isinstance(optional, Mapping):
            children.update(
                _canonical_requirement_name(str(value.get("name", "")))
                for group in optional.values()
                if isinstance(group, list)
                for value in group
                if isinstance(value, Mapping)
            )
        children.discard("")
        if not children.issubset(available_names):
            raise GeoAdaptV2Error(
                f"uv.lock contains unresolved dependency edges for {package_name}"
            )
        dependency_graph.setdefault(package_name, set()).update(children)
    reachable = {project_name}
    frontier = [project_name]
    dependency_graph.setdefault(project_name, set()).update(
        name for pins in dependency_group_pins.values() for name in pins
    )
    while frontier:
        parent = frontier.pop()
        for child in dependency_graph.get(parent, ()):
            if child not in reachable:
                reachable.add(child)
                frontier.append(child)
    if reachable != available_names:
        raise GeoAdaptV2Error(
            f"uv.lock contains packages outside the project closure: "
            f"{sorted(available_names - reachable)}"
        )
    runtime_reachable = {project_name}
    runtime_frontier = [project_name]
    while runtime_frontier:
        parent = runtime_frontier.pop()
        for child in runtime_dependency_graph.get(parent, ()):
            if child not in runtime_reachable:
                runtime_reachable.add(child)
                runtime_frontier.append(child)
    runtime_package_versions: dict[str, str] = {}
    for name in sorted(runtime_reachable - {project_name}):
        versions = locked_versions.get(name)
        if versions is None or len(versions) != 1:
            raise GeoAdaptV2Error(
                f"runtime package {name} does not have one exact locked version"
            )
        runtime_package_versions[name] = next(iter(versions))
    return {
        "pyproject_sha256": _sha256_file(pyproject),
        "uv_lock_sha256": _sha256_file(lock),
        "required_runtime_direct_dependencies": sorted(
            REQUIRED_RUNTIME_DIRECT_DEPENDENCIES
        ),
        "runtime_direct_dependency_pins": {
            name: runtime_pins[name]
            for name in sorted(REQUIRED_RUNTIME_DIRECT_DEPENDENCIES)
        },
        "project_direct_dependency_pins": {
            name: runtime_pins[name] for name in sorted(runtime_pins)
        },
        "test_dependency_group": "test",
        "required_test_group_dependencies": sorted(REQUIRED_TEST_GROUP_DEPENDENCIES),
        "test_group_dependency_pins": {
            name: test_pins[name] for name in sorted(test_pins)
        },
        "required_docs_group_dependencies": sorted(REQUIRED_DOCS_GROUP_DEPENDENCIES),
        "docs_group_dependency_pins": {
            name: docs_pins[name] for name in sorted(docs_pins)
        },
        "dependency_group_pins": {
            group: {name: pins[name] for name in sorted(pins)}
            for group, pins in sorted(dependency_group_pins.items())
        },
        "runtime_locked_package_versions": runtime_package_versions,
        "runtime_locked_package_closure_sha256": _sha256_bytes(
            _canonical_bytes(runtime_package_versions)
        ),
        "locked_package_closure_sha256": _sha256_bytes(_canonical_bytes(packages)),
        "project_name": project_name,
        "project_version": project_version,
        "requires_python": requires_python,
        "uv_lock_version": int(lock_document["version"]),
        "uv_lock_revision": int(lock_document["revision"]),
    }


def _require_uv_virtual_environment(
    project_identity: Mapping[str, Any] | None = None,
) -> None:
    if Path(sys.prefix).resolve() == Path(sys.base_prefix).resolve():
        raise GeoAdaptV2Error(
            "production execution requires an isolated virtual environment"
        )
    virtual_environment = os.environ.get("VIRTUAL_ENV")
    if (
        virtual_environment is None
        or Path(virtual_environment).resolve() != Path(sys.prefix).resolve()
    ):
        raise GeoAdaptV2Error(
            "VIRTUAL_ENV must identify the active isolated UV environment"
        )
    try:
        result = subprocess.run(
            ["uv", "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as error:
        raise GeoAdaptV2Error("uv is required on PATH") from error
    if not result.stdout.strip().startswith("uv "):
        raise GeoAdaptV2Error("uv --version returned an unexpected identity")
    if project_identity is not None:
        project_root = Path(__file__).resolve().parents[1]
        try:
            subprocess.run(
                [
                    "uv",
                    "sync",
                    "--check",
                    "--active",
                    "--frozen",
                    "--no-default-groups",
                    "--no-install-project",
                    "--project",
                    str(project_root),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
        except (FileNotFoundError, subprocess.SubprocessError) as error:
            raise GeoAdaptV2Error(
                "active UV environment is not the exact frozen "
                "runtime-only project closure"
            ) from error
        pins = project_identity.get("runtime_locked_package_versions")
        if not isinstance(pins, Mapping):
            raise GeoAdaptV2Error("UV project identity has no exact runtime closure")
        installed: dict[str, str] = {}
        for distribution in importlib.metadata.distributions():
            name = _canonical_requirement_name(
                str(distribution.metadata.get("Name", ""))
            )
            version = str(distribution.version)
            if not name or not version or name in installed:
                raise GeoAdaptV2Error(
                    "active UV environment has a malformed/duplicate distribution"
                )
            installed[name] = version
        expected_pins = {str(name): str(version) for name, version in pins.items()}
        if installed != expected_pins:
            raise GeoAdaptV2Error(
                "active UV environment differs from the exact runtime-only "
                "lock closure; test/docs/extras must be excluded"
            )


def _source_identity(root: Path | None = None) -> dict[str, str]:
    project_root = (
        Path(__file__).resolve().parents[1]
        if root is None
        else _assert_safe_path(root, leaf_kind="directory")
    )
    result: dict[str, str] = {}
    for relative in SOURCE_FILES:
        path = project_root / relative
        try:
            source_status = _anchored_lstat(path)
        except FileNotFoundError:
            raise GeoAdaptV2Error(f"source-closure file is absent: {relative}")
        if not stat.S_ISREG(source_status.st_mode):
            raise GeoAdaptV2Error(
                f"source-closure path is not a regular file: {relative}"
            )
        _require_single_link(source_status, path)
        result[relative] = _sha256_file(path)
    return result


def _nvidia_inventory() -> list[dict[str, Any]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    rows: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", 4)]
        if len(fields) != 5:
            raise GeoAdaptV2Error("nvidia-smi returned malformed inventory")
        rows.append(
            {
                "index": int(fields[0]),
                "uuid": fields[1],
                "name": fields[2],
                "driver_version": fields[3],
                "memory_total_mib": int(fields[4]),
            }
        )
    return rows


def _nvidia_csv_rows(arguments: Sequence[str]) -> list[list[str]]:
    try:
        result = subprocess.run(
            ["nvidia-smi", *arguments, "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as error:
        raise GeoAdaptV2Error(
            "nvidia-smi cooperative resource probe failed closed"
        ) from error
    return [
        [field.strip() for field in row]
        for row in csv.reader(io.StringIO(result.stdout))
        if row and any(field.strip() for field in row)
    ]


def _cooperative_gpu_status(
    gpu_uuid: str,
    *,
    allow_active_owner: bool,
) -> dict[str, Any]:
    """Reject a selected GPU that appears busy for another workstation user."""

    canonical = "GPU-" + str(gpu_uuid)[4:].lower()
    gpu_rows = _nvidia_csv_rows(["--query-gpu=uuid,utilization.gpu,memory.used"])
    matches = [
        row
        for row in gpu_rows
        if len(row) == 3
        and GPU_UUID_RE.fullmatch(row[0])
        and "GPU-" + row[0][4:].lower() == canonical
    ]
    if len(matches) != 1:
        raise GeoAdaptV2Error(
            "selected GPU UUID is absent or duplicated in cooperative probe"
        )
    try:
        utilization = float(matches[0][1])
        memory_used = float(matches[0][2])
    except ValueError as error:
        raise GeoAdaptV2Error(
            "nvidia-smi returned nonnumeric GPU utilization/memory"
        ) from error
    if (
        not math.isfinite(utilization)
        or not 0.0 <= utilization <= 100.0
        or not math.isfinite(memory_used)
        or memory_used < 0.0
    ):
        raise GeoAdaptV2Error("nvidia-smi returned an invalid GPU resource value")
    process_rows = _nvidia_csv_rows(
        [
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
        ]
    )
    foreign: list[dict[str, Any]] = []
    for row in process_rows:
        if len(row) != 4:
            raise GeoAdaptV2Error("nvidia-smi returned a malformed process row")
        if not GPU_UUID_RE.fullmatch(row[0]):
            raise GeoAdaptV2Error("nvidia-smi returned an invalid process GPU UUID")
        row_uuid = "GPU-" + row[0][4:].lower()
        if row_uuid != canonical:
            continue
        try:
            pid = int(row[1])
            used = float(row[3])
        except ValueError as error:
            raise GeoAdaptV2Error(
                "nvidia-smi returned an invalid GPU process identity"
            ) from error
        if pid <= 0 or not row[2] or not math.isfinite(used) or used < 0.0:
            raise GeoAdaptV2Error("nvidia-smi returned an invalid GPU process resource")
        if pid != os.getpid():
            foreign.append(
                {
                    "pid": pid,
                    "process_name": row[2],
                    "used_gpu_memory_mib": float(used),
                }
            )
    utilization_limit = 100.0 if allow_active_owner else 10.0
    safe = not foreign and utilization <= utilization_limit
    return {
        "schema": "ieee-mi-geoadapt-v2-cooperative-gpu-status-v1",
        "gpu_uuid": canonical,
        "utilization_percent": float(utilization),
        "memory_used_mib": float(memory_used),
        "foreign_processes": foreign,
        "allow_active_owner": bool(allow_active_owner),
        "utilization_limit_percent": float(utilization_limit),
        "safe": bool(safe),
        "reason": (
            "resources available"
            if safe
            else (
                "foreign compute process detected"
                if foreign
                else "GPU utilization exceeds the cooperative limit"
            )
        ),
    }


def _environment_identity() -> dict[str, Any]:
    packages = sorted(
        (
            str(distribution.metadata.get("Name", "unknown")).lower(),
            str(distribution.version),
        )
        for distribution in importlib.metadata.distributions()
    )
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
    try:
        import torch

        torch_identity: dict[str, Any] = {
            "version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        }
    except ImportError:
        torch_identity = {"version": None, "cuda_build": None, "cudnn": None}
    return {
        "schema": "ieee-mi-portable-compute-contract-v1",
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "uv_version": uv_version,
        "packages": [[name, version] for name, version in packages],
        "packages_sha256": _sha256_bytes(_canonical_bytes(packages)),
        "torch": torch_identity,
        "deterministic_torch_contract": copy.deepcopy(
            dict(DETERMINISTIC_TORCH_CONTRACT)
        ),
    }


def _feature_contract() -> dict[str, Any]:
    from scipy import signal

    coefficients = np.stack(
        [
            signal.firwin(
                FIR_TAPS,
                (low, high),
                pass_zero=False,
                fs=128.0,
                window=FIR_WINDOW,
                scale=True,
            )
            for low, high in FILTER_BANDS_HZ
        ],
        axis=0,
    ).astype(np.float64, copy=False)
    return {
        "schema": "ieee-mi-geoadapt-four-band-spd-view-v1",
        "input": (
            "harmonized-v2 broadband trials after split-specific source-only "
            "channel scaling"
        ),
        "sfreq_hz": 128.0,
        "bands_hz": [list(value) for value in FILTER_BANDS_HZ],
        "fir_taps": FIR_TAPS,
        "fir_window": FIR_WINDOW,
        "fir_design": "scipy.signal.firwin bandpass scale=True",
        "application": "single symmetric FFT convolution",
        "boundary": "reflection pad floor(fir_taps/2), then valid convolution",
        "post_filter_epoch_demean": True,
        "coefficient_sha256": _array_sha256(coefficients),
        "covariance": {
            "computation_dtype": "float64",
            "demean": True,
            "normalization": "divide by n_times",
            "shrinkage": COVARIANCE_SHRINKAGE,
            "target": "trace(C)/n_channels * identity",
            "scale_aware_positive_floor": True,
            "output_dtype": "float32",
        },
        "fit_scope": {
            "selection_scaler": "training rows only",
            "refit_scaler": "training plus validation rows",
            "filter_and_covariance": "deterministic per trial; no cross-trial fit",
            "held_out_outcomes": "never used",
        },
    }


def _cache_and_split_identity(
    cache_root: Path,
    contracts: Mapping[str, Any],
    *,
    cache_loader: Callable[..., Mapping[str, Any]] | None = None,
    splitter: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from .data import load_subject_cache, split_indices

    load = load_subject_cache if cache_loader is None else cache_loader
    split = split_indices if splitter is None else splitter
    caches: dict[str, Any] = {}
    splits: dict[str, Any] = {}
    for dataset, contract in contracts.items():
        for raw_subject in contract["subjects"]:
            subject = int(raw_subject)
            data = load(
                dataset,
                subject,
                cache_root=cache_root,
                montage_profile=MONTAGE_PROFILE,
            )
            identity = copy.deepcopy(dict(data["identity"]))
            digest = identity.get("array_sha256")
            if not _is_sha256(digest):
                raise GeoAdaptV2Error(
                    f"cache {dataset} S{subject} has no valid array digest"
                )
            caches[_subject_key(dataset, subject)] = identity
            for raw_fold in contract["folds"]:
                fold = int(raw_fold)
                train, validation, test = split(
                    dataset,
                    np.asarray(data["y"]),
                    np.asarray(data["sessions"]),
                    np.asarray(data["runs"]),
                    fold=fold,
                    subject=subject,
                )
                splits[_split_key(dataset, subject, fold)] = _one_split_identity(
                    dataset=dataset,
                    subject=subject,
                    fold=fold,
                    cache_array_sha256=str(digest),
                    trial_count=len(data["y"]),
                    train_rows=np.asarray(train, dtype=np.int64),
                    validation_rows=np.asarray(validation, dtype=np.int64),
                    test_rows=np.asarray(test, dtype=np.int64),
                )
    return caches, splits


def plan_sha256(plan: Mapping[str, Any]) -> str:
    value = copy.deepcopy(dict(plan))
    value.pop("plan_sha256", None)
    return _sha256_bytes(_canonical_bytes(value))


def assemble_plan(
    *,
    dataset_contracts: Mapping[str, Any],
    stable_configs: Mapping[str, Mapping[str, Any]],
    eligibility: Mapping[str, Mapping[str, Mapping[str, Any]]],
    seeds: Sequence[int],
    cache_identity: Mapping[str, Any],
    split_identity: Mapping[str, Any],
    source_identity: Mapping[str, str],
    environment_identity: Mapping[str, Any],
    project_identity: Mapping[str, Any],
    feature_contract: Mapping[str, Any],
    registry_records: Mapping[str, Any] | None = None,
    minimum_free_gib: float = DEFAULT_MIN_FREE_GIB,
    worker_cpu_threads: int = DEFAULT_WORKER_CPU_THREADS,
) -> dict[str, Any]:
    """Pure canonical plan constructor used by production and tiny tests."""

    configs = copy.deepcopy(dict(stable_configs))
    if (
        not configs
        or len(configs) != len(set(configs))
        or any(
            not isinstance(value, str) or not PATH_COMPONENT_RE.fullmatch(value)
            for value in configs
        )
    ):
        raise ValueError("stable configs must be nonempty and unique")
    raw_seeds = tuple(seeds)
    if (
        not raw_seeds
        or any(not _is_exact_int(value, minimum=0) for value in raw_seeds)
        or len(raw_seeds) != len(set(raw_seeds))
    ):
        raise ValueError("seeds must be unique nonnegative integers")
    seed_values = tuple(raw_seeds)
    if not _is_exact_float(minimum_free_gib, minimum=0.0):
        raise ValueError("minimum_free_gib must be finite and nonnegative")
    if not _is_exact_int(worker_cpu_threads, minimum=1):
        raise ValueError("worker_cpu_threads must be positive")
    registry = (
        _registry_snapshot(tuple(configs))
        if registry_records is None
        else copy.deepcopy(dict(registry_records))
    )
    if set(registry) != set(configs) or any(
        not isinstance(record, Mapping)
        or record.get("stable_id") != stable_id
        or record.get("track") != "procedure"
        or record.get("class_support") != "binary"
        for stable_id, record in registry.items()
    ):
        raise ValueError("registry snapshots do not match stable configs")

    contracts: dict[str, Any] = {}
    for dataset, raw_contract in dataset_contracts.items():
        if not isinstance(dataset, str) or not PATH_COMPONENT_RE.fullmatch(dataset):
            raise ValueError(f"unsafe dataset identity {dataset!r}")
        contract = copy.deepcopy(dict(raw_contract))
        raw_subjects = tuple(contract["subjects"])
        raw_folds = tuple(contract["folds"])
        if (
            not raw_subjects
            or not raw_folds
            or len(raw_subjects) != len(set(raw_subjects))
            or len(raw_folds) != len(set(raw_folds))
            or any(not _is_exact_int(value, minimum=1) for value in raw_subjects)
            or any(not _is_exact_int(value, minimum=0) for value in raw_folds)
            or not _is_exact_int(contract["n_classes"])
            or contract["n_classes"] != 2
        ):
            raise ValueError(f"{dataset} requires unique binary subject/fold contracts")
        contract["subjects"] = list(raw_subjects)
        contract["folds"] = list(raw_folds)
        contract["n_classes"] = 2
        contracts[dataset] = json.loads(_canonical_bytes(contract))
    if not contracts:
        raise ValueError("at least one eligible dataset is required")

    matrix = copy.deepcopy(
        {
            str(dataset): {
                str(stable_id): dict(cell) for stable_id, cell in cells.items()
            }
            for dataset, cells in eligibility.items()
        }
    )
    required_matrix_datasets = set(contracts)
    if set(configs) == set(STABLE_IDS):
        required_matrix_datasets.add(PUBLIC_NA_DATASET)
    if not required_matrix_datasets.issubset(matrix):
        raise ValueError("eligibility matrix omits a required dataset")
    for dataset in contracts:
        if set(matrix[dataset]) != set(configs):
            raise ValueError(f"{dataset} eligibility does not cover stable configs")
        if not any(
            cell.get("status") == "eligible" for cell in matrix[dataset].values()
        ):
            raise ValueError(f"{dataset} has no eligible procedure")
    for dataset, cells in matrix.items():
        for stable_id, cell in cells.items():
            if stable_id not in configs:
                raise ValueError("eligibility references an unknown stable config")
            if set(cell) != {"status", "reason"}:
                raise ValueError("eligibility cells require exact status/reason keys")
            if cell["status"] not in {"eligible", "not_applicable"}:
                raise ValueError("eligibility status is invalid")
            if cell["status"] == "eligible" and cell["reason"] is not None:
                raise ValueError("eligible cells cannot carry an N/A reason")
            if cell["status"] == "not_applicable" and not isinstance(
                cell["reason"], str
            ):
                raise ValueError("N/A cells require a reason")

    expected_cache_keys = {
        _subject_key(dataset, subject)
        for dataset, contract in contracts.items()
        for subject in contract["subjects"]
    }
    expected_split_keys = {
        _split_key(dataset, subject, fold)
        for dataset, contract in contracts.items()
        for subject in contract["subjects"]
        for fold in contract["folds"]
    }
    caches = copy.deepcopy(dict(cache_identity))
    splits = copy.deepcopy(dict(split_identity))
    if set(caches) != expected_cache_keys:
        raise ValueError("cache identities differ from the subject grid")
    if set(splits) != expected_split_keys:
        raise ValueError("split identities differ from the subject/fold grid")
    for key, cache in caches.items():
        if not isinstance(cache, Mapping) or not _is_sha256(cache.get("array_sha256")):
            raise ValueError(f"cache identity {key} is invalid")
        if "shape" in cache and (
            not isinstance(cache["shape"], list)
            or any(not _is_exact_int(value, minimum=1) for value in cache["shape"])
        ):
            raise ValueError(f"cache identity {key} has invalid shape scalars")
    for key, split in splits.items():
        if not isinstance(split, Mapping):
            raise ValueError(f"split identity {key} is invalid")
        if (
            not isinstance(split.get("dataset"), str)
            or not _is_exact_int(split.get("subject"), minimum=1)
            or not _is_exact_int(split.get("fold"), minimum=0)
            or not _is_exact_int(split.get("trial_count"), minimum=1)
        ):
            raise ValueError(f"split identity {key} has invalid scalar types")
        partitions = split.get("partitions")
        if not isinstance(partitions, Mapping) or set(partitions) != {
            "train",
            "validation",
            "source",
            "test",
        }:
            raise ValueError(f"split identity {key} has invalid partitions")
        if any(
            not isinstance(partitions[name], Mapping)
            or not _is_exact_int(partitions[name].get("count"), minimum=1)
            for name in ("train", "validation", "source", "test")
        ):
            raise ValueError(f"split identity {key} has invalid partition counts")
        counts = {
            name: partitions[name]["count"]
            for name in ("train", "validation", "source", "test")
        }
        if (
            counts["source"] != counts["train"] + counts["validation"]
            or counts["source"] + counts["test"] != split["trial_count"]
        ):
            raise ValueError(f"split identity {key} does not cover its cache")
        for partition in partitions.values():
            if not _is_sha256(partition["rows_sha256"]):
                raise ValueError(f"split identity {key} has invalid row identity")

    n_jobs = 0
    for dataset, contract in contracts.items():
        eligible_count = sum(
            matrix[dataset][stable_id]["status"] == "eligible" for stable_id in configs
        )
        n_jobs += (
            len(contract["subjects"])
            * len(contract["folds"])
            * len(seed_values)
            * eligible_count
        )
    payload: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "created_at": _utc_now(),
        "purpose": "geoadapt_family_harmonized_v2_procedure_predictions",
        "track": TRACK,
        "evidence_scope": EVIDENCE_SCOPE,
        "confirmation_evidence": False,
        "score_blind": True,
        "dataset_order": list(contracts),
        "datasets": contracts,
        "stable_id_order": list(configs),
        "stable_configs": configs,
        "registry_records": registry,
        "eligibility": matrix,
        "seeds": list(seed_values),
        "n_jobs": int(n_jobs),
        "cache_identity": caches,
        "split_identity": splits,
        "feature_contract": copy.deepcopy(dict(feature_contract)),
        "channel_scaling": json.loads(_canonical_bytes(CHANNEL_SCALING)),
        "source_identity": copy.deepcopy(dict(source_identity)),
        "environment_identity": copy.deepcopy(dict(environment_identity)),
        "project_identity": copy.deepcopy(dict(project_identity)),
        "execution_config": {
            "minimum_free_gib": float(minimum_free_gib),
            "worker_cpu_threads": int(worker_cpu_threads),
            "prediction_calls_per_job": 1,
            "test_outcomes_available_to_backend": False,
            "stale_foreign_claim_timeout_seconds": 0.0,
            "claim_recovery_policy": (
                "reclaim_only_definitively_dead_local_owner;"
                "freeze_foreign_unknown_or_malformed"
            ),
            "deterministic_torch": copy.deepcopy(dict(DETERMINISTIC_TORCH_CONTRACT)),
        },
    }
    payload["plan_sha256"] = plan_sha256(payload)
    return payload


def build_plan(
    *,
    cache_root: Path,
    minimum_free_gib: float = DEFAULT_MIN_FREE_GIB,
    worker_cpu_threads: int = DEFAULT_WORKER_CPU_THREADS,
) -> dict[str, Any]:
    """Build the exact 6,495-job production plan without opening N/A cells."""

    project_root = Path(__file__).resolve().parents[1]
    project = _validate_uv_project_contract(project_root)
    _require_uv_virtual_environment(project)
    contracts = _dataset_contracts()
    cache_root = _assert_safe_path(cache_root, leaf_kind="directory")
    caches, splits = _cache_and_split_identity(cache_root, contracts)
    plan = assemble_plan(
        dataset_contracts=contracts,
        stable_configs=STABLE_CONFIGS,
        eligibility=_eligibility_matrix(),
        seeds=SEEDS,
        cache_identity=caches,
        split_identity=splits,
        source_identity=_source_identity(project_root),
        environment_identity=_environment_identity(),
        project_identity=project,
        feature_contract=_feature_contract(),
        minimum_free_gib=minimum_free_gib,
        worker_cpu_threads=worker_cpu_threads,
    )
    validate_production_semantics(plan)
    return plan


def _validate_plan_shape(plan: Mapping[str, Any]) -> None:
    required = {
        "schema",
        "created_at",
        "purpose",
        "track",
        "evidence_scope",
        "confirmation_evidence",
        "score_blind",
        "dataset_order",
        "datasets",
        "stable_id_order",
        "stable_configs",
        "registry_records",
        "eligibility",
        "seeds",
        "n_jobs",
        "cache_identity",
        "split_identity",
        "feature_contract",
        "channel_scaling",
        "source_identity",
        "environment_identity",
        "project_identity",
        "execution_config",
        "plan_sha256",
    }
    if set(plan) != required:
        raise GeoAdaptV2Error("plan has an invalid exact schema")
    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("purpose") != "geoadapt_family_harmonized_v2_procedure_predictions"
        or plan.get("track") != TRACK
        or plan.get("evidence_scope") != EVIDENCE_SCOPE
        or plan.get("confirmation_evidence") is not False
        or plan.get("score_blind") is not True
        or not _is_utc_timestamp(plan.get("created_at"))
        or not _is_exact_int(plan.get("n_jobs"), minimum=1)
        or plan.get("plan_sha256") != plan_sha256(plan)
    ):
        raise GeoAdaptV2Error("plan identity is invalid")
    dataset_order = plan.get("dataset_order")
    datasets = plan.get("datasets")
    stable_order = plan.get("stable_id_order")
    stable_configs = plan.get("stable_configs")
    registry_records = plan.get("registry_records")
    eligibility = plan.get("eligibility")
    seeds = plan.get("seeds")
    execution = plan.get("execution_config")
    if (
        not isinstance(dataset_order, list)
        or not dataset_order
        or len(dataset_order) != len(set(dataset_order))
        or any(
            not isinstance(value, str) or not PATH_COMPONENT_RE.fullmatch(value)
            for value in dataset_order
        )
        or not isinstance(datasets, Mapping)
        or list(datasets) != dataset_order
        or not isinstance(stable_order, list)
        or not stable_order
        or len(stable_order) != len(set(stable_order))
        or any(
            not isinstance(value, str) or not PATH_COMPONENT_RE.fullmatch(value)
            for value in stable_order
        )
        or not isinstance(stable_configs, Mapping)
        or list(stable_configs) != stable_order
        or any(
            stable_id in STABLE_CONFIGS
            and (
                stable_configs[stable_id] != STABLE_CONFIGS[stable_id]
                or not _exact_scalar_types_match(
                    stable_configs[stable_id],
                    STABLE_CONFIGS[stable_id],
                )
            )
            for stable_id in stable_order
        )
        or not isinstance(registry_records, Mapping)
        or set(registry_records) != set(stable_order)
        or any(
            not isinstance(record, Mapping)
            or record.get("stable_id") != stable_id
            or record.get("track") != "procedure"
            or record.get("class_support") != "binary"
            for stable_id, record in registry_records.items()
        )
        or not isinstance(eligibility, Mapping)
        or not isinstance(seeds, list)
        or not seeds
        or len(seeds) != len(set(seeds))
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in seeds
        )
        or not isinstance(execution, Mapping)
        or set(execution)
        != {
            "minimum_free_gib",
            "worker_cpu_threads",
            "prediction_calls_per_job",
            "test_outcomes_available_to_backend",
            "stale_foreign_claim_timeout_seconds",
            "claim_recovery_policy",
            "deterministic_torch",
        }
        or not _is_exact_int(execution["worker_cpu_threads"], minimum=1)
        or not _is_exact_float(execution["minimum_free_gib"], minimum=0.0)
        or not _is_exact_int(execution["prediction_calls_per_job"], minimum=1)
        or execution["prediction_calls_per_job"] != 1
        or execution["test_outcomes_available_to_backend"] is not False
        or not _is_exact_float(
            execution["stale_foreign_claim_timeout_seconds"],
            minimum=0.0,
        )
        or execution["stale_foreign_claim_timeout_seconds"] != 0.0
        or execution["claim_recovery_policy"]
        != (
            "reclaim_only_definitively_dead_local_owner;"
            "freeze_foreign_unknown_or_malformed"
        )
        or execution["deterministic_torch"] != DETERMINISTIC_TORCH_CONTRACT
        or not _exact_scalar_types_match(
            execution["deterministic_torch"],
            DETERMINISTIC_TORCH_CONTRACT,
        )
    ):
        raise GeoAdaptV2Error("plan grid/execution contract is invalid")
    for dataset in dataset_order:
        contract = datasets.get(dataset)
        cells = eligibility.get(dataset)
        subjects = contract.get("subjects") if isinstance(contract, Mapping) else None
        folds = contract.get("folds") if isinstance(contract, Mapping) else None
        if (
            not isinstance(contract, Mapping)
            or not isinstance(subjects, list)
            or not subjects
            or len(subjects) != len(set(subjects))
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in subjects
            )
            or not isinstance(folds, list)
            or not folds
            or len(folds) != len(set(folds))
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in folds
            )
            or not _is_exact_int(contract.get("n_classes"))
            or contract["n_classes"] != 2
            or not isinstance(cells, Mapping)
            or set(cells) != set(stable_order)
        ):
            raise GeoAdaptV2Error(f"plan dataset contract is invalid for {dataset}")
        for stable_id in stable_order:
            cell = cells[stable_id]
            if (
                not isinstance(cell, Mapping)
                or set(cell) != {"status", "reason"}
                or cell["status"] not in {"eligible", "not_applicable"}
                or (cell["status"] == "eligible" and cell["reason"] is not None)
                or (
                    cell["status"] == "not_applicable"
                    and not isinstance(cell["reason"], str)
                )
            ):
                raise GeoAdaptV2Error("plan eligibility contract is invalid")
    jobs = tuple(iter_jobs(plan, validate_plan=False))
    if len(jobs) != plan["n_jobs"] or len({job.job_id for job in jobs}) != len(jobs):
        raise GeoAdaptV2Error("plan job Cartesian product is invalid")


def _validate_production_semantics_impl(plan: Mapping[str, Any]) -> None:
    """Assert the literal, non-negotiable GeoAdapt harmonized-v2 experiment."""

    _validate_plan_shape(plan)
    expected_contracts = _dataset_contracts()
    expected_eligibility = _eligibility_matrix()
    expected_registry = _registry_snapshot(STABLE_IDS)
    if plan["dataset_order"] != list(DATASET_ORDER):
        raise GeoAdaptV2Error("production dataset roster/order drifted")
    if plan["datasets"] != expected_contracts:
        raise GeoAdaptV2Error("production preprocessing/subject/fold contract drifted")
    if plan["stable_id_order"] != list(STABLE_IDS):
        raise GeoAdaptV2Error("production stable-ID roster/order drifted")
    if plan["stable_configs"] != STABLE_CONFIGS:
        raise GeoAdaptV2Error("literal production model/trainer configuration drifted")
    if plan["seeds"] != list(SEEDS):
        raise GeoAdaptV2Error("production five-seed roster drifted")
    if plan["eligibility"] != expected_eligibility:
        raise GeoAdaptV2Error("production eligibility/N/A matrix drifted")
    if plan["registry_records"] != expected_registry:
        raise GeoAdaptV2Error("production registry snapshot drifted")
    for name, expected_value in (
        ("datasets", expected_contracts),
        ("stable_configs", STABLE_CONFIGS),
        ("seeds", list(SEEDS)),
        ("eligibility", expected_eligibility),
        ("registry_records", expected_registry),
        ("channel_scaling", json.loads(_canonical_bytes(CHANNEL_SCALING))),
        ("feature_contract", _feature_contract()),
    ):
        _require_exact_scalar_types(
            plan[name],
            expected_value,
            path=f"plan.{name}",
        )
    for stable_id, record in expected_registry.items():
        decisions = {
            str(value["dataset"]): {
                "status": "eligible" if bool(value["eligible"]) else "not_applicable",
                "reason": value["reason"],
            }
            for value in record["dataset_eligibility"]
        }
        expected_cells = {
            dataset: expected_eligibility[dataset][stable_id]
            for dataset in (*DATASET_ORDER, PUBLIC_NA_DATASET)
        }
        if decisions != expected_cells:
            raise GeoAdaptV2Error(f"registry/plan eligibility conflict for {stable_id}")
    if plan["channel_scaling"] != json.loads(_canonical_bytes(CHANNEL_SCALING)):
        raise GeoAdaptV2Error("production channel-scaling contract drifted")
    if plan["feature_contract"] != _feature_contract():
        raise GeoAdaptV2Error("production four-band SPD preprocessing drifted")
    source_identity = plan["source_identity"]
    if (
        not isinstance(source_identity, Mapping)
        or set(source_identity) != set(SOURCE_FILES)
        or not all(_is_sha256(value) for value in source_identity.values())
    ):
        raise GeoAdaptV2Error("production source closure is incomplete")
    project_identity = plan["project_identity"]
    expected_project_keys = {
        "pyproject_sha256",
        "uv_lock_sha256",
        "required_runtime_direct_dependencies",
        "runtime_direct_dependency_pins",
        "project_direct_dependency_pins",
        "test_dependency_group",
        "required_test_group_dependencies",
        "test_group_dependency_pins",
        "required_docs_group_dependencies",
        "docs_group_dependency_pins",
        "dependency_group_pins",
        "runtime_locked_package_versions",
        "runtime_locked_package_closure_sha256",
        "locked_package_closure_sha256",
        "project_name",
        "project_version",
        "requires_python",
        "uv_lock_version",
        "uv_lock_revision",
    }
    if (
        not isinstance(project_identity, Mapping)
        or set(project_identity) != expected_project_keys
        or project_identity.get("required_runtime_direct_dependencies")
        != sorted(REQUIRED_RUNTIME_DIRECT_DEPENDENCIES)
        or set(project_identity.get("runtime_direct_dependency_pins", {}))
        != REQUIRED_RUNTIME_DIRECT_DEPENDENCIES
        or not isinstance(
            project_identity.get("project_direct_dependency_pins"), Mapping
        )
        or not REQUIRED_RUNTIME_DIRECT_DEPENDENCIES.issubset(
            set(project_identity.get("project_direct_dependency_pins", {}))
        )
        or project_identity.get("test_dependency_group") != "test"
        or project_identity.get("required_test_group_dependencies")
        != sorted(REQUIRED_TEST_GROUP_DEPENDENCIES)
        or set(project_identity.get("test_group_dependency_pins", {}))
        != REQUIRED_TEST_GROUP_DEPENDENCIES
        or project_identity.get("required_docs_group_dependencies")
        != sorted(REQUIRED_DOCS_GROUP_DEPENDENCIES)
        or set(project_identity.get("docs_group_dependency_pins", {}))
        != REQUIRED_DOCS_GROUP_DEPENDENCIES
        or not isinstance(project_identity.get("dependency_group_pins"), Mapping)
        or not set(REQUIRED_DEPENDENCY_GROUPS).issubset(
            set(project_identity.get("dependency_group_pins", {}))
        )
        or project_identity["dependency_group_pins"].get("test")
        != project_identity.get("test_group_dependency_pins")
        or project_identity["dependency_group_pins"].get("docs")
        != project_identity.get("docs_group_dependency_pins")
        or not isinstance(
            project_identity.get("runtime_locked_package_versions"), Mapping
        )
        or not project_identity.get("runtime_locked_package_versions")
        or any(
            not isinstance(value, str) or not value
            for value in project_identity.get(
                "runtime_direct_dependency_pins", {}
            ).values()
        )
        or any(
            not isinstance(name, str)
            or not name
            or not isinstance(value, str)
            or not value
            for name, value in project_identity.get(
                "project_direct_dependency_pins", {}
            ).items()
        )
        or any(
            not isinstance(value, str) or not value
            for value in project_identity.get("test_group_dependency_pins", {}).values()
        )
        or any(
            not isinstance(value, str) or not value
            for value in project_identity.get("docs_group_dependency_pins", {}).values()
        )
        or any(
            not isinstance(group, str)
            or PATH_COMPONENT_RE.fullmatch(group) is None
            or not isinstance(pins, Mapping)
            or any(
                not isinstance(name, str)
                or not name
                or not isinstance(version, str)
                or not version
                for name, version in pins.items()
            )
            for group, pins in project_identity.get("dependency_group_pins", {}).items()
        )
        or any(
            not isinstance(name, str)
            or not name
            or not isinstance(version, str)
            or not version
            for name, version in project_identity.get(
                "runtime_locked_package_versions", {}
            ).items()
        )
        or not _is_sha256(project_identity.get("pyproject_sha256"))
        or not _is_sha256(project_identity.get("uv_lock_sha256"))
        or not _is_sha256(project_identity.get("locked_package_closure_sha256"))
        or not _is_sha256(project_identity.get("runtime_locked_package_closure_sha256"))
        or project_identity.get("runtime_locked_package_closure_sha256")
        != _sha256_bytes(
            _canonical_bytes(project_identity.get("runtime_locked_package_versions"))
        )
        or not isinstance(project_identity.get("project_name"), str)
        or not project_identity.get("project_name")
        or not isinstance(project_identity.get("project_version"), str)
        or not project_identity.get("project_version")
        or not isinstance(project_identity.get("requires_python"), str)
        or not project_identity.get("requires_python")
        or not _is_exact_int(project_identity.get("uv_lock_version"), minimum=1)
        or not _is_exact_int(project_identity.get("uv_lock_revision"), minimum=0)
    ):
        raise GeoAdaptV2Error("production UV project/lock contract is incomplete")
    environment = plan["environment_identity"]
    expected_environment_keys = {
        "schema",
        "python_version",
        "python_implementation",
        "platform_system",
        "platform_machine",
        "uv_version",
        "packages",
        "packages_sha256",
        "torch",
        "deterministic_torch_contract",
    }
    if (
        not isinstance(environment, Mapping)
        or set(environment) != expected_environment_keys
        or environment.get("schema") != "ieee-mi-portable-compute-contract-v1"
        or environment.get("deterministic_torch_contract")
        != DETERMINISTIC_TORCH_CONTRACT
        or any(
            not isinstance(environment.get(key), str) or not environment.get(key)
            for key in (
                "python_version",
                "python_implementation",
                "platform_system",
                "platform_machine",
            )
        )
        or not isinstance(environment.get("uv_version"), str)
        or not environment.get("uv_version", "").startswith("uv ")
        or not isinstance(environment.get("packages"), list)
        or environment.get("packages_sha256")
        != _sha256_bytes(_canonical_bytes(environment.get("packages")))
        or not isinstance(environment.get("torch"), Mapping)
        or set(environment.get("torch", {})) != {"version", "cuda_build", "cudnn"}
        or not isinstance(environment["torch"].get("version"), str)
        or not environment["torch"].get("version")
        or (
            environment["torch"].get("cuda_build") is not None
            and not isinstance(environment["torch"].get("cuda_build"), str)
        )
        or (
            environment["torch"].get("cudnn") is not None
            and not _is_exact_int(environment["torch"].get("cudnn"), minimum=1)
        )
    ):
        raise GeoAdaptV2Error(
            "plan compute identity is not a portable deterministic contract"
        )
    package_versions: dict[str, str] = {}
    for entry in environment["packages"]:
        if (
            not isinstance(entry, list)
            or len(entry) != 2
            or not all(isinstance(value, str) and value for value in entry)
        ):
            raise GeoAdaptV2Error("portable package inventory is malformed")
        name = _canonical_requirement_name(entry[0])
        if not name or name in package_versions:
            raise GeoAdaptV2Error(
                "portable package inventory has malformed/duplicate names"
            )
        package_versions[name] = entry[1]
    if package_versions != project_identity["runtime_locked_package_versions"]:
        raise GeoAdaptV2Error(
            "portable installed versions differ from the exact runtime-only "
            "UV lock closure"
        )
    execution = plan["execution_config"]
    if (
        not _is_exact_float(execution["minimum_free_gib"], minimum=0.0)
        or execution["minimum_free_gib"] != DEFAULT_MIN_FREE_GIB
        or not _is_exact_int(execution["worker_cpu_threads"], minimum=1)
        or not _is_exact_int(execution["prediction_calls_per_job"], minimum=1)
        or execution["prediction_calls_per_job"] != 1
        or execution["test_outcomes_available_to_backend"] is not False
        or not _is_exact_float(
            execution["stale_foreign_claim_timeout_seconds"],
            minimum=0.0,
        )
        or execution["stale_foreign_claim_timeout_seconds"] != 0.0
        or execution["deterministic_torch"] != DETERMINISTIC_TORCH_CONTRACT
        or not _exact_scalar_types_match(
            execution["deterministic_torch"],
            DETERMINISTIC_TORCH_CONTRACT,
        )
    ):
        raise GeoAdaptV2Error("production execution safety contract drifted")

    expected_cache_keys = {
        _subject_key(dataset, subject)
        for dataset, contract in expected_contracts.items()
        for subject in contract["subjects"]
    }
    expected_split_keys = {
        _split_key(dataset, subject, fold)
        for dataset, contract in expected_contracts.items()
        for subject in contract["subjects"]
        for fold in contract["folds"]
    }
    if set(plan["cache_identity"]) != expected_cache_keys:
        raise GeoAdaptV2Error("production cache roster drifted")
    if set(plan["split_identity"]) != expected_split_keys:
        raise GeoAdaptV2Error("production split roster drifted")
    for key, cache in plan["cache_identity"].items():
        match = re.fullmatch(r"([^:]+):s([0-9]{3})", key)
        dataset = match.group(1) if match is not None else ""
        subject = int(match.group(2)) if match is not None else -1
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
        if (
            match is None
            or dataset not in expected_contracts
            or subject not in expected_contracts[dataset]["subjects"]
            or not isinstance(cache, Mapping)
            or set(cache) != expected_keys
            or not _is_sha256(cache["array_sha256"])
            or cache["dataset"]
            != json.loads(_canonical_bytes(asdict(dataset_spec(dataset))))
            or cache["subject"] != subject
            or not _is_exact_int(cache["subject"], minimum=1)
            or cache["montage_profile"] != MONTAGE_PROFILE
            or cache["preprocessing"] != preprocessing_for_dataset(dataset)
            or not _exact_scalar_types_match(
                cache["preprocessing"],
                preprocessing_for_dataset(dataset),
            )
            or not isinstance(cache["channels"], list)
            or not cache["channels"]
            or any(
                not isinstance(channel, str) or not channel
                for channel in cache["channels"]
            )
            or len(cache["channels"]) != len(set(cache["channels"]))
            or (
                channels_for_dataset(dataset, MONTAGE_PROFILE) is not None
                and cache["channels"]
                != list(channels_for_dataset(dataset, MONTAGE_PROFILE) or ())
            )
            or cache["coordinates"]
            != coordinate_contract_for_dataset(
                dataset,
                MONTAGE_PROFILE,
                tuple(cache["channels"]),
            )
            or not _exact_scalar_types_match(
                cache["coordinates"],
                coordinate_contract_for_dataset(
                    dataset,
                    MONTAGE_PROFILE,
                    tuple(cache["channels"]),
                ),
            )
            or not isinstance(cache["shape"], list)
            or len(cache["shape"]) != 3
            or any(not _is_exact_int(value, minimum=1) for value in cache["shape"])
            or cache["shape"][1] != len(cache["channels"])
            or cache["shape"][2] != int(preprocessing_for_dataset(dataset)["n_times"])
        ):
            raise GeoAdaptV2Error(f"production cache identity is invalid: {key}")
        if dataset == "local_exp4":
            manifest = cache["source_manifest"]
            if not isinstance(manifest, list) or not manifest:
                raise GeoAdaptV2Error(
                    f"production local cache source manifest is invalid: {key}"
                )
            for source in manifest:
                if (
                    not isinstance(source, Mapping)
                    or set(source)
                    != {"filename", "run", "sha256", "size_bytes", "subject"}
                    or not isinstance(source["filename"], str)
                    or not source["filename"]
                    or not _is_exact_int(source["run"], minimum=1)
                    or not _is_sha256(source["sha256"])
                    or not _is_exact_int(source["size_bytes"], minimum=1)
                    or source["subject"] != subject
                    or not _is_exact_int(source["subject"], minimum=1)
                ):
                    raise GeoAdaptV2Error(
                        f"production local cache source identity is invalid: {key}"
                    )
    for key, split in plan["split_identity"].items():
        if not isinstance(split, Mapping):
            raise GeoAdaptV2Error(f"production split identity is invalid: {key}")
        if (
            not isinstance(split.get("dataset"), str)
            or not _is_exact_int(split.get("subject"), minimum=1)
            or not _is_exact_int(split.get("fold"), minimum=0)
            or not _is_exact_int(split.get("trial_count"), minimum=1)
        ):
            raise GeoAdaptV2Error(f"production split scalar identity is invalid: {key}")
        expected_key = _split_key(
            str(split.get("dataset")),
            split["subject"],
            split["fold"],
        )
        partitions = split.get("partitions")
        cache_key = _subject_key(
            str(split.get("dataset")),
            split["subject"],
        )
        if (
            expected_key != key
            or cache_key not in plan["cache_identity"]
            or split.get("cache_array_sha256")
            != plan["cache_identity"][cache_key]["array_sha256"]
            or not isinstance(partitions, Mapping)
            or set(partitions) != {"train", "validation", "source", "test"}
        ):
            raise GeoAdaptV2Error(f"production split identity is inconsistent: {key}")
        if any(
            not isinstance(partitions[name], Mapping)
            or not _is_exact_int(partitions[name].get("count"), minimum=1)
            for name in ("train", "validation", "source", "test")
        ):
            raise GeoAdaptV2Error(
                f"production split partition types are invalid: {key}"
            )
        counts = {
            name: partitions[name]["count"]
            for name in ("train", "validation", "source", "test")
        }
        if (
            min(counts.values()) <= 0
            or counts["source"] != counts["train"] + counts["validation"]
            or counts["source"] + counts["test"] != split["trial_count"]
            or any(
                not _is_sha256(partitions[name].get("rows_sha256"))
                for name in partitions
            )
        ):
            raise GeoAdaptV2Error(f"production split coverage is invalid: {key}")

    jobs = tuple(iter_jobs(plan, validate_plan=False))
    counts = Counter(job.stable_id for job in jobs)
    expected_counts = {
        "architecture.geoadaptnet": 2_150,
        "architecture.geoadaptnet_fb": 2_195,
        "architecture.geoadaptnet_fbsp": 2_150,
    }
    if (
        len(jobs) != 6_495
        or not _is_exact_int(plan["n_jobs"])
        or plan["n_jobs"] != 6_495
        or counts != expected_counts
        or len({job.job_id for job in jobs}) != 6_495
    ):
        raise GeoAdaptV2Error(
            "production Cartesian roster is not the exact 6,495-job contract"
        )


def validate_production_semantics(plan: Mapping[str, Any]) -> None:
    """Fail closed on any structural or semantic production-plan deviation."""

    try:
        _validate_production_semantics_impl(plan)
    except GeoAdaptV2Error:
        raise
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise GeoAdaptV2Error(
            "production plan cannot satisfy its exact semantic contract"
        ) from error


def _quarantine_plan_authority_residue(
    run_root: Path,
    path: Path,
    *,
    reason: str,
) -> Path:
    """Move pre-plan transaction residue without requiring a plan ledger."""

    run_root = _assert_safe_path(run_root, leaf_kind="directory")
    path = _assert_safe_path(path)
    try:
        path.relative_to(run_root)
    except ValueError as error:
        raise GeoAdaptV2Error("plan residue is outside its run root") from error
    identity = _anchored_lstat(path)
    root = run_root / ".plan-authority-quarantine"
    _safe_mkdir(root)
    destination = root / f"{path.name}.{reason}.{time.time_ns()}.{uuid.uuid4().hex}"
    _rename_noreplace(
        path,
        destination,
        expected_source_dev=int(identity.st_dev),
        expected_source_ino=int(identity.st_ino),
    )
    _fsync_directory(run_root)
    _fsync_directory(root)
    return destination


def _recover_plan_staging_directories(run_root: Path) -> tuple[Path, ...]:
    recovered: list[Path] = []
    prefix = ".plan.publish-"
    suffix = ".staged"
    for name, observed in sorted(_anchored_directory_entries(run_root).items()):
        if not name.startswith(prefix) or not name.endswith(suffix):
            continue
        nonce = name[len(prefix) : -len(suffix)]
        path = run_root / name
        if re.fullmatch(r"[0-9a-f]{32}", nonce) is None:
            raise GeoAdaptV2Error(f"plan staging identity is invalid: {path}")
        if not stat.S_ISDIR(observed.st_mode):
            raise GeoAdaptV2Error(f"plan staging path is not a directory: {path}")
        recovered.append(
            _quarantine_plan_authority_residue(
                run_root,
                path,
                reason="powercut",
            )
        )
    return tuple(recovered)


def _create_plan_transaction(
    run_root: Path,
    transaction: Path,
    *,
    payloads: Mapping[str, bytes],
) -> None:
    staged = run_root / f".plan.publish-{uuid.uuid4().hex}.staged"
    _safe_mkdir(staged, parents=False)
    staged_identity = _anchored_lstat(staged)
    try:
        for name, payload in payloads.items():
            if name not in {"plan.json", "plan.sha256"}:
                raise GeoAdaptV2Error("plan transaction has an unknown member")
            _write_bytes_exclusive(staged / name, payload)
        _fsync_directory(staged)
        _rename_noreplace(
            staged,
            transaction,
            expected_source_dev=int(staged_identity.st_dev),
            expected_source_ino=int(staged_identity.st_ino),
        )
    except Exception:
        if _path_exists(staged):
            _quarantine_plan_authority_residue(
                run_root,
                staged,
                reason="failed",
            )
        raise


def write_or_validate_plan(
    run_root: Path, expected: Mapping[str, Any]
) -> dict[str, Any]:
    run_root = _absolute_path(run_root)
    _assert_safe_path(run_root, leaf_kind="directory", allow_missing=True)
    plan_path = run_root / "plan.json"
    digest_path = run_root / "plan.sha256"
    value = copy.deepcopy(dict(expected))
    validate_production_semantics(value)
    digest = str(value["plan_sha256"])
    _safe_mkdir(run_root)
    transaction = run_root / ".plan.publication.transaction"
    expected_payloads = {
        "plan.json": _canonical_bytes(value) + b"\n",
        "plan.sha256": (digest + "\n").encode("ascii"),
    }
    with _authority_publication_lock(plan_path):
        _recover_plan_staging_directories(run_root)
        final_exists = {
            "plan.json": _path_exists(plan_path),
            "plan.sha256": _path_exists(digest_path),
        }
        if all(final_exists.values()):
            try:
                observed = load_plan(run_root)
            except (GeoAdaptV2Error, OSError, ValueError):
                for path in (plan_path, digest_path):
                    if _path_exists(path):
                        _quarantine_plan_authority_residue(
                            run_root,
                            path,
                            reason="torn",
                        )
            else:
                if observed != value:
                    raise GeoAdaptV2Error(
                        "existing immutable plan differs from request"
                    )
                if _path_exists(transaction):
                    _quarantine_plan_authority_residue(
                        run_root,
                        transaction,
                        reason="superseded",
                    )
                return observed
        else:
            for name, path in (
                ("plan.json", plan_path),
                ("plan.sha256", digest_path),
            ):
                if not final_exists[name]:
                    continue
                try:
                    matches = _safe_read_bytes(path) == expected_payloads[name]
                except (GeoAdaptV2Error, OSError, ValueError):
                    matches = False
                if not matches:
                    _quarantine_plan_authority_residue(
                        run_root,
                        path,
                        reason="torn",
                    )

        missing = {
            name
            for name, path in (
                ("plan.json", plan_path),
                ("plan.sha256", digest_path),
            )
            if not _path_exists(path)
        }
        if _path_exists(transaction):
            valid_transaction = True
            try:
                transaction = _assert_safe_path(
                    transaction,
                    leaf_kind="directory",
                )
                entries = _anchored_directory_entries(transaction)
                valid_transaction = set(entries) == missing
                for name, observed in entries.items():
                    _require_single_link(observed, transaction / name)
                    if (
                        name not in expected_payloads
                        or not stat.S_ISREG(observed.st_mode)
                        or _safe_read_bytes(transaction / name)
                        != expected_payloads[name]
                    ):
                        valid_transaction = False
            except (GeoAdaptV2Error, OSError, ValueError):
                valid_transaction = False
            if not valid_transaction:
                _quarantine_plan_authority_residue(
                    run_root,
                    transaction,
                    reason="torn",
                )
        if not _path_exists(transaction):
            _create_plan_transaction(
                run_root,
                transaction,
                payloads={name: expected_payloads[name] for name in sorted(missing)},
            )

        transaction_identity = _anchored_lstat(transaction)
        for name, destination in (
            ("plan.json", plan_path),
            ("plan.sha256", digest_path),
        ):
            source = transaction / name
            if not _path_exists(source):
                continue
            source_identity = _anchored_lstat(source)
            _rename_noreplace(
                source,
                destination,
                expected_source_dev=int(source_identity.st_dev),
                expected_source_ino=int(source_identity.st_ino),
            )
        _remove_owned_empty_directory(
            transaction,
            expected_dev=int(transaction_identity.st_dev),
            expected_ino=int(transaction_identity.st_ino),
        )
        observed = load_plan(run_root)
        if observed != value:
            raise GeoAdaptV2Error("published immutable plan differs from request")
        return observed


def load_plan(run_root: Path) -> dict[str, Any]:
    run_root = _assert_safe_path(run_root, leaf_kind="directory")
    plan_path = run_root / "plan.json"
    digest_path = run_root / "plan.sha256"
    try:
        _assert_safe_path(plan_path, leaf_kind="file")
        _assert_safe_path(digest_path, leaf_kind="file")
    except FileNotFoundError:
        raise FileNotFoundError(f"immutable plan absent under {run_root}")
    plan_bytes = _safe_read_bytes(plan_path)
    plan = _strict_json_bytes(plan_bytes, source=str(plan_path))
    validate_production_semantics(plan)
    if (
        _safe_read_text(digest_path, encoding="ascii").strip() != plan["plan_sha256"]
        or plan_bytes != _canonical_bytes(plan) + b"\n"
    ):
        raise GeoAdaptV2Error("immutable plan bytes/checksum are invalid")
    return plan


def iter_jobs(plan: Mapping[str, Any], *, validate_plan: bool = True) -> Iterator[Job]:
    if validate_plan:
        if plan.get("schema") != PLAN_SCHEMA:
            raise GeoAdaptV2Error("cannot iterate a non-GeoAdapt v2 plan")
    count = 0
    for dataset in plan["dataset_order"]:
        contract = plan["datasets"][dataset]
        for stable_id in plan["stable_id_order"]:
            cell = plan["eligibility"][dataset][stable_id]
            if cell["status"] != "eligible":
                continue
            for subject in contract["subjects"]:
                for fold in contract["folds"]:
                    for seed in plan["seeds"]:
                        count += 1
                        yield Job(
                            dataset=str(dataset),
                            stable_id=str(stable_id),
                            subject=int(subject),
                            fold=int(fold),
                            seed=int(seed),
                        )
    if count != int(plan["n_jobs"]):
        raise GeoAdaptV2Error("iterated job count differs from plan")


def _validate_job_identity(plan: Mapping[str, Any], job: Job) -> None:
    # Applicability is checked before any cache access.  In particular, this
    # prevents the impossible FBSP x BNCI2014-004 cell from being silently
    # clamped or outcome-opened.
    try:
        cell = plan["eligibility"][job.dataset][job.stable_id]
    except (KeyError, TypeError) as error:
        raise NotApplicableError(
            "job is outside the frozen procedure matrix"
        ) from error
    if cell["status"] != "eligible":
        raise NotApplicableError(str(cell["reason"]))
    try:
        contract = plan["datasets"][job.dataset]
        valid = (
            job.stable_id in plan["stable_id_order"]
            and job.subject in contract["subjects"]
            and job.fold in contract["folds"]
            and job.seed in plan["seeds"]
        )
    except (KeyError, TypeError):
        valid = False
    if not valid:
        raise GeoAdaptV2Error("job identity is outside the immutable plan")


def verify_static_identity(plan: Mapping[str, Any]) -> None:
    validate_production_semantics(plan)
    project_root = Path(__file__).resolve().parents[1]
    project = _validate_uv_project_contract(project_root)
    _require_uv_virtual_environment(project)
    if _source_identity(project_root) != plan.get("source_identity"):
        raise GeoAdaptV2Error("source closure differs from immutable plan")
    if _registry_snapshot(plan["stable_id_order"]) != plan.get("registry_records"):
        raise GeoAdaptV2Error("model registry snapshots differ from immutable plan")
    if _environment_identity() != plan.get("environment_identity"):
        raise GeoAdaptV2Error("UV environment differs from immutable plan")
    if project != plan.get("project_identity"):
        raise GeoAdaptV2Error("UV project/lock identity differs from immutable plan")
    if _feature_contract() != plan.get("feature_contract"):
        raise GeoAdaptV2Error("four-band feature contract differs from immutable plan")


def verify_cache_and_splits(plan: Mapping[str, Any], *, cache_root: Path) -> None:
    validate_production_semantics(plan)
    cache_root = _assert_safe_path(cache_root, leaf_kind="directory")
    caches, splits = _cache_and_split_identity(cache_root, plan["datasets"])
    if caches != plan.get("cache_identity"):
        raise GeoAdaptV2Error("cache identities differ from immutable plan")
    if splits != plan.get("split_identity"):
        raise GeoAdaptV2Error("split identities differ from immutable plan")


def _verify_job_cache_and_split(
    plan: Mapping[str, Any],
    *,
    cache_root: Path,
    job: Job,
) -> None:
    """Rebuild the exact cache/split identities consumed by one atomic job."""

    from .data import load_subject_cache, split_indices

    cache_root = _assert_safe_path(cache_root, leaf_kind="directory")
    data = load_subject_cache(
        job.dataset,
        job.subject,
        cache_root=cache_root,
        montage_profile=MONTAGE_PROFILE,
    )
    observed_cache = copy.deepcopy(dict(data["identity"]))
    if observed_cache != dict(_planned_cache(plan, job)):
        raise GeoAdaptV2Error("live job cache differs from immutable plan")
    train, validation, test = split_indices(
        job.dataset,
        np.asarray(data["y"]),
        np.asarray(data["sessions"]),
        np.asarray(data["runs"]),
        fold=job.fold,
        subject=job.subject,
    )
    observed_split = _one_split_identity(
        dataset=job.dataset,
        subject=job.subject,
        fold=job.fold,
        cache_array_sha256=str(observed_cache["array_sha256"]),
        trial_count=len(data["y"]),
        train_rows=np.asarray(train, dtype=np.int64),
        validation_rows=np.asarray(validation, dtype=np.int64),
        test_rows=np.asarray(test, dtype=np.int64),
    )
    if observed_split != dict(_planned_split(plan, job)):
        raise GeoAdaptV2Error("live job split differs from immutable plan")


def _rebind_authoritative_inputs(
    *,
    run_root: Path,
    plan: Mapping[str, Any],
    cache_root: Path,
    job: Job | None = None,
    claim: Claim | None = None,
    metadata: Mapping[str, Any] | None = None,
    gpu_lease: project_gpu_leases.GPULease | None = None,
    expected_gpu_lease_receipt: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Rebind every frozen authority immediately around a canonical commit."""

    live_plan = load_plan(run_root)
    if _canonical_bytes(live_plan) != _canonical_bytes(plan):
        raise GeoAdaptV2Error("immutable plan changed at publication boundary")
    verify_static_identity(live_plan)
    if job is None:
        verify_cache_and_splits(live_plan, cache_root=cache_root)
    else:
        _validate_job_identity(live_plan, job)
        _verify_job_cache_and_split(live_plan, cache_root=cache_root, job=job)
        if metadata is not None and (
            metadata.get("cache_array_sha256")
            != _planned_cache(live_plan, job).get("array_sha256")
            or metadata.get("split") != _planned_split(live_plan, job)
        ):
            raise GeoAdaptV2Error(
                "record metadata cache/split differs at publication boundary"
            )
    if claim is not None:
        if job is None or claim.job != job:
            raise GeoAdaptV2Error("claim/job mismatch at publication boundary")
        _validate_claim_payload(
            run_root,
            live_plan,
            claim,
            metadata=metadata,
        )
        if expected_gpu_lease_receipt != claim.gpu_lease_receipt:
            raise GeoAdaptV2Error(
                "GPU lease receipt changed at publication boundary"
            )
        _assert_bound_gpu_lease(gpu_lease, expected_gpu_lease_receipt)
    elif gpu_lease is not None or expected_gpu_lease_receipt is not None:
        raise GeoAdaptV2Error("GPU authority cannot exist without a job claim")
    return live_plan


def fixed_filter_bank(x: np.ndarray, *, sfreq: float = 128.0) -> np.ndarray:
    """Apply the frozen four-band FIR view independently to every trial."""

    from scipy import signal

    values = np.asarray(x, dtype=np.float64)
    if (
        values.ndim != 3
        or values.shape[-1] <= FIR_TAPS
        or not np.all(np.isfinite(values))
    ):
        raise ValueError(
            "x must be finite (trials, channels, time) and exceed FIR length"
        )
    coefficients = np.stack(
        [
            signal.firwin(
                FIR_TAPS,
                (low, high),
                pass_zero=False,
                fs=float(sfreq),
                window=FIR_WINDOW,
                scale=True,
            )
            for low, high in FILTER_BANDS_HZ
        ],
        axis=0,
    )
    half = FIR_TAPS // 2
    padded = np.pad(values, ((0, 0), (0, 0), (half, half)), mode="reflect")
    bands: list[np.ndarray] = []
    for taps in coefficients:
        filtered = signal.fftconvolve(
            padded,
            taps.reshape(1, 1, -1),
            mode="valid",
            axes=-1,
        )
        filtered -= filtered.mean(axis=-1, keepdims=True)
        bands.append(filtered)
    result = np.stack(bands, axis=1)
    expected = (len(values), len(FILTER_BANDS_HZ), *values.shape[1:])
    if result.shape != expected:
        raise GeoAdaptV2Error("fixed filter bank returned an unexpected shape")
    return np.ascontiguousarray(result, dtype=np.float32)


def spd_covariances(epochs: np.ndarray) -> np.ndarray:
    """Construct deterministic SPD matrices that remain positive in float32."""

    values = np.asarray(epochs, dtype=np.float64)
    if values.ndim != 4 or values.shape[-1] < 2 or not np.all(np.isfinite(values)):
        raise ValueError("epochs must be finite (trials, bands, channels, time)")
    values = values - values.mean(axis=-1, keepdims=True)
    covariances = np.einsum("nbct,nbdt->nbcd", values, values, optimize=True) / float(
        values.shape[-1]
    )
    channels = covariances.shape[-1]
    trace_scale = np.trace(covariances, axis1=-2, axis2=-1) / float(channels)
    identity = np.eye(channels, dtype=np.float64)
    covariances = (
        1.0 - COVARIANCE_SHRINKAGE
    ) * covariances + COVARIANCE_SHRINKAGE * trace_scale[..., None, None] * identity
    covariances = 0.5 * (covariances + covariances.swapaxes(-1, -2))
    # The persisted dtype, rather than the float64 workspace, determines the
    # necessary eigenvalue margin.  A float64-positive matrix can acquire a
    # zero/negative eigenvalue when rounded to float32, especially for zero,
    # constant, or rank-deficient trials.
    floor = np.maximum(
        np.abs(trace_scale) * np.finfo(np.float32).eps * 64.0,
        np.finfo(np.float32).tiny * 64.0,
    )
    minimum = np.linalg.eigvalsh(covariances)[..., 0]
    covariances += np.maximum(0.0, floor - minimum)[..., None, None] * identity
    result = np.ascontiguousarray(covariances, dtype=np.float32)
    # Recheck the actual stored matrices and add a representable diagonal
    # margin if a platform's float32 rounding consumed the first margin.
    for _ in range(3):
        stored64 = result.astype(np.float64)
        stored_scale = np.maximum(
            np.mean(
                np.abs(np.diagonal(stored64, axis1=-2, axis2=-1)),
                axis=-1,
            ),
            np.finfo(np.float32).tiny,
        )
        stored_floor = np.maximum(
            stored_scale * np.finfo(np.float32).eps * 64.0,
            np.finfo(np.float32).tiny * 64.0,
        )
        stored_minimum = np.linalg.eigvalsh(stored64)[..., 0]
        adjustment = np.maximum(0.0, stored_floor - stored_minimum)
        if not np.any(adjustment > 0.0):
            break
        diagonal = np.diagonal(result, axis1=-2, axis2=-1).copy()
        diagonal = (diagonal.astype(np.float64) + adjustment[..., None]).astype(
            np.float32
        )
        indices = np.arange(channels)
        result[..., indices, indices] = diagonal
    result = np.ascontiguousarray(
        0.5 * (result + result.swapaxes(-1, -2)), dtype=np.float32
    )
    if (
        not np.all(np.isfinite(result))
        or np.min(np.linalg.eigvalsh(result.astype(np.float64))) <= 0.0
    ):
        raise GeoAdaptV2Error("covariance construction failed to produce SPD matrices")
    return result


def derive_four_band_spd(x: np.ndarray, *, sfreq: float = 128.0) -> np.ndarray:
    return spd_covariances(fixed_filter_bank(x, sfreq=sfreq))


def _planned_split(plan: Mapping[str, Any], job: Job) -> Mapping[str, Any]:
    return plan["split_identity"][_split_key(job.dataset, job.subject, job.fold)]


def _planned_cache(plan: Mapping[str, Any], job: Job) -> Mapping[str, Any]:
    return plan["cache_identity"][_subject_key(job.dataset, job.subject)]


def prepare_job(
    *,
    job: Job,
    plan: Mapping[str, Any],
    cache_root: Path,
    cache_loader: Callable[..., Mapping[str, Any]] | None = None,
    splitter: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
) -> PreparedJob:
    """Load and prepare one job while withholding held-out outcomes."""

    _validate_job_identity(plan, job)
    from .data import (
        apply_channel_scaler,
        fit_channel_scaler,
        load_subject_cache,
        split_indices,
    )

    load = load_subject_cache if cache_loader is None else cache_loader
    split = split_indices if splitter is None else splitter
    data = load(
        job.dataset,
        job.subject,
        cache_root=cache_root,
        montage_profile=MONTAGE_PROFILE,
    )
    if dict(data["identity"]) != dict(_planned_cache(plan, job)):
        raise GeoAdaptV2Error("runtime cache identity differs from immutable plan")
    train_rows, validation_rows, test_rows = split(
        job.dataset,
        np.asarray(data["y"]),
        np.asarray(data["sessions"]),
        np.asarray(data["runs"]),
        fold=job.fold,
        subject=job.subject,
    )
    train_rows = np.asarray(train_rows, dtype=np.int64)
    validation_rows = np.asarray(validation_rows, dtype=np.int64)
    test_rows = np.asarray(test_rows, dtype=np.int64)
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
    if observed_split != _planned_split(plan, job):
        raise GeoAdaptV2Error("runtime split differs from immutable plan")
    source_rows = np.sort(np.concatenate((train_rows, validation_rows))).astype(
        np.int64, copy=False
    )
    channel_names = tuple(str(value) for value in data["channel_names"].tolist())
    selection_mean, selection_std = fit_channel_scaler(
        np.asarray(data["x"])[train_rows], channel_names
    )
    refit_mean, refit_std = fit_channel_scaler(
        np.asarray(data["x"])[source_rows], channel_names
    )
    x_train = apply_channel_scaler(
        np.asarray(data["x"])[train_rows], selection_mean, selection_std
    )
    x_validation = apply_channel_scaler(
        np.asarray(data["x"])[validation_rows], selection_mean, selection_std
    )
    x_source = apply_channel_scaler(
        np.asarray(data["x"])[source_rows], refit_mean, refit_std
    )
    x_test = apply_channel_scaler(
        np.asarray(data["x"])[test_rows], refit_mean, refit_std
    )
    use_spd = (
        plan["stable_configs"][job.stable_id]["input_representation"]
        == "deterministic_four_band_spd"
    )
    sfreq = float(plan["datasets"][job.dataset]["preprocessing"]["sfreq_hz"])
    if use_spd:
        cov_train = derive_four_band_spd(x_train, sfreq=sfreq)
        cov_validation = derive_four_band_spd(x_validation, sfreq=sfreq)
        cov_source = derive_four_band_spd(x_source, sfreq=sfreq)
        cov_test = derive_four_band_spd(x_test, sfreq=sfreq)
    else:
        cov_train = cov_validation = cov_source = cov_test = None

    # Only source-side outcomes survive this boundary.  The backend receives no
    # full cache object and no held-out outcome vector.
    y = np.asarray(data["y"], dtype=np.int64)
    return PreparedJob(
        channel_names=channel_names,
        positions=np.ascontiguousarray(data["positions"], dtype=np.float32),
        train_rows=train_rows,
        validation_rows=validation_rows,
        source_rows=source_rows,
        test_rows=test_rows,
        y_train=np.ascontiguousarray(y[train_rows]),
        y_validation=np.ascontiguousarray(y[validation_rows]),
        y_source=np.ascontiguousarray(y[source_rows]),
        x_train=np.ascontiguousarray(x_train),
        x_validation=np.ascontiguousarray(x_validation),
        x_source=np.ascontiguousarray(x_source),
        x_test=np.ascontiguousarray(x_test),
        cov_train=cov_train,
        cov_validation=cov_validation,
        cov_source=cov_source,
        cov_test=cov_test,
        selection_mean=np.ascontiguousarray(selection_mean),
        selection_std=np.ascontiguousarray(selection_std),
        refit_mean=np.ascontiguousarray(refit_mean),
        refit_std=np.ascontiguousarray(refit_std),
        cache_array_sha256=str(data["identity"]["array_sha256"]),
        split_identity=observed_split,
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


def _configure_torch(seed: int, *, deterministic: bool) -> None:
    import torch

    if deterministic is not True:
        raise GeoAdaptV2Error(
            "GeoAdapt harmonized-v2 forbids nondeterministic Torch execution"
        )
    if (
        torch.cuda.is_available()
        and os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        != DETERMINISTIC_TORCH_CONTRACT["cublas_workspace_config_on_cuda"]
    ):
        raise GeoAdaptV2Error(
            "deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8 "
            "before Python starts"
        )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    observed = _torch_determinism_identity()
    for key in (
        "deterministic_algorithms",
        "warn_only",
        "cudnn_deterministic",
        "cudnn_benchmark",
        "cuda_matmul_tf32",
        "cudnn_tf32",
        "float32_matmul_precision",
    ):
        if observed[key] != DETERMINISTIC_TORCH_CONTRACT[key]:
            raise GeoAdaptV2Error(
                f"Torch failed to install deterministic setting {key}"
            )
    if (
        observed["cuda_available"]
        and observed["cublas_workspace_config"]
        != (DETERMINISTIC_TORCH_CONTRACT["cublas_workspace_config_on_cuda"])
    ):
        raise GeoAdaptV2Error("Torch failed to install the deterministic contract")


def _torch_determinism_identity() -> dict[str, Any]:
    import torch

    return {
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "warn_only": bool(torch.is_deterministic_algorithms_warn_only_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cuda_matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cuda_available": bool(torch.cuda.is_available()),
    }


def _parameter_count(model: Any) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _fit_fixed_geoadapt(
    *,
    job: Job,
    config: Mapping[str, Any],
    prepared: PreparedJob,
    device: str,
) -> tuple[dict[str, Any], np.ndarray]:
    import torch

    from deepnet.engine import (
        CovarianceDataset,
        TrainConfig,
        predict_proba,
        set_reproducible_seed,
        train_fixed_epochs,
        train_model,
    )
    from deepnet.model import GeoAdaptNet

    if any(
        value is None
        for value in (
            prepared.cov_train,
            prepared.cov_validation,
            prepared.cov_source,
            prepared.cov_test,
        )
    ):
        raise GeoAdaptV2Error("fixed GeoAdaptNet requires the SPD feature view")
    trainer_values = dict(config["trainer"])
    trainer_values.update({"seed": job.seed, "device": device})
    trainer = TrainConfig(**trainer_values)
    model_values = dict(config["model"])
    model_values["channels"] = len(prepared.channel_names)

    _configure_torch(job.seed, deterministic=True)
    set_reproducible_seed(job.seed, deterministic=True)
    model = GeoAdaptNet(**model_values)
    initial_state = copy.deepcopy(model.state_dict())
    initial_hash = _state_sha256(model)
    target = torch.device(device)
    if target.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target)

    selection = train_model(
        model,
        CovarianceDataset(prepared.cov_train, prepared.y_train),
        CovarianceDataset(prepared.cov_validation, prepared.y_validation),
        trainer,
    )
    selection_hash = _state_sha256(selection.model)
    selection.model.load_state_dict(initial_state, strict=True)
    reset_hash = _state_sha256(selection.model)
    if reset_hash != initial_hash:
        raise GeoAdaptV2Error("GeoAdaptNet reset did not restore initialization")
    selected_epochs = int(selection.best_epoch) + 1
    refit = train_fixed_epochs(
        selection.model,
        CovarianceDataset(prepared.cov_source, prepared.y_source),
        epochs=selected_epochs,
        config=trainer,
    )
    if target.type == "cuda":
        torch.cuda.synchronize(target)
    inference_started = time.perf_counter()
    # Synthetic zeros satisfy the dataset container only; actual held-out
    # outcomes are absent from PreparedJob and never reach this call.
    prediction_data = CovarianceDataset(
        prepared.cov_test,
        np.zeros(len(prepared.test_rows), dtype=np.int64),
    )
    probabilities, _ = predict_proba(
        refit.model,
        prediction_data,
        device=device,
        branch="full",
    )
    if target.type == "cuda":
        torch.cuda.synchronize(target)
    inference_seconds = time.perf_counter() - inference_started
    peak = int(torch.cuda.max_memory_allocated(target)) if target.type == "cuda" else 0
    metadata = {
        "initial_state_sha256": initial_hash,
        "selection_state_sha256": selection_hash,
        "reset_state_sha256": reset_hash,
        "reset_verified": True,
        "source_selected_epoch_index": int(selection.best_epoch),
        "selection_epochs_run": int(selection.epochs_ran),
        "refit_epochs_run": int(refit.epochs_ran),
        "refit_state_sha256": _state_sha256(refit.model),
        "parameter_count": _parameter_count(refit.model),
        "selection_fit_seconds": float(selection.train_seconds),
        "refit_fit_seconds": float(refit.train_seconds),
        "test_inference_seconds": inference_seconds,
        "cuda_peak_memory_bytes": peak,
    }
    return metadata, np.asarray(probabilities, dtype=np.float64)


def _make_filterbank_model(
    *,
    config: Mapping[str, Any],
    n_channels: int,
    sfreq: float,
) -> Any:
    from deepnet.config import BANDS
    from deepnet.filterbank_net import FilterBankSPDNet

    values = dict(config["model"])
    warm_start = bool(values.pop("warm_start_bands"))
    init_bands = (
        list(BANDS) if warm_start and int(values["n_bands"]) == len(BANDS) else None
    )
    return FilterBankSPDNet(
        channels=n_channels,
        sfreq=sfreq,
        init_bands=init_bands,
        **values,
    )


def _filterbank_epoch(
    *,
    model: Any,
    x: Any,
    y: Any,
    optimizer: Any,
    loss_fn: Any,
    batch_size: int,
    gradient_clip: float,
    generator: Any,
) -> None:
    import torch

    model.train()
    order = torch.randperm(len(x), device=x.device, generator=generator)
    for start in range(0, len(x), batch_size):
        indices = order[start : start + batch_size]
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x[indices]), y[indices])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()


def _fit_filterbank(
    *,
    job: Job,
    plan: Mapping[str, Any],
    config: Mapping[str, Any],
    prepared: PreparedJob,
    device: str,
) -> tuple[dict[str, Any], np.ndarray]:
    import torch
    from torch import nn

    trainer = dict(config["trainer"])
    deterministic = bool(trainer["deterministic"])
    _configure_torch(job.seed, deterministic=deterministic)
    target = torch.device(device)
    sfreq = float(plan["datasets"][job.dataset]["preprocessing"]["sfreq_hz"])
    model = _make_filterbank_model(
        config=config,
        n_channels=len(prepared.channel_names),
        sfreq=sfreq,
    ).to(target)
    initial_state = copy.deepcopy(model.state_dict())
    initial_hash = _state_sha256(model)
    if target.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target)

    x_train = torch.as_tensor(prepared.x_train, dtype=torch.float32, device=target)
    y_train = torch.as_tensor(prepared.y_train, dtype=torch.long, device=target)
    x_validation = torch.as_tensor(
        prepared.x_validation, dtype=torch.float32, device=target
    )
    y_validation = torch.as_tensor(
        prepared.y_validation, dtype=torch.long, device=target
    )
    loss_fn = nn.CrossEntropyLoss(label_smoothing=float(trainer["label_smoothing"]))
    selection_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.0)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(trainer["learning_rate"]),
        weight_decay=float(trainer["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(trainer["epochs"])
    )
    generator = torch.Generator(device=target).manual_seed(job.seed)
    best_state: dict[str, Any] | None = None
    best_value = float("inf")
    best_epoch = -1
    stale = 0
    selection_started = time.perf_counter()
    epochs_run = 0
    for epoch in range(int(trainer["epochs"])):
        _filterbank_epoch(
            model=model,
            x=x_train,
            y=y_train,
            optimizer=optimizer,
            loss_fn=loss_fn,
            batch_size=int(trainer["batch_size"]),
            gradient_clip=float(trainer["gradient_clip"]),
            generator=generator,
        )
        scheduler.step()
        model.eval()
        with torch.no_grad():
            value = float(selection_loss_fn(model(x_validation), y_validation))
        if not math.isfinite(value):
            raise GeoAdaptV2Error("filterbank validation objective became non-finite")
        epochs_run = epoch + 1
        if value < best_value - float(trainer["min_delta"]):
            best_value = value
            best_epoch = epoch
            stale = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= int(trainer["patience"]):
                break
    selection_seconds = time.perf_counter() - selection_started
    if best_state is None or best_epoch < 0:
        raise GeoAdaptV2Error("filterbank selection produced no checkpoint")
    model.load_state_dict(best_state, strict=True)
    selection_hash = _state_sha256(model)
    model.load_state_dict(initial_state, strict=True)
    reset_hash = _state_sha256(model)
    if reset_hash != initial_hash:
        raise GeoAdaptV2Error("filterbank reset did not restore initialization")

    _configure_torch(job.seed, deterministic=deterministic)
    x_source = torch.as_tensor(prepared.x_source, dtype=torch.float32, device=target)
    y_source = torch.as_tensor(prepared.y_source, dtype=torch.long, device=target)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(trainer["learning_rate"]),
        weight_decay=float(trainer["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(trainer["epochs"])
    )
    generator = torch.Generator(device=target).manual_seed(job.seed)
    refit_started = time.perf_counter()
    for _ in range(best_epoch + 1):
        _filterbank_epoch(
            model=model,
            x=x_source,
            y=y_source,
            optimizer=optimizer,
            loss_fn=loss_fn,
            batch_size=int(trainer["batch_size"]),
            gradient_clip=float(trainer["gradient_clip"]),
            generator=generator,
        )
        scheduler.step()
    refit_seconds = time.perf_counter() - refit_started
    model.eval()

    x_test = torch.as_tensor(prepared.x_test, dtype=torch.float32, device=target)
    if target.type == "cuda":
        torch.cuda.synchronize(target)
    inference_started = time.perf_counter()
    # This is the only invocation of the held-out prediction path.
    with torch.no_grad():
        probabilities = torch.softmax(model(x_test), dim=1).cpu().numpy()
    if target.type == "cuda":
        torch.cuda.synchronize(target)
    inference_seconds = time.perf_counter() - inference_started
    peak = int(torch.cuda.max_memory_allocated(target)) if target.type == "cuda" else 0
    metadata = {
        "initial_state_sha256": initial_hash,
        "selection_state_sha256": selection_hash,
        "reset_state_sha256": reset_hash,
        "reset_verified": True,
        "source_selected_epoch_index": int(best_epoch),
        "selection_epochs_run": int(epochs_run),
        "refit_epochs_run": int(best_epoch + 1),
        "refit_state_sha256": _state_sha256(model),
        "parameter_count": _parameter_count(model),
        "selection_fit_seconds": float(selection_seconds),
        "refit_fit_seconds": float(refit_seconds),
        "test_inference_seconds": float(inference_seconds),
        "cuda_peak_memory_bytes": peak,
    }
    return metadata, np.asarray(probabilities, dtype=np.float64)


def fit_procedure_backend(
    *,
    job: Job,
    plan: Mapping[str, Any],
    prepared: PreparedJob,
    device: str,
) -> tuple[Mapping[str, Any], np.ndarray]:
    config = plan["stable_configs"][job.stable_id]
    if job.stable_id == "architecture.geoadaptnet":
        return _fit_fixed_geoadapt(
            job=job,
            config=config,
            prepared=prepared,
            device=device,
        )
    if job.stable_id in {
        "architecture.geoadaptnet_fb",
        "architecture.geoadaptnet_fbsp",
    }:
        return _fit_filterbank(
            job=job,
            plan=plan,
            config=config,
            prepared=prepared,
            device=device,
        )
    raise GeoAdaptV2Error(f"no backend for {job.stable_id}")


def _device_resource_identity(device: str) -> dict[str, Any]:
    """Resolve the live device resources that a claim is binding."""

    import torch

    resolved = torch.device(device)
    cuda: dict[str, Any] | None = None
    if resolved.type == "cuda":
        index = (
            torch.cuda.current_device() if resolved.index is None else resolved.index
        )
        properties = torch.cuda.get_device_properties(index)
        cuda = {
            "logical_index": int(index),
            "name": properties.name,
            "capability": [int(properties.major), int(properties.minor)],
            "total_memory_bytes": int(properties.total_memory),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "inventory": _nvidia_inventory(),
        }
    return {
        "requested_device": str(device),
        "resolved_device": str(resolved),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "node_inventory": {
            "host": socket.gethostname(),
            "cuda": cuda,
        },
    }


def _physical_gpu_uuid_from_device_resource(
    device_resource: Mapping[str, Any],
) -> str:
    """Resolve the one physical NVIDIA UUID pinned to a CUDA worker.

    Formal GeoAdapt workers must expose exactly one complete physical GPU UUID
    through ``CUDA_VISIBLE_DEVICES``.  Numeric ordinals and UUID prefixes are
    deliberately rejected because their meaning can vary with host ordering.
    """

    _validate_device_resource_mapping(device_resource)
    node = device_resource["node_inventory"]
    cuda = node["cuda"]
    if cuda is None or not str(device_resource["resolved_device"]).startswith("cuda"):
        raise GeoAdaptV2Error("a physical GPU UUID was requested for a CPU worker")
    visible = device_resource["cuda_visible_devices"]
    if not isinstance(visible, str):
        raise GeoAdaptV2Error(
            "CUDA workers require one full UUID in CUDA_VISIBLE_DEVICES"
        )
    tokens = [token.strip() for token in visible.split(",")]
    if (
        len(tokens) != 1
        or not tokens[0]
        or not GPU_UUID_RE.fullmatch(tokens[0])
        or int(cuda["logical_index"]) != 0
    ):
        raise GeoAdaptV2Error(
            "CUDA_VISIBLE_DEVICES must pin exactly one full physical GPU UUID"
        )
    canonical = "GPU-" + tokens[0][4:].lower()
    inventory_matches = [
        row
        for row in cuda["inventory"]
        if isinstance(row.get("uuid"), str)
        and GPU_UUID_RE.fullmatch(row["uuid"])
        and "GPU-" + row["uuid"][4:].lower() == canonical
    ]
    if len(inventory_matches) != 1:
        raise GeoAdaptV2Error(
            "visible physical GPU UUID is absent or duplicated in node inventory"
        )
    return canonical


@contextlib.contextmanager
def _project_gpu_worker_lease(
    *,
    run_root: Path,
    plan: Mapping[str, Any],
    device: str,
) -> Iterator[project_gpu_leases.GPULease | None]:
    """Hold the shared physical-GPU lease for one complete worker loop."""

    requested = str(device).strip().lower()
    if not requested.startswith("cuda"):
        opening_resource = _device_resource_identity(device)
        if opening_resource["node_inventory"]["cuda"] is not None:
            raise GeoAdaptV2Error(
                "non-CUDA worker unexpectedly resolved a CUDA resource"
            )
        yield None
        return
    if requested not in {"cuda", "cuda:0"}:
        raise GeoAdaptV2Error(
            "one-UUID CUDA isolation permits only device cuda or cuda:0"
        )
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    tokens = [] if visible is None else [token.strip() for token in visible.split(",")]
    if len(tokens) != 1 or not GPU_UUID_RE.fullmatch(tokens[0]):
        raise GeoAdaptV2Error(
            "CUDA workers require exactly one full physical GPU UUID in "
            "CUDA_VISIBLE_DEVICES before any CUDA runtime probe"
        )
    gpu_uuid = "GPU-" + tokens[0][4:].lower()
    try:
        lease = acquire_gpu_lease(
            project_root=Path(__file__).resolve().parents[1],
            run_root=run_root,
            plan_sha256=str(plan["plan_sha256"]),
            gpu_uuid=gpu_uuid,
            track_scope=GPU_LEASE_TRACK_SCOPE,
        )
    except project_gpu_leases.GPUWorkerUnavailable as error:
        raise GPUWorkerUnavailable(str(error)) from error
    except project_gpu_leases.ProjectGPULeaseError as error:
        raise GeoAdaptV2Error(
            f"project-wide GPU lease registry rejected the worker: {error}"
        ) from error
    try:
        try:
            assert_gpu_lease(lease)
            leased_resource = _device_resource_identity(device)
        except project_gpu_leases.ProjectGPULeaseError as error:
            raise GeoAdaptV2Error(
                "project-wide GPU lease was lost before CUDA runtime probing"
            ) from error
        if (
            _physical_gpu_uuid_from_device_resource(leased_resource)
            != lease.value["gpu_uuid"]
        ):
            raise GeoAdaptV2Error(
                "physical GPU resource changed while acquiring its shared lease"
            )
        yield lease
    finally:
        try:
            release_gpu_lease(lease)
        except project_gpu_leases.ProjectGPULeaseError as error:
            raise GeoAdaptV2Error(
                f"project-wide GPU lease release failed closed: {error}"
            ) from error


def _assert_bound_gpu_lease(
    lease: project_gpu_leases.GPULease | None,
    expected_receipt: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    if expected_receipt is None:
        if lease is not None:
            raise GeoAdaptV2Error("CPU resource unexpectedly has a GPU lease")
        return None
    if lease is None:
        raise GeoAdaptV2Error("CUDA resource has no live GPU lease handle")
    try:
        assert_gpu_lease(lease)
        observed = gpu_lease_receipt(lease)
    except project_gpu_leases.ProjectGPULeaseError as error:
        raise GeoAdaptV2Error("project-wide GPU lease ownership was lost") from error
    if observed != dict(expected_receipt):
        raise GeoAdaptV2Error("project-wide GPU lease receipt changed")
    return observed


@contextlib.contextmanager
def _guard_bound_gpu_lease(
    lease: project_gpu_leases.GPULease | None,
    expected_receipt: Mapping[str, Any] | None,
) -> Iterator[Mapping[str, Any] | None]:
    """Hold the shared registry fence across one authoritative mutation."""

    if expected_receipt is None:
        if lease is not None:
            raise GeoAdaptV2Error("CPU publication unexpectedly has a GPU lease")
        yield None
        return
    if lease is None:
        raise GeoAdaptV2Error("CUDA publication has no live GPU lease handle")
    try:
        with guard_gpu_lease(lease) as observed:
            if observed != dict(expected_receipt):
                raise GeoAdaptV2Error("held project-wide GPU lease receipt changed")
            yield observed
    except project_gpu_leases.ProjectGPULeaseError as error:
        raise GeoAdaptV2Error(
            "project-wide GPU lease ownership was lost at publication"
        ) from error


def _runtime_identity(
    device: str,
    *,
    worker_cpu_threads: int,
    portable_compute_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    import torch

    resource = _device_resource_identity(device)
    return {
        "device": resource["resolved_device"],
        "portable_compute_contract_sha256": _sha256_bytes(
            _canonical_bytes(
                _environment_identity()
                if portable_compute_contract is None
                else portable_compute_contract
            )
        ),
        "torch_determinism": _torch_determinism_identity(),
        "worker_cpu_threads": int(worker_cpu_threads),
        "worker_interop_threads": int(torch.get_num_interop_threads()),
        "thread_environment": {
            name: os.environ.get(name) for name in THREAD_ENVIRONMENT_VARIABLES
        },
        "node_inventory": resource["node_inventory"],
    }


def _feature_hashes(prepared: PreparedJob) -> dict[str, Any]:
    result: dict[str, Any] = {
        "selection_train_scaled_sha256": _array_sha256(prepared.x_train),
        "selection_validation_scaled_sha256": _array_sha256(prepared.x_validation),
        "refit_source_scaled_sha256": _array_sha256(prepared.x_source),
        "prediction_test_scaled_sha256": _array_sha256(prepared.x_test),
    }
    if prepared.cov_train is not None:
        result.update(
            {
                "selection_train_spd_sha256": _array_sha256(prepared.cov_train),
                "selection_validation_spd_sha256": _array_sha256(
                    prepared.cov_validation
                ),
                "refit_source_spd_sha256": _array_sha256(prepared.cov_source),
                "prediction_test_spd_sha256": _array_sha256(prepared.cov_test),
            }
        )
    return result


def execute_job(
    *,
    job: Job,
    plan: Mapping[str, Any],
    cache_root: Path,
    device: str,
    backend: ProcedureBackend = fit_procedure_backend,
    cache_loader: Callable[..., Mapping[str, Any]] | None = None,
    splitter: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Execute one reset/refit procedure with an injectable tiny backend."""

    _validate_job_identity(plan, job)
    started = time.perf_counter()
    prepared = prepare_job(
        job=job,
        plan=plan,
        cache_root=cache_root,
        cache_loader=cache_loader,
        splitter=splitter,
    )
    fit_metadata, probabilities = backend(
        job=job,
        plan=plan,
        prepared=prepared,
        device=device,
    )
    values = np.asarray(probabilities, dtype=np.float64)
    expected = (len(prepared.test_rows), 2)
    if (
        values.shape != expected
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(values > 1.0)
        or not np.allclose(values.sum(axis=1), 1.0, rtol=0.0, atol=1e-6)
    ):
        raise GeoAdaptV2Error("backend returned invalid binary probabilities")
    worker_threads = int(plan["execution_config"]["worker_cpu_threads"])
    metadata = {
        "cache_array_sha256": prepared.cache_array_sha256,
        "channels": list(prepared.channel_names),
        "positions_sha256": _array_sha256(prepared.positions),
        "split": copy.deepcopy(dict(prepared.split_identity)),
        "scalers": {
            "selection_mean_sha256": _array_sha256(prepared.selection_mean),
            "selection_std_sha256": _array_sha256(prepared.selection_std),
            "refit_mean_sha256": _array_sha256(prepared.refit_mean),
            "refit_std_sha256": _array_sha256(prepared.refit_std),
        },
        "features": _feature_hashes(prepared),
        "fit": {
            "stable_config": copy.deepcopy(plan["stable_configs"][job.stable_id]),
            "seed_installed_before_construction": True,
            **copy.deepcopy(dict(fit_metadata)),
        },
        "protocol": {
            "selection_scaler_rows": "train_only",
            "selection_optimization_rows": "train_only",
            "epoch_selection_rows": "validation_only",
            "selection_decision": (
                "minimum_unsmoothed_validation_cross_entropy_not_persisted"
            ),
            "refit_scaler_rows": "train_plus_validation",
            "refit_optimization_rows": "train_plus_validation",
            "test_use": "single_probability_prediction_call_only",
            "test_outcomes_available_to_backend": False,
            "test_performance_computed": False,
            "transductive_calibration": False,
        },
        "timing_seconds": {
            "selection_fit": float(fit_metadata["selection_fit_seconds"]),
            "refit_fit": float(fit_metadata["refit_fit_seconds"]),
            "test_inference": float(fit_metadata["test_inference_seconds"]),
            "job_total": time.perf_counter() - started,
        },
        "runtime": _runtime_identity(
            device,
            worker_cpu_threads=worker_threads,
            portable_compute_contract=plan["environment_identity"],
        ),
    }
    for transient in (
        "selection_fit_seconds",
        "refit_fit_seconds",
        "test_inference_seconds",
    ):
        metadata["fit"].pop(transient, None)
    return (
        metadata,
        np.ascontiguousarray(prepared.test_rows, dtype=np.int64),
        np.ascontiguousarray(values, dtype=np.float64),
    )


def _record_directory(run_root: Path, job: Job) -> Path:
    name = f"s{job.subject:03d}-f{job.fold:02d}-seed{job.seed:010d}"
    return run_root / "records" / job.dataset / job.stable_id / name


def _claim_path(run_root: Path, job: Job) -> Path:
    return run_root / "claims" / job.job_id[-2:] / f"{job.job_id}.json"


def _parse_linux_proc_stat_start_marker(text: str, *, pid: int) -> str | None:
    if not _is_exact_int(pid, minimum=1):
        return None
    prefix = f"{pid} ("
    if not text.startswith(prefix):
        return None
    opening = len(prefix) - 1
    closing = text.rfind(")")
    if (
        closing <= opening
        or closing + 1 >= len(text)
        or not text[closing + 1].isspace()
    ):
        return None
    # /proc/<pid>/stat field 2 is a parenthesized command name that may itself
    # contain spaces or parentheses. The last ')' is the only safe boundary;
    # field 22 (starttime) is suffix index 19 because the suffix begins at
    # field 3 (state).
    suffix = text[closing + 1 :].split()
    if len(suffix) <= 19:
        return None
    marker = suffix[19]
    if not marker.isascii() or not marker.isdigit():
        return None
    return marker if int(marker) > 0 else None


def _parse_linux_boot_id(payload: bytes) -> str | None:
    try:
        marker = payload.decode("ascii").strip().lower()
    except UnicodeDecodeError:
        return None
    return (
        marker if project_gpu_leases.BOOT_ID_RE.fullmatch(marker) is not None else None
    )


def _process_start_marker(pid: int) -> str | None:
    if not _is_exact_int(pid, minimum=1):
        return None
    if sys.platform.startswith("linux"):
        try:
            payload = _read_procfs_regular_bounded(
                Path("/proc") / str(pid) / "stat",
                maximum_bytes=64 * 1024,
            )
            text = payload.decode("ascii")
        except (OSError, GeoAdaptV2Error, UnicodeDecodeError):
            return None
        return _parse_linux_proc_stat_start_marker(text, pid=pid)

    if pid == os.getpid():
        return _FALLBACK_PROCESS_START_MARKER
    try:
        result = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(pid)],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    marker = result.stdout.strip()
    return marker or None


def _boot_marker() -> str | None:
    if sys.platform.startswith("linux"):
        try:
            payload = _read_procfs_regular_bounded(
                Path("/proc/sys/kernel/random/boot_id"),
                maximum_bytes=128,
            )
        except (OSError, GeoAdaptV2Error):
            return None
        marker = _parse_linux_boot_id(payload)
        return (
            hashlib.sha256(marker.encode("ascii")).hexdigest()
            if marker is not None
            else None
        )

    try:
        result = subprocess.run(
            ["sysctl", "-n", "kern.boottime"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return _FALLBACK_BOOT_MARKER
    marker = result.stdout.strip()
    return (
        hashlib.sha256(marker.encode("utf-8")).hexdigest()
        if marker
        else _FALLBACK_BOOT_MARKER
    )


def _process_identity() -> dict[str, Any]:
    pid = os.getpid()
    start_marker = _process_start_marker(pid)
    boot_marker = _boot_marker()
    if start_marker is None or boot_marker is None:
        raise GeoAdaptV2Error(
            "cannot establish non-null process start and boot markers"
        )
    return {
        "host": socket.gethostname(),
        "pid": pid,
        "start_marker": start_marker,
        "boot_marker": boot_marker,
    }


def _owner_state(
    owner: Mapping[str, Any],
    *,
    created_at: str | None = None,
    foreign_timeout_seconds: float = 0.0,
) -> str:
    del created_at
    if foreign_timeout_seconds != 0.0:
        raise GeoAdaptV2Error(
            "claim timeout stealing is forbidden; timeout must remain zero"
        )
    try:
        host = str(owner["host"])
        pid = owner["pid"]
        marker = owner["start_marker"]
        boot_marker = owner["boot_marker"]
    except (KeyError, TypeError):
        return "unknown_frozen"
    if (
        not host
        or not _is_exact_int(pid, minimum=1)
        or not isinstance(marker, str)
        or not marker
        or not isinstance(boot_marker, str)
        or not boot_marker
    ):
        return "unknown_frozen"
    if host != socket.gethostname():
        return "foreign_frozen"
    live_boot_marker = _boot_marker()
    if live_boot_marker is None:
        return "unknown_frozen"
    if boot_marker != live_boot_marker:
        return "dead_local"
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return "dead_local"
    except (PermissionError, ValueError):
        return "unknown_frozen"
    observed = _process_start_marker(pid)
    if observed is None:
        return "unknown_frozen"
    if marker != observed:
        return "dead_local"
    return "live_local"


def _owner_is_live(
    owner: Mapping[str, Any],
    *,
    created_at: str | None = None,
    foreign_timeout_seconds: float = 0.0,
) -> bool:
    return (
        _owner_state(
            owner,
            created_at=created_at,
            foreign_timeout_seconds=foreign_timeout_seconds,
        )
        != "dead_local"
    )


def _coordination_lock_path(run_root: Path) -> Path:
    return run_root / ".geoadapt-v2-coordination.lock"


def _analysis_gate_path(run_root: Path) -> Path:
    return run_root / "analysis.gate.json"


def _validate_analysis_gate_payload(
    value: Any,
    *,
    plan: Mapping[str, Any],
) -> None:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"schema", "created_at", "plan_sha256", "nonce", "owner"}
        or value["schema"] != ANALYSIS_GATE_SCHEMA
        or not _is_utc_timestamp(value["created_at"])
        or value["plan_sha256"] != plan["plan_sha256"]
        or not _is_uuid4_hex(value["nonce"])
    ):
        raise GeoAdaptV2Error("analysis gate has an invalid exact schema")
    _validate_owner_mapping(value["owner"])


def _open_coordination_lock(run_root: Path, *, exclusive: bool) -> int:
    run_root = _safe_mkdir(run_root)
    path = _coordination_lock_path(run_root)
    parent = _open_directory_absolute(run_root)
    try:
        descriptor = os.open(
            path.name,
            _nofollow_flags(os.O_RDWR | os.O_CREAT | getattr(os, "O_NONBLOCK", 0)),
            0o600,
            dir_fd=parent,
        )
        os.fsync(parent)
    finally:
        os.close(parent)
    try:
        descriptor_status = os.fstat(descriptor)
        if not stat.S_ISREG(descriptor_status.st_mode):
            raise GeoAdaptV2Error("coordination lock is not a regular file")
        _require_single_link(descriptor_status, path)
        fcntl.flock(
            descriptor,
            fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH,
        )
        _assert_descriptor_matches_path(
            descriptor,
            path,
            kind="coordination lock",
        )
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


@contextlib.contextmanager
def _worker_claim_gate(run_root: Path) -> Iterator[None]:
    descriptor = _open_coordination_lock(run_root, exclusive=False)
    try:
        _assert_descriptor_matches_path(
            descriptor,
            _coordination_lock_path(_absolute_path(run_root)),
            kind="worker coordination lock",
        )
        marker = _analysis_gate_path(_absolute_path(run_root))
        if _path_exists(marker):
            raise ClaimUnavailable(
                "analysis/publication gate is active; worker claims are frozen"
            )
        yield
    finally:
        try:
            _assert_descriptor_matches_path(
                descriptor,
                _coordination_lock_path(_absolute_path(run_root)),
                kind="worker coordination lock",
            )
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)


def acquire_analysis_gate(
    run_root: Path,
    plan: Mapping[str, Any],
) -> AnalysisGate:
    """Freeze new worker claims through quiescence, score joining, and publish."""

    validate_production_semantics(plan)
    run_root = _assert_safe_path(run_root, leaf_kind="directory")
    descriptor = _open_coordination_lock(run_root, exclusive=True)
    marker = _analysis_gate_path(run_root)
    try:
        _quarantine_authority_stages(
            run_root,
            marker,
            category="analysis-gate-publications",
        )
        if _path_exists(marker):
            marker_status = _anchored_lstat(marker)
            marker_identity = (
                int(marker_status.st_dev),
                int(marker_status.st_ino),
            )
            try:
                observed = strict_load(marker)
                _validate_analysis_gate_payload(observed, plan=plan)
            except (GeoAdaptV2Error, OSError, ValueError):
                _quarantine(
                    run_root,
                    marker,
                    category="analysis-gates",
                    reason="malformed",
                    expected_identity=marker_identity,
                )
            else:
                state = _owner_state(observed["owner"])
                if state != "dead_local":
                    raise ClaimUnavailable(f"analysis gate owner is {state}")
                _quarantine(
                    run_root,
                    marker,
                    category="analysis-gates",
                    reason="powercut",
                    expected_identity=marker_identity,
                )
        nonce = uuid.uuid4().hex
        owner = _process_identity()
        _write_json_exclusive(
            marker,
            {
                "schema": ANALYSIS_GATE_SCHEMA,
                "created_at": _utc_now(),
                "plan_sha256": plan["plan_sha256"],
                "nonce": nonce,
                "owner": owner,
            },
        )
        marker_identity = _anchored_lstat(marker)
        return AnalysisGate(
            run_root=run_root,
            marker_path=marker,
            nonce=nonce,
            owner=owner,
            marker_st_dev=int(marker_identity.st_dev),
            marker_st_ino=int(marker_identity.st_ino),
            lock_descriptor=descriptor,
        )
    except Exception:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
        raise


def assert_analysis_gate(
    gate: AnalysisGate,
    run_root: Path,
    plan: Mapping[str, Any],
) -> None:
    validate_production_semantics(plan)
    expected_root = _assert_safe_path(run_root, leaf_kind="directory")
    if gate.run_root != expected_root or gate.marker_path != _analysis_gate_path(
        expected_root
    ):
        raise GeoAdaptV2Error("analysis gate belongs to another run root")
    _assert_descriptor_matches_path(
        gate.lock_descriptor,
        _coordination_lock_path(expected_root),
        kind="analysis coordination lock",
    )
    observed = strict_load(gate.marker_path)
    marker_identity = _anchored_lstat(gate.marker_path)
    _validate_analysis_gate_payload(observed, plan=plan)
    if (
        observed["nonce"] != gate.nonce
        or observed["owner"] != dict(gate.owner)
        or (marker_identity.st_dev, marker_identity.st_ino)
        != (gate.marker_st_dev, gate.marker_st_ino)
    ):
        raise GeoAdaptV2Error("analysis gate identity changed")


def release_analysis_gate(gate: AnalysisGate) -> None:
    try:
        _assert_descriptor_matches_path(
            gate.lock_descriptor,
            _coordination_lock_path(gate.run_root),
            kind="analysis coordination lock",
        )
        observed = strict_load(gate.marker_path)
        # The exact gate schema is checked without trusting values stored only in
        # the caller object. Its plan digest cannot be reconstructed here, but all
        # other fields and the nonce/owner identity remain mandatory.
        if (
            set(observed) != {"schema", "created_at", "plan_sha256", "nonce", "owner"}
            or observed["schema"] != ANALYSIS_GATE_SCHEMA
            or not _is_utc_timestamp(observed["created_at"])
            or not _is_sha256(observed["plan_sha256"])
            or not _is_uuid4_hex(observed["nonce"])
        ):
            raise GeoAdaptV2Error("analysis gate has an invalid exact schema")
        _validate_owner_mapping(observed["owner"])
        if observed["nonce"] != gate.nonce or observed["owner"] != dict(gate.owner):
            raise GeoAdaptV2Error("refusing to release another analysis gate")
        _unlink_owned_regular(
            gate.marker_path,
            expected_dev=gate.marker_st_dev,
            expected_ino=gate.marker_st_ino,
        )
    finally:
        fcntl.flock(gate.lock_descriptor, fcntl.LOCK_UN)
        os.close(gate.lock_descriptor)


@contextlib.contextmanager
def analysis_gate(
    run_root: Path,
    plan: Mapping[str, Any],
) -> Iterator[AnalysisGate]:
    gate = acquire_analysis_gate(run_root, plan)
    try:
        yield gate
    finally:
        release_analysis_gate(gate)


def probe_disk(path: Path, *, minimum_free_gib: float) -> dict[str, Any]:
    target = _absolute_path(path)
    while not _path_exists(target) and target != target.parent:
        target = target.parent
    target = _assert_safe_path(target, leaf_kind="directory")
    descriptor = _open_directory_absolute(target)
    try:
        usage = os.fstatvfs(descriptor)
    finally:
        os.close(descriptor)
    free_bytes = int(usage.f_bavail * usage.f_frsize)
    total_bytes = int(usage.f_blocks * usage.f_frsize)
    free_gib = free_bytes / 1024**3
    return {
        "safe": free_gib >= float(minimum_free_gib),
        "path": str(target),
        "free_bytes": free_bytes,
        "total_bytes": total_bytes,
        "free_gib": float(free_gib),
        "minimum_free_gib": float(minimum_free_gib),
    }


def _require_disk_floor(
    path: Path,
    *,
    plan: Mapping[str, Any],
    phase: str,
) -> dict[str, Any]:
    disk = probe_disk(
        path,
        minimum_free_gib=float(plan["execution_config"]["minimum_free_gib"]),
    )
    if disk["safe"] is not True:
        raise DiskUnavailable(f"hard disk floor failed immediately before {phase}")
    return disk


def _probe_worker_resource_guard(
    *,
    run_root: Path,
    plan: Mapping[str, Any],
    device: str,
    allow_active_owner: bool,
) -> dict[str, Any]:
    minimum = float(plan["execution_config"]["minimum_free_gib"])
    disk = probe_disk(run_root, minimum_free_gib=minimum)
    if not disk["safe"]:
        raise DiskUnavailable(
            f"{disk['free_gib']:.2f} GiB free is below the hard "
            f"{disk['minimum_free_gib']:.2f} GiB floor"
        )
    device_resource = _device_resource_identity(str(device))
    cuda = device_resource["node_inventory"]["cuda"]
    gpu_status = (
        None
        if cuda is None
        else _cooperative_gpu_status(
            _physical_gpu_uuid_from_device_resource(device_resource),
            allow_active_owner=allow_active_owner,
        )
    )
    if gpu_status is not None and gpu_status["safe"] is not True:
        raise GPUWorkerUnavailable(str(gpu_status["reason"]))
    return {
        "disk": disk,
        "requested_device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "worker_cpu_threads": int(plan["execution_config"]["worker_cpu_threads"]),
        "device_resource": device_resource,
        "gpu_status": gpu_status,
        "gpu_lease_receipt": None,
    }


def _bind_gpu_lease_receipt(
    resource_guard: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    run_root: Path,
    receipt: Mapping[str, Any] | None,
) -> dict[str, Any]:
    bound = copy.deepcopy(dict(resource_guard))
    bound["gpu_lease_receipt"] = (
        None if receipt is None else copy.deepcopy(dict(receipt))
    )
    _validate_resource_guard_mapping(bound, plan=plan, run_root=run_root)
    return bound


def _validate_device_resource_mapping(value: Any) -> None:
    if (
        not isinstance(value, Mapping)
        or set(value)
        != {
            "requested_device",
            "resolved_device",
            "cuda_visible_devices",
            "node_inventory",
        }
        or not isinstance(value["requested_device"], str)
        or not value["requested_device"]
        or not isinstance(value["resolved_device"], str)
        or not value["resolved_device"]
        or (
            value["cuda_visible_devices"] is not None
            and not isinstance(value["cuda_visible_devices"], str)
        )
    ):
        raise GeoAdaptV2Error("claim device resource has an invalid exact schema")
    node = value["node_inventory"]
    if (
        not isinstance(node, Mapping)
        or set(node) != {"host", "cuda"}
        or not isinstance(node["host"], str)
        or not node["host"]
    ):
        raise GeoAdaptV2Error("claim node inventory has an invalid exact schema")
    cuda = node["cuda"]
    if cuda is None:
        return
    if (
        not isinstance(cuda, Mapping)
        or set(cuda)
        != {
            "logical_index",
            "name",
            "capability",
            "total_memory_bytes",
            "cuda_visible_devices",
            "inventory",
        }
        or not _is_exact_int(cuda["logical_index"], minimum=0)
        or not isinstance(cuda["name"], str)
        or not cuda["name"]
        or not isinstance(cuda["capability"], list)
        or len(cuda["capability"]) != 2
        or any(not _is_exact_int(part, minimum=0) for part in cuda["capability"])
        or not _is_exact_int(cuda["total_memory_bytes"], minimum=1)
        or (
            cuda["cuda_visible_devices"] is not None
            and not isinstance(cuda["cuda_visible_devices"], str)
        )
        or not isinstance(cuda["inventory"], list)
    ):
        raise GeoAdaptV2Error("claim CUDA resource has an invalid exact schema")
    for row in cuda["inventory"]:
        if (
            not isinstance(row, Mapping)
            or set(row)
            != {
                "index",
                "uuid",
                "name",
                "driver_version",
                "memory_total_mib",
            }
            or not _is_exact_int(row["index"], minimum=0)
            or not _is_exact_int(row["memory_total_mib"], minimum=1)
            or any(
                not isinstance(row[key], str) or not row[key]
                for key in ("uuid", "name", "driver_version")
            )
        ):
            raise GeoAdaptV2Error(
                "claim CUDA inventory row has an invalid exact schema"
            )
    if cuda["cuda_visible_devices"] != value["cuda_visible_devices"]:
        raise GeoAdaptV2Error("claim CUDA visibility identities disagree")


def _validate_resource_guard_mapping(
    value: Any,
    *,
    plan: Mapping[str, Any],
    run_root: Path,
) -> None:
    canonical_run_root = _assert_safe_path(run_root, leaf_kind="directory")
    if (
        not isinstance(value, Mapping)
        or set(value)
        != {
            "disk",
            "requested_device",
            "cuda_visible_devices",
            "worker_cpu_threads",
            "device_resource",
            "gpu_status",
            "gpu_lease_receipt",
        }
        or not isinstance(value["requested_device"], str)
        or not value["requested_device"]
        or (
            value["cuda_visible_devices"] is not None
            and not isinstance(value["cuda_visible_devices"], str)
        )
        or not _is_exact_int(value["worker_cpu_threads"], minimum=1)
        or value["worker_cpu_threads"] != plan["execution_config"]["worker_cpu_threads"]
    ):
        raise GeoAdaptV2Error("claim resource guard has an invalid exact schema")
    disk = value["disk"]
    if (
        not isinstance(disk, Mapping)
        or set(disk)
        != {
            "safe",
            "path",
            "free_bytes",
            "total_bytes",
            "free_gib",
            "minimum_free_gib",
        }
        or disk["safe"] is not True
        or not isinstance(disk["path"], str)
        or not disk["path"]
        or _absolute_path(disk["path"]) != canonical_run_root
        or not _is_exact_int(disk["free_bytes"], minimum=0)
        or not _is_exact_int(disk["total_bytes"], minimum=1)
        or disk["free_bytes"] > disk["total_bytes"]
        or not isinstance(disk["free_gib"], float)
        or not math.isfinite(disk["free_gib"])
        or disk["free_gib"] < 0.0
        or not isinstance(disk["minimum_free_gib"], float)
        or not math.isfinite(disk["minimum_free_gib"])
        or disk["minimum_free_gib"] < 0.0
        or float(disk["minimum_free_gib"])
        != float(plan["execution_config"]["minimum_free_gib"])
    ):
        raise GeoAdaptV2Error("claim disk guard has an invalid exact schema")
    _validate_device_resource_mapping(value["device_resource"])
    resource = value["device_resource"]
    if (
        resource["requested_device"] != value["requested_device"]
        or resource["cuda_visible_devices"] != value["cuda_visible_devices"]
    ):
        raise GeoAdaptV2Error("claim resource guard identities disagree")
    resolved_is_cuda = str(resource["resolved_device"]).startswith("cuda")
    has_cuda_inventory = resource["node_inventory"]["cuda"] is not None
    if resolved_is_cuda != has_cuda_inventory:
        raise GeoAdaptV2Error(
            "claim resolved-device and CUDA inventory identities disagree"
        )
    if has_cuda_inventory:
        gpu_uuid = _physical_gpu_uuid_from_device_resource(resource)
        status = value["gpu_status"]
        if (
            not isinstance(status, Mapping)
            or set(status)
            != {
                "schema",
                "gpu_uuid",
                "utilization_percent",
                "memory_used_mib",
                "foreign_processes",
                "allow_active_owner",
                "utilization_limit_percent",
                "safe",
                "reason",
            }
            or status["schema"] != "ieee-mi-geoadapt-v2-cooperative-gpu-status-v1"
            or status["gpu_uuid"] != gpu_uuid
            or not isinstance(status["utilization_percent"], float)
            or not math.isfinite(status["utilization_percent"])
            or not 0.0 <= status["utilization_percent"] <= 100.0
            or not isinstance(status["memory_used_mib"], float)
            or not math.isfinite(status["memory_used_mib"])
            or status["memory_used_mib"] < 0.0
            or not isinstance(status["allow_active_owner"], bool)
            or not isinstance(status["utilization_limit_percent"], float)
            or status["utilization_limit_percent"]
            != (100.0 if status["allow_active_owner"] else 10.0)
            or status["safe"] is not True
            or not isinstance(status["reason"], str)
            or not status["reason"]
            or not isinstance(status["foreign_processes"], list)
            or status["foreign_processes"]
        ):
            raise GeoAdaptV2Error("claim cooperative GPU status is unsafe")
        receipt = value["gpu_lease_receipt"]
        try:
            project_gpu_leases.validate_gpu_lease_receipt(
                receipt,
                project_root=Path(__file__).resolve().parents[1],
                run_root=canonical_run_root,
                plan_sha256=str(plan["plan_sha256"]),
                gpu_uuid=gpu_uuid,
                track_scope=GPU_LEASE_TRACK_SCOPE,
            )
        except project_gpu_leases.ProjectGPULeaseError as error:
            raise GeoAdaptV2Error("claim GPU lease receipt is invalid") from error
    elif value["gpu_status"] is not None or value["gpu_lease_receipt"] is not None:
        raise GeoAdaptV2Error("non-CUDA claim cannot bind GPU status or lease")


def _load_validate_claim(
    path: Path,
    *,
    run_root: Path,
    plan: Mapping[str, Any],
    job: Job,
) -> dict[str, Any]:
    encoded = _safe_read_bytes(path)
    observed = _strict_json_bytes(encoded, source=str(path))
    if (
        set(observed)
        != {
            "schema",
            "created_at",
            "plan_sha256",
            "job_id",
            "job",
            "nonce",
            "owner",
            "resource_guard",
            "gpu_lease_receipt",
        }
        or observed["schema"] != CLAIM_SCHEMA
        or not _is_utc_timestamp(observed["created_at"])
        or observed["plan_sha256"] != plan["plan_sha256"]
        or observed["job_id"] != job.job_id
        or not _is_uuid4_hex(observed["nonce"])
    ):
        raise GeoAdaptV2Error("claim payload has an invalid exact schema or identity")
    _validate_job_mapping(observed["job"], expected=job)
    _validate_owner_mapping(observed["owner"])
    _validate_resource_guard_mapping(
        observed["resource_guard"],
        plan=plan,
        run_root=run_root,
    )
    if observed["gpu_lease_receipt"] != observed["resource_guard"]["gpu_lease_receipt"]:
        raise GeoAdaptV2Error("claim GPU lease receipt identities disagree")
    if (
        observed["owner"]["host"]
        != observed["resource_guard"]["device_resource"]["node_inventory"]["host"]
    ):
        raise GeoAdaptV2Error("claim owner and device resource hosts disagree")
    if encoded != _canonical_bytes(observed) + b"\n":
        raise GeoAdaptV2Error("claim payload is not canonical JSON")
    return observed


def _quarantine(
    run_root: Path,
    path: Path,
    *,
    category: str,
    reason: str,
    expected_identity: tuple[int, int],
) -> Path:
    run_root = _assert_safe_path(run_root, leaf_kind="directory")
    path = _assert_safe_path(path)
    if not _path_exists(path):
        raise FileNotFoundError(path)
    source_identity = _anchored_lstat(path)
    source_identity_pair = (
        int(source_identity.st_dev),
        int(source_identity.st_ino),
    )
    if source_identity_pair != tuple(map(int, expected_identity)):
        raise ClaimUnavailable(
            "quarantine source identity changed before invalidation"
        )
    try:
        origin_relative = str(path.relative_to(run_root))
    except ValueError as error:
        raise GeoAdaptV2Error(
            "quarantine source must be inside the run root"
        ) from error
    destination_root = run_root / "quarantine" / category
    if not PATH_COMPONENT_RE.fullmatch(category) or not PATH_COMPONENT_RE.fullmatch(
        reason
    ):
        raise GeoAdaptV2Error("unsafe quarantine category/reason")
    _safe_mkdir(destination_root)
    destination = (
        destination_root
        / f"{path.name}.{reason}.{int(time.time_ns())}.{uuid.uuid4().hex}"
    )
    _assert_safe_path(destination, allow_missing=True)
    source_is_directory = stat.S_ISDIR(source_identity.st_mode)
    if source_is_directory:
        descriptor = _open_directory_absolute(path)
        try:
            observed = os.fstat(descriptor)
            if (observed.st_dev, observed.st_ino) != (
                source_identity.st_dev,
                source_identity.st_ino,
            ):
                raise GeoAdaptV2Error(f"quarantine directory identity changed: {path}")
            os.fchmod(descriptor, 0o700)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _fsync_directory(path.parent)
    try:
        _rename_noreplace(
            path,
            destination,
            expected_source_dev=source_identity_pair[0],
            expected_source_ino=source_identity_pair[1],
        )
    except BaseException:
        # A no-replace helper may move the exact inode and then fail while
        # fsyncing or verifying. The canonical source is already hidden, so
        # finish this exact-inode quarantine instead of losing its ledger.
        if not _path_binds_identity(destination, source_identity_pair):
            raise
    if not _path_binds_identity(destination, source_identity_pair):
        raise GeoAdaptV2Error(
            "quarantine destination does not bind the validated source inode"
        )
    if source_is_directory:
        _seal_directory_read_only(
            destination,
            expected_dev=int(source_identity.st_dev),
            expected_ino=int(source_identity.st_ino),
        )
    _fsync_directory(path.parent)
    _fsync_directory(destination_root)
    plan_document = strict_load(run_root / "plan.json")
    plan_sha256 = plan_document.get("plan_sha256")
    if not _is_sha256(plan_sha256):
        raise GeoAdaptV2Error("cannot bind quarantine record to the run plan")
    payload_sha256 = _safe_tree_sha256(destination)
    ledger_root = run_root / "quarantine-ledger"
    _safe_mkdir(ledger_root)
    ledger_id = uuid.uuid4().hex
    _write_json_exclusive(
        ledger_root / f"{ledger_id}.json",
        {
            "schema": QUARANTINE_SCHEMA,
            "created_at": _utc_now(),
            "plan_sha256": plan_sha256,
            "ledger_id": ledger_id,
            "origin_relative_path": origin_relative,
            "destination_relative_path": str(destination.relative_to(run_root)),
            "category": category,
            "reason": reason,
            "payload_kind": (
                "directory"
                if stat.S_ISDIR(_anchored_lstat(destination).st_mode)
                else "file"
            ),
            "payload_sha256": payload_sha256,
        },
    )
    return destination


def _quarantine_authority_stages(
    run_root: Path,
    target: Path,
    *,
    category: str,
) -> tuple[Path, ...]:
    recovered: list[Path] = []
    for staged, staged_identity in _staged_publication_entries(target):
        try:
            recovered.append(
                _quarantine(
                    run_root,
                    staged,
                    category=category,
                    reason="powercut",
                    expected_identity=staged_identity,
                )
            )
        except FileNotFoundError:
            continue
    return tuple(recovered)


def _quarantine_partials(run_root: Path, job: Job) -> tuple[Path, ...]:
    root = run_root / "partials"
    recovered: list[Path] = []
    candidates: list[tuple[Path, str, str]] = []
    if _path_exists(root):
        _assert_safe_path(root, leaf_kind="directory")
        candidates.append((root, f"{job.job_id}.", ".partial"))
    destination = _record_directory(run_root, job)
    if _path_exists(destination.parent):
        candidates.append(
            (
                destination.parent,
                f".{destination.name}.",
                ".partial",
            )
        )
    for candidate_root, prefix, suffix in candidates:
        for name, entry_status in sorted(
            _anchored_directory_entries(candidate_root).items()
        ):
            if not name.startswith(prefix) or not name.endswith(suffix):
                continue
            path = candidate_root / name
            if stat.S_ISLNK(entry_status.st_mode):
                raise GeoAdaptV2Error(f"partial path is a symbolic link: {path}")
            if not (
                stat.S_ISREG(entry_status.st_mode) or stat.S_ISDIR(entry_status.st_mode)
            ):
                raise GeoAdaptV2Error(f"partial path is a special node: {path}")
            _require_single_link(entry_status, path)
            try:
                recovered.append(
                    _quarantine(
                        run_root,
                        path,
                        category="partials",
                        reason="recovered",
                        expected_identity=(
                            int(entry_status.st_dev),
                            int(entry_status.st_ino),
                        ),
                    )
                )
            except FileNotFoundError:
                continue
    return tuple(recovered)


def _recover_existing_claim_authority(
    run_root: Path,
    plan: Mapping[str, Any],
    job: Job,
    path: Path,
    *,
    recover_stale: bool,
    foreign_claim_timeout_seconds: float,
) -> None:
    with _authority_publication_lock(
        path,
        lock_root=run_root / ".authority-publication-locks" / "claims",
    ):
        _quarantine_authority_stages(
            run_root,
            path,
            category="claim-publications",
        )
        if not _path_exists(path):
            return
        claim_status = _anchored_lstat(path)
        if stat.S_ISLNK(claim_status.st_mode):
            raise GeoAdaptV2Error(f"symbolic links are forbidden: {path}")
        _require_single_link(claim_status, path)
        if not stat.S_ISREG(claim_status.st_mode):
            raise GeoAdaptV2Error(f"claim path is not a regular file: {path}")
        try:
            observed = _load_validate_claim(
                path,
                run_root=run_root,
                plan=plan,
                job=job,
            )
        except (GeoAdaptV2Error, OSError, ValueError) as error:
            if not recover_stale:
                raise ClaimUnavailable(f"{job.job_id} has a malformed claim") from error
            _quarantine(
                run_root,
                path,
                category="claims",
                reason="malformed",
                expected_identity=(
                    int(claim_status.st_dev),
                    int(claim_status.st_ino),
                ),
            )
            return
        owner_state = _owner_state(
            observed["owner"],
            created_at=observed["created_at"],
            foreign_timeout_seconds=foreign_claim_timeout_seconds,
        )
        if owner_state != "dead_local":
            raise ClaimUnavailable(
                f"{job.job_id} claim is {owner_state} and cannot be stolen"
            )
        if not recover_stale:
            raise ClaimUnavailable(f"{job.job_id} has a stale claim")
        _quarantine(
            run_root,
            path,
            category="claims",
            reason="stale",
            expected_identity=(
                int(claim_status.st_dev),
                int(claim_status.st_ino),
            ),
        )


def acquire_claim(
    run_root: Path,
    plan: Mapping[str, Any],
    job: Job,
    *,
    device: str,
    resource_guard: Mapping[str, Any] | None = None,
    gpu_lease: project_gpu_leases.GPULease | None = None,
    recover_stale: bool = True,
    foreign_claim_timeout_seconds: float = 0.0,
) -> Claim:
    _validate_job_identity(plan, job)
    if foreign_claim_timeout_seconds != 0.0:
        raise GeoAdaptV2Error(
            "foreign claim timeouts are forbidden; foreign/unknown claims freeze"
        )
    run_root = _assert_safe_path(run_root, leaf_kind="directory")
    output = _record_directory(run_root, job)
    if _path_exists(output):
        output_status = _anchored_lstat(output)
        output_identity = (
            int(output_status.st_dev),
            int(output_status.st_ino),
        )
        try:
            validate_completion(run_root, plan, job)
        except (GeoAdaptV2Error, OSError, ValueError):
            if not recover_stale:
                raise ClaimUnavailable(f"{job.job_id} has an invalid record")
            _quarantine(
                run_root,
                output,
                category="records",
                reason="invalid",
                expected_identity=output_identity,
            )
        else:
            if recover_stale:
                recover_completed_job(
                    run_root,
                    plan,
                    job,
                    foreign_claim_timeout_seconds=(foreign_claim_timeout_seconds),
                )
            raise ClaimUnavailable(f"{job.job_id} is already complete")
    path = _claim_path(run_root, job)
    _recover_existing_claim_authority(
        run_root,
        plan,
        job,
        path,
        recover_stale=recover_stale,
        foreign_claim_timeout_seconds=foreign_claim_timeout_seconds,
    )
    _quarantine_partials(run_root, job)
    nonce = uuid.uuid4().hex
    owner = _process_identity()
    if resource_guard is None:
        unbound = _probe_worker_resource_guard(
            run_root=run_root,
            plan=plan,
            device=str(device),
            allow_active_owner=False,
        )
        guard = _bind_gpu_lease_receipt(
            unbound,
            plan=plan,
            run_root=run_root,
            receipt=None,
        )
    else:
        guard = copy.deepcopy(dict(resource_guard))
        _validate_resource_guard_mapping(
            guard,
            plan=plan,
            run_root=run_root,
        )
        if guard["requested_device"] != str(device):
            raise GeoAdaptV2Error("claim device differs from probed resource guard")
    lease_receipt = guard["gpu_lease_receipt"]
    payload = {
        "schema": CLAIM_SCHEMA,
        "created_at": _utc_now(),
        "plan_sha256": plan["plan_sha256"],
        "job_id": job.job_id,
        "job": job.identity(),
        "nonce": nonce,
        "owner": owner,
        "resource_guard": guard,
        "gpu_lease_receipt": copy.deepcopy(lease_receipt),
    }
    claim_published = False
    try:
        with _guard_bound_gpu_lease(gpu_lease, lease_receipt):
            with _authority_publication_lock(
                path,
                lock_root=run_root / ".authority-publication-locks" / "claims",
            ):
                _quarantine_authority_stages(
                    run_root,
                    path,
                    category="claim-publications",
                )
                if _path_exists(path):
                    raise ClaimUnavailable(f"{job.job_id} lost its claim race")
                try:
                    _write_json_exclusive(path, payload)
                except FileExistsError as error:
                    raise ClaimUnavailable(
                        f"{job.job_id} lost its claim race"
                    ) from error
            claim_published = True
            identity = _anchored_lstat(path)
    except Exception:
        if claim_published and _path_exists(path):
            _quarantine(
                run_root,
                path,
                category="claims",
                reason="leaselost",
                expected_identity=(int(identity.st_dev), int(identity.st_ino)),
            )
        raise
    return Claim(
        job=job,
        path=path,
        nonce=nonce,
        owner=owner,
        resource_guard=guard,
        gpu_lease_receipt=lease_receipt,
        st_dev=int(identity.st_dev),
        st_ino=int(identity.st_ino),
    )


def release_claim(claim: Claim, plan: Mapping[str, Any]) -> None:
    run_root = _absolute_path(claim.path).parents[2]
    output = _record_directory(run_root, claim.job)

    def invalidate_owned_record(*, reason: str) -> bool:
        expected = claim.publication_state.record_identity
        if expected is None:
            if _path_exists(output):
                raise GeoAdaptV2Error(
                    "claim loss found a record not published by this invocation; "
                    "replacement left untouched"
                )
            return False
        if not _path_exists(output):
            claim.publication_state.record_identity = None
            return False
        try:
            _quarantine(
                run_root,
                output,
                category="records",
                reason=reason,
                expected_identity=expected,
            )
        except ClaimUnavailable as error:
            claim.publication_state.record_identity = None
            raise GeoAdaptV2Error(
                "owned record inode was replaced during claim release; "
                "replacement left untouched"
            ) from error
        claim.publication_state.record_identity = None
        return True

    if not _path_exists(claim.path):
        if invalidate_owned_record(reason="claimlostrelease"):
            raise GeoAdaptV2Error(
                "owned claim disappeared after record commit; record quarantined"
            )
        return
    try:
        observed = _load_validate_claim(
            claim.path,
            run_root=run_root,
            plan=plan,
            job=claim.job,
        )
    except Exception:
        invalidate_owned_record(reason="claimreplacedrelease")
        raise
    if observed.get("nonce") != claim.nonce or observed.get("owner") != dict(
        claim.owner
    ):
        invalidate_owned_record(reason="claimreplacedrelease")
        raise GeoAdaptV2Error("refusing to release another worker's claim")
    _unlink_owned_regular(
        claim.path,
        expected_dev=claim.st_dev,
        expected_ino=claim.st_ino,
    )
    claim.publication_state.record_identity = None


def _recursive_forbidden_keys(value: Any, prefix: str = "record") -> list[str]:
    safe_normalized = {
        "scoreblind",
        "labelsmoothing",
        "selectmetric",
        "testoutcomesavailabletobackend",
        "testperformancecomputed",
    }
    suspicious_fragments = (
        "label",
        "target",
        "outcome",
        "truth",
        "metric",
        "score",
        "groundtruth",
        "ytrue",
        "ytest",
        "testlabel",
        "classlabel",
        "predictedlabel",
        "predictionlabel",
        "actualoutcome",
        "validationloss",
        "validationbalancedaccuracy",
        "accuracy",
        "kappa",
        "auc",
    )
    violations: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}"
            normalized = re.sub(r"[^a-z0-9]+", "", str(key).casefold())
            if normalized not in safe_normalized and (
                normalized in FORBIDDEN_SCORE_BLIND_KEY_ALIASES
                or any(fragment in normalized for fragment in suspicious_fragments)
            ):
                violations.append(path)
            violations.extend(_recursive_forbidden_keys(child, path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            violations.extend(_recursive_forbidden_keys(child, f"{prefix}[{index}]"))
    return violations


def _validate_metadata(
    plan: Mapping[str, Any],
    job: Job,
    metadata: Mapping[str, Any],
) -> None:
    required = {
        "cache_array_sha256",
        "channels",
        "positions_sha256",
        "split",
        "scalers",
        "features",
        "fit",
        "protocol",
        "timing_seconds",
        "runtime",
    }
    if not isinstance(metadata, Mapping) or set(metadata) != required:
        raise GeoAdaptV2Error("record metadata has an invalid exact schema")
    if (
        metadata["cache_array_sha256"] != _planned_cache(plan, job)["array_sha256"]
        or metadata["split"] != _planned_split(plan, job)
        or not _is_sha256(metadata["positions_sha256"])
    ):
        raise GeoAdaptV2Error("record metadata cache/split identity is invalid")
    if (
        not isinstance(metadata["channels"], list)
        or not metadata["channels"]
        or any(
            not isinstance(value, str) or not value for value in metadata["channels"]
        )
        or len(metadata["channels"]) != len(set(metadata["channels"]))
    ):
        raise GeoAdaptV2Error("record channel identity is invalid")
    scalers = metadata["scalers"]
    if (
        not isinstance(scalers, Mapping)
        or set(scalers)
        != {
            "selection_mean_sha256",
            "selection_std_sha256",
            "refit_mean_sha256",
            "refit_std_sha256",
        }
        or not all(_is_sha256(value) for value in scalers.values())
    ):
        raise GeoAdaptV2Error("record scaler provenance is invalid")
    features = metadata["features"]
    expected_features = {
        "selection_train_scaled_sha256",
        "selection_validation_scaled_sha256",
        "refit_source_scaled_sha256",
        "prediction_test_scaled_sha256",
    }
    if job.stable_id == "architecture.geoadaptnet":
        expected_features |= {
            "selection_train_spd_sha256",
            "selection_validation_spd_sha256",
            "refit_source_spd_sha256",
            "prediction_test_spd_sha256",
        }
    if (
        not isinstance(features, Mapping)
        or set(features) != expected_features
        or not all(_is_sha256(value) for value in features.values())
    ):
        raise GeoAdaptV2Error("record feature provenance is invalid")
    fit = metadata["fit"]
    required_fit = {
        "stable_config",
        "seed_installed_before_construction",
        "initial_state_sha256",
        "selection_state_sha256",
        "reset_state_sha256",
        "reset_verified",
        "source_selected_epoch_index",
        "selection_epochs_run",
        "refit_epochs_run",
        "refit_state_sha256",
        "parameter_count",
        "cuda_peak_memory_bytes",
    }
    if (
        not isinstance(fit, Mapping)
        or set(fit) != required_fit
        or fit["stable_config"] != plan["stable_configs"][job.stable_id]
        or fit["seed_installed_before_construction"] is not True
        or fit["reset_verified"] is not True
        or fit["initial_state_sha256"] != fit["reset_state_sha256"]
        or not all(
            _is_sha256(fit[key])
            for key in (
                "initial_state_sha256",
                "selection_state_sha256",
                "reset_state_sha256",
                "refit_state_sha256",
            )
        )
        or not _is_exact_int(fit["source_selected_epoch_index"], minimum=0)
        or not _is_exact_int(fit["selection_epochs_run"], minimum=1)
        or not _is_exact_int(fit["refit_epochs_run"], minimum=1)
        or fit["refit_epochs_run"] != fit["source_selected_epoch_index"] + 1
        or not _is_exact_int(fit["parameter_count"], minimum=1)
        or not _is_exact_int(fit["cuda_peak_memory_bytes"], minimum=0)
    ):
        raise GeoAdaptV2Error("record fit/reset/refit provenance is invalid")
    expected_protocol = {
        "selection_scaler_rows": "train_only",
        "selection_optimization_rows": "train_only",
        "epoch_selection_rows": "validation_only",
        "selection_decision": (
            "minimum_unsmoothed_validation_cross_entropy_not_persisted"
        ),
        "refit_scaler_rows": "train_plus_validation",
        "refit_optimization_rows": "train_plus_validation",
        "test_use": "single_probability_prediction_call_only",
        "test_outcomes_available_to_backend": False,
        "test_performance_computed": False,
        "transductive_calibration": False,
    }
    if (
        not isinstance(metadata["protocol"], Mapping)
        or dict(metadata["protocol"]) != expected_protocol
    ):
        raise GeoAdaptV2Error("record source/test protocol is invalid")
    timings = metadata["timing_seconds"]
    if not isinstance(timings, Mapping) or set(timings) != {
        "selection_fit",
        "refit_fit",
        "test_inference",
        "job_total",
    }:
        raise GeoAdaptV2Error("record timing schema is invalid")
    if any(
        not isinstance(value, float) or not math.isfinite(value) or value < 0.0
        for value in timings.values()
    ):
        raise GeoAdaptV2Error("record timing value is invalid")
    runtime = metadata["runtime"]
    if (
        not isinstance(runtime, Mapping)
        or set(runtime)
        != {
            "device",
            "portable_compute_contract_sha256",
            "torch_determinism",
            "worker_cpu_threads",
            "worker_interop_threads",
            "thread_environment",
            "node_inventory",
        }
        or runtime.get("portable_compute_contract_sha256")
        != _sha256_bytes(_canonical_bytes(plan["environment_identity"]))
        or not _is_exact_int(runtime.get("worker_cpu_threads"), minimum=1)
        or runtime["worker_cpu_threads"]
        != plan["execution_config"]["worker_cpu_threads"]
        or not _is_exact_int(runtime.get("worker_interop_threads"), minimum=1)
        or not isinstance(runtime.get("device"), str)
        or not runtime["device"]
        or not isinstance(runtime.get("thread_environment"), Mapping)
        or set(runtime["thread_environment"]) != set(THREAD_ENVIRONMENT_VARIABLES)
        or any(
            value != str(plan["execution_config"]["worker_cpu_threads"])
            for value in runtime["thread_environment"].values()
        )
        or not isinstance(runtime.get("node_inventory"), Mapping)
        or set(runtime["node_inventory"]) != {"host", "cuda"}
        or not isinstance(runtime["node_inventory"].get("host"), str)
        or not runtime["node_inventory"]["host"]
    ):
        raise GeoAdaptV2Error("record runtime provenance is invalid")
    runtime_cuda = runtime["node_inventory"]["cuda"]
    _validate_device_resource_mapping(
        {
            "requested_device": runtime["device"],
            "resolved_device": runtime["device"],
            "cuda_visible_devices": (
                None
                if runtime_cuda is None
                else runtime_cuda.get("cuda_visible_devices")
            ),
            "node_inventory": runtime["node_inventory"],
        }
    )
    determinism = runtime["torch_determinism"]
    if (
        not isinstance(determinism, Mapping)
        or set(determinism)
        != {
            "deterministic_algorithms",
            "warn_only",
            "cudnn_deterministic",
            "cudnn_benchmark",
            "cuda_matmul_tf32",
            "cudnn_tf32",
            "float32_matmul_precision",
            "cublas_workspace_config",
            "cuda_available",
        }
        or not isinstance(determinism["cuda_available"], bool)
        or (
            determinism["cublas_workspace_config"] is not None
            and not isinstance(determinism["cublas_workspace_config"], str)
        )
    ):
        raise GeoAdaptV2Error("record Torch determinism provenance is invalid")
    for key in (
        "deterministic_algorithms",
        "warn_only",
        "cudnn_deterministic",
        "cudnn_benchmark",
        "cuda_matmul_tf32",
        "cudnn_tf32",
        "float32_matmul_precision",
    ):
        if (
            key != "float32_matmul_precision"
            and not isinstance(determinism.get(key), bool)
        ) or determinism.get(key) != DETERMINISTIC_TORCH_CONTRACT[key]:
            raise GeoAdaptV2Error(f"record Torch determinism setting drifted: {key}")
    if (
        determinism.get("cuda_available")
        and determinism.get("cublas_workspace_config")
        != DETERMINISTIC_TORCH_CONTRACT["cublas_workspace_config_on_cuda"]
    ):
        raise GeoAdaptV2Error("record deterministic cuBLAS setting is invalid")


def _validate_claim_payload(
    run_root: Path,
    plan: Mapping[str, Any],
    claim: Claim,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    run_root = _assert_safe_path(run_root, leaf_kind="directory")
    expected_path = _absolute_path(_claim_path(run_root, claim.job))
    if _absolute_path(claim.path) != expected_path:
        raise GeoAdaptV2Error("claim path is outside the frozen job identity")
    observed = _load_validate_claim(
        claim.path,
        run_root=run_root,
        plan=plan,
        job=claim.job,
    )
    path_identity = _anchored_lstat(claim.path)
    if (
        observed["nonce"] != claim.nonce
        or observed["owner"] != dict(claim.owner)
        or observed["resource_guard"] != dict(claim.resource_guard)
        or observed["gpu_lease_receipt"]
        != (None if claim.gpu_lease_receipt is None else dict(claim.gpu_lease_receipt))
        or (path_identity.st_dev, path_identity.st_ino) != (claim.st_dev, claim.st_ino)
    ):
        raise GeoAdaptV2Error("job claim object differs from its immutable payload")
    guard = observed["resource_guard"]
    expected_threads = str(plan["execution_config"]["worker_cpu_threads"])
    if os.environ.get("CUDA_VISIBLE_DEVICES") != guard["cuda_visible_devices"] or any(
        os.environ.get(name) != expected_threads
        for name in THREAD_ENVIRONMENT_VARIABLES
    ):
        raise GeoAdaptV2Error(
            "live CUDA visibility or worker-thread resource differs from claim"
        )
    live_resource = _device_resource_identity(guard["requested_device"])
    if live_resource != guard["device_resource"]:
        raise GeoAdaptV2Error("live device resource differs from claimed resource")
    if metadata is not None:
        runtime = metadata.get("runtime")
        if (
            not isinstance(runtime, Mapping)
            or runtime.get("device") != live_resource["resolved_device"]
            or runtime.get("node_inventory") != live_resource["node_inventory"]
        ):
            raise GeoAdaptV2Error(
                "record runtime device inventory differs from claimed live resource"
            )


def commit_job_output(
    run_root: Path,
    plan: Mapping[str, Any],
    claim: Claim,
    *,
    cache_root: Path | None = None,
    metadata: Mapping[str, Any],
    test_rows: np.ndarray,
    probabilities: np.ndarray,
    resource_recheck: Mapping[str, Any] | None = None,
    commit_gpu_lease_receipt: Mapping[str, Any] | None = None,
    gpu_lease: project_gpu_leases.GPULease | None = None,
) -> Path:
    run_root = _assert_safe_path(run_root, leaf_kind="directory")
    cache_root = _assert_safe_path(
        run_root if cache_root is None else cache_root,
        leaf_kind="directory",
    )
    job = claim.job
    if claim.publication_state.record_identity is not None:
        raise GeoAdaptV2Error("claim already carries a published record identity")
    _validate_job_identity(plan, job)
    rows = np.asarray(test_rows)
    values = np.asarray(probabilities)
    expected = _planned_split(plan, job)
    test = expected["partitions"]["test"]
    if (
        rows.dtype != np.dtype(np.int64)
        or rows.ndim != 1
        or len(rows) != int(test["count"])
        or _rows_sha256(rows) != test["rows_sha256"]
        or values.dtype != np.dtype(np.float64)
        or values.shape != (len(rows), 2)
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(values > 1.0)
        or not np.allclose(values.sum(axis=1), 1.0, rtol=0.0, atol=1e-6)
    ):
        raise GeoAdaptV2Error("invalid score-blind prediction arrays")
    _validate_metadata(plan, job, metadata)
    # Re-read the immutable claim and freshly resolve the live device immediately
    # before opening the partial output directory.
    _validate_claim_payload(
        run_root,
        plan,
        claim,
        metadata=metadata,
    )
    if resource_recheck is None:
        unbound_commit_resource = _probe_worker_resource_guard(
            run_root=run_root,
            plan=plan,
            device=str(claim.resource_guard["requested_device"]),
            allow_active_owner=True,
        )
        commit_resource = _bind_gpu_lease_receipt(
            unbound_commit_resource,
            plan=plan,
            run_root=run_root,
            receipt=commit_gpu_lease_receipt,
        )
    else:
        commit_resource = copy.deepcopy(dict(resource_recheck))
        _validate_resource_guard_mapping(
            commit_resource,
            plan=plan,
            run_root=run_root,
        )
    claimed_receipt = claim.resource_guard["gpu_lease_receipt"]
    if (
        commit_gpu_lease_receipt != claimed_receipt
        or commit_resource["gpu_lease_receipt"] != claimed_receipt
    ):
        raise GeoAdaptV2Error(
            "commit must reassert the exact GPU lease bound by the claim"
        )
    if (
        commit_resource["requested_device"] != claim.resource_guard["requested_device"]
        or commit_resource["device_resource"]["resolved_device"]
        != claim.resource_guard["device_resource"]["resolved_device"]
    ):
        raise GeoAdaptV2Error("commit device differs from claimed device")
    _assert_bound_gpu_lease(gpu_lease, commit_gpu_lease_receipt)
    record = {
        "schema": RECORD_SCHEMA,
        "created_at": _utc_now(),
        "plan_sha256": plan["plan_sha256"],
        "track": TRACK,
        "job_id": job.job_id,
        "job": job.identity(),
        "score_blind": True,
        "claim_resource_guard": copy.deepcopy(dict(claim.resource_guard)),
        "commit_resource_guard": copy.deepcopy(commit_resource),
        "commit_gpu_lease_receipt": copy.deepcopy(commit_gpu_lease_receipt),
        "test_count": len(rows),
        "test_rows_sha256": _rows_sha256(rows),
        "metadata": copy.deepcopy(dict(metadata)),
    }
    violations = _recursive_forbidden_keys(record)
    if violations:
        raise GeoAdaptV2Error(
            f"score-blind record contains forbidden fields: {violations[:5]}"
        )
    partial_root = run_root / "partials"
    _safe_mkdir(partial_root)
    partial = partial_root / f"{job.job_id}.{claim.nonce}.partial"
    _safe_mkdir(partial, parents=False)
    partial_identity = _anchored_lstat(partial)
    record_path = partial / "record.json"
    prediction_path = partial / "predictions.npz"
    completion_path = partial / "completion.json"
    destination = _record_directory(run_root, job)
    publication_stage = (
        destination.parent / f".{destination.name}.{claim.nonce}.partial"
    )
    staged_identity = (
        int(partial_identity.st_dev),
        int(partial_identity.st_ino),
    )
    published_identity: tuple[int, int] | None = None
    try:
        # The hard disk floor is refreshed at the last possible point before
        # result bytes are created. CUDA commits then reassert the exact lease.
        immediate_disk = _require_disk_floor(
            run_root,
            plan=plan,
            phase="record write",
        )
        record["commit_resource_guard"]["disk"] = immediate_disk
        _validate_resource_guard_mapping(
            record["commit_resource_guard"],
            plan=plan,
            run_root=run_root,
        )
        _assert_bound_gpu_lease(gpu_lease, commit_gpu_lease_receipt)
        _write_json_exclusive(record_path, record)
        _require_disk_floor(
            run_root,
            plan=plan,
            phase="prediction write",
        )
        partial_descriptor = _open_directory_absolute(partial)
        try:
            descriptor = os.open(
                prediction_path.name,
                _nofollow_flags(
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NONBLOCK", 0)
                ),
                0o600,
                dir_fd=partial_descriptor,
            )
            try:
                descriptor_status = os.fstat(descriptor)
                if not stat.S_ISREG(descriptor_status.st_mode):
                    raise GeoAdaptV2Error("prediction artifact is not a regular file")
                _require_single_link(descriptor_status, prediction_path)
                with os.fdopen(descriptor, "wb", closefd=False) as handle:
                    np.savez_compressed(
                        handle,
                        rows=np.ascontiguousarray(rows),
                        probabilities=np.ascontiguousarray(values),
                    )
                    handle.flush()
                    os.fsync(handle.fileno())
                os.fchmod(descriptor, 0o444)
            finally:
                os.close(descriptor)
            os.fsync(partial_descriptor)
        finally:
            os.close(partial_descriptor)
        completion = {
            "schema": COMPLETION_SCHEMA,
            "created_at": _utc_now(),
            "plan_sha256": plan["plan_sha256"],
            "job_id": job.job_id,
            "job": job.identity(),
            "files": {
                "record.json": _sha256_file(record_path),
                "predictions.npz": _sha256_file(prediction_path),
            },
        }
        _require_disk_floor(
            run_root,
            plan=plan,
            phase="completion write",
        )
        _write_json_exclusive(completion_path, completion)
        _fsync_directory(partial)
        _safe_mkdir(destination.parent)
        _assert_safe_path(
            publication_stage,
            leaf_kind="directory",
            allow_missing=True,
        )
        _rename_noreplace(
            partial,
            publication_stage,
            expected_source_dev=int(partial_identity.st_dev),
            expected_source_ino=int(partial_identity.st_ino),
        )
        partial = publication_stage
        _seal_directory_read_only(
            partial,
            expected_dev=int(partial_identity.st_dev),
            expected_ino=int(partial_identity.st_ino),
        )
        _assert_safe_path(destination, leaf_kind="directory", allow_missing=True)
        _validate_claim_payload(
            run_root,
            plan,
            claim,
            metadata=metadata,
        )
        commit_cuda = commit_resource["device_resource"]["node_inventory"]["cuda"]
        if commit_cuda is not None:
            final_gpu_status = _cooperative_gpu_status(
                _physical_gpu_uuid_from_device_resource(
                    commit_resource["device_resource"]
                ),
                allow_active_owner=True,
            )
            if final_gpu_status["safe"] is not True:
                raise GPUWorkerUnavailable(str(final_gpu_status["reason"]))
        rename_conflict = False
        with _guard_bound_gpu_lease(gpu_lease, commit_gpu_lease_receipt):
            _validate_claim_payload(
                run_root,
                plan,
                claim,
                metadata=metadata,
            )
            _require_disk_floor(
                run_root,
                plan=plan,
                phase="record commit",
            )
            _rebind_authoritative_inputs(
                run_root=run_root,
                plan=plan,
                cache_root=cache_root,
                job=job,
                claim=claim,
                metadata=metadata,
                gpu_lease=gpu_lease,
                expected_gpu_lease_receipt=commit_gpu_lease_receipt,
            )
            try:
                _rename_noreplace(
                    partial,
                    destination,
                    expected_source_dev=staged_identity[0],
                    expected_source_ino=staged_identity[1],
                )
            except BaseException as publication_error:
                if _path_binds_identity(destination, staged_identity):
                    # Publication identity is established independently of the
                    # helper's return: it may move, fsync, and then raise.
                    published_identity = staged_identity
                    claim.publication_state.record_identity = published_identity
                    _rebind_authoritative_inputs(
                        run_root=run_root,
                        plan=plan,
                        cache_root=cache_root,
                        job=job,
                        claim=claim,
                        metadata=metadata,
                        gpu_lease=gpu_lease,
                        expected_gpu_lease_receipt=commit_gpu_lease_receipt,
                    )
                    raise
                if isinstance(publication_error, FileExistsError):
                    rename_conflict = True
                else:
                    raise
            else:
                published_identity = staged_identity
                claim.publication_state.record_identity = published_identity
                _rebind_authoritative_inputs(
                    run_root=run_root,
                    plan=plan,
                    cache_root=cache_root,
                    job=job,
                    claim=claim,
                    metadata=metadata,
                    gpu_lease=gpu_lease,
                    expected_gpu_lease_receipt=commit_gpu_lease_receipt,
                )
                _fsync_directory(destination.parent)
        if rename_conflict:
            validate_completion(run_root, plan, job)
            _quarantine(
                run_root,
                partial,
                category="partials",
                reason="duplicate",
                expected_identity=(
                    int(partial_identity.st_dev),
                    int(partial_identity.st_ino),
                ),
            )
            return destination
        try:
            _validate_claim_payload(
                run_root,
                plan,
                claim,
                metadata=metadata,
            )
            _assert_bound_gpu_lease(gpu_lease, commit_gpu_lease_receipt)
        except Exception:
            _quarantine(
                run_root,
                destination,
                category="records",
                reason="claimlost",
                expected_identity=published_identity,
            )
            published_identity = None
            claim.publication_state.record_identity = None
            raise
        validate_completion(run_root, plan, job)
        _validate_claim_payload(
            run_root,
            plan,
            claim,
            metadata=metadata,
        )
        _assert_bound_gpu_lease(gpu_lease, commit_gpu_lease_receipt)
        return destination
    except BaseException:
        if published_identity is not None and _path_exists(destination):
            try:
                _quarantine(
                    run_root,
                    destination,
                    category="records",
                    reason="failed",
                    expected_identity=published_identity,
                )
            except (FileNotFoundError, ClaimUnavailable):
                # A replacement is a competing winner and must survive.
                pass
            finally:
                published_identity = None
                claim.publication_state.record_identity = None
        if _path_exists(partial):
            try:
                _quarantine(
                    run_root,
                    partial,
                    category="partials",
                    reason="failed",
                    expected_identity=(
                        int(partial_identity.st_dev),
                        int(partial_identity.st_ino),
                    ),
                )
            except (FileNotFoundError, ClaimUnavailable):
                pass
        raise


def _capture_completion_snapshot(
    run_root: Path, plan: Mapping[str, Any], job: Job
) -> CompletionSnapshot:
    """Validate and decode one completion from one coherent held-FD snapshot."""

    run_root = _assert_safe_path(run_root, leaf_kind="directory")
    _validate_job_identity(plan, job)
    directory = _record_directory(run_root, job)
    directory_descriptor = _open_directory_absolute(directory)
    descriptors: dict[str, int] = {}
    artifact_bytes: dict[str, bytes] = {}
    entry_fingerprints: dict[str, tuple[int, ...]] = {}
    try:
        directory_before = os.fstat(directory_descriptor)
        if (
            not stat.S_ISDIR(directory_before.st_mode)
            or stat.S_IMODE(directory_before.st_mode) != 0o555
        ):
            raise GeoAdaptV2Error(
                f"{directory} is not an immutable 0555 completion directory"
            )
        directory_fingerprint = _completion_stat_fingerprint(directory_before)
        initial_names = tuple(sorted(os.listdir(directory_descriptor)))
        entries = {
            name: os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
            for name in initial_names
        }
        if set(entries) != FINAL_FILENAMES:
            raise GeoAdaptV2Error(f"{directory} has an invalid file set")
        for name in sorted(FINAL_FILENAMES):
            status = entries[name]
            _require_single_link(status, directory / name)
            if not stat.S_ISREG(status.st_mode) or status.st_mode & 0o222:
                raise GeoAdaptV2Error(
                    f"{directory / name} is not a unique read-only completion artifact"
                )
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
            fingerprint = _completion_stat_fingerprint(status)
            if (
                not stat.S_ISREG(held.st_mode)
                or held.st_nlink != 1
                or held.st_mode & 0o222
                or _completion_stat_fingerprint(held) != fingerprint
            ):
                raise GeoAdaptV2Error(
                    f"{directory / name} changed while opening completion package"
                )
            entry_fingerprints[name] = fingerprint

        # Rebind every canonical child name before reading any bytes. The
        # descriptors all remain open until the complete package is read.
        before_read_entries = tuple(
            (
                name,
                _completion_stat_fingerprint(
                    os.stat(
                        name,
                        dir_fd=directory_descriptor,
                        follow_symlinks=False,
                    )
                ),
            )
            for name in sorted(os.listdir(directory_descriptor))
        )
        expected_entries = tuple(sorted(entry_fingerprints.items()))
        if (
            before_read_entries != expected_entries
            or _completion_stat_fingerprint(os.fstat(directory_descriptor))
            != directory_fingerprint
        ):
            raise GeoAdaptV2Error(
                f"{directory} changed before completion package read"
            )

        for name in sorted(FINAL_FILENAMES):
            chunks: list[bytes] = []
            while True:
                try:
                    chunk = os.read(descriptors[name], 1024 * 1024)
                except InterruptedError:
                    continue
                if not chunk:
                    break
                chunks.append(chunk)
            payload = b"".join(chunks)
            expected_size = entry_fingerprints[name][
                _COMPLETION_STABLE_FIELDS.index("st_size")
            ]
            if len(payload) != expected_size:
                raise GeoAdaptV2Error(
                    f"{directory / name} completion package read was incomplete"
                )
            artifact_bytes[name] = payload

        # Rebind every still-held descriptor to its exact canonical name only
        # after all bytes are read. Full directory and sorted-entry stat
        # fingerprints reject coherent A/B/A package substitution.
        after_read_entries: list[tuple[str, tuple[int, ...]]] = []
        for name in sorted(os.listdir(directory_descriptor)):
            path_after = os.stat(
                name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            path_fingerprint = _completion_stat_fingerprint(path_after)
            after_read_entries.append((name, path_fingerprint))
            descriptor = descriptors.get(name)
            if (
                descriptor is None
                or _completion_stat_fingerprint(os.fstat(descriptor))
                != entry_fingerprints.get(name)
                or path_fingerprint != entry_fingerprints.get(name)
            ):
                raise GeoAdaptV2Error(
                    f"{directory / name} changed during completion package read"
                )
        directory_after = os.fstat(directory_descriptor)
        canonical_after = _anchored_lstat(directory)
        if (
            tuple(after_read_entries) != expected_entries
            or _completion_stat_fingerprint(directory_after)
            != directory_fingerprint
            or _completion_stat_fingerprint(canonical_after)
            != directory_fingerprint
        ):
            raise GeoAdaptV2Error(
                f"{directory} changed while completion package was validated"
            )
    finally:
        for descriptor in descriptors.values():
            os.close(descriptor)
        os.close(directory_descriptor)
    completion_bytes = artifact_bytes["completion.json"]
    record_bytes = artifact_bytes["record.json"]
    prediction_bytes = artifact_bytes["predictions.npz"]
    completion = _strict_json_bytes(
        completion_bytes,
        source=str(directory / "completion.json"),
    )
    if completion_bytes != _canonical_bytes(completion) + b"\n":
        raise GeoAdaptV2Error(f"{directory} completion JSON is not canonical")
    if (
        set(completion)
        != {
            "schema",
            "created_at",
            "plan_sha256",
            "job_id",
            "job",
            "files",
        }
        or completion["schema"] != COMPLETION_SCHEMA
        or not _is_utc_timestamp(completion["created_at"])
        or completion["plan_sha256"] != plan["plan_sha256"]
        or completion["job_id"] != job.job_id
        or not isinstance(completion["files"], Mapping)
        or set(completion["files"]) != {"record.json", "predictions.npz"}
        or not all(_is_sha256(value) for value in completion["files"].values())
    ):
        raise GeoAdaptV2Error(
            f"{directory} completion has an invalid exact schema or type"
        )
    _validate_job_mapping(completion["job"], expected=job)
    if dict(completion["files"]) != {
        "record.json": _sha256_bytes(record_bytes),
        "predictions.npz": _sha256_bytes(prediction_bytes),
    }:
        raise GeoAdaptV2Error(f"{directory} completion checksums are invalid")
    record = _strict_json_bytes(
        record_bytes,
        source=str(directory / "record.json"),
    )
    if record_bytes != _canonical_bytes(record) + b"\n":
        raise GeoAdaptV2Error(f"{directory} record JSON is not canonical")
    if (
        set(record)
        != {
            "schema",
            "created_at",
            "plan_sha256",
            "track",
            "job_id",
            "job",
            "score_blind",
            "claim_resource_guard",
            "commit_resource_guard",
            "commit_gpu_lease_receipt",
            "test_count",
            "test_rows_sha256",
            "metadata",
        }
        or record["schema"] != RECORD_SCHEMA
        or not _is_utc_timestamp(record["created_at"])
        or record["plan_sha256"] != plan["plan_sha256"]
        or record["track"] != TRACK
        or record["job_id"] != job.job_id
        or record["score_blind"] is not True
        or not _is_exact_int(record["test_count"], minimum=1)
        or not _is_sha256(record["test_rows_sha256"])
    ):
        raise GeoAdaptV2Error(f"{directory} record has an invalid exact schema or type")
    _validate_job_mapping(record["job"], expected=job)
    _validate_resource_guard_mapping(
        record["claim_resource_guard"],
        plan=plan,
        run_root=run_root,
    )
    _validate_resource_guard_mapping(
        record["commit_resource_guard"],
        plan=plan,
        run_root=run_root,
    )
    if (
        record["commit_gpu_lease_receipt"]
        != record["claim_resource_guard"]["gpu_lease_receipt"]
        or record["commit_gpu_lease_receipt"]
        != record["commit_resource_guard"]["gpu_lease_receipt"]
        or record["claim_resource_guard"]["requested_device"]
        != record["commit_resource_guard"]["requested_device"]
        or record["claim_resource_guard"]["device_resource"]["resolved_device"]
        != record["commit_resource_guard"]["device_resource"]["resolved_device"]
    ):
        raise GeoAdaptV2Error("record GPU lease/resource binding changed")
    violations = _recursive_forbidden_keys(record)
    if violations:
        raise GeoAdaptV2Error(f"{directory} violates score-blind schema")
    metadata = record.get("metadata")
    if not isinstance(metadata, Mapping):
        raise GeoAdaptV2Error(f"{directory} has no metadata object")
    _validate_metadata(plan, job, metadata)
    try:
        with zipfile.ZipFile(io.BytesIO(prediction_bytes), mode="r") as raw_archive:
            raw_members = [entry.filename for entry in raw_archive.infolist()]
            if raw_members != ["rows.npy", "probabilities.npy"]:
                raise GeoAdaptV2Error(
                    "prediction archive has an invalid exact ZIP member order"
                )
        with np.load(io.BytesIO(prediction_bytes), allow_pickle=False) as archive:
            if archive.files != ["rows", "probabilities"]:
                raise GeoAdaptV2Error(
                    "prediction archive has an invalid exact array order"
                )
            rows = archive["rows"].copy()
            probabilities = archive["probabilities"].copy()
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise GeoAdaptV2Error("prediction archive cannot be loaded") from error
    expected = _planned_split(plan, job)
    test = expected["partitions"]["test"]
    if (
        rows.dtype != np.dtype(np.int64)
        or rows.ndim != 1
        or len(rows) != int(record["test_count"])
        or len(rows) != int(test["count"])
        or _rows_sha256(rows) != record["test_rows_sha256"]
        or _rows_sha256(rows) != test["rows_sha256"]
        or probabilities.dtype != np.dtype(np.float64)
        or probabilities.shape != (len(rows), 2)
        or not np.all(np.isfinite(probabilities))
        or np.any(probabilities < 0.0)
        or np.any(probabilities > 1.0)
        or not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-6)
    ):
        raise GeoAdaptV2Error(f"{directory} prediction arrays are invalid")
    rows = np.ascontiguousarray(rows, dtype=np.int64)
    probabilities = np.ascontiguousarray(probabilities, dtype=np.float64)
    rows.setflags(write=False)
    probabilities.setflags(write=False)
    return CompletionSnapshot(
        record=record,
        completion=completion,
        rows=rows,
        probabilities=probabilities,
        artifact_bytes=dict(artifact_bytes),
        artifact_sha256={
            name: _sha256_bytes(payload)
            for name, payload in sorted(artifact_bytes.items())
        },
        directory_fingerprint=directory_fingerprint,
        entry_fingerprints=tuple(sorted(entry_fingerprints.items())),
    )


def validate_completion(
    run_root: Path, plan: Mapping[str, Any], job: Job
) -> dict[str, Any]:
    snapshot = _capture_completion_snapshot(run_root, plan, job)
    return {
        "record": snapshot.record,
        "completion": snapshot.completion,
        "rows": snapshot.rows,
        "probabilities": snapshot.probabilities,
        "artifact_bytes": snapshot.artifact_bytes,
        "artifact_sha256": snapshot.artifact_sha256,
    }


def recover_completed_job(
    run_root: Path,
    plan: Mapping[str, Any],
    job: Job,
    *,
    foreign_claim_timeout_seconds: float = 0.0,
) -> tuple[Path, ...]:
    """Recover stale claim/partial residue only after validating completion."""

    if foreign_claim_timeout_seconds != 0.0:
        raise GeoAdaptV2Error("foreign claim timeout stealing is forbidden")
    validate_completion(run_root, plan, job)
    recovered: list[Path] = []
    claim_path = _claim_path(run_root, job)
    if _path_exists(claim_path):
        claim_status = _anchored_lstat(claim_path)
        try:
            observed = _load_validate_claim(
                claim_path,
                run_root=run_root,
                plan=plan,
                job=job,
            )
        except (GeoAdaptV2Error, OSError, ValueError) as error:
            raise ClaimUnavailable(
                "valid completion has a malformed claim; frozen for manual review"
            ) from error
        owner_state = _owner_state(
            observed["owner"],
            created_at=observed["created_at"],
            foreign_timeout_seconds=foreign_claim_timeout_seconds,
        )
        if owner_state != "dead_local":
            raise ClaimUnavailable(
                f"valid completion still has a {owner_state} claim owner"
            )
        recovered.append(
            _quarantine(
                run_root,
                claim_path,
                category="claims",
                reason="postcommit",
                expected_identity=(
                    int(claim_status.st_dev),
                    int(claim_status.st_ino),
                ),
            )
        )
    recovered.extend(_quarantine_partials(run_root, job))
    return tuple(recovered)


def _failure_root(run_root: Path, job: Job) -> Path:
    return run_root / "failures" / job.dataset / job.stable_id


def _record_failure(
    run_root: Path,
    plan: Mapping[str, Any],
    claim: Claim,
    error: BaseException,
) -> Path:
    root = _failure_root(run_root, claim.job)
    _safe_mkdir(root)
    authority = (
        run_root / ".failure-publication-locks" / f"{claim.job.job_id}.authority"
    )
    with _authority_publication_lock(authority):
        attempt_numbers: list[int] = []
        prefix = f"{claim.job.job_id}.attempt-"
        final_pattern = re.compile(rf"^{re.escape(prefix)}([0-9]{{4}})\.json$")
        stage_pattern = re.compile(
            rf"^\.(?P<target>{re.escape(prefix)}[0-9]{{4}}\.json)"
            r"\.publish-[0-9a-f]{32}\.staged$"
        )
        for name, observed in sorted(_anchored_directory_entries(root).items()):
            stage_match = stage_pattern.fullmatch(name)
            if stage_match is not None:
                staged = root / name
                if not stat.S_ISREG(observed.st_mode):
                    raise GeoAdaptV2Error(
                        f"failure staging path is not regular: {staged}"
                    )
                _require_single_link(observed, staged)
                _quarantine(
                    run_root,
                    staged,
                    category="failure-publications",
                    reason="powercut",
                    expected_identity=(
                        int(observed.st_dev),
                        int(observed.st_ino),
                    ),
                )
                continue
            final_match = final_pattern.fullmatch(name)
            if final_match is None:
                continue
            path = root / name
            if not stat.S_ISREG(observed.st_mode):
                raise GeoAdaptV2Error(f"failure authority is not regular: {path}")
            _require_single_link(observed, path)
            try:
                _validate_failure_record(path, plan=plan, job=claim.job)
            except (GeoAdaptV2Error, OSError, ValueError):
                _quarantine(
                    run_root,
                    path,
                    category="failures",
                    reason="malformed",
                    expected_identity=(
                        int(observed.st_dev),
                        int(observed.st_ino),
                    ),
                )
                continue
            attempt_numbers.append(int(final_match.group(1)))
        attempt = max(attempt_numbers, default=0) + 1
        path = root / f"{claim.job.job_id}.attempt-{attempt:04d}.json"
        payload = {
            "schema": FAILURE_SCHEMA,
            "created_at": _utc_now(),
            "plan_sha256": plan["plan_sha256"],
            "job_id": claim.job.job_id,
            "job": claim.job.identity(),
            "owner": dict(claim.owner),
            "error_type": type(error).__name__,
            "error_message": str(error)[:4000],
        }
        _write_json_exclusive(path, payload)
        _validate_failure_record(path, plan=plan, job=claim.job)
        return path


def _validate_failure_record(
    path: Path,
    *,
    plan: Mapping[str, Any],
    job: Job,
) -> dict[str, Any]:
    encoded = _safe_read_bytes(path)
    payload = _strict_json_bytes(encoded, source=str(path))
    if (
        set(payload)
        != {
            "schema",
            "created_at",
            "plan_sha256",
            "job_id",
            "job",
            "owner",
            "error_type",
            "error_message",
        }
        or payload["schema"] != FAILURE_SCHEMA
        or not _is_utc_timestamp(payload["created_at"])
        or payload["plan_sha256"] != plan["plan_sha256"]
        or payload["job_id"] != job.job_id
        or not isinstance(payload["error_type"], str)
        or not payload["error_type"]
        or not isinstance(payload["error_message"], str)
        or encoded != _canonical_bytes(payload) + b"\n"
    ):
        raise GeoAdaptV2Error("failure record has an invalid exact schema")
    _validate_job_mapping(payload["job"], expected=job)
    _validate_owner_mapping(payload["owner"])
    return payload


def _audit_quarantine(
    run_root: Path,
    plan: Mapping[str, Any],
) -> tuple[int, tuple[str, ...]]:
    ledger_root = run_root / "quarantine-ledger"
    payload_root = run_root / "quarantine"
    ledger_paths: list[Path] = []
    if _path_exists(ledger_root):
        _assert_safe_path(ledger_root, leaf_kind="directory")
        ledger_paths = [
            path
            for path in _walk_safe_tree(ledger_root)
            if stat.S_ISREG(_anchored_lstat(path).st_mode)
        ]
        if any(path.parent != ledger_root for path in ledger_paths):
            return 0, ("nested quarantine ledger path",)
    referenced: set[str] = set()
    errors: list[str] = []
    for path in sorted(ledger_paths):
        try:
            encoded = _safe_read_bytes(path)
            payload = _strict_json_bytes(encoded, source=str(path))
            expected_keys = {
                "schema",
                "created_at",
                "plan_sha256",
                "ledger_id",
                "origin_relative_path",
                "destination_relative_path",
                "category",
                "reason",
                "payload_kind",
                "payload_sha256",
            }
            if (
                set(payload) != expected_keys
                or payload["schema"] != QUARANTINE_SCHEMA
                or not _is_utc_timestamp(payload["created_at"])
                or payload["plan_sha256"] != plan["plan_sha256"]
                or not _is_uuid4_hex(payload["ledger_id"])
                or path.name != f"{payload['ledger_id']}.json"
                or not isinstance(payload["origin_relative_path"], str)
                or not payload["origin_relative_path"]
                or not isinstance(payload["destination_relative_path"], str)
                or not payload["destination_relative_path"]
                or not isinstance(payload["category"], str)
                or PATH_COMPONENT_RE.fullmatch(payload["category"]) is None
                or not isinstance(payload["reason"], str)
                or PATH_COMPONENT_RE.fullmatch(payload["reason"]) is None
                or payload["payload_kind"] not in {"file", "directory"}
                or not _is_sha256(payload["payload_sha256"])
                or encoded != _canonical_bytes(payload) + b"\n"
            ):
                raise GeoAdaptV2Error("invalid quarantine ledger schema")
            destination_relative = Path(payload["destination_relative_path"])
            origin_relative = Path(payload["origin_relative_path"])
            if (
                destination_relative.is_absolute()
                or ".." in destination_relative.parts
                or origin_relative.is_absolute()
                or ".." in origin_relative.parts
                or len(destination_relative.parts) < 3
                or destination_relative.parts[:2] != ("quarantine", payload["category"])
            ):
                raise GeoAdaptV2Error("invalid quarantine path binding")
            destination_key = str(destination_relative)
            if destination_key in referenced:
                raise GeoAdaptV2Error("duplicate quarantine destination ledger")
            destination = _assert_safe_path(run_root / destination_relative)
            destination_status = _anchored_lstat(destination)
            actual_kind = (
                "directory" if stat.S_ISDIR(destination_status.st_mode) else "file"
            )
            if (
                actual_kind != payload["payload_kind"]
                or _safe_tree_sha256(destination) != payload["payload_sha256"]
            ):
                raise GeoAdaptV2Error("quarantine payload changed")
            referenced.add(destination_key)
        except (GeoAdaptV2Error, OSError, ValueError, KeyError, TypeError) as error:
            errors.append(f"{path.relative_to(run_root)}: {error}")
    observed: set[str] = set()
    if _path_exists(payload_root):
        _assert_safe_path(payload_root, leaf_kind="directory")
        for category_name, category_status in sorted(
            _anchored_directory_entries(payload_root).items()
        ):
            if not stat.S_ISDIR(category_status.st_mode):
                errors.append(f"unsafe quarantine category node: {category_name}")
                continue
            category_path = _assert_safe_path(
                payload_root / category_name, leaf_kind="directory"
            )
            if PATH_COMPONENT_RE.fullmatch(category_name) is None:
                errors.append(f"unsafe quarantine category: {category_name}")
                continue
            for entry_name, entry_status in sorted(
                _anchored_directory_entries(category_path).items()
            ):
                entry_path = category_path / entry_name
                if not (
                    stat.S_ISREG(entry_status.st_mode)
                    or stat.S_ISDIR(entry_status.st_mode)
                ):
                    errors.append(f"unsafe quarantine payload: {entry_path}")
                    continue
                _require_single_link(entry_status, entry_path)
                observed.add(str(entry_path.relative_to(run_root)))
    if observed != referenced:
        errors.append(
            "quarantine payload/ledger sets differ: "
            f"unledgered={sorted(observed - referenced)} "
            f"missing={sorted(referenced - observed)}"
        )
    return len(referenced), tuple(errors)


def audit_grid(run_root: Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    validate_production_semantics(plan)
    run_root = _assert_safe_path(run_root, leaf_kind="directory", allow_missing=True)
    jobs = tuple(iter_jobs(plan))
    expected_records = {_record_directory(run_root, job): job for job in jobs}
    valid: list[str] = []
    missing: list[str] = []
    corrupt: list[str] = []
    for path, job in expected_records.items():
        if not _path_exists(path):
            missing.append(job.job_id)
            continue
        try:
            validate_completion(run_root, plan, job)
        except (GeoAdaptV2Error, OSError, ValueError):
            corrupt.append(job.job_id)
        else:
            valid.append(job.job_id)

    observed_record_dirs: set[Path] = set()
    record_publication_partials: set[Path] = set()
    records_root = run_root / "records"
    if _path_exists(records_root):
        for path in _walk_safe_tree(records_root):
            observed = _anchored_lstat(path)
            if (
                stat.S_ISDIR(observed.st_mode)
                and path.name.startswith(".")
                and path.name.endswith(".partial")
            ):
                record_publication_partials.add(path)
            elif stat.S_ISREG(observed.st_mode):
                observed_record_dirs.add(path.parent)
    unknown_records = sorted(
        str(path.relative_to(run_root))
        for path in (
            observed_record_dirs - set(expected_records) - record_publication_partials
        )
    )

    expected_claims = {_claim_path(run_root, job): job for job in jobs}
    live_claims: list[str] = []
    stale_claims: list[str] = []
    unknown_claims: list[str] = []
    claims_root = run_root / "claims"
    if _path_exists(claims_root):
        for path in sorted(
            value
            for value in _walk_safe_tree(claims_root)
            if stat.S_ISREG(_anchored_lstat(value).st_mode)
        ):
            job = expected_claims.get(path)
            if job is None:
                unknown_claims.append(str(path.relative_to(run_root)))
                continue
            try:
                claim = _load_validate_claim(
                    path,
                    run_root=run_root,
                    plan=plan,
                    job=job,
                )
            except (GeoAdaptV2Error, OSError, ValueError):
                unknown_claims.append(str(path.relative_to(run_root)))
                continue
            target = (
                live_claims
                if _owner_is_live(
                    claim["owner"],
                    created_at=claim["created_at"],
                )
                else stale_claims
            )
            target.append(job.job_id)
    partials: list[str] = []
    partial_root = run_root / "partials"
    if _path_exists(partial_root):
        _assert_safe_path(partial_root, leaf_kind="directory")
        for entry_name, entry_status in _anchored_directory_entries(
            partial_root
        ).items():
            if stat.S_ISLNK(entry_status.st_mode) or not (
                stat.S_ISREG(entry_status.st_mode) or stat.S_ISDIR(entry_status.st_mode)
            ):
                raise GeoAdaptV2Error(
                    "partials contain an alias or special node: "
                    f"{partial_root / entry_name}"
                )
            _require_single_link(entry_status, partial_root / entry_name)
            partials.append(str((partial_root / entry_name).relative_to(run_root)))
    partials.extend(
        str(path.relative_to(run_root)) for path in sorted(record_publication_partials)
    )
    partials.sort()
    jobs_by_id = {job.job_id: job for job in jobs}
    failure_records = 0
    invalid_failures: list[str] = []
    failures_root = run_root / "failures"
    if _path_exists(failures_root):
        for path in _walk_safe_tree(failures_root):
            if not stat.S_ISREG(_anchored_lstat(path).st_mode):
                continue
            match = re.fullmatch(
                r"(geoadapt-v2-[0-9a-f]{24})\.attempt-([0-9]{4})\.json",
                path.name,
            )
            job = jobs_by_id.get(match.group(1)) if match is not None else None
            if (
                job is None
                or path.parent != _failure_root(run_root, job)
                or int(match.group(2)) <= 0
            ):
                invalid_failures.append(str(path.relative_to(run_root)))
                continue
            try:
                _validate_failure_record(path, plan=plan, job=job)
            except (GeoAdaptV2Error, OSError, ValueError):
                invalid_failures.append(str(path.relative_to(run_root)))
            else:
                failure_records += 1
    quarantine_records, invalid_quarantine = _audit_quarantine(run_root, plan)
    quiescent = (
        not live_claims and not stale_claims and not unknown_claims and not partials
    )
    exact = (
        len(valid) == len(jobs)
        and not missing
        and not corrupt
        and not unknown_records
        and not invalid_failures
        and not invalid_quarantine
        and quiescent
    )
    return {
        "schema": AUDIT_SCHEMA,
        "created_at": _utc_now(),
        "plan_sha256": plan["plan_sha256"],
        "track": TRACK,
        "expected_jobs": len(jobs),
        "valid_records": len(valid),
        "missing_records": len(missing),
        "corrupt_records": len(corrupt),
        "unknown_records": len(unknown_records),
        "live_claims": len(live_claims),
        "stale_claims": len(stale_claims),
        "unknown_claims": len(unknown_claims),
        "partials": len(partials),
        "failure_records": failure_records,
        "invalid_failures": len(invalid_failures),
        "quarantine_records": quarantine_records,
        "invalid_quarantine": len(invalid_quarantine),
        "quiescent": quiescent,
        "exact_cartesian_complete": exact,
        "details": {
            "missing_job_ids": missing,
            "corrupt_job_ids": corrupt,
            "unknown_record_paths": unknown_records,
            "live_claim_job_ids": live_claims,
            "stale_claim_job_ids": stale_claims,
            "unknown_claim_paths": unknown_claims,
            "partial_paths": partials,
            "invalid_failure_paths": invalid_failures,
            "invalid_quarantine": list(invalid_quarantine),
        },
    }


def _configure_worker_threads(count: int) -> None:
    value = str(int(count))
    for name in THREAD_ENVIRONMENT_VARIABLES:
        observed = os.environ.get(name)
        if observed != value:
            raise GeoAdaptV2Error(
                f"{name}={observed!r} differs from frozen worker value {value!r}"
            )
    import torch

    torch.set_num_threads(int(count))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise


def run_worker(
    *,
    run_root: Path,
    plan: Mapping[str, Any],
    cache_root: Path,
    device: str,
    stable_ids: Sequence[str] | None = None,
    max_jobs: int | None = None,
    recover_stale: bool = True,
    foreign_claim_timeout_seconds: float = 0.0,
    backend: ProcedureBackend = fit_procedure_backend,
) -> dict[str, Any]:
    validate_production_semantics(plan)
    run_root = _assert_safe_path(run_root, leaf_kind="directory")
    if foreign_claim_timeout_seconds != 0.0:
        raise GeoAdaptV2Error(
            "foreign claim timeouts are forbidden; foreign/unknown claims freeze"
        )
    _configure_worker_threads(int(plan["execution_config"]["worker_cpu_threads"]))
    selected = (
        set(plan["stable_id_order"])
        if stable_ids is None
        else {str(value) for value in stable_ids}
    )
    if not selected or not selected.issubset(set(plan["stable_id_order"])):
        raise ValueError("worker stable-ID filter is invalid")
    completed = 0
    skipped = 0
    failed = 0
    with _project_gpu_worker_lease(
        run_root=run_root,
        plan=plan,
        device=device,
    ) as gpu_lease:
        for job in iter_jobs(plan):
            if job.stable_id not in selected:
                continue
            if max_jobs is not None and completed >= max_jobs:
                break
            try:
                with _worker_claim_gate(run_root):
                    unbound_claim_resource = _probe_worker_resource_guard(
                        run_root=run_root,
                        plan=plan,
                        device=device,
                        allow_active_owner=False,
                    )
                    claim_lease_receipt = (
                        None
                        if gpu_lease is None
                        else _assert_bound_gpu_lease(
                            gpu_lease,
                            gpu_lease_receipt(gpu_lease),
                        )
                    )
                    claim_resource = _bind_gpu_lease_receipt(
                        unbound_claim_resource,
                        plan=plan,
                        run_root=run_root,
                        receipt=claim_lease_receipt,
                    )
                    claim = acquire_claim(
                        run_root,
                        plan,
                        job,
                        device=device,
                        resource_guard=claim_resource,
                        gpu_lease=gpu_lease,
                        recover_stale=recover_stale,
                        foreign_claim_timeout_seconds=(foreign_claim_timeout_seconds),
                    )
            except ClaimUnavailable:
                skipped += 1
                continue
            try:
                metadata, rows, probabilities = execute_job(
                    job=job,
                    plan=plan,
                    cache_root=cache_root,
                    device=device,
                    backend=backend,
                )
                unbound_commit_resource = _probe_worker_resource_guard(
                    run_root=run_root,
                    plan=plan,
                    device=device,
                    allow_active_owner=True,
                )
                live_commit_receipt = (
                    None
                    if gpu_lease is None
                    else _assert_bound_gpu_lease(
                        gpu_lease,
                        claim.gpu_lease_receipt,
                    )
                )
                commit_resource = _bind_gpu_lease_receipt(
                    unbound_commit_resource,
                    plan=plan,
                    run_root=run_root,
                    receipt=live_commit_receipt,
                )
                commit_job_output(
                    run_root,
                    plan,
                    claim,
                    cache_root=cache_root,
                    metadata=metadata,
                    test_rows=rows,
                    probabilities=probabilities,
                    resource_recheck=commit_resource,
                    commit_gpu_lease_receipt=live_commit_receipt,
                    gpu_lease=gpu_lease,
                )
                completed += 1
            except Exception as error:
                _record_failure(run_root, plan, claim, error)
                failed += 1
                raise
            finally:
                release_claim(claim, plan)
                try:
                    import gc
                    import torch

                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:  # pragma: no cover - production requires torch
                    pass
    return {
        "completed_this_worker": completed,
        "skipped_this_worker": skipped,
        "failed_this_worker": failed,
        "audit": audit_grid(run_root, plan),
    }


def _parse_stable_ids(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    result = tuple(token.strip() for token in value.split(",") if token.strip())
    if not result or len(result) != len(set(result)):
        raise ValueError("--stable-ids must be unique comma-separated values")
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="freeze the exact production plan")
    plan.add_argument("--run-root", required=True)
    plan.add_argument("--cache-root", required=True)
    plan.add_argument("--minimum-free-gib", type=float, default=DEFAULT_MIN_FREE_GIB)
    plan.add_argument(
        "--worker-cpu-threads", type=int, default=DEFAULT_WORKER_CPU_THREADS
    )

    worker = subparsers.add_parser("worker", help="run dynamically claimed jobs")
    worker.add_argument("--run-root", required=True)
    worker.add_argument("--cache-root", required=True)
    worker.add_argument("--device", required=True)
    worker.add_argument("--stable-ids")
    worker.add_argument("--max-jobs", type=int)
    worker.add_argument(
        "--no-recover-stale", action="store_true", help="fail on stale residue"
    )
    worker.add_argument(
        "--skip-full-cache-preflight",
        action="store_true",
        help="verify each cache at job load; analyzer still performs a full audit",
    )

    audit = subparsers.add_parser("audit", help="audit exact completion/quiescence")
    audit.add_argument("--run-root", required=True)
    audit.add_argument("--output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "plan":
        run_root = _absolute_path(args.run_root)
        expected = build_plan(
            cache_root=_absolute_path(args.cache_root),
            minimum_free_gib=args.minimum_free_gib,
            worker_cpu_threads=args.worker_cpu_threads,
        )
        plan = write_or_validate_plan(run_root, expected)
        print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
        return 0

    run_root = _absolute_path(args.run_root)
    plan = load_plan(run_root)
    verify_static_identity(plan)
    if args.command == "worker":
        if args.max_jobs is not None and (
            isinstance(args.max_jobs, bool) or args.max_jobs <= 0
        ):
            raise ValueError("--max-jobs must be positive")
        cache_root = _absolute_path(args.cache_root)
        if not args.skip_full_cache_preflight:
            verify_cache_and_splits(plan, cache_root=cache_root)
        try:
            result = run_worker(
                run_root=run_root,
                plan=plan,
                cache_root=cache_root,
                device=args.device,
                stable_ids=_parse_stable_ids(args.stable_ids),
                max_jobs=args.max_jobs,
                recover_stale=not args.no_recover_stale,
                foreign_claim_timeout_seconds=0.0,
            )
        except GPUWorkerUnavailable as error:
            print(str(error), file=sys.stderr, flush=True)
            return 75
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return 0 if result["failed_this_worker"] == 0 else 1

    report = audit_grid(run_root, plan)
    if args.output:
        _atomic_json(_absolute_path(args.output), report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if report["exact_cartesian_complete"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ANALYSIS_GATE_SCHEMA",
    "AnalysisGate",
    "AUDIT_SCHEMA",
    "BNCI001_NA_REASON",
    "CLAIM_SCHEMA",
    "COMPLETION_SCHEMA",
    "CompletionSnapshot",
    "DATASET_FOLDS",
    "DATASET_ORDER",
    "DATASET_SUBJECTS",
    "DETERMINISTIC_TORCH_CONTRACT",
    "EVIDENCE_SCOPE",
    "FBSP_BNCI004_NA_REASON",
    "GEOADAPT_BNCI004_NA_REASON",
    "FILTER_BANDS_HZ",
    "GPUWorkerUnavailable",
    "Job",
    "NotApplicableError",
    "PLAN_SCHEMA",
    "PreparedJob",
    "RECORD_SCHEMA",
    "SEEDS",
    "STABLE_CONFIGS",
    "STABLE_IDS",
    "TRACK",
    "acquire_claim",
    "acquire_analysis_gate",
    "analysis_gate",
    "assemble_plan",
    "audit_grid",
    "build_plan",
    "commit_job_output",
    "derive_four_band_spd",
    "execute_job",
    "fixed_filter_bank",
    "fit_procedure_backend",
    "iter_jobs",
    "load_plan",
    "main",
    "prepare_job",
    "recover_completed_job",
    "release_analysis_gate",
    "release_claim",
    "run_worker",
    "spd_covariances",
    "validate_completion",
    "validate_production_semantics",
    "verify_cache_and_splits",
    "verify_static_identity",
    "write_or_validate_plan",
]
