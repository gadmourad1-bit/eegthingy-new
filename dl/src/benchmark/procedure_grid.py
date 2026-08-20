"""Score-blind harmonized-v2 runner for the frozen binary neural procedures.

This module is deliberately separate from the common-training-recipe grid.
It executes only four procedures whose configuration existed as an immutable,
outcome-free file before this adapter was written:

* ``architecture.cameo``;
* ``architecture.hemiparity``;
* ``architecture.parity_fuse``; and
* ``architecture.orbit_v3``.

ORBIT-v1/v2/v4/v5 remain blocked.  Their historical result artifacts are useful
provenance, but are not eligible runtime configuration sources.  The
``inventory`` command reports that boundary without opening or extracting an
outcome from those artifacts.

Each atomic job performs validation-only epoch/route selection, reconstructs a
fresh estimator from the same random initialization, refits source-only
preprocessing and weights on train+validation for the selected duration, and
calls ``predict_proba`` exactly once on the held-out rows.  The writer publishes
row indices and probabilities but no held-out labels or score.  A separate
module, :mod:`benchmark.procedure_grid_analysis`, is the only score-joining layer.

The runner uses exclusive filesystem claims, power-cut-recoverable partial
directories, immutable checksummed completions, cooperative GPU/disk guards,
and an exact quiescent audit.  Production commands require a UV-managed virtual
environment and never install or modify a package.
"""

from __future__ import annotations

import argparse
import copy
import csv
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
import re
import socket
import stat
import subprocess
import sys
import time
import uuid
import zipfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import project_gpu_leases
from .config import (
    LOCAL_EXP4_SUBJECT_RUNS,
    PROJECT_ROOT,
    coordinate_contract_for_dataset,
    dataset_spec,
    preprocessing_for_dataset,
)

PLAN_SCHEMA = "eeg-mi-binary-procedure-grid-plan-v1"
RECORD_SCHEMA = "eeg-mi-binary-procedure-prediction-record-v1"
COMPLETION_SCHEMA = "eeg-mi-binary-procedure-completion-v1"
CLAIM_SCHEMA = "eeg-mi-binary-procedure-claim-v1"
CLAIM_RECEIPT_SCHEMA = "eeg-mi-binary-procedure-claim-receipt-v1"
FAILURE_SCHEMA = "eeg-mi-binary-procedure-failure-v1"
AUDIT_SCHEMA = "eeg-mi-binary-procedure-audit-v1"
VIEW_SCHEMA = "eeg-mi-paired-broadband-spd-tangent-v1"

BINARY_DATASETS: tuple[str, ...] = (
    "local_exp4",
    "bnci2014_004",
    "cho2017",
    "physionet_mi",
)
DATASET_FOLDS: Mapping[str, tuple[int, ...]] = {
    "local_exp4": (0,),
    "bnci2014_004": (0,),
    "cho2017": (0, 1, 2, 3, 4),
    "physionet_mi": (0, 1, 2),
}
FORMAL_SEEDS: tuple[int, ...] = (7, 17, 27, 37, 47)
SOURCE_BANDS_HZ: tuple[tuple[float, float], ...] = (
    (8.0, 12.0),
    (11.0, 15.0),
    (14.0, 20.0),
    (20.0, 30.0),
)
FILTER_ORDER = 4
DEFAULT_MIN_FREE_GIB = 50.0
FORMAL_EXPECTED_JOBS = 8_780
FORMAL_PUBLICATION_MODE = "formal_publishable"
TEST_PUBLICATION_MODE = "test_nonpublishable"
FORMAL_EXECUTOR = "benchmark.procedure_grid:execute_procedure_job"
FORMAL_ANALYSIS_CONFIG: Mapping[str, Any] = {"ece_bins": 15}
REQUIRED_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
THREAD_ENVIRONMENT_VARIABLES: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
HEX_64_RE = re.compile(r"^[0-9a-f]{64}$")
FINAL_FILENAMES = frozenset({"record.json", "predictions.npz", "completion.json"})
PREDICTION_ARCHIVE_MEMBERS = ("rows.npy", "probabilities.npy")
PREDICTION_ARCHIVE_ARRAYS = ("rows", "probabilities")
RUN_DIRECTORY_NAMES = frozenset({"records", "claims", "partials"})
PUBLICATION_FENCE_FILENAME = ".procedure-publication.lock"
PUBLICATION_FENCE_SCHEMA = "eeg-mi-procedure-publication-fence-v1"
MAX_PROJECT_GPU_WORKERS = project_gpu_leases.MAX_ACTIVE_LEASES
GPU_LEASE_TRACK_SCOPE = "binary-neural-procedures"
FORENSIC_ROOT_SUFFIX = ".procedure-forensics"
PLAN_CREATED_AT_SENTINEL = "pending-atomic-publication"
GPU_UUID_RE = re.compile(
    r"^GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
BOOT_ID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12}$"
)
_NONLINUX_BOOT_ID = str(uuid.uuid4())
_NONLINUX_PROCESS_START_TOKEN = str(max(1, time.monotonic_ns()))


@dataclass(frozen=True)
class ProcedureSpec:
    key: str
    stable_id: str
    config_path: str
    config_bytes: int
    config_sha256: str


PROCEDURES: tuple[ProcedureSpec, ...] = (
    ProcedureSpec(
        key="cameo",
        stable_id="architecture.cameo",
        config_path="configs/local_procedures/cameo_v1.json",
        config_bytes=829,
        config_sha256=(
            "e21d87705155f630be58f108a6cb80d5db2c7d59553459b0d638c24aeb321db6"
        ),
    ),
    ProcedureSpec(
        key="hemiparity",
        stable_id="architecture.hemiparity",
        config_path="configs/local_procedures/hemiparity_v1.json",
        config_bytes=569,
        config_sha256=(
            "8f4c5cef14e4a0edc782c350e06de33a661d7205016d995adecf09595ec9898f"
        ),
    ),
    ProcedureSpec(
        key="parity_fuse",
        stable_id="architecture.parity_fuse",
        config_path="configs/local_procedures/parity_fuse_v1.json",
        config_bytes=628,
        config_sha256=(
            "c2c45117d3aa9448ad715732f0c9f370ee96653944b0a76efefc72f317ae3bbc"
        ),
    ),
    ProcedureSpec(
        key="orbit_v3",
        stable_id="architecture.orbit_v3",
        config_path="configs/local_procedures/orbit_v3.json",
        config_bytes=848,
        config_sha256=(
            "aed0c4303a6dcd046dc49b233dee3a2fc926859d42382506e8b823d3e195e806"
        ),
    ),
)
PROCEDURE_BY_ID: Mapping[str, ProcedureSpec] = {
    value.stable_id: value for value in PROCEDURES
}
PROCEDURE_BY_KEY: Mapping[str, ProcedureSpec] = {
    value.key: value for value in PROCEDURES
}

# Read-only historical provenance.  These paths are never part of the runtime
# configuration closure and no function in this module parses their JSON.
BLOCKED_ORBIT_PROVENANCE: tuple[dict[str, Any], ...] = (
    {
        "stable_id": "architecture.orbit_v1",
        "status": "blocked_historical_post_hoc",
        "historical_artifact": ("predecessor-workspace/research-results/parity/orbit_v1_cho_dev8_formal.json"),
        "historical_artifact_sha256": (
            "b877b6ec6b758c1ee4f16d330d89e1febfe47561c99704b59a2075fa75c08b34"
        ),
        "historical_command": [
            (
                "/home/user/Desktop/eegthingy_codex_arch_20260718/"
                "src/benchmark/research/orbit_transport_benchmark.py"
            ),
            "--mode",
            "cho-dev",
            "--subjects",
            "1-8",
            "--folds",
            "5",
            "--seeds",
            "7",
            "--output",
            "predecessor-workspace/research-results/parity/orbit_v1_cho_dev8_formal.json",
        ],
        "complete_config_object_in_historical_artifact": False,
        "missing_current_config_fields": ["normalization"],
        "deterministic_config_only_extraction_without_inference": False,
        "stable_id_mapping_evidence": (
            "benchmark.model_registry._ORBIT_CONFIGURATIONS and "
            "docs/MODEL_REGISTRY.md map v1 to GroupNorm"
        ),
        "reason": (
            "the only settings object is outcome-bearing and omits the current "
            "explicit normalization field; reconstructing it would infer a "
            "historical default"
        ),
    },
    {
        "stable_id": "architecture.orbit_v2",
        "status": "blocked_historical_post_hoc",
        "historical_artifact": ("predecessor-workspace/research-results/parity/orbit_v2_batch_cho_dev8.json"),
        "historical_artifact_sha256": (
            "626b65a3463e7ead4cffb04f79bc79d9d5e75c25b86592c707f7e64291673b51"
        ),
        "historical_command": [
            (
                "/home/user/Desktop/eegthingy_codex_arch_20260718/"
                "src/benchmark/research/orbit_transport_benchmark.py"
            ),
            "--mode",
            "cho-dev",
            "--subjects",
            "1-8",
            "--folds",
            "5",
            "--seeds",
            "7",
            "--normalization",
            "batch",
            "--output",
            "predecessor-workspace/research-results/parity/orbit_v2_batch_cho_dev8.json",
        ],
        "complete_config_object_in_historical_artifact": True,
        "missing_current_config_fields": [],
        "deterministic_config_only_extraction_without_inference": True,
        "stable_id_mapping_evidence": (
            "benchmark.model_registry._ORBIT_CONFIGURATIONS maps v2 to the "
            "view-symmetric BatchNorm configuration"
        ),
        "reason": (
            "mechanical extraction is technically possible, but doing it after "
            "outcomes were observed would create a post-hoc runtime config"
        ),
    },
    {
        "stable_id": "architecture.orbit_v4",
        "status": "blocked_historical_post_hoc",
        "historical_artifact": (
            "predecessor-workspace/research-results/parity/orbit_v4_transport_only_cho_dev8.json"
        ),
        "historical_artifact_sha256": (
            "c337f8f3d91a2b6781b7cc59fb6bfde7687cb122fb9ed50d56c1e360b71cf6b5"
        ),
        "historical_command": [
            (
                "/home/user/Desktop/eegthingy_codex_arch_20260718/"
                "src/benchmark/research/orbit_transport_benchmark.py"
            ),
            "--mode",
            "cho-dev",
            "--subjects",
            "1-8",
            "--folds",
            "5",
            "--seeds",
            "7",
            "--normalization",
            "batch",
            "--transport-auxiliary-weight",
            "0.35",
            "--view-auxiliary-weight",
            "0",
            "--orientation-penalty",
            "0",
            "--output",
            "predecessor-workspace/research-results/parity/orbit_v4_transport_only_cho_dev8.json",
        ],
        "complete_config_object_in_historical_artifact": True,
        "missing_current_config_fields": [],
        "deterministic_config_only_extraction_without_inference": True,
        "stable_id_mapping_evidence": (
            "benchmark.model_registry._ORBIT_CONFIGURATIONS maps v4 to the "
            "v3 transport-only ablation"
        ),
        "reason": (
            "mechanical extraction is technically possible, but doing it after "
            "outcomes were observed would create a post-hoc runtime config"
        ),
    },
    {
        "stable_id": "architecture.orbit_v5",
        "status": "blocked_historical_post_hoc",
        "historical_artifact": (
            "predecessor-workspace/research-results/parity/orbit_v5_rawmean_only_cho_dev8.json"
        ),
        "historical_artifact_sha256": (
            "b22597098cf127070a295f1338c33c7231fe8d15c10471ddb22388fd532a79af"
        ),
        "historical_command": ["-c"],
        "complete_config_object_in_historical_artifact": True,
        "missing_current_config_fields": [],
        "deterministic_config_only_extraction_without_inference": True,
        "stable_id_mapping_evidence": (
            "benchmark.model_registry._ORBIT_CONFIGURATIONS maps v5 to the "
            "raw-mean-only candidate restriction"
        ),
        "reason": (
            "the result object contains complete-looking settings but only a "
            "generic '-c' command receipt; extraction would still be post-hoc"
        ),
    },
)

SOURCE_FILES: tuple[str, ...] = (
    "src/benchmark/__init__.py",
    "src/benchmark/procedure_grid.py",
    "src/benchmark/procedure_grid_analysis.py",
    "src/benchmark/project_gpu_leases.py",
    "src/benchmark/model_registry.py",
    # Resource guards are reused from this module; pin the transitive runtime
    # source rather than treating a safety helper as outside the closure.
    "src/benchmark/full_grid.py",
    "src/benchmark/config.py",
    "src/benchmark/data.py",
    "src/benchmark/research/__init__.py",
    "src/benchmark/local_outer_refit_benchmark.py",
    "src/benchmark/research/config.py",
    "src/benchmark/research/cameo_net.py",
    "src/benchmark/research/parity_net.py",
    "src/benchmark/research/parity_fuse_net.py",
    "src/benchmark/research/orbit_transport_net.py",
    "src/benchmark/research/data.py",
    "src/benchmark/shared/augment.py",
    "src/benchmark/shared/spd.py",
    "pyproject.toml",
    "uv.lock",
    *(value.config_path for value in PROCEDURES),
)

PROCEDURE_CONFIG_FIELDS: Mapping[str, Mapping[str, tuple[str, ...]]] = {
    "cameo": {
        "architecture": (
            "dropout",
            "dynamics_channels",
            "dynamics_kernel",
            "pool_kernel",
            "pool_stride",
            "temporal_filters",
            "temporal_kernel",
        ),
        "training": (
            "auxiliary_weight",
            "batch_size",
            "epochs",
            "gradient_clip",
            "learning_rate",
            "min_delta",
            "mirror_penalty",
            "mixture_names",
            "patience",
            "rho_grid",
            "swap_probability",
            "weight_decay",
        ),
    },
    "hemiparity": {
        "architecture": (
            "dropout",
            "dynamics_channels",
            "raw_rank",
            "tangent_rank",
            "temporal_filters",
        ),
        "training": (
            "auxiliary_weight",
            "batch_size",
            "deterministic",
            "epochs",
            "gate_balance_weight",
            "gradient_clip",
            "learning_rate",
            "min_delta",
            "patience",
            "tangent_learning_rate_scale",
            "weight_decay",
        ),
    },
    "parity_fuse": {
        "architecture": (
            "dynamics_channels",
            "dynamics_kernel",
            "fusion_dim",
            "gate_hidden",
            "raw_dim",
            "tangent_band_dim",
            "tangent_dim",
            "tangent_hidden",
            "temporal_filters",
            "temporal_kernel",
        ),
        "training": (
            "batch_size",
            "deterministic",
            "epochs",
            "gradient_clip",
            "learning_rate",
            "min_delta",
            "patience",
            "residual_penalty",
            "weight_decay",
        ),
    },
    "orbit_v3": {
        "architecture": (
            "dynamics_channels",
            "dynamics_kernel",
            "gate_hidden",
            "normalization",
            "orientation_hidden",
            "orientation_rank",
            "pool_kernel",
            "pool_stride",
            "temporal_filters",
            "temporal_kernel",
        ),
        "training": (
            "batch_size",
            "candidate_names",
            "deterministic",
            "epochs",
            "gradient_clip",
            "learning_rate",
            "min_delta",
            "orientation_penalty",
            "patience",
            "transport_auxiliary_weight",
            "view_auxiliary_weight",
            "weight_decay",
        ),
    },
}

RESET_EXCLUSIONS_BY_PROCEDURE: Mapping[str, tuple[str, ...]] = {
    "cameo": (),
    "hemiparity": ("tangent_base.weight",),
    "parity_fuse": ("tangent_anchor_weight",),
    "orbit_v3": (),
}

OUTCOME_KEY_TOKENS = frozenset(
    {
        "accuracy",
        "correctness",
        "groundtruth",
        "label",
        "labels",
        "metric",
        "metrics",
        "outcome",
        "outcomes",
        "prediction",
        "predictions",
        "probability",
        "probabilities",
        "score",
        "scores",
        "target",
        "targets",
        "truth",
        "ytrue",
    }
)
LEGITIMATE_OUTCOME_TOKEN_KEYS = frozenset(
    {
        "score_blind",
        "predictions.npz",
        "swap_probability",
        "test_performance_computed",
        "test_rows_sha256",
    }
)
REQUIRED_UV_RUNTIME_PACKAGES = frozenset(
    {"mne", "numpy", "pyriemann", "scikit-learn", "scipy", "torch"}
)
REQUIRED_UV_TEST_PACKAGES = frozenset({"pytest"})
REQUIRED_UV_DOCS_PACKAGES = frozenset({"pdfplumber", "pypdf", "reportlab"})
REQUIRED_UV_DEPENDENCY_GROUPS: Mapping[str, frozenset[str]] = {
    "docs": REQUIRED_UV_DOCS_PACKAGES,
    "test": REQUIRED_UV_TEST_PACKAGES,
}

FORBIDDEN_RECORD_KEYS = frozenset(
    {
        "accuracy",
        "balanced_accuracy",
        "cohen_kappa",
        "f1",
        "label",
        "labels",
        "metric",
        "metrics",
        "prediction",
        "predictions",
        "roc_auc",
        "score",
        "scores",
        "target",
        "targets",
        "test_label",
        "test_labels",
        "y",
    }
)


class ProcedureGridError(RuntimeError):
    """Raised when a procedure-grid contract cannot be preserved."""


class ClaimUnavailable(ProcedureGridError):
    """Raised when a live worker already owns a job."""


class PublicationActive(ClaimUnavailable):
    """Raised when an exclusive analysis publication fence is held."""


class PublicationFenceLost(ProcedureGridError):
    """Raised when the locked publication-gate inode or token is replaced."""


class GPUWorkerUnavailable(ProcedureGridError):
    """Raised when a physical GPU lease or the project-wide cap is unavailable."""


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
        digest = _sha256_bytes(_canonical_bytes(self.identity()))
        return f"procedure-{digest[:24]}"


@dataclass(frozen=True)
class Claim:
    job: Job
    path: Path
    nonce: str
    owner: Mapping[str, Any]
    resource_guard: Mapping[str, Any]
    st_dev: int
    st_ino: int
    value: Mapping[str, Any]
    fence_fd: int


GPULease = project_gpu_leases.GPULease


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(child) for child in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("non-finite floating value")
        return result
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite floating value")
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


def _require_no_symlink_ancestors(
    path: Path,
    *,
    include_leaf: bool = True,
    allow_missing_tail: bool = False,
) -> Path:
    """Reject every symlink or non-directory ancestor of an absolute path."""

    absolute = _absolute_path(path)
    parts = absolute.parts
    current = Path(absolute.anchor)
    stop = len(parts) if include_leaf else max(1, len(parts) - 1)
    missing = False
    for component in parts[1:stop]:
        current /= component
        if missing:
            continue
        try:
            observed = current.lstat()
        except FileNotFoundError:
            if not allow_missing_tail:
                raise
            missing = True
            continue
        is_leaf = current == absolute
        if stat.S_ISLNK(observed.st_mode):
            raise ProcedureGridError(f"path contains a symlink component: {current}")
        if not is_leaf and not stat.S_ISDIR(observed.st_mode):
            raise ProcedureGridError(
                f"path ancestor is not a real directory: {current}"
            )
    return absolute


def _secure_mkdir_absolute(path: Path, *, mode: int = 0o700) -> Path:
    """Create an absolute directory tree one no-follow descriptor at a time."""

    absolute = _absolute_path(path)
    if absolute == Path(absolute.anchor):
        return absolute
    descriptor = os.open(
        absolute.anchor,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0),
    )
    try:
        for component in absolute.parts[1:]:
            if component in {"", ".", ".."}:
                raise ProcedureGridError(f"invalid absolute path component: {path}")
            try:
                child = os.open(
                    component,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_NONBLOCK", 0),
                    dir_fd=descriptor,
                )
            except FileNotFoundError:
                try:
                    os.mkdir(component, mode, dir_fd=descriptor)
                except FileExistsError:
                    pass
                child = os.open(
                    component,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_NONBLOCK", 0),
                    dir_fd=descriptor,
                )
            except OSError as error:
                raise ProcedureGridError(
                    f"directory path is not a no-follow real tree: {absolute}"
                ) from error
            observed = os.fstat(child)
            if not stat.S_ISDIR(observed.st_mode):
                os.close(child)
                raise ProcedureGridError(
                    f"directory path component is not a directory: {absolute}"
                )
            os.close(descriptor)
            descriptor = child
    finally:
        os.close(descriptor)
    return absolute


def _open_directory_absolute(path: Path) -> int:
    """Open a directory by traversing every absolute component no-follow."""

    absolute = _absolute_path(path)
    descriptor = os.open(
        absolute.anchor,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        for component in absolute.parts[1:]:
            child = os.open(
                component,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=descriptor,
            )
            observed = os.fstat(child)
            if not stat.S_ISDIR(observed.st_mode):
                os.close(child)
                raise ProcedureGridError(
                    f"path component is not a directory: {absolute}"
                )
            os.close(descriptor)
            descriptor = child
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _anchored_lstat(path: Path) -> os.stat_result:
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


def _open_unique_regular_readonly(path: Path) -> int:
    absolute = _absolute_path(path)
    parent = _open_directory_absolute(absolute.parent)
    try:
        descriptor = os.open(
            absolute.name,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent,
        )
        path_stat = os.stat(
            absolute.name,
            dir_fd=parent,
            follow_symlinks=False,
        )
    except Exception:
        os.close(parent)
        raise
    os.close(parent)
    descriptor_stat = os.fstat(descriptor)
    if (
        not stat.S_ISREG(descriptor_stat.st_mode)
        or descriptor_stat.st_nlink != 1
        or (descriptor_stat.st_dev, descriptor_stat.st_ino)
        != (path_stat.st_dev, path_stat.st_ino)
    ):
        os.close(descriptor)
        raise ProcedureGridError(f"{absolute} is not a unique regular file")
    return descriptor


def _read_regular_file(path: Path) -> bytes:
    absolute = _require_no_symlink_ancestors(path, include_leaf=False)
    if absolute.name in {"", ".", ".."}:
        raise ProcedureGridError(f"invalid regular-file path: {absolute}")
    parent_descriptor = _open_directory_absolute(absolute.parent)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        path_stat = os.stat(
            absolute.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if not stat.S_ISREG(path_stat.st_mode) or path_stat.st_nlink != 1:
            raise ProcedureGridError(f"{absolute} is not a unique regular file")
        descriptor = os.open(
            absolute.name,
            flags,
            dir_fd=parent_descriptor,
        )
    except Exception:
        os.close(parent_descriptor)
        raise
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or (before.st_dev, before.st_ino) != (path_stat.st_dev, path_stat.st_ino)
        ):
            raise ProcedureGridError(f"{absolute} changed while it was opened")
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
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(getattr(before, name) != getattr(after, name) for name in stable_fields):
            raise ProcedureGridError(f"{absolute} changed during its verified read")
        final_path = os.stat(
            absolute.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (final_path.st_dev, final_path.st_ino) != (
            before.st_dev,
            before.st_ino,
        ) or final_path.st_nlink != 1:
            raise ProcedureGridError(f"{absolute} changed after its verified read")
        payload = b"".join(chunks)
        if len(payload) != before.st_size:
            raise ProcedureGridError(f"{absolute} produced an incomplete read")
        return payload
    finally:
        os.close(descriptor)
        os.close(parent_descriptor)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(_read_regular_file(path))
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.view(np.uint8).tobytes())
    return digest.hexdigest()


def _array_manifest(value: np.ndarray) -> dict[str, Any]:
    array = np.asarray(value)
    return {
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "sha256": _array_sha256(array),
    }


# Private compatibility domain separator; retained as numeric bytes so
# existing split identities remain valid without serializing a retired token.
_ROWS_HASH_COMPATIBILITY_PREFIX = bytes(
    (
        105, 101, 101, 101, 45, 109, 105, 45, 112, 114, 111, 99, 101, 100,
        117, 114, 101, 45, 114, 111, 119, 45, 105, 110, 100, 105, 99, 101,
        115, 45, 118, 49, 0,
    )
)


def _rows_sha256(value: np.ndarray | Sequence[int]) -> str:
    rows = np.ascontiguousarray(np.asarray(value, dtype="<i8"))
    if rows.ndim != 1:
        raise ValueError("row indices must be one-dimensional")
    digest = hashlib.sha256()
    digest.update(_ROWS_HASH_COMPATIBILITY_PREFIX)
    digest.update(np.asarray(rows.shape, dtype="<i8").tobytes())
    digest.update(rows.tobytes())
    return digest.hexdigest()


def _strict_json_bytes(payload: bytes, *, source: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r}")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ProcedureGridError(f"{source} is not UTF-8 JSON") from error
    value = json.loads(
        text,
        object_pairs_hook=unique,
        parse_constant=reject_constant,
    )
    if not isinstance(value, dict):
        raise ProcedureGridError(f"{source} is not a JSON object")
    return value


def _strict_json_load(path: Path) -> dict[str, Any]:
    return _strict_json_bytes(_read_regular_file(path), source=str(path))


def _read_stable_descriptor_bytes(descriptor: int, *, source: str) -> bytes:
    """Read and verify one immutable regular-file descriptor snapshot."""

    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise ProcedureGridError(f"{source} is not a unique regular file")
    os.lseek(descriptor, 0, os.SEEK_SET)
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
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(getattr(before, name) != getattr(after, name) for name in stable_fields):
        raise ProcedureGridError(f"{source} changed during its descriptor read")
    payload = b"".join(chunks)
    if len(payload) != before.st_size:
        raise ProcedureGridError(f"{source} produced an incomplete read")
    return payload


def _read_unique_regular_at(
    directory_descriptor: int,
    name: str,
    *,
    source: str,
    read_only: bool,
) -> bytes:
    """Read one file through a bound directory and retain its inode identity."""

    if not name or name in {".", ".."} or "/" in name:
        raise ProcedureGridError("invalid descriptor-anchored input path")
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
            or not stat.S_ISREG(descriptor_stat.st_mode)
            or descriptor_stat.st_nlink != 1
            or (path_stat.st_dev, path_stat.st_ino)
            != (descriptor_stat.st_dev, descriptor_stat.st_ino)
            or (read_only and descriptor_stat.st_mode & 0o222)
        ):
            raise ProcedureGridError(f"{source} is not a unique immutable file")
        payload = _read_stable_descriptor_bytes(descriptor, source=source)
        final_path = os.stat(
            name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if final_path.st_nlink != 1 or (final_path.st_dev, final_path.st_ino) != (
            descriptor_stat.st_dev,
            descriptor_stat.st_ino,
        ):
            raise ProcedureGridError(f"{source} changed after its verified read")
        return payload
    finally:
        os.close(descriptor)


def _strict_json_from_descriptor(descriptor: int, *, source: str) -> dict[str, Any]:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise ProcedureGridError(f"{source} is not a unique regular file")
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    try:
        payload = b"".join(chunks)
        text = payload.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=lambda pairs: _unique_json_pairs(pairs, source=source),
            parse_constant=lambda raw: _reject_json_constant(raw, source=source),
        )
    except UnicodeDecodeError as error:
        raise ProcedureGridError(f"{source} is not UTF-8 JSON") from error
    after = os.fstat(descriptor)
    stable_fields = (
        "st_dev",
        "st_ino",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(getattr(before, name) != getattr(after, name) for name in stable_fields):
        raise ProcedureGridError(f"{source} changed during its descriptor read")
    if not isinstance(value, dict):
        raise ProcedureGridError(f"{source} is not a JSON object")
    if payload != _canonical_bytes(value) + b"\n":
        raise ProcedureGridError(f"{source} is not canonical JSON")
    return value


def _unique_json_pairs(pairs: list[tuple[str, Any]], *, source: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProcedureGridError(f"{source} has duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(raw: str, *, source: str) -> None:
    raise ProcedureGridError(f"{source} has non-finite JSON constant {raw!r}")


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _safe_run_root(run_root: Path, *, create: bool = False) -> Path:
    root = _absolute_path(run_root)
    if create and not root.exists():
        _secure_mkdir_absolute(root)
    _require_no_symlink_ancestors(
        root,
        include_leaf=True,
        allow_missing_tail=not root.exists(),
    )
    try:
        _require_real_directory(root)
    except FileNotFoundError:
        if create:
            raise
        return root
    return root


def _ensure_real_directory(root: Path, relative: str | Path) -> Path:
    base = _safe_run_root(root, create=True)
    relative_path = Path(relative)
    if relative_path.is_absolute() or any(
        part in {"", ".", ".."} for part in relative_path.parts
    ):
        raise ProcedureGridError(f"invalid run-relative directory {relative}")
    current = _secure_mkdir_absolute(base / relative_path)
    try:
        current.relative_to(base)
    except ValueError as error:
        raise ProcedureGridError(
            f"run directory escapes its anchor: {relative}"
        ) from error
    return current


def _require_real_directory(path: Path) -> os.stat_result:
    absolute = _require_no_symlink_ancestors(path)
    descriptor = _open_directory_absolute(absolute)
    try:
        observed = os.fstat(descriptor)
        if not stat.S_ISDIR(observed.st_mode):
            raise ProcedureGridError(f"{absolute} is not a real directory")
        return observed
    finally:
        os.close(descriptor)


def _require_real_path_components(root: Path, path: Path) -> None:
    base = _safe_run_root(root)
    try:
        relative = _absolute_path(path).relative_to(base)
    except ValueError as error:
        raise ProcedureGridError(f"path escapes run root: {path}") from error
    current = base
    for part in relative.parts:
        current = current / part
        _require_real_directory(current)


def _require_unique_regular(path: Path, *, read_only: bool = False) -> os.stat_result:
    absolute = _require_no_symlink_ancestors(path, include_leaf=False)
    parent_descriptor = _open_directory_absolute(absolute.parent)
    try:
        descriptor = os.open(
            absolute.name,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_descriptor,
        )
        try:
            observed = os.fstat(descriptor)
            path_stat = os.stat(
                absolute.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISREG(observed.st_mode)
                or observed.st_nlink != 1
                or (observed.st_dev, observed.st_ino)
                != (path_stat.st_dev, path_stat.st_ino)
                or (read_only and observed.st_mode & 0o222)
            ):
                qualifier = "read-only " if read_only else ""
                raise ProcedureGridError(
                    f"{absolute} is not a unique {qualifier}regular file"
                )
            return observed
        finally:
            os.close(descriptor)
    finally:
        os.close(parent_descriptor)


def _exact_keys(value: Any, expected: set[str] | frozenset[str], *, path: str) -> None:
    if not isinstance(value, Mapping) or set(value) != set(expected):
        observed = (
            sorted(str(key) for key in value) if isinstance(value, Mapping) else []
        )
        raise ProcedureGridError(
            f"{path} schema is invalid; expected {sorted(expected)}, got {observed}"
        )


def _recursive_outcome_aliases(value: Any, path: str = "record") -> list[str]:
    violations: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).strip().lower()
            compact = re.sub(r"[^a-z0-9]+", "", key)
            if key not in LEGITIMATE_OUTCOME_TOKEN_KEYS and any(
                token in compact for token in OUTCOME_KEY_TOKENS
            ):
                violations.append(f"{path}.{raw_key}")
            violations.extend(_recursive_outcome_aliases(child, f"{path}.{raw_key}"))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            violations.extend(_recursive_outcome_aliases(child, f"{path}[{index}]"))
    return violations


def _recursive_cache_outcome_aliases(value: Any, path: str = "cache") -> list[str]:
    """Reject hidden outcomes while allowing the frozen preprocessing label note."""

    violations: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).strip().lower()
            compact = re.sub(r"[^a-z0-9]+", "", key)
            preprocessing_label_contract = key == "labels" and path.endswith(
                ".preprocessing"
            )
            if not preprocessing_label_contract and any(
                token in compact for token in OUTCOME_KEY_TOKENS
            ):
                violations.append(f"{path}.{raw_key}")
            violations.extend(
                _recursive_cache_outcome_aliases(child, f"{path}.{raw_key}")
            )
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            violations.extend(
                _recursive_cache_outcome_aliases(child, f"{path}[{index}]")
            )
    return violations


def _fsync_directory(path: Path) -> None:
    _require_real_directory(path)
    descriptor = _open_directory_absolute(path)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_bytes_exclusive(path: Path, payload: bytes) -> None:
    _secure_mkdir_absolute(path.parent)
    _require_real_directory(path.parent)
    parent_descriptor = _open_directory_absolute(path.parent)
    try:
        descriptor = os.open(
            path.name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=parent_descriptor,
        )
        try:
            offset = 0
            while offset < len(payload):
                offset += os.write(descriptor, payload[offset:])
            os.fchmod(descriptor, 0o444)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    _write_bytes_exclusive(path, _canonical_bytes(value) + b"\n")


def _write_bytes_exclusive_at(
    directory_descriptor: int,
    name: str,
    payload: bytes,
) -> str:
    """Write one immutable file beneath an already-bound directory."""

    if (
        not isinstance(name, str)
        or not name
        or name in {".", ".."}
        or "/" in name
        or not stat.S_ISDIR(os.fstat(directory_descriptor).st_mode)
    ):
        raise ProcedureGridError("invalid descriptor-anchored output path")
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
                raise OSError("descriptor-anchored write made no progress")
            offset += written
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.fsync(directory_descriptor)
    return _sha256_bytes(payload)


def _assert_directory_descriptor_path(descriptor: int, path: Path) -> os.stat_result:
    """Require a pathname to still name the exact open directory inode."""

    bound = os.fstat(descriptor)
    observed = _anchored_lstat(path)
    if (
        not stat.S_ISDIR(bound.st_mode)
        or not stat.S_ISDIR(observed.st_mode)
        or (bound.st_dev, bound.st_ino) != (observed.st_dev, observed.st_ino)
    ):
        raise ProcedureGridError(f"directory inode ownership was lost: {path}")
    return bound


def _seal_directory_read_only(descriptor: int, path: Path) -> os.stat_result:
    """Durably seal the exact held directory inode as mode 0555."""

    before = _assert_directory_descriptor_path(descriptor, path)
    os.fsync(descriptor)
    os.fchmod(descriptor, 0o555)
    os.fsync(descriptor)
    sealed = _assert_directory_descriptor_path(descriptor, path)
    path_status = _anchored_lstat(path)
    if (
        (sealed.st_dev, sealed.st_ino) != (before.st_dev, before.st_ino)
        or stat.S_IMODE(sealed.st_mode) != 0o555
        or (path_status.st_dev, path_status.st_ino, path_status.st_mode)
        != (sealed.st_dev, sealed.st_ino, sealed.st_mode)
    ):
        raise ProcedureGridError(f"directory seal did not persist: {path}")
    return sealed


def _stage_run_bytes(
    run_root: Path,
    *,
    basename: str,
    payload: bytes,
) -> Path:
    if (
        not basename
        or "/" in basename
        or basename in {".", ".."}
        or not re.fullmatch(r"[A-Za-z0-9_.-]+", basename)
    ):
        raise ProcedureGridError("invalid staged run receipt name")
    root = _safe_run_root(run_root)
    partial_root = _secure_mkdir_absolute(
        root.parent / f".{root.name}{FORENSIC_ROOT_SUFFIX}" / "partials"
    )
    path = partial_root / f"{basename}.{uuid.uuid4().hex}.partial"
    _write_bytes_exclusive(path, payload)
    if _read_regular_file(path) != payload:
        raise ProcedureGridError("staged run bytes changed after creation")
    return path


def _stage_run_json(
    run_root: Path,
    *,
    basename: str,
    value: Mapping[str, Any],
) -> Path:
    """Create a complete non-authoritative receipt outside the run tree."""

    path = _stage_run_bytes(
        run_root,
        basename=basename,
        payload=_canonical_bytes(value) + b"\n",
    )
    if _strict_json_load(path) != dict(value):
        raise ProcedureGridError("staged run receipt changed after creation")
    return path


def _atomic_rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish one path and fail if the destination already exists."""

    source = _absolute_path(source)
    destination = _absolute_path(destination)
    _require_no_symlink_ancestors(source, include_leaf=False)
    _require_no_symlink_ancestors(
        destination,
        include_leaf=False,
        allow_missing_tail=False,
    )
    libc = ctypes.CDLL(None, use_errno=True)
    source_parent = _open_directory_absolute(source.parent)
    try:
        destination_parent = _open_directory_absolute(destination.parent)
    except Exception:
        os.close(source_parent)
        raise
    try:
        source_bytes = os.fsencode(source.name)
        destination_bytes = os.fsencode(destination.name)
        if sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
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
        elif sys.platform == "darwin" and hasattr(libc, "renameatx_np"):
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
        else:
            raise ProcedureGridError(
                "anchored atomic no-replace publication is unavailable"
            )
    finally:
        os.close(source_parent)
        os.close(destination_parent)
    if result != 0:
        observed_errno = ctypes.get_errno()
        if observed_errno in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(
                observed_errno, os.strerror(observed_errno), destination
            )
        raise OSError(observed_errno, os.strerror(observed_errno), destination)


def _recursive_forbidden_keys(value: Any, path: str = "record") -> list[str]:
    violations: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).strip().lower()
            if key in FORBIDDEN_RECORD_KEYS:
                violations.append(f"{path}.{raw_key}")
            violations.extend(_recursive_forbidden_keys(child, f"{path}.{raw_key}"))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            violations.extend(_recursive_forbidden_keys(child, f"{path}[{index}]"))
    return violations


def _subject_key(dataset: str, subject: int) -> str:
    return f"{dataset}:s{int(subject):03d}"


def _split_key(dataset: str, subject: int, fold: int) -> str:
    return f"{dataset}:s{int(subject):03d}:f{int(fold):02d}"


def _one_split_identity(
    *,
    dataset: str,
    subject: int,
    fold: int,
    cache_array_sha256: str,
    trial_count: int,
    train_rows: Sequence[int],
    validation_rows: Sequence[int],
    test_rows: Sequence[int],
) -> dict[str, Any]:
    train = np.asarray(train_rows, dtype=np.int64)
    validation = np.asarray(validation_rows, dtype=np.int64)
    test = np.asarray(test_rows, dtype=np.int64)
    source = np.sort(np.concatenate((train, validation)))
    return {
        "dataset": dataset,
        "subject": int(subject),
        "fold": int(fold),
        "cache_array_sha256": cache_array_sha256,
        "trial_count": int(trial_count),
        "partitions": {
            name: {
                "count": len(rows),
                "rows_sha256": _rows_sha256(rows),
            }
            for name, rows in (
                ("train", train),
                ("validation", validation),
                ("source", source),
                ("test", test),
            )
        },
    }


def _dataset_contracts() -> dict[str, Any]:
    contracts: dict[str, Any] = {}
    for dataset in BINARY_DATASETS:
        spec = dataset_spec(dataset)
        if spec.n_classes != 2:
            raise ProcedureGridError(f"{dataset} is no longer binary")
        contracts[dataset] = {
            "subjects": [int(value) for value in spec.development_subjects],
            "folds": [int(value) for value in DATASET_FOLDS[dataset]],
            "n_classes": 2,
            "events": list(spec.events),
            "protocol": spec.protocol,
            "preprocessing": preprocessing_for_dataset(dataset),
        }
    return contracts


def _cache_and_split_identity(
    cache_root: Path,
    contracts: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    from .data import load_subject_cache, split_indices

    cache_identity: dict[str, Any] = {}
    split_identity: dict[str, Any] = {}
    for dataset, contract in contracts.items():
        for raw_subject in contract["subjects"]:
            subject = int(raw_subject)
            data = load_subject_cache(str(dataset), subject, cache_root=cache_root)
            identity = _cache_identity_from_loaded(data)
            cache_identity[_subject_key(str(dataset), subject)] = identity
            for raw_fold in contract["folds"]:
                fold = int(raw_fold)
                train, validation, test = split_indices(
                    str(dataset),
                    data["y"],
                    data["sessions"],
                    data["runs"],
                    fold=fold,
                    subject=subject,
                )
                split_identity[_split_key(str(dataset), subject, fold)] = (
                    _one_split_identity(
                        dataset=str(dataset),
                        subject=subject,
                        fold=fold,
                        cache_array_sha256=str(data["identity"]["array_sha256"]),
                        trial_count=len(data["y"]),
                        train_rows=train,
                        validation_rows=validation,
                        test_rows=test,
                    )
                )
    return cache_identity, split_identity


def _cache_identity_from_loaded(data: Mapping[str, Any]) -> dict[str, Any]:
    """Bind exact cache metadata and array geometry from one verified archive."""

    required = {
        "x",
        "y",
        "positions",
        "channel_names",
        "sessions",
        "runs",
        "identity",
    }
    if set(data) != required:
        raise ProcedureGridError("verified cache loader returned an unknown schema")
    values = np.asarray(data["x"])
    positions = np.asarray(data["positions"])
    channels = [str(value) for value in np.asarray(data["channel_names"]).tolist()]
    identity = copy.deepcopy(dict(data["identity"]))
    identity.update(
        {
            "trial_count": int(len(data["y"])),
            "n_channels": int(values.shape[1]),
            "positions_manifest": _array_manifest(positions),
        }
    )
    if (
        values.ndim != 3
        or identity.get("shape") != list(values.shape)
        or identity.get("channels") != channels
        or identity["trial_count"] != values.shape[0]
        or identity["n_channels"] != len(channels)
        or positions.shape != (len(channels), 3)
    ):
        raise ProcedureGridError("verified cache geometry is internally inconsistent")
    aliases = _recursive_cache_outcome_aliases(identity)
    if aliases:
        raise ProcedureGridError(
            f"cache identity contains outcome aliases: {aliases[:5]}"
        )
    return identity


def _source_identity() -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in SOURCE_FILES:
        path = PROJECT_ROOT / relative
        try:
            payload = _read_regular_file(path)
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"procedure source closure is missing {path}"
            ) from error
        result[relative] = _sha256_bytes(payload)
    return result


def _uv_version() -> str | None:
    try:
        completed = subprocess.run(
            ["uv", "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def _nvidia_identity() -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,pci.bus_id,name,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return {"available": False, "gpus": []}
    rows: list[dict[str, str]] = []
    seen_indices: set[int] = set()
    seen_uuids: set[str] = set()
    for fields in _strict_nvidia_csv_rows(completed.stdout, width=5):
        index_text, raw_uuid, pci_bus_id, name, driver_version = fields
        try:
            gpu_uuid = project_gpu_leases.canonical_gpu_uuid(raw_uuid)
        except project_gpu_leases.ProjectGPULeaseError as error:
            raise ProcedureGridError("invalid nvidia-smi GPU UUID") from error
        index = int(index_text) if index_text.isdigit() else -1
        if (
            index < 0
            or index in seen_indices
            or gpu_uuid in seen_uuids
            or not re.fullmatch(
                r"[0-9A-Fa-f]{4,8}:[0-9A-Fa-f]{2}:"
                r"[0-9A-Fa-f]{2}\.[0-7]",
                pci_bus_id,
            )
            or any(ord(character) < 32 for character in name + driver_version)
        ):
            raise ProcedureGridError("invalid nvidia-smi GPU identity")
        seen_indices.add(index)
        seen_uuids.add(gpu_uuid)
        rows.append(
            {
                "index": str(index),
                "uuid": gpu_uuid,
                "pci_bus_id": pci_bus_id,
                "name": name,
                "driver_version": driver_version,
            }
        )
    return {"available": bool(rows), "gpus": rows}


def _environment_identity() -> dict[str, Any]:
    packages = sorted(
        (
            str(distribution.metadata.get("Name", "unknown")).lower(),
            str(distribution.version),
        )
        for distribution in importlib.metadata.distributions()
    )
    try:
        import torch

        torch_identity: dict[str, Any] = {
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
        }
    except ImportError:
        torch_identity = {
            "version": None,
            "cuda_runtime": None,
            "cuda_available": False,
        }
    return {
        "schema": "eeg-mi-procedure-environment-v1",
        "python": sys.version,
        "executable": str(Path(sys.executable).resolve()),
        "prefix": str(Path(sys.prefix).resolve()),
        "base_prefix": str(Path(sys.base_prefix).resolve()),
        "virtual_environment": os.environ.get("VIRTUAL_ENV"),
        "platform": platform.platform(),
        "uv_version": _uv_version(),
        "packages": [list(value) for value in packages],
        "packages_sha256": _sha256_bytes(_canonical_bytes(packages)),
        "torch": torch_identity,
        "nvidia": _nvidia_identity(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "thread_environment": {
            name: os.environ.get(name) for name in THREAD_ENVIRONMENT_VARIABLES
        },
    }


def _require_uv_virtual_environment() -> None:
    virtual_environment = os.environ.get("VIRTUAL_ENV")
    if (
        sys.prefix == sys.base_prefix
        or not virtual_environment
        or virtual_environment != sys.prefix
    ):
        raise ProcedureGridError(
            "production commands require VIRTUAL_ENV to equal the active UV prefix"
        )
    if _uv_version() is None:
        raise ProcedureGridError("UV is required but `uv --version` failed")


def _release_uv_manifest_texts() -> tuple[str, str]:
    try:
        pyproject = _read_regular_file(PROJECT_ROOT / "pyproject.toml").decode("utf-8")
        uv_lock = _read_regular_file(PROJECT_ROOT / "uv.lock").decode("utf-8")
    except UnicodeDecodeError as error:
        raise ProcedureGridError("release UV manifests are not UTF-8") from error
    return pyproject, uv_lock


def _require_formal_release_runtime() -> None:
    _require_uv_virtual_environment()
    identity = validate_uv_dependency_closure(*_release_uv_manifest_texts())
    expected = identity["runtime_locked_package_versions"]
    installed: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = _canonical_requirement_name(str(distribution.metadata.get("Name", "")))
        version = str(distribution.version)
        if not name or not version or name in installed:
            raise ProcedureGridError(
                "active UV environment has a malformed/duplicate distribution"
            )
        installed[name] = version
    if installed != expected:
        raise ProcedureGridError(
            "active UV environment differs from the exact runtime-only lock "
            "closure; test/docs/default-group extras must be absent"
        )


def validate_uv_dependency_closure(
    pyproject_text: str, uv_lock_text: str
) -> dict[str, Any]:
    """Validate an exact runtime/test/docs UV closure without installing."""

    import tomllib

    try:
        from packaging.markers import InvalidMarker, Marker
    except ImportError as error:
        raise ProcedureGridError(
            "the UV closure validator requires the locked packaging runtime"
        ) from error

    try:
        project_document = tomllib.loads(pyproject_text)
        lock_document = tomllib.loads(uv_lock_text)
    except (tomllib.TOMLDecodeError, TypeError) as error:
        raise ProcedureGridError("UV dependency manifests are invalid TOML") from error

    project = project_document.get("project")
    if not isinstance(project, Mapping):
        raise ProcedureGridError("pyproject.toml has no [project] table")
    project_name = _canonical_requirement_name(str(project.get("name", "")))
    project_version = str(project.get("version", ""))
    requires_python = project.get("requires-python")
    if (
        not project_name
        or not project_version
        or not isinstance(requires_python, str)
        or not requires_python
    ):
        raise ProcedureGridError("pyproject project identity is incomplete")

    def exact_pins(values: Any, *, scope: str) -> dict[str, str]:
        if not isinstance(values, list):
            raise ProcedureGridError(f"{scope} dependencies must be a list")
        result: dict[str, str] = {}
        for raw in values:
            parsed = _exact_requirement_pin(raw)
            if parsed is None:
                raise ProcedureGridError(
                    f"every {scope} direct dependency must be exactly pinned: {raw!r}"
                )
            name, version = parsed
            if name in result:
                raise ProcedureGridError(f"duplicate {scope} direct dependency: {name}")
            result[name] = version
        return result

    runtime_pins = exact_pins(project.get("dependencies"), scope="runtime")
    if not REQUIRED_UV_RUNTIME_PACKAGES.issubset(runtime_pins):
        raise ProcedureGridError(
            "runtime pins do not contain the complete EEG/Torch procedure stack"
        )
    optional = project.get("optional-dependencies")
    if optional not in (None, {}):
        raise ProcedureGridError(
            "release dependencies must use explicit non-default groups, not extras"
        )
    dependency_groups = project_document.get("dependency-groups")
    if not isinstance(dependency_groups, Mapping):
        raise ProcedureGridError(
            "pyproject.toml must define exact test and docs groups"
        )
    group_pins: dict[str, dict[str, str]] = {}
    for raw_group, raw_dependencies in dependency_groups.items():
        group = str(raw_group)
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", group):
            raise ProcedureGridError("dependency-group name is invalid")
        group_pins[group] = exact_pins(raw_dependencies, scope=f"group {group}")
    if not set(REQUIRED_UV_DEPENDENCY_GROUPS).issubset(group_pins):
        raise ProcedureGridError("pyproject omits the required test/docs groups")
    for group, required in REQUIRED_UV_DEPENDENCY_GROUPS.items():
        if not required.issubset(group_pins[group]):
            raise ProcedureGridError(
                f"pyproject {group} group omits required frozen tools"
            )
    grouped_names = {name for pins in group_pins.values() for name in pins}
    if set(runtime_pins).intersection(grouped_names):
        raise ProcedureGridError(
            "test/docs dependency-group tools must not be runtime dependencies"
        )
    uv_settings = project_document.get("tool", {}).get("uv")
    if not isinstance(uv_settings, Mapping) or uv_settings.get("default-groups") != []:
        raise ProcedureGridError(
            "[tool.uv] default-groups must be [] so test/docs never install "
            "in the production runtime"
        )

    packages = lock_document.get("package")
    if (
        type(lock_document.get("version")) is not int
        or type(lock_document.get("revision")) is not int
        or lock_document.get("requires-python") != requires_python
        or not isinstance(packages, list)
    ):
        raise ProcedureGridError("uv.lock header is incomplete or inconsistent")
    by_name: dict[str, Mapping[str, Any]] = {}
    locked_versions: dict[str, str] = {}
    project_package: Mapping[str, Any] | None = None
    for package in packages:
        if not isinstance(package, Mapping):
            raise ProcedureGridError("uv.lock contains a malformed package")
        name = _canonical_requirement_name(str(package.get("name", "")))
        version = str(package.get("version", ""))
        if not name or not version or name in by_name:
            raise ProcedureGridError(
                "uv.lock package names/versions must be unique and nonempty"
            )
        by_name[name] = package
        locked_versions[name] = version
        source = package.get("source")
        if (
            name == project_name
            and version == project_version
            and isinstance(source, Mapping)
            and source.get("virtual") == "."
        ):
            project_package = package
    if project_package is None:
        raise ProcedureGridError(
            "uv.lock does not contain the exact local project package"
        )

    def dependency_names(values: Any, *, scope: str) -> list[str]:
        if not isinstance(values, list):
            raise ProcedureGridError(f"uv.lock {scope} ledger is malformed")
        result: list[str] = []
        for value in values:
            if not isinstance(value, Mapping):
                raise ProcedureGridError(
                    f"uv.lock {scope} contains a malformed dependency"
                )
            name = _canonical_requirement_name(str(value.get("name", "")))
            if not name:
                raise ProcedureGridError(
                    f"uv.lock {scope} contains an empty dependency"
                )
            result.append(name)
        return result

    root_dependencies = dependency_names(
        project_package.get("dependencies", []),
        scope="project dependencies",
    )
    if len(root_dependencies) != len(set(root_dependencies)) or set(
        root_dependencies
    ) != set(runtime_pins):
        raise ProcedureGridError(
            "uv.lock project runtime dependencies differ from pyproject"
        )
    root_groups = project_package.get("dev-dependencies")
    if not isinstance(root_groups, Mapping) or set(root_groups) != set(group_pins):
        raise ProcedureGridError(
            "uv.lock project dependency groups differ from pyproject"
        )
    for group, pins in group_pins.items():
        names = dependency_names(root_groups[group], scope=f"project group {group}")
        if len(names) != len(set(names)) or set(names) != set(pins):
            raise ProcedureGridError(
                f"uv.lock project group {group} differs from pyproject"
            )

    metadata = project_package.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ProcedureGridError("uv.lock project metadata is absent")

    def metadata_pins(values: Any, *, scope: str) -> dict[str, str]:
        if not isinstance(values, list):
            raise ProcedureGridError(f"uv.lock metadata {scope} is malformed")
        result: dict[str, str] = {}
        for value in values:
            if not isinstance(value, Mapping):
                raise ProcedureGridError(
                    f"uv.lock metadata {scope} contains a malformed entry"
                )
            name = _canonical_requirement_name(str(value.get("name", "")))
            specifier = str(value.get("specifier", ""))
            if not name or name in result:
                raise ProcedureGridError(f"uv.lock metadata {scope} names are invalid")
            result[name] = specifier
        return result

    locked_runtime_pins = metadata_pins(
        metadata.get("requires-dist"), scope="requires-dist"
    )
    if set(locked_runtime_pins) != set(runtime_pins):
        raise ProcedureGridError(
            "uv.lock requires-dist differs from pyproject runtime pins"
        )
    requires_dev = metadata.get("requires-dev")
    if not isinstance(requires_dev, Mapping) or set(requires_dev) != set(group_pins):
        raise ProcedureGridError("uv.lock requires-dev groups differ from pyproject")
    locked_group_pins = {
        group: metadata_pins(requires_dev[group], scope=f"requires-dev.{group}")
        for group in group_pins
    }
    for name, version in runtime_pins.items():
        if (
            locked_runtime_pins.get(name) != f"=={version}"
            or locked_versions.get(name) != version
        ):
            raise ProcedureGridError(
                f"uv.lock runtime pin differs from pyproject for {name}"
            )
    for group, pins in group_pins.items():
        if set(locked_group_pins[group]) != set(pins):
            raise ProcedureGridError(
                f"uv.lock requires-dev.{group} differs from pyproject"
            )
        for name, version in pins.items():
            if (
                locked_group_pins[group].get(name) != f"=={version}"
                or locked_versions.get(name) != version
            ):
                raise ProcedureGridError(
                    f"uv.lock {group} pin differs from pyproject for {name}"
                )

    all_graph: dict[str, set[str]] = {}
    runtime_graph: dict[str, set[str]] = {}
    for name, package in by_name.items():
        children = dependency_names(
            package.get("dependencies", []),
            scope=f"package {name}",
        )
        all_graph[name] = set(children)
        selected: set[str] = set()
        for dependency, raw in zip(
            children,
            package.get("dependencies", []),
            strict=True,
        ):
            marker = raw.get("marker")
            if marker is None:
                selected.add(dependency)
                continue
            if not isinstance(marker, str) or not marker:
                raise ProcedureGridError(
                    f"uv.lock dependency marker is invalid for {name}"
                )
            try:
                if Marker(marker).evaluate():
                    selected.add(dependency)
            except InvalidMarker as error:
                raise ProcedureGridError(
                    f"uv.lock dependency marker is invalid for {name}"
                ) from error
        runtime_graph[name] = selected
        if not all_graph[name].issubset(by_name):
            raise ProcedureGridError(
                f"uv.lock contains unresolved dependency edges for {name}"
            )
    all_graph[project_name].update(
        name for pins in group_pins.values() for name in pins
    )

    def closure(graph: Mapping[str, set[str]]) -> set[str]:
        reached = {project_name}
        frontier = [project_name]
        while frontier:
            parent = frontier.pop()
            for child in graph.get(parent, set()):
                if child not in reached:
                    reached.add(child)
                    frontier.append(child)
        return reached

    full_closure = closure(all_graph)
    if full_closure != set(by_name):
        raise ProcedureGridError(
            "uv.lock contains packages outside the runtime/test/docs closure"
        )
    runtime_closure = closure(runtime_graph) - {project_name}
    runtime_versions = {name: locked_versions[name] for name in sorted(runtime_closure)}
    if set(runtime_versions).intersection(grouped_names):
        raise ProcedureGridError(
            "runtime closure includes a direct test/docs group dependency"
        )
    return {
        "project_name": project_name,
        "project_version": project_version,
        "requires_python": requires_python,
        "runtime_direct_dependency_pins": {
            name: runtime_pins[name] for name in sorted(runtime_pins)
        },
        "dependency_group_pins": {
            group: {name: pins[name] for name in sorted(pins)}
            for group, pins in sorted(group_pins.items())
        },
        "runtime_locked_package_versions": runtime_versions,
        "runtime_locked_package_closure_sha256": _sha256_bytes(
            _canonical_bytes(runtime_versions)
        ),
    }


def _canonical_requirement_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value.strip().lower())


def _exact_requirement_pin(value: Any) -> tuple[str, str] | None:
    if not isinstance(value, str):
        return None
    match = re.fullmatch(
        r"\s*([A-Za-z0-9_.-]+)(?:\[[A-Za-z0-9_,.-]+\])?"
        r"==([A-Za-z0-9][A-Za-z0-9_.+!-]*)\s*",
        value,
    )
    if match is None:
        return None
    return _canonical_requirement_name(match.group(1)), match.group(2)


def _validate_frozen_config_payload(
    spec: ProcedureSpec, payload: Mapping[str, Any]
) -> dict[str, Any]:
    _exact_keys(
        payload,
        {"schema", "version", "model", "architecture", "training"},
        path=f"config[{spec.stable_id}]",
    )
    if (
        payload["schema"] != "eeg-mi-local-procedure-config-v2"
        or payload["version"] != 1
        or payload["model"] != spec.stable_id
    ):
        raise ProcedureGridError(f"config identity is invalid for {spec.stable_id}")
    fields = PROCEDURE_CONFIG_FIELDS[spec.key]
    _exact_keys(
        payload["architecture"],
        set(fields["architecture"]),
        path=f"config[{spec.stable_id}].architecture",
    )
    _exact_keys(
        payload["training"],
        set(fields["training"]),
        path=f"config[{spec.stable_id}].training",
    )
    settings = {
        **copy.deepcopy(dict(payload["architecture"])),
        **copy.deepcopy(dict(payload["training"])),
    }
    if (
        isinstance(settings.get("epochs"), bool)
        or not isinstance(settings.get("epochs"), int)
        or int(settings["epochs"]) <= 0
    ):
        raise ProcedureGridError(f"config epochs are invalid for {spec.stable_id}")
    if spec.key == "cameo":
        mixtures = settings.get("mixture_names")
        rhos = settings.get("rho_grid")
        if (
            not isinstance(mixtures, list)
            or not mixtures
            or len(mixtures) != len(set(mixtures))
            or any(not isinstance(value, str) or not value for value in mixtures)
            or not isinstance(rhos, list)
            or not rhos
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in rhos
            )
            or len([float(value) for value in rhos])
            != len(set(float(value) for value in rhos))
        ):
            raise ProcedureGridError("CAMEO candidate grid is invalid")
    if spec.key == "orbit_v3":
        candidates = settings.get("candidate_names")
        if (
            not isinstance(candidates, list)
            or not candidates
            or len(candidates) != len(set(candidates))
            or any(not isinstance(value, str) or not value for value in candidates)
        ):
            raise ProcedureGridError("ORBIT-v3 candidate grid is invalid")
    return settings


def _config_inventory() -> dict[str, Any]:
    available: list[dict[str, Any]] = []
    for spec in PROCEDURES:
        path = PROJECT_ROOT / spec.config_path
        try:
            payload_bytes = _read_regular_file(path)
        except (FileNotFoundError, ProcedureGridError):
            raise ProcedureGridError(f"frozen config is missing: {path}")
        observed_size = len(payload_bytes)
        observed_sha = _sha256_bytes(payload_bytes)
        if observed_size != spec.config_bytes or observed_sha != spec.config_sha256:
            raise ProcedureGridError(f"frozen config changed for {spec.stable_id}")
        payload = _strict_json_bytes(payload_bytes, source=str(path))
        settings = _validate_frozen_config_payload(spec, payload)
        if _recursive_forbidden_keys(payload):
            raise ProcedureGridError(
                f"config-only file contains an outcome field: {spec.config_path}"
            )
        available.append(
            {
                "stable_id": spec.stable_id,
                "procedure_key": spec.key,
                "status": "available_exact_outcome_free",
                "config_path": spec.config_path,
                "config_bytes": observed_size,
                "config_sha256": observed_sha,
                "settings": settings,
                "settings_sha256": _sha256_bytes(_canonical_bytes(settings)),
            }
        )

    blocked: list[dict[str, Any]] = []
    for provenance in BLOCKED_ORBIT_PROVENANCE:
        stable_id = str(provenance["stable_id"])
        suffix = stable_id.rsplit("_", 1)[-1]
        candidate = (
            PROJECT_ROOT / "configs" / "local_procedures" / f"orbit_{suffix}.json"
        )
        if candidate.exists():
            status = "blocked_unreviewed_config_file_present"
            observed = {
                "unreviewed_config_path": str(candidate.relative_to(PROJECT_ROOT)),
                "unreviewed_config_sha256": _sha256_file(candidate),
            }
        else:
            status = str(provenance["status"])
            observed = {
                "expected_config_only_path": str(candidate.relative_to(PROJECT_ROOT)),
                "expected_config_only_path_present": False,
            }
        blocked.append({**copy.deepcopy(provenance), **observed, "status": status})

    return {
        "schema": "eeg-mi-orbit-config-boundary-v1",
        "available": available,
        "blocked": blocked,
        "runtime_reads_historical_result_artifacts": False,
        "post_hoc_extraction_performed": False,
    }


def assemble_plan(
    *,
    dataset_contracts: Mapping[str, Any],
    cache_identity: Mapping[str, Any],
    split_identity: Mapping[str, Any],
    source_identity: Mapping[str, str],
    environment_identity: Mapping[str, Any],
    config_inventory: Mapping[str, Any],
    procedure_ids: Sequence[str] = tuple(value.stable_id for value in PROCEDURES),
    seeds: Sequence[int] = FORMAL_SEEDS,
    cpu_threads: int = 4,
    executor: str = FORMAL_EXECUTOR,
    publishable: bool = True,
) -> dict[str, Any]:
    procedure_ids = tuple(str(value) for value in procedure_ids)
    seeds = tuple(int(value) for value in seeds)
    if (
        not procedure_ids
        or len(set(procedure_ids)) != len(procedure_ids)
        or any(value not in PROCEDURE_BY_ID for value in procedure_ids)
    ):
        raise ValueError("procedure IDs must be unique frozen stable IDs")
    if not seeds or len(set(seeds)) != len(seeds) or any(value < 0 for value in seeds):
        raise ValueError("seeds must be unique nonnegative integers")
    if isinstance(cpu_threads, bool) or not isinstance(cpu_threads, int):
        raise TypeError("cpu_threads must be a positive integer")
    if cpu_threads <= 0:
        raise ValueError("cpu_threads must be a positive integer")
    dataset_order = list(dataset_contracts)
    if not dataset_order:
        raise ValueError("at least one dataset is required")
    expected_jobs = 0
    for contract in dataset_contracts.values():
        if int(contract["n_classes"]) != 2:
            raise ValueError("procedure grid is binary only")
        expected_jobs += (
            len(contract["subjects"])
            * len(contract["folds"])
            * len(procedure_ids)
            * len(seeds)
        )
    nvidia = environment_identity.get("nvidia")
    gpu_uuids = (
        sorted(
            str(value["uuid"])
            for value in nvidia.get("gpus", ())
            if isinstance(value, Mapping) and isinstance(value.get("uuid"), str)
        )
        if isinstance(nvidia, Mapping)
        else []
    )
    plan: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "created_at": PLAN_CREATED_AT_SENTINEL,
        "publication_mode": (
            FORMAL_PUBLICATION_MODE if publishable else TEST_PUBLICATION_MODE
        ),
        "track": "binary_neural_procedures_separate_from_common_recipe",
        "evidence_scope": (
            "opened development cohorts only; no confirmation or SOTA claim"
        ),
        "dataset_order": dataset_order,
        "datasets": copy.deepcopy(dict(dataset_contracts)),
        "procedure_ids": list(procedure_ids),
        "procedure_key_by_id": {
            stable_id: PROCEDURE_BY_ID[stable_id].key for stable_id in procedure_ids
        },
        "blocked_procedure_ids": [
            str(value["stable_id"]) for value in BLOCKED_ORBIT_PROVENANCE
        ],
        "seeds": list(seeds),
        "n_jobs": int(expected_jobs),
        "cache_identity": copy.deepcopy(dict(cache_identity)),
        "split_identity": copy.deepcopy(dict(split_identity)),
        "source_identity": copy.deepcopy(dict(source_identity)),
        "environment_identity": copy.deepcopy(dict(environment_identity)),
        "config_inventory": copy.deepcopy(dict(config_inventory)),
        "view_contract": covariance_view_contract(),
        "protocol": {
            "selection": (
                "fit preprocessing/model on train only; validation selects "
                "epoch count and predeclared route/candidate"
            ),
            "refit": (
                "fresh deterministic reset; fit preprocessing/model on "
                "train+validation for selected epoch count"
            ),
            "test": ("one predict_proba call after refit; no test score in producer"),
            "paired_views": (
                "deterministic original/reflected broadband, four-band SPD, "
                "and source-fitted tangent representations"
            ),
        },
        "executor": executor,
        "execution_config": {
            "cpu_threads_per_worker": int(cpu_threads),
            "torch_interop_threads": 1,
            "recommended_max_concurrent_gpu_workers_on_shared_four_gpu_host": 3,
            "minimum_free_gib": DEFAULT_MIN_FREE_GIB,
            "device": "cuda:0" if publishable else "cpu",
            "physical_gpu_uuid_roster": gpu_uuids,
        },
        "analysis_config": copy.deepcopy(dict(FORMAL_ANALYSIS_CONFIG)),
    }
    plan["plan_sha256"] = plan_sha256(plan)
    return plan


def plan_sha256(plan: Mapping[str, Any]) -> str:
    value = copy.deepcopy(dict(plan))
    value.pop("plan_sha256", None)
    return _sha256_bytes(_canonical_bytes(value))


def _expected_protocol() -> dict[str, str]:
    return {
        "selection": (
            "fit preprocessing/model on train only; validation selects "
            "epoch count and predeclared route/candidate"
        ),
        "refit": (
            "fresh deterministic reset; fit preprocessing/model on "
            "train+validation for selected epoch count"
        ),
        "test": "one predict_proba call after refit; no test score in producer",
        "paired_views": (
            "deterministic original/reflected broadband, four-band SPD, "
            "and source-fitted tangent representations"
        ),
    }


def _validate_inventory_structure(
    inventory: Any, *, procedure_ids: Sequence[str], formal: bool
) -> None:
    _exact_keys(
        inventory,
        {
            "schema",
            "available",
            "blocked",
            "runtime_reads_historical_result_artifacts",
            "post_hoc_extraction_performed",
        },
        path="plan.config_inventory",
    )
    if (
        inventory["schema"] != "eeg-mi-orbit-config-boundary-v1"
        or inventory["runtime_reads_historical_result_artifacts"] is not False
        or inventory["post_hoc_extraction_performed"] is not False
        or not isinstance(inventory["available"], list)
        or not isinstance(inventory["blocked"], list)
    ):
        raise ProcedureGridError("config boundary flags are invalid")
    expected_available_keys = {
        "stable_id",
        "procedure_key",
        "status",
        "config_path",
        "config_bytes",
        "config_sha256",
        "settings",
        "settings_sha256",
    }
    observed_ids: list[str] = []
    for index, value in enumerate(inventory["available"]):
        _exact_keys(
            value,
            expected_available_keys,
            path=f"plan.config_inventory.available[{index}]",
        )
        stable_id = value["stable_id"]
        if stable_id not in PROCEDURE_BY_ID:
            raise ProcedureGridError("config inventory has an unknown procedure")
        spec = PROCEDURE_BY_ID[stable_id]
        settings = value["settings"]
        if (
            value["procedure_key"] != spec.key
            or value["status"] != "available_exact_outcome_free"
            or value["config_path"] != spec.config_path
            or value["config_bytes"] != spec.config_bytes
            or value["config_sha256"] != spec.config_sha256
            or not isinstance(settings, Mapping)
            or set(settings)
            != (
                set(PROCEDURE_CONFIG_FIELDS[spec.key]["architecture"])
                | set(PROCEDURE_CONFIG_FIELDS[spec.key]["training"])
            )
            or value["settings_sha256"] != _sha256_bytes(_canonical_bytes(settings))
        ):
            raise ProcedureGridError("config inventory identity is invalid")
        observed_ids.append(str(stable_id))
    if observed_ids != list(procedure_ids):
        raise ProcedureGridError("config inventory order differs from procedures")
    if [value.get("stable_id") for value in inventory["blocked"]] != [
        value["stable_id"] for value in BLOCKED_ORBIT_PROVENANCE
    ]:
        raise ProcedureGridError("blocked ORBIT provenance roster is invalid")
    if formal and inventory != _config_inventory():
        raise ProcedureGridError("formal config boundary differs from frozen files")


def _validate_cache_identity_contract(
    value: Mapping[str, Any],
    *,
    dataset: str,
    subject: int,
    formal: bool,
) -> None:
    aliases = _recursive_cache_outcome_aliases(
        value,
        path=f"plan.cache_identity[{_subject_key(dataset, subject)}]",
    )
    if aliases:
        raise ProcedureGridError(f"cache identity has outcome aliases: {aliases[:5]}")
    if not formal:
        required = {"array_sha256", "trial_count", "n_channels"}
        if not required.issubset(value):
            raise ProcedureGridError("test cache identity lacks its minimum geometry")
        return
    expected_keys = {
        "dataset",
        "subject",
        "montage_profile",
        "preprocessing",
        "coordinates",
        "shape",
        "channels",
        "array_sha256",
        "trial_count",
        "n_channels",
        "positions_manifest",
    }
    if dataset == "local_exp4":
        expected_keys.add("source_manifest")
    _exact_keys(
        value,
        expected_keys,
        path=f"plan.cache_identity[{_subject_key(dataset, subject)}]",
    )
    expected_dataset = _json_ready(asdict(dataset_spec(dataset)))
    channels = value["channels"]
    shape = value["shape"]
    if (
        value["dataset"] != expected_dataset
        or type(value["subject"]) is not int
        or value["subject"] != subject
        or value["montage_profile"] != "harmonized"
        or value["preprocessing"] != preprocessing_for_dataset(dataset)
        or not isinstance(channels, list)
        or not channels
        or any(not isinstance(channel, str) or not channel for channel in channels)
        or len(channels) != len(set(channels))
        or not isinstance(shape, list)
        or len(shape) != 3
        or any(type(dimension) is not int or dimension <= 0 for dimension in shape)
        or shape
        != [
            value["trial_count"],
            value["n_channels"],
            int(value["preprocessing"]["n_times"]),
        ]
        or len(channels) != value["n_channels"]
        or value["coordinates"]
        != coordinate_contract_for_dataset(
            dataset,
            "harmonized",
            tuple(channels),
        )
    ):
        raise ProcedureGridError("formal cache identity geometry/contract is invalid")
    positions = value["positions_manifest"]
    _exact_keys(
        positions,
        {"shape", "dtype", "sha256"},
        path=f"plan.cache_identity[{_subject_key(dataset, subject)}].positions",
    )
    if (
        positions["shape"] != [value["n_channels"], 3]
        or positions["dtype"] != np.dtype(np.float32).str
        or not isinstance(positions["sha256"], str)
        or not HEX_64_RE.fullmatch(positions["sha256"])
    ):
        raise ProcedureGridError("formal cache position binding is invalid")
    if dataset == "local_exp4":
        manifest = value["source_manifest"]
        expected_runs = LOCAL_EXP4_SUBJECT_RUNS[subject]
        if not isinstance(manifest, list) or len(manifest) != len(expected_runs):
            raise ProcedureGridError("local cache source manifest roster is invalid")
        for entry, run in zip(manifest, expected_runs, strict=True):
            _exact_keys(
                entry,
                {"subject", "run", "filename", "size_bytes", "sha256"},
                path="plan.cache_identity.source_manifest[]",
            )
            if (
                type(entry["subject"]) is not int
                or entry["subject"] != subject
                or type(entry["run"]) is not int
                or entry["run"] != run
                or entry["filename"]
                != f"exp4_subject{subject}_training_{run}_mi_raw.fif"
                or type(entry["size_bytes"]) is not int
                or entry["size_bytes"] <= 0
                or not isinstance(entry["sha256"], str)
                or not HEX_64_RE.fullmatch(entry["sha256"])
            ):
                raise ProcedureGridError("local cache source manifest is invalid")


def _validate_split_structure(plan: Mapping[str, Any], *, formal: bool) -> None:
    cache_identity = plan["cache_identity"]
    split_identity = plan["split_identity"]
    if not isinstance(cache_identity, Mapping) or not isinstance(
        split_identity, Mapping
    ):
        raise ProcedureGridError("cache/split identity is not an object")
    expected_cache_keys: set[str] = set()
    expected_split_keys: set[str] = set()
    for dataset in plan["dataset_order"]:
        contract = plan["datasets"][dataset]
        for subject in contract["subjects"]:
            cache_key = _subject_key(str(dataset), int(subject))
            expected_cache_keys.add(cache_key)
            value = cache_identity.get(cache_key)
            if not isinstance(value, Mapping):
                if formal:
                    raise ProcedureGridError(
                        f"formal cache identity is absent: {cache_key}"
                    )
                continue
            _validate_cache_identity_contract(
                value,
                dataset=str(dataset),
                subject=int(subject),
                formal=formal,
            )
            if (
                not isinstance(value.get("array_sha256"), str)
                or not HEX_64_RE.fullmatch(str(value["array_sha256"]))
                or isinstance(value.get("trial_count"), bool)
                or not isinstance(value.get("trial_count"), int)
                or int(value["trial_count"]) <= 0
                or isinstance(value.get("n_channels"), bool)
                or not isinstance(value.get("n_channels"), int)
                or int(value["n_channels"]) <= 0
            ):
                raise ProcedureGridError(f"cache identity is invalid: {cache_key}")
            for fold in contract["folds"]:
                split_key = _split_key(str(dataset), int(subject), int(fold))
                expected_split_keys.add(split_key)
                split = split_identity.get(split_key)
                if not isinstance(split, Mapping):
                    if formal:
                        raise ProcedureGridError(
                            f"formal split identity is absent: {split_key}"
                        )
                    continue
                _exact_keys(
                    split,
                    {
                        "dataset",
                        "subject",
                        "fold",
                        "cache_array_sha256",
                        "trial_count",
                        "partitions",
                    },
                    path=f"plan.split_identity[{split_key}]",
                )
                if (
                    split["dataset"] != dataset
                    or type(split["subject"]) is not int
                    or split["subject"] != int(subject)
                    or type(split["fold"]) is not int
                    or split["fold"] != int(fold)
                    or not isinstance(split["cache_array_sha256"], str)
                    or split["cache_array_sha256"] != value["array_sha256"]
                    or type(split["trial_count"]) is not int
                    or split["trial_count"] != value["trial_count"]
                ):
                    raise ProcedureGridError(f"split identity mismatch: {split_key}")
                _exact_keys(
                    split["partitions"],
                    {"train", "validation", "source", "test"},
                    path=f"plan.split_identity[{split_key}].partitions",
                )
                for name, partition in split["partitions"].items():
                    _exact_keys(
                        partition,
                        {"count", "rows_sha256"},
                        path=f"plan.split_identity[{split_key}].partitions.{name}",
                    )
                    if (
                        isinstance(partition["count"], bool)
                        or not isinstance(partition["count"], int)
                        or int(partition["count"]) <= 0
                        or not isinstance(partition["rows_sha256"], str)
                        or not HEX_64_RE.fullmatch(partition["rows_sha256"])
                    ):
                        raise ProcedureGridError(
                            f"split partition is invalid: {split_key}/{name}"
                        )
                if (
                    split["partitions"]["source"]["count"]
                    != split["partitions"]["train"]["count"]
                    + split["partitions"]["validation"]["count"]
                    or split["partitions"]["source"]["count"]
                    + split["partitions"]["test"]["count"]
                    != split["trial_count"]
                ):
                    raise ProcedureGridError(
                        f"split source count is invalid: {split_key}"
                    )
    if formal and set(cache_identity) != expected_cache_keys:
        raise ProcedureGridError("formal cache identity roster is not exact")
    if formal and set(split_identity) != expected_split_keys:
        raise ProcedureGridError("formal split identity roster is not exact")


def validate_plan_semantics(
    plan: Mapping[str, Any], *, allow_unsealed: bool = False
) -> None:
    """Validate the complete formal contract or an explicit nonpublishable test plan."""

    _exact_keys(
        plan,
        {
            "schema",
            "created_at",
            "publication_mode",
            "track",
            "evidence_scope",
            "dataset_order",
            "datasets",
            "procedure_ids",
            "procedure_key_by_id",
            "blocked_procedure_ids",
            "seeds",
            "n_jobs",
            "cache_identity",
            "split_identity",
            "source_identity",
            "environment_identity",
            "config_inventory",
            "view_contract",
            "protocol",
            "executor",
            "execution_config",
            "analysis_config",
            "plan_sha256",
        },
        path="plan",
    )
    if plan["schema"] != PLAN_SCHEMA or plan["track"] != (
        "binary_neural_procedures_separate_from_common_recipe"
    ):
        raise ProcedureGridError("plan schema/track is invalid")
    formal = plan["publication_mode"] == FORMAL_PUBLICATION_MODE
    if plan["publication_mode"] not in {
        FORMAL_PUBLICATION_MODE,
        TEST_PUBLICATION_MODE,
    }:
        raise ProcedureGridError("plan publication mode is invalid")
    created_at = plan["created_at"]
    if created_at == PLAN_CREATED_AT_SENTINEL:
        if not allow_unsealed:
            raise ProcedureGridError("published plan still has an unsealed timestamp")
    else:
        try:
            parsed = datetime.fromisoformat(str(created_at))
        except (TypeError, ValueError) as error:
            raise ProcedureGridError("plan creation timestamp is invalid") from error
        if parsed.tzinfo is None:
            raise ProcedureGridError("plan creation timestamp lacks a timezone")
    if plan["plan_sha256"] != plan_sha256(plan):
        raise ProcedureGridError("plan digest does not match its semantics")
    if (
        plan["evidence_scope"]
        != "opened development cohorts only; no confirmation or SOTA claim"
        or plan["view_contract"] != covariance_view_contract()
        or plan["protocol"] != _expected_protocol()
        or plan["executor"] != FORMAL_EXECUTOR
        or plan["analysis_config"] != dict(FORMAL_ANALYSIS_CONFIG)
    ):
        raise ProcedureGridError(
            "plan protocol/view/executor/analysis contract drifted"
        )
    if (
        not isinstance(plan["dataset_order"], list)
        or not plan["dataset_order"]
        or len(plan["dataset_order"]) != len(set(plan["dataset_order"]))
        or not isinstance(plan["datasets"], Mapping)
        or set(plan["datasets"]) != set(plan["dataset_order"])
        or not isinstance(plan["procedure_ids"], list)
        or not plan["procedure_ids"]
        or len(plan["procedure_ids"]) != len(set(plan["procedure_ids"]))
        or any(value not in PROCEDURE_BY_ID for value in plan["procedure_ids"])
        or plan["procedure_key_by_id"]
        != {
            stable_id: PROCEDURE_BY_ID[stable_id].key
            for stable_id in plan["procedure_ids"]
        }
        or plan["blocked_procedure_ids"]
        != [str(value["stable_id"]) for value in BLOCKED_ORBIT_PROVENANCE]
        or not isinstance(plan["seeds"], list)
        or not plan["seeds"]
        or len(plan["seeds"]) != len(set(plan["seeds"]))
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in plan["seeds"]
        )
    ):
        raise ProcedureGridError("plan roster is invalid")
    for dataset, contract in plan["datasets"].items():
        aliases = _recursive_cache_outcome_aliases(
            contract, path=f"plan.datasets.{dataset}"
        )
        if aliases:
            raise ProcedureGridError(
                f"dataset contract contains outcome aliases: {aliases[:5]}"
            )
        _exact_keys(
            contract,
            {"subjects", "folds", "n_classes", "events", "protocol", "preprocessing"},
            path=f"plan.datasets.{dataset}",
        )
        if (
            contract["n_classes"] != 2
            or not isinstance(contract["subjects"], list)
            or not contract["subjects"]
            or len(contract["subjects"]) != len(set(contract["subjects"]))
            or not isinstance(contract["folds"], list)
            or not contract["folds"]
            or len(contract["folds"]) != len(set(contract["folds"]))
            or any(
                type(value) is not int or value <= 0 for value in contract["subjects"]
            )
            or any(type(value) is not int or value < 0 for value in contract["folds"])
            or not isinstance(contract["events"], list)
            or len(contract["events"]) != 2
            or len(contract["events"]) != len(set(contract["events"]))
            or any(
                not isinstance(value, str) or not value for value in contract["events"]
            )
            or not isinstance(contract["protocol"], str)
            or not contract["protocol"]
            or not isinstance(contract["preprocessing"], Mapping)
        ):
            raise ProcedureGridError(f"dataset contract is invalid: {dataset}")
    _exact_keys(
        plan["execution_config"],
        {
            "cpu_threads_per_worker",
            "torch_interop_threads",
            "recommended_max_concurrent_gpu_workers_on_shared_four_gpu_host",
            "minimum_free_gib",
            "device",
            "physical_gpu_uuid_roster",
        },
        path="plan.execution_config",
    )
    execution = plan["execution_config"]
    if (
        isinstance(execution["cpu_threads_per_worker"], bool)
        or not isinstance(execution["cpu_threads_per_worker"], int)
        or not 1 <= execution["cpu_threads_per_worker"] <= 16
        or execution["torch_interop_threads"] != 1
        or execution["recommended_max_concurrent_gpu_workers_on_shared_four_gpu_host"]
        != 3
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
        or execution["device"] != ("cuda:0" if formal else "cpu")
    ):
        raise ProcedureGridError("plan execution contract is invalid")
    expected_jobs = sum(
        len(plan["datasets"][dataset]["subjects"])
        * len(plan["datasets"][dataset]["folds"])
        * len(plan["procedure_ids"])
        * len(plan["seeds"])
        for dataset in plan["dataset_order"]
    )
    jobs = tuple(iter_jobs(plan))
    if (
        plan["n_jobs"] != expected_jobs
        or len(jobs) != expected_jobs
        or len({job.job_id for job in jobs}) != expected_jobs
    ):
        raise ProcedureGridError("plan job roster is inconsistent")
    _validate_inventory_structure(
        plan["config_inventory"],
        procedure_ids=plan["procedure_ids"],
        formal=formal,
    )
    if (
        not isinstance(plan["source_identity"], Mapping)
        or not plan["source_identity"]
        or any(
            not isinstance(name, str)
            or not name
            or not isinstance(digest, str)
            or not HEX_64_RE.fullmatch(digest)
            for name, digest in plan["source_identity"].items()
        )
        or not isinstance(plan["environment_identity"], Mapping)
    ):
        raise ProcedureGridError("source/environment identity is invalid")
    _validate_split_structure(plan, formal=formal)
    if formal:
        environment = plan["environment_identity"]
        _exact_keys(
            environment,
            {
                "schema",
                "python",
                "executable",
                "prefix",
                "base_prefix",
                "virtual_environment",
                "platform",
                "uv_version",
                "packages",
                "packages_sha256",
                "torch",
                "nvidia",
                "cublas_workspace_config",
                "thread_environment",
            },
            path="plan.environment_identity",
        )
        nvidia = environment["nvidia"]
        torch_identity = environment["torch"]
        _exact_keys(
            torch_identity,
            {"version", "cuda_runtime", "cuda_available"},
            path="plan.environment_identity.torch",
        )
        if (
            not isinstance(environment["packages"], list)
            or any(
                not isinstance(value, list)
                or len(value) != 2
                or any(not isinstance(item, str) for item in value)
                for value in environment["packages"]
            )
            or environment["packages_sha256"]
            != _sha256_bytes(
                _canonical_bytes([tuple(value) for value in environment["packages"]])
            )
            or environment["cublas_workspace_config"]
            != REQUIRED_CUBLAS_WORKSPACE_CONFIG
            or environment["thread_environment"]
            != {
                name: str(execution["cpu_threads_per_worker"])
                for name in THREAD_ENVIRONMENT_VARIABLES
            }
            or torch_identity["cuda_available"] is not True
        ):
            raise ProcedureGridError("formal environment contract is invalid")
        if isinstance(nvidia, Mapping) and isinstance(nvidia.get("gpus"), list):
            for index, gpu in enumerate(nvidia["gpus"]):
                _exact_keys(
                    gpu,
                    {"index", "uuid", "pci_bus_id", "name", "driver_version"},
                    path=f"plan.environment_identity.nvidia.gpus[{index}]",
                )
                if (
                    not isinstance(gpu["index"], str)
                    or not gpu["index"].isdigit()
                    or not GPU_UUID_RE.fullmatch(str(gpu["uuid"]))
                ):
                    raise ProcedureGridError("formal GPU inventory is invalid")
        if (
            tuple(plan["dataset_order"]) != BINARY_DATASETS
            or plan["datasets"] != _dataset_contracts()
            or tuple(plan["procedure_ids"])
            != tuple(value.stable_id for value in PROCEDURES)
            or tuple(plan["seeds"]) != FORMAL_SEEDS
            or plan["n_jobs"] != FORMAL_EXPECTED_JOBS
            or plan["source_identity"] != _source_identity()
            or set(plan["source_identity"]) != set(SOURCE_FILES)
            or not isinstance(nvidia, Mapping)
            or set(nvidia) != {"available", "gpus"}
            or nvidia["available"] is not True
            or not execution["physical_gpu_uuid_roster"]
            or execution["physical_gpu_uuid_roster"]
            != sorted(str(value["uuid"]) for value in nvidia["gpus"])
        ):
            raise ProcedureGridError("formal plan semantics are not exact")


def _plan_semantic_payload(plan: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(plan))
    value.pop("created_at", None)
    value.pop("plan_sha256", None)
    return value


def write_or_validate_plan(run_root: Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    expected = copy.deepcopy(dict(plan))
    validate_plan_semantics(expected, allow_unsealed=True)
    if expected["publication_mode"] == FORMAL_PUBLICATION_MODE:
        _require_formal_release_runtime()
    root = _safe_run_root(run_root, create=True)
    path = root / "plan.json"
    digest_path = root / "plan.sha256"
    if path.exists() or digest_path.exists():
        observed = load_or_repair_plan(root)
        if _plan_semantic_payload(observed) != _plan_semantic_payload(expected):
            raise ProcedureGridError(
                "existing immutable plan differs from requested plan"
            )
        return observed
    if expected["created_at"] == PLAN_CREATED_AT_SENTINEL:
        expected["created_at"] = _utc_now()
        expected["plan_sha256"] = plan_sha256(expected)
    validate_plan_semantics(expected)
    staged_plan = _stage_run_json(
        root,
        basename="plan",
        value=expected,
    )
    try:
        _atomic_rename_noreplace(staged_plan, path)
    except FileExistsError:
        observed = load_or_repair_plan(root)
        if _plan_semantic_payload(observed) != _plan_semantic_payload(expected):
            raise ProcedureGridError(
                "concurrently published plan differs from requested plan"
            )
        return observed
    _fsync_directory(root)
    # If this block is interrupted, the canonical orphaned plan is safely
    # repairable by ``load_or_repair_plan``.
    staged_digest = _stage_run_bytes(
        root,
        basename="plan-sha256",
        payload=(str(expected["plan_sha256"]) + "\n").encode("ascii"),
    )
    try:
        _atomic_rename_noreplace(staged_digest, digest_path)
    except FileExistsError:
        pass
    _fsync_directory(root)
    return load_plan(root)


def load_plan(run_root: Path) -> dict[str, Any]:
    root = _safe_run_root(run_root)
    path = root / "plan.json"
    digest_path = root / "plan.sha256"
    if not path.exists() or not digest_path.exists():
        raise FileNotFoundError("plan.json/plan.sha256 is absent")
    for candidate in (path, digest_path):
        _require_unique_regular(candidate, read_only=True)
    plan_bytes = _read_regular_file(path)
    plan = _strict_json_bytes(plan_bytes, source=str(path))
    digest_bytes = _read_regular_file(digest_path)
    try:
        observed = digest_bytes.decode("ascii").strip()
    except UnicodeDecodeError as error:
        raise ProcedureGridError("plan digest is not ASCII") from error
    expected = plan_sha256(plan)
    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("plan_sha256") != expected
        or observed != expected
        or plan_bytes != _canonical_bytes(plan) + b"\n"
        or digest_bytes != (expected + "\n").encode("ascii")
    ):
        raise ProcedureGridError("immutable plan is noncanonical or corrupt")
    validate_plan_semantics(plan)
    if plan["publication_mode"] == FORMAL_PUBLICATION_MODE:
        _require_formal_release_runtime()
    return plan


def load_or_repair_plan(run_root: Path) -> dict[str, Any]:
    """Repair only the safe plan-write power-cut state.

    ``plan.json`` is published first and is already canonical, checksummed
    internally, and read-only.  A power loss can occur before ``plan.sha256``
    is created.  This function validates the complete orphaned plan before
    creating only the missing digest receipt.  Every other partial state fails
    closed.
    """

    root = _safe_run_root(run_root)
    path = root / "plan.json"
    digest_path = root / "plan.sha256"
    if path.exists() and digest_path.exists():
        return load_plan(root)
    if not path.exists() or digest_path.exists():
        raise ProcedureGridError("plan power-cut state is not safely repairable")
    plan_bytes = _read_regular_file(path)
    plan = _strict_json_bytes(plan_bytes, source=str(path))
    expected = plan_sha256(plan)
    try:
        _require_unique_regular(path, read_only=True)
        validate_plan_semantics(plan)
        if plan["publication_mode"] == FORMAL_PUBLICATION_MODE:
            _require_formal_release_runtime()
    except (FileNotFoundError, ProcedureGridError):
        raise ProcedureGridError("orphaned plan.json is invalid")
    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("plan_sha256") != expected
        or plan_bytes != _canonical_bytes(plan) + b"\n"
    ):
        raise ProcedureGridError("orphaned plan.json is invalid")
    staged_digest = _stage_run_bytes(
        root,
        basename="plan-sha256-repair",
        payload=(expected + "\n").encode("ascii"),
    )
    try:
        _atomic_rename_noreplace(staged_digest, digest_path)
    except FileExistsError:
        pass
    _fsync_directory(root)
    return load_plan(root)


def verify_runtime_identity(plan: Mapping[str, Any], *, cache_root: Path) -> None:
    validate_plan_semantics(plan)
    if plan.get("publication_mode") != FORMAL_PUBLICATION_MODE:
        raise ProcedureGridError("runtime identity verification requires a formal plan")
    if plan.get("config_inventory") != _config_inventory():
        raise ProcedureGridError("config boundary drifted")
    if plan.get("source_identity") != _source_identity():
        raise ProcedureGridError("source closure drifted")
    planned_environment = copy.deepcopy(dict(plan["environment_identity"]))
    observed_environment = _environment_identity()
    if planned_environment != observed_environment:
        raise ProcedureGridError("UV execution environment drifted")
    cache, splits = _cache_and_split_identity(cache_root, plan["datasets"])
    if cache != plan.get("cache_identity"):
        raise ProcedureGridError("cache identity drifted")
    if splits != plan.get("split_identity"):
        raise ProcedureGridError("split identity drifted")


def iter_jobs(plan: Mapping[str, Any]) -> Iterator[Job]:
    for stable_id in plan["procedure_ids"]:
        for dataset in plan["dataset_order"]:
            contract = plan["datasets"][dataset]
            for raw_subject in contract["subjects"]:
                for raw_fold in contract["folds"]:
                    for raw_seed in plan["seeds"]:
                        yield Job(
                            dataset=str(dataset),
                            stable_id=str(stable_id),
                            subject=int(raw_subject),
                            fold=int(raw_fold),
                            seed=int(raw_seed),
                        )


def _planned_cache(plan: Mapping[str, Any], job: Job) -> Mapping[str, Any]:
    value = plan["cache_identity"].get(_subject_key(job.dataset, job.subject))
    if not isinstance(value, Mapping):
        raise ProcedureGridError("planned cache identity is absent")
    return value


def _planned_split(plan: Mapping[str, Any], job: Job) -> Mapping[str, Any]:
    value = plan["split_identity"].get(_split_key(job.dataset, job.subject, job.fold))
    if not isinstance(value, Mapping):
        raise ProcedureGridError("planned split identity is absent")
    return value


def _filter_designs(sfreq: float) -> tuple[np.ndarray, ...]:
    from scipy import signal

    if not math.isclose(float(sfreq), 128.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"procedure view requires 128 Hz, got {sfreq}")
    return tuple(
        signal.butter(
            FILTER_ORDER,
            (low, high),
            btype="bandpass",
            fs=float(sfreq),
            output="sos",
        ).astype(np.float64, copy=False)
        for low, high in SOURCE_BANDS_HZ
    )


def covariance_view_contract(sfreq: float = 128.0) -> dict[str, Any]:
    designs = _filter_designs(sfreq)
    return {
        "schema": VIEW_SCHEMA,
        "input": "unaltered harmonized eeg-mi-cache-v2 broadband trial",
        "fit_scope": (
            "independent deterministic filtering/covariance per trial; no "
            "labels and no cross-trial statistics"
        ),
        "reflection": (
            "benchmark.shared.augment.left_right_swap_index; raw P*x and SPD P*C*P^T"
        ),
        "bands_hz": [list(value) for value in SOURCE_BANDS_HZ],
        "filter": {
            "family": "scipy.signal.butter",
            "order": FILTER_ORDER,
            "representation": "second-order sections",
            "application": "scipy.signal.sosfiltfilt on time axis",
            "phase": "zero",
            "padtype": "odd",
            "padlen": "scipy default",
            "sfreq_hz": float(sfreq),
            "sos_sha256": [_array_sha256(value) for value in designs],
        },
        "covariance": {
            "implementation": "benchmark.research.data.make_spd_covariances",
            "demean": True,
            "shrinkage": 1e-3,
            "shrinkage_method": "fixed",
            "dtype": "float32",
        },
        "tangent": {
            "implementation": "benchmark.research.cameo_net.FrozenTangentAnchor",
            "selection_fit_rows": "train only plus deterministic reflections",
            "refit_fit_rows": ("train+validation only plus deterministic reflections"),
            "test_fit_use": False,
        },
    }


def _left_right_swap_index(channel_names: Sequence[str]) -> np.ndarray:
    """Return the exact odd/even 10-20 reflection without importing PyTorch."""

    channels = tuple(str(value) for value in channel_names)
    name_to_index = {name: index for index, name in enumerate(channels)}
    permutation = list(range(len(channels)))
    for index, name in enumerate(channels):
        if not name or not name[-1].isdigit():
            continue
        digit = int(name[-1])
        partner_digit = digit + 1 if digit % 2 == 1 else digit - 1
        partner = f"{name[:-1]}{partner_digit}"
        if partner in name_to_index:
            permutation[index] = name_to_index[partner]
    return np.asarray(permutation, dtype=np.int64)


def derive_paired_views(
    raw_epochs: NDArray[np.floating],
    *,
    channel_names: Sequence[str],
    sfreq: float = 128.0,
) -> dict[str, np.ndarray]:
    """Construct deterministic original/reflected broadband and four-band SPD."""

    from scipy import signal

    from benchmark.research.data import make_spd_covariances

    raw = np.ascontiguousarray(raw_epochs, dtype=np.float32)
    channels = tuple(str(value) for value in channel_names)
    if (
        raw.ndim != 3
        or raw.shape[1] != len(channels)
        or raw.shape[-1] not in {256, 320}
        or not np.all(np.isfinite(raw))
        or not channels
        or len(set(channels)) != len(channels)
    ):
        raise ValueError("invalid harmonized-v2 broadband array")
    mirror = _left_right_swap_index(channels)
    if (
        mirror.shape != (len(channels),)
        or sorted(mirror.tolist()) != list(range(len(channels)))
        or not np.array_equal(mirror[mirror], np.arange(len(channels)))
    ):
        raise ProcedureGridError("left/right reflection is not an involution")
    reflected_raw = np.ascontiguousarray(raw[:, mirror, :], dtype=np.float32)
    if not np.array_equal(
        reflected_raw[:, mirror, :].view(np.uint8), raw.view(np.uint8)
    ):
        raise ProcedureGridError("raw reflection round trip failed")

    filtered: list[np.ndarray] = []
    for design in _filter_designs(sfreq):
        value = signal.sosfiltfilt(
            design,
            raw.astype(np.float64, copy=False),
            axis=-1,
            padtype="odd",
            padlen=None,
        )
        filtered.append(value.astype(np.float32, copy=False))
    filter_bank = np.ascontiguousarray(np.stack(filtered, axis=1))
    covariance = np.ascontiguousarray(
        make_spd_covariances(
            filter_bank,
            shrinkage=1e-3,
            method="fixed",
            demean=True,
            dtype=np.float32,
        ),
        dtype=np.float32,
    )
    reflected_covariance = np.ascontiguousarray(
        covariance[..., mirror, :][..., :, mirror], dtype=np.float32
    )
    covariance_round_trip = np.ascontiguousarray(
        reflected_covariance[..., mirror, :][..., :, mirror],
        dtype=np.float32,
    )
    if not np.array_equal(
        covariance_round_trip.view(np.uint8),
        covariance.view(np.uint8),
    ):
        raise ProcedureGridError("SPD reflection round trip failed")
    return {
        "raw": raw,
        "raw_reflected": reflected_raw,
        "covariance": covariance,
        "covariance_reflected": reflected_covariance,
        "mirror_index": mirror,
    }


def _tangent_pair_manifest(
    classifier: Any,
    covariance: np.ndarray,
    covariance_reflected: np.ndarray,
) -> dict[str, Any]:
    original = np.ascontiguousarray(
        classifier.anchor_.transform(covariance), dtype=np.float32
    )
    reflected = np.ascontiguousarray(
        classifier.anchor_.transform(covariance_reflected), dtype=np.float32
    )
    if original.shape != reflected.shape or not (
        np.all(np.isfinite(original)) and np.all(np.isfinite(reflected))
    ):
        raise ProcedureGridError("paired tangent view is malformed")
    return {
        "original": _array_manifest(original),
        "reflected": _array_manifest(reflected),
    }


def _configure_determinism(seed: int, cpu_threads: int) -> None:
    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", REQUIRED_CUBLAS_WORKSPACE_CONFIG)
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != REQUIRED_CUBLAS_WORKSPACE_CONFIG:
        raise ProcedureGridError(
            "CUBLAS_WORKSPACE_CONFIG differs from the deterministic contract"
        )
    for name in THREAD_ENVIRONMENT_VARIABLES:
        os.environ[name] = str(int(cpu_threads))
    torch.set_num_threads(int(cpu_threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)


def _install_thread_environment(cpu_threads: int) -> None:
    if isinstance(cpu_threads, bool) or int(cpu_threads) <= 0:
        raise ValueError("cpu_threads must be positive")
    for name in THREAD_ENVIRONMENT_VARIABLES:
        os.environ[name] = str(int(cpu_threads))


def execute_procedure_job(
    *,
    job: Job,
    plan: Mapping[str, Any],
    cache_root: Path,
    device: str,
    cpu_threads: int = 4,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Fit one procedure record and return score-blind publication material."""

    import torch

    from . import local_outer_refit_benchmark as legacy

    from .data import load_subject_cache, split_indices

    if job.stable_id not in PROCEDURE_BY_ID:
        raise ProcedureGridError(f"{job.stable_id} has no outcome-free exact config")
    dataset_contract = plan.get("datasets", {}).get(job.dataset)
    if (
        not isinstance(dataset_contract, Mapping)
        or dataset_contract.get("n_classes") != 2
    ):
        raise ProcedureGridError("procedure executor is binary-only")
    started = time.perf_counter()
    _configure_determinism(job.seed, cpu_threads)
    target_device = torch.device(device)
    if str(target_device) != str(plan["execution_config"]["device"]):
        raise ProcedureGridError("executor device differs from immutable plan")
    physical_gpu_uuid: str | None = None
    if target_device.type == "cuda":
        if not torch.cuda.is_available():
            raise ProcedureGridError("CUDA was requested but is unavailable")
        from .full_grid import _visible_cuda_uuid

        physical_gpu_uuid = _visible_cuda_uuid()
        if (
            physical_gpu_uuid
            not in plan["execution_config"]["physical_gpu_uuid_roster"]
        ):
            raise ProcedureGridError("CUDA-visible physical GPU is outside the plan")
        torch.cuda.reset_peak_memory_stats(target_device)

    data = load_subject_cache(job.dataset, job.subject, cache_root=cache_root)
    planned_cache = _planned_cache(plan, job)
    observed_cache = _cache_identity_from_loaded(data)
    if observed_cache != planned_cache:
        raise ProcedureGridError("runtime cache differs from plan")
    train, validation, test = split_indices(
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
        train_rows=train,
        validation_rows=validation,
        test_rows=test,
    )
    if observed_split != _planned_split(plan, job):
        raise ProcedureGridError("runtime split differs from plan")
    source = np.sort(np.concatenate((train, validation)))
    channels = tuple(str(value) for value in data["channel_names"].tolist())
    sfreq = float(plan["datasets"][job.dataset]["preprocessing"]["sfreq_hz"])
    views = derive_paired_views(data["x"], channel_names=channels, sfreq=sfreq)
    spec = PROCEDURE_BY_ID[job.stable_id]
    config, config_identity = legacy._load_frozen_config(
        spec.key, seed=job.seed, device=device
    )
    expected_config = next(
        value
        for value in plan["config_inventory"]["available"]
        if value["stable_id"] == job.stable_id
    )
    if (
        config_identity["config_file_sha256"] != expected_config["config_sha256"]
        or config_identity["config_file_bytes"] != expected_config["config_bytes"]
    ):
        raise ProcedureGridError("runtime config differs from plan")

    selection_started = time.perf_counter()
    selection, selection_detail = legacy._selection_fit(
        spec.key,
        config,
        raw_train=views["raw"][train],
        covariance_train=views["covariance"][train],
        labels_train=np.asarray(data["y"][train], dtype=np.int64),
        raw_validation=views["raw"][validation],
        covariance_validation=views["covariance"][validation],
        labels_validation=np.asarray(data["y"][validation], dtype=np.int64),
        channels=channels,
    )
    selection_seconds = time.perf_counter() - selection_started
    selection_tangent = {
        "train": _tangent_pair_manifest(
            selection,
            views["covariance"][train],
            views["covariance_reflected"][train],
        ),
        "validation": _tangent_pair_manifest(
            selection,
            views["covariance"][validation],
            views["covariance_reflected"][validation],
        ),
    }

    refit_started = time.perf_counter()
    refitted, refit_detail = legacy._refit(
        spec.key,
        config,
        selection,
        selection_detail,
        raw_source=views["raw"][source],
        covariance_source=views["covariance"][source],
        labels_source=np.asarray(data["y"][source], dtype=np.int64),
        channels=channels,
    )
    refit_seconds = time.perf_counter() - refit_started
    refit_tangent = {
        "source": _tangent_pair_manifest(
            refitted,
            views["covariance"][source],
            views["covariance_reflected"][source],
        ),
        "test": _tangent_pair_manifest(
            refitted,
            views["covariance"][test],
            views["covariance_reflected"][test],
        ),
    }

    # This is the single held-out estimator call.  No diagnostic helper or
    # producer-side score is allowed to call the estimator a second time.
    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)
    inference_started = time.perf_counter()
    probabilities = np.asarray(
        refitted.predict_proba(views["raw"][test], views["covariance"][test]),
        dtype=np.float64,
    )
    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)
    inference_seconds = time.perf_counter() - inference_started
    if (
        probabilities.shape != (len(test), 2)
        or not np.all(np.isfinite(probabilities))
        or np.any(probabilities < 0.0)
        or np.any(probabilities > 1.0)
        or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    ):
        raise ProcedureGridError("procedure returned invalid probabilities")

    exclusions = legacy._RESET_EXCLUSIONS[spec.key]
    selection_reset = legacy._state_sha256(
        selection.initial_model_state_, exclude=exclusions
    )
    metadata = {
        "cache_array_sha256": str(data["identity"]["array_sha256"]),
        "channels": list(channels),
        "split": copy.deepcopy(observed_split),
        "frozen_config": copy.deepcopy(config_identity),
        "view_pipeline": {
            "schema": VIEW_SCHEMA,
            "mirror_index": views["mirror_index"].tolist(),
            "raw": {
                "original": _array_manifest(views["raw"]),
                "reflected": _array_manifest(views["raw_reflected"]),
            },
            "four_band_spd": {
                "original": _array_manifest(views["covariance"]),
                "reflected": _array_manifest(views["covariance_reflected"]),
            },
            "selection_tangent": selection_tangent,
            "refit_tangent": refit_tangent,
        },
        "fit": {
            "seed_installed_before_construction": True,
            "selected_epoch_index": int(selection_detail["best_epoch_zero_based"]),
            "selected_epoch_count": int(selection_detail["selected_epoch_count"]),
            "selection_epochs_run": int(selection_detail["epochs_run"]),
            "selection_decision": copy.deepcopy(selection_detail["selection_decision"]),
            "selection_start_reset_sha256": selection_reset,
            "refit_start_reset_sha256": str(refit_detail["refit_start_reset_sha256"]),
            "reset_hash_exclusions": list(exclusions),
            "reset_verified": bool(refit_detail["reset_verified"]),
            "refit_epochs_run": int(refit_detail["epoch_count"]),
            "refit_state_sha256": str(refit_detail["model_state_sha256"]),
            "parameter_count": int(refit_detail["parameter_count"]),
            "trainable_parameter_count": int(refit_detail["trainable_parameter_count"]),
            "selection_preprocessing": copy.deepcopy(
                selection_detail["fitted_preprocessing"]
            ),
            "refit_preprocessing": copy.deepcopy(refit_detail["fitted_preprocessing"]),
        },
        "protocol": {
            "selection_preprocessing_rows": "train_only",
            "selection_optimization_rows": "train_only",
            "selection_decision_rows": "validation_only",
            "refit_preprocessing_rows": "train_plus_validation",
            "refit_optimization_rows": "train_plus_validation",
            "test_use": "single_predict_proba_call_only",
            "test_performance_computed": False,
            "transductive_calibration": False,
        },
        "timing_seconds": {
            "selection_fit": float(selection_seconds),
            "refit_fit": float(refit_seconds),
            "test_inference": float(inference_seconds),
            "job_total": float(time.perf_counter() - started),
        },
        "runtime": {
            "cpu_threads": int(cpu_threads),
            "torch_interop_threads": int(torch.get_num_interop_threads()),
            "device": str(target_device),
            "physical_gpu_uuid": physical_gpu_uuid,
            "cuda_visible_devices": (
                os.environ.get("CUDA_VISIBLE_DEVICES")
                if target_device.type == "cuda"
                else None
            ),
            "thread_environment": {
                name: os.environ.get(name) for name in THREAD_ENVIRONMENT_VARIABLES
            },
            "cuda_peak_memory_bytes": (
                int(torch.cuda.max_memory_allocated(target_device))
                if target_device.type == "cuda"
                else 0
            ),
        },
    }
    if (
        metadata["fit"]["selection_start_reset_sha256"]
        != metadata["fit"]["refit_start_reset_sha256"]
        or metadata["fit"]["reset_verified"] is not True
        or metadata["fit"]["refit_epochs_run"]
        != metadata["fit"]["selected_epoch_count"]
    ):
        raise ProcedureGridError("reset/refit invariant failed")
    del selection, refitted
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metadata, np.asarray(test, dtype=np.int64), probabilities


def _record_directory(run_root: Path, job: Job) -> Path:
    return (
        _safe_run_root(run_root)
        / "records"
        / job.stable_id
        / job.dataset
        / f"s{job.subject:03d}"
        / f"f{job.fold:02d}"
        / f"seed{job.seed}"
    )


def _claim_path(run_root: Path, job: Job) -> Path:
    return _safe_run_root(run_root) / "claims" / f"{job.job_id}.json"


def _validate_resource_guard(
    value: Any, *, plan: Mapping[str, Any], require_safe: bool = True
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
    if (
        (require_safe and value["safe"] is not True)
        or value["gpu"]["safe"] is not True
        or value["disk"]["safe"] is not True
        or not isinstance(value["reason"], str)
        or not value["reason"]
        or isinstance(value["disk"]["minimum_free_gib"], bool)
        or not isinstance(value["disk"]["minimum_free_gib"], (int, float))
        or not math.isfinite(float(value["disk"]["minimum_free_gib"]))
        or not math.isclose(
            float(value["disk"]["minimum_free_gib"]),
            float(plan["execution_config"]["minimum_free_gib"]),
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or not isinstance(value["gpu"]["foreign_processes"], list)
        or value["gpu"]["foreign_processes"]
        or not isinstance(value["disk"]["path"], str)
        or not value["disk"]["path"]
        or isinstance(value["disk"]["free_bytes"], bool)
        or not isinstance(value["disk"]["free_bytes"], int)
        or value["disk"]["free_bytes"] < 0
        or isinstance(value["disk"]["total_bytes"], bool)
        or not isinstance(value["disk"]["total_bytes"], int)
        or value["disk"]["total_bytes"] < value["disk"]["free_bytes"]
        or isinstance(value["disk"]["free_gib"], bool)
        or not isinstance(value["disk"]["free_gib"], (int, float))
        or not math.isfinite(float(value["disk"]["free_gib"]))
        or float(value["disk"]["free_gib"])
        < float(plan["execution_config"]["minimum_free_gib"])
        or value["disk"]["free_bytes"]
        < math.ceil(float(plan["execution_config"]["minimum_free_gib"]) * 1024**3)
        or not math.isclose(
            float(value["disk"]["free_gib"]),
            float(value["disk"]["free_bytes"]) / float(1024**3),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ProcedureGridError("resource guard is invalid")
    for process in value["gpu"]["foreign_processes"]:
        _exact_keys(
            process,
            {"pid", "process_name", "used_gpu_memory_mib"},
            path="resource_guard.gpu.foreign_processes[]",
        )
    formal = plan["publication_mode"] == FORMAL_PUBLICATION_MODE
    if formal:
        if (
            value["gpu"]["gpu_uuid"]
            not in plan["execution_config"]["physical_gpu_uuid_roster"]
            or value["gpu"]["gpu"] != value["gpu"]["gpu_uuid"]
            or not isinstance(value["gpu"]["pci_bus_id"], str)
            or not value["gpu"]["pci_bus_id"]
            or not isinstance(value["gpu"]["name"], str)
            or not value["gpu"]["name"]
            or any(
                isinstance(value["gpu"][name], bool)
                or not isinstance(value["gpu"][name], (int, float))
                or not math.isfinite(float(value["gpu"][name]))
                or float(value["gpu"][name]) < 0.0
                for name in (
                    "utilization_percent",
                    "memory_used_mib",
                    "own_compute_memory_mib",
                )
            )
        ):
            raise ProcedureGridError("resource guard GPU is not physically bound")
    elif value["gpu"]["gpu_uuid"] is not None:
        raise ProcedureGridError("test-only resource guard must not claim a GPU UUID")


def _validate_claim_value(
    value: Mapping[str, Any],
    *,
    run_root: Path,
    plan: Mapping[str, Any],
    job: Job,
    expected_nonce: str | None = None,
    expected_owner: Mapping[str, Any] | None = None,
) -> None:
    _exact_keys(
        value,
        {
            "schema",
            "created_at",
            "plan_sha256",
            "job_id",
            "job",
            "nonce",
            "owner",
            "resource_guard",
            "gpu_lease_receipt",
        },
        path="claim",
    )
    _exact_keys(
        value["owner"],
        {"hostname", "pid", "boot_id", "process_start_token"},
        path="claim.owner",
    )
    try:
        created = datetime.fromisoformat(str(value["created_at"]))
    except (TypeError, ValueError) as error:
        raise ProcedureGridError("claim timestamp is invalid") from error
    if (
        created.tzinfo is None
        or value["schema"] != CLAIM_SCHEMA
        or value["plan_sha256"] != plan["plan_sha256"]
        or value["job_id"] != job.job_id
        or value["job"] != job.identity()
        or not isinstance(value["nonce"], str)
        or not re.fullmatch(r"[0-9a-f]{32}", value["nonce"])
        or (expected_nonce is not None and value["nonce"] != expected_nonce)
        or (expected_owner is not None and value["owner"] != dict(expected_owner))
        or not _owner_identity_is_valid(value["owner"])
    ):
        raise ProcedureGridError("claim identity is invalid")
    _validate_resource_guard(value["resource_guard"], plan=plan)
    formal = plan["publication_mode"] == FORMAL_PUBLICATION_MODE
    if not formal:
        if value["gpu_lease_receipt"] is not None:
            raise ProcedureGridError("test-only claim must not bind a formal GPU lease")
        return
    try:
        project_gpu_leases.validate_gpu_lease_receipt(
            value["gpu_lease_receipt"],
            project_root=PROJECT_ROOT,
            run_root=_safe_run_root(run_root),
            plan_sha256=str(plan["plan_sha256"]),
            gpu_uuid=str(value["resource_guard"]["gpu"]["gpu_uuid"]),
            track_scope=GPU_LEASE_TRACK_SCOPE,
        )
    except project_gpu_leases.ProjectGPULeaseError as error:
        raise ProcedureGridError("claim GPU lease receipt is invalid") from error


def _publication_fence_path(run_root: Path) -> Path:
    return _safe_run_root(run_root) / PUBLICATION_FENCE_FILENAME


def _publication_fence_value(
    plan: Mapping[str, Any] | str,
) -> dict[str, str]:
    digest = str(plan["plan_sha256"] if isinstance(plan, Mapping) else plan)
    return {
        "schema": PUBLICATION_FENCE_SCHEMA,
        "plan_sha256": digest,
        "token": _sha256_bytes(
            b"eeg-mi-procedure-publication-fence-v1\0" + digest.encode("ascii")
        ),
    }


def _assert_publication_fence_descriptor(
    run_root: Path,
    plan: Mapping[str, Any] | str,
    descriptor: int,
) -> None:
    root = _safe_run_root(run_root)
    path = _publication_fence_path(root)
    descriptor_stat = os.fstat(descriptor)
    try:
        path_stat = _anchored_lstat(path)
    except FileNotFoundError as error:
        raise PublicationFenceLost("publication fence path disappeared") from error
    if (
        not stat.S_ISREG(descriptor_stat.st_mode)
        or descriptor_stat.st_nlink != 1
        or descriptor_stat.st_dev != path_stat.st_dev
        or descriptor_stat.st_ino != path_stat.st_ino
    ):
        raise PublicationFenceLost("publication fence inode ownership was lost")
    try:
        observed = _strict_json_from_descriptor(descriptor, source=str(path))
    except ProcedureGridError as error:
        raise PublicationFenceLost("publication fence token became invalid") from error
    if observed != _publication_fence_value(plan):
        raise PublicationFenceLost("publication fence token differs from the plan")


def _acquire_publication_fence(
    run_root: Path,
    plan: Mapping[str, Any],
    *,
    exclusive: bool,
    blocking: bool,
) -> int:
    root = _safe_run_root(run_root, create=True)
    path = root / PUBLICATION_FENCE_FILENAME
    expected = _publication_fence_value(plan)
    if not path.exists() and not path.is_symlink():
        staged = _stage_run_json(
            root,
            basename="publication-fence",
            value=expected,
        )
        try:
            _atomic_rename_noreplace(staged, path)
            _fsync_directory(root)
        except FileExistsError:
            # Another process installed the same authoritative fence.  The
            # complete losing receipt remains outside the publishable run.
            pass
    _require_unique_regular(path, read_only=True)
    descriptor = _open_unique_regular_readonly(path)
    try:
        mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        if not blocking:
            mode |= fcntl.LOCK_NB
        try:
            fcntl.flock(descriptor, mode)
        except BlockingIOError as error:
            raise PublicationActive("publication fence is active") from error
        _assert_publication_fence_descriptor(root, plan, descriptor)
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


@contextmanager
def publication_fence(
    run_root: Path,
    *,
    plan: Mapping[str, Any],
    exclusive: bool,
    blocking: bool = True,
) -> Iterator[int]:
    """Fence worker claims/commits against a quiescent publication transaction."""

    root = _safe_run_root(run_root, create=True)
    descriptor = _acquire_publication_fence(
        root,
        plan,
        exclusive=exclusive,
        blocking=blocking,
    )
    fence_error: Exception | None = None
    try:
        yield descriptor
    finally:
        try:
            _assert_publication_fence_descriptor(root, plan, descriptor)
        except Exception as error:
            fence_error = error
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
        if fence_error is not None:
            raise fence_error


def _process_start_token(pid: int) -> str | None:
    if not sys.platform.startswith("linux"):
        return _NONLINUX_PROCESS_START_TOKEN if int(pid) == os.getpid() else None
    path = Path(f"/proc/{int(pid)}/stat")
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            observed_stat = os.fstat(descriptor)
            if not stat.S_ISREG(observed_stat.st_mode) or observed_stat.st_nlink != 1:
                return None
            text = os.read(descriptor, 64 * 1024).decode("ascii")
        finally:
            os.close(descriptor)
        token = text.rsplit(")", 1)[1].split()[19]
        return token if token.isdigit() and int(token) > 0 else None
    except (OSError, UnicodeDecodeError, IndexError):
        return None


def _boot_id() -> str | None:
    if not sys.platform.startswith("linux"):
        return _NONLINUX_BOOT_ID
    try:
        descriptor = os.open(
            "/proc/sys/kernel/random/boot_id",
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            observed_stat = os.fstat(descriptor)
            if not stat.S_ISREG(observed_stat.st_mode) or observed_stat.st_nlink != 1:
                return None
            observed = os.read(descriptor, 128).decode("ascii").strip().lower()
        finally:
            os.close(descriptor)
        return observed if BOOT_ID_RE.fullmatch(observed) else None
    except (OSError, UnicodeDecodeError):
        return None


def _process_identity(pid: int | None = None) -> dict[str, Any]:
    selected = os.getpid() if pid is None else int(pid)
    value = {
        "hostname": socket.gethostname(),
        "pid": selected,
        "boot_id": _boot_id(),
        "process_start_token": _process_start_token(selected),
    }
    if not _owner_identity_is_valid(value):
        raise ProcedureGridError("cannot establish a durable process identity")
    return value


def _owner_identity_is_valid(value: Any) -> bool:
    return bool(
        isinstance(value, Mapping)
        and set(value) == {"hostname", "pid", "boot_id", "process_start_token"}
        and isinstance(value["hostname"], str)
        and value["hostname"]
        and type(value["pid"]) is int
        and value["pid"] > 0
        and isinstance(value["boot_id"], str)
        and BOOT_ID_RE.fullmatch(value["boot_id"])
        and isinstance(value["process_start_token"], str)
        and value["process_start_token"].isdigit()
        and int(value["process_start_token"]) > 0
    )


def _claim_is_live(
    value: Mapping[str, Any], *, foreign_timeout_seconds: float = 86400.0
) -> bool:
    owner = value.get("owner")
    if not _owner_identity_is_valid(owner):
        return False
    if owner.get("hostname") != socket.gethostname():
        # A local filesystem cannot prove that a process on another host died.
        # Never steal a foreign claim merely because wall-clock time elapsed.
        return True
    current_boot = _boot_id()
    if current_boot is None:
        return True
    if owner["boot_id"] != current_boot:
        return False
    try:
        pid = int(owner["pid"])
        os.kill(pid, 0)
    except (KeyError, TypeError, ValueError, ProcessLookupError):
        return False
    except PermissionError:
        return True
    token = owner["process_start_token"]
    observed = _process_start_token(pid)
    return observed is None or str(token) == str(observed)


def _quarantine(
    run_root: Path,
    path: Path,
    *,
    category: str,
    expected_descriptor: int | None = None,
) -> Path:
    root = _safe_run_root(run_root)
    allowed_categories = {
        "stale_claims",
        "power_cut_partials",
        "post_rename_stale_claims",
        "duplicate_partials",
        "failed_partials",
        "claim_lost_after_commit",
        "publication_fence_lost",
        "gpu_leases",
    }
    if category not in allowed_categories:
        raise ProcedureGridError(f"unknown forensic quarantine category {category}")
    observed = _anchored_lstat(path)
    if stat.S_ISLNK(observed.st_mode) or not (
        stat.S_ISREG(observed.st_mode) or stat.S_ISDIR(observed.st_mode)
    ):
        raise ProcedureGridError(f"refusing to quarantine special/aliased node {path}")
    forensic_root = _secure_mkdir_absolute(
        root.parent / f".{root.name}{FORENSIC_ROOT_SUFFIX}"
    )
    destination_root = _secure_mkdir_absolute(forensic_root / "quarantine" / category)
    destination = destination_root / f"{path.name}.{uuid.uuid4().hex}"
    expected_identity: tuple[int, int] | None = None
    directory_descriptor: int | None = None
    close_directory_descriptor = False
    directory_mode_mutated = False
    source_moved = False
    if expected_descriptor is not None:
        descriptor_stat = os.fstat(expected_descriptor)
        path_stat = _anchored_lstat(path)
        expected_identity = (descriptor_stat.st_dev, descriptor_stat.st_ino)
        if expected_identity != (path_stat.st_dev, path_stat.st_ino):
            raise ProcedureGridError("quarantine source inode ownership was lost")
    original_mode: int | None = None
    try:
        if stat.S_ISDIR(observed.st_mode):
            if expected_descriptor is None:
                directory_descriptor = _open_directory_absolute(path)
                close_directory_descriptor = True
            else:
                directory_descriptor = expected_descriptor
            bound = _assert_directory_descriptor_path(directory_descriptor, path)
            if expected_identity is None:
                expected_identity = (bound.st_dev, bound.st_ino)
            original_mode = stat.S_IMODE(bound.st_mode)
            os.fchmod(directory_descriptor, 0o700)
            directory_mode_mutated = True
            os.fsync(directory_descriptor)
            _assert_directory_descriptor_path(directory_descriptor, path)
            _fsync_directory(path.parent)
        _atomic_rename_noreplace(path, destination)
        source_moved = True
        if expected_identity is not None:
            destination_stat = _anchored_lstat(destination)
            descriptor_stat = (
                os.fstat(expected_descriptor)
                if expected_descriptor is not None
                else os.fstat(directory_descriptor)
            )
            if (
                destination_stat.st_dev,
                destination_stat.st_ino,
            ) != expected_identity or (
                descriptor_stat.st_dev,
                descriptor_stat.st_ino,
            ) != expected_identity:
                raise ProcedureGridError(
                    "quarantine destination does not own the locked inode"
                )
        if directory_descriptor is not None:
            _seal_directory_read_only(directory_descriptor, destination)
            directory_mode_mutated = False
        _fsync_directory(path.parent)
        _fsync_directory(destination.parent)
        return destination
    finally:
        if (
            directory_mode_mutated
            and directory_descriptor is not None
            and original_mode is not None
        ):
            os.fchmod(
                directory_descriptor,
                0o555 if source_moved else original_mode,
            )
            os.fsync(directory_descriptor)
        if close_directory_descriptor and directory_descriptor is not None:
            os.close(directory_descriptor)


def _quarantine_stale_claim_atomically(
    run_root: Path,
    path: Path,
    *,
    plan: Mapping[str, Any],
    job: Job,
    category: str,
    foreign_timeout_seconds: float = 86400.0,
) -> Path:
    """Move exactly one still-stale claim while holding its inode lock."""

    try:
        descriptor = _open_unique_regular_readonly(path)
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            descriptor_stat = os.fstat(handle.fileno())
            path_stat = _anchored_lstat(path)
            if (
                descriptor_stat.st_dev != path_stat.st_dev
                or descriptor_stat.st_ino != path_stat.st_ino
            ):
                raise ClaimUnavailable("claim path changed during recovery")
            value = _strict_json_from_descriptor(handle.fileno(), source=str(path))
            _validate_claim_value(value, run_root=run_root, plan=plan, job=job)
            if _claim_is_live(value, foreign_timeout_seconds=foreign_timeout_seconds):
                raise ClaimUnavailable("claim became live during recovery")
            path_stat = _anchored_lstat(path)
            if (
                descriptor_stat.st_dev != path_stat.st_dev
                or descriptor_stat.st_ino != path_stat.st_ino
            ):
                raise ClaimUnavailable("claim path changed before quarantine")
            return _quarantine(
                run_root,
                path,
                category=category,
                expected_descriptor=handle.fileno(),
            )
    except (BlockingIOError, FileNotFoundError) as error:
        raise ClaimUnavailable("another worker won stale-claim recovery") from error


def _acquire_claim_guarded(
    run_root: Path,
    plan: Mapping[str, Any],
    job: Job,
    *,
    resource_guard: Mapping[str, Any],
    gpu_lease_receipt: Mapping[str, Any] | None = None,
    recover_stale: bool,
    foreign_timeout_seconds: float = 86400.0,
) -> Claim:
    validate_plan_semantics(plan)
    _validate_resource_guard(_json_ready(resource_guard), plan=plan)
    root = _safe_run_root(run_root)
    _ensure_real_directory(root, "claims")
    path = _claim_path(root, job)
    owner = _process_identity()
    nonce = uuid.uuid4().hex
    value = {
        "schema": CLAIM_SCHEMA,
        "created_at": _utc_now(),
        "plan_sha256": plan["plan_sha256"],
        "job_id": job.job_id,
        "job": job.identity(),
        "nonce": nonce,
        "owner": owner,
        "resource_guard": copy.deepcopy(dict(resource_guard)),
        "gpu_lease_receipt": copy.deepcopy(gpu_lease_receipt),
    }
    _validate_claim_value(value, run_root=root, plan=plan, job=job)
    fence_fd = _acquire_publication_fence(
        root,
        plan,
        exclusive=False,
        blocking=False,
    )
    try:
        if plan["publication_mode"] == FORMAL_PUBLICATION_MODE:
            _require_hard_disk_floor(
                root,
                float(plan["execution_config"]["minimum_free_gib"]),
            )
        staged = _stage_run_json(
            root,
            basename=f"claim-{job.job_id}-{nonce}",
            value=value,
        )
        try:
            if plan["publication_mode"] == FORMAL_PUBLICATION_MODE:
                _require_hard_disk_floor(
                    root,
                    float(plan["execution_config"]["minimum_free_gib"]),
                )
            _atomic_rename_noreplace(staged, path)
            _fsync_directory(path.parent)
        except FileExistsError as error:
            existing = _strict_json_load(path)
            _validate_claim_value(existing, run_root=root, plan=plan, job=job)
            if _claim_is_live(
                existing, foreign_timeout_seconds=foreign_timeout_seconds
            ):
                raise ClaimUnavailable(job.job_id) from error
            if not recover_stale:
                raise ClaimUnavailable(
                    f"stale claim requires --recover-stale: {job.job_id}"
                ) from error
            _quarantine_stale_claim_atomically(
                root,
                path,
                plan=plan,
                job=job,
                category="stale_claims",
                foreign_timeout_seconds=foreign_timeout_seconds,
            )
            if plan["publication_mode"] == FORMAL_PUBLICATION_MODE:
                _require_hard_disk_floor(
                    root,
                    float(plan["execution_config"]["minimum_free_gib"]),
                )
            _atomic_rename_noreplace(staged, path)
            _fsync_directory(path.parent)
        observed_value = _strict_json_load(path)
        _validate_claim_value(
            observed_value,
            run_root=root,
            plan=plan,
            job=job,
            expected_nonce=nonce,
            expected_owner=owner,
        )
        observed_stat = _require_unique_regular(path, read_only=True)
        _assert_publication_fence_descriptor(root, plan, fence_fd)
        return Claim(
            job=job,
            path=path,
            nonce=nonce,
            owner=owner,
            resource_guard=copy.deepcopy(observed_value["resource_guard"]),
            st_dev=int(observed_stat.st_dev),
            st_ino=int(observed_stat.st_ino),
            value=copy.deepcopy(observed_value),
            fence_fd=fence_fd,
        )
    except Exception:
        try:
            fcntl.flock(fence_fd, fcntl.LOCK_UN)
        finally:
            os.close(fence_fd)
        raise


def acquire_claim(
    run_root: Path,
    plan: Mapping[str, Any],
    job: Job,
    *,
    resource_guard: Mapping[str, Any],
    gpu_lease: GPULease | None = None,
    recover_stale: bool,
    foreign_timeout_seconds: float = 86400.0,
) -> Claim:
    """Publish one claim while holding the exact formal GPU lease guard."""

    validate_plan_semantics(plan)
    formal = plan["publication_mode"] == FORMAL_PUBLICATION_MODE
    if not formal:
        if gpu_lease is not None:
            raise ProcedureGridError("test-only claim must not hold a formal GPU lease")
        return _acquire_claim_guarded(
            run_root,
            plan,
            job,
            resource_guard=resource_guard,
            gpu_lease_receipt=None,
            recover_stale=recover_stale,
            foreign_timeout_seconds=foreign_timeout_seconds,
        )
    if gpu_lease is None:
        raise ProcedureGridError("formal claim requires a guarded GPU lease")
    claim: Claim | None = None
    try:
        with guard_gpu_lease(gpu_lease) as lease_receipt:
            claim = _acquire_claim_guarded(
                run_root,
                plan,
                job,
                resource_guard=resource_guard,
                gpu_lease_receipt=lease_receipt,
                recover_stale=recover_stale,
                foreign_timeout_seconds=foreign_timeout_seconds,
            )
        return claim
    except Exception:
        if claim is not None:
            release_claim(claim)
        raise


def release_claim(claim: Claim) -> None:
    root = _safe_run_root(claim.path.parent.parent)
    try:
        _assert_publication_fence_descriptor(
            root,
            str(claim.value["plan_sha256"]),
            claim.fence_fd,
        )
        try:
            descriptor = _open_unique_regular_readonly(claim.path)
        except FileNotFoundError as error:
            raise ProcedureGridError(
                "owned claim disappeared before release"
            ) from error
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            descriptor_stat = os.fstat(descriptor)
            path_stat = _anchored_lstat(claim.path)
            observed = _strict_json_from_descriptor(descriptor, source=str(claim.path))
            if (
                descriptor_stat.st_dev != claim.st_dev
                or descriptor_stat.st_ino != claim.st_ino
                or path_stat.st_dev != claim.st_dev
                or path_stat.st_ino != claim.st_ino
                or observed != dict(claim.value)
                or observed.get("nonce") != claim.nonce
                or observed.get("owner") != dict(claim.owner)
            ):
                raise ProcedureGridError("refusing to release a claim owned elsewhere")
            archive_root = _secure_mkdir_absolute(
                root.parent / f".{root.name}{FORENSIC_ROOT_SUFFIX}" / "released_claims"
            )
            archived = archive_root / f"{claim.path.name}.{claim.nonce}"
            _atomic_rename_noreplace(claim.path, archived)
            archived_stat = _anchored_lstat(archived)
            archived_value = _strict_json_from_descriptor(
                descriptor, source=str(archived)
            )
            if (
                archived_stat.st_dev != claim.st_dev
                or archived_stat.st_ino != claim.st_ino
                or archived_value != dict(claim.value)
            ):
                raise ProcedureGridError(
                    "released claim archive does not own the locked inode"
                )
            _fsync_directory(claim.path.parent)
            _fsync_directory(archive_root)
        finally:
            os.close(descriptor)
        _assert_publication_fence_descriptor(
            root,
            str(claim.value["plan_sha256"]),
            claim.fence_fd,
        )
    finally:
        try:
            fcntl.flock(claim.fence_fd, fcntl.LOCK_UN)
        finally:
            os.close(claim.fence_fd)


def _recover_job_partials(run_root: Path, job: Job) -> list[str]:
    root = _safe_run_root(run_root)
    partial_root = root / "partials"
    recovered: list[str] = []
    if partial_root.exists():
        _require_real_directory(partial_root)
        for path in sorted(partial_root.glob(f"{job.job_id}.*.partial")):
            _require_real_directory(path)
            recovered.append(
                str(_quarantine(root, path, category="power_cut_partials"))
            )
    destination = _record_directory(root, job)
    if destination.parent.exists():
        _require_real_directory(destination.parent)
        for path in sorted(destination.parent.glob(f".{destination.name}.*.partial")):
            _require_real_directory(path)
            recovered.append(
                str(_quarantine(root, path, category="power_cut_partials"))
            )
    return recovered


def _recover_completed_claim(
    run_root: Path,
    plan: Mapping[str, Any],
    job: Job,
    *,
    recover_stale: bool,
) -> str | None:
    """Recover a claim left after an atomic record-directory rename."""

    path = _claim_path(run_root, job)
    if not path.exists():
        return None
    value = _strict_json_load(path)
    _validate_claim_value(value, run_root=run_root, plan=plan, job=job)
    if _claim_is_live(value):
        return "live_claim_left_for_owner"
    if not recover_stale:
        raise ClaimUnavailable(
            f"completed job has a stale claim; use --recover-stale: {job.job_id}"
        )
    try:
        recovered = _quarantine_stale_claim_atomically(
            run_root,
            path,
            plan=plan,
            job=job,
            category="post_rename_stale_claims",
        )
    except ClaimUnavailable:
        if not path.exists():
            return "stale_claim_recovered_by_another_worker"
        if _claim_is_live(_strict_json_load(path)):
            return "live_claim_left_for_owner"
        raise
    return str(recovered)


def _validate_metadata(
    plan: Mapping[str, Any], job: Job, metadata: Mapping[str, Any]
) -> None:
    _exact_keys(
        metadata,
        {
            "cache_array_sha256",
            "channels",
            "split",
            "frozen_config",
            "view_pipeline",
            "fit",
            "protocol",
            "timing_seconds",
            "runtime",
        },
        path="record.metadata",
    )
    aliases = _recursive_outcome_aliases(metadata, "record.metadata")
    if aliases:
        raise ProcedureGridError(f"metadata contains outcome aliases: {aliases[:5]}")
    expected_cache = _planned_cache(plan, job)
    if metadata.get("cache_array_sha256") != expected_cache.get("array_sha256"):
        raise ProcedureGridError("metadata cache identity differs from plan")
    if metadata.get("split") != _planned_split(plan, job):
        raise ProcedureGridError("metadata split differs from plan")
    config = metadata.get("frozen_config")
    expected = next(
        value
        for value in plan["config_inventory"]["available"]
        if value["stable_id"] == job.stable_id
    )
    runtime_overrides = (
        config.get("runtime_overrides") if isinstance(config, Mapping) else None
    )
    if (
        not isinstance(config, Mapping)
        or set(config)
        != {
            "schema",
            "version",
            "model",
            "config_file",
            "config_file_bytes",
            "config_file_sha256",
            "settings_sha256",
            "config",
            "runtime_overrides",
        }
        or config.get("schema") != "eeg-mi-local-procedure-config-v2"
        or config.get("version") != 1
        or config.get("model") != job.stable_id
        or config.get("config_file") != expected["config_path"]
        or config.get("config_file_bytes") != expected["config_bytes"]
        or config.get("config_file_sha256") != expected["config_sha256"]
        or config.get("settings_sha256") != expected["settings_sha256"]
        or not isinstance(config.get("config"), Mapping)
        or config.get("config") != expected.get("settings")
        or config.get("settings_sha256")
        != _sha256_bytes(_canonical_bytes(config.get("config")))
        or not isinstance(runtime_overrides, Mapping)
        or set(runtime_overrides) != {"seed", "device"}
        or runtime_overrides.get("seed") != job.seed
        or runtime_overrides.get("device") != plan["execution_config"]["device"]
    ):
        raise ProcedureGridError("metadata config identity differs from plan")
    fit = metadata.get("fit")
    _exact_keys(
        fit,
        {
            "seed_installed_before_construction",
            "selected_epoch_index",
            "selected_epoch_count",
            "selection_epochs_run",
            "selection_decision",
            "selection_start_reset_sha256",
            "refit_start_reset_sha256",
            "reset_hash_exclusions",
            "reset_verified",
            "refit_epochs_run",
            "refit_state_sha256",
            "parameter_count",
            "trainable_parameter_count",
            "selection_preprocessing",
            "refit_preprocessing",
        },
        path="record.metadata.fit",
    )
    hashes = (
        fit.get("selection_start_reset_sha256"),
        fit.get("refit_start_reset_sha256"),
        fit.get("refit_state_sha256"),
    )
    integer_fit_fields = (
        "selected_epoch_index",
        "selected_epoch_count",
        "selection_epochs_run",
        "refit_epochs_run",
        "parameter_count",
        "trainable_parameter_count",
    )
    if (
        any(
            isinstance(fit.get(name), bool) or not isinstance(fit.get(name), int)
            for name in integer_fit_fields
        )
        or any(
            not isinstance(value, str) or not HEX_64_RE.fullmatch(value)
            for value in hashes
        )
        or hashes[0] != hashes[1]
        or fit.get("seed_installed_before_construction") is not True
        or fit.get("reset_verified") is not True
        or int(fit.get("selected_epoch_count", -1))
        < (0 if job.stable_id == "architecture.parity_fuse" else 1)
        or int(fit.get("selected_epoch_index", -2))
        != int(fit.get("selected_epoch_count", -1)) - 1
        or int(fit.get("selection_epochs_run", -1))
        < int(fit.get("selected_epoch_count", -1))
        or int(fit.get("refit_epochs_run", -2))
        != int(fit.get("selected_epoch_count", -1))
        or int(fit.get("parameter_count", 0)) <= 0
        or int(fit.get("trainable_parameter_count", 0)) <= 0
        or int(fit.get("trainable_parameter_count", 0))
        > int(fit.get("parameter_count", 0))
    ):
        raise ProcedureGridError("metadata reset/refit contract is invalid")
    maximum_epochs = config["config"].get("epochs")
    if (
        isinstance(maximum_epochs, bool)
        or not isinstance(maximum_epochs, int)
        or maximum_epochs <= 0
        or int(fit["selected_epoch_count"]) > maximum_epochs
        or int(fit["selection_epochs_run"]) > maximum_epochs
        or not isinstance(fit.get("reset_hash_exclusions"), list)
        or fit["reset_hash_exclusions"]
        != list(RESET_EXCLUSIONS_BY_PROCEDURE[PROCEDURE_BY_ID[job.stable_id].key])
    ):
        raise ProcedureGridError("metadata epoch/config contract is invalid")
    decision = fit.get("selection_decision")
    if not isinstance(decision, Mapping):
        raise ProcedureGridError("metadata selection decision is absent")
    if job.stable_id == "architecture.cameo":
        rho = decision.get("selected_rho")
        if (
            set(decision) != {"selected_mixture", "selected_rho"}
            or decision.get("selected_mixture") not in config["config"]["mixture_names"]
            or isinstance(rho, bool)
            or not isinstance(rho, (int, float))
            or not math.isfinite(float(rho))
            or float(rho)
            not in {float(value) for value in config["config"]["rho_grid"]}
        ):
            raise ProcedureGridError("CAMEO selection decision is invalid")
    elif job.stable_id in {
        "architecture.hemiparity",
        "architecture.parity_fuse",
    }:
        if decision:
            raise ProcedureGridError(
                f"{job.stable_id} must not persist a route decision"
            )
    elif job.stable_id == "architecture.orbit_v3":
        if (
            set(decision) != {"selected_candidate"}
            or decision.get("selected_candidate")
            not in config["config"]["candidate_names"]
        ):
            raise ProcedureGridError("ORBIT-v3 selection decision is invalid")
    else:
        raise ProcedureGridError("unrecognized procedure selection contract")

    def require_manifest(
        value: Any,
        *,
        expected_shape: Sequence[int] | None = None,
        expected_dtype: str | None = None,
    ) -> None:
        if (
            not isinstance(value, Mapping)
            or set(value) != {"shape", "dtype", "sha256"}
            or not isinstance(value["shape"], list)
            or any(
                isinstance(raw, bool) or not isinstance(raw, int) or raw < 0
                for raw in value["shape"]
            )
            or not isinstance(value["dtype"], str)
            or not isinstance(value["sha256"], str)
            or not HEX_64_RE.fullmatch(value["sha256"])
            or (expected_shape is not None and value["shape"] != list(expected_shape))
            or (expected_dtype is not None and value["dtype"] != expected_dtype)
        ):
            raise ProcedureGridError("metadata array manifest is invalid")

    preprocessing_keys = {
        "raw_mean",
        "raw_std",
        "anchor_log_reference",
        "anchor_scaler_mean",
        "anchor_scaler_scale",
        "anchor_logistic_coef",
        "anchor_logistic_intercept",
    }
    for phase in ("selection_preprocessing", "refit_preprocessing"):
        preprocessing = fit.get(phase)
        if (
            not isinstance(preprocessing, Mapping)
            or set(preprocessing) != preprocessing_keys
        ):
            raise ProcedureGridError(f"metadata {phase} manifest is invalid")
        for manifest in preprocessing.values():
            require_manifest(manifest)
    expected_protocol = {
        "selection_preprocessing_rows": "train_only",
        "selection_optimization_rows": "train_only",
        "selection_decision_rows": "validation_only",
        "refit_preprocessing_rows": "train_plus_validation",
        "refit_optimization_rows": "train_plus_validation",
        "test_use": "single_predict_proba_call_only",
        "test_performance_computed": False,
        "transductive_calibration": False,
    }
    if metadata.get("protocol") != expected_protocol:
        raise ProcedureGridError("metadata source/test protocol is invalid")
    view = metadata.get("view_pipeline")
    channels = metadata.get("channels")
    if (
        not isinstance(channels, list)
        or not channels
        or any(not isinstance(value, str) or not value for value in channels)
        or len(set(channels)) != len(channels)
    ):
        raise ProcedureGridError("metadata channel contract is invalid")
    n_channels = len(channels)
    trial_count = int(expected_cache["trial_count"])
    n_times = int(plan["datasets"][job.dataset]["preprocessing"]["n_times"])
    if (
        not isinstance(view, Mapping)
        or set(view)
        != {
            "schema",
            "mirror_index",
            "raw",
            "four_band_spd",
            "selection_tangent",
            "refit_tangent",
        }
        or view.get("schema") != VIEW_SCHEMA
        or view.get("mirror_index") != _left_right_swap_index(channels).tolist()
    ):
        raise ProcedureGridError("metadata paired-view contract is absent")
    for name in ("raw", "four_band_spd"):
        pair = view[name]
        if not isinstance(pair, Mapping) or set(pair) != {
            "original",
            "reflected",
        }:
            raise ProcedureGridError(f"metadata {name} pair is invalid")
    require_manifest(
        view["raw"]["original"],
        expected_shape=(trial_count, n_channels, n_times),
        expected_dtype=np.dtype(np.float32).str,
    )
    require_manifest(
        view["raw"]["reflected"],
        expected_shape=(trial_count, n_channels, n_times),
        expected_dtype=np.dtype(np.float32).str,
    )
    require_manifest(
        view["four_band_spd"]["original"],
        expected_shape=(trial_count, 4, n_channels, n_channels),
        expected_dtype=np.dtype(np.float32).str,
    )
    require_manifest(
        view["four_band_spd"]["reflected"],
        expected_shape=(trial_count, 4, n_channels, n_channels),
        expected_dtype=np.dtype(np.float32).str,
    )
    tangent_features = 4 * n_channels * (n_channels + 1) // 2
    partition_counts = _planned_split(plan, job)["partitions"]
    tangent_contracts = (
        ("selection_tangent", "train", partition_counts["train"]["count"]),
        (
            "selection_tangent",
            "validation",
            partition_counts["validation"]["count"],
        ),
        ("refit_tangent", "source", partition_counts["source"]["count"]),
        ("refit_tangent", "test", partition_counts["test"]["count"]),
    )
    if set(view["selection_tangent"]) != {"train", "validation"} or set(
        view["refit_tangent"]
    ) != {"source", "test"}:
        raise ProcedureGridError("metadata tangent partition roster is invalid")
    for phase, partition, count in tangent_contracts:
        pair = view[phase][partition]
        if not isinstance(pair, Mapping) or set(pair) != {
            "original",
            "reflected",
        }:
            raise ProcedureGridError("metadata tangent pair is invalid")
        require_manifest(
            pair["original"],
            expected_shape=(int(count), tangent_features),
            expected_dtype=np.dtype(np.float32).str,
        )
        require_manifest(
            pair["reflected"],
            expected_shape=(int(count), tangent_features),
            expected_dtype=np.dtype(np.float32).str,
        )
    timing = metadata.get("timing_seconds")
    if not isinstance(timing, Mapping) or set(timing) != {
        "selection_fit",
        "refit_fit",
        "test_inference",
        "job_total",
    }:
        raise ProcedureGridError("metadata timing contract is invalid")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
        for value in timing.values()
    ):
        raise ProcedureGridError("metadata timing contains invalid values")
    runtime = metadata.get("runtime")
    expected_threads = int(plan["execution_config"]["cpu_threads_per_worker"])
    if (
        not isinstance(runtime, Mapping)
        or set(runtime)
        != {
            "cpu_threads",
            "torch_interop_threads",
            "device",
            "physical_gpu_uuid",
            "cuda_visible_devices",
            "thread_environment",
            "cuda_peak_memory_bytes",
        }
        or runtime.get("cpu_threads") != expected_threads
        or runtime.get("torch_interop_threads") != 1
        or runtime.get("device") != plan["execution_config"]["device"]
        or runtime.get("thread_environment")
        != {name: str(expected_threads) for name in THREAD_ENVIRONMENT_VARIABLES}
        or isinstance(runtime.get("cuda_peak_memory_bytes"), bool)
        or not isinstance(runtime.get("cuda_peak_memory_bytes"), int)
        or runtime["cuda_peak_memory_bytes"] < 0
    ):
        raise ProcedureGridError("metadata worker runtime is invalid")
    if plan["publication_mode"] == FORMAL_PUBLICATION_MODE:
        if (
            not isinstance(runtime.get("physical_gpu_uuid"), str)
            or runtime["physical_gpu_uuid"]
            not in plan["execution_config"]["physical_gpu_uuid_roster"]
            or runtime.get("cuda_visible_devices") != runtime["physical_gpu_uuid"]
        ):
            raise ProcedureGridError("metadata CUDA physical-device binding is invalid")
    elif (
        runtime.get("physical_gpu_uuid") is not None
        or runtime.get("cuda_visible_devices") is not None
    ):
        raise ProcedureGridError("test-only CPU metadata claims a CUDA binding")


@contextmanager
def _owned_claim_lock(claim: Claim, *, plan: Mapping[str, Any]) -> Iterator[int]:
    try:
        descriptor = _open_unique_regular_readonly(claim.path)
    except FileNotFoundError as error:
        raise ProcedureGridError("owned claim is absent") from error
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        descriptor_stat = os.fstat(descriptor)
        path_stat = _anchored_lstat(claim.path)
        observed = _strict_json_from_descriptor(descriptor, source=str(claim.path))
        _validate_claim_value(
            observed,
            run_root=claim.path.parent.parent,
            plan=plan,
            job=claim.job,
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
            raise ProcedureGridError("claim token/inode ownership was lost")
        yield descriptor
    finally:
        os.close(descriptor)


def _recheck_owned_claim_path(claim: Claim, descriptor: int) -> None:
    descriptor_stat = os.fstat(descriptor)
    try:
        path_stat = _anchored_lstat(claim.path)
    except FileNotFoundError as error:
        raise ProcedureGridError("owned claim disappeared before commit") from error
    if (
        descriptor_stat.st_dev != claim.st_dev
        or descriptor_stat.st_ino != claim.st_ino
        or path_stat.st_dev != claim.st_dev
        or path_stat.st_ino != claim.st_ino
    ):
        raise ProcedureGridError("owned claim path changed before commit")


def _claim_receipt(claim: Claim) -> dict[str, Any]:
    value = copy.deepcopy(dict(claim.value))
    return {
        "schema": CLAIM_RECEIPT_SCHEMA,
        "claim": value,
        "claim_sha256": _sha256_bytes(_canonical_bytes(value)),
        "claim_st_dev": int(claim.st_dev),
        "claim_st_ino": int(claim.st_ino),
    }


def _validate_claim_receipt(
    receipt: Any,
    *,
    run_root: Path,
    plan: Mapping[str, Any],
    job: Job,
) -> None:
    _exact_keys(
        receipt,
        {
            "schema",
            "claim",
            "claim_sha256",
            "claim_st_dev",
            "claim_st_ino",
        },
        path="record.claim_receipt",
    )
    if receipt["schema"] != CLAIM_RECEIPT_SCHEMA:
        raise ProcedureGridError("record claim receipt schema is invalid")
    _validate_claim_value(receipt["claim"], run_root=run_root, plan=plan, job=job)
    if (
        receipt["claim_sha256"] != _sha256_bytes(_canonical_bytes(receipt["claim"]))
        or type(receipt["claim_st_dev"]) is not int
        or receipt["claim_st_dev"] < 0
        or type(receipt["claim_st_ino"]) is not int
        or receipt["claim_st_ino"] <= 0
    ):
        raise ProcedureGridError("record claim receipt is invalid")


def _validate_record_object(
    record: Mapping[str, Any],
    *,
    run_root: Path,
    plan: Mapping[str, Any],
    job: Job,
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
            "test_count",
            "test_rows_sha256",
            "claim_receipt",
            "claim_resource_guard",
            "commit_resource_guard",
            "commit_gpu_lease_receipt",
            "metadata",
        },
        path="record",
    )
    aliases = _recursive_outcome_aliases(record)
    if aliases:
        raise ProcedureGridError(
            f"score-blind record has outcome aliases: {aliases[:5]}"
        )
    try:
        created = datetime.fromisoformat(str(record["created_at"]))
    except (TypeError, ValueError) as error:
        raise ProcedureGridError("record timestamp is invalid") from error
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
        raise ProcedureGridError("score-blind record identity is invalid")
    _validate_claim_receipt(
        record["claim_receipt"],
        run_root=run_root,
        plan=plan,
        job=job,
    )
    _validate_resource_guard(record["claim_resource_guard"], plan=plan)
    _validate_resource_guard(record["commit_resource_guard"], plan=plan)
    if (
        record["claim_resource_guard"]
        != record["claim_receipt"]["claim"]["resource_guard"]
    ):
        raise ProcedureGridError("record claim resource differs from its receipt")
    if (
        record["commit_gpu_lease_receipt"]
        != record["claim_receipt"]["claim"]["gpu_lease_receipt"]
    ):
        raise ProcedureGridError(
            "record commit GPU lease differs from the claimed lease"
        )
    _validate_metadata(plan, job, record["metadata"])
    if plan["publication_mode"] == FORMAL_PUBLICATION_MODE and (
        record["claim_resource_guard"]["gpu"]["gpu_uuid"]
        != record["commit_resource_guard"]["gpu"]["gpu_uuid"]
        or record["metadata"]["runtime"]["physical_gpu_uuid"]
        != record["commit_resource_guard"]["gpu"]["gpu_uuid"]
    ):
        raise ProcedureGridError("record GPU binding changed between claim and commit")


def _validate_completion_object(
    completion: Mapping[str, Any], *, plan: Mapping[str, Any], job: Job
) -> None:
    _exact_keys(
        completion,
        {
            "schema",
            "created_at",
            "plan_sha256",
            "job_id",
            "job",
            "claim_receipt_sha256",
            "files",
        },
        path="completion",
    )
    _exact_keys(
        completion["files"],
        {"record.json", "predictions.npz"},
        path="completion.files",
    )
    aliases = _recursive_outcome_aliases(completion, "completion")
    if aliases:
        raise ProcedureGridError(f"completion has outcome aliases: {aliases[:5]}")
    try:
        created = datetime.fromisoformat(str(completion["created_at"]))
    except (TypeError, ValueError) as error:
        raise ProcedureGridError("completion timestamp is invalid") from error
    if (
        created.tzinfo is None
        or completion["schema"] != COMPLETION_SCHEMA
        or completion["plan_sha256"] != plan["plan_sha256"]
        or completion["job_id"] != job.job_id
        or completion["job"] != job.identity()
        or not isinstance(completion["claim_receipt_sha256"], str)
        or not HEX_64_RE.fullmatch(completion["claim_receipt_sha256"])
        or any(
            not isinstance(value, str) or not HEX_64_RE.fullmatch(value)
            for value in completion["files"].values()
        )
    ):
        raise ProcedureGridError("completion receipt identity is invalid")


def commit_job_output(
    run_root: Path,
    plan: Mapping[str, Any],
    claim: Claim,
    *,
    metadata: Mapping[str, Any],
    test_rows: np.ndarray,
    probabilities: np.ndarray,
    resource_recheck: Mapping[str, Any] | None = None,
    gpu_lease: GPULease | None = None,
) -> Path:
    validate_plan_semantics(plan)
    job = claim.job
    rows = np.asarray(test_rows)
    values = np.asarray(probabilities)
    expected_test = _planned_split(plan, job)["partitions"]["test"]
    if (
        rows.dtype != np.int64
        or rows.ndim != 1
        or len(rows) != int(expected_test["count"])
        or len(np.unique(rows)) != len(rows)
        or _rows_sha256(rows) != expected_test["rows_sha256"]
        or values.dtype != np.float64
        or values.shape != (len(rows), 2)
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(values > 1.0)
        or not np.allclose(values.sum(axis=1), 1.0, atol=1e-6)
    ):
        raise ProcedureGridError("executor returned invalid publication arrays")
    _validate_metadata(plan, job, metadata)
    if resource_recheck is None:
        if plan["publication_mode"] == FORMAL_PUBLICATION_MODE:
            raise ProcedureGridError("formal commit requires a fresh resource recheck")
        resource_recheck = claim.resource_guard
    claim_resource = _json_ready(claim.resource_guard)
    commit_resource = _json_ready(resource_recheck)
    _validate_resource_guard(claim_resource, plan=plan)
    _validate_resource_guard(commit_resource, plan=plan)
    bound_lease_receipt = claim.value.get("gpu_lease_receipt")
    if plan["publication_mode"] == FORMAL_PUBLICATION_MODE:
        if gpu_lease is None:
            raise ProcedureGridError("formal commit requires a guarded GPU lease")
        commit_lease_receipt: Mapping[str, Any] | None = gpu_lease_receipt(gpu_lease)
        if commit_lease_receipt != bound_lease_receipt:
            raise ProcedureGridError(
                "formal commit must reassert the exact claimed GPU lease"
            )
    else:
        if gpu_lease is not None or bound_lease_receipt is not None:
            raise ProcedureGridError(
                "test-only commit must not bind a formal GPU lease"
            )
        commit_lease_receipt = None
    claim_receipt = _claim_receipt(claim)
    record = {
        "schema": RECORD_SCHEMA,
        "created_at": _utc_now(),
        "plan_sha256": plan["plan_sha256"],
        "job_id": job.job_id,
        "job": job.identity(),
        "score_blind": True,
        "test_count": len(rows),
        "test_rows_sha256": _rows_sha256(rows),
        "claim_receipt": claim_receipt,
        "claim_resource_guard": copy.deepcopy(claim_resource),
        "commit_resource_guard": copy.deepcopy(commit_resource),
        "commit_gpu_lease_receipt": copy.deepcopy(commit_lease_receipt),
        "metadata": copy.deepcopy(dict(metadata)),
    }
    _validate_record_object(record, run_root=run_root, plan=plan, job=job)
    root = _safe_run_root(run_root)
    formal = plan["publication_mode"] == FORMAL_PUBLICATION_MODE
    minimum_free_gib = float(plan["execution_config"]["minimum_free_gib"])
    _assert_publication_fence_descriptor(root, plan, claim.fence_fd)
    partial_root = _ensure_real_directory(root, "partials")
    partial = partial_root / f"{job.job_id}.{claim.nonce}.partial"
    with _owned_claim_lock(claim, plan=plan) as claim_descriptor:
        if formal:
            _require_hard_disk_floor(root, minimum_free_gib)
        partial_parent = _open_directory_absolute(partial_root)
        try:
            os.mkdir(partial.name, 0o700, dir_fd=partial_parent)
            os.fsync(partial_parent)
        finally:
            os.close(partial_parent)
        _require_real_directory(partial)
        partial_descriptor = _open_directory_absolute(partial)
        destination = _record_directory(root, job)
        publication_stage = (
            destination.parent / f".{destination.name}.{claim.nonce}.partial"
        )
        staged_path = partial
        destination_published = False
        try:
            record_path = partial / "record.json"
            prediction_path = partial / "predictions.npz"
            completion_path = partial / "completion.json"
            if formal:
                _require_hard_disk_floor(root, minimum_free_gib)
            record_payload = _canonical_bytes(record) + b"\n"
            record_digest = _write_bytes_exclusive_at(
                partial_descriptor,
                record_path.name,
                record_payload,
            )
            if formal:
                _require_hard_disk_floor(root, minimum_free_gib)
            descriptor = os.open(
                prediction_path.name,
                os.O_RDWR
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0)
                | getattr(os, "O_CLOEXEC", 0),
                0o600,
                dir_fd=partial_descriptor,
            )
            try:
                with os.fdopen(descriptor, "wb", closefd=False) as handle:
                    np.savez_compressed(
                        handle,
                        rows=np.ascontiguousarray(rows),
                        probabilities=np.ascontiguousarray(values),
                    )
                    handle.flush()
                    os.fchmod(descriptor, 0o444)
                    os.fsync(handle.fileno())
                prediction_payload = _read_stable_descriptor_bytes(
                    descriptor,
                    source=str(prediction_path),
                )
            finally:
                os.close(descriptor)
            os.fsync(partial_descriptor)
            prediction_digest = _sha256_bytes(prediction_payload)
            completion = {
                "schema": COMPLETION_SCHEMA,
                "created_at": _utc_now(),
                "plan_sha256": plan["plan_sha256"],
                "job_id": job.job_id,
                "job": job.identity(),
                "claim_receipt_sha256": _sha256_bytes(_canonical_bytes(claim_receipt)),
                "files": {
                    "record.json": record_digest,
                    "predictions.npz": prediction_digest,
                },
            }
            _validate_completion_object(completion, plan=plan, job=job)
            if formal:
                _require_hard_disk_floor(root, minimum_free_gib)
            _write_bytes_exclusive_at(
                partial_descriptor,
                completion_path.name,
                _canonical_bytes(completion) + b"\n",
            )
            if set(os.listdir(partial_descriptor)) != FINAL_FILENAMES:
                raise ProcedureGridError(
                    "partial publication contains unexpected files"
                )
            for filename in FINAL_FILENAMES:
                observed_file = os.stat(
                    filename,
                    dir_fd=partial_descriptor,
                    follow_symlinks=False,
                )
                if (
                    not stat.S_ISREG(observed_file.st_mode)
                    or observed_file.st_nlink != 1
                    or observed_file.st_mode & 0o222
                ):
                    raise ProcedureGridError(
                        "partial publication contains a mutable/aliased file"
                    )
            os.fsync(partial_descriptor)
            relative_parent = destination.parent.relative_to(root)
            _ensure_real_directory(root, relative_parent)
            _assert_directory_descriptor_path(partial_descriptor, partial)
            _atomic_rename_noreplace(partial, publication_stage)
            staged_path = publication_stage
            _assert_directory_descriptor_path(partial_descriptor, publication_stage)
            _fsync_directory(partial_root)
            _fsync_directory(destination.parent)
            _seal_directory_read_only(partial_descriptor, publication_stage)
            lease_guard = (
                guard_gpu_lease(gpu_lease)
                if gpu_lease is not None
                else nullcontext(None)
            )
            with lease_guard as guarded_lease_receipt:
                if (
                    plan["publication_mode"] == FORMAL_PUBLICATION_MODE
                    and guarded_lease_receipt != commit_lease_receipt
                ):
                    raise ProcedureGridError(
                        "guarded commit GPU lease differs from the claim"
                    )
                _recheck_owned_claim_path(claim, claim_descriptor)
                _assert_publication_fence_descriptor(root, plan, claim.fence_fd)
                _assert_directory_descriptor_path(partial_descriptor, publication_stage)
                if formal:
                    _require_hard_disk_floor(root, minimum_free_gib)
                try:
                    _atomic_rename_noreplace(publication_stage, destination)
                except FileExistsError:
                    existing = validate_completion(root, plan, job)
                    if (
                        existing["record"]["claim_receipt"]["claim_sha256"]
                        != claim_receipt["claim_sha256"]
                    ):
                        raise ProcedureGridError(
                            "completed destination belongs to a different claim"
                        )
                    _quarantine(
                        root,
                        publication_stage,
                        category="duplicate_partials",
                        expected_descriptor=partial_descriptor,
                    )
                    return destination
                destination_published = True
                _assert_directory_descriptor_path(partial_descriptor, destination)
                _fsync_directory(destination.parent)
                try:
                    _recheck_owned_claim_path(claim, claim_descriptor)
                    observed_claim = _strict_json_from_descriptor(
                        claim_descriptor,
                        source=str(claim.path),
                    )
                    if observed_claim != dict(claim.value):
                        raise ProcedureGridError(
                            "claim token changed after destination publication"
                        )
                    _assert_publication_fence_descriptor(root, plan, claim.fence_fd)
                except Exception as error:
                    category = (
                        "publication_fence_lost"
                        if isinstance(error, PublicationFenceLost)
                        else "claim_lost_after_commit"
                    )
                    _quarantine(
                        root,
                        destination,
                        category=category,
                        expected_descriptor=partial_descriptor,
                    )
                    _fsync_directory(destination.parent)
                    destination_published = False
                    raise
                validate_completion(root, plan, job)
            return destination
        except Exception:
            if staged_path.exists():
                _quarantine(
                    root,
                    staged_path,
                    category="failed_partials",
                    expected_descriptor=partial_descriptor,
                )
            elif destination_published and destination.exists():
                _quarantine(
                    root,
                    destination,
                    category="claim_lost_after_commit",
                    expected_descriptor=partial_descriptor,
                )
            raise
        finally:
            os.close(partial_descriptor)


def validate_completion(
    run_root: Path, plan: Mapping[str, Any], job: Job
) -> dict[str, Any]:
    validate_plan_semantics(plan)
    root = _safe_run_root(run_root)
    directory = _record_directory(root, job)
    try:
        _require_real_path_components(root, directory)
    except FileNotFoundError:
        raise FileNotFoundError(directory)
    directory_descriptor = _open_directory_absolute(directory)
    try:
        directory_before = _assert_directory_descriptor_path(
            directory_descriptor, directory
        )
        path_before = _anchored_lstat(directory)
        expected_directory_identity = (
            directory_before.st_dev,
            directory_before.st_ino,
            directory_before.st_mode,
        )
        if (
            not stat.S_ISDIR(directory_before.st_mode)
            or stat.S_IMODE(directory_before.st_mode) != 0o555
            or (
                path_before.st_dev,
                path_before.st_ino,
                path_before.st_mode,
            )
            != expected_directory_identity
        ):
            raise ProcedureGridError(
                f"{directory} is not an immutable 0555 completion directory"
            )
        if set(os.listdir(directory_descriptor)) != FINAL_FILENAMES:
            raise ProcedureGridError(f"{directory} has unexpected files")
        completion_bytes = _read_unique_regular_at(
            directory_descriptor,
            "completion.json",
            source=str(directory / "completion.json"),
            read_only=True,
        )
        completion = _strict_json_bytes(
            completion_bytes,
            source=str(directory / "completion.json"),
        )
        _validate_completion_object(completion, plan=plan, job=job)
        record_bytes = _read_unique_regular_at(
            directory_descriptor,
            "record.json",
            source=str(directory / "record.json"),
            read_only=True,
        )
        prediction_bytes = _read_unique_regular_at(
            directory_descriptor,
            "predictions.npz",
            source=str(directory / "predictions.npz"),
            read_only=True,
        )
        if (
            completion.get("files")
            != {
                "record.json": _sha256_bytes(record_bytes),
                "predictions.npz": _sha256_bytes(prediction_bytes),
            }
            or completion_bytes != _canonical_bytes(completion) + b"\n"
        ):
            raise ProcedureGridError("completion receipt is invalid")
        record = _strict_json_bytes(
            record_bytes,
            source=str(directory / "record.json"),
        )
        _validate_record_object(record, run_root=root, plan=plan, job=job)
        if record_bytes != _canonical_bytes(record) + b"\n" or completion[
            "claim_receipt_sha256"
        ] != _sha256_bytes(_canonical_bytes(record["claim_receipt"])):
            raise ProcedureGridError("score-blind record is noncanonical")
        try:
            with zipfile.ZipFile(io.BytesIO(prediction_bytes), mode="r") as raw_archive:
                raw_members = tuple(entry.filename for entry in raw_archive.infolist())
                if raw_members != PREDICTION_ARCHIVE_MEMBERS:
                    raise ProcedureGridError(
                        "prediction archive has an invalid exact ZIP member order"
                    )
            with np.load(io.BytesIO(prediction_bytes), allow_pickle=False) as archive:
                if tuple(archive.files) != PREDICTION_ARCHIVE_ARRAYS:
                    raise ProcedureGridError(
                        "prediction archive has an invalid exact array order"
                    )
                rows = archive["rows"].copy()
                probabilities = archive["probabilities"].copy()
        except ProcedureGridError:
            raise
        except (
            EOFError,
            KeyError,
            OSError,
            TypeError,
            ValueError,
            zipfile.BadZipFile,
        ) as error:
            raise ProcedureGridError("prediction archive cannot be loaded") from error
        directory_after = os.fstat(directory_descriptor)
        path_after = _anchored_lstat(directory)
        if (
            directory_after.st_dev,
            directory_after.st_ino,
            directory_after.st_mode,
        ) != expected_directory_identity or (
            path_after.st_dev,
            path_after.st_ino,
            path_after.st_mode,
        ) != expected_directory_identity:
            raise ProcedureGridError(
                f"{directory} inode or immutable seal changed during validation"
            )
    finally:
        os.close(directory_descriptor)
    expected_test = _planned_split(plan, job)["partitions"]["test"]
    if (
        rows.dtype != np.int64
        or rows.ndim != 1
        or len(rows) != int(record["test_count"])
        or len(rows) != int(expected_test["count"])
        or _rows_sha256(rows) != expected_test["rows_sha256"]
        or _rows_sha256(rows) != record["test_rows_sha256"]
        or probabilities.dtype != np.float64
        or probabilities.shape != (len(rows), 2)
        or not np.all(np.isfinite(probabilities))
        or np.any(probabilities < 0.0)
        or np.any(probabilities > 1.0)
        or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    ):
        raise ProcedureGridError("prediction archive is invalid")
    return {
        "record": record,
        "completion": completion,
        "rows": rows,
        "probabilities": probabilities,
        "file_sha256": {
            "record.json": _sha256_bytes(record_bytes),
            "predictions.npz": _sha256_bytes(prediction_bytes),
            "completion.json": _sha256_bytes(completion_bytes),
        },
    }


def _record_failure(
    run_root: Path, plan: Mapping[str, Any], claim: Claim, error: Exception
) -> Path:
    run = _safe_run_root(run_root)
    forensic_root = _secure_mkdir_absolute(
        run.parent / f".{run.name}{FORENSIC_ROOT_SUFFIX}"
    )
    root = _secure_mkdir_absolute(forensic_root / "failures" / claim.job.job_id)
    attempts = len(tuple(root.glob("attempt-*.json"))) + 1
    path = root / f"attempt-{attempts:04d}.json"
    _write_json_exclusive(
        path,
        {
            "schema": FAILURE_SCHEMA,
            "created_at": _utc_now(),
            "plan_sha256": plan["plan_sha256"],
            "job_id": claim.job.job_id,
            "job": claim.job.identity(),
            "owner": dict(claim.owner),
            "error_type": type(error).__name__,
            "error": str(error),
        },
    )
    return path


def acquire_gpu_lease(
    run_root: Path,
    plan: Mapping[str, Any],
    gpu_uuid: str,
) -> GPULease:
    validate_plan_semantics(plan)
    if (
        plan["publication_mode"] != FORMAL_PUBLICATION_MODE
        or gpu_uuid not in plan["execution_config"]["physical_gpu_uuid_roster"]
        or not GPU_UUID_RE.fullmatch(gpu_uuid)
    ):
        raise ProcedureGridError("formal GPU lease request is outside the plan")
    try:
        return project_gpu_leases.acquire_gpu_lease(
            project_root=PROJECT_ROOT,
            run_root=_safe_run_root(run_root),
            plan_sha256=str(plan["plan_sha256"]),
            gpu_uuid=gpu_uuid,
            track_scope=GPU_LEASE_TRACK_SCOPE,
        )
    except project_gpu_leases.GPUWorkerUnavailable as error:
        raise GPUWorkerUnavailable(str(error)) from error
    except project_gpu_leases.ProjectGPULeaseError as error:
        raise ProcedureGridError(str(error)) from error


def release_gpu_lease(lease: GPULease) -> None:
    try:
        project_gpu_leases.release_gpu_lease(lease)
    except project_gpu_leases.ProjectGPULeaseError as error:
        raise ProcedureGridError(str(error)) from error


def assert_gpu_lease(lease: GPULease) -> None:
    try:
        project_gpu_leases.assert_gpu_lease(lease)
    except project_gpu_leases.ProjectGPULeaseError as error:
        raise ProcedureGridError(str(error)) from error


def gpu_lease_receipt(lease: GPULease) -> dict[str, Any]:
    try:
        return project_gpu_leases.gpu_lease_receipt(lease)
    except project_gpu_leases.ProjectGPULeaseError as error:
        raise ProcedureGridError(str(error)) from error


@contextmanager
def guard_gpu_lease(lease: GPULease) -> Iterator[dict[str, Any]]:
    try:
        with project_gpu_leases.guard_gpu_lease(lease) as receipt:
            yield receipt
    except project_gpu_leases.ProjectGPULeaseError as error:
        raise ProcedureGridError(str(error)) from error


def _strict_nvidia_csv_rows(stdout: str, *, width: int) -> list[list[str]]:
    try:
        parsed = list(csv.reader(io.StringIO(stdout), strict=True))
    except csv.Error as error:
        raise ProcedureGridError("nvidia-smi returned malformed CSV") from error
    rows: list[list[str]] = []
    for row in parsed:
        if not row or all(not field.strip() for field in row):
            continue
        fields = [field.strip() for field in row]
        if len(fields) != width or any(not field for field in fields):
            raise ProcedureGridError("nvidia-smi returned an invalid row")
        rows.append(fields)
    return rows


def _strict_gpu_status(
    *,
    gpu: str,
    allow_active_owner: bool,
) -> dict[str, Any]:
    """Probe one exact physical UUID and fail closed on any malformed row."""

    requested = project_gpu_leases.canonical_gpu_uuid(gpu)
    try:
        identity_result = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                requested,
                "--query-gpu=uuid,pci.bus_id,name,utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        identity_rows = _strict_nvidia_csv_rows(identity_result.stdout, width=5)
        if len(identity_rows) != 1:
            raise ProcedureGridError(
                "nvidia-smi did not resolve exactly one physical GPU"
            )
        (
            observed_uuid,
            pci_bus_id,
            gpu_name,
            utilization_text,
            memory_text,
        ) = identity_rows[0]
        if (
            project_gpu_leases.canonical_gpu_uuid(observed_uuid) != requested
            or not re.fullmatch(
                r"[0-9A-Fa-f]{4,8}:[0-9A-Fa-f]{2}:"
                r"[0-9A-Fa-f]{2}\.[0-7]",
                pci_bus_id,
            )
            or any(ord(character) < 32 for character in gpu_name)
        ):
            raise ProcedureGridError("nvidia-smi GPU identity is invalid")
        utilization = float(utilization_text)
        memory_used = float(memory_text)
        if (
            not math.isfinite(utilization)
            or not 0.0 <= utilization <= 100.0
            or not math.isfinite(memory_used)
            or memory_used < 0.0
        ):
            raise ProcedureGridError("nvidia-smi GPU counters are invalid")
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
        process_rows = _strict_nvidia_csv_rows(process_result.stdout, width=4)
        own_memory = 0.0
        foreign: list[dict[str, Any]] = []
        seen: set[tuple[str, int]] = set()
        for pid_text, raw_uuid, process_name, used_text in process_rows:
            process_uuid = project_gpu_leases.canonical_gpu_uuid(raw_uuid)
            if not pid_text.isdigit() or int(pid_text) <= 0:
                raise ProcedureGridError("nvidia-smi process PID is invalid")
            pid = int(pid_text)
            used_memory = float(used_text)
            if (
                not math.isfinite(used_memory)
                or used_memory < 0.0
                or any(ord(character) < 32 for character in process_name)
                or (process_uuid, pid) in seen
            ):
                raise ProcedureGridError("nvidia-smi process accounting is invalid")
            seen.add((process_uuid, pid))
            if process_uuid != requested:
                continue
            process = {
                "pid": pid,
                "process_name": process_name,
                "used_gpu_memory_mib": used_memory,
            }
            if pid == os.getpid():
                own_memory += used_memory
            else:
                foreign.append(process)
        unexplained_memory = max(0.0, memory_used - own_memory)
        maximum_utilization = 100.0 if allow_active_owner else 10.0
        reasons: list[str] = []
        if foreign:
            reasons.append(f"{len(foreign)} foreign compute process(es)")
        if utilization > maximum_utilization:
            reasons.append(
                f"utilization {utilization:.1f}% > {maximum_utilization:.1f}%"
            )
        if unexplained_memory > 1024.0:
            reasons.append(f"foreign memory {unexplained_memory:.1f} MiB > 1024.0 MiB")
        return {
            "safe": not reasons,
            "gpu": requested,
            "utilization_percent": utilization,
            "memory_used_mib": memory_used,
            "own_compute_memory_mib": own_memory,
            "foreign_processes": foreign,
            "reason": "idle" if not reasons else "; ".join(reasons),
            "gpu_uuid": requested,
            "pci_bus_id": pci_bus_id,
            "name": gpu_name,
        }
    except (
        FileNotFoundError,
        subprocess.SubprocessError,
        ValueError,
        project_gpu_leases.ProjectGPULeaseError,
        ProcedureGridError,
    ) as error:
        return {
            "safe": False,
            "gpu": requested,
            "utilization_percent": None,
            "memory_used_mib": None,
            "own_compute_memory_mib": 0.0,
            "foreign_processes": [],
            "reason": f"nvidia-smi probe failed closed: {error}",
            "gpu_uuid": requested,
            "pci_bus_id": None,
            "name": None,
        }


def _resource_status(
    *,
    gpu: str,
    run_root: Path,
    minimum_free_gib: float,
    allow_active_owner: bool = False,
) -> dict[str, Any]:
    from .full_grid import probe_disk

    gpu_status = _strict_gpu_status(
        gpu=gpu,
        allow_active_owner=allow_active_owner,
    )
    disk_status = probe_disk(
        run_root,
        minimum_free_gib=minimum_free_gib,
    )
    return {
        "safe": bool(gpu_status["safe"] and disk_status.safe),
        "gpu": gpu_status,
        "disk": disk_status.as_dict(),
        "reason": (
            "resources available"
            if gpu_status["safe"] and disk_status.safe
            else f"GPU: {gpu_status['reason']}; disk: {disk_status.reason}"
        ),
    }


def _require_hard_disk_floor(path: Path, minimum_free_gib: float) -> None:
    """Check the bound filesystem immediately before an authoritative write."""

    if (
        isinstance(minimum_free_gib, bool)
        or not isinstance(minimum_free_gib, (int, float))
        or not math.isfinite(float(minimum_free_gib))
        or float(minimum_free_gib) < DEFAULT_MIN_FREE_GIB
    ):
        raise ProcedureGridError(
            f"disk floor cannot be below {DEFAULT_MIN_FREE_GIB:.0f} GiB"
        )
    descriptor = _open_directory_absolute(_safe_run_root(path))
    try:
        observed = os.fstatvfs(descriptor)
        free_bytes = int(observed.f_bavail) * int(observed.f_frsize)
    finally:
        os.close(descriptor)
    required_bytes = math.ceil(float(minimum_free_gib) * 1024**3)
    if free_bytes < required_bytes:
        raise ProcedureGridError(
            f"filesystem has {free_bytes} bytes free; "
            f"{required_bytes} bytes are required before publication"
        )


def _worker_claim_loop(
    *,
    run_root: Path,
    cache_root: Path,
    plan: Mapping[str, Any],
    device: str,
    worker_index: int,
    cpu_threads: int,
    recover_stale: bool,
    max_jobs: int | None,
    selected_probe: Callable[[], Mapping[str, Any]],
    selected_commit_probe: Callable[[], Mapping[str, Any]],
    selected_executor: Callable[..., tuple[dict[str, Any], np.ndarray, np.ndarray]],
    gpu_lease: GPULease | None,
) -> int:
    jobs = tuple(iter_jobs(plan))
    completed_now = 0
    cursor = (int(worker_index) * math.ceil(len(jobs) / MAX_PROJECT_GPU_WORKERS)) % max(
        len(jobs), 1
    )
    ordered = jobs[cursor:] + jobs[:cursor]
    for job in ordered:
        destination = _record_directory(run_root, job)
        if destination.exists():
            validate_completion(run_root, plan, job)
            _recover_completed_claim(run_root, plan, job, recover_stale=recover_stale)
            continue
        if max_jobs is not None and completed_now >= int(max_jobs):
            break
        resource = _json_ready(selected_probe())
        if resource.get("safe") is not True:
            return 75
        _validate_resource_guard(resource, plan=plan)
        try:
            claim = acquire_claim(
                run_root,
                plan,
                job,
                resource_guard=resource,
                gpu_lease=gpu_lease,
                recover_stale=recover_stale,
            )
        except PublicationActive:
            return 75
        except ClaimUnavailable:
            continue
        try:
            _recover_job_partials(run_root, job)
            metadata, rows, probabilities = selected_executor(
                job=job,
                plan=plan,
                cache_root=cache_root,
                device=device,
                cpu_threads=cpu_threads,
            )
            final_resource = _json_ready(selected_commit_probe())
            _validate_resource_guard(final_resource, plan=plan)
            commit_job_output(
                run_root,
                plan,
                claim,
                metadata=metadata,
                test_rows=rows,
                probabilities=probabilities,
                resource_recheck=final_resource,
                gpu_lease=gpu_lease,
            )
            completed_now += 1
        except Exception as error:
            _record_failure(run_root, plan, claim, error)
            raise
        finally:
            try:
                release_claim(claim)
            except ProcedureGridError:
                if destination.exists() and not destination.is_symlink():
                    _quarantine(
                        run_root,
                        destination,
                        category="claim_lost_after_commit",
                    )
                raise
    return 0


def worker_loop(
    *,
    run_root: Path,
    cache_root: Path,
    gpu: str,
    device: str,
    worker_index: int,
    cpu_threads: int,
    recover_stale: bool,
    minimum_free_gib: float = DEFAULT_MIN_FREE_GIB,
    max_jobs: int | None = None,
    executor: Callable[..., tuple[dict[str, Any], np.ndarray, np.ndarray]]
    | None = None,
    resource_probe: Callable[[], Mapping[str, Any]] | None = None,
    verify_identity: bool = True,
) -> int:
    plan = load_plan(run_root)
    execution = plan["execution_config"]
    if plan["publication_mode"] == FORMAL_PUBLICATION_MODE and not verify_identity:
        raise ProcedureGridError("formal workers cannot bypass runtime identity")
    if plan["publication_mode"] == FORMAL_PUBLICATION_MODE and (
        executor is not None or resource_probe is not None
    ):
        raise ProcedureGridError("formal workers cannot inject executor/resource hooks")
    if int(cpu_threads) != int(plan["execution_config"]["cpu_threads_per_worker"]):
        raise ProcedureGridError("worker CPU thread count differs from immutable plan")
    if (
        not math.isfinite(float(minimum_free_gib))
        or not math.isclose(
            float(minimum_free_gib),
            float(execution["minimum_free_gib"]),
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or str(device) != str(execution["device"])
    ):
        raise ProcedureGridError("worker resource/device override differs from plan")
    selected_probe = resource_probe or (
        lambda: _resource_status(
            gpu=gpu,
            run_root=run_root,
            minimum_free_gib=minimum_free_gib,
        )
    )
    selected_commit_probe = resource_probe or (
        lambda: _resource_status(
            gpu=gpu,
            run_root=run_root,
            minimum_free_gib=minimum_free_gib,
            allow_active_owner=True,
        )
    )
    opening_resource = _json_ready(selected_probe())
    if opening_resource.get("safe") is not True:
        return 75
    _validate_resource_guard(opening_resource, plan=plan)
    if plan["publication_mode"] == FORMAL_PUBLICATION_MODE:
        if (
            type(worker_index) is not int
            or not 0 <= worker_index < MAX_PROJECT_GPU_WORKERS
        ):
            raise ProcedureGridError(
                "formal worker index must be 0, 1, or 2; GPU 3 is never a default"
            )
        expected_uuid = opening_resource["gpu"]["gpu_uuid"]
        if str(gpu) != expected_uuid:
            raise ProcedureGridError("formal --gpu must be the probed full GPU UUID")
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible not in {None, expected_uuid}:
            raise ProcedureGridError("CUDA_VISIBLE_DEVICES conflicts with --gpu")
        os.environ["CUDA_VISIBLE_DEVICES"] = expected_uuid
        from .full_grid import _verify_visible_cuda_device

        _verify_visible_cuda_device(expected_uuid)
    lease: GPULease | None = None
    if plan["publication_mode"] == FORMAL_PUBLICATION_MODE:
        try:
            lease = acquire_gpu_lease(run_root, plan, str(gpu))
        except GPUWorkerUnavailable:
            return 75
    try:
        if lease is not None:
            leased_resource = _json_ready(selected_probe())
            if leased_resource.get("safe") is not True:
                return 75
            _validate_resource_guard(leased_resource, plan=plan)
            if (
                project_gpu_leases.canonical_gpu_uuid(
                    str(leased_resource["gpu"]["gpu_uuid"])
                )
                != lease.value["gpu_uuid"]
            ):
                raise ProcedureGridError(
                    "leased physical GPU changed after acquisition"
                )
        if verify_identity:
            verify_runtime_identity(plan, cache_root=_absolute_path(cache_root))
        return _worker_claim_loop(
            run_root=run_root,
            cache_root=cache_root,
            plan=plan,
            device=device,
            worker_index=worker_index,
            cpu_threads=cpu_threads,
            recover_stale=recover_stale,
            max_jobs=max_jobs,
            selected_probe=selected_probe,
            selected_commit_probe=selected_commit_probe,
            selected_executor=executor or execute_procedure_job,
            gpu_lease=lease,
        )
    finally:
        if lease is not None:
            release_gpu_lease(lease)


def audit_grid(run_root: Path, plan: Mapping[str, Any] | None = None) -> dict[str, Any]:
    root = _safe_run_root(run_root)
    selected_plan = load_plan(root) if plan is None else dict(plan)
    validate_plan_semantics(selected_plan)
    jobs = tuple(iter_jobs(selected_plan))
    expected = {_record_directory(root, job): job for job in jobs}
    complete: list[str] = []
    missing: list[str] = []
    corrupt: dict[str, str] = {}
    for directory, job in expected.items():
        if not directory.exists():
            missing.append(job.job_id)
            continue
        try:
            validate_completion(root, selected_plan, job)
        except (
            KeyError,
            OSError,
            TypeError,
            ValueError,
            ProcedureGridError,
        ) as error:
            corrupt[job.job_id] = str(error)
        else:
            complete.append(job.job_id)
    observed_record_directories: set[Path] = set()
    records_root = root / "records"
    unexpected_record_paths: list[str] = []
    unsafe_paths: list[str] = []
    if records_root.exists():
        try:
            _require_real_directory(records_root)
        except ProcedureGridError as error:
            unsafe_paths.append(f"records: {error}")
        expected_files = {
            directory / name for directory in expected for name in FINAL_FILENAMES
        }
        observed_files: set[Path] = set()
        if not records_root.is_symlink():
            for path in records_root.rglob("*"):
                try:
                    observed = path.lstat()
                except FileNotFoundError:
                    unsafe_paths.append(f"{path.relative_to(root)}: disappeared")
                    continue
                if path.is_symlink():
                    unsafe_paths.append(f"{path.relative_to(root)}: symlink")
                elif stat.S_ISREG(observed.st_mode):
                    observed_files.add(path)
                    if observed.st_nlink != 1:
                        unsafe_paths.append(
                            f"{path.relative_to(root)}: hardlinked file"
                        )
                elif not stat.S_ISDIR(observed.st_mode):
                    unsafe_paths.append(f"{path.relative_to(root)}: special node")
        observed_record_directories = {path.parent for path in observed_files}
        unexpected_record_paths = sorted(
            str(path.relative_to(root)) for path in observed_files - expected_files
        )
        allowed_directories = {records_root}
        for directory in expected:
            current = directory
            while current != root:
                allowed_directories.add(current)
                if current == records_root:
                    break
                current = current.parent
        if not records_root.is_symlink():
            unexpected_record_paths.extend(
                sorted(
                    str(path.relative_to(root))
                    for path in records_root.rglob("*")
                    if path.is_dir()
                    and not path.is_symlink()
                    and path not in allowed_directories
                )
            )
    extra = sorted(
        str(value.relative_to(root))
        for value in observed_record_directories - set(expected)
    )

    def residual_tree(name: str) -> list[str]:
        directory = root / name
        if not directory.exists() and not directory.is_symlink():
            return []
        try:
            _require_real_directory(directory)
        except ProcedureGridError as error:
            unsafe_paths.append(f"{name}: {error}")
            return [name]
        values: list[str] = []
        for path in directory.rglob("*"):
            values.append(str(path.relative_to(root)))
            observed = path.lstat()
            if path.is_symlink():
                unsafe_paths.append(f"{path.relative_to(root)}: symlink")
            elif stat.S_ISREG(observed.st_mode):
                if observed.st_nlink != 1:
                    unsafe_paths.append(f"{path.relative_to(root)}: hardlinked file")
            elif not stat.S_ISDIR(observed.st_mode):
                unsafe_paths.append(f"{path.relative_to(root)}: special node")
        return sorted(values)

    claim_paths = residual_tree("claims")
    partial_paths = residual_tree("partials")
    for forensic_name in ("failures", "quarantine"):
        directory = root / forensic_name
        if not directory.exists() and not directory.is_symlink():
            continue
        try:
            _require_real_directory(directory)
        except ProcedureGridError as error:
            unsafe_paths.append(f"{forensic_name}: {error}")
            continue
        for path in directory.rglob("*"):
            observed = path.lstat()
            if path.is_symlink():
                unsafe_paths.append(f"{path.relative_to(root)}: symlink")
            elif stat.S_ISREG(observed.st_mode):
                if observed.st_nlink != 1:
                    unsafe_paths.append(f"{path.relative_to(root)}: hardlinked file")
            elif not stat.S_ISDIR(observed.st_mode):
                unsafe_paths.append(f"{path.relative_to(root)}: special node")
    failures_root = root / "failures"
    if failures_root.exists() and not failures_root.is_symlink():
        expected_job_ids = {job.job_id for job in jobs}
        for job_path in failures_root.iterdir():
            if (
                not job_path.is_dir()
                or job_path.is_symlink()
                or job_path.name not in expected_job_ids
            ):
                unsafe_paths.append(
                    f"{job_path.relative_to(root)}: unexpected failure path"
                )
                continue
            for receipt in job_path.iterdir():
                if (
                    receipt.is_symlink()
                    or not receipt.is_file()
                    or not re.fullmatch(r"attempt-[0-9]{4}\.json", receipt.name)
                ):
                    unsafe_paths.append(
                        f"{receipt.relative_to(root)}: unexpected failure receipt"
                    )
    quarantine_root = root / "quarantine"
    allowed_quarantine_categories = {
        "stale_claims",
        "power_cut_partials",
        "post_rename_stale_claims",
        "duplicate_partials",
        "failed_partials",
    }
    if quarantine_root.exists() and not quarantine_root.is_symlink():
        for category in quarantine_root.iterdir():
            if (
                not category.is_dir()
                or category.is_symlink()
                or category.name not in allowed_quarantine_categories
            ):
                unsafe_paths.append(
                    f"{category.relative_to(root)}: unexpected quarantine category"
                )
    expected_root_entries = {
        "plan.json",
        "plan.sha256",
        "records",
        "claims",
        "partials",
        PUBLICATION_FENCE_FILENAME,
    }
    for path in root.iterdir():
        if path.name in RUN_DIRECTORY_NAMES:
            try:
                _require_real_directory(path)
            except ProcedureGridError as error:
                unsafe_paths.append(f"{path.name}: {error}")
        elif path.name in {"plan.json", "plan.sha256", PUBLICATION_FENCE_FILENAME}:
            try:
                _require_unique_regular(path, read_only=True)
            except ProcedureGridError as error:
                unsafe_paths.append(f"{path.name}: {error}")
    fence_path = root / PUBLICATION_FENCE_FILENAME
    if not fence_path.exists() and not fence_path.is_symlink():
        unsafe_paths.append("publication fence is absent")
    else:
        try:
            if _strict_json_load(fence_path) != _publication_fence_value(selected_plan):
                unsafe_paths.append("publication fence token is invalid")
        except (OSError, ProcedureGridError) as error:
            unsafe_paths.append(f"publication fence: {error}")
    unexpected_root = sorted(
        path.name for path in root.iterdir() if path.name not in expected_root_entries
    )
    exact = (
        len(complete) == len(jobs)
        and not missing
        and not corrupt
        and not extra
        and not unexpected_record_paths
        and not claim_paths
        and not partial_paths
        and not unexpected_root
        and not unsafe_paths
    )
    return {
        "schema": AUDIT_SCHEMA,
        "plan_sha256": selected_plan["plan_sha256"],
        "publication_mode": selected_plan["publication_mode"],
        "formal_publication_eligible": (
            selected_plan["publication_mode"] == FORMAL_PUBLICATION_MODE
        ),
        "expected_jobs": len(jobs),
        "complete_jobs": len(complete),
        "missing_jobs": len(missing),
        "corrupt_jobs": len(corrupt),
        "complete": exact,
        "complete_job_ids": complete,
        "missing_job_ids": missing,
        "corrupt_job_ids": corrupt,
        "extra_record_directories": extra,
        "unexpected_record_paths": unexpected_record_paths,
        "residual_claim_paths": claim_paths,
        "residual_partial_paths": partial_paths,
        "unexpected_root_entries": unexpected_root,
        "unsafe_paths": sorted(set(unsafe_paths)),
    }


def _parse_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory_parser = subparsers.add_parser(
        "inventory", help="print the outcome-free/blocked config boundary"
    )
    inventory_parser.add_argument("--pretty", action="store_true")

    plan_parser = subparsers.add_parser(
        "plan", help="create or verify the immutable formal plan"
    )
    plan_parser.add_argument("--run-root", type=Path, required=True)
    plan_parser.add_argument("--cache-root", type=Path, required=True)
    plan_parser.add_argument("--cpu-threads", type=_parse_int, default=4)

    worker_parser = subparsers.add_parser(
        "worker", help="run one resumable single-GPU worker"
    )
    worker_parser.add_argument("--run-root", type=Path, required=True)
    worker_parser.add_argument("--cache-root", type=Path, required=True)
    worker_parser.add_argument("--gpu", required=True)
    worker_parser.add_argument("--device", default="cuda")
    worker_parser.add_argument("--worker-index", type=int, default=0)
    worker_parser.add_argument("--cpu-threads", type=_parse_int, default=4)
    worker_parser.add_argument("--recover-stale", action="store_true")
    worker_parser.add_argument(
        "--minimum-free-gib", type=float, default=DEFAULT_MIN_FREE_GIB
    )
    worker_parser.add_argument("--max-jobs", type=_parse_int)

    audit_parser = subparsers.add_parser(
        "audit", help="validate exact completion and quiescence"
    )
    audit_parser.add_argument("--run-root", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "inventory":
        value = _config_inventory()
        print(
            json.dumps(
                value,
                indent=2 if args.pretty else None,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "plan":
        _require_formal_release_runtime()
        os.environ.setdefault(
            "CUBLAS_WORKSPACE_CONFIG", REQUIRED_CUBLAS_WORKSPACE_CONFIG
        )
        if os.environ["CUBLAS_WORKSPACE_CONFIG"] != REQUIRED_CUBLAS_WORKSPACE_CONFIG:
            raise ProcedureGridError(
                "CUBLAS_WORKSPACE_CONFIG differs from the formal contract"
            )
        _install_thread_environment(int(args.cpu_threads))
        contracts = _dataset_contracts()
        cache_root = _absolute_path(args.cache_root)
        _require_no_symlink_ancestors(cache_root)
        _require_real_directory(cache_root)
        cache_identity, split_identity = _cache_and_split_identity(
            cache_root, contracts
        )
        plan = assemble_plan(
            dataset_contracts=contracts,
            cache_identity=cache_identity,
            split_identity=split_identity,
            source_identity=_source_identity(),
            environment_identity=_environment_identity(),
            config_inventory=_config_inventory(),
            cpu_threads=int(args.cpu_threads),
        )
        observed = write_or_validate_plan(args.run_root, plan)
        print(
            json.dumps(
                {
                    "plan_sha256": observed["plan_sha256"],
                    "n_jobs": observed["n_jobs"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "worker":
        _require_formal_release_runtime()
        if args.worker_index < 0:
            raise ValueError("worker index must be nonnegative")
        os.environ.setdefault(
            "CUBLAS_WORKSPACE_CONFIG", REQUIRED_CUBLAS_WORKSPACE_CONFIG
        )
        if os.environ["CUBLAS_WORKSPACE_CONFIG"] != REQUIRED_CUBLAS_WORKSPACE_CONFIG:
            raise ProcedureGridError(
                "CUBLAS_WORKSPACE_CONFIG differs from the formal contract"
            )
        _install_thread_environment(int(args.cpu_threads))
        return worker_loop(
            run_root=args.run_root,
            cache_root=args.cache_root,
            gpu=str(args.gpu),
            device=str(args.device),
            worker_index=int(args.worker_index),
            cpu_threads=int(args.cpu_threads),
            recover_stale=bool(args.recover_stale),
            minimum_free_gib=float(args.minimum_free_gib),
            max_jobs=args.max_jobs,
        )
    if args.command == "audit":
        value = audit_grid(args.run_root)
        print(json.dumps(value, indent=2, sort_keys=True))
        return 0 if value["complete"] else 1
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUDIT_SCHEMA",
    "BINARY_DATASETS",
    "BLOCKED_ORBIT_PROVENANCE",
    "COMPLETION_SCHEMA",
    "FORMAL_SEEDS",
    "FORMAL_EXPECTED_JOBS",
    "FORMAL_PUBLICATION_MODE",
    "PLAN_SCHEMA",
    "PROCEDURES",
    "RECORD_SCHEMA",
    "SOURCE_FILES",
    "VIEW_SCHEMA",
    "TEST_PUBLICATION_MODE",
    "Job",
    "assemble_plan",
    "audit_grid",
    "commit_job_output",
    "covariance_view_contract",
    "derive_paired_views",
    "execute_procedure_job",
    "iter_jobs",
    "load_or_repair_plan",
    "load_plan",
    "main",
    "plan_sha256",
    "publication_fence",
    "validate_plan_semantics",
    "validate_uv_dependency_closure",
    "validate_completion",
    "verify_runtime_identity",
    "worker_loop",
    "write_or_validate_plan",
]
