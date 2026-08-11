"""Score-blind, resumable CPU grid for the three deterministic MI controls.

This runner is deliberately separate from the stochastic neural full grid.  A
job is exactly one ``(control, dataset, subject, fold)`` protocol execution:
hyperparameters are selected on validation NLL, all learned state is rebuilt
on train+validation, and the held-out rows are passed to ``predict_proba``
exactly once.  Deterministic controls do not acquire seed identities.

Only held-out row indices and class probabilities are published.  Trial labels
and performance scores are never written by this module.  Statistical analysis
must join an audited prediction artifact to the locked cache in a separate
step.

The production CLI requires a UV-managed virtual environment.  It never
installs a package, reserves a GPU, or launches distributed training.  Worker
processes cooperate through filesystem CPU slots and cap BLAS, OpenMP, NumExpr,
Numba, and Torch CPU thread pools.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import ctypes
import errno
import fcntl
import hashlib
import io
import importlib.metadata
import json
import os
import pickle
import platform
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
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import log_loss

from .config import CHANNEL_SCALING, dataset_spec, preprocessing_for_dataset

PLAN_SCHEMA = "ieee-mi-score-blind-control-grid-plan-v1"
RECORD_SCHEMA = "ieee-mi-score-blind-control-record-v1"
COMPLETION_SCHEMA = "ieee-mi-score-blind-control-completion-v1"
CLAIM_SCHEMA = "ieee-mi-control-grid-claim-v1"
CPU_SLOT_SCHEMA = "ieee-mi-control-grid-cpu-slot-v1"
FAILURE_SCHEMA = "ieee-mi-control-grid-failure-v1"
AUDIT_SCHEMA = "ieee-mi-score-blind-control-audit-v1"

OPENED_DATASETS: tuple[str, ...] = (
    "local_exp4",
    "bnci2014_001",
    "bnci2014_004",
    "cho2017",
    "physionet_mi",
)
DATASET_SUBJECTS: Mapping[str, tuple[int, ...]] = {
    "local_exp4": (1, 3, 4, 5, 6, 7, 8, 10),
    "bnci2014_001": tuple(range(1, 10)),
    "bnci2014_004": tuple(range(1, 10)),
    "cho2017": tuple(range(1, 53)),
    "physionet_mi": tuple(range(1, 55)),
}
DATASET_FOLDS: Mapping[str, tuple[int, ...]] = {
    "local_exp4": (0,),
    "bnci2014_001": (0,),
    "bnci2014_004": (0,),
    "cho2017": (0, 1, 2, 3, 4),
    "physionet_mi": (0, 1, 2),
}
CONTROLS: tuple[str, ...] = (
    "control.riemann",
    "control.tangent_anchor",
    "control.ea_fbcsp",
)
RUNTIME_KEYS: Mapping[str, str] = {
    "control.riemann": "riemann",
    "control.tangent_anchor": "tangent_anchor",
    "control.ea_fbcsp": "ea_fbcsp",
}
IMPLEMENTATIONS: Mapping[str, str] = {
    "control.riemann": "deepnet.baselines.RiemannianTangentLogistic",
    "control.tangent_anchor": "deepnet.tangent_anchor.TangentAnchorClassifier",
    "control.ea_fbcsp": "deepnet.baselines.EAFilterBankCSP",
}
BASE_HYPERPARAMETER_GRID: Mapping[str, tuple[dict[str, float | int], ...]] = {
    "control.riemann": tuple(
        {"c": value} for value in (0.01, 0.1, 1.0, 10.0)
    ),
    "control.tangent_anchor": tuple(
        {"C": value} for value in (0.01, 0.1, 1.0, 10.0)
    ),
    "control.ea_fbcsp": tuple(
        {"n_components": value} for value in (2, 4, 6)
    ),
}

FITS_PER_CONTROL = sum(
    len(DATASET_SUBJECTS[dataset]) * len(DATASET_FOLDS[dataset])
    for dataset in OPENED_DATASETS
)
EXPECTED_JOB_COUNT = len(CONTROLS) * FITS_PER_CONTROL
DEFAULT_CPU_BUDGET = 16
DEFAULT_THREADS_PER_WORKER = 4
DEFAULT_MIN_FREE_GIB = 50.0
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
SOURCE_FILES: tuple[str, ...] = (
    "ieee_mi/control_grid.py",
    "ieee_mi/config.py",
    "ieee_mi/data.py",
    "ieee_mi/local_geometric_controls.py",
    "ieee_mi/model_registry.py",
    "deepnet/baselines.py",
    "deepnet/spd.py",
    "deepnet/tangent_anchor.py",
)
REQUIRED_DIRECT_DEPENDENCIES = frozenset(
    {
        "mne",
        "numpy",
        "pyriemann",
        "scikit-learn",
        "scipy",
        "threadpoolctl",
        "torch",
    }
)
FINAL_FILENAMES = frozenset({"record.json", "predictions.npz", "completion.json"})
PREDICTION_ZIP_MEMBERS: tuple[str, ...] = (
    "rows.npy",
    "probabilities.npy",
)
COORDINATION_LOCK_NAME = ".control-grid.coordination.lock"
STAT_FINGERPRINT_FIELDS: tuple[str, ...] = (
    "st_dev",
    "st_ino",
    "st_mode",
    "st_nlink",
    "st_size",
    "st_mtime_ns",
    "st_ctime_ns",
)
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
        "validation_nll",
        "test_nll",
        "ground_truth",
        "truth",
        "y_test",
        "y_true",
    }
)


class ControlGridError(RuntimeError):
    """Base class for an audited control-grid contract violation."""


class RegistryDriftError(ControlGridError):
    """The executable control roster no longer matches the registry."""


class ClaimUnavailable(ControlGridError):
    """A job is complete or another live worker owns it."""


class CPUUnavailable(ControlGridError):
    """Every cooperative CPU slot is occupied."""


class DiskUnavailable(ControlGridError):
    """The output filesystem is below the frozen low-water mark."""


class EstimatorCapabilityError(RegistryDriftError):
    """A registered estimator cannot satisfy its binary/multiclass contract."""


@dataclass(frozen=True, order=True)
class Job:
    dataset: str
    control: str
    subject: int
    fold: int

    def identity(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "control": self.control,
            "subject": int(self.subject),
            "fold": int(self.fold),
        }

    @property
    def job_id(self) -> str:
        digest = hashlib.sha256(_canonical_bytes(self.identity())).hexdigest()
        return f"control-{digest[:24]}"


@dataclass(frozen=True)
class Claim:
    job: Job
    path: Path
    nonce: str
    owner: Mapping[str, Any]
    resource_guard: Mapping[str, Any]
    descriptor: int = -1
    st_dev: int = -1
    st_ino: int = -1
    payload: bytes = b""


@dataclass(frozen=True)
class CPUSlot:
    index: int
    path: Path
    nonce: str
    owner: Mapping[str, Any]
    threads: int
    descriptor: int = -1
    st_dev: int = -1
    st_ino: int = -1
    payload: bytes = b""


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
class PreparedFeatures:
    """Split-specific features with no held-out outcome vector."""

    train_epochs: np.ndarray
    validation_epochs: np.ndarray
    source_epochs: np.ndarray
    test_epochs: np.ndarray
    train_covariances: np.ndarray
    validation_covariances: np.ndarray
    source_covariances: np.ndarray
    test_covariances: np.ndarray
    train_outcomes: np.ndarray
    validation_outcomes: np.ndarray
    source_outcomes: np.ndarray
    test_rows: np.ndarray
    selection_mean: np.ndarray
    selection_std: np.ndarray
    refit_mean: np.ndarray
    refit_std: np.ndarray
    channel_names: tuple[str, ...]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


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


def _absolute(path: Path | str) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _stat_fingerprint(observed: os.stat_result) -> tuple[int, ...]:
    return tuple(
        int(getattr(observed, field))
        for field in STAT_FINGERPRINT_FIELDS
    )


def _directory_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _open_directory_absolute(path: Path | str) -> int:
    """Open an absolute directory by a descriptor-pinned no-follow walk."""

    absolute = _absolute(path)
    descriptor = os.open(absolute.anchor, _directory_flags())
    try:
        for component in absolute.parts[1:]:
            child = os.open(component, _directory_flags(), dir_fd=descriptor)
            observed = os.fstat(child)
            if not stat.S_ISDIR(observed.st_mode):
                os.close(child)
                raise ControlGridError(
                    f"unsafe directory component while opening {absolute}"
                )
            os.close(descriptor)
            descriptor = child
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _assert_directory_path(path: Path | str, descriptor: int) -> None:
    replacement = _open_directory_absolute(path)
    try:
        held = os.fstat(descriptor)
        current = os.fstat(replacement)
        if (
            not stat.S_ISDIR(held.st_mode)
            or (int(held.st_dev), int(held.st_ino))
            != (int(current.st_dev), int(current.st_ino))
        ):
            raise ControlGridError(
                f"directory namespace changed during transaction: {_absolute(path)}"
            )
    finally:
        os.close(replacement)


def _safe_mkdir(path: Path | str, *, mode: int = 0o700) -> Path:
    """Create a hierarchy without following a symlinked ancestor."""

    absolute = _absolute(path)
    descriptor = os.open(absolute.anchor, _directory_flags())
    try:
        for component in absolute.parts[1:]:
            try:
                child = os.open(
                    component, _directory_flags(), dir_fd=descriptor
                )
            except FileNotFoundError:
                os.mkdir(component, mode=mode, dir_fd=descriptor)
                os.fsync(descriptor)
                child = os.open(
                    component, _directory_flags(), dir_fd=descriptor
                )
            observed = os.fstat(child)
            if not stat.S_ISDIR(observed.st_mode):
                os.close(child)
                raise ControlGridError(
                    f"unsafe directory component while creating {absolute}"
                )
            os.close(descriptor)
            descriptor = child
    finally:
        os.close(descriptor)
    return absolute


def _anchored_lstat(path: Path | str) -> os.stat_result:
    absolute = _absolute(path)
    parent = _open_directory_absolute(absolute.parent)
    try:
        return os.stat(
            absolute.name, dir_fd=parent, follow_symlinks=False
        )
    finally:
        os.close(parent)


def _path_exists(path: Path | str) -> bool:
    try:
        _anchored_lstat(path)
        return True
    except FileNotFoundError:
        return False


def _assert_unique_regular(
    observed: os.stat_result,
    *,
    source: str,
    allow_empty: bool = False,
    readonly: bool = False,
) -> None:
    if (
        not stat.S_ISREG(observed.st_mode)
        or observed.st_nlink != 1
        or (observed.st_size <= 0 and not allow_empty)
        or (readonly and observed.st_mode & 0o222)
    ):
        raise ControlGridError(
            f"{source} is not a unique"
            f"{' read-only' if readonly else ''} regular file"
        )


def _read_descriptor_bytes(
    descriptor: int,
    *,
    source: str,
    allow_empty: bool = False,
) -> bytes:
    before = os.fstat(descriptor)
    _assert_unique_regular(
        before, source=source, allow_empty=allow_empty
    )
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    remaining = int(before.st_size)
    while remaining:
        try:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
        except InterruptedError:
            continue
        if not chunk:
            raise ControlGridError(f"{source} was truncated while reading")
        chunks.append(chunk)
        remaining -= len(chunk)
    after = os.fstat(descriptor)
    if _stat_fingerprint(before) != _stat_fingerprint(after):
        raise ControlGridError(f"{source} changed while reading")
    return b"".join(chunks)


def _read_unique_regular_at(
    parent_descriptor: int,
    name: str,
    *,
    source: str,
    allow_empty: bool = False,
    readonly: bool = False,
) -> tuple[bytes, os.stat_result]:
    if name in {"", ".", ".."} or "/" in name:
        raise ControlGridError(f"{source} has an invalid leaf name")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptor = -1
    try:
        before_path = os.stat(
            name, dir_fd=parent_descriptor, follow_symlinks=False
        )
        _assert_unique_regular(
            before_path,
            source=source,
            allow_empty=allow_empty,
            readonly=readonly,
        )
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
        opened = os.fstat(descriptor)
        _assert_unique_regular(
            opened,
            source=source,
            allow_empty=allow_empty,
            readonly=readonly,
        )
        if _stat_fingerprint(opened) != _stat_fingerprint(before_path):
            raise ControlGridError(f"{source} changed while opening")
        payload = _read_descriptor_bytes(
            descriptor, source=source, allow_empty=allow_empty
        )
        final_descriptor = os.fstat(descriptor)
        final_path = os.stat(
            name, dir_fd=parent_descriptor, follow_symlinks=False
        )
        if (
            _stat_fingerprint(final_descriptor)
            != _stat_fingerprint(opened)
            or _stat_fingerprint(final_path)
            != _stat_fingerprint(opened)
        ):
            raise ControlGridError(f"{source} changed during snapshot")
        return payload, opened
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _read_unique_regular_bytes(
    path: Path | str,
    *,
    allow_empty: bool = False,
    readonly: bool = False,
) -> bytes:
    absolute = _absolute(path)
    parent = _open_directory_absolute(absolute.parent)
    parent_fingerprint = _stat_fingerprint(os.fstat(parent))
    try:
        payload, _ = _read_unique_regular_at(
            parent,
            absolute.name,
            source=str(absolute),
            allow_empty=allow_empty,
            readonly=readonly,
        )
        _assert_directory_path(absolute.parent, parent)
        if _stat_fingerprint(os.fstat(parent)) != parent_fingerprint:
            raise ControlGridError(
                f"{absolute.parent} changed during verified read"
            )
        return payload
    finally:
        os.close(parent)


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(_read_unique_regular_bytes(path))


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _rows_sha256(value: np.ndarray | Sequence[int]) -> str:
    rows = np.asarray(value, dtype="<i8")
    if rows.ndim != 1:
        raise ValueError("row indices must be one-dimensional")
    rows = np.ascontiguousarray(rows)
    digest = hashlib.sha256()
    digest.update(b"ieee-mi-control-row-indices-v1\0")
    digest.update(np.asarray(rows.shape, dtype="<i8").tobytes())
    digest.update(rows.tobytes())
    return digest.hexdigest()


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _strict_json_bytes(payload: bytes, *, source: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ControlGridError(f"{source} is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ControlGridError(f"{source} does not contain a JSON object")
    if payload != _canonical_bytes(value) + b"\n":
        raise ControlGridError(f"{source} is not canonical JSON")
    return value


def strict_load(path: Path) -> dict[str, Any]:
    return _strict_json_bytes(
        _read_unique_regular_bytes(path), source=str(_absolute(path))
    )


def _snapshot_regular_files(
    directory: Path | str,
    names: Sequence[str],
    *,
    exact_roster: frozenset[str] | None = None,
    directory_mode: int | None = None,
    child_mode: int | None = None,
    stable_parent: bool = False,
) -> tuple[
    dict[str, bytes],
    tuple[int, ...],
    dict[str, tuple[int, ...]],
]:
    """Read a flat package from one retained directory and all retained FDs."""

    absolute = _absolute(directory)
    parent_descriptor = -1
    parent_fingerprint: tuple[int, ...] | None = None
    if stable_parent:
        parent_descriptor = _open_directory_absolute(absolute.parent)
        parent_fingerprint = _stat_fingerprint(
            os.fstat(parent_descriptor)
        )
        directory_descriptor = os.open(
            absolute.name,
            _directory_flags(),
            dir_fd=parent_descriptor,
        )
    else:
        directory_descriptor = _open_directory_absolute(absolute)
    directory_before = os.fstat(directory_descriptor)
    directory_fingerprint = _stat_fingerprint(directory_before)
    descriptors: dict[str, int] = {}
    child_fingerprints: dict[str, tuple[int, ...]] = {}
    payloads: dict[str, bytes] = {}
    try:
        if (
            directory_mode is not None
            and stat.S_IMODE(directory_before.st_mode) != directory_mode
        ):
            raise ControlGridError(
                f"{absolute} does not have required mode {directory_mode:o}"
            )
        if exact_roster is not None and frozenset(
            os.listdir(directory_descriptor)
        ) != exact_roster:
            raise ControlGridError(f"{absolute} has an incomplete file set")
        for name in names:
            if name in {"", ".", ".."} or "/" in name:
                raise ControlGridError("snapshot has an invalid child name")
            before_path = os.stat(
                name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            _assert_unique_regular(
                before_path,
                source=str(absolute / name),
                readonly=(child_mode is not None),
            )
            if (
                child_mode is not None
                and stat.S_IMODE(before_path.st_mode) != child_mode
            ):
                raise ControlGridError(
                    f"{absolute / name} does not have required mode "
                    f"{child_mode:o}"
                )
            descriptor = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0),
                dir_fd=directory_descriptor,
            )
            opened = os.fstat(descriptor)
            if _stat_fingerprint(opened) != _stat_fingerprint(before_path):
                os.close(descriptor)
                raise ControlGridError(
                    f"{absolute / name} changed while opening"
                )
            descriptors[name] = descriptor
            child_fingerprints[name] = _stat_fingerprint(opened)
        for name in names:
            payloads[name] = _read_descriptor_bytes(
                descriptors[name], source=str(absolute / name)
            )
        for name in names:
            held = os.fstat(descriptors[name])
            current = os.stat(
                name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            expected = child_fingerprints[name]
            if (
                _stat_fingerprint(held) != expected
                or _stat_fingerprint(current) != expected
            ):
                raise ControlGridError(
                    f"{absolute / name} changed during package snapshot"
                )
        if (
            _stat_fingerprint(os.fstat(directory_descriptor))
            != directory_fingerprint
            or (
                exact_roster is not None
                and frozenset(os.listdir(directory_descriptor))
                != exact_roster
            )
        ):
            raise ControlGridError(
                f"{absolute} changed during package snapshot"
            )
        _assert_directory_path(absolute, directory_descriptor)
        if stable_parent:
            current = os.stat(
                absolute.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if (
                _stat_fingerprint(current)
                != directory_fingerprint
                or _stat_fingerprint(os.fstat(parent_descriptor))
                != parent_fingerprint
            ):
                raise ControlGridError(
                    f"{absolute.parent} changed during package snapshot"
                )
            _assert_directory_path(
                absolute.parent, parent_descriptor
            )
        return payloads, directory_fingerprint, child_fingerprints
    finally:
        for descriptor in descriptors.values():
            os.close(descriptor)
        os.close(directory_descriptor)
        if parent_descriptor >= 0:
            os.close(parent_descriptor)


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str] | frozenset[str], context: str
) -> None:
    observed = set(value)
    if observed != set(expected):
        missing = sorted(set(expected) - observed)
        extra = sorted(observed - set(expected))
        raise ControlGridError(
            f"{context} fields differ from schema; missing={missing}, extra={extra}"
        )


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _require_sha256(value: Any, context: str) -> None:
    if not _is_sha256(value):
        raise ControlGridError(f"{context} is not a lowercase SHA-256 digest")


def _fsync_directory(path: Path) -> None:
    descriptor = _open_directory_absolute(path)
    try:
        os.fsync(descriptor)
        _assert_directory_path(path, descriptor)
    finally:
        os.close(descriptor)


def _write_exclusive_at(
    parent_descriptor: int,
    name: str,
    payload: bytes,
    *,
    mode: int = 0o444,
) -> os.stat_result:
    if name in {"", ".", ".."} or "/" in name:
        raise ControlGridError("exclusive write requires one safe leaf")
    descriptor = os.open(
        name,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=parent_descriptor,
    )
    try:
        offset = 0
        while offset < len(payload):
            try:
                written = os.write(descriptor, payload[offset:])
            except InterruptedError:
                continue
            if written <= 0:
                raise ControlGridError("short immutable artifact write")
            offset += written
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
        observed = os.fstat(descriptor)
        _assert_unique_regular(
            observed,
            source=name,
            allow_empty=(len(payload) == 0),
            readonly=not bool(mode & 0o222),
        )
        if observed.st_size != len(payload):
            raise ControlGridError("immutable artifact size differs after write")
        return observed
    finally:
        os.close(descriptor)


def _rename_noreplace_at(
    source_parent: int,
    source_name: str,
    destination_parent: int,
    destination_name: str,
) -> None:
    """Perform one native dirfd-relative atomic no-replace rename."""

    for name in (source_name, destination_name):
        if name in {"", ".", ".."} or "/" in name:
            raise ControlGridError("atomic rename requires safe leaf names")
    libc = ctypes.CDLL(None, use_errno=True)
    source = os.fsencode(source_name)
    destination = os.fsencode(destination_name)
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
            source_parent, source, destination_parent, destination, 1
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
            source,
            destination_parent,
            destination,
            0x00000004,
        )
    else:
        raise ControlGridError(
            "platform lacks native dirfd-relative no-replace rename"
        )
    if result != 0:
        observed_errno = ctypes.get_errno()
        if observed_errno in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(
                observed_errno,
                os.strerror(observed_errno),
                destination_name,
            )
        raise OSError(
            observed_errno,
            os.strerror(observed_errno),
            destination_name,
        )


def _hide_exact_regular(
    *,
    parent_descriptor: int,
    canonical_name: str,
    staged_name: str,
    expected_identity: tuple[int, int],
) -> bool:
    """Move an exact canonical inode back to its private stage name."""

    try:
        observed = os.stat(
            canonical_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return False
    if (int(observed.st_dev), int(observed.st_ino)) != expected_identity:
        return False
    try:
        _rename_noreplace_at(
            parent_descriptor,
            canonical_name,
            parent_descriptor,
            staged_name,
        )
    except Exception:
        try:
            current = os.stat(
                canonical_name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return True
        if (int(current.st_dev), int(current.st_ino)) == expected_identity:
            raise ControlGridError(
                "failed publication left its exact canonical inode visible"
            )
        return False
    return True


def _atomic_bytes(
    path: Path | str,
    payload: bytes,
    *,
    mode: int = 0o444,
) -> tuple[int, int]:
    """Publish one immutable leaf without replacement or error-state residue."""

    absolute = _absolute(path)
    _safe_mkdir(absolute.parent)
    parent = _open_directory_absolute(absolute.parent)
    staged_name = f".{absolute.name}.stage-{uuid.uuid4().hex}"
    identity: tuple[int, int] | None = None
    try:
        written = _write_exclusive_at(
            parent, staged_name, payload, mode=mode
        )
        identity = (int(written.st_dev), int(written.st_ino))
        _rename_noreplace_at(
            parent, staged_name, parent, absolute.name
        )
        os.fsync(parent)
        observed_payload, observed = _read_unique_regular_at(
            parent,
            absolute.name,
            source=str(absolute),
            allow_empty=(len(payload) == 0),
            readonly=not bool(mode & 0o222),
        )
        if (
            (int(observed.st_dev), int(observed.st_ino)) != identity
            or stat.S_IMODE(observed.st_mode) != mode
            or observed_payload != payload
        ):
            raise ControlGridError(
                "published immutable artifact differs from its stage"
            )
        return identity
    except Exception:
        if identity is not None:
            _hide_exact_regular(
                parent_descriptor=parent,
                canonical_name=absolute.name,
                staged_name=staged_name,
                expected_identity=identity,
            )
        try:
            staged = os.stat(
                staged_name, dir_fd=parent, follow_symlinks=False
            )
        except FileNotFoundError:
            pass
        else:
            if identity is None or (
                int(staged.st_dev),
                int(staged.st_ino),
            ) == identity:
                os.unlink(staged_name, dir_fd=parent)
                try:
                    os.fsync(parent)
                except OSError:
                    pass
        raise
    finally:
        os.close(parent)


def _write_bytes_exclusive(
    path: Path, payload: bytes, *, mode: int = 0o444
) -> None:
    _atomic_bytes(path, payload, mode=mode)


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    _write_bytes_exclusive(path, _canonical_bytes(dict(payload)) + b"\n")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish one immutable audit object without replacing an authority."""

    _write_json_exclusive(path, payload)


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
        or len(np.unique(values)) != len(values)
    ):
        raise ControlGridError(
            "split rows must be nonempty, unique, nonnegative int64 vectors"
        )
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
    source_rows = np.sort(
        np.concatenate((train_rows, validation_rows))
    ).astype(np.int64, copy=False)
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


def _dataset_contracts() -> dict[str, Any]:
    return {
        dataset: {
            "subjects": list(DATASET_SUBJECTS[dataset]),
            "folds": list(DATASET_FOLDS[dataset]),
            "n_classes": int(dataset_spec(dataset).n_classes),
            "protocol": dataset_spec(dataset).protocol,
            "preprocessing": preprocessing_for_dataset(dataset),
        }
        for dataset in OPENED_DATASETS
    }


def _validate_registry_contract() -> None:
    from .model_registry import BENCHMARK_DATASETS, MODEL_REGISTRY

    if tuple(BENCHMARK_DATASETS) != OPENED_DATASETS:
        raise RegistryDriftError("opened dataset roster differs from model registry")
    indexed = {record.stable_id: record for record in MODEL_REGISTRY}
    observed_controls = tuple(
        record.stable_id for record in MODEL_REGISTRY if record.track == "control"
    )
    if observed_controls != CONTROLS:
        raise RegistryDriftError(
            f"registry control roster {observed_controls!r} differs from {CONTROLS!r}"
        )
    for stable_id in CONTROLS:
        record = indexed[stable_id]
        if (
            record.identity_level != "classical_control"
            or record.class_support != "binary_and_multiclass"
            or record.implementation != IMPLEMENTATIONS[stable_id]
        ):
            raise RegistryDriftError(
                f"{stable_id} identity/class/implementation contract drifted"
            )
        decisions = {
            decision.dataset: decision.eligible
            for decision in record.dataset_eligibility
        }
        if decisions != {dataset: True for dataset in OPENED_DATASETS}:
            raise RegistryDriftError(
                f"{stable_id} is not registry-eligible on every opened dataset"
            )


def _canonical_requirement_name(requirement: str) -> str:
    name = re.split(r"[\s<>=!~;\[]", requirement, maxsplit=1)[0]
    return re.sub(r"[-_.]+", "-", name.strip().lower())


def _validate_uv_project_contract(root: Path) -> None:
    """Fail before planning if the UV lock cannot reproduce the control stack."""

    pyproject_path = root / "pyproject.toml"
    lock_path = root / "uv.lock"
    try:
        pyproject = tomllib.loads(
            _read_unique_regular_bytes(pyproject_path).decode("utf-8")
        )
        lock = tomllib.loads(
            _read_unique_regular_bytes(lock_path).decode("utf-8")
        )
        dependencies = pyproject["project"]["dependencies"]
        packages = lock["package"]
    except (KeyError, TypeError, tomllib.TOMLDecodeError) as error:
        raise ControlGridError(f"UV project metadata cannot be parsed: {error}") from error
    except (OSError, UnicodeDecodeError, ControlGridError) as error:
        raise ControlGridError(
            "pyproject.toml and uv.lock must be unique regular files"
        ) from error
    if not isinstance(dependencies, list) or not all(
        isinstance(value, str) for value in dependencies
    ):
        raise ControlGridError("project.dependencies must be a string list")
    if not isinstance(packages, list):
        raise ControlGridError("uv.lock package table must be an array")
    direct = {_canonical_requirement_name(value) for value in dependencies}
    locked = {
        _canonical_requirement_name(str(value["name"]))
        for value in packages
        if isinstance(value, Mapping) and "name" in value
    }
    missing_direct = sorted(REQUIRED_DIRECT_DEPENDENCIES - direct)
    missing_locked = sorted(REQUIRED_DIRECT_DEPENDENCIES - locked)
    if missing_direct or missing_locked:
        raise ControlGridError(
            "UV project does not fully declare the control runtime; "
            f"missing_direct={missing_direct}, missing_locked={missing_locked}"
        )


def _source_identity() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    _validate_uv_project_contract(root)
    paths = {name: root / name for name in SOURCE_FILES}
    for name in ("pyproject.toml", "uv.lock"):
        paths[name] = root / name
    try:
        return {
            name: _sha256_file(path)
            for name, path in sorted(paths.items())
        }
    except (OSError, ControlGridError) as error:
        raise ControlGridError(
            "source closure contains an absent or unsafe file"
        ) from error


def _uv_version() -> str | None:
    try:
        return subprocess.run(
            ["uv", "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


def _environment_identity() -> dict[str, Any]:
    packages = sorted(
        (
            str(distribution.metadata.get("Name", "unknown")).lower(),
            str(distribution.version),
        )
        for distribution in importlib.metadata.distributions()
    )
    return {
        "python_version": sys.version,
        "python_executable": str(Path(sys.executable).resolve()),
        "python_prefix": str(Path(sys.prefix).resolve()),
        "base_prefix": str(Path(sys.base_prefix).resolve()),
        "platform": platform.platform(),
        "uv_version": _uv_version(),
        "packages": [[name, version] for name, version in packages],
        "packages_sha256": _sha256_bytes(_canonical_bytes(packages)),
    }


def _require_uv_virtual_environment() -> None:
    if Path(sys.prefix).resolve() == Path(sys.base_prefix).resolve():
        raise ControlGridError(
            "control-grid execution requires a virtual environment; use `uv venv` "
            "and `uv run` (or activate that environment)"
        )
    if _uv_version() is None:
        raise ControlGridError("UV is not available on PATH")
    configuration_path = Path(sys.prefix).resolve() / "pyvenv.cfg"
    try:
        configuration = _read_unique_regular_bytes(
            configuration_path
        ).decode("utf-8")
    except (OSError, UnicodeDecodeError, ControlGridError) as error:
        raise ControlGridError(
            f"UV virtual-environment marker is unavailable: {error}"
        ) from error
    fields = {
        key.strip().lower(): value.strip()
        for line in configuration.splitlines()
        if "=" in line
        for key, value in (line.split("=", 1),)
    }
    if "uv" not in fields or fields.get("include-system-site-packages", "").lower() != (
        "false"
    ):
        raise ControlGridError(
            "execution requires an isolated `uv venv` with system site packages disabled"
        )


def _cache_and_split_identity(
    cache_root: Path,
    contracts: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    from .data import load_subject_cache, split_indices

    caches: dict[str, Any] = {}
    splits: dict[str, Any] = {}
    for dataset, contract in contracts.items():
        for subject in contract["subjects"]:
            data = load_subject_cache(dataset, int(subject), cache_root=cache_root)
            key = _subject_key(dataset, int(subject))
            identity = copy.deepcopy(dict(data["identity"]))
            identity["trial_count"] = len(data["y"])
            identity["n_channels"] = int(data["x"].shape[1])
            caches[key] = identity
            for fold in contract["folds"]:
                train, validation, test = split_indices(
                    dataset,
                    data["y"],
                    data["sessions"],
                    data["runs"],
                    fold=int(fold),
                    subject=int(subject),
                )
                splits[_split_key(dataset, int(subject), int(fold))] = (
                    _one_split_identity(
                        dataset=dataset,
                        subject=int(subject),
                        fold=int(fold),
                        cache_array_sha256=str(
                            data["identity"]["array_sha256"]
                        ),
                        trial_count=len(data["y"]),
                        train_rows=train,
                        validation_rows=validation,
                        test_rows=test,
                    )
                )
    return caches, splits


def plan_sha256(plan: Mapping[str, Any]) -> str:
    payload = copy.deepcopy(dict(plan))
    payload.pop("plan_sha256", None)
    return _sha256_bytes(_canonical_bytes(payload))


def assemble_plan(
    *,
    dataset_contracts: Mapping[str, Any],
    controls: Sequence[str],
    cache_identity: Mapping[str, Any],
    split_identity: Mapping[str, Any],
    source_identity: Mapping[str, str],
    environment_identity: Mapping[str, Any],
    cpu_budget_threads: int = DEFAULT_CPU_BUDGET,
    threads_per_worker: int = DEFAULT_THREADS_PER_WORKER,
    minimum_free_gib: float = DEFAULT_MIN_FREE_GIB,
) -> dict[str, Any]:
    """Construct a canonical plan from explicit, already-validated identities."""

    control_values = tuple(str(value) for value in controls)
    if not control_values or len(set(control_values)) != len(control_values):
        raise ValueError("controls must be nonempty and unique")
    cpu_budget_threads = int(cpu_budget_threads)
    threads_per_worker = int(threads_per_worker)
    if cpu_budget_threads <= 0 or threads_per_worker <= 0:
        raise ValueError("CPU thread counts must be positive")
    slot_count = cpu_budget_threads // threads_per_worker
    if slot_count < 1:
        raise ValueError("CPU budget must permit at least one worker")
    minimum_free_gib = float(minimum_free_gib)
    if minimum_free_gib < DEFAULT_MIN_FREE_GIB:
        raise ValueError(
            f"minimum_free_gib may not be below {DEFAULT_MIN_FREE_GIB:.1f}"
        )

    contracts = copy.deepcopy(dict(dataset_contracts))
    expected_cache_keys: set[str] = set()
    expected_split_keys: set[str] = set()
    subject_fold_count = 0
    for dataset, contract in contracts.items():
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
        expected_cache_keys.update(_subject_key(dataset, value) for value in subjects)
        expected_split_keys.update(
            _split_key(dataset, subject, fold)
            for subject in subjects
            for fold in folds
        )
        subject_fold_count += len(subjects) * len(folds)
    caches = copy.deepcopy(dict(cache_identity))
    splits = copy.deepcopy(dict(split_identity))
    if set(caches) != expected_cache_keys:
        raise ValueError("cache identity keys differ from the subject grid")
    if set(splits) != expected_split_keys:
        raise ValueError("split identity keys differ from the subject/fold grid")

    for key, split in splits.items():
        partitions = split.get("partitions")
        if not isinstance(partitions, Mapping) or set(partitions) != {
            "train",
            "validation",
            "source",
            "test",
        }:
            raise ValueError(f"{key} has an invalid partition identity")
        counts = {
            name: int(partitions[name]["count"])
            for name in ("train", "validation", "source", "test")
        }
        if (
            counts["source"] != counts["train"] + counts["validation"]
            or counts["source"] + counts["test"] != int(split["trial_count"])
        ):
            raise ValueError(f"{key} split counts do not cover its cache")

    payload: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "purpose": "deterministic_classical_control_cross_dataset_predictions",
        "evidence_scope": "opened_development_datasets_only_not_confirmation",
        "confirmation_evidence": False,
        "score_blind": True,
        "dataset_order": list(contracts),
        "datasets": contracts,
        "controls": list(control_values),
        "runtime_keys": {
            stable_id: RUNTIME_KEYS.get(stable_id, stable_id)
            for stable_id in control_values
        },
        "hyperparameter_grid": {
            stable_id: [
                dict(candidate)
                for candidate in BASE_HYPERPARAMETER_GRID.get(stable_id, ())
            ]
            for stable_id in control_values
        },
        "cache_identity": caches,
        "split_identity": splits,
        "source_identity": copy.deepcopy(dict(source_identity)),
        "environment_identity": copy.deepcopy(dict(environment_identity)),
        "channel_scaling": copy.deepcopy(CHANNEL_SCALING),
        "execution_config": {
            "cpu_budget_threads": cpu_budget_threads,
            "threads_per_worker": threads_per_worker,
            "cpu_slot_count": slot_count,
            "minimum_free_gib": minimum_free_gib,
            "gpu_reservation": "none_cpu_only",
        },
        "job_order": ["dataset", "subject", "fold", "control"],
        "seed_policy": "one deterministic record; no seed field or seed aliases",
        "n_jobs": subject_fold_count * len(control_values),
        "output_contract": {
            "outcomes_present": False,
            "performance_scores_present": False,
            "prediction_arrays": ["rows", "probabilities"],
            "atomic_unit": "control/dataset/subject/fold",
            "selection": (
                "train-only fitting; prespecified hyperparameter selection by "
                "validation NLL; validation score not persisted"
            ),
            "final_refit": (
                "new estimator and source-only scaler fit on train+validation"
            ),
            "test_use": "one predict_proba call; no calibrate call",
        },
    }
    payload["plan_sha256"] = plan_sha256(payload)
    return payload


def _validate_plan_contract(plan: Mapping[str, Any]) -> None:
    _require_exact_keys(
        plan,
        {
            "schema",
            "purpose",
            "evidence_scope",
            "confirmation_evidence",
            "score_blind",
            "dataset_order",
            "datasets",
            "controls",
            "runtime_keys",
            "hyperparameter_grid",
            "cache_identity",
            "split_identity",
            "source_identity",
            "environment_identity",
            "channel_scaling",
            "execution_config",
            "job_order",
            "seed_policy",
            "n_jobs",
            "output_contract",
            "plan_sha256",
        },
        "plan",
    )
    try:
        config = plan["execution_config"]
        if not isinstance(config, Mapping):
            raise TypeError("execution_config is not an object")
        _require_exact_keys(
            config,
            {
                "cpu_budget_threads",
                "threads_per_worker",
                "cpu_slot_count",
                "minimum_free_gib",
                "gpu_reservation",
            },
            "plan execution_config",
        )
        expected = assemble_plan(
            dataset_contracts=plan["datasets"],
            controls=plan["controls"],
            cache_identity=plan["cache_identity"],
            split_identity=plan["split_identity"],
            source_identity=plan["source_identity"],
            environment_identity=plan["environment_identity"],
            cpu_budget_threads=int(config["cpu_budget_threads"]),
            threads_per_worker=int(config["threads_per_worker"]),
            minimum_free_gib=float(config["minimum_free_gib"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ControlGridError(f"plan contract is invalid: {error}") from error
    if dict(plan) != expected:
        raise ControlGridError("plan differs from the canonical control-grid contract")


def build_plan(
    *,
    cache_root: Path,
    cpu_budget_threads: int = DEFAULT_CPU_BUDGET,
    threads_per_worker: int = DEFAULT_THREADS_PER_WORKER,
    minimum_free_gib: float = DEFAULT_MIN_FREE_GIB,
) -> dict[str, Any]:
    _require_uv_virtual_environment()
    _validate_registry_contract()
    contracts = _dataset_contracts()
    caches, splits = _cache_and_split_identity(cache_root.resolve(), contracts)
    plan = assemble_plan(
        dataset_contracts=contracts,
        controls=CONTROLS,
        cache_identity=caches,
        split_identity=splits,
        source_identity=_source_identity(),
        environment_identity=_environment_identity(),
        cpu_budget_threads=cpu_budget_threads,
        threads_per_worker=threads_per_worker,
        minimum_free_gib=minimum_free_gib,
    )
    if plan["n_jobs"] != EXPECTED_JOB_COUNT:
        raise ControlGridError(
            f"production cardinality drift: {plan['n_jobs']} != {EXPECTED_JOB_COUNT}"
        )
    return plan


def _initialize_run_layout(run_root: Path) -> Path:
    root = _safe_mkdir(run_root)
    for name in (
        "records",
        "partial",
        "claims",
        "cpu_slots",
        "failures",
        "quarantine",
    ):
        _safe_mkdir(root / name)
    with _coordination_lock(
        root, exclusive=True, nonblocking=False
    ):
        pass
    return root


def write_or_validate_plan(run_root: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    run_root = _initialize_run_layout(run_root)
    plan_path = run_root / "plan.json"
    digest_path = run_root / "plan.sha256"
    value = copy.deepcopy(dict(expected))
    digest = plan_sha256(value)
    if value.get("plan_sha256") != digest:
        raise ControlGridError("expected plan has an invalid internal digest")
    if _path_exists(plan_path) and _path_exists(digest_path):
        observed = load_plan(run_root)
        if observed != value:
            raise ControlGridError("existing immutable plan differs from request")
        return observed
    _validate_plan_contract(value)
    if _path_exists(plan_path):
        observed = strict_load(plan_path)
        if (
            observed != value
            or observed.get("schema") != PLAN_SCHEMA
            or observed.get("plan_sha256") != digest
            or _read_unique_regular_bytes(plan_path)
            != _canonical_bytes(observed) + b"\n"
        ):
            raise ControlGridError(
                "orphaned plan.json differs from the requested immutable plan"
            )
        try:
            _write_bytes_exclusive(digest_path, f"{digest}\n".encode("ascii"))
        except FileExistsError:
            pass
        return load_plan(run_root)
    if _path_exists(digest_path):
        try:
            observed_digest = _read_unique_regular_bytes(
                digest_path
            ).decode("ascii").strip()
        except UnicodeDecodeError as error:
            raise ControlGridError(
                "orphaned plan.sha256 is not ASCII"
            ) from error
        if observed_digest != digest:
            raise ControlGridError(
                "orphaned plan.sha256 differs from the requested immutable plan"
            )
        try:
            _write_json_exclusive(plan_path, value)
        except FileExistsError:
            pass
        return load_plan(run_root)
    _write_json_exclusive(plan_path, value)
    try:
        _write_bytes_exclusive(digest_path, f"{digest}\n".encode("ascii"))
    except FileExistsError:
        pass
    return load_plan(run_root)


def load_plan(run_root: Path) -> dict[str, Any]:
    root = _absolute(run_root)
    payloads, _, _ = _snapshot_regular_files(
        root,
        ("plan.json", "plan.sha256"),
        child_mode=0o444,
    )
    plan = _strict_json_bytes(
        payloads["plan.json"], source=str(root / "plan.json")
    )
    # Canonical JSON sorts object keys, while dataset order is a prespecified
    # scientific identity carried by dataset_order. Restore that explicit
    # order in memory before rebuilding the canonical plan contract.
    dataset_order = plan.get("dataset_order")
    datasets = plan.get("datasets")
    if (
        isinstance(dataset_order, list)
        and all(isinstance(value, str) for value in dataset_order)
        and len(dataset_order) == len(set(dataset_order))
        and isinstance(datasets, Mapping)
        and set(dataset_order) == set(datasets)
    ):
        plan["datasets"] = {
            dataset: datasets[dataset] for dataset in dataset_order
        }
    digest = plan_sha256(plan)
    try:
        observed_digest = payloads["plan.sha256"].decode("ascii").strip()
    except UnicodeDecodeError as error:
        raise ControlGridError("immutable plan digest is not ASCII") from error
    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("plan_sha256") != digest
        or observed_digest != digest
        or payloads["plan.json"] != _canonical_bytes(plan) + b"\n"
        or payloads["plan.sha256"] != f"{digest}\n".encode("ascii")
    ):
        raise ControlGridError("immutable plan checksum/schema is invalid")
    _validate_plan_contract(plan)
    return plan


def iter_jobs(plan: Mapping[str, Any]) -> Iterator[Job]:
    count = 0
    seen: set[str] = set()
    controls = tuple(str(value) for value in plan["controls"])
    for dataset in plan["dataset_order"]:
        contract = plan["datasets"][dataset]
        for subject in contract["subjects"]:
            for fold in contract["folds"]:
                for control in controls:
                    job = Job(
                        dataset=str(dataset),
                        control=control,
                        subject=int(subject),
                        fold=int(fold),
                    )
                    if job.job_id in seen:
                        raise ControlGridError("job ID collision")
                    seen.add(job.job_id)
                    count += 1
                    yield job
    if count != int(plan["n_jobs"]):
        raise ControlGridError("iterated job count differs from immutable plan")


def verify_runtime_identity(plan: Mapping[str, Any], *, cache_root: Path) -> None:
    _require_uv_virtual_environment()
    _validate_registry_contract()
    if tuple(plan.get("dataset_order", ())) != OPENED_DATASETS:
        raise ControlGridError("plan does not contain the exact opened datasets")
    if tuple(plan.get("controls", ())) != CONTROLS:
        raise ControlGridError("plan does not contain the exact control roster")
    if plan.get("datasets") != _dataset_contracts():
        raise ControlGridError("dataset contracts differ from plan")
    if _source_identity() != plan.get("source_identity"):
        raise ControlGridError("source identities differ from plan")
    if _environment_identity() != plan.get("environment_identity"):
        raise ControlGridError("execution environment differs from plan")
    caches, splits = _cache_and_split_identity(
        cache_root.resolve(), plan["datasets"]
    )
    if caches != plan.get("cache_identity"):
        raise ControlGridError("cache identities differ from plan")
    if splits != plan.get("split_identity"):
        raise ControlGridError("split identities differ from plan")
    jobs = tuple(iter_jobs(plan))
    if len(jobs) != EXPECTED_JOB_COUNT:
        raise ControlGridError("formal job cardinality differs from 1,344")
    per_control = Counter(job.control for job in jobs)
    if per_control != Counter({control: FITS_PER_CONTROL for control in CONTROLS}):
        raise ControlGridError("per-control cardinality differs from 448")


def _record_directory(run_root: Path, job: Job) -> Path:
    return (
        run_root
        / "records"
        / job.dataset
        / job.control.replace(".", "_")
        / f"s{job.subject:03d}"
        / f"f{job.fold:02d}"
    )


def _claim_path(run_root: Path, job: Job) -> Path:
    return run_root / "claims" / f"{job.job_id}.json"


def _quarantine_path(
    run_root: Path,
    category: str,
    basename: str,
) -> Path:
    if re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", category) is None:
        raise ControlGridError("invalid quarantine category")
    root = _safe_mkdir(
        _absolute(run_root) / "quarantine" / category
    )
    return root / f"{basename}-{uuid.uuid4().hex}"


def _quarantine_leaf(
    run_root: Path,
    source: Path,
    *,
    category: str,
    expected_identity: tuple[int, int],
    expected_payload: bytes | None = None,
) -> Path:
    """Hide one exact inode before permission recovery or cross-parent work."""

    source = _absolute(source)
    source_parent = _open_directory_absolute(source.parent)
    hidden_name = f".{source.name}.quarantine-{uuid.uuid4().hex}"
    hidden_path = source.parent / hidden_name
    source_descriptor = -1
    destination_parent = -1
    destination: Path | None = None
    original_mode: int | None = None
    try:
        observed = os.stat(
            source.name,
            dir_fd=source_parent,
            follow_symlinks=False,
        )
        identity = (int(observed.st_dev), int(observed.st_ino))
        if identity != expected_identity:
            raise ControlGridError(
                "quarantine source is not the expected inode"
            )
        if not (
            stat.S_ISREG(observed.st_mode)
            or stat.S_ISDIR(observed.st_mode)
        ):
            raise ControlGridError(
                "quarantine source is not a regular file or directory"
            )
        original_mode = stat.S_IMODE(observed.st_mode)
        flags = (
            _directory_flags()
            if stat.S_ISDIR(observed.st_mode)
            else (
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0)
            )
        )
        source_descriptor = os.open(
            source.name, flags, dir_fd=source_parent
        )
        held = os.fstat(source_descriptor)
        if (int(held.st_dev), int(held.st_ino)) != identity:
            raise ControlGridError(
                "quarantine source changed while binding its inode"
            )
        _rename_noreplace_at(
            source_parent,
            source.name,
            source_parent,
            hidden_name,
        )
        hidden = os.stat(
            hidden_name,
            dir_fd=source_parent,
            follow_symlinks=False,
        )
        if (int(hidden.st_dev), int(hidden.st_ino)) != identity:
            raise ControlGridError(
                "hidden quarantine artifact changed during authority removal"
            )
        os.fsync(source_parent)
        if expected_payload is not None:
            if not stat.S_ISREG(held.st_mode):
                raise ControlGridError(
                    "payload-bound quarantine source is not a regular file"
                )
            if _read_descriptor_bytes(
                source_descriptor, source=str(hidden_path)
            ) != expected_payload:
                raise ControlGridError(
                    "hidden quarantine artifact differs from expected bytes"
                )
        if stat.S_ISDIR(held.st_mode) and not (
            held.st_mode & stat.S_IWUSR
        ):
            os.fchmod(source_descriptor, 0o700)
            os.fsync(source_descriptor)
        destination = _quarantine_path(
            run_root, category, source.name
        )
        destination_parent = _open_directory_absolute(
            destination.parent
        )
        _rename_noreplace_at(
            source_parent,
            hidden_name,
            destination_parent,
            destination.name,
        )
        final = os.stat(
            destination.name,
            dir_fd=destination_parent,
            follow_symlinks=False,
        )
        if (int(final.st_dev), int(final.st_ino)) != identity:
            raise ControlGridError(
                "quarantined artifact changed during rename"
            )
        if stat.S_IMODE(os.fstat(source_descriptor).st_mode) != original_mode:
            os.fchmod(source_descriptor, original_mode)
            os.fsync(source_descriptor)
        os.fsync(destination_parent)
        os.fsync(source_parent)
    finally:
        if source_descriptor >= 0:
            os.close(source_descriptor)
        os.close(source_parent)
        if destination_parent >= 0:
            os.close(destination_parent)
    if destination is None:
        raise ControlGridError(
            f"quarantine destination was not established for {source}"
        )
    return destination


def _open_locked_leaf(
    path: Path,
    *,
    nonblocking: bool,
) -> tuple[int, os.stat_result, bytes]:
    absolute = _absolute(path)
    parent = _open_directory_absolute(absolute.parent)
    descriptor = -1
    try:
        before = os.stat(
            absolute.name, dir_fd=parent, follow_symlinks=False
        )
        _assert_unique_regular(
            before, source=str(absolute), readonly=True
        )
        descriptor = os.open(
            absolute.name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent,
        )
        opened = os.fstat(descriptor)
        if _stat_fingerprint(opened) != _stat_fingerprint(before):
            raise ControlGridError(f"{absolute} changed while opening")
        operation = fcntl.LOCK_EX
        if nonblocking:
            operation |= fcntl.LOCK_NB
        fcntl.flock(descriptor, operation)
        payload = _read_descriptor_bytes(
            descriptor, source=str(absolute)
        )
        current = os.stat(
            absolute.name, dir_fd=parent, follow_symlinks=False
        )
        if _stat_fingerprint(current) != _stat_fingerprint(opened):
            raise ControlGridError(f"{absolute} changed while locking")
        return descriptor, opened, payload
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    finally:
        os.close(parent)


def _assert_locked_authority(
    path: Path,
    descriptor: int,
    *,
    expected_identity: tuple[int, int],
    expected_payload: bytes,
) -> None:
    held = os.fstat(descriptor)
    if (
        (int(held.st_dev), int(held.st_ino)) != expected_identity
        or not stat.S_ISREG(held.st_mode)
        or held.st_nlink != 1
        or stat.S_IMODE(held.st_mode) != 0o444
        or _read_descriptor_bytes(descriptor, source=str(path))
        != expected_payload
    ):
        raise ControlGridError(f"{path} descriptor authority changed")
    current = _anchored_lstat(path)
    if (
        (int(current.st_dev), int(current.st_ino)) != expected_identity
        or _stat_fingerprint(current) != _stat_fingerprint(held)
    ):
        raise ControlGridError(f"{path} path authority changed")


def _publish_locked_json(
    run_root: Path,
    path: Path,
    value: Mapping[str, Any],
    *,
    failure_category: str,
) -> tuple[int, tuple[int, int], bytes]:
    """Publish and retain an exclusively locked immutable JSON authority."""

    absolute = _absolute(path)
    _safe_mkdir(absolute.parent)
    parent = _open_directory_absolute(absolute.parent)
    payload = _canonical_bytes(dict(value)) + b"\n"
    staged_name = f".{absolute.name}.stage-{uuid.uuid4().hex}"
    identity: tuple[int, int] | None = None
    descriptor = -1
    try:
        written = _write_exclusive_at(
            parent, staged_name, payload, mode=0o444
        )
        identity = (int(written.st_dev), int(written.st_ino))
        descriptor = os.open(
            staged_name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent,
        )
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        held = os.fstat(descriptor)
        if (
            (int(held.st_dev), int(held.st_ino)) != identity
            or _read_descriptor_bytes(
                descriptor, source=str(absolute.parent / staged_name)
            )
            != payload
        ):
            raise ControlGridError(
                "locked JSON stage differs from its intended authority"
            )
        _rename_noreplace_at(
            parent, staged_name, parent, absolute.name
        )
        os.fsync(parent)
        _assert_locked_authority(
            absolute,
            descriptor,
            expected_identity=identity,
            expected_payload=payload,
        )
        return descriptor, identity, payload
    except Exception:
        if identity is not None:
            for candidate in (
                absolute,
                absolute.parent / staged_name,
            ):
                if not _path_exists(candidate):
                    continue
                observed = _anchored_lstat(candidate)
                if (
                    int(observed.st_dev),
                    int(observed.st_ino),
                ) != identity:
                    continue
                _quarantine_leaf(
                    run_root,
                    candidate,
                    category=failure_category,
                    expected_identity=identity,
                    expected_payload=payload,
                )
        if descriptor >= 0:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
        raise
    finally:
        os.close(parent)


def _coordination_lock_path(run_root: Path) -> Path:
    return _absolute(run_root) / COORDINATION_LOCK_NAME


@contextlib.contextmanager
def _coordination_lock(
    run_root: Path,
    *,
    exclusive: bool,
    nonblocking: bool = False,
) -> Iterator[int]:
    root = _safe_mkdir(run_root)
    path = _coordination_lock_path(root)
    payload = b"ieee-mi-control-grid-coordination-v1\n"
    if not _path_exists(path):
        try:
            _atomic_bytes(path, payload, mode=0o444)
        except FileExistsError:
            pass
    parent = _open_directory_absolute(path.parent)
    descriptor = -1
    raised: BaseException | None = None
    try:
        before = os.stat(
            path.name, dir_fd=parent, follow_symlinks=False
        )
        _assert_unique_regular(
            before, source=str(path), readonly=True
        )
        descriptor = os.open(
            path.name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent,
        )
        operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        if nonblocking:
            operation |= fcntl.LOCK_NB
        fcntl.flock(descriptor, operation)
        identity = (int(before.st_dev), int(before.st_ino))
        _assert_locked_authority(
            path,
            descriptor,
            expected_identity=identity,
            expected_payload=payload,
        )
        try:
            yield descriptor
        except BaseException as error:
            raised = error
            raise
        finally:
            try:
                _assert_locked_authority(
                    path,
                    descriptor,
                    expected_identity=identity,
                    expected_payload=payload,
                )
            except BaseException:
                if raised is None:
                    raise
    finally:
        if descriptor >= 0:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
        os.close(parent)


@contextlib.contextmanager
def analysis_fence(run_root: Path) -> Iterator[int]:
    """Exclude all claim/recovery/publication mutations while labels are open."""

    with _coordination_lock(
        run_root, exclusive=True, nonblocking=False
    ) as descriptor:
        yield descriptor


def _process_start_marker(pid: int) -> str | None:
    proc_stat = Path(f"/proc/{pid}/stat")
    try:
        fields = proc_stat.read_text(encoding="utf-8").split()
        if len(fields) > 21:
            return f"proc:{fields[21]}"
    except (OSError, ValueError):
        pass
    try:
        value = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(pid)],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        return f"ps:{value}" if value else None
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


def _process_identity() -> dict[str, Any]:
    return {
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "start_marker": _process_start_marker(os.getpid()),
    }


def _owner_is_live(owner: Mapping[str, Any]) -> bool:
    host = owner.get("host")
    if not isinstance(host, str) or not host:
        return False
    if host != socket.gethostname():
        return True
    try:
        if isinstance(owner.get("pid"), bool):
            return False
        pid = int(owner["pid"])
    except (KeyError, TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    expected = owner.get("start_marker")
    observed = _process_start_marker(pid)
    # A transient /proc or ps failure is not permission to delete another
    # process's claim.  Ambiguous live owners fail closed; an administrator may
    # quarantine them manually after resolving process ownership.
    if expected is None or observed is None:
        return True
    return observed == expected


def probe_disk(path: Path, *, minimum_free_gib: float) -> DiskStatus:
    resolved = path.resolve()
    try:
        stat = os.statvfs(resolved)
        free_bytes = int(stat.f_bavail) * int(stat.f_frsize)
        total_bytes = int(stat.f_blocks) * int(stat.f_frsize)
        free_gib = free_bytes / float(1024**3)
        safe = free_gib >= minimum_free_gib
        return DiskStatus(
            safe=safe,
            path=str(resolved),
            free_bytes=free_bytes,
            total_bytes=total_bytes,
            free_gib=free_gib,
            minimum_free_gib=float(minimum_free_gib),
            reason=(
                f"{free_gib:.2f} GiB free "
                f"{'>=' if safe else '<'} {minimum_free_gib:.2f} GiB"
            ),
        )
    except OSError as error:
        return DiskStatus(
            safe=False,
            path=str(resolved),
            free_bytes=None,
            total_bytes=None,
            free_gib=None,
            minimum_free_gib=float(minimum_free_gib),
            reason=f"statvfs failed closed: {error}",
        )


def _cpu_slot_path(run_root: Path, index: int) -> Path:
    return run_root / "cpu_slots" / f"slot-{index:03d}.json"


def acquire_cpu_slot(run_root: Path, plan: Mapping[str, Any]) -> CPUSlot:
    config = plan["execution_config"]
    owner = _process_identity()
    threads = int(config["threads_per_worker"])
    run_root = _safe_mkdir(run_root)
    _safe_mkdir(run_root / "cpu_slots")
    with _coordination_lock(
        run_root, exclusive=False, nonblocking=True
    ):
        for index in range(int(config["cpu_slot_count"])):
            path = _cpu_slot_path(run_root, index)
            if _path_exists(path):
                try:
                    descriptor, observed_stat, observed_payload = (
                        _open_locked_leaf(path, nonblocking=True)
                    )
                except BlockingIOError:
                    continue
                try:
                    observed = _strict_json_bytes(
                        observed_payload, source=str(path)
                    )
                    if _owner_is_live(observed.get("owner", {})):
                        continue
                    _require_exact_keys(
                        observed,
                        {
                            "schema",
                            "created_at",
                            "plan_sha256",
                            "slot",
                            "threads",
                            "nonce",
                            "owner",
                        },
                        "stale CPU slot",
                    )
                    if (
                        observed["schema"] != CPU_SLOT_SCHEMA
                        or observed["plan_sha256"]
                        != plan["plan_sha256"]
                        or observed["slot"] != index
                        or observed["threads"] != threads
                    ):
                        raise ControlGridError(
                            "stale CPU slot differs from the current plan"
                        )
                    _quarantine_leaf(
                        run_root,
                        path,
                        category="stale_cpu_slots",
                        expected_identity=(
                            int(observed_stat.st_dev),
                            int(observed_stat.st_ino),
                        ),
                        expected_payload=observed_payload,
                    )
                finally:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                    os.close(descriptor)
            nonce = uuid.uuid4().hex
            value = {
                "schema": CPU_SLOT_SCHEMA,
                "created_at": _utc_now(),
                "plan_sha256": plan["plan_sha256"],
                "slot": index,
                "threads": threads,
                "nonce": nonce,
                "owner": owner,
            }
            try:
                descriptor, identity, payload = _publish_locked_json(
                    run_root,
                    path,
                    value,
                    failure_category="cpu_slot_authority_loss",
                )
            except FileExistsError:
                continue
            return CPUSlot(
                index=index,
                path=path,
                nonce=nonce,
                owner=owner,
                threads=threads,
                descriptor=descriptor,
                st_dev=identity[0],
                st_ino=identity[1],
                payload=payload,
            )
    raise CPUUnavailable("every cooperative CPU slot is occupied")


def _validate_cpu_slot(
    run_root: Path, plan: Mapping[str, Any], slot: CPUSlot
) -> dict[str, Any]:
    config = plan["execution_config"]
    expected_path = _cpu_slot_path(run_root, slot.index)
    if (
        _absolute(slot.path) != _absolute(expected_path)
        or slot.descriptor < 0
        or slot.st_dev < 0
        or slot.st_ino < 0
        or not slot.payload
    ):
        raise ControlGridError("CPU slot path is absent, aliased, or outside the plan")
    _assert_locked_authority(
        slot.path,
        slot.descriptor,
        expected_identity=(slot.st_dev, slot.st_ino),
        expected_payload=slot.payload,
    )
    observed = _strict_json_bytes(
        slot.payload, source=str(slot.path)
    )
    _require_exact_keys(
        observed,
        {
            "schema",
            "created_at",
            "plan_sha256",
            "slot",
            "threads",
            "nonce",
            "owner",
        },
        "CPU slot",
    )
    expected = {
        "schema": CPU_SLOT_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "slot": slot.index,
        "threads": int(config["threads_per_worker"]),
        "nonce": slot.nonce,
        "owner": dict(slot.owner),
    }
    for key, value in expected.items():
        if observed.get(key) != value:
            raise ControlGridError(f"CPU slot has invalid {key}")
    if not 0 <= slot.index < int(config["cpu_slot_count"]):
        raise ControlGridError("CPU slot index is outside the immutable budget")
    if slot.threads != int(config["threads_per_worker"]):
        raise ControlGridError("CPU slot thread count differs from the plan")
    if dict(slot.owner) != _process_identity():
        raise ControlGridError("CPU slot is not owned by the calling worker")
    return observed


def release_cpu_slot(slot: CPUSlot) -> None:
    try:
        with _coordination_lock(
            slot.path.parents[1],
            exclusive=False,
            nonblocking=False,
        ):
            _assert_locked_authority(
                slot.path,
                slot.descriptor,
                expected_identity=(slot.st_dev, slot.st_ino),
                expected_payload=slot.payload,
            )
            observed = _strict_json_bytes(
                slot.payload, source=str(slot.path)
            )
            if (
                observed.get("nonce") != slot.nonce
                or observed.get("owner") != dict(slot.owner)
            ):
                raise ControlGridError(
                    "refusing to release a CPU slot owned by another worker"
                )
            parent = _open_directory_absolute(slot.path.parent)
            try:
                current = os.stat(
                    slot.path.name,
                    dir_fd=parent,
                    follow_symlinks=False,
                )
                if (int(current.st_dev), int(current.st_ino)) != (
                    slot.st_dev,
                    slot.st_ino,
                ):
                    raise ControlGridError(
                        "refusing to release a replaced CPU slot"
                    )
                os.unlink(slot.path.name, dir_fd=parent)
                os.fsync(parent)
            finally:
                os.close(parent)
    finally:
        if slot.descriptor >= 0:
            try:
                fcntl.flock(slot.descriptor, fcntl.LOCK_UN)
            finally:
                os.close(slot.descriptor)


def _recover_job_partials(
    run_root: Path,
    job: Job,
) -> tuple[Path, ...]:
    """Quarantine exact abandoned stages only after claim authority is absent."""

    recovered: list[Path] = []
    old_partial_root = _safe_mkdir(_absolute(run_root) / "partial")
    old_prefix = f"{job.job_id}."
    old_partial_descriptor = _open_directory_absolute(old_partial_root)
    try:
        old_partial_names = sorted(os.listdir(old_partial_descriptor))
    finally:
        os.close(old_partial_descriptor)
    for name in old_partial_names:
        if not name.startswith(old_prefix):
            continue
        if (
            re.fullmatch(
                rf"{re.escape(job.job_id)}\.[0-9a-f]{{32}}",
                name,
            )
            is None
        ):
            raise ControlGridError(
                f"unrecognized partial for {job.job_id}: {name}"
            )
        path = old_partial_root / name
        observed = _anchored_lstat(path)
        if not stat.S_ISDIR(observed.st_mode):
            raise ControlGridError("job partial is not a directory")
        recovered.append(
            _quarantine_leaf(
                run_root,
                path,
                category="abandoned_job_stages",
                expected_identity=(
                    int(observed.st_dev),
                    int(observed.st_ino),
                ),
            )
        )
    final = _record_directory(run_root, job)
    parent = _safe_mkdir(final.parent)
    prefix = f".{final.name}.partial-{job.job_id}."
    directory = _open_directory_absolute(parent)
    try:
        names = sorted(os.listdir(directory))
    finally:
        os.close(directory)
    for name in names:
        if not name.startswith(prefix):
            continue
        if (
            re.fullmatch(
                rf"\.{re.escape(final.name)}\.partial-"
                rf"{re.escape(job.job_id)}\.[0-9a-f]{{32}}\.[0-9a-f]{{32}}",
                name,
            )
            is None
        ):
            raise ControlGridError(
                f"unrecognized publication stage for {job.job_id}: {name}"
            )
        path = parent / name
        observed = _anchored_lstat(path)
        if not stat.S_ISDIR(observed.st_mode):
            raise ControlGridError("publication stage is not a directory")
        recovered.append(
            _quarantine_leaf(
                run_root,
                path,
                category="abandoned_job_stages",
                expected_identity=(
                    int(observed.st_dev),
                    int(observed.st_ino),
                ),
            )
        )
    return tuple(recovered)


def acquire_claim(
    run_root: Path,
    plan: Mapping[str, Any],
    job: Job,
    *,
    cpu_slot: CPUSlot,
) -> Claim:
    _validate_job_identity(plan, job)
    run_root = _safe_mkdir(run_root)
    _safe_mkdir(run_root / "claims")
    with _coordination_lock(
        run_root, exclusive=False, nonblocking=True
    ):
        _validate_cpu_slot(run_root, plan, cpu_slot)
        output = _record_directory(run_root, job)
        if _path_exists(output):
            validate_completion(run_root, plan, job)
            raise ClaimUnavailable(f"{job.job_id} is already complete")
        path = _claim_path(run_root, job)
        if _path_exists(path):
            try:
                descriptor, claim_stat, claim_payload = _open_locked_leaf(
                    path, nonblocking=True
                )
            except BlockingIOError as error:
                raise ClaimUnavailable(
                    f"{job.job_id} has a live claim"
                ) from error
            try:
                observed = _strict_json_bytes(
                    claim_payload, source=str(path)
                )
                if _owner_is_live(observed.get("owner", {})):
                    raise ClaimUnavailable(
                        f"{job.job_id} has a live or ambiguous claim"
                    )
                stale = Claim(
                    job=job,
                    path=path,
                    nonce=str(observed.get("nonce", "")),
                    owner=copy.deepcopy(observed.get("owner", {})),
                    resource_guard=copy.deepcopy(
                        observed.get("resource_guard", {})
                    ),
                    descriptor=descriptor,
                    st_dev=int(claim_stat.st_dev),
                    st_ino=int(claim_stat.st_ino),
                    payload=claim_payload,
                )
                _validate_claim_payload(run_root, plan, stale)
                _quarantine_leaf(
                    run_root,
                    path,
                    category="stale_claims",
                    expected_identity=(
                        int(claim_stat.st_dev),
                        int(claim_stat.st_ino),
                    ),
                    expected_payload=claim_payload,
                )
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)
        _recover_job_partials(run_root, job)
        disk = probe_disk(
            run_root,
            minimum_free_gib=float(
                plan["execution_config"]["minimum_free_gib"]
            ),
        )
        if not disk.safe:
            raise DiskUnavailable(disk.reason)
        nonce = uuid.uuid4().hex
        owner = _process_identity()
        guard = {
            "disk": disk.as_dict(),
            "cpu": {
                "slot": cpu_slot.index,
                "threads": cpu_slot.threads,
                "slot_nonce": cpu_slot.nonce,
                "cpu_budget_threads": int(
                    plan["execution_config"]["cpu_budget_threads"]
                ),
            },
            "gpu_reservation": "none_cpu_only",
        }
        value = {
            "schema": CLAIM_SCHEMA,
            "created_at": _utc_now(),
            "plan_sha256": plan["plan_sha256"],
            "job_id": job.job_id,
            "job": job.identity(),
            "nonce": nonce,
            "owner": owner,
            "resource_guard": guard,
        }
        try:
            descriptor, identity, payload = _publish_locked_json(
                run_root,
                path,
                value,
                failure_category="claim_authority_loss",
            )
        except FileExistsError as error:
            raise ClaimUnavailable(
                f"{job.job_id} lost its claim race"
            ) from error
        return Claim(
            job=job,
            path=path,
            nonce=nonce,
            owner=owner,
            resource_guard=guard,
            descriptor=descriptor,
            st_dev=identity[0],
            st_ino=identity[1],
            payload=payload,
        )


def release_claim(claim: Claim) -> None:
    run_root = claim.path.parents[1]
    try:
        with _coordination_lock(
            run_root, exclusive=False, nonblocking=False
        ):
            _assert_locked_authority(
                claim.path,
                claim.descriptor,
                expected_identity=(claim.st_dev, claim.st_ino),
                expected_payload=claim.payload,
            )
            observed = _strict_json_bytes(
                claim.payload, source=str(claim.path)
            )
            if (
                observed.get("nonce") != claim.nonce
                or observed.get("owner") != dict(claim.owner)
            ):
                raise ControlGridError(
                    "refusing to release another worker's claim"
                )
            parent = _open_directory_absolute(claim.path.parent)
            try:
                current = os.stat(
                    claim.path.name,
                    dir_fd=parent,
                    follow_symlinks=False,
                )
                if (int(current.st_dev), int(current.st_ino)) != (
                    claim.st_dev,
                    claim.st_ino,
                ):
                    raise ControlGridError(
                        "refusing to release a replaced claim"
                    )
                os.unlink(claim.path.name, dir_fd=parent)
                os.fsync(parent)
            finally:
                os.close(parent)
    finally:
        if claim.descriptor >= 0:
            try:
                fcntl.flock(claim.descriptor, fcntl.LOCK_UN)
            finally:
                os.close(claim.descriptor)


def _validate_resource_guard(
    run_root: Path, plan: Mapping[str, Any], guard: Mapping[str, Any]
) -> None:
    _require_exact_keys(
        guard, {"disk", "cpu", "gpu_reservation"}, "claim resource guard"
    )
    disk = guard.get("disk")
    cpu = guard.get("cpu")
    if not isinstance(disk, Mapping) or not isinstance(cpu, Mapping):
        raise ControlGridError("claim resource guard disk/CPU records are absent")
    _require_exact_keys(
        disk,
        {
            "safe",
            "path",
            "free_bytes",
            "total_bytes",
            "free_gib",
            "minimum_free_gib",
            "reason",
        },
        "claim disk guard",
    )
    _require_exact_keys(
        cpu,
        {
            "slot",
            "threads",
            "slot_nonce",
            "cpu_budget_threads",
        },
        "claim CPU guard",
    )
    config = plan["execution_config"]
    free_gib = disk.get("free_gib")
    minimum = float(config["minimum_free_gib"])
    try:
        observed_minimum = float(disk.get("minimum_free_gib"))
    except (TypeError, ValueError) as error:
        raise ControlGridError("claim disk guard has an invalid floor") from error
    free_bytes = disk.get("free_bytes")
    total_bytes = disk.get("total_bytes")
    if (
        disk.get("safe") is not True
        or disk.get("path") != str(run_root.resolve())
        or not isinstance(free_gib, (int, float))
        or isinstance(free_gib, bool)
        or not np.isfinite(float(free_gib))
        or float(free_gib) < minimum
        or observed_minimum != minimum
        or not isinstance(free_bytes, int)
        or isinstance(free_bytes, bool)
        or free_bytes < 0
        or not isinstance(total_bytes, int)
        or isinstance(total_bytes, bool)
        or total_bytes < free_bytes
        or not isinstance(disk.get("reason"), str)
    ):
        raise ControlGridError("claim disk guard differs from the frozen floor")
    slot = cpu.get("slot")
    if (
        not isinstance(slot, int)
        or isinstance(slot, bool)
        or not 0 <= slot < int(config["cpu_slot_count"])
        or cpu.get("threads") != int(config["threads_per_worker"])
        or cpu.get("cpu_budget_threads") != int(config["cpu_budget_threads"])
        or not isinstance(cpu.get("slot_nonce"), str)
        or not cpu["slot_nonce"]
        or guard.get("gpu_reservation") != "none_cpu_only"
    ):
        raise ControlGridError("claim CPU/GPU guard differs from the plan")


def _validate_claim_payload(
    run_root: Path, plan: Mapping[str, Any], claim: Claim
) -> dict[str, Any]:
    if (
        _absolute(claim.path)
        != _absolute(_claim_path(run_root, claim.job))
        or claim.descriptor < 0
        or claim.st_dev < 0
        or claim.st_ino < 0
        or not claim.payload
    ):
        raise ControlGridError("job claim path is aliased or outside the plan")
    _assert_locked_authority(
        claim.path,
        claim.descriptor,
        expected_identity=(claim.st_dev, claim.st_ino),
        expected_payload=claim.payload,
    )
    observed = _strict_json_bytes(
        claim.payload, source=str(claim.path)
    )
    _require_exact_keys(
        observed,
        {
            "schema",
            "created_at",
            "plan_sha256",
            "job_id",
            "job",
            "nonce",
            "owner",
            "resource_guard",
        },
        "job claim",
    )
    expected = {
        "schema": CLAIM_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "job_id": claim.job.job_id,
        "job": claim.job.identity(),
        "nonce": claim.nonce,
        "owner": dict(claim.owner),
        "resource_guard": dict(claim.resource_guard),
    }
    for key, value in expected.items():
        if observed.get(key) != value:
            raise ControlGridError(f"job claim has invalid {key}")
    if not isinstance(observed.get("created_at"), str):
        raise ControlGridError("job claim has invalid created_at")
    owner = observed["owner"]
    if not isinstance(owner, Mapping):
        raise ControlGridError("job claim owner is absent")
    _require_exact_keys(owner, {"host", "pid", "start_marker"}, "job claim owner")
    _validate_resource_guard(run_root, plan, observed["resource_guard"])
    return observed


def _recursive_forbidden_keys(value: Any, prefix: str = "") -> list[str]:
    violations: list[str] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).strip().lower()
            location = f"{prefix}.{key}" if prefix else str(key)
            if normalized in FORBIDDEN_SCORE_BLIND_KEYS:
                violations.append(location)
            violations.extend(_recursive_forbidden_keys(nested, location))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            violations.extend(
                _recursive_forbidden_keys(nested, f"{prefix}[{index}]")
            )
    return violations


def _planned_split(plan: Mapping[str, Any], job: Job) -> Mapping[str, Any]:
    return plan["split_identity"][_split_key(job.dataset, job.subject, job.fold)]


def _planned_cache(plan: Mapping[str, Any], job: Job) -> Mapping[str, Any]:
    return plan["cache_identity"][_subject_key(job.dataset, job.subject)]


def _validate_job_identity(plan: Mapping[str, Any], job: Job) -> None:
    if (
        job.dataset not in plan["datasets"]
        or job.control not in plan["controls"]
        or job.subject not in plan["datasets"][job.dataset]["subjects"]
        or job.fold not in plan["datasets"][job.dataset]["folds"]
    ):
        raise ControlGridError("job identity is outside the immutable plan")


def _admissible_grid(control: str, n_channels: int) -> tuple[dict[str, Any], ...]:
    candidates = tuple(
        dict(value) for value in BASE_HYPERPARAMETER_GRID[control]
    )
    if control == "control.ea_fbcsp":
        candidates = tuple(
            value
            for value in candidates
            if int(value["n_components"]) <= int(n_channels)
        )
    if not candidates:
        raise EstimatorCapabilityError(
            f"{control} has no admissible candidate for {n_channels} channels"
        )
    return candidates


_THREADPOOL_LIMITER: Any | None = None


def _apply_cpu_limits(threads: int) -> None:
    global _THREADPOOL_LIMITER
    value = str(int(threads))
    for variable in THREAD_ENVIRONMENT_VARIABLES:
        os.environ[variable] = value
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
    try:
        from threadpoolctl import threadpool_limits

        _THREADPOOL_LIMITER = threadpool_limits(limits=int(threads))
    except ImportError as error:
        raise ControlGridError(
            "threadpoolctl is required for the cooperative CPU cap"
        ) from error
    try:
        import torch

        torch.set_num_threads(int(threads))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            if torch.get_num_interop_threads() != 1:
                raise
    except ImportError:
        # Riemann and EA-FBCSP can run without Torch; TangentAnchor will fail
        # explicitly at estimator import rather than silently changing method.
        pass


def _new_estimator(control: str, parameters: Mapping[str, Any]) -> Any:
    if control == "control.riemann":
        from deepnet.baselines import RiemannianTangentLogistic

        return RiemannianTangentLogistic(
            c=float(parameters["c"]), max_iter=3000
        )
    if control == "control.tangent_anchor":
        from deepnet.tangent_anchor import TangentAnchorClassifier

        return TangentAnchorClassifier(
            C=float(parameters["C"]), max_iter=3000
        )
    if control == "control.ea_fbcsp":
        from deepnet.baselines import EAFilterBankCSP

        return EAFilterBankCSP(
            n_components=int(parameters["n_components"])
        )
    raise ValueError(f"unknown control {control!r}")


def _fit_estimator(
    control: str, estimator: Any, features: np.ndarray, outcomes: np.ndarray
) -> Any:
    if control == "control.ea_fbcsp":
        import mne

        with mne.use_log_level("ERROR"):
            return estimator.fit(features, outcomes)
    return estimator.fit(features, outcomes)


def _estimator_classes(estimator: Any) -> np.ndarray:
    if hasattr(estimator, "classes_"):
        return np.asarray(estimator.classes_, dtype=np.int64)
    if hasattr(estimator, "model_") and hasattr(estimator.model_, "classes_"):
        return np.asarray(estimator.model_.classes_, dtype=np.int64)
    raise EstimatorCapabilityError("estimator does not expose fitted class order")


def _valid_probabilities(
    estimator: Any,
    features: np.ndarray,
    *,
    n_classes: int,
) -> np.ndarray:
    expected_classes = np.arange(n_classes, dtype=np.int64)
    observed_classes = _estimator_classes(estimator)
    if not np.array_equal(observed_classes, expected_classes):
        raise EstimatorCapabilityError(
            f"estimator classes {observed_classes.tolist()} differ from "
            f"{expected_classes.tolist()}"
        )
    values = np.asarray(estimator.predict_proba(features), dtype=np.float64)
    if (
        values.shape != (len(features), n_classes)
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(values > 1.0)
        or not np.allclose(values.sum(axis=1), 1.0, rtol=0.0, atol=1e-8)
    ):
        raise EstimatorCapabilityError(
            f"invalid {n_classes}-class probability output {values.shape}"
        )
    return np.ascontiguousarray(values, dtype=np.float64)


def _estimator_state_sha256(estimator: Any) -> str:
    return _sha256_bytes(pickle.dumps(estimator, protocol=5))


def _parameter_count(estimator: Any) -> int:
    candidates: list[Any] = []
    if hasattr(estimator, "classifier_"):
        classifier = estimator.classifier_
        if hasattr(classifier, "named_steps"):
            candidates.extend(classifier.named_steps.values())
        candidates.append(classifier)
    if hasattr(estimator, "model_"):
        candidates.append(estimator.model_)
    for candidate in reversed(candidates):
        coefficient = getattr(candidate, "coef_", None)
        intercept = getattr(candidate, "intercept_", None)
        if coefficient is not None:
            return int(np.asarray(coefficient).size) + (
                int(np.asarray(intercept).size) if intercept is not None else 0
            )
    return 0


def _prepare_features(
    data: Mapping[str, Any],
    job: Job,
    plan: Mapping[str, Any],
) -> PreparedFeatures:
    from .data import (
        apply_channel_scaler,
        fit_channel_scaler,
        split_indices,
    )
    from .local_geometric_controls import fixed_filter_bank, spd_covariances

    if data["identity"] != {
        key: value
        for key, value in _planned_cache(plan, job).items()
        if key not in {"trial_count", "n_channels"}
    }:
        raise ControlGridError("loaded cache identity differs from immutable plan")
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
        raise ControlGridError("runtime split differs from immutable plan")
    source = np.sort(np.concatenate((train, validation))).astype(
        np.int64, copy=False
    )
    names = tuple(str(value) for value in data["channel_names"].tolist())
    selection_mean, selection_std = fit_channel_scaler(
        data["x"][train], names
    )
    refit_mean, refit_std = fit_channel_scaler(data["x"][source], names)
    sfreq = float(plan["datasets"][job.dataset]["preprocessing"]["sfreq_hz"])

    def transform(rows: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        scaled = apply_channel_scaler(data["x"][rows], mean, std)
        return fixed_filter_bank(scaled, sfreq=sfreq)

    train_epochs = transform(train, selection_mean, selection_std)
    validation_epochs = transform(validation, selection_mean, selection_std)
    source_epochs = transform(source, refit_mean, refit_std)
    test_epochs = transform(test, refit_mean, refit_std)
    return PreparedFeatures(
        train_epochs=train_epochs,
        validation_epochs=validation_epochs,
        source_epochs=source_epochs,
        test_epochs=test_epochs,
        train_covariances=spd_covariances(train_epochs),
        validation_covariances=spd_covariances(validation_epochs),
        source_covariances=spd_covariances(source_epochs),
        test_covariances=spd_covariances(test_epochs),
        train_outcomes=np.asarray(data["y"][train], dtype=np.int64),
        validation_outcomes=np.asarray(data["y"][validation], dtype=np.int64),
        source_outcomes=np.asarray(data["y"][source], dtype=np.int64),
        test_rows=np.asarray(test, dtype=np.int64),
        selection_mean=selection_mean,
        selection_std=selection_std,
        refit_mean=refit_mean,
        refit_std=refit_std,
        channel_names=names,
    )


def _feature_view(control: str, prepared: PreparedFeatures, split: str) -> np.ndarray:
    suffix = "epochs" if control == "control.ea_fbcsp" else "covariances"
    return np.asarray(getattr(prepared, f"{split}_{suffix}"))


def select_refit_predict(
    *,
    job: Job,
    plan: Mapping[str, Any],
    prepared: PreparedFeatures,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Run a deterministic control protocol and return score-blind material."""

    started = time.perf_counter()
    n_classes = int(plan["datasets"][job.dataset]["n_classes"])
    expected_classes = set(range(n_classes))
    for name, outcomes in (
        ("train", prepared.train_outcomes),
        ("validation", prepared.validation_outcomes),
        ("source", prepared.source_outcomes),
    ):
        if set(outcomes.tolist()) != expected_classes:
            raise EstimatorCapabilityError(
                f"{name} fitting outcomes do not contain all {n_classes} classes"
            )
    candidates = _admissible_grid(job.control, len(prepared.channel_names))
    train_x = _feature_view(job.control, prepared, "train")
    validation_x = _feature_view(job.control, prepared, "validation")
    source_x = _feature_view(job.control, prepared, "source")
    test_x = _feature_view(job.control, prepared, "test")

    best_loss = float("inf")
    best_parameters: dict[str, Any] | None = None
    selected_state_sha256: str | None = None
    selection_started = time.perf_counter()
    for parameters in candidates:
        try:
            estimator = _new_estimator(job.control, parameters)
            _fit_estimator(
                job.control, estimator, train_x, prepared.train_outcomes
            )
            probability = _valid_probabilities(
                estimator, validation_x, n_classes=n_classes
            )
        except ControlGridError:
            raise
        except Exception as error:
            if n_classes > 2:
                raise EstimatorCapabilityError(
                    f"{job.control} failed its registered multiclass contract "
                    f"for {job.dataset}; registry/implementation drift must be "
                    "resolved rather than changing the task"
                ) from error
            raise
        loss = float(
            log_loss(
                prepared.validation_outcomes,
                probability,
                labels=np.arange(n_classes, dtype=np.int64),
            )
        )
        if not np.isfinite(loss):
            raise EstimatorCapabilityError("validation selection produced nonfinite NLL")
        if loss < best_loss - 1e-12:
            best_loss = loss
            best_parameters = dict(parameters)
            selected_state_sha256 = _estimator_state_sha256(estimator)
    selection_seconds = time.perf_counter() - selection_started
    if best_parameters is None or selected_state_sha256 is None:
        raise EstimatorCapabilityError("hyperparameter selection produced no estimator")

    refit_started = time.perf_counter()
    try:
        final_estimator = _new_estimator(job.control, best_parameters)
        _fit_estimator(
            job.control, final_estimator, source_x, prepared.source_outcomes
        )
    except ControlGridError:
        raise
    except Exception as error:
        if n_classes > 2:
            raise EstimatorCapabilityError(
                f"{job.control} failed multiclass final refit for {job.dataset}; "
                "registry/implementation drift must be resolved"
            ) from error
        raise
    refit_seconds = time.perf_counter() - refit_started
    refit_state_sha256 = _estimator_state_sha256(final_estimator)
    inference_started = time.perf_counter()
    # This is the only final-estimator test call.  No estimator calibrate method
    # is invoked anywhere in this runner.
    try:
        probabilities = _valid_probabilities(
            final_estimator, test_x, n_classes=n_classes
        )
    except ControlGridError:
        raise
    except Exception as error:
        if n_classes > 2:
            raise EstimatorCapabilityError(
                f"{job.control} failed multiclass probability production for "
                f"{job.dataset}; registry/implementation drift must be resolved"
            ) from error
        raise
    inference_seconds = time.perf_counter() - inference_started
    if _estimator_state_sha256(final_estimator) != refit_state_sha256:
        raise EstimatorCapabilityError(
            "final estimator mutated during held-out predict_proba"
        )

    split = _planned_split(plan, job)
    metadata = {
        "cache_array_sha256": str(
            _planned_cache(plan, job)["array_sha256"]
        ),
        "n_channels": len(prepared.channel_names),
        "n_classes": n_classes,
        "split": copy.deepcopy(dict(split)),
        "scalers": {
            "selection_mean_sha256": _array_sha256(prepared.selection_mean),
            "selection_std_sha256": _array_sha256(prepared.selection_std),
            "refit_mean_sha256": _array_sha256(prepared.refit_mean),
            "refit_std_sha256": _array_sha256(prepared.refit_std),
        },
        "fit": {
            "deterministic": True,
            "seed_identity": "none",
            "candidate_grid": [dict(value) for value in candidates],
            "selection_candidate_count": len(candidates),
            "selected_parameters": best_parameters,
            "selection_state_sha256": selected_state_sha256,
            "refit_state_sha256": refit_state_sha256,
            "estimator_fit_calls": len(candidates) + 1,
            "parameter_count": _parameter_count(final_estimator),
        },
        "protocol": {
            "selection_scaler_rows": "train_only",
            "selection_estimator_rows": "train_only",
            "selection_decision_rows": "validation_only",
            "selection_objective": "negative_log_likelihood_not_persisted",
            "refit_scaler_rows": "train_plus_validation",
            "refit_estimator_rows": "train_plus_validation",
            "test_use": "single_predict_proba_call_only",
            "transductive_calibration": False,
            "test_performance_computed": False,
        },
        "timing_seconds": {
            "selection_fit": selection_seconds,
            "refit_fit": refit_seconds,
            "test_inference": inference_seconds,
            "job_total": time.perf_counter() - started,
        },
        "runtime": {
            "cpu_threads": int(
                plan["execution_config"]["threads_per_worker"]
            ),
            "thread_environment": {
                name: os.environ.get(name)
                for name in THREAD_ENVIRONMENT_VARIABLES
            },
            "gpu_reservation": "none_cpu_only",
        },
    }
    return metadata, prepared.test_rows, probabilities


def _validate_metadata(
    plan: Mapping[str, Any], job: Job, metadata: Mapping[str, Any]
) -> None:
    if _recursive_forbidden_keys(metadata):
        raise ControlGridError("metadata violates the score-blind key contract")
    _require_exact_keys(
        metadata,
        {
            "cache_array_sha256",
            "n_channels",
            "n_classes",
            "split",
            "scalers",
            "fit",
            "protocol",
            "timing_seconds",
            "runtime",
        },
        "metadata",
    )
    if metadata.get("cache_array_sha256") != _planned_cache(plan, job).get(
        "array_sha256"
    ):
        raise ControlGridError("record cache identity differs from plan")
    if metadata.get("split") != _planned_split(plan, job):
        raise ControlGridError("record split identity differs from plan")
    if int(metadata.get("n_classes", -1)) != int(
        plan["datasets"][job.dataset]["n_classes"]
    ):
        raise ControlGridError("record class count differs from plan")
    fit = metadata.get("fit")
    scalers = metadata.get("scalers")
    protocol = metadata.get("protocol")
    timing = metadata.get("timing_seconds")
    runtime = metadata.get("runtime")
    if not all(
        isinstance(value, Mapping)
        for value in (fit, scalers, protocol, timing, runtime)
    ):
        raise ControlGridError("record metadata has an absent nested contract")
    n_channels = int(metadata.get("n_channels", -1))
    if n_channels != int(_planned_cache(plan, job).get("n_channels", -1)):
        raise ControlGridError("record channel count differs from cache identity")
    candidates = _admissible_grid(job.control, n_channels)
    _require_exact_keys(
        scalers,
        {
            "selection_mean_sha256",
            "selection_std_sha256",
            "refit_mean_sha256",
            "refit_std_sha256",
        },
        "metadata scaler",
    )
    for key, value in scalers.items():
        _require_sha256(value, f"metadata scaler {key}")
    _require_exact_keys(
        fit,
        {
            "deterministic",
            "seed_identity",
            "candidate_grid",
            "selection_candidate_count",
            "selected_parameters",
            "selection_state_sha256",
            "refit_state_sha256",
            "estimator_fit_calls",
            "parameter_count",
        },
        "metadata fit",
    )
    if fit.get("candidate_grid") != [dict(value) for value in candidates]:
        raise ControlGridError("record candidate grid differs from protocol")
    if fit.get("selected_parameters") not in candidates:
        raise ControlGridError("record selected parameters are outside the grid")
    parameter_count = fit.get("parameter_count")
    if (
        fit.get("deterministic") is not True
        or fit.get("seed_identity") != "none"
        or fit.get("selection_candidate_count") != len(candidates)
        or fit.get("estimator_fit_calls") != len(candidates) + 1
        or not isinstance(parameter_count, int)
        or isinstance(parameter_count, bool)
        or parameter_count < 0
    ):
        raise ControlGridError("record fit counts/determinism differ from protocol")
    for key in ("selection_state_sha256", "refit_state_sha256"):
        _require_sha256(fit.get(key), f"record {key}")
    expected_protocol = {
        "selection_scaler_rows": "train_only",
        "selection_estimator_rows": "train_only",
        "selection_decision_rows": "validation_only",
        "selection_objective": "negative_log_likelihood_not_persisted",
        "refit_scaler_rows": "train_plus_validation",
        "refit_estimator_rows": "train_plus_validation",
        "test_use": "single_predict_proba_call_only",
        "transductive_calibration": False,
        "test_performance_computed": False,
    }
    _require_exact_keys(protocol, set(expected_protocol), "metadata protocol")
    if protocol != expected_protocol:
        raise ControlGridError("record protocol differs from audited contract")
    expected_timing = {
        "selection_fit",
        "refit_fit",
        "test_inference",
        "job_total",
    }
    _require_exact_keys(timing, expected_timing, "metadata timing")
    timing_values: dict[str, float] = {}
    for key in expected_timing:
        value = timing.get(key)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not np.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ControlGridError(f"record timing {key} is invalid")
        timing_values[key] = float(value)
    component_total = sum(
        timing_values[key]
        for key in ("selection_fit", "refit_fit", "test_inference")
    )
    if timing_values["job_total"] + 1e-9 < component_total:
        raise ControlGridError("record job_total is shorter than timed components")
    threads = int(plan["execution_config"]["threads_per_worker"])
    _require_exact_keys(
        runtime,
        {"cpu_threads", "thread_environment", "gpu_reservation"},
        "metadata runtime",
    )
    if (
        runtime.get("cpu_threads") != threads
        or runtime.get("gpu_reservation") != "none_cpu_only"
        or runtime.get("thread_environment")
        != {name: str(threads) for name in THREAD_ENVIRONMENT_VARIABLES}
    ):
        raise ControlGridError("record runtime differs from CPU-only contract")


def _prediction_npz_bytes(
    rows: np.ndarray,
    probabilities: np.ndarray,
) -> bytes:
    stream = io.BytesIO()
    np.savez_compressed(
        stream,
        rows=np.ascontiguousarray(rows),
        probabilities=np.ascontiguousarray(probabilities),
    )
    return stream.getvalue()


def _load_prediction_npz_exact(
    payload: bytes,
    *,
    source: str,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
            members = archive.infolist()
            names = tuple(member.filename for member in members)
            if (
                names != PREDICTION_ZIP_MEMBERS
                or len(names) != len(set(names))
                or any(
                    member.is_dir()
                    or member.flag_bits & 0x1
                    or member.file_size <= 0
                    for member in members
                )
            ):
                raise ControlGridError(
                    f"{source} raw ZIP member roster is invalid"
                )
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if archive.files != ["rows", "probabilities"]:
                raise ControlGridError(
                    f"{source} NumPy member roster is invalid"
                )
            rows = archive["rows"].copy()
            probabilities = archive["probabilities"].copy()
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise ControlGridError(
            f"{source} prediction archive cannot load: {error}"
        ) from error
    return rows, probabilities


def _is_formal_plan(plan: Mapping[str, Any]) -> bool:
    return (
        tuple(plan.get("dataset_order", ())) == OPENED_DATASETS
        and tuple(plan.get("controls", ())) == CONTROLS
        and plan.get("datasets") == _dataset_contracts()
        and plan.get("n_jobs") == EXPECTED_JOB_COUNT
    )


def _verify_job_cache_split_identity(
    plan: Mapping[str, Any],
    *,
    cache_root: Path,
    job: Job,
) -> None:
    from .data import load_subject_cache, split_indices

    data = load_subject_cache(
        job.dataset, job.subject, cache_root=cache_root
    )
    planned_cache = _planned_cache(plan, job)
    observed_cache = copy.deepcopy(dict(data["identity"]))
    observed_cache["trial_count"] = len(data["y"])
    observed_cache["n_channels"] = int(data["x"].shape[1])
    if observed_cache != planned_cache:
        raise ControlGridError(
            f"live cache identity differs for {job.job_id}"
        )
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
        cache_array_sha256=str(
            data["identity"]["array_sha256"]
        ),
        trial_count=len(data["y"]),
        train_rows=train,
        validation_rows=validation,
        test_rows=test,
    )
    if observed_split != _planned_split(plan, job):
        raise ControlGridError(
            f"live split identity differs for {job.job_id}"
        )


def _verify_worker_publication_identity(
    *,
    run_root: Path,
    plan: Mapping[str, Any],
    claim: Claim,
    cache_root: Path | None,
    cpu_slot: CPUSlot | None,
) -> None:
    _validate_plan_contract(plan)
    if load_plan(run_root) != dict(plan):
        raise ControlGridError("live immutable plan differs before publication")
    _validate_claim_payload(run_root, plan, claim)
    formal = _is_formal_plan(plan)
    if cpu_slot is not None:
        _validate_cpu_slot(run_root, plan, cpu_slot)
        cpu_guard = claim.resource_guard["cpu"]
        if (
            cpu_guard["slot"] != cpu_slot.index
            or cpu_guard["slot_nonce"] != cpu_slot.nonce
            or cpu_guard["threads"] != cpu_slot.threads
        ):
            raise ControlGridError(
                "claim no longer binds the live CPU slot"
            )
    elif formal:
        raise ControlGridError(
            "formal publication requires the held CPU-slot lease"
        )
    if formal:
        if _source_identity() != plan.get("source_identity"):
            raise ControlGridError(
                "source identity drifted before publication"
            )
        if _environment_identity() != plan.get("environment_identity"):
            raise ControlGridError(
                "UV execution environment drifted before publication"
            )
        _validate_registry_contract()
        if plan.get("datasets") != _dataset_contracts():
            raise ControlGridError(
                "dataset/config contract drifted before publication"
            )
    expected_grid = {
        stable_id: [dict(value) for value in BASE_HYPERPARAMETER_GRID[stable_id]]
        for stable_id in plan["controls"]
    }
    if plan.get("hyperparameter_grid") != expected_grid:
        raise ControlGridError(
            "hyperparameter configuration drifted before publication"
        )
    if formal:
        threads = int(plan["execution_config"]["threads_per_worker"])
        if any(
            os.environ.get(name) != str(threads)
            for name in THREAD_ENVIRONMENT_VARIABLES
        ):
            raise ControlGridError(
                "worker thread environment drifted before publication"
            )
        if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
            raise ControlGridError(
                "control publication is not CPU-only"
            )
        if any(
            os.environ.get(name)
            for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK")
        ):
            raise ControlGridError(
                "distributed worker variables are forbidden"
            )
    disk = probe_disk(
        run_root,
        minimum_free_gib=float(
            plan["execution_config"]["minimum_free_gib"]
        ),
    )
    if not disk.safe:
        raise DiskUnavailable(disk.reason)
    if cache_root is not None:
        _verify_job_cache_split_identity(
            plan, cache_root=cache_root, job=claim.job
        )
    elif formal:
        raise ControlGridError(
            "formal publication requires cache/split revalidation"
        )


def commit_job_output(
    run_root: Path,
    plan: Mapping[str, Any],
    claim: Claim,
    *,
    metadata: Mapping[str, Any],
    test_rows: np.ndarray,
    probabilities: np.ndarray,
    cache_root: Path | None = None,
    cpu_slot: CPUSlot | None = None,
) -> Path:
    _validate_job_identity(plan, claim.job)
    _validate_claim_payload(run_root, plan, claim)
    job = claim.job
    _validate_metadata(plan, job, metadata)
    rows = np.asarray(test_rows)
    values = np.asarray(probabilities)
    test_identity = _planned_split(plan, job)["partitions"]["test"]
    n_classes = int(plan["datasets"][job.dataset]["n_classes"])
    if (
        rows.dtype != np.int64
        or rows.ndim != 1
        or len(rows) != int(test_identity["count"])
        or _rows_sha256(rows) != test_identity["rows_sha256"]
        or values.dtype != np.float64
        or values.shape != (len(rows), n_classes)
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(values > 1.0)
        or not np.allclose(values.sum(axis=1), 1.0, rtol=0.0, atol=1e-8)
    ):
        raise ControlGridError("score-blind prediction arrays are invalid")
    run_root = _absolute(run_root)
    final = _record_directory(run_root, job)
    final_parent = _safe_mkdir(final.parent)
    stage_name = (
        f".{final.name}.partial-{job.job_id}."
        f"{claim.nonce}.{uuid.uuid4().hex}"
    )
    stage = final_parent / stage_name
    record = {
        "schema": RECORD_SCHEMA,
        "created_at": _utc_now(),
        "plan_sha256": plan["plan_sha256"],
        "job_id": job.job_id,
        "job": job.identity(),
        "score_blind": True,
        "test_count": len(rows),
        "test_rows_sha256": _rows_sha256(rows),
        "claim_resource_guard": copy.deepcopy(dict(claim.resource_guard)),
        "metadata": copy.deepcopy(dict(metadata)),
    }
    violations = _recursive_forbidden_keys(record)
    if violations:
        raise ControlGridError(f"record contains forbidden keys: {violations[:5]}")
    record_payload = _canonical_bytes(record) + b"\n"
    prediction_payload = _prediction_npz_bytes(rows, values)
    completion = {
        "schema": COMPLETION_SCHEMA,
        "created_at": _utc_now(),
        "plan_sha256": plan["plan_sha256"],
        "job_id": job.job_id,
        "job": job.identity(),
        "files": {
            "record.json": _sha256_bytes(record_payload),
            "predictions.npz": _sha256_bytes(prediction_payload),
        },
    }
    completion_payload = _canonical_bytes(completion) + b"\n"
    stage_identity: tuple[int, int] | None = None
    parent_descriptor = _open_directory_absolute(final_parent)
    stage_descriptor = -1
    published = False
    try:
        with _coordination_lock(
            run_root, exclusive=False, nonblocking=True
        ):
            _validate_claim_payload(run_root, plan, claim)
            if cpu_slot is not None:
                _validate_cpu_slot(run_root, plan, cpu_slot)
            if _path_exists(final):
                raise ClaimUnavailable(
                    f"{job.job_id} became complete before publication"
                )
            os.mkdir(stage_name, mode=0o700, dir_fd=parent_descriptor)
            os.fsync(parent_descriptor)
            stage_descriptor = os.open(
                stage_name,
                _directory_flags(),
                dir_fd=parent_descriptor,
            )
            stage_status = os.fstat(stage_descriptor)
            stage_identity = (
                int(stage_status.st_dev),
                int(stage_status.st_ino),
            )
            _write_exclusive_at(
                stage_descriptor,
                "record.json",
                record_payload,
                mode=0o444,
            )
            _write_exclusive_at(
                stage_descriptor,
                "predictions.npz",
                prediction_payload,
                mode=0o444,
            )
            _write_exclusive_at(
                stage_descriptor,
                "completion.json",
                completion_payload,
                mode=0o444,
            )
            os.fsync(stage_descriptor)
            os.fchmod(stage_descriptor, 0o555)
            os.fsync(stage_descriptor)
            validate_completion(
                run_root, plan, job, _directory=stage
            )
            _verify_worker_publication_identity(
                run_root=run_root,
                plan=plan,
                claim=claim,
                cache_root=cache_root,
                cpu_slot=cpu_slot,
            )
            _rename_noreplace_at(
                parent_descriptor,
                stage_name,
                parent_descriptor,
                final.name,
            )
            published = True
            os.fsync(parent_descriptor)
            current = os.stat(
                final.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if (
                int(current.st_dev),
                int(current.st_ino),
            ) != stage_identity:
                raise ControlGridError(
                    "published result differs from its staged inode"
                )
            validate_completion(run_root, plan, job)
            _verify_worker_publication_identity(
                run_root=run_root,
                plan=plan,
                claim=claim,
                cache_root=cache_root,
                cpu_slot=cpu_slot,
            )
            final_status = os.stat(
                final.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if (
                int(final_status.st_dev),
                int(final_status.st_ino),
            ) != stage_identity:
                raise ControlGridError(
                    "result authority changed before publication return"
                )
            return final
    except Exception:
        if stage_identity is not None:
            for candidate, category in (
                (final, "result_authority_loss"),
                (stage, "abandoned_job_stages"),
            ):
                if not _path_exists(candidate):
                    continue
                observed = _anchored_lstat(candidate)
                if (
                    int(observed.st_dev),
                    int(observed.st_ino),
                ) != stage_identity:
                    continue
                _quarantine_leaf(
                    run_root,
                    candidate,
                    category=category,
                    expected_identity=stage_identity,
                )
        raise
    finally:
        if stage_descriptor >= 0:
            os.close(stage_descriptor)
        os.close(parent_descriptor)


def validate_completion(
    run_root: Path,
    plan: Mapping[str, Any],
    job: Job,
    *,
    _directory: Path | None = None,
    _stable_parent: bool = False,
) -> dict[str, Any]:
    _validate_job_identity(plan, job)
    directory = (
        _record_directory(run_root, job)
        if _directory is None
        else _directory
    )
    payloads, directory_fingerprint, child_fingerprints = (
        _snapshot_regular_files(
            directory,
            ("record.json", "predictions.npz", "completion.json"),
            exact_roster=FINAL_FILENAMES,
            directory_mode=0o555,
            child_mode=0o444,
            stable_parent=_stable_parent,
        )
    )
    completion = _strict_json_bytes(
        payloads["completion.json"],
        source=str(directory / "completion.json"),
    )
    _require_exact_keys(
        completion,
        {
            "schema",
            "created_at",
            "plan_sha256",
            "job_id",
            "job",
            "files",
        },
        "completion",
    )
    if not isinstance(completion.get("created_at"), str):
        raise ControlGridError(f"{directory} completion has invalid created_at")
    if payloads["completion.json"] != _canonical_bytes(completion) + b"\n":
        raise ControlGridError(f"{directory} completion is not canonical JSON")
    expected = {
        "schema": COMPLETION_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "job_id": job.job_id,
        "job": job.identity(),
    }
    for key, value in expected.items():
        if completion.get(key) != value:
            raise ControlGridError(f"{directory} completion has invalid {key}")
    expected_hashes = {
        "record.json": _sha256_bytes(payloads["record.json"]),
        "predictions.npz": _sha256_bytes(payloads["predictions.npz"]),
    }
    files = completion.get("files")
    if not isinstance(files, Mapping):
        raise ControlGridError(f"{directory} completion file manifest is absent")
    _require_exact_keys(files, set(expected_hashes), "completion file manifest")
    if completion.get("files") != expected_hashes:
        raise ControlGridError(f"{directory} completion checksums are invalid")
    record = _strict_json_bytes(
        payloads["record.json"], source=str(directory / "record.json")
    )
    _require_exact_keys(
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
            "claim_resource_guard",
            "metadata",
        },
        "record",
    )
    if (
        not isinstance(record.get("created_at"), str)
        or payloads["record.json"] != _canonical_bytes(record) + b"\n"
    ):
        raise ControlGridError(f"{directory} record JSON/timestamp is invalid")
    for key, value in {
        "schema": RECORD_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "job_id": job.job_id,
        "job": job.identity(),
        "score_blind": True,
    }.items():
        if record.get(key) != value:
            raise ControlGridError(f"{directory} record has invalid {key}")
    violations = _recursive_forbidden_keys(record)
    if violations:
        raise ControlGridError(f"{directory} contains forbidden keys {violations[:5]}")
    resource_guard = record.get("claim_resource_guard")
    if not isinstance(resource_guard, Mapping):
        raise ControlGridError(f"{directory} claim resource guard is absent")
    _validate_resource_guard(run_root, plan, resource_guard)
    test_count = record.get("test_count")
    if (
        not isinstance(test_count, int)
        or isinstance(test_count, bool)
        or test_count <= 0
    ):
        raise ControlGridError(f"{directory} has an invalid test_count")
    _require_sha256(record.get("test_rows_sha256"), "record test row identity")
    metadata = record.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ControlGridError(f"{directory} record metadata is absent")
    _validate_metadata(plan, job, metadata)
    rows, probabilities = _load_prediction_npz_exact(
        payloads["predictions.npz"],
        source=str(directory / "predictions.npz"),
    )
    test_identity = _planned_split(plan, job)["partitions"]["test"]
    n_classes = int(plan["datasets"][job.dataset]["n_classes"])
    if (
        rows.dtype != np.int64
        or rows.ndim != 1
        or len(np.unique(rows)) != len(rows)
        or len(rows) != test_count
        or len(rows) != int(test_identity["count"])
        or _rows_sha256(rows) != record["test_rows_sha256"]
        or _rows_sha256(rows) != test_identity["rows_sha256"]
        or probabilities.dtype != np.float64
        or probabilities.shape != (len(rows), n_classes)
        or not np.all(np.isfinite(probabilities))
        or np.any(probabilities < 0.0)
        or np.any(probabilities > 1.0)
        or not np.allclose(
            probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-8
        )
    ):
        raise ControlGridError(f"{directory} prediction arrays are invalid")
    return {
        "record": record,
        "completion": completion,
        "rows": rows,
        "probabilities": probabilities,
        "file_sha256": {
            name: _sha256_bytes(payload)
            for name, payload in payloads.items()
        },
        "snapshot": {
            "directory": list(directory_fingerprint),
            "children": {
                name: list(fingerprint)
                for name, fingerprint in child_fingerprints.items()
            },
        },
    }


def _record_failure(
    run_root: Path,
    plan: Mapping[str, Any],
    claim: Claim,
    error: Exception,
) -> Path:
    root = _safe_mkdir(
        _absolute(run_root) / "failures" / claim.job.dataset
    )
    path = root / (
        f"{claim.job.job_id}.{int(time.time())}.{uuid.uuid4().hex}.json"
    )
    with _coordination_lock(
        run_root, exclusive=False, nonblocking=False
    ):
        _validate_claim_payload(run_root, plan, claim)
        _write_json_exclusive(
            path,
            {
                "schema": FAILURE_SCHEMA,
                "created_at": _utc_now(),
                "plan_sha256": plan["plan_sha256"],
                "job_id": claim.job.job_id,
                "job": claim.job.identity(),
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
    return path


def audit_grid(
    run_root: Path,
    plan: Mapping[str, Any],
    *,
    _fenced: bool = False,
) -> dict[str, Any]:
    if not _fenced:
        with analysis_fence(run_root):
            return audit_grid(run_root, plan, _fenced=True)
    jobs = tuple(iter_jobs(plan))
    complete: list[str] = []
    missing: list[str] = []
    corrupt: dict[str, str] = {}
    per_control_complete: Counter[str] = Counter()
    per_dataset_complete: Counter[str] = Counter()
    for job in jobs:
        directory = _record_directory(run_root, job)
        if not directory.exists():
            missing.append(job.job_id)
            continue
        try:
            validate_completion(run_root, plan, job)
        except Exception as error:  # noqa: BLE001 - corruption audit is fail-soft
            corrupt[job.job_id] = str(error)
        else:
            complete.append(job.job_id)
            per_control_complete[job.control] += 1
            per_dataset_complete[job.dataset] += 1
    expected_directories = {_record_directory(run_root, job) for job in jobs}
    records_root = run_root / "records"
    allowed_record_directories: set[Path] = {records_root}
    allowed_record_files: set[Path] = set()
    for directory in expected_directories:
        current = directory
        while current != records_root:
            allowed_record_directories.add(current)
            current = current.parent
        allowed_record_files.update(
            directory / filename for filename in FINAL_FILENAMES
        )
    unexpected_record_paths: list[str] = []
    if records_root.exists():
        for path in records_root.rglob("*"):
            expected_directory = path in allowed_record_directories
            expected_file = path in allowed_record_files
            invalid_type = (
                expected_directory
                and (path.is_symlink() or not path.is_dir())
            ) or (
                expected_file
                and (path.is_symlink() or not path.is_file())
            )
            if (not expected_directory and not expected_file) or invalid_type:
                unexpected_record_paths.append(str(path.relative_to(run_root)))
    unexpected_record_paths.sort()
    extras = sorted(
        value
        for value in unexpected_record_paths
        if (run_root / value).is_dir()
    )

    def residual_paths(name: str) -> list[str]:
        root = run_root / name
        if not root.exists():
            return []
        result = [str(path.relative_to(run_root)) for path in root.rglob("*")]
        return sorted(result)

    residual_partial_paths = residual_paths("partial")
    residual_claim_paths = residual_paths("claims")
    residual_cpu_slot_paths = residual_paths("cpu_slots")
    allowed_root_entries = {
        "plan.json",
        "plan.sha256",
        "records",
        "partial",
        "claims",
        "cpu_slots",
        "failures",
        "quarantine",
        "audit.json",
        COORDINATION_LOCK_NAME,
    }
    unexpected_root_entries = sorted(
        path.name
        for path in run_root.iterdir()
        if path.name not in allowed_root_entries
    )
    payload = {
        "schema": AUDIT_SCHEMA,
        "created_at": _utc_now(),
        "plan_sha256": plan["plan_sha256"],
        "score_blind": True,
        "expected_jobs": len(jobs),
        "complete_jobs": len(complete),
        "missing_jobs": len(missing),
        "corrupt_jobs": len(corrupt),
        "extra_directories": extras,
        "unexpected_record_paths": unexpected_record_paths,
        "residual_partial_paths": residual_partial_paths,
        "residual_claim_paths": residual_claim_paths,
        "residual_cpu_slot_paths": residual_cpu_slot_paths,
        "unexpected_root_entries": unexpected_root_entries,
        "complete_job_ids": sorted(complete),
        "missing_job_ids": sorted(missing),
        "corrupt_job_ids": dict(sorted(corrupt.items())),
        "expected_per_control": {
            control: sum(
                len(plan["datasets"][dataset]["subjects"])
                * len(plan["datasets"][dataset]["folds"])
                for dataset in plan["dataset_order"]
            )
            for control in plan["controls"]
        },
        "complete_per_control": {
            control: per_control_complete[control]
            for control in plan["controls"]
        },
        "complete_per_dataset": {
            dataset: per_dataset_complete[dataset]
            for dataset in plan["dataset_order"]
        },
        "complete": (
            len(complete) == len(jobs)
            and not corrupt
            and not unexpected_record_paths
            and not residual_partial_paths
            and not residual_claim_paths
            and not residual_cpu_slot_paths
            and not unexpected_root_entries
        ),
    }
    if _recursive_forbidden_keys(payload):
        raise ControlGridError("audit payload violates score-blind contract")
    return payload


def _worker(
    *,
    run_root: Path,
    cache_root: Path,
    required_plan_sha256: str,
) -> int:
    plan = load_plan(run_root)
    if plan["plan_sha256"] != required_plan_sha256:
        raise ControlGridError("worker required plan digest differs from plan.json")
    threads = int(plan["execution_config"]["threads_per_worker"])
    _apply_cpu_limits(threads)
    verify_runtime_identity(plan, cache_root=cache_root)
    slot = acquire_cpu_slot(run_root, plan)
    prepared_key: tuple[str, int, int] | None = None
    prepared: PreparedFeatures | None = None
    try:
        for job in iter_jobs(plan):
            try:
                claim = acquire_claim(
                    run_root, plan, job, cpu_slot=slot
                )
            except ClaimUnavailable:
                continue
            try:
                key = (job.dataset, job.subject, job.fold)
                if key != prepared_key or prepared is None:
                    from .data import load_subject_cache

                    data = load_subject_cache(
                        job.dataset, job.subject, cache_root=cache_root
                    )
                    prepared = _prepare_features(data, job, plan)
                    prepared_key = key
                metadata, rows, probabilities = select_refit_predict(
                    job=job, plan=plan, prepared=prepared
                )
                commit_job_output(
                    run_root,
                    plan,
                    claim,
                    metadata=metadata,
                    test_rows=rows,
                    probabilities=probabilities,
                    cache_root=cache_root,
                    cpu_slot=slot,
                )
            except Exception as error:
                _record_failure(run_root, plan, claim, error)
                raise
            finally:
                release_claim(claim)
    finally:
        release_cpu_slot(slot)
    return 0


def _spawn_workers(
    *,
    run_root: Path,
    cache_root: Path,
    plan: Mapping[str, Any],
    workers: int,
) -> int:
    slots = int(plan["execution_config"]["cpu_slot_count"])
    if not 1 <= workers <= slots:
        raise ValueError(f"workers must be in [1, {slots}]")
    threads = int(plan["execution_config"]["threads_per_worker"])
    environment = os.environ.copy()
    for variable in THREAD_ENVIRONMENT_VARIABLES:
        environment[variable] = str(threads)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment.setdefault("MNE_DONTWRITE_HOME", "true")
    commands: list[subprocess.Popen[Any]] = []
    try:
        for _ in range(workers):
            commands.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "ieee_mi.control_grid",
                        "worker",
                        "--run-root",
                        str(run_root),
                        "--cache-root",
                        str(cache_root),
                        "--plan-sha256",
                        str(plan["plan_sha256"]),
                    ],
                    env=environment,
                )
            )
        while True:
            exit_codes = [process.poll() for process in commands]
            if any(value is not None and value != 0 for value in exit_codes):
                _terminate_owned_workers(commands)
                return 1
            if all(value == 0 for value in exit_codes):
                return 0
            time.sleep(0.1)
    except BaseException:
        _terminate_owned_workers(commands)
        raise


def _terminate_owned_workers(processes: Sequence[subprocess.Popen[Any]]) -> None:
    """Stop only children launched by this supervisor, never foreign processes."""

    running = [process for process in processes if process.poll() is None]
    for process in running:
        process.terminate()
    for process in running:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def _validated_audit_output_path(run_root: Path, output: Path) -> Path:
    """Allow only the designated audit file inside the immutable run tree."""

    root = Path(os.path.abspath(run_root))
    candidate = Path(os.path.abspath(output))
    try:
        relative = candidate.relative_to(root)
    except ValueError:
        return candidate
    if relative != Path("audit.json") or candidate.is_symlink():
        raise ControlGridError(
            "audit output inside run_root must be the non-symlink file audit.json"
        )
    return candidate


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    plan_parser = commands.add_parser("plan", help="create/validate immutable plan")
    plan_parser.add_argument("--run-root", type=Path, required=True)
    plan_parser.add_argument("--cache-root", type=Path, required=True)
    plan_parser.add_argument(
        "--cpu-budget-threads", type=int, default=DEFAULT_CPU_BUDGET
    )
    plan_parser.add_argument(
        "--threads-per-worker", type=int, default=DEFAULT_THREADS_PER_WORKER
    )
    plan_parser.add_argument(
        "--minimum-free-gib", type=float, default=DEFAULT_MIN_FREE_GIB
    )

    run_parser = commands.add_parser("run", help="resume CPU workers")
    run_parser.add_argument("--run-root", type=Path, required=True)
    run_parser.add_argument("--cache-root", type=Path, required=True)
    run_parser.add_argument("--workers", type=int)
    run_parser.add_argument("--plan-sha256", required=True)

    worker_parser = commands.add_parser("worker", help=argparse.SUPPRESS)
    worker_parser.add_argument("--run-root", type=Path, required=True)
    worker_parser.add_argument("--cache-root", type=Path, required=True)
    worker_parser.add_argument("--plan-sha256", required=True)

    audit_parser = commands.add_parser("audit", help="score-blind completion audit")
    audit_parser.add_argument("--run-root", type=Path, required=True)
    audit_parser.add_argument("--output", type=Path)

    args = parser.parse_args(argv)
    if args.command == "plan":
        plan = build_plan(
            cache_root=args.cache_root,
            cpu_budget_threads=args.cpu_budget_threads,
            threads_per_worker=args.threads_per_worker,
            minimum_free_gib=args.minimum_free_gib,
        )
        observed = write_or_validate_plan(args.run_root, plan)
        print(
            json.dumps(
                {
                    "plan_sha256": observed["plan_sha256"],
                    "n_jobs": observed["n_jobs"],
                    "cpu_slots": observed["execution_config"]["cpu_slot_count"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "worker":
        return _worker(
            run_root=args.run_root.resolve(),
            cache_root=args.cache_root.resolve(),
            required_plan_sha256=str(args.plan_sha256),
        )
    plan = load_plan(args.run_root.resolve())
    if args.command == "run":
        if str(args.plan_sha256) != plan["plan_sha256"]:
            raise ControlGridError("--plan-sha256 differs from immutable plan")
        _require_uv_virtual_environment()
        verify_runtime_identity(plan, cache_root=args.cache_root.resolve())
        workers = (
            int(args.workers)
            if args.workers is not None
            else int(plan["execution_config"]["cpu_slot_count"])
        )
        return _spawn_workers(
            run_root=args.run_root.resolve(),
            cache_root=args.cache_root.resolve(),
            plan=plan,
            workers=workers,
        )
    if args.command == "audit":
        audit = audit_grid(args.run_root.resolve(), plan)
        if args.output is not None:
            output = _validated_audit_output_path(
                args.run_root.resolve(), args.output
            )
            _atomic_json(output, audit)
        print(json.dumps(audit, indent=2, sort_keys=True))
        return 0 if audit["complete"] else 2
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONTROLS",
    "DATASET_FOLDS",
    "DATASET_SUBJECTS",
    "EXPECTED_JOB_COUNT",
    "FITS_PER_CONTROL",
    "CPUUnavailable",
    "Claim",
    "ClaimUnavailable",
    "ControlGridError",
    "DiskStatus",
    "DiskUnavailable",
    "EstimatorCapabilityError",
    "Job",
    "PreparedFeatures",
    "RegistryDriftError",
    "assemble_plan",
    "audit_grid",
    "build_plan",
    "commit_job_output",
    "iter_jobs",
    "load_plan",
    "main",
    "plan_sha256",
    "probe_disk",
    "select_refit_predict",
    "strict_load",
    "validate_completion",
    "verify_runtime_identity",
    "write_or_validate_plan",
]
