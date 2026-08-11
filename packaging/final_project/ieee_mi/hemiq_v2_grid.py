"""Formal score-blind runner for the HemiQ-Field harmonized-v2 adaptation.

The runner is deliberately dataset-specific.  It accepts only the immutable
harmonized BNCI2014-004 supplied-bipolar cache, subjects 1--9, chronological
fold 0, and seeds 7/17/27/37/47.  Producers publish test row numbers and
probabilities but never held-out labels or metrics.

Formal execution is split into four authority boundaries:

* an immutable, outcome-free 45-job plan;
* one synthetic CUDA preflight receipt per physical GPU used by a worker;
* fenced filesystem claims held for the full duration of a job; and
* crash-atomic result-directory publication under the project GPU lease guard.

The companion :mod:`ieee_mi.hemiq_v2_analysis` acquires the exclusive
publication fence, performs a label-free exact-grid audit, and only then opens
held-out labels.
"""

from __future__ import annotations

import argparse
import copy
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
import shutil
import stat
import subprocess
import sys
import time
import uuid
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from . import data as data_module
from . import project_gpu_leases
from .config import (
    PREPROCESSING,
    coordinate_contract_for_dataset,
    dataset_spec,
    preprocessing_for_dataset,
)
from .hemiq_v2_model import (
    CONFIG_RELATIVE_PATH,
    CONFIG_SCHEMA,
    EXPECTED_N_TIMES,
    EXPECTED_SFREQ,
    HemiQHarmonizedV2Classifier,
    derive_hemiq_views,
    hemiq_view_contract,
    load_frozen_config,
)


PLAN_SCHEMA = "ieee-mi-hemiq-harmonized-v2-plan-v1"
PREFLIGHT_SCHEMA = "ieee-mi-hemiq-harmonized-v2-cuda-preflight-v1"
CLAIM_SCHEMA = "ieee-mi-hemiq-harmonized-v2-claim-v1"
RECORD_SCHEMA = "ieee-mi-hemiq-harmonized-v2-score-blind-record-v1"
COMPLETION_SCHEMA = "ieee-mi-hemiq-harmonized-v2-completion-v1"
AUDIT_SCHEMA = "ieee-mi-hemiq-harmonized-v2-quiescent-audit-v1"
FAILURE_SCHEMA = "ieee-mi-hemiq-harmonized-v2-failure-v1"
ANALYSIS_FENCE_SCHEMA = "ieee-mi-hemiq-harmonized-v2-analysis-fence-v1"

TRACK = "hemiq_field_harmonized_v2_development"
TRACK_SCOPE = "hemiq-v2"
MODEL_ID = "architecture.hemi_q_field"
DATASET = "bnci2014_004"
MONTAGE_PROFILE = "harmonized"
SUBJECTS = tuple(range(1, 10))
FOLDS = (0,)
SEEDS = (7, 17, 27, 37, 47)
EXPECTED_JOBS = 45
MINIMUM_FREE_GIB = 50.0
MAX_GPU_WORKERS = 3
REQUIRED_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
FORMAL_PUBLICATION_MODE = "formal_publishable"
TEST_PUBLICATION_MODE = "test_nonpublishable"

RUN_DIRECTORIES = frozenset(
    {
        ".authority",
        "claims",
        "failures",
        "preflight",
        "quarantine",
        "records",
        "staging",
    }
)
RESULT_FILENAMES = frozenset(
    {"record.json", "predictions.npz", "completion.json"}
)
AUTHORITY_FILENAMES = frozenset({"coordination.lock"})
SOURCE_FILES = (
    "configs/hemiq/hemiq_field_harmonized_v2.json",
    "deepnet/__init__.py",
    "deepnet/augment.py",
    "deepnet/cameo_net.py",
    "deepnet/config.py",
    "deepnet/data.py",
    "deepnet/hemi_q_field_net.py",
    "deepnet/spd.py",
    "ieee_mi/__init__.py",
    "ieee_mi/config.py",
    "ieee_mi/data.py",
    "ieee_mi/hemiq_v2_analysis.py",
    "ieee_mi/hemiq_v2_grid.py",
    "ieee_mi/hemiq_v2_model.py",
    "ieee_mi/project_gpu_leases.py",
)
DIRECT_DEPENDENCIES = {
    "numpy": "2.4.4",
    "scikit-learn": "1.8.0",
    "scipy": "1.18.0",
    "torch": "2.6.0+cu124",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GPU_UUID_RE = re.compile(
    r"^GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
JOB_ID_RE = re.compile(
    r"^bnci2014_004-s(?P<subject>[0-9]{3})-f00-seed(?P<seed>[0-9]{10})$"
)
FORBIDDEN_RESULT_KEYS = frozenset(
    {
        "accuracy",
        "balanced_accuracy",
        "cohen_kappa",
        "confusion_matrix",
        "correct",
        "f1",
        "kappa",
        "label",
        "labels",
        "macro_f1",
        "score",
        "scores",
        "test_label",
        "test_labels",
        "y_pred",
        "y_test",
        "y_true",
    }
)
STAT_FINGERPRINT_FIELDS = (
    "st_dev",
    "st_ino",
    "st_mode",
    "st_nlink",
    "st_size",
    "st_mtime_ns",
    "st_ctime_ns",
)


class HemiQGridError(RuntimeError):
    """Raised when a formal HemiQ-v2 execution invariant is violated."""


class ClaimUnavailable(HemiQGridError):
    """Raised when another live worker owns one job."""


class AnalysisActive(ClaimUnavailable):
    """Raised when the analysis publication fence excludes workers."""


@dataclass(frozen=True, order=True)
class Job:
    subject: int
    fold: int
    seed: int
    dataset: str = DATASET
    model_id: str = MODEL_ID

    @property
    def job_id(self) -> str:
        return (
            f"{self.dataset}-s{self.subject:03d}-f{self.fold:02d}-"
            f"seed{self.seed:010d}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "fold": self.fold,
            "model_id": self.model_id,
            "seed": self.seed,
            "subject": self.subject,
        }


@dataclass
class Claim:
    job: Job
    path: Path
    descriptor: int
    st_dev: int
    st_ino: int
    nonce: str
    value: Mapping[str, Any]


@dataclass
class AnalysisFence:
    run_root: Path
    root_descriptor: int
    root_st_dev: int
    root_st_ino: int
    authority_descriptor: int
    authority_st_dev: int
    authority_st_ino: int
    lock_path: Path
    descriptor: int
    st_dev: int
    st_ino: int
    marker_path: Path
    marker_st_dev: int
    marker_st_ino: int
    marker_value: Mapping[str, Any]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _absolute(path: Path | str) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _is_exact_int(value: Any, *, minimum: int | None = None) -> bool:
    return (
        type(value) is int
        and (minimum is None or value >= minimum)
    )


def _is_finite_number(value: Any, *, minimum: float | None = None) -> bool:
    return (
        type(value) in {int, float}
        and math.isfinite(float(value))
        and (minimum is None or float(value) >= minimum)
    )


def _same_typed_value(observed: Any, expected: Any) -> bool:
    """Compare JSON-like values without Python's bool/int aliasing."""

    if type(observed) is not type(expected):
        return False
    if isinstance(expected, Mapping):
        return (
            set(observed) == set(expected)
            and all(
                _same_typed_value(observed[key], expected[key])
                for key in expected
            )
        )
    if isinstance(expected, list):
        return (
            len(observed) == len(expected)
            and all(
                _same_typed_value(left, right)
                for left, right in zip(observed, expected, strict=True)
            )
        )
    return bool(observed == expected)


def _stat_fingerprint(observed: os.stat_result) -> tuple[int, ...]:
    return tuple(
        int(getattr(observed, field))
        for field in STAT_FINGERPRINT_FIELDS
    )


def _exact_keys(value: Any, expected: set[str] | frozenset[str], *, path: str) -> None:
    if not isinstance(value, Mapping) or set(value) != set(expected):
        observed = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise HemiQGridError(f"{path} has an unknown schema: {observed}")


def _reject_json_constant(raw: str) -> None:
    raise HemiQGridError(f"JSON contains a non-finite constant: {raw}")


def _unique_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise HemiQGridError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _strict_json_bytes(payload: bytes, *, source: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_unique_json_pairs,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HemiQGridError(f"{source} is not strict JSON") from error
    if not isinstance(value, dict):
        raise HemiQGridError(f"{source} is not a JSON object")
    if _canonical_bytes(value) != payload:
        raise HemiQGridError(f"{source} is not canonical JSON")
    return value


def _directory_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _open_directory_absolute(path: Path | str) -> int:
    """Open an absolute directory by descriptor-pinned no-follow traversal."""

    absolute = _absolute(path)
    descriptor = os.open(absolute.anchor, _directory_flags())
    try:
        for component in absolute.parts[1:]:
            child = os.open(component, _directory_flags(), dir_fd=descriptor)
            observed = os.fstat(child)
            if not stat.S_ISDIR(observed.st_mode):
                os.close(child)
                raise HemiQGridError(f"unsafe directory component: {absolute}")
            os.close(descriptor)
            descriptor = child
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _safe_mkdir(path: Path | str, *, mode: int = 0o700) -> Path:
    """Create a directory hierarchy without following an ancestor symlink."""

    absolute = _absolute(path)
    descriptor = os.open(absolute.anchor, _directory_flags())
    try:
        for component in absolute.parts[1:]:
            try:
                child = os.open(component, _directory_flags(), dir_fd=descriptor)
            except FileNotFoundError:
                os.mkdir(component, mode=mode, dir_fd=descriptor)
                os.fsync(descriptor)
                child = os.open(component, _directory_flags(), dir_fd=descriptor)
            observed = os.fstat(child)
            if not stat.S_ISDIR(observed.st_mode):
                os.close(child)
                raise HemiQGridError(f"unsafe directory component: {absolute}")
            os.close(descriptor)
            descriptor = child
    finally:
        os.close(descriptor)
    return absolute


def _anchored_lstat(path: Path | str) -> os.stat_result:
    absolute = _absolute(path)
    parent = _open_directory_absolute(absolute.parent)
    try:
        return os.stat(absolute.name, dir_fd=parent, follow_symlinks=False)
    finally:
        os.close(parent)


def _path_exists(path: Path | str) -> bool:
    try:
        _anchored_lstat(path)
        return True
    except FileNotFoundError:
        return False


def _assert_unique_regular(status: os.stat_result, *, source: str, allow_empty: bool = False) -> None:
    if (
        not stat.S_ISREG(status.st_mode)
        or status.st_nlink != 1
        or (status.st_size <= 0 and not allow_empty)
    ):
        raise HemiQGridError(f"{source} is not a unique regular file")


def _read_unique_regular_bytes(path: Path | str, *, allow_empty: bool = False) -> bytes:
    """Read the exact opened leaf and reject symlinks, aliases, and mutation."""

    absolute = _absolute(path)
    parent = _open_directory_absolute(absolute.parent)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptor = -1
    try:
        before_path = os.stat(
            absolute.name, dir_fd=parent, follow_symlinks=False
        )
        _assert_unique_regular(
            before_path, source=str(absolute), allow_empty=allow_empty
        )
        descriptor = os.open(absolute.name, flags, dir_fd=parent)
        before = os.fstat(descriptor)
        _assert_unique_regular(before, source=str(absolute), allow_empty=allow_empty)
        stable = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(getattr(before, key) != getattr(before_path, key) for key in stable):
            raise HemiQGridError(f"{absolute} changed while opening")
        chunks: list[bytes] = []
        remaining = int(before.st_size)
        while remaining:
            try:
                chunk = os.read(descriptor, min(1024 * 1024, remaining))
            except InterruptedError:
                continue
            if not chunk:
                raise HemiQGridError(f"{absolute} was truncated while reading")
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        after_path = os.stat(
            absolute.name, dir_fd=parent, follow_symlinks=False
        )
        for observed in (after, after_path):
            if any(getattr(before, key) != getattr(observed, key) for key in stable):
                raise HemiQGridError(f"{absolute} changed while reading")
        return b"".join(chunks)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent)


def _strict_json_file(path: Path | str) -> dict[str, Any]:
    return _strict_json_bytes(
        _read_unique_regular_bytes(path), source=str(_absolute(path))
    )


def _fsync_directory(path: Path | str) -> None:
    descriptor = _open_directory_absolute(path)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_exclusive_at(
    parent_descriptor: int,
    name: str,
    payload: bytes,
    *,
    mode: int = 0o600,
) -> os.stat_result:
    if "/" in name or name in {"", ".", ".."}:
        raise ValueError("exclusive leaf name is invalid")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(name, flags, mode, dir_fd=parent_descriptor)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise HemiQGridError("short artifact write")
            offset += written
        os.fsync(descriptor)
        observed = os.fstat(descriptor)
        _assert_unique_regular(
            observed, source=name, allow_empty=(len(payload) == 0)
        )
        if observed.st_size != len(payload):
            raise HemiQGridError("artifact size differs after write")
        return observed
    finally:
        os.close(descriptor)


def _rename_noreplace_at(
    source_parent: int,
    source_name: str,
    destination_parent: int,
    destination_name: str,
) -> None:
    """Atomically rename one anchored leaf without replacing a destination."""

    libc = ctypes.CDLL(None, use_errno=True)
    source = os.fsencode(source_name)
    destination = os.fsencode(destination_name)
    result: int
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
            source_parent, source, destination_parent, destination, 0x00000004
        )
    else:
        raise HemiQGridError("atomic no-replace rename is unavailable")
    if result != 0:
        observed_errno = ctypes.get_errno()
        if observed_errno in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(
                observed_errno, os.strerror(observed_errno), destination_name
            )
        raise OSError(observed_errno, os.strerror(observed_errno), destination_name)


def _atomic_bytes(path: Path | str, payload: bytes) -> tuple[int, int]:
    """Publish immutable bytes or leave no canonical leaf on caught failure.

    The staged leaf is complete, fsynced, and already read-only before its
    no-replace rename.  If the parent-directory fsync fails after that rename,
    the exact inode is first moved back to its same-parent stage name before
    cleanup.  Thus a caller that observes an exception cannot accidentally
    accept a writable or half-authoritative canonical JSON leaf.
    """

    absolute = _absolute(path)
    parent = _open_directory_absolute(absolute.parent)
    staged = f".{absolute.name}.stage-{uuid.uuid4().hex}"
    staged_identity: tuple[int, int] | None = None
    renamed = False
    try:
        written = _write_exclusive_at(parent, staged, payload, mode=0o400)
        staged_identity = (int(written.st_dev), int(written.st_ino))
        _rename_noreplace_at(parent, staged, parent, absolute.name)
        renamed = True
        os.fsync(parent)
        published_payload, published_status = _read_unique_regular_at(
            parent,
            absolute.name,
            source=str(absolute),
        )
        published_identity = (
            int(published_status.st_dev),
            int(published_status.st_ino),
        )
        if (
            staged_identity is None
            or published_identity != staged_identity
            or stat.S_IMODE(published_status.st_mode) != 0o400
            or published_payload != payload
        ):
            raise HemiQGridError(
                "atomic publication differs from its staged inode or bytes"
            )
        return published_identity
    except Exception:
        try:
            canonical = os.stat(
                absolute.name,
                dir_fd=parent,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            renamed = False
        else:
            canonical_identity = (
                int(canonical.st_dev),
                int(canonical.st_ino),
            )
            if staged_identity is not None and (
                canonical_identity == staged_identity
            ):
                try:
                    _rename_noreplace_at(
                        parent,
                        absolute.name,
                        parent,
                        staged,
                    )
                except Exception:
                    # A fault injector may perform the rename and then raise.
                    # Recheck the authoritative name before deciding whether
                    # rollback itself failed.
                    try:
                        still_visible = os.stat(
                            absolute.name,
                            dir_fd=parent,
                            follow_symlinks=False,
                        )
                    except FileNotFoundError:
                        pass
                    else:
                        if (
                            int(still_visible.st_dev),
                            int(still_visible.st_ino),
                        ) == staged_identity:
                            raise HemiQGridError(
                                "atomic publication could not hide its "
                                "canonical inode after failure"
                            )
                        raise HemiQGridError(
                            "atomic publication authority was replaced "
                            "during rollback"
                        )
                renamed = False
                try:
                    os.fsync(parent)
                except OSError:
                    # The canonical name is already hidden. Preserve the
                    # original publication failure even if durability cannot
                    # be confirmed by this process.
                    pass
            elif renamed:
                raise HemiQGridError(
                    "atomic publication authority changed after rename"
                )
        try:
            os.unlink(staged, dir_fd=parent)
        except FileNotFoundError:
            pass
        raise
    finally:
        os.close(parent)


def _atomic_json(path: Path | str, value: Mapping[str, Any]) -> bytes:
    payload = _canonical_bytes(dict(value))
    _atomic_bytes(path, payload)
    return payload


def _atomic_json_with_identity(
    path: Path | str, value: Mapping[str, Any]
) -> tuple[bytes, tuple[int, int]]:
    """Publish canonical JSON and return its exact bytes and inode identity."""

    payload = _canonical_bytes(dict(value))
    return payload, _atomic_bytes(path, payload)


def _chmod_unique_regular(path: Path | str, mode: int) -> None:
    absolute = _absolute(path)
    parent = _open_directory_absolute(absolute.parent)
    descriptor = -1
    try:
        path_status = os.stat(
            absolute.name, dir_fd=parent, follow_symlinks=False
        )
        _assert_unique_regular(path_status, source=str(absolute))
        descriptor = os.open(
            absolute.name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent,
        )
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (
            path_status.st_dev,
            path_status.st_ino,
        ):
            raise HemiQGridError(f"{absolute} changed before chmod")
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
        final = os.stat(
            absolute.name, dir_fd=parent, follow_symlinks=False
        )
        if (
            (final.st_dev, final.st_ino) != (opened.st_dev, opened.st_ino)
            or stat.S_IMODE(final.st_mode) != mode
        ):
            raise HemiQGridError(f"{absolute} changed during chmod")
        os.fsync(parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent)


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(_canonical_bytes(list(array.shape)))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _array_manifest(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": _array_sha256(array),
    }


def _rows_manifest(value: np.ndarray | Sequence[int]) -> dict[str, Any]:
    rows = np.ascontiguousarray(value, dtype=np.int64)
    if rows.ndim != 1:
        raise ValueError("row indices must be one-dimensional")
    return _array_manifest(rows)


def _load_npz_exact(
    payload: bytes,
    *,
    expected_keys: Sequence[str],
    source: str,
) -> dict[str, np.ndarray]:
    """Load NPZ only after exact raw-ZIP and NumPy member validation."""

    expected_names = [f"{key}.npy" for key in expected_keys]
    try:
        with zipfile.ZipFile(io.BytesIO(payload), mode="r") as raw_archive:
            infos = raw_archive.infolist()
            raw_names = [info.filename for info in infos]
            if (
                raw_names != expected_names
                or len(raw_names) != len(set(raw_names))
                or any(
                    info.is_dir()
                    or info.flag_bits & 0x1
                    or info.file_size <= 0
                    for info in infos
                )
            ):
                raise HemiQGridError(
                    f"{source} raw ZIP members are ambiguous or non-canonical"
                )
    except (zipfile.BadZipFile, OSError) as error:
        raise HemiQGridError(f"{source} is not a valid NPZ archive") from error
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if archive.files != list(expected_keys):
                raise HemiQGridError(
                    f"{source} NumPy member roster is not exact and ordered"
                )
            return {name: archive[name].copy() for name in expected_keys}
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as error:
        raise HemiQGridError(f"{source} contains an invalid NPY member") from error


def _recursive_forbidden_keys(value: Any, path: str = "record") -> list[str]:
    violations: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            token = str(raw_key).strip().lower()
            if token in FORBIDDEN_RESULT_KEYS:
                violations.append(f"{path}.{raw_key}")
            violations.extend(
                _recursive_forbidden_keys(child, f"{path}.{raw_key}")
            )
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            violations.extend(
                _recursive_forbidden_keys(child, f"{path}[{index}]")
            )
    return violations


def _job_from_mapping(value: Any) -> Job:
    _exact_keys(
        value,
        {"dataset", "fold", "model_id", "seed", "subject"},
        path="job",
    )
    if (
        value["dataset"] != DATASET
        or value["model_id"] != MODEL_ID
        or not _is_exact_int(value["subject"])
        or not _is_exact_int(value["fold"])
        or not _is_exact_int(value["seed"], minimum=0)
    ):
        raise HemiQGridError("job identity has invalid values")
    return Job(
        subject=value["subject"],
        fold=value["fold"],
        seed=value["seed"],
    )


def expected_jobs() -> tuple[Job, ...]:
    jobs = tuple(
        Job(subject=subject, fold=fold, seed=seed)
        for subject in SUBJECTS
        for fold in FOLDS
        for seed in SEEDS
    )
    if len(jobs) != EXPECTED_JOBS or len({job.job_id for job in jobs}) != len(jobs):
        raise RuntimeError("internal HemiQ-v2 cardinality drifted")
    return jobs


def parse_job_id(value: str) -> Job:
    match = JOB_ID_RE.fullmatch(value)
    if match is None:
        raise HemiQGridError(f"invalid HemiQ-v2 job ID: {value!r}")
    job = Job(
        subject=int(match.group("subject")),
        fold=0,
        seed=int(match.group("seed")),
    )
    if job not in expected_jobs() or job.job_id != value:
        raise HemiQGridError(f"job ID lies outside the formal grid: {value!r}")
    return job


def plan_sha256(plan: Mapping[str, Any]) -> str:
    value = copy.deepcopy(dict(plan))
    value.pop("plan_sha256", None)
    return _sha256_bytes(_canonical_bytes(value))


def _cache_path(cache_root: Path, subject: int) -> Path:
    return (
        _absolute(cache_root)
        / str(PREPROCESSING["schema"])
        / MONTAGE_PROFILE
        / DATASET
        / f"subject_{subject:03d}.npz"
    )


def _bnci004_rows(
    cache: Mapping[str, Any],
    *,
    validate_held_out_classes: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derive the chronological split, optionally validating test outcomes.

    Formal workers call this with ``False``.  Their only held-out operation is
    the one probability call; test labels are first indexed by the analyzer
    after a quiescent score-blind audit.
    """

    sessions = np.asarray(cache["sessions"]).astype(str)
    y = np.asarray(cache["y"], dtype=np.int64)
    if sessions.shape != y.shape or sessions.ndim != 1:
        raise HemiQGridError("BNCI2014-004 cache row vectors are invalid")
    rows = np.arange(len(sessions), dtype=np.int64)
    train = rows[np.isin(sessions, ("0train", "1train"))]
    validation = rows[sessions == "2train"]
    test = rows[np.isin(sessions, ("3test", "4test"))]
    if (
        not len(train)
        or not len(validation)
        or not len(test)
        or len(train) + len(validation) + len(test) != len(rows)
        or any(
            np.intersect1d(first, second).size
            for first, second in (
                (train, validation),
                (train, test),
                (validation, test),
            )
        )
        or set(y[train].tolist()) != {0, 1}
        or set(y[validation].tolist()) != {0, 1}
    ):
        raise HemiQGridError("BNCI2014-004 chronological split is invalid")
    if validate_held_out_classes and set(y[test].tolist()) != {0, 1}:
        raise HemiQGridError("BNCI2014-004 held-out partition is invalid")
    return train, validation, test


def _cache_identity(cache_root: Path, subject: int) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _cache_path(cache_root, subject)
    payload = _read_unique_regular_bytes(path)
    _load_npz_exact(
        payload,
        expected_keys=(
            "x",
            "y",
            "positions",
            "channel_names",
            "sessions",
            "runs",
            "identity",
        ),
        source=str(path),
    )
    cache = data_module.load_subject_cache(
        DATASET,
        subject,
        cache_root=cache_root,
        montage_profile=MONTAGE_PROFILE,
    )
    channels = tuple(str(value) for value in cache["channel_names"].tolist())
    if channels != ("C3", "Cz", "C4"):
        raise HemiQGridError("HemiQ-v2 requires supplied-bipolar C3,Cz,C4 order")
    x = np.asarray(cache["x"])
    if x.dtype != np.float32 or x.shape[1:] != (3, EXPECTED_N_TIMES):
        raise HemiQGridError("BNCI2014-004 cache geometry is incompatible")
    identity = {
        "archive": {
            "relative_path": str(path.relative_to(_absolute(cache_root))),
            "sha256": _sha256_bytes(payload),
            "size_bytes": len(payload),
        },
        "arrays": {
            name: _array_manifest(np.asarray(cache[name]))
            for name in ("channel_names", "positions", "runs", "sessions", "x", "y")
        },
        "embedded_identity": copy.deepcopy(cache["identity"]),
        "montage_interpretation": (
            "supplied bipolar derivations in exact C3,Cz,C4 order; "
            "nominal coordinates are not point-electrode geometry"
        ),
    }
    train, validation, test = _bnci004_rows(
        cache, validate_held_out_classes=False
    )
    source = np.sort(np.concatenate((train, validation)))
    split = {
        "cache_array_sha256": str(cache["identity"]["array_sha256"]),
        "dataset": DATASET,
        "fold": 0,
        "partitions": {
            "source": _rows_manifest(source),
            "test": _rows_manifest(test),
            "train": _rows_manifest(train),
            "validation": _rows_manifest(validation),
        },
        "subject": subject,
        "trial_count": len(x),
    }
    return identity, split


def collect_cache_and_split_identity(
    cache_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cache_identity: dict[str, Any] = {}
    split_identity: dict[str, Any] = {}
    for subject in SUBJECTS:
        subject_key = f"{DATASET}:s{subject:03d}"
        split_key = f"{subject_key}:f00"
        cache_identity[subject_key], split_identity[split_key] = _cache_identity(
            cache_root, subject
        )
    return cache_identity, split_identity


def collect_source_identity(project_root: Path) -> dict[str, dict[str, Any]]:
    root = _absolute(project_root)
    identity: dict[str, dict[str, Any]] = {}
    for relative in SOURCE_FILES:
        payload = _read_unique_regular_bytes(root / relative)
        identity[relative] = {
            "sha256": _sha256_bytes(payload),
            "size_bytes": len(payload),
        }
    return identity


def _run_text_command(arguments: Sequence[str]) -> str:
    try:
        completed = subprocess.run(
            list(arguments),
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise HemiQGridError(f"command failed: {list(arguments)!r}") from error
    return completed.stdout.strip()


def collect_uv_identity(project_root: Path) -> dict[str, Any]:
    root = _absolute(project_root)
    pyproject = _read_unique_regular_bytes(root / "pyproject.toml")
    lock = _read_unique_regular_bytes(root / "uv.lock")
    python_version = _read_unique_regular_bytes(root / ".python-version")
    virtual_environment = os.environ.get("VIRTUAL_ENV", "")
    if not virtual_environment:
        raise HemiQGridError("formal HemiQ-v2 execution requires a UV virtual environment")
    venv = _absolute(virtual_environment)
    executable = _absolute(sys.executable)
    try:
        executable.relative_to(venv)
    except ValueError as error:
        raise HemiQGridError("Python executable is outside VIRTUAL_ENV") from error
    if sys.prefix == sys.base_prefix or _absolute(sys.prefix) != venv:
        raise HemiQGridError("active Python is not the declared isolated venv")
    installed: dict[str, str] = {}
    for name, expected in DIRECT_DEPENDENCIES.items():
        try:
            observed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise HemiQGridError(f"required dependency is absent: {name}") from error
        if observed != expected:
            raise HemiQGridError(
                f"{name} version differs: expected {expected}, observed {observed}"
            )
        installed[name] = observed
    return {
        "active_venv": str(venv),
        "direct_dependencies": installed,
        "lock": {
            "relative_path": "uv.lock",
            "sha256": _sha256_bytes(lock),
            "size_bytes": len(lock),
        },
        "project": {
            "relative_path": "pyproject.toml",
            "sha256": _sha256_bytes(pyproject),
            "size_bytes": len(pyproject),
        },
        "python": {
            "executable": str(executable),
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "python_version_file": {
            "relative_path": ".python-version",
            "sha256": _sha256_bytes(python_version),
            "size_bytes": len(python_version),
        },
        "uv_version": _run_text_command(("uv", "--version")),
    }


def _nvidia_inventory() -> list[dict[str, Any]]:
    text = _run_text_command(
        (
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        )
    )
    rows: list[dict[str, Any]] = []
    for raw_line in text.splitlines():
        fields = [value.strip() for value in raw_line.split(",")]
        if len(fields) != 5:
            raise HemiQGridError("nvidia-smi inventory has an unknown shape")
        raw_index, gpu_uuid, name, driver, raw_memory = fields
        if (
            not raw_index.isdigit()
            or GPU_UUID_RE.fullmatch(gpu_uuid) is None
            or not name
            or not driver
        ):
            raise HemiQGridError("nvidia-smi inventory contains invalid values")
        try:
            memory_mib = int(raw_memory)
        except ValueError as error:
            raise HemiQGridError("nvidia-smi memory is not an integer") from error
        if memory_mib <= 0:
            raise HemiQGridError("nvidia-smi memory is not positive")
        rows.append(
            {
                "driver_version": driver,
                "index": int(raw_index),
                "memory_total_mib": memory_mib,
                "name": name,
                "uuid": gpu_uuid,
            }
        )
    if not rows or len({row["uuid"] for row in rows}) != len(rows):
        raise HemiQGridError("physical GPU inventory is empty or duplicated")
    return sorted(rows, key=lambda row: row["uuid"])


def collect_environment_identity() -> dict[str, Any]:
    import scipy
    import torch

    gpus = _nvidia_inventory()
    drivers = {row["driver_version"] for row in gpus}
    if len(drivers) != 1:
        raise HemiQGridError("physical GPUs report different driver versions")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise HemiQGridError("formal HemiQ-v2 environment has no CUDA device")
    if not isinstance(torch.version.cuda, str) or not torch.version.cuda:
        raise HemiQGridError("Torch CUDA build identity is unavailable")
    return {
        "cuda": {
            "cublas_workspace_config": REQUIRED_CUBLAS_WORKSPACE_CONFIG,
            "cudnn_version": int(torch.backends.cudnn.version()),
            "torch_cuda_version": torch.version.cuda,
        },
        "hardware": {
            "architecture": platform.machine(),
            "gpus": gpus,
            "nvidia_driver_version": next(iter(drivers)),
            "system": platform.system(),
        },
        "libraries": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
        },
        "parallelism": {
            "distributed_execution": "forbidden",
            "maximum_project_gpu_workers": MAX_GPU_WORKERS,
        },
    }


def _config_identity(project_root: Path) -> dict[str, Any]:
    _, identity = load_frozen_config(
        project_root, seed=SEEDS[0], device="cuda:0"
    )
    value = copy.deepcopy(identity)
    value["runtime_overrides"] = ["device", "seed"]
    return value


def assemble_plan(
    *,
    cache_identity: Mapping[str, Any],
    split_identity: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    uv_identity: Mapping[str, Any],
    environment_identity: Mapping[str, Any],
    config_identity: Mapping[str, Any],
    view_contract: Mapping[str, Any],
    publication_mode: str = FORMAL_PUBLICATION_MODE,
) -> dict[str, Any]:
    jobs = expected_jobs()
    plan: dict[str, Any] = {
        "analysis_protocol": {
            "label_boundary": (
                "held-out labels may be indexed only after an exclusive-fence "
                "exact quiescent score-blind audit"
            ),
            "primary_aggregation": (
                "fold concatenation, seed mean within participant, equal "
                "participant mean for BNCI2014-004"
            ),
        },
        "cache_identity": copy.deepcopy(dict(cache_identity)),
        "configuration_identity": copy.deepcopy(dict(config_identity)),
        "created_at": "pending-atomic-publication",
        "dataset_contract": {
            "dataset": DATASET,
            "events": ["left_hand", "right_hand"],
            "folds": list(FOLDS),
            "montage_interpretation": (
                "three supplied bipolar derivations; nominal coordinates are "
                "never used as point-electrode geometry"
            ),
            "montage_profile": MONTAGE_PROFILE,
            "n_classes": 2,
            "preprocessing": preprocessing_for_dataset(DATASET),
            "subjects": list(SUBJECTS),
        },
        "environment_identity": copy.deepcopy(dict(environment_identity)),
        "evidence_scope": (
            "all nine subjects are opened development evidence; historical "
            "HemiQ confirmation artifacts are neither read nor rewritten"
        ),
        "execution": {
            "cublas_workspace_config": REQUIRED_CUBLAS_WORKSPACE_CONFIG,
            "device": "cuda:0",
            "distributed_execution": "forbidden",
            "maximum_concurrent_gpu_workers": MAX_GPU_WORKERS,
            "minimum_free_gib": MINIMUM_FREE_GIB,
            "physical_gpu_uuid_roster": sorted(
                str(row["uuid"])
                for row in environment_identity["hardware"]["gpus"]
            ),
        },
        "jobs": [job.as_dict() for job in jobs],
        "model_id": MODEL_ID,
        "n_jobs": len(jobs),
        "protocol": {
            "prediction": (
                "exactly one predict_proba call on held-out sessions 3test/4test; "
                "publish rows and probabilities without labels or metrics"
            ),
            "refit": (
                "fresh seeded reset; refit scaler and teacher on train plus "
                "validation; optimize exactly the selected zero-or-more epochs"
            ),
            "selection": (
                "fit scaler and teacher on train only; validation BCE selects "
                "duration and the untrained zero-epoch checkpoint is eligible"
            ),
            "teacher_schedule": (
                "retain the frozen 240-epoch decay horizon during reset/refit"
            ),
        },
        "publication_mode": publication_mode,
        "schema": PLAN_SCHEMA,
        "seeds": list(SEEDS),
        "source_identity": copy.deepcopy(dict(source_identity)),
        "split_identity": copy.deepcopy(dict(split_identity)),
        "track": TRACK,
        "uv_identity": copy.deepcopy(dict(uv_identity)),
        "view_contract": copy.deepcopy(dict(view_contract)),
    }
    plan["plan_sha256"] = plan_sha256(plan)
    validate_plan(plan)
    return plan


def build_plan(*, project_root: Path, cache_root: Path) -> dict[str, Any]:
    cache_identity, split_identity = collect_cache_and_split_identity(cache_root)
    return assemble_plan(
        cache_identity=cache_identity,
        split_identity=split_identity,
        source_identity=collect_source_identity(project_root),
        uv_identity=collect_uv_identity(project_root),
        environment_identity=collect_environment_identity(),
        config_identity=_config_identity(project_root),
        view_contract=hemiq_view_contract(),
    )


def _validate_array_manifest(
    value: Any,
    *,
    path: str,
    expected_shape: Sequence[int] | None = None,
    expected_dtype: str | None = None,
) -> None:
    _exact_keys(value, {"dtype", "shape", "sha256"}, path=path)
    if (
        not isinstance(value["dtype"], str)
        or not isinstance(value["shape"], list)
        or any(not _is_exact_int(item, minimum=0) for item in value["shape"])
        or not isinstance(value["sha256"], str)
        or SHA256_RE.fullmatch(value["sha256"]) is None
    ):
        raise HemiQGridError(f"{path} contains invalid values")
    if expected_shape is not None and value["shape"] != list(expected_shape):
        raise HemiQGridError(f"{path} has an unexpected shape")
    if expected_dtype is not None and value["dtype"] != expected_dtype:
        raise HemiQGridError(f"{path} has an unexpected dtype")


def _validate_file_identity(value: Any, *, path: str) -> None:
    _exact_keys(value, {"sha256", "size_bytes"}, path=path)
    if (
        not isinstance(value["sha256"], str)
        or SHA256_RE.fullmatch(value["sha256"]) is None
        or not _is_exact_int(value["size_bytes"], minimum=1)
    ):
        raise HemiQGridError(f"{path} contains invalid values")


def validate_plan(plan: Mapping[str, Any]) -> None:
    expected_top = {
        "analysis_protocol",
        "cache_identity",
        "configuration_identity",
        "created_at",
        "dataset_contract",
        "environment_identity",
        "evidence_scope",
        "execution",
        "jobs",
        "model_id",
        "n_jobs",
        "plan_sha256",
        "protocol",
        "publication_mode",
        "schema",
        "seeds",
        "source_identity",
        "split_identity",
        "track",
        "uv_identity",
        "view_contract",
    }
    _exact_keys(plan, expected_top, path="plan")
    if (
        plan["schema"] != PLAN_SCHEMA
        or plan["track"] != TRACK
        or plan["model_id"] != MODEL_ID
        or plan["created_at"] != "pending-atomic-publication"
        or plan["publication_mode"]
        not in {FORMAL_PUBLICATION_MODE, TEST_PUBLICATION_MODE}
        or not isinstance(plan["plan_sha256"], str)
        or SHA256_RE.fullmatch(plan["plan_sha256"]) is None
        or plan_sha256(plan) != plan["plan_sha256"]
        or not _is_exact_int(plan["n_jobs"], minimum=1)
        or plan["n_jobs"] != EXPECTED_JOBS
        or not _same_typed_value(plan["seeds"], list(SEEDS))
    ):
        raise HemiQGridError("plan identity or cardinality is invalid")

    expected_roster = [job.as_dict() for job in expected_jobs()]
    if plan["jobs"] != expected_roster:
        raise HemiQGridError("plan job roster is not the exact formal 45-job grid")
    if len({_job_from_mapping(value).job_id for value in plan["jobs"]}) != EXPECTED_JOBS:
        raise HemiQGridError("plan job roster contains duplicates")

    _exact_keys(
        plan["dataset_contract"],
        {
            "dataset",
            "events",
            "folds",
            "montage_interpretation",
            "montage_profile",
            "n_classes",
            "preprocessing",
            "subjects",
        },
        path="plan.dataset_contract",
    )
    contract = plan["dataset_contract"]
    spec = dataset_spec(DATASET)
    if (
        contract["dataset"] != DATASET
        or not _same_typed_value(contract["events"], list(spec.events))
        or not _same_typed_value(contract["folds"], [0])
        or contract["montage_profile"] != MONTAGE_PROFILE
        or not _is_exact_int(contract["n_classes"], minimum=2)
        or contract["n_classes"] != 2
        or not _same_typed_value(
            contract["preprocessing"],
            preprocessing_for_dataset(DATASET),
        )
        or not _same_typed_value(contract["subjects"], list(SUBJECTS))
        or "bipolar" not in str(contract["montage_interpretation"]).lower()
        or "point-electrode" not in str(contract["montage_interpretation"]).lower()
    ):
        raise HemiQGridError("plan dataset contract drifted")

    _exact_keys(
        plan["execution"],
        {
            "cublas_workspace_config",
            "device",
            "distributed_execution",
            "maximum_concurrent_gpu_workers",
            "minimum_free_gib",
            "physical_gpu_uuid_roster",
        },
        path="plan.execution",
    )
    execution = plan["execution"]
    roster = execution["physical_gpu_uuid_roster"]
    if (
        execution["cublas_workspace_config"] != REQUIRED_CUBLAS_WORKSPACE_CONFIG
        or execution["device"] != "cuda:0"
        or execution["distributed_execution"] != "forbidden"
        or not _is_exact_int(
            execution["maximum_concurrent_gpu_workers"], minimum=1
        )
        or execution["maximum_concurrent_gpu_workers"] != MAX_GPU_WORKERS
        or type(execution["minimum_free_gib"]) is not float
        or not _is_finite_number(
            execution["minimum_free_gib"], minimum=MINIMUM_FREE_GIB
        )
        or execution["minimum_free_gib"] != MINIMUM_FREE_GIB
        or not isinstance(roster, list)
        or roster != sorted(roster)
        or len(roster) != len(set(roster))
        or not roster
        or any(
            not isinstance(value, str) or GPU_UUID_RE.fullmatch(value) is None
            for value in roster
        )
        or roster
        != sorted(
            str(row["uuid"])
            for row in plan["environment_identity"]["hardware"]["gpus"]
        )
    ):
        raise HemiQGridError("plan execution contract drifted")

    if (
        not isinstance(plan["source_identity"], Mapping)
        or set(plan["source_identity"]) != set(SOURCE_FILES)
    ):
        raise HemiQGridError("plan source closure is not exact")
    for relative, value in plan["source_identity"].items():
        _validate_file_identity(value, path=f"plan.source_identity.{relative}")

    cache_keys = {f"{DATASET}:s{subject:03d}" for subject in SUBJECTS}
    split_keys = {f"{key}:f00" for key in cache_keys}
    if (
        not isinstance(plan["cache_identity"], Mapping)
        or set(plan["cache_identity"]) != cache_keys
        or not isinstance(plan["split_identity"], Mapping)
        or set(plan["split_identity"]) != split_keys
    ):
        raise HemiQGridError("plan cache/split inventory is not exact")
    for subject in SUBJECTS:
        key = f"{DATASET}:s{subject:03d}"
        cache = plan["cache_identity"][key]
        _exact_keys(
            cache,
            {
                "archive",
                "arrays",
                "embedded_identity",
                "montage_interpretation",
            },
            path=f"plan.cache_identity.{key}",
        )
        _exact_keys(
            cache["archive"],
            {"relative_path", "sha256", "size_bytes"},
            path=f"plan.cache_identity.{key}.archive",
        )
        expected_relative = (
            f"{PREPROCESSING['schema']}/{MONTAGE_PROFILE}/{DATASET}/"
            f"subject_{subject:03d}.npz"
        )
        if (
            cache["archive"]["relative_path"] != expected_relative
            or not isinstance(cache["archive"]["sha256"], str)
            or SHA256_RE.fullmatch(cache["archive"]["sha256"]) is None
            or not _is_exact_int(cache["archive"]["size_bytes"], minimum=1)
            or set(cache["arrays"])
            != {"channel_names", "positions", "runs", "sessions", "x", "y"}
            or "bipolar" not in str(cache["montage_interpretation"]).lower()
        ):
            raise HemiQGridError(f"plan cache identity {key} is invalid")
        for name, manifest in cache["arrays"].items():
            _validate_array_manifest(
                manifest, path=f"plan.cache_identity.{key}.arrays.{name}"
            )
        embedded = cache["embedded_identity"]
        _exact_keys(
            embedded,
            {
                "array_sha256",
                "channels",
                "coordinates",
                "dataset",
                "montage_profile",
                "preprocessing",
                "shape",
                "subject",
            },
            path=f"plan.cache_identity.{key}.embedded_identity",
        )
        expected_dataset = {
            "confirmation_subjects": list(spec.confirmation_subjects),
            "development_subjects": list(spec.development_subjects),
            "events": list(spec.events),
            "key": spec.key,
            "moabb_class": spec.moabb_class,
            "n_classes": spec.n_classes,
            "protocol": spec.protocol,
            "subjects": list(spec.subjects),
        }
        if (
            cache["arrays"]["x"]["dtype"] != np.dtype(np.float32).str
            or cache["arrays"]["x"]["shape"][1:] != [3, EXPECTED_N_TIMES]
            or cache["arrays"]["y"]["dtype"] != np.dtype(np.int64).str
            or cache["arrays"]["channel_names"]["shape"] != [3]
            or not _same_typed_value(
                embedded["channels"],
                ["C3", "Cz", "C4"],
            )
            or embedded["montage_profile"] != MONTAGE_PROFILE
            or not isinstance(embedded["array_sha256"], str)
            or SHA256_RE.fullmatch(embedded["array_sha256"]) is None
            or not _is_exact_int(embedded["subject"], minimum=1)
            or embedded["subject"] != subject
            or not _same_typed_value(
                embedded["shape"], cache["arrays"]["x"]["shape"]
            )
            or not _same_typed_value(
                embedded["dataset"], expected_dataset
            )
            or not _same_typed_value(
                embedded["preprocessing"],
                preprocessing_for_dataset(DATASET),
            )
            or not _same_typed_value(
                embedded["coordinates"],
                coordinate_contract_for_dataset(
                    DATASET,
                    MONTAGE_PROFILE,
                    ("C3", "Cz", "C4"),
                ),
            )
        ):
            raise HemiQGridError(f"plan cache geometry {key} is invalid")

        split_key = f"{key}:f00"
        split = plan["split_identity"][split_key]
        _exact_keys(
            split,
            {
                "cache_array_sha256",
                "dataset",
                "fold",
                "partitions",
                "subject",
                "trial_count",
            },
            path=f"plan.split_identity.{split_key}",
        )
        if (
            split["dataset"] != DATASET
            or not _is_exact_int(split["subject"], minimum=1)
            or split["subject"] != subject
            or not _is_exact_int(split["fold"], minimum=0)
            or split["fold"] != 0
            or split["cache_array_sha256"]
            != cache["embedded_identity"].get("array_sha256")
            or not _is_exact_int(split["trial_count"], minimum=1)
            or split["trial_count"] != cache["arrays"]["x"]["shape"][0]
            or set(split["partitions"])
            != {"source", "test", "train", "validation"}
        ):
            raise HemiQGridError(f"plan split identity {split_key} is invalid")
        for name, manifest in split["partitions"].items():
            _validate_array_manifest(
                manifest,
                path=f"plan.split_identity.{split_key}.partitions.{name}",
                expected_dtype=np.dtype(np.int64).str,
            )
            if len(manifest["shape"]) != 1 or manifest["shape"][0] <= 0:
                raise HemiQGridError(f"plan split partition {split_key}/{name} is empty")
        counts = {
            name: split["partitions"][name]["shape"][0]
            for name in ("train", "validation", "test")
        }
        if (
            counts["train"] + counts["validation"] + counts["test"]
            != split["trial_count"]
            or split["partitions"]["source"]["shape"][0]
            != counts["train"] + counts["validation"]
        ):
            raise HemiQGridError(f"plan split cardinality {split_key} is invalid")

    _exact_keys(
        plan["configuration_identity"],
        {
            "config_file",
            "config_file_bytes",
            "config_file_sha256",
            "immutable_settings_sha256",
            "parent_frozen_manifest",
            "runtime_overrides",
            "schema",
        },
        path="plan.configuration_identity",
    )
    configuration = plan["configuration_identity"]
    if (
        configuration["schema"] != CONFIG_SCHEMA
        or configuration["config_file"] != CONFIG_RELATIVE_PATH
        or configuration["runtime_overrides"] != ["device", "seed"]
        or not _is_exact_int(configuration["config_file_bytes"], minimum=1)
        or any(
            not isinstance(configuration[name], str)
            or SHA256_RE.fullmatch(configuration[name]) is None
            for name in ("config_file_sha256", "immutable_settings_sha256")
        )
    ):
        raise HemiQGridError("plan HemiQ configuration identity drifted")
    parent = configuration["parent_frozen_manifest"]
    _exact_keys(
        parent,
        {"path", "sha256", "size_bytes"},
        path="plan.configuration_identity.parent_frozen_manifest",
    )
    if (
        parent["path"]
        != "deepnet/results/hemi_q/hemi_q_final_frozen_manifest.json"
        or parent["sha256"]
        != "2bbe28d4415a74cb007328579cf8153585ceaaa28219246aecd0622398987401"
        or not _is_exact_int(parent["size_bytes"], minimum=1)
        or parent["size_bytes"] != 6688
    ):
        raise HemiQGridError("plan historical parent identity drifted")

    if not _same_typed_value(plan["view_contract"], hemiq_view_contract()):
        raise HemiQGridError("plan deterministic view contract drifted")
    _validate_uv_identity(plan["uv_identity"])
    _validate_environment_identity(plan["environment_identity"])
    _exact_keys(
        plan["protocol"],
        {"prediction", "refit", "selection", "teacher_schedule"},
        path="plan.protocol",
    )
    _exact_keys(
        plan["analysis_protocol"],
        {"label_boundary", "primary_aggregation"},
        path="plan.analysis_protocol",
    )
    if (
        not isinstance(plan["evidence_scope"], str)
        or "development" not in plan["evidence_scope"].lower()
        or "confirmation" not in plan["evidence_scope"].lower()
        or any(
            not isinstance(value, str) or not value
            for value in plan["protocol"].values()
        )
        or any(
            not isinstance(value, str) or not value
            for value in plan["analysis_protocol"].values()
        )
    ):
        raise HemiQGridError("plan evidence boundary is missing")


def _validate_uv_identity(value: Any) -> None:
    _exact_keys(
        value,
        {
            "active_venv",
            "direct_dependencies",
            "lock",
            "project",
            "python",
            "python_version_file",
            "uv_version",
        },
        path="plan.uv_identity",
    )
    if (
        not isinstance(value["active_venv"], str)
        or not value["active_venv"].startswith("/")
        or value["direct_dependencies"] != DIRECT_DEPENDENCIES
        or not isinstance(value["uv_version"], str)
        or not value["uv_version"].startswith("uv ")
    ):
        raise HemiQGridError("plan UV identity is invalid")
    for name in ("lock", "project", "python_version_file"):
        entry = value[name]
        _exact_keys(
            entry,
            {"relative_path", "sha256", "size_bytes"},
            path=f"plan.uv_identity.{name}",
        )
        if (
            not isinstance(entry["relative_path"], str)
            or "/" in entry["relative_path"]
            or not isinstance(entry["sha256"], str)
            or SHA256_RE.fullmatch(entry["sha256"]) is None
            or not _is_exact_int(entry["size_bytes"], minimum=1)
        ):
            raise HemiQGridError(f"plan UV identity {name} is invalid")
    _exact_keys(
        value["python"],
        {"executable", "implementation", "version"},
        path="plan.uv_identity.python",
    )
    if any(not isinstance(item, str) or not item for item in value["python"].values()):
        raise HemiQGridError("plan Python identity is invalid")


def _validate_environment_identity(value: Any) -> None:
    _exact_keys(
        value,
        {"cuda", "hardware", "libraries", "parallelism"},
        path="plan.environment_identity",
    )
    _exact_keys(
        value["cuda"],
        {"cublas_workspace_config", "cudnn_version", "torch_cuda_version"},
        path="plan.environment_identity.cuda",
    )
    _exact_keys(
        value["hardware"],
        {
            "architecture",
            "gpus",
            "nvidia_driver_version",
            "system",
        },
        path="plan.environment_identity.hardware",
    )
    _exact_keys(
        value["libraries"],
        {"numpy", "scipy", "torch"},
        path="plan.environment_identity.libraries",
    )
    _exact_keys(
        value["parallelism"],
        {"distributed_execution", "maximum_project_gpu_workers"},
        path="plan.environment_identity.parallelism",
    )
    cuda = value["cuda"]
    hardware = value["hardware"]
    gpus = hardware["gpus"]
    if (
        cuda["cublas_workspace_config"] != REQUIRED_CUBLAS_WORKSPACE_CONFIG
        or not _is_exact_int(cuda["cudnn_version"], minimum=1)
        or not isinstance(cuda["torch_cuda_version"], str)
        or not cuda["torch_cuda_version"]
        or hardware["system"] != "Linux"
        or hardware["architecture"] not in {"x86_64", "AMD64"}
        or not isinstance(hardware["nvidia_driver_version"], str)
        or not isinstance(gpus, list)
        or not gpus
        or not _same_typed_value(
            value["parallelism"],
            {
                "distributed_execution": "forbidden",
                "maximum_project_gpu_workers": MAX_GPU_WORKERS,
            },
        )
        or any(
            not isinstance(value["libraries"][name], str)
            or not value["libraries"][name]
            for name in ("numpy", "scipy", "torch")
        )
    ):
        raise HemiQGridError("plan CUDA/environment contract is invalid")
    observed_uuids: list[str] = []
    for index, row in enumerate(gpus):
        _exact_keys(
            row,
            {"driver_version", "index", "memory_total_mib", "name", "uuid"},
            path=f"plan.environment_identity.hardware.gpus[{index}]",
        )
        if (
            row["driver_version"] != hardware["nvidia_driver_version"]
            or not _is_exact_int(row["index"], minimum=0)
            or not _is_exact_int(row["memory_total_mib"], minimum=1)
            or not isinstance(row["name"], str)
            or not row["name"]
            or not isinstance(row["uuid"], str)
            or GPU_UUID_RE.fullmatch(row["uuid"]) is None
        ):
            raise HemiQGridError("plan physical GPU inventory is invalid")
        observed_uuids.append(row["uuid"])
    if observed_uuids != sorted(set(observed_uuids)):
        raise HemiQGridError("plan physical GPU inventory is not canonical")


def _validate_runtime_identity(
    value: Any,
    *,
    plan: Mapping[str, Any],
    gpu_uuid: str | None,
    path: str,
) -> str:
    _exact_keys(
        value,
        {"cuda_device_count", "device", "gpu", "torch_cuda_version"},
        path=path,
    )
    gpu = value["gpu"]
    _exact_keys(
        gpu,
        {"driver_version", "index", "memory_total_mib", "name", "uuid"},
        path=f"{path}.gpu",
    )
    observed_uuid = gpu["uuid"]
    if (
        not isinstance(observed_uuid, str)
        or GPU_UUID_RE.fullmatch(observed_uuid) is None
        or (gpu_uuid is not None and observed_uuid != gpu_uuid)
        or not isinstance(gpu["driver_version"], str)
        or not gpu["driver_version"]
        or not _is_exact_int(gpu["index"], minimum=0)
        or not _is_exact_int(gpu["memory_total_mib"], minimum=1)
        or not isinstance(gpu["name"], str)
        or not gpu["name"]
        or not _is_exact_int(value["cuda_device_count"], minimum=1)
        or value["cuda_device_count"] != 1
        or value["device"] != "cuda:0"
        or not isinstance(value["device"], str)
        or not isinstance(value["torch_cuda_version"], str)
        or value["torch_cuda_version"]
        != plan["environment_identity"]["cuda"]["torch_cuda_version"]
    ):
        raise HemiQGridError(f"{path} contains invalid runtime values")
    planned = [
        row
        for row in plan["environment_identity"]["hardware"]["gpus"]
        if type(row.get("uuid")) is str and row["uuid"] == observed_uuid
    ]
    if (
        len(planned) != 1
        or not _same_typed_value(gpu, planned[0])
        or gpu["driver_version"]
        != plan["environment_identity"]["hardware"][
            "nvidia_driver_version"
        ]
    ):
        raise HemiQGridError(
            f"{path} GPU/driver row differs from the unique planned row"
        )
    return observed_uuid


def _initialize_run_root(run_root: Path) -> Path:
    root = _safe_mkdir(run_root)
    for name in sorted(RUN_DIRECTORIES):
        _safe_mkdir(root / name)
    lock_path = root / ".authority" / "coordination.lock"
    if not _path_exists(lock_path):
        parent = _open_directory_absolute(lock_path.parent)
        try:
            try:
                _write_exclusive_at(parent, lock_path.name, b"", mode=0o600)
                os.fsync(parent)
            except FileExistsError:
                pass
        finally:
            os.close(parent)
    status = _anchored_lstat(lock_path)
    _assert_unique_regular(status, source=str(lock_path), allow_empty=True)
    return root


def write_or_validate_plan(run_root: Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    validate_plan(plan)
    root = _initialize_run_root(run_root)
    path = root / "plan.json"
    payload = _canonical_bytes(dict(plan))
    with _coordination_lock(root, exclusive=True):
        _recover_regular_stages_guarded(
            run_root=root,
            parent=root,
            associated_prefix=".plan.json.stage-",
            pattern=re.compile(r"^\.plan\.json\.stage-[0-9a-f]{32}$"),
            category="abandoned_plan_stage",
        )
        try:
            _atomic_bytes(path, payload)
        except FileExistsError:
            observed = _read_unique_regular_bytes(path)
            if observed != payload:
                raise HemiQGridError("immutable HemiQ-v2 plan differs")
        if stat.S_IMODE(_anchored_lstat(path).st_mode) != 0o400:
            raise HemiQGridError("immutable HemiQ-v2 plan is writable")
    loaded = _strict_json_file(path)
    validate_plan(loaded)
    if loaded != dict(plan):
        raise HemiQGridError("published HemiQ-v2 plan changed")
    return loaded


def load_plan(run_root: Path) -> dict[str, Any]:
    root = _absolute(run_root)
    if stat.S_IMODE(_anchored_lstat(root / "plan.json").st_mode) != 0o400:
        raise HemiQGridError("immutable HemiQ-v2 plan is writable")
    loaded = _strict_json_file(root / "plan.json")
    validate_plan(loaded)
    return loaded


def _verify_file_identity(
    path: Path, expected: Mapping[str, Any], *, source: str
) -> None:
    payload = _read_unique_regular_bytes(path)
    if (
        len(payload) != expected["size_bytes"]
        or _sha256_bytes(payload) != expected["sha256"]
    ):
        raise HemiQGridError(f"{source} identity differs from the plan")


def verify_source_identity(plan: Mapping[str, Any], *, project_root: Path) -> None:
    validate_plan(plan)
    root = _absolute(project_root)
    for relative, identity in plan["source_identity"].items():
        _verify_file_identity(
            root / relative, identity, source=f"source file {relative}"
        )


def verify_uv_identity(plan: Mapping[str, Any], *, project_root: Path) -> None:
    validate_plan(plan)
    observed = collect_uv_identity(project_root)
    if observed != plan["uv_identity"]:
        raise HemiQGridError("active UV environment differs from the plan")


def verify_environment_identity(plan: Mapping[str, Any]) -> None:
    validate_plan(plan)
    observed = collect_environment_identity()
    if observed != plan["environment_identity"]:
        raise HemiQGridError("CUDA/driver environment differs from the plan")


def _verify_loaded_cache(
    plan: Mapping[str, Any],
    *,
    cache_root: Path,
    subject: int,
    allow_test_class_validation: bool,
) -> tuple[dict[str, Any], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if subject not in SUBJECTS:
        raise HemiQGridError("subject lies outside HemiQ-v2")
    key = f"{DATASET}:s{subject:03d}"
    split_key = f"{key}:f00"
    expected_cache = plan["cache_identity"][key]
    path = _cache_path(cache_root, subject)
    payload = _read_unique_regular_bytes(path)
    if (
        len(payload) != expected_cache["archive"]["size_bytes"]
        or _sha256_bytes(payload) != expected_cache["archive"]["sha256"]
    ):
        raise HemiQGridError(f"cache {key} identity differs from the plan")
    _load_npz_exact(
        payload,
        expected_keys=(
            "x",
            "y",
            "positions",
            "channel_names",
            "sessions",
            "runs",
            "identity",
        ),
        source=str(path),
    )
    cache = data_module.load_subject_cache(
        DATASET,
        subject,
        cache_root=cache_root,
        montage_profile=MONTAGE_PROFILE,
    )
    observed_arrays = {
        name: _array_manifest(np.asarray(cache[name]))
        for name in ("channel_names", "positions", "runs", "sessions", "x", "y")
    }
    if (
        observed_arrays != expected_cache["arrays"]
        or cache["identity"] != expected_cache["embedded_identity"]
        or tuple(str(value) for value in cache["channel_names"].tolist())
        != ("C3", "Cz", "C4")
    ):
        raise HemiQGridError(f"loaded cache {key} differs from the plan")
    rows = _bnci004_rows(
        cache, validate_held_out_classes=allow_test_class_validation
    )
    train, validation, test = rows
    source = np.sort(np.concatenate((train, validation)))
    observed_split = {
        "cache_array_sha256": str(cache["identity"]["array_sha256"]),
        "dataset": DATASET,
        "fold": 0,
        "partitions": {
            "source": _rows_manifest(source),
            "test": _rows_manifest(test),
            "train": _rows_manifest(train),
            "validation": _rows_manifest(validation),
        },
        "subject": subject,
        "trial_count": len(cache["x"]),
    }
    if observed_split != plan["split_identity"][split_key]:
        raise HemiQGridError(f"split {split_key} differs from the plan")
    return cache, rows


def verify_cache_and_splits(
    plan: Mapping[str, Any],
    *,
    cache_root: Path,
) -> None:
    validate_plan(plan)
    for subject in SUBJECTS:
        _verify_loaded_cache(
            plan,
            cache_root=cache_root,
            subject=subject,
            allow_test_class_validation=False,
        )


def verify_static_identity(
    plan: Mapping[str, Any],
    *,
    project_root: Path,
    cache_root: Path,
) -> None:
    verify_source_identity(plan, project_root=project_root)
    verify_uv_identity(plan, project_root=project_root)
    verify_environment_identity(plan)
    verify_cache_and_splits(plan, cache_root=cache_root)
    observed_config = _config_identity(project_root)
    if observed_config != plan["configuration_identity"]:
        raise HemiQGridError("outcome-free HemiQ configuration differs from plan")
    if hemiq_view_contract() != plan["view_contract"]:
        raise HemiQGridError("HemiQ deterministic view contract differs from plan")


def _verify_worker_publication_identity(
    *,
    plan: Mapping[str, Any],
    project_root: Path,
) -> None:
    """Rebind the executable worker closure immediately around publication."""

    verify_source_identity(plan, project_root=project_root)
    verify_uv_identity(plan, project_root=project_root)
    observed_config = _config_identity(project_root)
    if not _same_typed_value(
        observed_config,
        plan["configuration_identity"],
    ):
        raise HemiQGridError(
            "worker HemiQ configuration differs from the plan"
        )


def _hard_disk_guard(path: Path, minimum_free_gib: float = MINIMUM_FREE_GIB) -> dict[str, Any]:
    if not _is_finite_number(minimum_free_gib, minimum=MINIMUM_FREE_GIB):
        raise HemiQGridError("minimum free-space floor may not be weakened")
    usage = shutil.disk_usage(_absolute(path))
    free_gib = usage.free / (1024.0**3)
    if not math.isfinite(free_gib) or free_gib < float(minimum_free_gib):
        raise HemiQGridError(
            f"free space {free_gib:.3f} GiB is below the 50 GiB hard floor"
        )
    return {
        "free_bytes": int(usage.free),
        "minimum_free_bytes": int(math.ceil(float(minimum_free_gib) * 1024**3)),
        "path": str(_absolute(path)),
        "status": "pass",
    }


def _require_single_gpu_runtime(
    *,
    plan: Mapping[str, Any],
    gpu_uuid: str,
    device: str,
) -> dict[str, Any]:
    import torch

    if plan["publication_mode"] == FORMAL_PUBLICATION_MODE:
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != REQUIRED_CUBLAS_WORKSPACE_CONFIG:
            raise HemiQGridError("CUBLAS_WORKSPACE_CONFIG is not the frozen value")
        if device != "cuda:0":
            raise HemiQGridError("formal HemiQ worker requires device cuda:0")
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if visible != gpu_uuid:
            raise HemiQGridError(
                "formal worker requires CUDA_VISIBLE_DEVICES to equal its physical UUID"
            )
    for name in ("RANK", "LOCAL_RANK", "GROUP_RANK", "NODE_RANK"):
        if name in os.environ:
            raise HemiQGridError(f"distributed environment variable {name} is forbidden")
    world_size = os.environ.get("WORLD_SIZE")
    if world_size not in {None, "", "1"}:
        raise HemiQGridError("DDP/world-size execution is forbidden")
    if gpu_uuid not in plan["execution"]["physical_gpu_uuid_roster"]:
        raise HemiQGridError("worker GPU UUID is absent from the plan")
    inventory = _nvidia_inventory()
    matched = [row for row in inventory if row["uuid"] == gpu_uuid]
    if len(matched) != 1:
        raise HemiQGridError("worker physical GPU UUID is not unique")
    planned = [
        row
        for row in plan["environment_identity"]["hardware"]["gpus"]
        if row["uuid"] == gpu_uuid
    ]
    if matched != planned:
        raise HemiQGridError("worker GPU/driver identity differs from the plan")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise HemiQGridError("formal worker must expose exactly one CUDA device")
    return {
        "cuda_device_count": int(torch.cuda.device_count()),
        "device": device,
        "gpu": copy.deepcopy(matched[0]),
        "torch_cuda_version": torch.version.cuda,
    }


@contextmanager
def _coordination_lock(
    run_root: Path,
    *,
    exclusive: bool,
    nonblocking: bool = False,
) -> Iterator[int]:
    root = _absolute(run_root)
    path = root / ".authority" / "coordination.lock"
    parent = _open_directory_absolute(path.parent)
    flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    try:
        path_status = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
        _assert_unique_regular(path_status, source=str(path), allow_empty=True)
        descriptor = os.open(path.name, flags, dir_fd=parent)
        observed = os.fstat(descriptor)
        if (
            (observed.st_dev, observed.st_ino)
            != (path_status.st_dev, path_status.st_ino)
        ):
            raise HemiQGridError("coordination lock changed while opening")
        operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        if nonblocking:
            operation |= fcntl.LOCK_NB
        try:
            fcntl.flock(descriptor, operation)
        except BlockingIOError as error:
            raise AnalysisActive("analysis publication fence is active") from error
        yield descriptor
        final = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
        if (
            (final.st_dev, final.st_ino)
            != (observed.st_dev, observed.st_ino)
        ):
            raise HemiQGridError("coordination lock authority was replaced")
    finally:
        if descriptor >= 0:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
        os.close(parent)


def _assert_lease_bound(
    lease: project_gpu_leases.GPULease,
    *,
    project_root: Path,
    run_root: Path,
    plan: Mapping[str, Any],
    gpu_uuid: str,
) -> None:
    project_gpu_leases.assert_gpu_lease(lease)
    receipt = project_gpu_leases.gpu_lease_receipt(lease)
    project_gpu_leases.validate_gpu_lease_receipt(
        receipt,
        project_root=_absolute(project_root),
        run_root=_absolute(run_root),
        plan_sha256=str(plan["plan_sha256"]),
        gpu_uuid=gpu_uuid,
        track_scope=TRACK_SCOPE,
    )


def _synthetic_cuda_preflight(
    *,
    project_root: Path,
    device: str,
) -> list[dict[str, str]]:
    """Exercise outcome-free transform/reset/equivariance invariants on CUDA."""

    import torch

    rng = np.random.default_rng(20_260_729)
    raw = rng.normal(scale=1e-6, size=(8, 3, EXPECTED_N_TIMES)).astype(
        np.float32
    )
    labels = np.arange(8, dtype=np.int64) % 2
    views = derive_hemiq_views(raw, channel_names=("C3", "Cz", "C4"))
    config, _ = load_frozen_config(project_root, seed=SEEDS[0], device=device)
    classifier = HemiQHarmonizedV2Classifier(config).fit_fixed_epochs(
        views["raw"], views["covariance"], labels, epochs=0
    )
    initial = classifier.initial_model_state_sha256_
    if initial != classifier.model_state_sha256_:
        raise HemiQGridError("zero-epoch reset changed the synthetic model")
    prepared = torch.from_numpy(classifier._prepare_raw(views["raw"])).to(
        classifier.device_
    )
    reflected = prepared[:, (2, 1, 0), :]
    classifier.model_.eval()
    with torch.no_grad():
        direct = classifier.model_(prepared)
        mirrored = classifier.model_(reflected)
    if not torch.equal(direct, -mirrored):
        raise HemiQGridError("synthetic CUDA reflection is not exactly odd")
    replay = HemiQHarmonizedV2Classifier(config).fit_fixed_epochs(
        views["raw"], views["covariance"], labels, epochs=0
    )
    if replay.initial_model_state_sha256_ != initial:
        raise HemiQGridError("synthetic CUDA seeded initialization did not replay")
    first_step = HemiQHarmonizedV2Classifier(config).fit_fixed_epochs(
        views["raw"], views["covariance"], labels, epochs=1
    )
    first_probability = first_step.predict_proba(views["raw"])
    second_step = HemiQHarmonizedV2Classifier(config).fit_fixed_epochs(
        views["raw"], views["covariance"], labels, epochs=1
    )
    second_probability = second_step.predict_proba(views["raw"])
    if (
        first_step.initial_model_state_sha256_
        != second_step.initial_model_state_sha256_
        or first_step.model_state_sha256_
        != second_step.model_state_sha256_
        or first_step.history_ != second_step.history_
        or not np.array_equal(first_probability, second_probability)
    ):
        raise HemiQGridError(
            "synthetic CUDA forward/backward/optimizer replay differed"
        )
    covariance_eigenvalues = np.linalg.eigvalsh(
        views["covariance"].astype(np.float64)
    )
    if (
        views["raw"].dtype != np.float32
        or views["covariance"].dtype != np.float32
        or np.any(covariance_eigenvalues <= 0.0)
    ):
        raise HemiQGridError("synthetic deterministic views failed their contract")
    return [
        {"check": "config_identity", "status": "pass"},
        {"check": "cuda_available", "status": "pass"},
        {"check": "deterministic_seeded_reset", "status": "pass"},
        {
            "check": "deterministic_forward_backward_optimizer_replay",
            "status": "pass",
        },
        {"check": "deterministic_probability_replay", "status": "pass"},
        {"check": "exact_reflection_anti_equivariance", "status": "pass"},
        {"check": "float32_spd_covariances", "status": "pass"},
        {"check": "supplied_bipolar_order", "status": "pass"},
        {"check": "zero_epoch_preserved", "status": "pass"},
    ]


def _preflight_path(run_root: Path, gpu_uuid: str) -> Path:
    if GPU_UUID_RE.fullmatch(gpu_uuid) is None:
        raise HemiQGridError("invalid preflight GPU UUID")
    return _absolute(run_root) / "preflight" / f"{gpu_uuid}.json"


def _validate_preflight_value(
    value: Any,
    *,
    plan: Mapping[str, Any],
    project_root: Path,
    run_root: Path,
    gpu_uuid: str,
) -> None:
    _exact_keys(
        value,
        {
            "checks",
            "created_at",
            "disk_guard",
            "gpu_lease_receipt",
            "gpu_uuid",
            "plan_sha256",
            "runtime",
            "schema",
            "status",
        },
        path="preflight",
    )
    checks = value["checks"]
    if (
        value["schema"] != PREFLIGHT_SCHEMA
        or value["status"] != "pass"
        or value["plan_sha256"] != plan["plan_sha256"]
        or value["gpu_uuid"] != gpu_uuid
        or not isinstance(value["created_at"], str)
        or not isinstance(checks, list)
        or len(checks) != 9
        or [item.get("check") for item in checks]
        != [
            "config_identity",
            "cuda_available",
            "deterministic_seeded_reset",
            "deterministic_forward_backward_optimizer_replay",
            "deterministic_probability_replay",
            "exact_reflection_anti_equivariance",
            "float32_spd_covariances",
            "supplied_bipolar_order",
            "zero_epoch_preserved",
        ]
        or any(
            not isinstance(item, Mapping)
            or set(item) != {"check", "status"}
            or item["status"] != "pass"
            for item in checks
        )
    ):
        raise HemiQGridError("CUDA preflight receipt is invalid")
    _exact_keys(
        value["disk_guard"],
        {"free_bytes", "minimum_free_bytes", "path", "status"},
        path="preflight.disk_guard",
    )
    if (
        value["disk_guard"]["status"] != "pass"
        or not _is_exact_int(value["disk_guard"]["free_bytes"], minimum=1)
        or not _is_exact_int(
            value["disk_guard"]["minimum_free_bytes"],
            minimum=int(MINIMUM_FREE_GIB * 1024**3),
        )
        or value["disk_guard"]["path"] != str(_absolute(run_root))
    ):
        raise HemiQGridError("preflight disk guard is invalid")
    _validate_runtime_identity(
        value["runtime"],
        plan=plan,
        gpu_uuid=gpu_uuid,
        path="preflight.runtime",
    )
    project_gpu_leases.validate_gpu_lease_receipt(
        value["gpu_lease_receipt"],
        project_root=_absolute(project_root),
        run_root=_absolute(run_root),
        plan_sha256=str(plan["plan_sha256"]),
        gpu_uuid=gpu_uuid,
        track_scope=TRACK_SCOPE,
    )


def validate_preflight(
    *,
    plan: Mapping[str, Any],
    project_root: Path,
    run_root: Path,
    gpu_uuid: str,
) -> dict[str, Any]:
    path = _preflight_path(run_root, gpu_uuid)
    if stat.S_IMODE(_anchored_lstat(path).st_mode) != 0o400:
        raise HemiQGridError("CUDA preflight receipt is writable")
    value = _strict_json_file(path)
    _validate_preflight_value(
        value,
        plan=plan,
        project_root=project_root,
        run_root=run_root,
        gpu_uuid=gpu_uuid,
    )
    return value


def run_preflight(
    *,
    plan: Mapping[str, Any],
    project_root: Path,
    cache_root: Path,
    run_root: Path,
    gpu_uuid: str,
    device: str,
    lease: project_gpu_leases.GPULease,
) -> dict[str, Any]:
    """Publish one immutable synthetic-only CUDA preflight under the GPU guard."""

    validate_plan(plan)
    _assert_lease_bound(
        lease,
        project_root=project_root,
        run_root=run_root,
        plan=plan,
        gpu_uuid=gpu_uuid,
    )
    path = _preflight_path(run_root, gpu_uuid)
    already_published = _path_exists(path)
    runtime: dict[str, Any] | None = None
    checks: list[dict[str, str]] | None = None
    disk_guard: dict[str, Any] | None = None
    if not already_published:
        verify_static_identity(
            plan, project_root=project_root, cache_root=cache_root
        )
        runtime = _require_single_gpu_runtime(
            plan=plan, gpu_uuid=gpu_uuid, device=device
        )
        checks = _synthetic_cuda_preflight(
            project_root=project_root, device=device
        )
        disk_guard = _hard_disk_guard(run_root)
    published_identity: tuple[int, int] | None = None
    published_payload: bytes | None = None
    try:
        with project_gpu_leases.guard_gpu_lease(lease) as lease_receipt:
            with _coordination_lock(run_root, exclusive=False, nonblocking=True):
                _recover_regular_stages_guarded(
                    run_root=run_root,
                    parent=_absolute(run_root) / "preflight",
                    associated_prefix=f".{gpu_uuid}.json.stage-",
                    pattern=re.compile(
                        rf"^\.{re.escape(gpu_uuid)}\.json\.stage-[0-9a-f]{{32}}$"
                    ),
                    category="abandoned_preflight_stage",
                )
                if not _path_exists(path):
                    if runtime is None or checks is None or disk_guard is None:
                        raise HemiQGridError(
                            "preflight disappeared after identity selection"
                        )
                    value = {
                        "checks": checks,
                        "created_at": _utc_now(),
                        "disk_guard": disk_guard,
                        "gpu_lease_receipt": lease_receipt,
                        "gpu_uuid": gpu_uuid,
                        "plan_sha256": plan["plan_sha256"],
                        "runtime": runtime,
                        "schema": PREFLIGHT_SCHEMA,
                        "status": "pass",
                    }
                    _validate_preflight_value(
                        value,
                        plan=plan,
                        project_root=project_root,
                        run_root=run_root,
                        gpu_uuid=gpu_uuid,
                    )
                    try:
                        (
                            published_payload,
                            published_identity,
                        ) = _atomic_json_with_identity(path, value)
                    except FileExistsError:
                        pass
        result = validate_preflight(
            plan=plan,
            project_root=project_root,
            run_root=run_root,
            gpu_uuid=gpu_uuid,
        )
        if stat.S_IMODE(_anchored_lstat(path).st_mode) != 0o400:
            raise HemiQGridError("CUDA preflight receipt is writable")
        return result
    except Exception:
        if published_identity is not None and _path_exists(path):
            _quarantine_leaf(
                run_root,
                path,
                category="preflight_authority_loss",
                expected_identity=published_identity,
                expected_payload=published_payload,
            )
        raise


def _record_directory(run_root: Path, job: Job) -> Path:
    return _absolute(run_root) / "records" / job.job_id


def _claim_path(run_root: Path, job: Job) -> Path:
    return _absolute(run_root) / "claims" / f"{job.job_id}.json"


def _failure_path(run_root: Path, job: Job, nonce: str) -> Path:
    return _absolute(run_root) / "failures" / f"{job.job_id}-{nonce}.json"


def _quarantine_path(run_root: Path, category: str, basename: str) -> Path:
    if not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", category):
        raise HemiQGridError("invalid quarantine category")
    root = _safe_mkdir(_absolute(run_root) / "quarantine" / category)
    return root / f"{basename}-{uuid.uuid4().hex}"


def _quarantine_leaf(
    run_root: Path,
    source: Path,
    *,
    category: str,
    expected_identity: tuple[int, int] | None = None,
    expected_payload: bytes | None = None,
) -> Path:
    """Hide an exact canonical inode before any forensic recovery work.

    The first mutation is always an inode-preserving rename within the
    canonical leaf's current parent.  Permission recovery and the later move
    into the quarantine tree operate only on that hidden name.  Consequently,
    failures while chmodding, crossing parents, inspecting, or fsyncing cannot
    leave the failed artifact visible at its authoritative canonical name.
    """

    if not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", category):
        raise HemiQGridError("invalid quarantine category")
    source = _absolute(source)
    destination: Path | None = None
    source_parent = _open_directory_absolute(source.parent)
    hidden_name = f".{source.name}.quarantine-{uuid.uuid4().hex}"
    hidden_path = source.parent / hidden_name
    source_descriptor = -1
    destination_parent = -1
    identity: tuple[int, int] | None = None
    original_mode: int | None = None
    try:
        source_status = os.stat(
            source.name, dir_fd=source_parent, follow_symlinks=False
        )
        identity = (int(source_status.st_dev), int(source_status.st_ino))
        if expected_identity is not None and identity != expected_identity:
            raise HemiQGridError("quarantine source is not the expected inode")
        if not (
            stat.S_ISREG(source_status.st_mode)
            or stat.S_ISDIR(source_status.st_mode)
        ):
            raise HemiQGridError(
                "quarantine source is not a regular file or directory"
            )
        original_mode = stat.S_IMODE(source_status.st_mode)
        flags = (
            _directory_flags()
            if stat.S_ISDIR(source_status.st_mode)
            else (
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0)
            )
        )
        source_descriptor = os.open(
            source.name,
            flags,
            dir_fd=source_parent,
        )
        opened = os.fstat(source_descriptor)
        if (int(opened.st_dev), int(opened.st_ino)) != identity:
            raise HemiQGridError(
                "quarantine source changed while binding its inode"
            )

        # Authority-removing operation: nothing that can require chmod or a
        # cross-parent filesystem policy check occurs before this rename.
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
            raise HemiQGridError(
                "hidden quarantine artifact changed during authority removal"
            )
        os.fsync(source_parent)
        if expected_payload is not None:
            if not stat.S_ISREG(opened.st_mode):
                raise HemiQGridError(
                    "payload-bound quarantine source is not a regular file"
                )
            hidden_payload = _read_descriptor_bytes(
                source_descriptor,
                source=str(hidden_path),
            )
            hidden_after = os.stat(
                hidden_name,
                dir_fd=source_parent,
                follow_symlinks=False,
            )
            if (
                hidden_payload != expected_payload
                or (int(hidden_after.st_dev), int(hidden_after.st_ino))
                != identity
            ):
                raise HemiQGridError(
                    "hidden quarantine artifact differs from its published bytes"
                )

        if stat.S_ISDIR(opened.st_mode) and not (
            opened.st_mode & stat.S_IWUSR
        ):
            # The lab filesystem rejects cross-parent renames of read-only
            # directories. This chmod is safe because the canonical name has
            # already disappeared and the descriptor binds the hidden inode.
            os.fchmod(source_descriptor, 0o700)
            os.fsync(source_descriptor)
            hidden = os.stat(
                hidden_name,
                dir_fd=source_parent,
                follow_symlinks=False,
            )
            if (int(hidden.st_dev), int(hidden.st_ino)) != identity:
                raise HemiQGridError(
                    "hidden quarantine artifact changed during permission recovery"
                )

        # Preparing or opening the cross-parent forensic destination occurs
        # only after the canonical name is gone.
        destination = _quarantine_path(
            run_root,
            category,
            source.name,
        )
        destination_parent = _open_directory_absolute(destination.parent)
        _rename_noreplace_at(
            source_parent,
            hidden_name,
            destination_parent,
            destination.name,
        )
        quarantined = os.stat(
            destination.name,
            dir_fd=destination_parent,
            follow_symlinks=False,
        )
        if (int(quarantined.st_dev), int(quarantined.st_ino)) != identity:
            raise HemiQGridError("quarantined artifact changed during rename")
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
        raise HemiQGridError(
            f"quarantine destination was not established for {hidden_path}"
        )
    return destination


def _read_descriptor_bytes(descriptor: int, *, source: str) -> bytes:
    before = os.fstat(descriptor)
    _assert_unique_regular(before, source=source)
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    remaining = int(before.st_size)
    while remaining:
        chunk = os.read(descriptor, min(1024 * 1024, remaining))
        if not chunk:
            raise HemiQGridError(f"{source} was truncated")
        chunks.append(chunk)
        remaining -= len(chunk)
    after = os.fstat(descriptor)
    stable = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(getattr(before, key) != getattr(after, key) for key in stable):
        raise HemiQGridError(f"{source} changed while reading")
    return b"".join(chunks)


def _read_unique_regular_at(
    parent_descriptor: int,
    name: str,
    *,
    source: str,
    allow_empty: bool = False,
) -> tuple[bytes, os.stat_result]:
    """Read one descriptor-anchored leaf and retain its exact byte snapshot."""

    if "/" in name or name in {"", ".", ".."}:
        raise HemiQGridError(f"{source} has an invalid leaf name")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptor = -1
    stable = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    try:
        before_path = os.stat(
            name, dir_fd=parent_descriptor, follow_symlinks=False
        )
        _assert_unique_regular(
            before_path, source=source, allow_empty=allow_empty
        )
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
        opened = os.fstat(descriptor)
        _assert_unique_regular(opened, source=source, allow_empty=allow_empty)
        if any(
            getattr(before_path, field) != getattr(opened, field)
            for field in stable
        ):
            raise HemiQGridError(f"{source} changed while opening")
        payload = _read_descriptor_bytes(descriptor, source=source)
        final_descriptor = os.fstat(descriptor)
        final_path = os.stat(
            name, dir_fd=parent_descriptor, follow_symlinks=False
        )
        for observed in (final_descriptor, final_path):
            if any(
                getattr(opened, field) != getattr(observed, field)
                for field in stable
            ):
                raise HemiQGridError(f"{source} changed during snapshot")
        return payload, opened
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _open_locked_leaf(path: Path, *, nonblocking: bool) -> tuple[int, os.stat_result]:
    absolute = _absolute(path)
    parent = _open_directory_absolute(absolute.parent)
    descriptor = -1
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        path_status = os.stat(
            absolute.name, dir_fd=parent, follow_symlinks=False
        )
        _assert_unique_regular(path_status, source=str(absolute))
        descriptor = os.open(absolute.name, flags, dir_fd=parent)
        observed = os.fstat(descriptor)
        if (
            (observed.st_dev, observed.st_ino)
            != (path_status.st_dev, path_status.st_ino)
        ):
            raise HemiQGridError(f"{absolute} changed while opening")
        operation = fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0)
        fcntl.flock(descriptor, operation)
        return descriptor, observed
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    finally:
        os.close(parent)


def _recover_regular_stages_guarded(
    *,
    run_root: Path,
    parent: Path,
    associated_prefix: str,
    pattern: re.Pattern[str],
    category: str,
) -> list[Path]:
    """Quarantine exact unlocked atomic stages; malformed/live stages fail closed."""

    recovered: list[Path] = []
    directory = _open_directory_absolute(parent)
    try:
        names = sorted(os.listdir(directory))
    finally:
        os.close(directory)
    for name in names:
        if not name.startswith(associated_prefix):
            continue
        if pattern.fullmatch(name) is None:
            raise HemiQGridError(
                f"unrecognized atomic stage for {associated_prefix}: {name}"
            )
        path = _absolute(parent) / name
        try:
            descriptor, observed = _open_locked_leaf(
                path, nonblocking=True
            )
        except BlockingIOError as error:
            raise ClaimUnavailable(
                f"atomic stage is still live: {name}"
            ) from error
        try:
            payload = _read_descriptor_bytes(descriptor, source=str(path))
            recovered.append(
                _quarantine_leaf(
                    run_root,
                    path,
                    category=category,
                    expected_identity=(
                        int(observed.st_dev),
                        int(observed.st_ino),
                    ),
                    expected_payload=payload,
                )
            )
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
    return recovered


def _validate_claim_value(
    value: Any,
    *,
    job: Job,
    plan: Mapping[str, Any],
    project_root: Path,
    run_root: Path,
    gpu_uuid: str,
    preflight_payload: bytes | None = None,
) -> None:
    _exact_keys(
        value,
        {
            "created_at",
            "disk_guard",
            "gpu_lease_receipt",
            "gpu_uuid",
            "job",
            "nonce",
            "owner",
            "plan_sha256",
            "preflight",
            "schema",
        },
        path="claim",
    )
    if (
        value["schema"] != CLAIM_SCHEMA
        or value["plan_sha256"] != plan["plan_sha256"]
        or _job_from_mapping(value["job"]) != job
        or value["gpu_uuid"] != gpu_uuid
        or not isinstance(value["nonce"], str)
        or re.fullmatch(r"[0-9a-f]{32}", value["nonce"]) is None
        or not isinstance(value["created_at"], str)
        or not isinstance(value["owner"], Mapping)
    ):
        raise HemiQGridError("claim identity is invalid")
    _exact_keys(
        value["disk_guard"],
        {"free_bytes", "minimum_free_bytes", "path", "status"},
        path="claim.disk_guard",
    )
    if (
        value["disk_guard"]["status"] != "pass"
        or not _is_exact_int(value["disk_guard"]["free_bytes"], minimum=1)
        or not _is_exact_int(
            value["disk_guard"]["minimum_free_bytes"],
            minimum=int(MINIMUM_FREE_GIB * 1024**3),
        )
        or value["disk_guard"]["path"] != str(_absolute(run_root))
    ):
        raise HemiQGridError("claim disk guard is invalid")
    _exact_keys(
        value["preflight"],
        {"relative_path", "sha256", "size_bytes"},
        path="claim.preflight",
    )
    expected_preflight = _preflight_path(run_root, gpu_uuid)
    if (
        value["preflight"]["relative_path"]
        != str(expected_preflight.relative_to(_absolute(run_root)))
        or not isinstance(value["preflight"]["sha256"], str)
        or SHA256_RE.fullmatch(value["preflight"]["sha256"]) is None
        or not _is_exact_int(value["preflight"]["size_bytes"], minimum=1)
    ):
        raise HemiQGridError("claim preflight binding is invalid")
    if preflight_payload is None:
        _verify_file_identity(
            expected_preflight,
            value["preflight"],
            source="claim preflight",
        )
    elif (
        len(preflight_payload) != value["preflight"]["size_bytes"]
        or _sha256_bytes(preflight_payload)
        != value["preflight"]["sha256"]
    ):
        raise HemiQGridError(
            "descriptor-bound claim preflight differs from its receipt"
        )
    project_gpu_leases.validate_gpu_lease_receipt(
        value["gpu_lease_receipt"],
        project_root=_absolute(project_root),
        run_root=_absolute(run_root),
        plan_sha256=str(plan["plan_sha256"]),
        gpu_uuid=gpu_uuid,
        track_scope=TRACK_SCOPE,
    )


def _assert_claim_path(claim: Claim) -> None:
    observed = os.fstat(claim.descriptor)
    if (
        (observed.st_dev, observed.st_ino) != (claim.st_dev, claim.st_ino)
        or observed.st_nlink != 1
        or not stat.S_ISREG(observed.st_mode)
    ):
        raise HemiQGridError("claim descriptor authority changed")
    path_status = _anchored_lstat(claim.path)
    if (
        (path_status.st_dev, path_status.st_ino) != (claim.st_dev, claim.st_ino)
        or path_status.st_nlink != 1
        or not stat.S_ISREG(path_status.st_mode)
    ):
        raise HemiQGridError("claim path authority changed")
    payload = _read_descriptor_bytes(
        claim.descriptor, source=str(claim.path)
    )
    if _strict_json_bytes(payload, source=str(claim.path)) != dict(claim.value):
        raise HemiQGridError("claim bytes changed")


def _recover_existing_claim(
    *,
    plan: Mapping[str, Any],
    project_root: Path,
    run_root: Path,
    job: Job,
    completed_claim_receipt: Mapping[str, Any] | None = None,
) -> None:
    path = _claim_path(run_root, job)
    if not _path_exists(path):
        return
    try:
        descriptor, observed = _open_locked_leaf(path, nonblocking=True)
    except BlockingIOError as error:
        raise ClaimUnavailable(f"job {job.job_id} has a live claim") from error
    try:
        current = _anchored_lstat(path)
        if (current.st_dev, current.st_ino) != (
            observed.st_dev,
            observed.st_ino,
        ):
            raise HemiQGridError("stale claim path changed during recovery")
        payload = _read_descriptor_bytes(descriptor, source=str(path))
        value = _strict_json_bytes(payload, source=str(path))
        gpu_uuid = value.get("gpu_uuid")
        if not isinstance(gpu_uuid, str):
            raise HemiQGridError("stale claim lacks a physical GPU UUID")
        _validate_claim_value(
            value,
            job=job,
            plan=plan,
            project_root=project_root,
            run_root=run_root,
            gpu_uuid=gpu_uuid,
        )
        observed_receipt = {
            "claim": value,
            "claim_sha256": _sha256_bytes(payload),
            "claim_size_bytes": len(payload),
            "claim_st_dev": int(observed.st_dev),
            "claim_st_ino": int(observed.st_ino),
            "gpu_lease_receipt": copy.deepcopy(
                value["gpu_lease_receipt"]
            ),
            "nonce": value["nonce"],
        }
        if completed_claim_receipt is not None:
            if not _same_typed_value(
                observed_receipt, completed_claim_receipt
            ):
                raise HemiQGridError(
                    "completed result does not bind the unlocked claim"
                )
            category = "completed_claim"
        else:
            if _path_exists(_record_directory(run_root, job)):
                raise HemiQGridError(
                    "completed claim recovery requires a validated result"
                )
            category = "stale_claim"
        _quarantine_leaf(
            run_root,
            path,
            category=category,
            expected_identity=(
                int(observed.st_dev),
                int(observed.st_ino),
            ),
            expected_payload=payload,
        )
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _recover_claim_before_partials(
    *,
    plan: Mapping[str, Any],
    project_root: Path,
    run_root: Path,
    job: Job,
) -> tuple[bool, list[Path]]:
    """Resolve the claim authority before touching any job-owned stage.

    A committed result binds the exact claim inode and bytes through its
    completion receipt. An uncommitted claim is validated in full. In either
    case, a locked/live claim raises :class:`ClaimUnavailable` before a stage
    directory can be moved. Only an unlocked, validated claim is quarantined,
    after which abandoned stages are eligible for recovery.
    """

    destination = _record_directory(run_root, job)
    if _path_exists(destination):
        validated_completion = validate_completion(
            plan=plan,
            project_root=project_root,
            run_root=run_root,
            job=job,
        )
        _recover_existing_claim(
            plan=plan,
            project_root=project_root,
            run_root=run_root,
            job=job,
            completed_claim_receipt=validated_completion["completion"][
                "claim_receipt"
            ],
        )
        completed = True
    else:
        _recover_existing_claim(
            plan=plan,
            project_root=project_root,
            run_root=run_root,
            job=job,
        )
        completed = False
    recovered = _recover_job_partials_guarded(run_root=run_root, job=job)
    return completed, recovered


def acquire_claim(
    *,
    plan: Mapping[str, Any],
    project_root: Path,
    run_root: Path,
    job: Job,
    gpu_uuid: str,
    lease: project_gpu_leases.GPULease,
) -> Claim | None:
    validate_plan(plan)
    if job not in expected_jobs():
        raise HemiQGridError("cannot claim a job outside the exact plan")
    _assert_lease_bound(
        lease,
        project_root=project_root,
        run_root=run_root,
        plan=plan,
        gpu_uuid=gpu_uuid,
    )
    validate_preflight(
        plan=plan,
        project_root=project_root,
        run_root=run_root,
        gpu_uuid=gpu_uuid,
    )
    created_claim: Claim | None = None
    candidate_descriptor = -1
    candidate_identity: tuple[int, int] | None = None
    candidate_payload: bytes | None = None
    candidate_stage_path: Path | None = None
    completed = False
    try:
        with project_gpu_leases.guard_gpu_lease(lease) as lease_receipt:
            with _coordination_lock(run_root, exclusive=False, nonblocking=True):
                completed, _ = _recover_claim_before_partials(
                    plan=plan,
                    project_root=project_root,
                    run_root=run_root,
                    job=job,
                )
                if not completed:
                    disk_guard = _hard_disk_guard(run_root)
                    preflight_path = _preflight_path(run_root, gpu_uuid)
                    preflight_payload = _read_unique_regular_bytes(preflight_path)
                    nonce = uuid.uuid4().hex
                    value = {
                        "created_at": _utc_now(),
                        "disk_guard": disk_guard,
                        "gpu_lease_receipt": lease_receipt,
                        "gpu_uuid": gpu_uuid,
                        "job": job.as_dict(),
                        "nonce": nonce,
                        "owner": project_gpu_leases.owner_identity(),
                        "plan_sha256": plan["plan_sha256"],
                        "preflight": {
                            "relative_path": str(
                                preflight_path.relative_to(_absolute(run_root))
                            ),
                            "sha256": _sha256_bytes(preflight_payload),
                            "size_bytes": len(preflight_payload),
                        },
                        "schema": CLAIM_SCHEMA,
                    }
                    _validate_claim_value(
                        value,
                        job=job,
                        plan=plan,
                        project_root=project_root,
                        run_root=run_root,
                        gpu_uuid=gpu_uuid,
                    )
                    claims = _absolute(run_root) / "claims"
                    parent = _open_directory_absolute(claims)
                    staged_name = f".{job.job_id}.{nonce}.stage"
                    candidate_stage_path = claims / staged_name
                    candidate_payload = _canonical_bytes(value)
                    try:
                        written = _write_exclusive_at(
                            parent, staged_name, candidate_payload
                        )
                        candidate_identity = (
                            int(written.st_dev),
                            int(written.st_ino),
                        )
                        candidate_descriptor = os.open(
                            staged_name,
                            os.O_RDONLY
                            | getattr(os, "O_CLOEXEC", 0)
                            | getattr(os, "O_NOFOLLOW", 0),
                            dir_fd=parent,
                        )
                        fcntl.flock(candidate_descriptor, fcntl.LOCK_EX)
                        os.fchmod(candidate_descriptor, 0o400)
                        os.fsync(candidate_descriptor)
                        observed = os.fstat(candidate_descriptor)
                        if (
                            int(observed.st_dev),
                            int(observed.st_ino),
                        ) != candidate_identity:
                            raise HemiQGridError(
                                "claim stage changed while binding its inode"
                            )
                        if (
                            _read_descriptor_bytes(
                                candidate_descriptor,
                                source=str(candidate_stage_path),
                            )
                            != candidate_payload
                        ):
                            raise HemiQGridError(
                                "claim stage differs from its intended bytes"
                            )
                        _rename_noreplace_at(
                            parent, staged_name, parent, f"{job.job_id}.json"
                        )
                        os.fsync(parent)
                        candidate_claim = Claim(
                            job=job,
                            path=_claim_path(run_root, job),
                            descriptor=candidate_descriptor,
                            st_dev=int(observed.st_dev),
                            st_ino=int(observed.st_ino),
                            nonce=nonce,
                            value=value,
                        )
                        _assert_claim_path(candidate_claim)
                        created_claim = candidate_claim
                        candidate_descriptor = -1
                    except Exception:
                        if (
                            candidate_identity is not None
                            and candidate_payload is not None
                        ):
                            for candidate_path, category in (
                                (
                                    _claim_path(run_root, job),
                                    "claim_authority_loss",
                                ),
                                (
                                    candidate_stage_path,
                                    "abandoned_claim_stage",
                                ),
                            ):
                                if (
                                    candidate_path is None
                                    or not _path_exists(candidate_path)
                                ):
                                    continue
                                observed_path = _anchored_lstat(
                                    candidate_path
                                )
                                if (
                                    int(observed_path.st_dev),
                                    int(observed_path.st_ino),
                                ) != candidate_identity:
                                    # A replacement is a separate authority.
                                    # It must never be unlinked or relabeled as
                                    # this failed publication's evidence.
                                    continue
                                _quarantine_leaf(
                                    run_root,
                                    candidate_path,
                                    category=category,
                                    expected_identity=candidate_identity,
                                    expected_payload=candidate_payload,
                                )
                        raise
                    finally:
                        if candidate_descriptor >= 0:
                            try:
                                fcntl.flock(
                                    candidate_descriptor,
                                    fcntl.LOCK_UN,
                                )
                            finally:
                                os.close(candidate_descriptor)
                            candidate_descriptor = -1
                        os.close(parent)
    except Exception:
        if created_claim is not None:
            try:
                if _path_exists(created_claim.path):
                    _quarantine_leaf(
                        run_root,
                        created_claim.path,
                        category="claim_authority_loss",
                        expected_identity=(
                            created_claim.st_dev,
                            created_claim.st_ino,
                        ),
                        expected_payload=_canonical_bytes(
                            dict(created_claim.value)
                        ),
                    )
            finally:
                fcntl.flock(created_claim.descriptor, fcntl.LOCK_UN)
                os.close(created_claim.descriptor)
        raise
    if completed:
        return None
    if created_claim is None:
        raise HemiQGridError("claim publication produced no authority")
    return created_claim


def release_claim(
    claim: Claim,
    *,
    plan: Mapping[str, Any],
    project_root: Path,
    run_root: Path,
    gpu_uuid: str,
    lease: project_gpu_leases.GPULease,
) -> None:
    try:
        if _path_exists(claim.path):
            _assert_lease_bound(
                lease,
                project_root=project_root,
                run_root=run_root,
                plan=plan,
                gpu_uuid=gpu_uuid,
            )
            with project_gpu_leases.guard_gpu_lease(lease):
                with _coordination_lock(
                    run_root, exclusive=False, nonblocking=True
                ):
                    _assert_claim_path(claim)
                    parent = _open_directory_absolute(claim.path.parent)
                    try:
                        path_status = os.stat(
                            claim.path.name,
                            dir_fd=parent,
                            follow_symlinks=False,
                        )
                        if (path_status.st_dev, path_status.st_ino) != (
                            claim.st_dev,
                            claim.st_ino,
                        ):
                            raise HemiQGridError(
                                "refusing to remove a replaced claim"
                            )
                        os.unlink(claim.path.name, dir_fd=parent)
                        os.fsync(parent)
                    finally:
                        os.close(parent)
    finally:
        try:
            fcntl.flock(claim.descriptor, fcntl.LOCK_UN)
        finally:
            os.close(claim.descriptor)


def _feature_manifest(views: Mapping[str, np.ndarray]) -> dict[str, Any]:
    if set(views) != {"raw", "covariance"}:
        raise HemiQGridError("HemiQ view producer returned an unknown schema")
    return {
        "covariance": _array_manifest(np.asarray(views["covariance"])),
        "raw": _array_manifest(np.asarray(views["raw"])),
    }


def execute_job(
    *,
    plan: Mapping[str, Any],
    project_root: Path,
    cache_root: Path,
    job: Job,
    gpu_uuid: str,
    device: str = "cuda:0",
    classifier_type: type[HemiQHarmonizedV2Classifier] = HemiQHarmonizedV2Classifier,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Run one score-blind two-phase job without publishing any artifact."""

    validate_plan(plan)
    if job not in expected_jobs():
        raise HemiQGridError("job lies outside the exact HemiQ-v2 plan")
    started = time.perf_counter()
    runtime = _require_single_gpu_runtime(
        plan=plan, gpu_uuid=gpu_uuid, device=device
    )
    cache, (train_rows, validation_rows, test_rows) = _verify_loaded_cache(
        plan,
        cache_root=cache_root,
        subject=job.subject,
        allow_test_class_validation=False,
    )
    channels = tuple(str(value) for value in cache["channel_names"].tolist())
    x = np.asarray(cache["x"], dtype=np.float32)
    y = np.asarray(cache["y"], dtype=np.int64)

    feature_started = time.perf_counter()
    train_views = derive_hemiq_views(x[train_rows], channel_names=channels)
    validation_views = derive_hemiq_views(
        x[validation_rows], channel_names=channels
    )
    test_views = derive_hemiq_views(x[test_rows], channel_names=channels)
    feature_seconds = time.perf_counter() - feature_started

    config, config_identity = load_frozen_config(
        project_root, seed=job.seed, device=device
    )
    planned_config = copy.deepcopy(config_identity)
    planned_config["runtime_overrides"] = ["device", "seed"]
    if planned_config != plan["configuration_identity"]:
        raise HemiQGridError("job configuration differs from the immutable plan")

    selection_started = time.perf_counter()
    selection = classifier_type(config).fit_selection(
        train_views["raw"],
        train_views["covariance"],
        y[train_rows],
        validation_views["raw"],
        y[validation_rows],
    )
    selection_seconds = time.perf_counter() - selection_started
    selected_epochs = selection.selected_epoch_count_
    if (
        not _is_exact_int(selected_epochs, minimum=0)
        or selected_epochs > config.epochs
    ):
        raise HemiQGridError("selection returned an invalid epoch count")

    source_rows = np.sort(np.concatenate((train_rows, validation_rows)))
    source_views = {
        "raw": np.ascontiguousarray(
            np.concatenate(
                (train_views["raw"], validation_views["raw"]), axis=0
            ),
            dtype=np.float32,
        ),
        "covariance": np.ascontiguousarray(
            np.concatenate(
                (
                    train_views["covariance"],
                    validation_views["covariance"],
                ),
                axis=0,
            ),
            dtype=np.float32,
        ),
    }
    source_labels = np.concatenate(
        (y[train_rows], y[validation_rows]), axis=0
    )
    # ``source_rows`` is sorted, and the chronological train rows precede
    # validation rows for BNCI2014-004.  Assert rather than silently reorder
    # the source views relative to their labels.
    if not np.array_equal(
        source_rows, np.concatenate((train_rows, validation_rows))
    ):
        raise HemiQGridError("source partition ordering is not chronological")

    refit_started = time.perf_counter()
    refit = classifier_type(config).fit_fixed_epochs(
        source_views["raw"],
        source_views["covariance"],
        source_labels,
        epochs=selected_epochs,
    )
    refit_seconds = time.perf_counter() - refit_started
    if (
        refit.epochs_run_ != selected_epochs
        or selection.initial_model_state_sha256_
        != refit.initial_model_state_sha256_
    ):
        raise HemiQGridError("reset/refit did not reconstruct the seeded state")

    prediction_started = time.perf_counter()
    # This is intentionally the sole held-out predictor call.
    probabilities = np.asarray(
        refit.predict_proba(test_views["raw"]), dtype=np.float64
    )
    prediction_seconds = time.perf_counter() - prediction_started
    if (
        probabilities.shape != (len(test_rows), 2)
        or not np.all(np.isfinite(probabilities))
        or np.any(probabilities < 0.0)
        or np.any(probabilities > 1.0)
        or not np.allclose(
            probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-7
        )
    ):
        raise HemiQGridError("held-out probability output is invalid")
    test_rows = np.ascontiguousarray(test_rows, dtype=np.int64)
    probabilities = np.ascontiguousarray(probabilities, dtype=np.float64)

    record = {
        "cache": {
            "archive_sha256": plan["cache_identity"][
                f"{DATASET}:s{job.subject:03d}"
            ]["archive"]["sha256"],
            "cache_array_sha256": cache["identity"]["array_sha256"],
        },
        "evidence_scope": plan["evidence_scope"],
        "features": {
            "source": _feature_manifest(source_views),
            "test": _feature_manifest(test_views),
            "train": _feature_manifest(train_views),
            "validation_raw": _array_manifest(validation_views["raw"]),
        },
        "job": job.as_dict(),
        "model": {
            "config_file_sha256": config_identity["config_file_sha256"],
            "initial_state_sha256": refit.initial_model_state_sha256_,
            "parameter_count": refit.parameter_count_,
            "stable_id": MODEL_ID,
            "trainable_parameter_count": refit.trainable_parameter_count_,
        },
        "plan_sha256": plan["plan_sha256"],
        "prediction": {
            "call_count": 1,
            "class_order": [0, 1],
            "probabilities": _array_manifest(probabilities),
            "rows": _rows_manifest(test_rows),
            "threshold_logit": 0.0,
        },
        "refit": {
            "epochs_run": refit.epochs_run_,
            "initial_state_sha256": refit.initial_model_state_sha256_,
            "mode": refit.mode_,
            "model_state_sha256": refit.model_state_sha256_,
            "preprocessing_manifest": copy.deepcopy(
                refit.preprocessing_manifest_
            ),
            "teacher_status": (
                "used" if refit.teacher_was_used_ else "disabled"
            ),
        },
        "runtime": runtime,
        "schema": RECORD_SCHEMA,
        "selection": {
            "epochs_run": selection.epochs_run_,
            "initial_state_sha256": selection.initial_model_state_sha256_,
            "selected_epoch_count": selected_epochs,
            "teacher_status": (
                "used" if selection.teacher_was_used_ else "disabled"
            ),
        },
        "split": copy.deepcopy(
            plan["split_identity"][
                f"{DATASET}:s{job.subject:03d}:f00"
            ]
        ),
        "timing_seconds": {
            "features": feature_seconds,
            "prediction": prediction_seconds,
            "refit": refit_seconds,
            "selection": selection_seconds,
            "total": time.perf_counter() - started,
        },
        "view_contract_sha256": _sha256_bytes(
            _canonical_bytes(plan["view_contract"])
        ),
    }
    _validate_record(
        record, plan=plan, job=job, test_rows=test_rows, probabilities=probabilities
    )
    return record, test_rows, probabilities


def _validate_record(
    value: Any,
    *,
    plan: Mapping[str, Any],
    job: Job,
    test_rows: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    _validate_persisted_predictions(
        plan=plan,
        job=job,
        test_rows=test_rows,
        probabilities=probabilities,
    )
    _exact_keys(
        value,
        {
            "cache",
            "evidence_scope",
            "features",
            "job",
            "model",
            "plan_sha256",
            "prediction",
            "refit",
            "runtime",
            "schema",
            "selection",
            "split",
            "timing_seconds",
            "view_contract_sha256",
        },
        path="record",
    )
    violations = _recursive_forbidden_keys(value)
    if violations:
        raise HemiQGridError(
            f"score-blind record contains outcome aliases: {violations}"
        )
    if (
        value["schema"] != RECORD_SCHEMA
        or value["plan_sha256"] != plan["plan_sha256"]
        or _job_from_mapping(value["job"]) != job
        or value["evidence_scope"] != plan["evidence_scope"]
        or not _same_typed_value(
            value["split"],
            plan["split_identity"][f"{DATASET}:s{job.subject:03d}:f00"],
        )
        or value["view_contract_sha256"]
        != _sha256_bytes(_canonical_bytes(plan["view_contract"]))
    ):
        raise HemiQGridError("record binding is invalid")
    _exact_keys(
        value["cache"],
        {"archive_sha256", "cache_array_sha256"},
        path="record.cache",
    )
    expected_cache = plan["cache_identity"][f"{DATASET}:s{job.subject:03d}"]
    if (
        value["cache"]["archive_sha256"] != expected_cache["archive"]["sha256"]
        or value["cache"]["cache_array_sha256"]
        != expected_cache["embedded_identity"]["array_sha256"]
    ):
        raise HemiQGridError("record cache binding is invalid")
    _exact_keys(
        value["features"],
        {"source", "test", "train", "validation_raw"},
        path="record.features",
    )
    split_partitions = value["split"]["partitions"]
    for name in ("source", "test", "train"):
        view = value["features"][name]
        _exact_keys(
            view, {"covariance", "raw"}, path=f"record.features.{name}"
        )
        count = split_partitions[name]["shape"][0]
        _validate_array_manifest(
            view["raw"],
            path=f"record.features.{name}.raw",
            expected_shape=(count, 3, EXPECTED_N_TIMES),
            expected_dtype=np.dtype(np.float32).str,
        )
        _validate_array_manifest(
            view["covariance"],
            path=f"record.features.{name}.covariance",
            expected_shape=(count, 4, 3, 3),
            expected_dtype=np.dtype(np.float32).str,
        )
    validation_count = split_partitions["validation"]["shape"][0]
    _validate_array_manifest(
        value["features"]["validation_raw"],
        path="record.features.validation_raw",
        expected_shape=(validation_count, 3, EXPECTED_N_TIMES),
        expected_dtype=np.dtype(np.float32).str,
    )
    _exact_keys(
        value["prediction"],
        {
            "call_count",
            "class_order",
            "probabilities",
            "rows",
            "threshold_logit",
        },
        path="record.prediction",
    )
    if (
        not _is_exact_int(value["prediction"]["call_count"], minimum=1)
        or value["prediction"]["call_count"] != 1
        or not _same_typed_value(
            value["prediction"]["class_order"], [0, 1]
        )
        or type(value["prediction"]["threshold_logit"]) is not float
        or value["prediction"]["threshold_logit"] != 0.0
    ):
        raise HemiQGridError("record prediction contract is invalid")
    _validate_array_manifest(
        value["prediction"]["rows"],
        path="record.prediction.rows",
        expected_shape=test_rows.shape,
        expected_dtype=np.dtype(np.int64).str,
    )
    _validate_array_manifest(
        value["prediction"]["probabilities"],
        path="record.prediction.probabilities",
        expected_shape=probabilities.shape,
        expected_dtype=np.dtype(np.float64).str,
    )
    if (
        value["prediction"]["rows"] != _rows_manifest(test_rows)
        or value["prediction"]["probabilities"]
        != _array_manifest(probabilities)
        or value["prediction"]["rows"]
        != value["split"]["partitions"]["test"]
    ):
        raise HemiQGridError("record prediction manifests differ from arrays")
    _exact_keys(
        value["selection"],
        {
            "epochs_run",
            "initial_state_sha256",
            "selected_epoch_count",
            "teacher_status",
        },
        path="record.selection",
    )
    _exact_keys(
        value["refit"],
        {
            "epochs_run",
            "initial_state_sha256",
            "mode",
            "model_state_sha256",
            "preprocessing_manifest",
            "teacher_status",
        },
        path="record.refit",
    )
    selection = value["selection"]
    refit = value["refit"]
    if (
        not _is_exact_int(selection["selected_epoch_count"], minimum=0)
        or selection["selected_epoch_count"] > 240
        or not _is_exact_int(selection["epochs_run"], minimum=0)
        or selection["epochs_run"] > 240
        or selection["selected_epoch_count"] > selection["epochs_run"]
        or not _is_exact_int(refit["epochs_run"], minimum=0)
        or refit["epochs_run"] != selection["selected_epoch_count"]
        or refit["mode"] != "fixed_refit"
        or selection["teacher_status"] != "used"
        or refit["teacher_status"] != "used"
        or selection["initial_state_sha256"]
        != refit["initial_state_sha256"]
        or any(
            not isinstance(value_hash, str)
            or SHA256_RE.fullmatch(value_hash) is None
            for value_hash in (
                selection["initial_state_sha256"],
                refit["model_state_sha256"],
            )
        )
    ):
        raise HemiQGridError("record reset/refit contract is invalid")
    preprocessing = refit["preprocessing_manifest"]
    _exact_keys(
        preprocessing,
        {"raw_mean", "raw_std", "teacher"},
        path="record.refit.preprocessing_manifest",
    )
    for name in ("raw_mean", "raw_std"):
        _validate_array_manifest(
            preprocessing[name],
            path=f"record.refit.preprocessing_manifest.{name}",
            expected_shape=(1, 3, 1),
            expected_dtype=np.dtype(np.float32).str,
        )
    teacher = preprocessing["teacher"]
    _exact_keys(
        teacher,
        {
            "log_reference",
            "logistic_coef",
            "logistic_intercept",
            "scaler_mean",
            "scaler_scale",
        },
        path="record.refit.preprocessing_manifest.teacher",
    )
    for name, manifest in teacher.items():
        _validate_array_manifest(
            manifest,
            path=f"record.refit.preprocessing_manifest.teacher.{name}",
        )
    _exact_keys(
        value["model"],
        {
            "config_file_sha256",
            "initial_state_sha256",
            "parameter_count",
            "stable_id",
            "trainable_parameter_count",
        },
        path="record.model",
    )
    if (
        value["model"]["stable_id"] != MODEL_ID
        or value["model"]["config_file_sha256"]
        != plan["configuration_identity"]["config_file_sha256"]
        or value["model"]["initial_state_sha256"]
        != refit["initial_state_sha256"]
        or not _is_exact_int(value["model"]["parameter_count"], minimum=1)
        or not _is_exact_int(
            value["model"]["trainable_parameter_count"], minimum=1
        )
        or value["model"]["trainable_parameter_count"] >= 50_000
    ):
        raise HemiQGridError("record model identity is invalid")
    _validate_runtime_identity(
        value["runtime"],
        plan=plan,
        gpu_uuid=None,
        path="record.runtime",
    )
    _exact_keys(
        value["timing_seconds"],
        {"features", "prediction", "refit", "selection", "total"},
        path="record.timing_seconds",
    )
    if any(
        type(item) is not float
        or not _is_finite_number(item, minimum=0.0)
        for item in value["timing_seconds"].values()
    ):
        raise HemiQGridError("record timings are invalid")


def _validate_persisted_predictions(
    *,
    plan: Mapping[str, Any],
    job: Job,
    test_rows: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    """Bind decoded prediction arrays to the exact planned held-out rows."""

    rows = np.asarray(test_rows)
    probability = np.asarray(probabilities)
    split_key = f"{DATASET}:s{job.subject:03d}:f00"
    try:
        planned_test = plan["split_identity"][split_key]["partitions"]["test"]
        expected_count = planned_test["shape"][0]
    except (KeyError, IndexError, TypeError) as error:
        raise HemiQGridError(
            "planned held-out row contract is unavailable"
        ) from error
    if (
        not _is_exact_int(expected_count, minimum=1)
        or rows.dtype != np.dtype(np.int64)
        or rows.ndim != 1
        or rows.shape != (expected_count,)
        or _array_manifest(rows) != planned_test
        or np.unique(rows).size != expected_count
        or (expected_count > 1 and np.any(np.diff(rows) <= 0))
    ):
        raise HemiQGridError(
            "prediction rows differ from the exact planned held-out trials"
        )
    if (
        probability.dtype != np.dtype(np.float64)
        or probability.ndim != 2
        or probability.shape != (expected_count, 2)
        or not np.all(np.isfinite(probability))
        or np.any(probability < 0.0)
        or np.any(probability > 1.0)
        or not np.allclose(
            probability.sum(axis=1),
            1.0,
            rtol=0.0,
            atol=1e-12,
        )
    ):
        raise HemiQGridError(
            "persisted probabilities violate the exact binary simplex contract"
        )


def _prediction_npz_bytes(
    test_rows: np.ndarray, probabilities: np.ndarray
) -> bytes:
    buffer = io.BytesIO()
    np.savez(
        buffer,
        test_rows=np.ascontiguousarray(test_rows, dtype=np.int64),
        probabilities=np.ascontiguousarray(probabilities, dtype=np.float64),
    )
    payload = buffer.getvalue()
    loaded = _load_npz_exact(
        payload,
        expected_keys=("test_rows", "probabilities"),
        source="staged predictions",
    )
    if (
        not np.array_equal(loaded["test_rows"], test_rows)
        or not np.array_equal(loaded["probabilities"], probabilities)
    ):
        raise HemiQGridError("staged prediction NPZ changed array bytes")
    return payload


def _claim_receipt(claim: Claim) -> dict[str, Any]:
    payload = _read_descriptor_bytes(claim.descriptor, source=str(claim.path))
    return {
        "claim": copy.deepcopy(dict(claim.value)),
        "claim_sha256": _sha256_bytes(payload),
        "claim_size_bytes": len(payload),
        "claim_st_dev": claim.st_dev,
        "claim_st_ino": claim.st_ino,
        "gpu_lease_receipt": copy.deepcopy(
            claim.value["gpu_lease_receipt"]
        ),
        "nonce": claim.nonce,
    }


def _validate_claim_receipt(
    value: Any,
    *,
    plan: Mapping[str, Any],
    project_root: Path,
    run_root: Path,
    gpu_uuid: str,
    job: Job,
    preflight_payload: bytes | None = None,
) -> None:
    _exact_keys(
        value,
        {
            "claim",
            "claim_sha256",
            "claim_size_bytes",
            "claim_st_dev",
            "claim_st_ino",
            "gpu_lease_receipt",
            "nonce",
        },
        path="completion.claim_receipt",
    )
    if (
        not isinstance(value["claim_sha256"], str)
        or SHA256_RE.fullmatch(value["claim_sha256"]) is None
        or not _is_exact_int(value["claim_size_bytes"], minimum=1)
        or not _is_exact_int(value["claim_st_dev"], minimum=0)
        or not _is_exact_int(value["claim_st_ino"], minimum=1)
        or not isinstance(value["nonce"], str)
        or re.fullmatch(r"[0-9a-f]{32}", value["nonce"]) is None
    ):
        raise HemiQGridError("completion claim receipt is invalid")
    canonical_claim = _canonical_bytes(value["claim"])
    if (
        value["claim_sha256"] != _sha256_bytes(canonical_claim)
        or value["claim_size_bytes"] != len(canonical_claim)
        or value["nonce"] != value["claim"].get("nonce")
    ):
        raise HemiQGridError("completion claim bytes are not self-consistent")
    _validate_claim_value(
        value["claim"],
        job=job,
        plan=plan,
        project_root=project_root,
        run_root=run_root,
        gpu_uuid=gpu_uuid,
        preflight_payload=preflight_payload,
    )
    project_gpu_leases.validate_gpu_lease_receipt(
        value["gpu_lease_receipt"],
        project_root=_absolute(project_root),
        run_root=_absolute(run_root),
        plan_sha256=str(plan["plan_sha256"]),
        gpu_uuid=gpu_uuid,
        track_scope=TRACK_SCOPE,
    )


def _completion_value(
    *,
    plan: Mapping[str, Any],
    job: Job,
    gpu_uuid: str,
    record_payload: bytes,
    prediction_payload: bytes,
    claim: Claim,
    commit_lease_receipt: Mapping[str, Any],
    directory_status: os.stat_result,
    commit_disk_guard: Mapping[str, Any],
) -> dict[str, Any]:
    artifacts = {
        "predictions.npz": {
            "sha256": _sha256_bytes(prediction_payload),
            "size_bytes": len(prediction_payload),
        },
        "record.json": {
            "sha256": _sha256_bytes(record_payload),
            "size_bytes": len(record_payload),
        },
    }
    return {
        "artifacts": artifacts,
        "claim_receipt": _claim_receipt(claim),
        "commit_disk_guard": copy.deepcopy(dict(commit_disk_guard)),
        "commit_gpu_lease_receipt": copy.deepcopy(dict(commit_lease_receipt)),
        "completed_at": _utc_now(),
        "directory_mode": "0555",
        "directory_st_dev": int(directory_status.st_dev),
        "directory_st_ino": int(directory_status.st_ino),
        "gpu_uuid": gpu_uuid,
        "job": job.as_dict(),
        "plan_sha256": plan["plan_sha256"],
        "schema": COMPLETION_SCHEMA,
        "seal_sha256": _sha256_bytes(_canonical_bytes(artifacts)),
    }


def _unlink_claim_path(claim: Claim) -> None:
    _assert_claim_path(claim)
    parent = _open_directory_absolute(claim.path.parent)
    try:
        observed = os.stat(
            claim.path.name, dir_fd=parent, follow_symlinks=False
        )
        if (observed.st_dev, observed.st_ino) != (claim.st_dev, claim.st_ino):
            raise HemiQGridError("claim path changed before completion")
        os.unlink(claim.path.name, dir_fd=parent)
        os.fsync(parent)
    finally:
        os.close(parent)


def commit_job_output(
    *,
    plan: Mapping[str, Any],
    project_root: Path,
    run_root: Path,
    job: Job,
    gpu_uuid: str,
    lease: project_gpu_leases.GPULease,
    claim: Claim,
    record: Mapping[str, Any],
    test_rows: np.ndarray,
    probabilities: np.ndarray,
) -> Path:
    """Crash-atomically publish one read-only result directory."""

    if claim.job != job:
        raise HemiQGridError("claim belongs to a different job")
    _assert_claim_path(claim)
    _validate_persisted_predictions(
        plan=plan,
        job=job,
        test_rows=np.asarray(test_rows),
        probabilities=np.asarray(probabilities),
    )
    rows = np.ascontiguousarray(test_rows)
    probability = np.ascontiguousarray(probabilities)
    _validate_record(
        record,
        plan=plan,
        job=job,
        test_rows=rows,
        probabilities=probability,
    )
    record_payload = _canonical_bytes(dict(record))
    prediction_payload = _prediction_npz_bytes(rows, probability)
    destination = _record_directory(run_root, job)
    published = False
    publication_identity: tuple[int, int] | None = None
    try:
        with project_gpu_leases.guard_gpu_lease(lease) as commit_receipt:
            with _coordination_lock(run_root, exclusive=False, nonblocking=True):
                _assert_claim_path(claim)
                commit_disk_guard = _hard_disk_guard(run_root)
                validate_preflight(
                    plan=plan,
                    project_root=project_root,
                    run_root=run_root,
                    gpu_uuid=gpu_uuid,
                )
                if _path_exists(destination):
                    raise FileExistsError(str(destination))
                # Stage under the records parent.  The lab filesystem permits
                # a same-parent atomic rename after chmod 0555 but rejects a
                # cross-parent rename of a read-only directory.
                staging_root = _absolute(run_root) / "records"
                stage_name = f"{job.job_id}.{claim.nonce}.stage"
                stage_path = staging_root / stage_name
                stage_parent = _open_directory_absolute(staging_root)
                stage_descriptor = -1
                stage_identity: tuple[int, int] | None = None
                try:
                    os.mkdir(stage_name, mode=0o700, dir_fd=stage_parent)
                    stage_descriptor = os.open(
                        stage_name, _directory_flags(), dir_fd=stage_parent
                    )
                    stage_status = os.fstat(stage_descriptor)
                    stage_identity = (
                        int(stage_status.st_dev),
                        int(stage_status.st_ino),
                    )
                    completion = _completion_value(
                        plan=plan,
                        job=job,
                        gpu_uuid=gpu_uuid,
                        record_payload=record_payload,
                        prediction_payload=prediction_payload,
                        claim=claim,
                        commit_lease_receipt=commit_receipt,
                        directory_status=stage_status,
                        commit_disk_guard=commit_disk_guard,
                    )
                    for name, payload in (
                        ("record.json", record_payload),
                        ("predictions.npz", prediction_payload),
                        ("completion.json", _canonical_bytes(completion)),
                    ):
                        _write_exclusive_at(
                            stage_descriptor, name, payload, mode=0o400
                        )
                    os.fsync(stage_descriptor)
                    os.fchmod(stage_descriptor, 0o555)
                    os.fsync(stage_descriptor)
                    readonly = os.fstat(stage_descriptor)
                    if (
                        (readonly.st_mode & 0o777) != 0o555
                        or (readonly.st_dev, readonly.st_ino)
                        != (stage_status.st_dev, stage_status.st_ino)
                    ):
                        raise HemiQGridError(
                            "staged result directory did not become immutable"
                        )
                    records_parent = _open_directory_absolute(destination.parent)
                    try:
                        stage_parent_status = os.fstat(stage_parent)
                        records_parent_status = os.fstat(records_parent)
                        if (
                            stage_parent_status.st_dev,
                            stage_parent_status.st_ino,
                        ) != (
                            records_parent_status.st_dev,
                            records_parent_status.st_ino,
                        ):
                            raise HemiQGridError(
                                "result publication requires a same-parent rename"
                            )
                        publication_identity = (
                            int(readonly.st_dev),
                            int(readonly.st_ino),
                        )
                        _verify_worker_publication_identity(
                            plan=plan,
                            project_root=project_root,
                        )
                        _assert_claim_path(claim)
                        _rename_noreplace_at(
                            stage_parent,
                            stage_name,
                            records_parent,
                            destination.name,
                        )
                        # This assignment is the first operation after a
                        # successful rename. Failures in either parent fsync
                        # or any later inspection quarantine this exact inode.
                        published = True
                        os.fsync(stage_parent)
                        os.fsync(records_parent)
                        published_status = os.stat(
                            destination.name,
                            dir_fd=records_parent,
                            follow_symlinks=False,
                        )
                        if (
                            (published_status.st_dev, published_status.st_ino)
                            != (readonly.st_dev, readonly.st_ino)
                            or not stat.S_ISDIR(published_status.st_mode)
                            or (published_status.st_mode & 0o222) != 0
                        ):
                            raise HemiQGridError(
                                "published result directory authority changed"
                            )
                    finally:
                        os.close(records_parent)
                    _unlink_claim_path(claim)
                except Exception:
                    if not published and _path_exists(stage_path):
                        _quarantine_leaf(
                            run_root,
                            stage_path,
                            category="failed_stage",
                            expected_identity=stage_identity,
                        )
                    raise
                finally:
                    if stage_descriptor >= 0:
                        os.close(stage_descriptor)
                    os.close(stage_parent)
    except Exception:
        if publication_identity is not None and _path_exists(destination):
            _quarantine_leaf(
                run_root,
                destination,
                category="post_commit_authority_loss",
                expected_identity=publication_identity,
            )
        raise
    try:
        validate_completion(
            plan=plan,
            project_root=project_root,
            run_root=run_root,
            job=job,
        )
        _verify_worker_publication_identity(
            plan=plan,
            project_root=project_root,
        )
        final_status = _anchored_lstat(destination)
        if publication_identity is None or (
            int(final_status.st_dev),
            int(final_status.st_ino),
        ) != publication_identity:
            raise HemiQGridError(
                "result authority changed after final worker identity rebind"
            )
    except Exception:
        if _path_exists(destination):
            _quarantine_leaf(
                run_root,
                destination,
                category="post_commit_validation_loss",
                expected_identity=publication_identity,
            )
        raise
    return destination


def _validate_completion_value(
    value: Any,
    *,
    plan: Mapping[str, Any],
    project_root: Path,
    run_root: Path,
    job: Job,
    record_payload: bytes,
    prediction_payload: bytes,
    directory_status: os.stat_result,
    preflight_payload: bytes,
) -> None:
    _exact_keys(
        value,
        {
            "artifacts",
            "claim_receipt",
            "commit_disk_guard",
            "commit_gpu_lease_receipt",
            "completed_at",
            "directory_mode",
            "directory_st_dev",
            "directory_st_ino",
            "gpu_uuid",
            "job",
            "plan_sha256",
            "schema",
            "seal_sha256",
        },
        path="completion",
    )
    gpu_uuid = value["gpu_uuid"]
    if (
        value["schema"] != COMPLETION_SCHEMA
        or value["plan_sha256"] != plan["plan_sha256"]
        or _job_from_mapping(value["job"]) != job
        or value["directory_mode"] != "0555"
        or not _is_exact_int(value["directory_st_dev"], minimum=0)
        or value["directory_st_dev"] != int(directory_status.st_dev)
        or not _is_exact_int(value["directory_st_ino"], minimum=1)
        or value["directory_st_ino"] != int(directory_status.st_ino)
        or not isinstance(value["completed_at"], str)
        or not isinstance(gpu_uuid, str)
        or gpu_uuid not in plan["execution"]["physical_gpu_uuid_roster"]
    ):
        raise HemiQGridError("completion identity is invalid")
    expected_artifacts = {
        "predictions.npz": {
            "sha256": _sha256_bytes(prediction_payload),
            "size_bytes": len(prediction_payload),
        },
        "record.json": {
            "sha256": _sha256_bytes(record_payload),
            "size_bytes": len(record_payload),
        },
    }
    if (
        not _same_typed_value(value["artifacts"], expected_artifacts)
        or value["seal_sha256"]
        != _sha256_bytes(_canonical_bytes(expected_artifacts))
    ):
        raise HemiQGridError("completion artifact seal is invalid")
    _validate_claim_receipt(
        value["claim_receipt"],
        plan=plan,
        project_root=project_root,
        run_root=run_root,
        gpu_uuid=gpu_uuid,
        job=job,
        preflight_payload=preflight_payload,
    )
    _exact_keys(
        value["commit_disk_guard"],
        {"free_bytes", "minimum_free_bytes", "path", "status"},
        path="completion.commit_disk_guard",
    )
    if (
        value["commit_disk_guard"]["status"] != "pass"
        or not _is_exact_int(
            value["commit_disk_guard"]["free_bytes"], minimum=1
        )
        or not _is_exact_int(
            value["commit_disk_guard"]["minimum_free_bytes"],
            minimum=int(MINIMUM_FREE_GIB * 1024**3),
        )
        or value["commit_disk_guard"]["path"] != str(_absolute(run_root))
    ):
        raise HemiQGridError("completion commit disk guard is invalid")
    project_gpu_leases.validate_gpu_lease_receipt(
        value["commit_gpu_lease_receipt"],
        project_root=_absolute(project_root),
        run_root=_absolute(run_root),
        plan_sha256=str(plan["plan_sha256"]),
        gpu_uuid=gpu_uuid,
        track_scope=TRACK_SCOPE,
    )


def validate_completion(
    *,
    plan: Mapping[str, Any],
    project_root: Path,
    run_root: Path,
    job: Job,
    fence: AnalysisFence | None = None,
) -> dict[str, Any]:
    """Validate one coherent descriptor-held result package.

    Every child descriptor is opened before any child is read. The held root,
    records directory, job directory, and all three leaves are then rebound to
    their exact names and full stat fingerprints after all reads. Analysis can
    additionally supply its live fence so traversal starts from the pinned
    run-root descriptor instead of resolving the canonical path again.
    """

    validate_plan(plan)
    root = _absolute(run_root)
    directory = _record_directory(root, job)
    if fence is not None:
        if fence.run_root != root:
            raise HemiQGridError(
                "completion root differs from the analysis fence"
            )
        assert_analysis_fence(fence, plan=plan)
        root_descriptor = os.dup(fence.root_descriptor)
    else:
        root_descriptor = _open_directory_absolute(root)
    records_descriptor = -1
    directory_descriptor = -1
    try:
        root_status = os.fstat(root_descriptor)
        root_fingerprint = _stat_fingerprint(root_status)
        if (
            not stat.S_ISDIR(root_status.st_mode)
            or (
                fence is not None
                and (int(root_status.st_dev), int(root_status.st_ino))
                != (fence.root_st_dev, fence.root_st_ino)
            )
            or (
                fence is None
                and _stat_fingerprint(_anchored_lstat(root))
                != root_fingerprint
            )
        ):
            raise HemiQGridError(
                "completion run-root descriptor is detached"
            )
        records_descriptor = os.open(
            "records",
            _directory_flags(),
            dir_fd=root_descriptor,
        )
        records_status = os.fstat(records_descriptor)
        records_fingerprint = _stat_fingerprint(records_status)
        records_path_status = os.stat(
            "records",
            dir_fd=root_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(records_status.st_mode)
            or _stat_fingerprint(records_path_status)
            != records_fingerprint
        ):
            raise HemiQGridError(
                "completion records directory is detached"
            )
        directory_descriptor = os.open(
            job.job_id,
            _directory_flags(),
            dir_fd=records_descriptor,
        )
        directory_status = os.fstat(directory_descriptor)
        directory_fingerprint = _stat_fingerprint(directory_status)
        path_status = os.stat(
            job.job_id,
            dir_fd=records_descriptor,
            follow_symlinks=False,
        )
        if (
            _stat_fingerprint(path_status) != directory_fingerprint
            or not stat.S_ISDIR(directory_status.st_mode)
            or (directory_status.st_mode & 0o222) != 0
            or (directory_status.st_mode & 0o777) != 0o555
        ):
            raise HemiQGridError(
                f"result directory is writable or replaced: {directory}"
            )
        names = sorted(os.listdir(directory_descriptor))
        if names != sorted(RESULT_FILENAMES):
            raise HemiQGridError(
                f"result directory has unexpected leaves: {directory}"
            )
        payloads: dict[str, bytes] = {}
        artifact_roster: list[dict[str, Any]] = []
        descriptors: dict[str, int] = {}
        fingerprints: dict[str, tuple[int, ...]] = {}
        try:
            # Hold the entire flat package before reading any member.
            for name in names:
                path = directory / name
                descriptor = os.open(
                    name,
                    os.O_RDONLY
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_NONBLOCK", 0),
                    dir_fd=directory_descriptor,
                )
                descriptors[name] = descriptor
                opened = os.fstat(descriptor)
                status = os.stat(
                    name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
                _assert_unique_regular(opened, source=str(path))
                fingerprint = _stat_fingerprint(opened)
                if (
                    stat.S_IMODE(opened.st_mode) != 0o400
                    or _stat_fingerprint(status) != fingerprint
                ):
                    raise HemiQGridError(
                        f"result leaf is writable or detached: {path}"
                    )
                fingerprints[name] = fingerprint
            if (
                _stat_fingerprint(os.fstat(directory_descriptor))
                != directory_fingerprint
                or sorted(os.listdir(directory_descriptor)) != names
                or _stat_fingerprint(
                    os.stat(
                        job.job_id,
                        dir_fd=records_descriptor,
                        follow_symlinks=False,
                    )
                )
                != directory_fingerprint
            ):
                raise HemiQGridError(
                    f"result package changed while opening: {directory}"
                )

            for name in names:
                payload = _read_descriptor_bytes(
                    descriptors[name],
                    source=str(directory / name),
                )
                expected_size = fingerprints[name][
                    STAT_FINGERPRINT_FIELDS.index("st_size")
                ]
                if len(payload) != expected_size:
                    raise HemiQGridError(
                        f"result leaf read was incomplete: {directory / name}"
                    )
                payloads[name] = payload

            # Rebind the full held package only after every read completes.
            for name in names:
                held_after = os.fstat(descriptors[name])
                path_after = os.stat(
                    name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
                if (
                    _stat_fingerprint(held_after) != fingerprints[name]
                    or _stat_fingerprint(path_after) != fingerprints[name]
                ):
                    raise HemiQGridError(
                        f"result leaf changed during package snapshot: "
                        f"{directory / name}"
                    )
            records_after = os.fstat(records_descriptor)
            records_path_after = os.stat(
                "records",
                dir_fd=root_descriptor,
                follow_symlinks=False,
            )
            records_identity = (
                int(records_status.st_dev),
                int(records_status.st_ino),
            )
            records_changed = (
                (
                    int(records_after.st_dev),
                    int(records_after.st_ino),
                )
                != records_identity
                or (
                    int(records_path_after.st_dev),
                    int(records_path_after.st_ino),
                )
                != records_identity
                or (
                    fence is not None
                    and (
                        _stat_fingerprint(records_after)
                        != records_fingerprint
                        or _stat_fingerprint(records_path_after)
                        != records_fingerprint
                    )
                )
            )
            if (
                _stat_fingerprint(os.fstat(directory_descriptor))
                != directory_fingerprint
                or _stat_fingerprint(
                    os.stat(
                        job.job_id,
                        dir_fd=records_descriptor,
                        follow_symlinks=False,
                    )
                )
                != directory_fingerprint
                or sorted(os.listdir(directory_descriptor)) != names
                or records_changed
            ):
                raise HemiQGridError(
                    f"result package changed during validation: {directory}"
                )
        finally:
            for descriptor in descriptors.values():
                os.close(descriptor)

        artifact_roster = [
            {
                "relative_path": str(
                    (directory / name).relative_to(root)
                ),
                "sha256": _sha256_bytes(payloads[name]),
                "size_bytes": len(payloads[name]),
            }
            for name in names
        ]
        record = _strict_json_bytes(
            payloads["record.json"], source=str(directory / "record.json")
        )
        arrays = _load_npz_exact(
            payloads["predictions.npz"],
            expected_keys=("test_rows", "probabilities"),
            source=str(directory / "predictions.npz"),
        )
        rows = np.asarray(arrays["test_rows"])
        probabilities = np.asarray(arrays["probabilities"])
        if rows.dtype != np.int64 or probabilities.dtype != np.float64:
            raise HemiQGridError("prediction NPZ dtypes are invalid")
        _validate_record(
            record,
            plan=plan,
            job=job,
            test_rows=rows,
            probabilities=probabilities,
        )
        completion = _strict_json_bytes(
            payloads["completion.json"],
            source=str(directory / "completion.json"),
        )
        gpu_uuid = completion.get("gpu_uuid")
        if not isinstance(gpu_uuid, str):
            raise HemiQGridError(
                "completion lacks a physical GPU UUID"
            )
        preflight_path = _preflight_path(root, gpu_uuid)
        preflight_descriptor = os.open(
            "preflight",
            _directory_flags(),
            dir_fd=root_descriptor,
        )
        try:
            preflight_directory_status = os.fstat(preflight_descriptor)
            preflight_directory_fingerprint = _stat_fingerprint(
                preflight_directory_status
            )
            if (
                not stat.S_ISDIR(preflight_directory_status.st_mode)
                or _stat_fingerprint(
                    os.stat(
                        "preflight",
                        dir_fd=root_descriptor,
                        follow_symlinks=False,
                    )
                )
                != preflight_directory_fingerprint
            ):
                raise HemiQGridError(
                    "completion preflight directory is detached"
                )
            preflight_payload, preflight_status = _read_unique_regular_at(
                preflight_descriptor,
                preflight_path.name,
                source=str(preflight_path),
            )
            if stat.S_IMODE(preflight_status.st_mode) != 0o400:
                raise HemiQGridError(
                    "completion preflight receipt is writable"
                )
            preflight_after = os.fstat(preflight_descriptor)
            preflight_path_after = os.stat(
                "preflight",
                dir_fd=root_descriptor,
                follow_symlinks=False,
            )
            preflight_identity = (
                int(preflight_directory_status.st_dev),
                int(preflight_directory_status.st_ino),
            )
            if (
                (
                    int(preflight_after.st_dev),
                    int(preflight_after.st_ino),
                )
                != preflight_identity
                or (
                    int(preflight_path_after.st_dev),
                    int(preflight_path_after.st_ino),
                )
                != preflight_identity
                or (
                    fence is not None
                    and (
                        _stat_fingerprint(preflight_after)
                        != preflight_directory_fingerprint
                        or _stat_fingerprint(preflight_path_after)
                        != preflight_directory_fingerprint
                    )
                )
            ):
                raise HemiQGridError(
                    "completion preflight package changed during snapshot"
                )
        finally:
            os.close(preflight_descriptor)
        _validate_completion_value(
            completion,
            plan=plan,
            project_root=project_root,
            run_root=run_root,
            job=job,
            record_payload=payloads["record.json"],
            prediction_payload=payloads["predictions.npz"],
            directory_status=directory_status,
            preflight_payload=preflight_payload,
        )
        if (
            record["runtime"]["gpu"]["uuid"]
            != completion["gpu_uuid"]
            or completion["claim_receipt"]["claim"]["gpu_uuid"]
            != completion["gpu_uuid"]
        ):
            raise HemiQGridError(
                "record, claim, and completion GPU identities differ"
            )
        if (
            _stat_fingerprint(os.fstat(directory_descriptor))
            != directory_fingerprint
            or _stat_fingerprint(
                os.stat(
                    job.job_id,
                    dir_fd=records_descriptor,
                    follow_symlinks=False,
                )
            )
            != directory_fingerprint
            or _stat_fingerprint(os.fstat(root_descriptor))
            != root_fingerprint
        ):
            raise HemiQGridError("result directory changed during validation")
        if fence is not None:
            assert_analysis_fence(fence, plan=plan)
        elif _stat_fingerprint(_anchored_lstat(root)) != root_fingerprint:
            raise HemiQGridError(
                "completion run root changed during validation"
            )
        return {
            "artifact_roster": artifact_roster,
            "completion": completion,
            "probabilities": probabilities,
            "record": record,
            "test_rows": rows,
        }
    finally:
        if directory_descriptor >= 0:
            os.close(directory_descriptor)
        if records_descriptor >= 0:
            os.close(records_descriptor)
        os.close(root_descriptor)


def _recover_job_partials_guarded(*, run_root: Path, job: Job) -> list[Path]:
    recovered = _recover_regular_stages_guarded(
        run_root=run_root,
        parent=_absolute(run_root) / "claims",
        associated_prefix=f".{job.job_id}.",
        pattern=re.compile(
            rf"^\.{re.escape(job.job_id)}\.[0-9a-f]{{32}}\.stage$"
        ),
        category="abandoned_claim_stage",
    )
    recovered.extend(
        _recover_regular_stages_guarded(
            run_root=run_root,
            parent=_absolute(run_root) / "failures",
            associated_prefix=f".{job.job_id}-",
            pattern=re.compile(
                rf"^\.{re.escape(job.job_id)}-[0-9a-f]{{32}}"
                r"\.json\.stage-[0-9a-f]{32}$"
            ),
            category="abandoned_failure_stage",
        )
    )
    for staging in (
        _absolute(run_root) / "records",
        _absolute(run_root) / "staging",
    ):
        descriptor = _open_directory_absolute(staging)
        try:
            for name in sorted(os.listdir(descriptor)):
                if not name.startswith(f"{job.job_id}.") or not name.endswith(".stage"):
                    continue
                status = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                if not stat.S_ISDIR(status.st_mode):
                    raise HemiQGridError(
                        f"job staging leaf is not a directory: {name}"
                    )
                recovered.append(
                    _quarantine_leaf(
                        run_root,
                        staging / name,
                        category="abandoned_stage",
                        expected_identity=(
                            int(status.st_dev),
                            int(status.st_ino),
                        ),
                    )
                )
        finally:
            os.close(descriptor)
    return recovered


def recover_job_partials(
    *,
    plan: Mapping[str, Any],
    project_root: Path,
    run_root: Path,
    job: Job,
    gpu_uuid: str,
    lease: project_gpu_leases.GPULease,
) -> list[Path]:
    """Guard and quarantine only abandoned, job-owned staging directories."""

    _assert_lease_bound(
        lease,
        project_root=project_root,
        run_root=run_root,
        plan=plan,
        gpu_uuid=gpu_uuid,
    )
    with project_gpu_leases.guard_gpu_lease(lease):
        with _coordination_lock(run_root, exclusive=False, nonblocking=True):
            _, recovered = _recover_claim_before_partials(
                plan=plan,
                project_root=project_root,
                run_root=run_root,
                job=job,
            )
            return recovered


def _record_failure(
    *,
    plan: Mapping[str, Any],
    project_root: Path,
    run_root: Path,
    job: Job,
    gpu_uuid: str,
    lease: project_gpu_leases.GPULease,
    error: BaseException,
) -> Path:
    nonce = uuid.uuid4().hex
    path = _failure_path(run_root, job, nonce)
    published_identity: tuple[int, int] | None = None
    published_payload: bytes | None = None
    try:
        with project_gpu_leases.guard_gpu_lease(lease) as lease_receipt:
            with _coordination_lock(run_root, exclusive=False, nonblocking=True):
                _recover_regular_stages_guarded(
                    run_root=run_root,
                    parent=_absolute(run_root) / "failures",
                    associated_prefix=f".{job.job_id}-",
                    pattern=re.compile(
                        rf"^\.{re.escape(job.job_id)}-[0-9a-f]{{32}}"
                        r"\.json\.stage-[0-9a-f]{32}$"
                    ),
                    category="abandoned_failure_stage",
                )
                value = {
                    "created_at": _utc_now(),
                    "error_message": str(error)[:2000],
                    "error_type": type(error).__name__,
                    "gpu_lease_receipt": lease_receipt,
                    "gpu_uuid": gpu_uuid,
                    "job": job.as_dict(),
                    "nonce": nonce,
                    "plan_sha256": plan["plan_sha256"],
                    "schema": FAILURE_SCHEMA,
                }
                (
                    published_payload,
                    published_identity,
                ) = _atomic_json_with_identity(path, value)
        return path
    except Exception:
        if published_identity is not None and _path_exists(path):
            _quarantine_leaf(
                run_root,
                path,
                category="failure_authority_loss",
                expected_identity=published_identity,
                expected_payload=published_payload,
            )
        raise


def _directory_entries(path: Path) -> dict[str, os.stat_result]:
    descriptor = _open_directory_absolute(path)
    try:
        return {
            name: os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            for name in sorted(os.listdir(descriptor))
        }
    finally:
        os.close(descriptor)


def _roster_entry(run_root: Path, path: Path) -> dict[str, Any]:
    payload = _read_unique_regular_bytes(path)
    return {
        "relative_path": str(path.relative_to(_absolute(run_root))),
        "sha256": _sha256_bytes(payload),
        "size_bytes": len(payload),
    }


def _validate_analysis_fence_marker(
    value: Any, *, plan: Mapping[str, Any]
) -> None:
    _exact_keys(
        value,
        {"created_at", "nonce", "owner", "plan_sha256", "schema"},
        path="analysis_fence",
    )
    if (
        value["schema"] != ANALYSIS_FENCE_SCHEMA
        or value["plan_sha256"] != plan["plan_sha256"]
        or not isinstance(value["created_at"], str)
        or not isinstance(value["owner"], Mapping)
        or not isinstance(value["nonce"], str)
        or re.fullmatch(r"[0-9a-f]{32}", value["nonce"]) is None
    ):
        raise HemiQGridError("analysis fence marker is invalid")


def assert_analysis_fence(
    fence: AnalysisFence, *, plan: Mapping[str, Any]
) -> None:
    root_status = os.fstat(fence.root_descriptor)
    root_path_status = _anchored_lstat(fence.run_root)
    authority_status = os.fstat(fence.authority_descriptor)
    lock_status = os.fstat(fence.descriptor)
    authority_path_status = os.stat(
        ".authority",
        dir_fd=fence.root_descriptor,
        follow_symlinks=False,
    )
    lock_path_status = os.stat(
        "coordination.lock",
        dir_fd=fence.authority_descriptor,
        follow_symlinks=False,
    )
    if (
        (root_status.st_dev, root_status.st_ino)
        != (fence.root_st_dev, fence.root_st_ino)
        or (root_path_status.st_dev, root_path_status.st_ino)
        != (fence.root_st_dev, fence.root_st_ino)
        or not stat.S_ISDIR(root_status.st_mode)
        or (authority_status.st_dev, authority_status.st_ino)
        != (fence.authority_st_dev, fence.authority_st_ino)
        or (authority_path_status.st_dev, authority_path_status.st_ino)
        != (fence.authority_st_dev, fence.authority_st_ino)
        or not stat.S_ISDIR(authority_status.st_mode)
        or (lock_status.st_dev, lock_status.st_ino)
        != (fence.st_dev, fence.st_ino)
        or (lock_path_status.st_dev, lock_path_status.st_ino)
        != (fence.st_dev, fence.st_ino)
        or not stat.S_ISREG(lock_status.st_mode)
        or lock_status.st_nlink != 1
    ):
        raise HemiQGridError(
            "analysis run-root namespace or coordination fence was replaced"
        )
    marker_payload, marker_status = _read_unique_regular_at(
        fence.authority_descriptor,
        fence.marker_path.name,
        source=str(fence.marker_path),
    )
    if (
        (int(marker_status.st_dev), int(marker_status.st_ino))
        != (fence.marker_st_dev, fence.marker_st_ino)
        or marker_status.st_mode & 0o222
    ):
        raise HemiQGridError("analysis fence marker authority changed")
    marker = _strict_json_bytes(
        marker_payload,
        source=str(fence.marker_path),
    )
    _validate_analysis_fence_marker(marker, plan=plan)
    if marker != dict(fence.marker_value):
        raise HemiQGridError("analysis fence marker changed")
    # Repeat the canonical root binding after the marker snapshot. This closes
    # the specific path-swap window between checking the lock and reading the
    # marker, while all traversal-capable descriptors remain open.
    final_root = _anchored_lstat(fence.run_root)
    final_authority = os.stat(
        ".authority",
        dir_fd=fence.root_descriptor,
        follow_symlinks=False,
    )
    if (
        (final_root.st_dev, final_root.st_ino)
        != (fence.root_st_dev, fence.root_st_ino)
        or (final_authority.st_dev, final_authority.st_ino)
        != (fence.authority_st_dev, fence.authority_st_ino)
    ):
        raise HemiQGridError(
            "analysis run-root namespace changed during fence assertion"
        )


@contextmanager
def analysis_fence(
    *,
    run_root: Path,
    plan: Mapping[str, Any],
) -> Iterator[AnalysisFence]:
    """Exclude all cooperating claims/commits for audit and analysis."""

    validate_plan(plan)
    root = _absolute(run_root)
    root_descriptor = _open_directory_absolute(root)
    authority_descriptor = -1
    descriptor = -1
    marker_created = False
    marker_name = ""
    marker_identity: tuple[int, int] | None = None
    try:
        root_status = os.fstat(root_descriptor)
        root_path_status = _anchored_lstat(root)
        if (
            (root_status.st_dev, root_status.st_ino)
            != (root_path_status.st_dev, root_path_status.st_ino)
            or not stat.S_ISDIR(root_status.st_mode)
        ):
            raise HemiQGridError(
                "analysis run root changed while binding its namespace"
            )
        authority_descriptor = os.open(
            ".authority",
            _directory_flags(),
            dir_fd=root_descriptor,
        )
        authority_status = os.fstat(authority_descriptor)
        authority_path_status = os.stat(
            ".authority",
            dir_fd=root_descriptor,
            follow_symlinks=False,
        )
        if (
            (authority_status.st_dev, authority_status.st_ino)
            != (authority_path_status.st_dev, authority_path_status.st_ino)
            or not stat.S_ISDIR(authority_status.st_mode)
        ):
            raise HemiQGridError(
                "analysis authority directory changed while opening"
            )
        lock_path_status = os.stat(
            "coordination.lock",
            dir_fd=authority_descriptor,
            follow_symlinks=False,
        )
        _assert_unique_regular(
            lock_path_status,
            source=str(root / ".authority" / "coordination.lock"),
            allow_empty=True,
        )
        descriptor = os.open(
            "coordination.lock",
            os.O_RDWR
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=authority_descriptor,
        )
        lock_status = os.fstat(descriptor)
        if (
            (lock_status.st_dev, lock_status.st_ino)
            != (lock_path_status.st_dev, lock_path_status.st_ino)
        ):
            raise HemiQGridError(
                "analysis coordination lock changed while opening"
            )
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        nonce = uuid.uuid4().hex
        marker_name = f"analysis-{nonce}.json"
        marker_path = root / ".authority" / marker_name
        value = {
            "created_at": _utc_now(),
            "nonce": nonce,
            "owner": project_gpu_leases.owner_identity(),
            "plan_sha256": plan["plan_sha256"],
            "schema": ANALYSIS_FENCE_SCHEMA,
        }
        _validate_analysis_fence_marker(value, plan=plan)
        marker_status = _write_exclusive_at(
            authority_descriptor,
            marker_name,
            _canonical_bytes(value),
            mode=0o400,
        )
        marker_created = True
        marker_identity = (
            int(marker_status.st_dev),
            int(marker_status.st_ino),
        )
        os.fsync(authority_descriptor)
        fence = AnalysisFence(
            run_root=root,
            root_descriptor=root_descriptor,
            root_st_dev=int(root_status.st_dev),
            root_st_ino=int(root_status.st_ino),
            authority_descriptor=authority_descriptor,
            authority_st_dev=int(authority_status.st_dev),
            authority_st_ino=int(authority_status.st_ino),
            lock_path=root / ".authority" / "coordination.lock",
            descriptor=descriptor,
            st_dev=int(lock_status.st_dev),
            st_ino=int(lock_status.st_ino),
            marker_path=marker_path,
            marker_st_dev=marker_identity[0],
            marker_st_ino=marker_identity[1],
            marker_value=value,
        )
        try:
            assert_analysis_fence(fence, plan=plan)
            yield fence
            assert_analysis_fence(fence, plan=plan)
        finally:
            if marker_created:
                try:
                    observed = os.stat(
                        marker_name,
                        dir_fd=authority_descriptor,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    marker_created = False
                else:
                    marker_payload, opened = _read_unique_regular_at(
                        authority_descriptor,
                        marker_name,
                        source=str(marker_path),
                    )
                    marker = _strict_json_bytes(
                        marker_payload,
                        source=str(marker_path),
                    )
                    if (
                        marker_identity is None
                        or (
                            int(observed.st_dev),
                            int(observed.st_ino),
                        )
                        != marker_identity
                        or (
                            int(opened.st_dev),
                            int(opened.st_ino),
                        )
                        != marker_identity
                        or marker != value
                    ):
                        raise HemiQGridError(
                            "refusing to remove a replaced analysis marker"
                        )
                    os.unlink(marker_name, dir_fd=authority_descriptor)
                    os.fsync(authority_descriptor)
                    marker_created = False
    finally:
        if marker_created and authority_descriptor >= 0 and marker_name:
            # Best-effort cleanup for failures between marker creation and
            # yielding the fully constructed fence. Never resolve through the
            # possibly replaced canonical run-root path.
            try:
                observed = os.stat(
                    marker_name,
                    dir_fd=authority_descriptor,
                    follow_symlinks=False,
                )
                if marker_identity is not None and (
                    int(observed.st_dev),
                    int(observed.st_ino),
                ) == marker_identity:
                    os.unlink(marker_name, dir_fd=authority_descriptor)
                    os.fsync(authority_descriptor)
            except FileNotFoundError:
                pass
        if descriptor >= 0:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
        if authority_descriptor >= 0:
            os.close(authority_descriptor)
        os.close(root_descriptor)


def audit_grid(
    *,
    plan: Mapping[str, Any],
    project_root: Path,
    run_root: Path,
    fence: AnalysisFence,
) -> dict[str, Any]:
    """Perform one descriptor-pinned, exact, label-free grid audit."""

    validate_plan(plan)
    assert_analysis_fence(fence, plan=plan)
    root = _absolute(run_root)
    if root != fence.run_root:
        raise HemiQGridError("audit root differs from the fenced run root")
    root_descriptor = os.dup(fence.root_descriptor)
    directories: dict[str, int] = {}
    fingerprints: dict[str, tuple[int, ...]] = {}
    try:
        root_fingerprint = _stat_fingerprint(os.fstat(root_descriptor))
        root_names = sorted(os.listdir(root_descriptor))
        if set(root_names) != set(RUN_DIRECTORIES) | {"plan.json"}:
            raise HemiQGridError("run root contains an unexpected artifact")
        for name in sorted(RUN_DIRECTORIES):
            descriptor = os.open(
                name,
                _directory_flags(),
                dir_fd=root_descriptor,
            )
            directories[name] = descriptor
            held = os.fstat(descriptor)
            path_status = os.stat(
                name,
                dir_fd=root_descriptor,
                follow_symlinks=False,
            )
            fingerprint = _stat_fingerprint(held)
            if (
                not stat.S_ISDIR(held.st_mode)
                or _stat_fingerprint(path_status) != fingerprint
            ):
                raise HemiQGridError(
                    f"run-root directory is detached: {name}"
                )
            fingerprints[name] = fingerprint

        if (
            int(os.fstat(directories[".authority"]).st_dev),
            int(os.fstat(directories[".authority"]).st_ino),
        ) != (fence.authority_st_dev, fence.authority_st_ino):
            raise HemiQGridError(
                "audit authority descriptor differs from the fence"
            )

        plan_payload, plan_status = _read_unique_regular_at(
            root_descriptor,
            "plan.json",
            source=str(root / "plan.json"),
        )
        if (
            stat.S_IMODE(plan_status.st_mode) != 0o400
            or _strict_json_bytes(
                plan_payload,
                source=str(root / "plan.json"),
            )
            != dict(plan)
        ):
            raise HemiQGridError(
                "run-root plan differs from analysis plan"
            )

        authority_names = set(os.listdir(directories[".authority"]))
        if authority_names != {
            "coordination.lock",
            fence.marker_path.name,
        }:
            raise HemiQGridError(
                "authority directory is not quiescent and exact"
            )
        if any(
            os.listdir(directories[name])
            for name in ("claims", "failures", "quarantine", "staging")
        ):
            raise HemiQGridError(
                "claims/failures/quarantine/staging are not quiescent"
            )
        assert_analysis_fence(fence, plan=plan)

        preflight_descriptor = directories["preflight"]
        preflight_names = sorted(os.listdir(preflight_descriptor))
        if not preflight_names:
            raise HemiQGridError("no CUDA preflight receipt exists")
        preflights: dict[str, dict[str, Any]] = {}
        preflight_payloads: dict[str, bytes] = {}
        for name in preflight_names:
            if not name.endswith(".json"):
                raise HemiQGridError(
                    "preflight directory contains an invalid leaf"
                )
            gpu_uuid = name.removesuffix(".json")
            payload, status = _read_unique_regular_at(
                preflight_descriptor,
                name,
                source=str(root / "preflight" / name),
            )
            if stat.S_IMODE(status.st_mode) != 0o400:
                raise HemiQGridError(
                    "preflight directory contains a writable leaf"
                )
            value = _strict_json_bytes(
                payload,
                source=str(root / "preflight" / name),
            )
            _validate_preflight_value(
                value,
                plan=plan,
                project_root=project_root,
                run_root=root,
                gpu_uuid=gpu_uuid,
            )
            preflights[gpu_uuid] = value
            preflight_payloads[gpu_uuid] = payload
            assert_analysis_fence(fence, plan=plan)

        records_descriptor = directories["records"]
        record_names = sorted(os.listdir(records_descriptor))
        expected_ids = {job.job_id for job in expected_jobs()}
        if set(record_names) != expected_ids:
            missing = sorted(expected_ids - set(record_names))
            extra = sorted(set(record_names) - expected_ids)
            raise HemiQGridError(
                f"record roster is not exact; missing={missing}, extra={extra}"
            )
        record_entries = {
            name: os.stat(
                name,
                dir_fd=records_descriptor,
                follow_symlinks=False,
            )
            for name in record_names
        }
        roster: list[dict[str, Any]] = [
            {
                "relative_path": "plan.json",
                "sha256": _sha256_bytes(plan_payload),
                "size_bytes": len(plan_payload),
            }
        ]
        used_gpus: set[str] = set()
        for job in expected_jobs():
            assert_analysis_fence(fence, plan=plan)
            status = record_entries[job.job_id]
            if (
                not stat.S_ISDIR(status.st_mode)
                or stat.S_IMODE(status.st_mode) != 0o555
            ):
                raise HemiQGridError(
                    f"result directory is unsafe: {job.job_id}"
                )
            validated = validate_completion(
                plan=plan,
                project_root=project_root,
                run_root=root,
                job=job,
                fence=fence,
            )
            used_gpus.add(validated["completion"]["gpu_uuid"])
            roster.extend(copy.deepcopy(validated["artifact_roster"]))
            assert_analysis_fence(fence, plan=plan)
        if not used_gpus or not used_gpus.issubset(preflights):
            raise HemiQGridError(
                "result GPU roster lacks exact preflight coverage"
            )
        for gpu_uuid in sorted(preflights):
            payload = preflight_payloads[gpu_uuid]
            roster.append(
                {
                    "relative_path": (
                        f"preflight/{gpu_uuid}.json"
                    ),
                    "sha256": _sha256_bytes(payload),
                    "size_bytes": len(payload),
                }
            )

        expected_final_names = {
            ".authority": {
                "coordination.lock",
                fence.marker_path.name,
            },
            "claims": set(),
            "failures": set(),
            "preflight": set(preflight_names),
            "quarantine": set(),
            "records": expected_ids,
            "staging": set(),
        }
        if (
            _stat_fingerprint(os.fstat(root_descriptor))
            != root_fingerprint
            or sorted(os.listdir(root_descriptor)) != root_names
        ):
            raise HemiQGridError(
                "run root changed during descriptor-bound audit"
            )
        for name, descriptor in directories.items():
            if (
                _stat_fingerprint(os.fstat(descriptor))
                != fingerprints[name]
                or _stat_fingerprint(
                    os.stat(
                        name,
                        dir_fd=root_descriptor,
                        follow_symlinks=False,
                    )
                )
                != fingerprints[name]
                or set(os.listdir(descriptor))
                != expected_final_names[name]
            ):
                raise HemiQGridError(
                    f"run-root directory changed during audit: {name}"
                )
        final_plan, _ = _read_unique_regular_at(
            root_descriptor,
            "plan.json",
            source=str(root / "plan.json"),
        )
        if final_plan != plan_payload:
            raise HemiQGridError("run-root plan changed during audit")
        assert_analysis_fence(fence, plan=plan)
        return {
            "audited_at": _utc_now(),
            "completed_jobs": EXPECTED_JOBS,
            "evidence_scope": plan["evidence_scope"],
            "plan_sha256": plan["plan_sha256"],
            "preflight_gpu_uuids": sorted(preflights),
            "roster": roster,
            "roster_sha256": _sha256_bytes(_canonical_bytes(roster)),
            "schema": AUDIT_SCHEMA,
            "status": "pass",
        }
    finally:
        for descriptor in directories.values():
            os.close(descriptor)
        os.close(root_descriptor)


def _worker_jobs(plan: Mapping[str, Any]) -> Iterator[Job]:
    validate_plan(plan)
    for value in plan["jobs"]:
        yield _job_from_mapping(value)


def run_worker(
    *,
    project_root: Path,
    cache_root: Path,
    run_root: Path,
    gpu_uuid: str,
    device: str = "cuda:0",
    max_jobs: int | None = None,
) -> dict[str, int]:
    """Run resumable formal jobs on one explicitly isolated physical GPU."""

    plan = load_plan(run_root)
    if max_jobs is not None and plan["publication_mode"] == FORMAL_PUBLICATION_MODE:
        raise HemiQGridError("formal workers may not truncate the exact grid")
    if max_jobs is not None and (
        not _is_exact_int(max_jobs, minimum=1)
    ):
        raise HemiQGridError("max_jobs must be a positive non-boolean integer")
    verify_source_identity(plan, project_root=project_root)
    verify_uv_identity(plan, project_root=project_root)
    verify_environment_identity(plan)
    lease = project_gpu_leases.acquire_gpu_lease(
        project_root=_absolute(project_root),
        run_root=_absolute(run_root),
        plan_sha256=str(plan["plan_sha256"]),
        gpu_uuid=gpu_uuid,
        track_scope=TRACK_SCOPE,
    )
    completed = 0
    skipped = 0
    try:
        run_preflight(
            plan=plan,
            project_root=project_root,
            cache_root=cache_root,
            run_root=run_root,
            gpu_uuid=gpu_uuid,
            device=device,
            lease=lease,
        )
        for job in _worker_jobs(plan):
            if max_jobs is not None and completed >= max_jobs:
                break
            try:
                claim = acquire_claim(
                    plan=plan,
                    project_root=project_root,
                    run_root=run_root,
                    job=job,
                    gpu_uuid=gpu_uuid,
                    lease=lease,
                )
            except ClaimUnavailable:
                skipped += 1
                continue
            if claim is None:
                skipped += 1
                continue
            try:
                record, rows, probabilities = execute_job(
                    plan=plan,
                    project_root=project_root,
                    cache_root=cache_root,
                    job=job,
                    gpu_uuid=gpu_uuid,
                    device=device,
                )
                commit_job_output(
                    plan=plan,
                    project_root=project_root,
                    run_root=run_root,
                    job=job,
                    gpu_uuid=gpu_uuid,
                    lease=lease,
                    claim=claim,
                    record=record,
                    test_rows=rows,
                    probabilities=probabilities,
                )
                completed += 1
            except Exception as error:
                try:
                    _record_failure(
                        plan=plan,
                        project_root=project_root,
                        run_root=run_root,
                        job=job,
                        gpu_uuid=gpu_uuid,
                        lease=lease,
                        error=error,
                    )
                finally:
                    release_claim(
                        claim,
                        plan=plan,
                        project_root=project_root,
                        run_root=run_root,
                        gpu_uuid=gpu_uuid,
                        lease=lease,
                    )
                raise
            else:
                release_claim(
                    claim,
                    plan=plan,
                    project_root=project_root,
                    run_root=run_root,
                    gpu_uuid=gpu_uuid,
                    lease=lease,
                )
    finally:
        project_gpu_leases.release_gpu_lease(lease)
    return {"completed": completed, "skipped": skipped}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-plan")
    build.add_argument("--project-root", type=Path, required=True)
    build.add_argument("--cache-root", type=Path, required=True)
    build.add_argument("--run-root", type=Path, required=True)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--project-root", type=Path, required=True)
    worker.add_argument("--cache-root", type=Path, required=True)
    worker.add_argument("--run-root", type=Path, required=True)
    worker.add_argument("--gpu-uuid", required=True)
    worker.add_argument("--device", default="cuda:0")
    audit = subparsers.add_parser("audit")
    audit.add_argument("--project-root", type=Path, required=True)
    audit.add_argument("--run-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "build-plan":
        plan = build_plan(
            project_root=arguments.project_root,
            cache_root=arguments.cache_root,
        )
        published = write_or_validate_plan(arguments.run_root, plan)
        print(
            json.dumps(
                {
                    "n_jobs": published["n_jobs"],
                    "plan_sha256": published["plan_sha256"],
                    "status": "plan_published_no_jobs_launched",
                },
                sort_keys=True,
            )
        )
        return 0
    if arguments.command == "worker":
        result = run_worker(
            project_root=arguments.project_root,
            cache_root=arguments.cache_root,
            run_root=arguments.run_root,
            gpu_uuid=arguments.gpu_uuid,
            device=arguments.device,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    if arguments.command == "audit":
        plan = load_plan(arguments.run_root)
        with analysis_fence(run_root=arguments.run_root, plan=plan) as fence:
            result = audit_grid(
                plan=plan,
                project_root=arguments.project_root,
                run_root=arguments.run_root,
                fence=fence,
            )
        print(json.dumps(result, sort_keys=True))
        return 0
    raise AssertionError("unreachable command")


if __name__ == "__main__":
    raise SystemExit(main())
