"""Atomic, score-blind execution of the separate author-recipe reference grid.

This runner is intentionally not part of :mod:`benchmark.full_grid`.  It applies
the released optimization recipes implemented in
:mod:`benchmark.reference_training` to the same five opened v2 caches and split
identities, then publishes only row identities and probabilities.  Scores are
joined to labels later by :mod:`benchmark.author_recipe_analysis`.

The two reference procedures have different selection/refit semantics:

* TCFormer has a prespecified dataset-specific fixed horizon and no
  outcome-selected epoch.  The final source fit therefore uses
  train+validation rows and that frozen horizon.
* FBCNet performs its released validation-inaccuracy selection on train versus
  validation, restores the best model and Adam state, then runs the prescribed
  train+validation stage-two fit.  The original validation subset is used only
  by the released stage-two stopping condition.

Neither fitter accepts test arrays.  The test tensor is passed exactly once to
``predict_probabilities`` after fitting.  The immutable plan binds the cache
and split vectors, complete UV runtime, in-repository source closure, official
reference provenance, and the exact hash-pinned TCFormer runtime snapshot.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import errno
import fcntl
import hashlib
import importlib
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
import tomllib
import uuid
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np

from . import full_grid
from .reference_training import (
    BRAIDECODE_VERSION,
    FBCNET_SOURCE_COMMIT,
    FBCNetReferenceConfig,
    TCFormerReferenceConfig,
    fbcnet_reference_metadata,
    tcformer_reference_config,
    tcformer_reference_metadata,
)
from .tcformer_source import verify_tcformer_source


PLAN_SCHEMA = "eeg-mi-author-recipe-grid-plan-v1"
RECORD_SCHEMA = "eeg-mi-author-recipe-score-blind-record-v1"
COMPLETION_SCHEMA = "eeg-mi-author-recipe-completion-v1"
CLAIM_SCHEMA = "eeg-mi-author-recipe-claim-v1"
FAILURE_SCHEMA = "eeg-mi-author-recipe-failure-v1"
AUDIT_SCHEMA = "eeg-mi-author-recipe-audit-v1"
ANALYSIS_CONTRACT_SCHEMA = "eeg-mi-author-recipe-analysis-contract-v1"
PUBLICATION_FENCE_SCHEMA = "eeg-mi-author-recipe-publication-fence-v1"

TRACK = "author_recipe_adapted_reference_separate"
TRACK_DESCRIPTION = "author-recipe adapted"
REFERENCES: tuple[str, ...] = ("reference.tcformer", "reference.fbcnet")
OPENED_DATASETS = full_grid.OPENED_DATASETS
DATASET_FOLDS = full_grid.DATASET_FOLDS
FORMAL_SEEDS = full_grid.FORMAL_SEEDS
MAX_GPU_WORKERS = full_grid.MAX_GPU_WORKERS
REQUIRED_CUBLAS_WORKSPACE_CONFIG = full_grid.REQUIRED_CUBLAS_WORKSPACE_CONFIG
DEFAULT_MIN_FREE_GIB = full_grid.DEFAULT_MIN_FREE_GIB
THREAD_ENVIRONMENT_VARIABLES = full_grid.THREAD_ENVIRONMENT_VARIABLES
EXPECTED_JOBS_PER_REFERENCE = 2_240
EXPECTED_JOB_COUNT = 4_480
DEFAULT_EXECUTOR = "benchmark.author_recipe_grid:execute_author_recipe_job"
PUBLICATION_GATE_FILENAME = ".author-recipe-publication.gate"
PUBLICATION_FENCE_FILENAME = ".author-recipe-publication.json"
PUBLICATION_GATE_BYTES = b"eeg-mi-author-recipe-publication-gate-v1\n"
FINAL_FILENAMES = frozenset({"record.json", "predictions.npz", "completion.json"})
HEX_64_RE = full_grid.HEX_64_RE

SOURCE_FILES: tuple[str, ...] = (
    "src/benchmark/__init__.py",
    "src/benchmark/author_recipe_grid.py",
    "src/benchmark/author_recipe_analysis.py",
    "src/benchmark/model_registry.py",
    "src/benchmark/reference_training.py",
    "src/benchmark/baselines.py",
    "src/benchmark/tcformer_source.py",
    "src/benchmark/models.py",
    "src/benchmark/training.py",
    "src/benchmark/data.py",
    "src/benchmark/config.py",
    # Dataset contracts, environment capture, and cooperative resource probes
    # are reused from this audited infrastructure module.
    "src/benchmark/full_grid.py",
    "pyproject.toml",
    "uv.lock",
)
REQUIRED_DIRECT_DEPENDENCIES = frozenset(
    {
        "torch",
        "braindecode",
        "einops",
        "mne",
        "numpy",
        "scikit-learn",
        "scipy",
    }
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
        "outcome",
        "outcomes",
        "truth",
        "ground_truth",
        "y_true",
        "y_test",
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


class AuthorRecipeError(RuntimeError):
    """Base error for an author-recipe contract violation."""


class ClaimUnavailable(AuthorRecipeError):
    """Raised when an atomic job cannot be claimed."""


class GPUUnavailable(AuthorRecipeError):
    """Raised when the cooperative GPU guard rejects a new claim."""


class DiskUnavailable(AuthorRecipeError):
    """Raised when the shared filesystem is below its low-water mark."""


@dataclass(frozen=True, order=True)
class Job:
    dataset: str
    reference: str
    subject: int
    fold: int
    seed: int

    def identity(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "reference": self.reference,
            "subject": self.subject,
            "fold": self.fold,
            "seed": self.seed,
        }

    @property
    def job_id(self) -> str:
        digest = hashlib.sha256(_canonical_bytes(self.identity())).hexdigest()
        return f"author-job-{digest[:24]}"


@dataclass
class ClaimPublicationState:
    """Process-local ownership of a record published by this claim."""

    record_identity: tuple[int, int] | None = None


@dataclass(frozen=True)
class Claim:
    job: Job
    run_root: Path
    path: Path
    path_identity: tuple[int, int]
    plan_sha256: str
    nonce: str
    owner: Mapping[str, Any]
    resource_guard: Mapping[str, Any]
    publication_gate_descriptor: int = field(repr=False, compare=False)
    publication_state: ClaimPublicationState = field(
        default_factory=ClaimPublicationState,
        repr=False,
        compare=False,
    )


@dataclass(frozen=True)
class PublicationFence:
    run_root: Path
    path: Path
    path_identity: tuple[int, int]
    plan_sha256: str
    nonce: str
    owner: Mapping[str, Any]
    destination: str
    gate_descriptor: int = field(repr=False, compare=False)


_SNAPSHOT_STABLE_FIELDS: tuple[str, ...] = (
    "st_dev",
    "st_ino",
    "st_mode",
    "st_nlink",
    "st_size",
    "st_mtime_ns",
    "st_ctime_ns",
)


@dataclass(frozen=True)
class CompletionSnapshot:
    """One coherent, descriptor-held completion package."""

    record: Mapping[str, Any]
    completion: Mapping[str, Any]
    rows: np.ndarray = field(repr=False)
    probabilities: np.ndarray = field(repr=False)
    artifact_bytes: Mapping[str, bytes] = field(repr=False)
    artifact_sha256: Mapping[str, str]
    directory_fingerprint: tuple[int, ...]
    entry_fingerprints: tuple[tuple[str, tuple[int, ...]], ...]


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


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(_read_regular_bytes(path))


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _rows_sha256(values: np.ndarray | Sequence[int]) -> str:
    return full_grid._rows_sha256(values)


def _subject_key(dataset: str, subject: int) -> str:
    return f"{dataset}:s{int(subject):03d}"


def _split_key(dataset: str, subject: int, fold: int) -> str:
    return f"{dataset}:s{int(subject):03d}:f{int(fold):02d}"


def _absolute_path(path: Path | str) -> Path:
    """Return a lexical absolute path without resolving aliases."""

    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def _stat_fingerprint(observed: os.stat_result) -> tuple[int, ...]:
    return tuple(
        int(getattr(observed, name)) for name in _SNAPSHOT_STABLE_FIELDS
    )


def _open_directory_anchored(path: Path | str) -> int:
    """Open an absolute directory one no-follow component at a time."""

    candidate = _absolute_path(path)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = os.open(candidate.anchor, flags)
    try:
        for component in candidate.parts[1:]:
            child = os.open(component, flags, dir_fd=descriptor)
            opened = os.fstat(child)
            if not stat.S_ISDIR(opened.st_mode):
                os.close(child)
                raise AuthorRecipeError(
                    f"path component is not a directory: {candidate}"
                )
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _anchored_lstat(path: Path | str) -> os.stat_result:
    candidate = _absolute_path(path)
    parent_descriptor = _open_directory_anchored(candidate.parent)
    try:
        return os.stat(
            candidate.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    finally:
        os.close(parent_descriptor)


def _path_binds_identity(
    path: Path | str, expected_identity: tuple[int, int]
) -> bool:
    try:
        observed = _anchored_lstat(path)
    except (FileNotFoundError, AuthorRecipeError, OSError):
        return False
    return (int(observed.st_dev), int(observed.st_ino)) == tuple(
        map(int, expected_identity)
    )


def _require_contained(path: Path, root: Path, context: str) -> tuple[Path, Path]:
    candidate = _absolute_path(path)
    boundary = _absolute_path(root)
    if candidate != boundary and boundary not in candidate.parents:
        raise AuthorRecipeError(f"{context} escapes its trusted root")
    return candidate, boundary


def _path_components(path: Path) -> Iterator[Path]:
    candidate = _absolute_path(path)
    current = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        current = current / part
        yield current


def _require_safe_ancestors(
    path: Path,
    *,
    root: Path | None = None,
    include_leaf: bool = False,
    allow_missing_leaf: bool = True,
    context: str = "path",
) -> None:
    """Reject symlink, non-directory, and special ancestors using ``lstat``."""

    candidate = _absolute_path(path)
    boundary: Path | None = None
    if root is not None:
        candidate, boundary = _require_contained(candidate, root, context)
    stop = candidate if include_leaf else candidate.parent
    for component in _path_components(stop):
        try:
            observed = os.lstat(component)
        except FileNotFoundError:
            continue
        is_leaf = component == candidate
        if stat.S_ISLNK(observed.st_mode):
            raise AuthorRecipeError(f"{context} contains a symlink: {component}")
        if is_leaf and include_leaf:
            if not stat.S_ISREG(observed.st_mode) and not stat.S_ISDIR(
                observed.st_mode
            ):
                raise AuthorRecipeError(
                    f"{context} is a special filesystem object: {component}"
                )
        elif not stat.S_ISDIR(observed.st_mode):
            raise AuthorRecipeError(
                f"{context} has a non-directory ancestor: {component}"
            )
    if include_leaf and not allow_missing_leaf and not os.path.lexists(candidate):
        raise FileNotFoundError(candidate)


def _safe_mkdir(
    path: Path,
    *,
    root: Path | None = None,
    mode: int = 0o755,
    exist_ok: bool = True,
) -> Path:
    candidate = _absolute_path(path)
    boundary = _absolute_path(root) if root is not None else None
    if boundary is not None:
        _require_contained(candidate, boundary, "directory")
    _require_safe_ancestors(
        candidate, root=boundary, include_leaf=False, context="directory"
    )
    existed = os.path.lexists(candidate)
    if existed and not exist_ok:
        raise FileExistsError(candidate)
    for component in _path_components(candidate):
        try:
            observed = os.lstat(component)
        except FileNotFoundError:
            try:
                os.mkdir(component, mode)
            except FileExistsError:
                if component == candidate and not exist_ok:
                    raise
            observed = os.lstat(component)
        if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
            raise AuthorRecipeError(
                f"directory path is aliased or non-directory: {component}"
            )
    return candidate


def _regular_identity(
    path: Path,
    *,
    root: Path | None = None,
    readonly: bool = False,
    single_link: bool = True,
    context: str = "file",
) -> tuple[os.stat_result, int]:
    candidate = _absolute_path(path)
    _require_safe_ancestors(candidate, root=root, context=context)
    try:
        before = os.lstat(candidate)
    except FileNotFoundError:
        raise
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or (single_link and before.st_nlink != 1)
        or (readonly and before.st_mode & 0o222)
    ):
        raise AuthorRecipeError(
            f"{context} must be a regular, unaliased"
            + (", read-only" if readonly else "")
            + f" file: {candidate}"
        )
    parent_descriptor = _open_directory_anchored(candidate.parent)
    try:
        parent_opened = os.fstat(parent_descriptor)
        parent_visible = _anchored_lstat(candidate.parent)
        if _stat_fingerprint(parent_opened) != _stat_fingerprint(parent_visible):
            raise AuthorRecipeError(
                f"{context} parent changed while opening: {candidate.parent}"
            )
        flags = (
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        descriptor = os.open(candidate.name, flags, dir_fd=parent_descriptor)
    except BaseException:
        os.close(parent_descriptor)
        raise
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or _stat_fingerprint(opened) != _stat_fingerprint(before)
            or (single_link and opened.st_nlink != 1)
            or (readonly and opened.st_mode & 0o222)
        ):
            raise AuthorRecipeError(f"{context} changed while opening: {candidate}")
    except BaseException:
        os.close(descriptor)
        os.close(parent_descriptor)
        raise
    os.close(parent_descriptor)
    return before, descriptor


def _read_regular_bytes(
    path: Path,
    *,
    root: Path | None = None,
    readonly: bool = False,
    context: str = "file",
) -> bytes:
    payload, _ = _read_regular_snapshot(
        path,
        root=root,
        readonly=readonly,
        context=context,
    )
    return payload


def _read_regular_snapshot(
    path: Path,
    *,
    root: Path | None = None,
    readonly: bool = False,
    context: str = "file",
    expected_identity: tuple[int, int] | None = None,
) -> tuple[bytes, tuple[int, ...]]:
    """Read one exact unique inode and rebind its full canonical fingerprint."""

    candidate = _absolute_path(path)
    before, descriptor = _regular_identity(
        candidate,
        root=root,
        readonly=readonly,
        context=context,
    )
    fingerprint = _stat_fingerprint(before)
    identity = (int(before.st_dev), int(before.st_ino))
    if expected_identity is not None and identity != tuple(
        map(int, expected_identity)
    ):
        os.close(descriptor)
        raise AuthorRecipeError(f"{context} inode differs from its authority")
    try:
        chunks: list[bytes] = []
        offset = 0
        while offset < int(before.st_size):
            chunk = os.pread(
                descriptor,
                min(1024 * 1024, int(before.st_size) - offset),
                offset,
            )
            if not chunk:
                raise AuthorRecipeError(
                    f"{context} ended before its declared size: {candidate}"
                )
            chunks.append(chunk)
            offset += len(chunk)
        if os.pread(descriptor, 1, offset):
            raise AuthorRecipeError(
                f"{context} grew while being read: {candidate}"
            )
        after_fd = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        after_path = _anchored_lstat(candidate)
    except FileNotFoundError as error:
        raise AuthorRecipeError(f"{context} vanished while reading: {candidate}") from error
    if (
        _stat_fingerprint(after_fd) != fingerprint
        or _stat_fingerprint(after_path) != fingerprint
        or stat.S_ISLNK(after_path.st_mode)
        or not stat.S_ISREG(after_path.st_mode)
        or after_path.st_nlink != 1
        or (readonly and after_path.st_mode & 0o222)
    ):
        raise AuthorRecipeError(f"{context} changed while reading: {candidate}")
    return b"".join(chunks), fingerprint


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _decode_json_object(payload: bytes, *, context: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise AuthorRecipeError(f"strict JSON load failed for {context}: {error}") from error
    if not isinstance(value, dict):
        raise AuthorRecipeError(f"JSON artifact is not an object: {context}")
    return value


def strict_load(
    path: Path,
    *,
    root: Path | None = None,
    readonly: bool = False,
) -> dict[str, Any]:
    payload = _read_regular_bytes(
        path,
        root=root,
        readonly=readonly,
        context="JSON artifact",
    )
    return _decode_json_object(payload, context=str(path))


def _fsync_directory(path: Path, *, root: Path | None = None) -> None:
    candidate = _absolute_path(path)
    _require_safe_ancestors(
        candidate,
        root=root,
        include_leaf=True,
        allow_missing_leaf=False,
        context="directory fsync",
    )
    observed = os.lstat(candidate)
    if not stat.S_ISDIR(observed.st_mode) or stat.S_ISLNK(observed.st_mode):
        raise AuthorRecipeError(f"cannot fsync non-directory {candidate}")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    descriptor = os.open(candidate, flags)
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (observed.st_dev, observed.st_ino):
            raise AuthorRecipeError(f"directory changed while opening: {candidate}")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_bytes_exclusive(
    path: Path,
    payload: bytes,
    *,
    mode: int = 0o444,
    root: Path | None = None,
) -> None:
    candidate = _absolute_path(path)
    boundary = _absolute_path(root) if root is not None else None
    if boundary is not None:
        _require_contained(candidate, boundary, "exclusive output")
    parent = _safe_mkdir(candidate.parent, root=boundary)
    parent_before = os.lstat(parent)
    parent_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    parent_descriptor = os.open(parent, parent_flags)
    parent_opened = os.fstat(parent_descriptor)
    if (parent_opened.st_dev, parent_opened.st_ino) != (
        parent_before.st_dev,
        parent_before.st_ino,
    ):
        os.close(parent_descriptor)
        raise AuthorRecipeError(f"exclusive output parent changed: {parent}")
    try:
        descriptor = os.open(
            candidate.name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent_descriptor,
        )
    except BaseException:
        os.close(parent_descriptor)
        raise
    created_identity: tuple[int, int] | None = None
    try:
        created = os.fstat(descriptor)
        if not stat.S_ISREG(created.st_mode) or created.st_nlink != 1:
            raise AuthorRecipeError(f"exclusive output is not regular: {candidate}")
        created_identity = (created.st_dev, created.st_ino)
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.fchmod(descriptor, mode)
    finally:
        os.close(descriptor)
        os.close(parent_descriptor)
    parent_after = os.lstat(parent)
    final = os.lstat(candidate)
    if (
        (parent_after.st_dev, parent_after.st_ino)
        != (parent_before.st_dev, parent_before.st_ino)
        or created_identity != (final.st_dev, final.st_ino)
        or not stat.S_ISREG(final.st_mode)
        or stat.S_ISLNK(final.st_mode)
        or final.st_nlink != 1
    ):
        raise AuthorRecipeError(f"exclusive output path changed: {candidate}")
    _fsync_directory(parent, root=boundary)


def _write_json_exclusive(
    path: Path,
    value: Mapping[str, Any],
    *,
    root: Path | None = None,
) -> None:
    _write_bytes_exclusive(
        path,
        _canonical_bytes(dict(value)) + b"\n",
        root=root,
    )


def _rename_leaf_nofollow(
    source: Path,
    destination: Path,
    *,
    root: Path,
    expected_source_identity: tuple[int, int] | None = None,
) -> tuple[int, int]:
    """Atomically rename one exact leaf and never replace a destination."""

    source, boundary = _require_contained(source, root, "rename source")
    destination, _ = _require_contained(
        destination, boundary, "rename destination"
    )
    if source.name in {"", ".", ".."} or destination.name in {"", ".", ".."}:
        raise AuthorRecipeError("rename requires exact leaf paths")
    _require_safe_ancestors(
        source, root=boundary, context="rename source"
    )
    _require_safe_ancestors(
        destination, root=boundary, context="rename destination"
    )
    source_parent_fd = _open_directory_anchored(source.parent)
    try:
        destination_parent_fd = _open_directory_anchored(destination.parent)
    except BaseException:
        os.close(source_parent_fd)
        raise
    try:
        source_parent_stat = os.fstat(source_parent_fd)
        destination_parent_stat = os.fstat(destination_parent_fd)
        source_parent_identity = (
            int(source_parent_stat.st_dev),
            int(source_parent_stat.st_ino),
        )
        destination_parent_identity = (
            int(destination_parent_stat.st_dev),
            int(destination_parent_stat.st_ino),
        )
        source_stat = os.stat(
            source.name,
            dir_fd=source_parent_fd,
            follow_symlinks=False,
        )
        source_identity = (
            int(source_stat.st_dev),
            int(source_stat.st_ino),
        )
        if (
            stat.S_ISLNK(source_stat.st_mode)
            or not (
                stat.S_ISREG(source_stat.st_mode)
                or stat.S_ISDIR(source_stat.st_mode)
            )
            or (
                stat.S_ISREG(source_stat.st_mode)
                and int(source_stat.st_nlink) != 1
            )
            or (
                expected_source_identity is not None
                and source_identity
                != tuple(map(int, expected_source_identity))
            )
        ):
            raise AuthorRecipeError(
                "rename source is aliased, special, or changed"
            )
        libc = ctypes.CDLL(None, use_errno=True)
        source_name = os.fsencode(source.name)
        destination_name = os.fsencode(destination.name)
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
                source_parent_fd,
                source_name,
                destination_parent_fd,
                destination_name,
                0x00000004,  # RENAME_EXCL
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
                source_parent_fd,
                source_name,
                destination_parent_fd,
                destination_name,
                0x00000001,  # RENAME_NOREPLACE
            )
        else:
            raise AuthorRecipeError(
                "atomic no-replace rename is unavailable on this platform"
            )
        if result != 0:
            error_number = ctypes.get_errno()
            if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
                raise FileExistsError(
                    error_number, os.strerror(error_number), destination
                )
            raise OSError(
                error_number,
                os.strerror(error_number),
                f"{source} -> {destination}",
            )
        moved = os.stat(
            destination.name,
            dir_fd=destination_parent_fd,
            follow_symlinks=False,
        )
        if source_identity != (int(moved.st_dev), int(moved.st_ino)):
            raise AuthorRecipeError("renamed leaf inode changed")
        os.fsync(source_parent_fd)
        if destination_parent_fd != source_parent_fd:
            os.fsync(destination_parent_fd)
        if (
            (
                int(os.fstat(source_parent_fd).st_dev),
                int(os.fstat(source_parent_fd).st_ino),
            )
            != source_parent_identity
            or (
                int(os.fstat(destination_parent_fd).st_dev),
                int(os.fstat(destination_parent_fd).st_ino),
            )
            != destination_parent_identity
        ):
            raise AuthorRecipeError("rename parent descriptor changed")
    finally:
        os.close(destination_parent_fd)
        os.close(source_parent_fd)
    try:
        visible_parent = _anchored_lstat(destination.parent)
        visible = _anchored_lstat(destination)
    except FileNotFoundError as error:
        raise AuthorRecipeError(
            "renamed leaf is no longer visible at its contained destination"
        ) from error
    if (
        (int(visible_parent.st_dev), int(visible_parent.st_ino))
        != destination_parent_identity
        or (int(visible.st_dev), int(visible.st_ino)) != source_identity
    ):
        raise AuthorRecipeError("rename destination path or parent was swapped")
    return source_identity


def _seal_directory_read_only(
    path: Path,
    *,
    root: Path,
    expected_identity: tuple[int, int],
) -> tuple[int, ...]:
    """Durably seal one exact directory to mode 0555."""

    candidate, _ = _require_contained(path, root, "directory seal")
    descriptor = _open_directory_anchored(candidate)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(before.st_mode)
            or (int(before.st_dev), int(before.st_ino))
            != tuple(map(int, expected_identity))
        ):
            raise AuthorRecipeError("directory seal authority changed")
        os.fchmod(descriptor, 0o555)
        os.fsync(descriptor)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    visible = _anchored_lstat(candidate)
    if (
        _stat_fingerprint(visible) != _stat_fingerprint(after)
        or stat.S_IMODE(after.st_mode) != 0o555
        or (int(after.st_dev), int(after.st_ino))
        != tuple(map(int, expected_identity))
    ):
        raise AuthorRecipeError("directory seal did not bind the exact inode")
    _fsync_directory(candidate.parent, root=root)
    return _stat_fingerprint(after)


def _canonical_requirement_name(requirement: str) -> str:
    name = requirement.strip().lower()
    for token in ("[", " ", "<", ">", "=", "!", "~", ";"):
        name = name.split(token, 1)[0]
    return name.replace("_", "-").replace(".", "-")


_EXACT_REQUIREMENT_RE = re.compile(
    r"^\s*([A-Za-z0-9_.-]+)(?:\[[^\]]+\])?\s*==\s*"
    r"([A-Za-z0-9.!+_-]+)(?:\s*;.*)?$"
)


def _run_uv_integrity_command(
    command: Sequence[str], *, project_root: Path
) -> str:
    try:
        result = subprocess.run(
            list(command),
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as error:
        rendered = " ".join(command)
        raise AuthorRecipeError(
            f"UV integrity command failed closed: {rendered}"
        ) from error
    return "\n".join(
        value.strip()
        for value in (result.stdout, result.stderr)
        if value.strip()
    )


def _validate_uv_project_contract(
    project_root: Path, *, check_runtime: bool = True
) -> dict[str, Any]:
    project_root = _absolute_path(project_root)
    _require_safe_ancestors(
        project_root,
        include_leaf=True,
        allow_missing_leaf=False,
        context="UV project root",
    )
    pyproject_path = project_root / "pyproject.toml"
    lock_path = project_root / "uv.lock"
    try:
        pyproject_bytes = _read_regular_bytes(
            pyproject_path, root=project_root, context="pyproject.toml"
        )
        lock_bytes = _read_regular_bytes(
            lock_path, root=project_root, context="uv.lock"
        )
    except FileNotFoundError:
        raise AuthorRecipeError(
            "formal author-recipe planning requires pyproject.toml and uv.lock"
        )
    try:
        pyproject = tomllib.loads(pyproject_bytes.decode("utf-8"))
        lock = tomllib.loads(lock_bytes.decode("utf-8"))
        dependencies = pyproject["project"]["dependencies"]
        packages = lock["package"]
    except (
        KeyError,
        TypeError,
        UnicodeDecodeError,
        tomllib.TOMLDecodeError,
    ) as error:
        raise AuthorRecipeError(f"UV project metadata cannot be parsed: {error}") from error
    if not isinstance(dependencies, list) or not all(
        isinstance(value, str) for value in dependencies
    ):
        raise AuthorRecipeError("project.dependencies must be a string list")
    if not isinstance(packages, list):
        raise AuthorRecipeError("uv.lock package table must be an array")
    pinned_direct: dict[str, str] = {}
    for requirement in dependencies:
        match = _EXACT_REQUIREMENT_RE.fullmatch(requirement)
        name = _canonical_requirement_name(requirement)
        if name not in REQUIRED_DIRECT_DEPENDENCIES:
            continue
        if match is None:
            raise AuthorRecipeError(
                f"author-recipe dependency {name} must use one exact == pin"
            )
        normalized = _canonical_requirement_name(match.group(1))
        if normalized in pinned_direct:
            raise AuthorRecipeError(
                f"author-recipe dependency {normalized} is declared more than once"
            )
        pinned_direct[normalized] = match.group(2)
    locked_versions: dict[str, set[str]] = {}
    for value in packages:
        if not isinstance(value, Mapping) or "name" not in value or "version" not in value:
            continue
        name = _canonical_requirement_name(str(value["name"]))
        locked_versions.setdefault(name, set()).add(str(value["version"]))
    missing_direct = sorted(REQUIRED_DIRECT_DEPENDENCIES - set(pinned_direct))
    missing_locked = sorted(REQUIRED_DIRECT_DEPENDENCIES - set(locked_versions))
    version_drift = {
        name: {
            "declared": version,
            "locked": sorted(locked_versions.get(name, set())),
        }
        for name, version in sorted(pinned_direct.items())
        if locked_versions.get(name) != {version}
    }
    if missing_direct or missing_locked or version_drift:
        raise AuthorRecipeError(
            "UV project does not fully declare the author-recipe runtime; "
            f"missing_direct={missing_direct}, missing_locked={missing_locked}, "
            f"version_drift={version_drift}"
        )
    result: dict[str, Any] = {
        "project_root": str(project_root),
        "pyproject_sha256": _sha256_bytes(pyproject_bytes),
        "lock_sha256": _sha256_bytes(lock_bytes),
        "required_exact_pins": dict(sorted(pinned_direct.items())),
        "lock_current": None,
        "project_venv": None,
        "environment_exact_no_extras": None,
        "pip_check": None,
    }
    if not check_runtime:
        return result
    venv = project_root / ".venv"
    _require_safe_ancestors(
        venv,
        root=project_root,
        include_leaf=True,
        allow_missing_leaf=False,
        context="UV project virtual environment",
    )
    venv_stat = os.lstat(venv)
    if not stat.S_ISDIR(venv_stat.st_mode) or stat.S_ISLNK(venv_stat.st_mode):
        raise AuthorRecipeError("the UV project .venv must be a real directory")
    if Path(sys.prefix).resolve() != venv.resolve():
        raise AuthorRecipeError(
            "formal author-recipe execution must use this project's .venv"
        )
    executable = _absolute_path(sys.executable)
    if venv != executable and venv not in executable.parents:
        raise AuthorRecipeError("Python executable is outside the project .venv")
    lock_check = _run_uv_integrity_command(
        (
            "uv",
            "lock",
            "--check",
            "--offline",
            "--project",
            str(project_root),
        ),
        project_root=project_root,
    )
    sync_check = _run_uv_integrity_command(
        (
            "uv",
            "sync",
            "--check",
            "--frozen",
            "--no-dev",
            "--offline",
            "--project",
            str(project_root),
            "--python",
            str(executable),
        ),
        project_root=project_root,
    )
    pip_check = _run_uv_integrity_command(
        (
            "uv",
            "pip",
            "check",
            "--offline",
            "--python",
            str(executable),
        ),
        project_root=project_root,
    )
    freeze = _run_uv_integrity_command(
        (
            "uv",
            "pip",
            "freeze",
            "--strict",
            "--offline",
            "--python",
            str(executable),
        ),
        project_root=project_root,
    )
    result.update(
        {
            "lock_current": True,
            "project_venv": str(venv),
            "environment_exact_no_extras": True,
            "pip_check": True,
            "uv_lock_check_output_sha256": _sha256_bytes(
                lock_check.encode("utf-8")
            ),
            "uv_sync_check_output_sha256": _sha256_bytes(
                sync_check.encode("utf-8")
            ),
            "uv_pip_check_output_sha256": _sha256_bytes(
                pip_check.encode("utf-8")
            ),
            "installed_freeze_sha256": _sha256_bytes(
                freeze.encode("utf-8")
            ),
        }
    )
    return result


def _source_identity() -> dict[str, str]:
    project_root = Path(__file__).resolve().parents[2]
    _validate_uv_project_contract(project_root, check_runtime=False)
    paths = {name: project_root / name for name in SOURCE_FILES}
    missing = sorted(name for name, path in paths.items() if not os.path.lexists(path))
    if missing:
        raise AuthorRecipeError(f"source-closure files are absent: {missing}")
    return {name: _sha256_file(path) for name, path in sorted(paths.items())}


def _environment_identity() -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[2]
    uv_integrity = _validate_uv_project_contract(project_root)
    identity = full_grid._environment_identity()
    if identity.get("uv_version") is None:
        raise AuthorRecipeError("formal runtime identity requires the uv executable")
    packages = {str(name): str(version) for name, version in identity["packages"]}
    expected = {
        "braindecode": BRAIDECODE_VERSION,
    }
    mismatched = {
        name: {"expected": version, "observed": packages.get(name)}
        for name, version in expected.items()
        if packages.get(name) != version
    }
    if mismatched:
        raise AuthorRecipeError(
            f"author-recipe runtime has incompatible package versions: {mismatched}"
        )
    identity["uv_project_integrity"] = uv_integrity
    return identity


def _normalized_worker_environment(identity: Mapping[str, Any]) -> dict[str, Any]:
    return full_grid._normalized_worker_environment_identity(identity)


def _tcformer_source_root() -> Path:
    value = os.environ.get("EEG_MI_TCFORMER_ROOT")
    if not value:
        raise AuthorRecipeError(
            "EEG_MI_TCFORMER_ROOT must name the pinned vendored source"
        )
    root = _absolute_path(value)
    _require_safe_ancestors(
        root,
        include_leaf=True,
        allow_missing_leaf=False,
        context="TCFormer source",
    )
    observed = os.lstat(root)
    if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
        raise AuthorRecipeError("TCFormer source root must be a real directory")
    return root


def _tcformer_source_identity() -> dict[str, Any]:
    root = _tcformer_source_root()
    try:
        return verify_tcformer_source(root)
    except RuntimeError as error:
        raise AuthorRecipeError(
            f"TCFormer vendored source cannot be verified at {root}"
        ) from error


def _reference_provenance() -> dict[str, Any]:
    return {
        "reference.tcformer": _tcformer_source_identity(),
        "reference.fbcnet": {
            "repository": "https://github.com/ravikiran-mane/FBCNet",
            "source_commit": FBCNET_SOURCE_COMMIT,
            "runtime_adapter": "braindecode.models.FBCNet",
            "braindecode_version": BRAIDECODE_VERSION,
            "checkout_used_at_runtime": False,
        },
    }


def _dataset_contracts() -> dict[str, Any]:
    return full_grid._dataset_contracts()


def _validate_registry_contract() -> None:
    from .model_registry import BENCHMARK_DATASETS, MODEL_REGISTRY

    if tuple(BENCHMARK_DATASETS) != OPENED_DATASETS:
        raise AuthorRecipeError(
            "opened dataset roster differs from the model registry"
        )
    indexed = {record.stable_id: record for record in MODEL_REGISTRY}
    expected_implementations = {
        "reference.tcformer": (
            "benchmark.reference_training.fit_tcformer_reference"
        ),
        "reference.fbcnet": (
            "benchmark.reference_training.fit_fbcnet_reference"
        ),
    }
    common_names = {
        str(record.common_roster_name)
        for record in MODEL_REGISTRY
        if record.common_roster_name is not None
    }
    if set(REFERENCES).intersection(common_names):
        raise AuthorRecipeError(
            "author-recipe stable IDs alias common-roster architecture names"
        )
    for stable_id in REFERENCES:
        record = indexed.get(stable_id)
        # ``author_faithful`` is the registry's existing routing enum. The
        # scientific output is deliberately labelled author-recipe adapted;
        # this check must not be read as an exact-reproduction claim.
        if (
            record is None
            or record.track != "author_faithful"
            or record.identity_level != "procedure_configuration"
            or record.common_roster_name is not None
            or record.implementation != expected_implementations[stable_id]
            or any(
                not record.eligibility_for(dataset).eligible
                for dataset in OPENED_DATASETS
            )
        ):
            raise AuthorRecipeError(
                f"registry contract drifted for {stable_id}"
            )


def _cache_and_split_identity(
    cache_root: Path, contracts: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    from .data import split_indices

    cache_result: dict[str, Any] = {}
    split_result: dict[str, Any] = {}
    for dataset in contracts:
        for raw_subject in contracts[dataset]["subjects"]:
            subject = int(raw_subject)
            cache = _load_subject_cache_secure(
                dataset, subject, cache_root=cache_root
            )
            identity = copy.deepcopy(cache["identity"])
            digest = identity.get("array_sha256")
            if not isinstance(digest, str) or not HEX_64_RE.fullmatch(digest):
                raise AuthorRecipeError(
                    f"cache {dataset} S{subject} has no valid array digest"
                )
            cache_result[_subject_key(dataset, subject)] = identity
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
                split_result[_split_key(dataset, subject, fold)] = (
                    full_grid._one_split_identity(
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


def _load_subject_cache_secure(
    dataset: str,
    subject: int,
    *,
    cache_root: Path,
) -> dict[str, Any]:
    """Decode one cache only from a contained, no-follow, single-link snapshot."""

    from . import data as data_module

    root = _absolute_path(cache_root)
    profile = data_module.validate_montage_profile(
        data_module.DEFAULT_MONTAGE_PROFILE
    )
    requested_channels = data_module.channels_for_dataset(dataset, profile)
    path = data_module._cache_path(root, dataset, int(subject), profile)
    payload = _read_regular_bytes(
        path,
        root=root,
        readonly=False,
        context=f"subject cache {dataset} S{int(subject)}",
    )
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            result = {key: archive[key].copy() for key in archive.files}
    except (OSError, ValueError) as error:
        raise AuthorRecipeError(
            f"cache {path} cannot be decoded: {error}"
        ) from error
    try:
        identity_scalar = result["identity"]
        if identity_scalar.ndim != 0:
            raise ValueError("identity is not scalar")
        result["identity"] = _decode_json_object(
            str(identity_scalar.item()).encode("utf-8"),
            context=f"cache identity {path}",
        )
    except (KeyError, TypeError, ValueError, AuthorRecipeError) as error:
        raise AuthorRecipeError(
            f"cache {path} has an invalid identity"
        ) from error
    identity = result["identity"]
    required = {
        "x",
        "y",
        "positions",
        "channel_names",
        "sessions",
        "runs",
        "identity",
    }
    if set(result) != required:
        raise AuthorRecipeError(
            f"cache {path} fields differ from the frozen schema: "
            f"{sorted(result)}"
        )
    dataset_identity = identity.get("dataset", {})
    if (
        not isinstance(dataset_identity, Mapping)
        or dataset_identity.get("key") != dataset
        or int(identity.get("subject", -1)) != int(subject)
        or identity.get("montage_profile") != profile
    ):
        raise AuthorRecipeError(
            f"cache identity does not match {dataset} S{int(subject)}"
        )
    expected_preprocessing = data_module.preprocessing_for_dataset(dataset)
    if identity.get("preprocessing") != expected_preprocessing:
        raise AuthorRecipeError(
            f"cache preprocessing is stale for {dataset} S{int(subject)}"
        )
    channel_names = tuple(
        str(value) for value in result["channel_names"].tolist()
    )
    if (
        not channel_names
        or len(set(channel_names)) != len(channel_names)
        or (
            requested_channels is not None
            and channel_names != requested_channels
        )
        or identity.get("channels") != list(channel_names)
    ):
        raise AuthorRecipeError(
            f"cache channel contract is stale for {dataset} S{int(subject)}"
        )
    expected_coordinates = data_module.coordinate_contract_for_dataset(
        dataset, profile, channel_names
    )
    if identity.get("coordinates") != expected_coordinates:
        raise AuthorRecipeError(
            f"cache coordinate contract is stale for {dataset} S{int(subject)}"
        )
    if list(result["x"].shape) != identity.get("shape"):
        raise AuthorRecipeError(
            f"cache shape identity is stale for {dataset} S{int(subject)}"
        )
    expected_n_times = int(expected_preprocessing["n_times"])
    if (
        result["x"].ndim != 3
        or result["x"].shape[1:]
        != (len(channel_names), expected_n_times)
        or result["x"].dtype != np.float32
        or result["positions"].dtype != np.float32
        or result["y"].dtype != np.int64
        or result["positions"].shape != (len(channel_names), 3)
    ):
        raise AuthorRecipeError(
            f"cache array contract is invalid for {dataset} S{int(subject)}"
        )
    expected_positions = data_module._atlas_unit_positions(channel_names)
    if (
        not np.allclose(
            result["positions"], expected_positions, rtol=0.0, atol=1e-7
        )
        or not np.allclose(
            np.linalg.norm(result["positions"], axis=1),
            1.0,
            rtol=0.0,
            atol=1e-6,
        )
    ):
        raise AuthorRecipeError(
            f"cache coordinates are invalid for {dataset} S{int(subject)}"
        )
    trial_count = result["x"].shape[0]
    if not all(
        len(result[key]) == trial_count
        for key in ("y", "sessions", "runs")
    ):
        raise AuthorRecipeError(
            f"cache row counts differ for {dataset} S{int(subject)}"
        )
    expected_labels = set(range(data_module.dataset_spec(dataset).n_classes))
    if (
        set(result["y"].tolist()) != expected_labels
        or not np.all(np.isfinite(result["x"]))
        or not np.all(np.isfinite(result["positions"]))
    ):
        raise AuthorRecipeError(
            f"cache values are invalid for {dataset} S{int(subject)}"
        )
    digest = hashlib.sha256()
    for key in ("x", "y", "positions", "sessions", "runs"):
        array = result[key]
        if key in {"sessions", "runs"}:
            array = array.astype("U")
        digest.update(np.ascontiguousarray(array).tobytes())
    if digest.hexdigest() != identity.get("array_sha256"):
        raise AuthorRecipeError(
            f"cache array digest failed for {dataset} S{int(subject)}"
        )
    return result


def _recipe_contracts() -> dict[str, Any]:
    tcformer_by_dataset = {
        dataset: asdict(tcformer_reference_config(dataset, device="cuda"))
        for dataset in OPENED_DATASETS
    }
    fbcnet = asdict(FBCNetReferenceConfig(device="cuda"))
    return {
        "reference.tcformer": {
            "component_callable": (
                "benchmark.reference_training.fit_tcformer_reference"
            ),
            "factory_callable": "benchmark.baselines.make_model(tcformer)",
            "per_dataset_config": tcformer_by_dataset,
            "selection": "prespecified_fixed_horizon_no_outcome_selection",
            "final_fit": "train_plus_validation_for_prespecified_horizon",
        },
        "reference.fbcnet": {
            "component_callable": (
                "benchmark.reference_training.fit_fbcnet_reference"
            ),
            "factory_callable": (
                "PrefilteredFBCNetAdapter(benchmark.baselines.make_model(fbcnet))"
            ),
            "config": fbcnet,
            "selection": "train_fit_validation_inaccuracy_patience",
            "final_fit": (
                "restore_best_model_and_adam_then_train_plus_validation_until_"
                "released_threshold_or_cap"
            ),
        },
    }


def _analysis_contract(
    environment_identity: Mapping[str, Any],
) -> dict[str, Any]:
    from . import author_recipe_analysis

    project_root = Path(__file__).resolve().parents[2]
    analysis_path = (
        project_root
        / "src"
        / "benchmark"
        / "author_recipe_analysis.py"
    )
    if not os.path.lexists(analysis_path):
        raise AuthorRecipeError("author_recipe_analysis.py is absent")
    decisions = author_recipe_analysis.decision_contract()
    return {
        "schema": ANALYSIS_CONTRACT_SCHEMA,
        "source_identity": {
            "src/benchmark/author_recipe_analysis.py": _sha256_file(analysis_path),
        },
        "environment_identity_sha256": _sha256_bytes(
            _canonical_bytes(_normalized_worker_environment(environment_identity))
        ),
        "track": TRACK,
        "track_description": TRACK_DESCRIPTION,
        "decision_contract": decisions,
        "decision_contract_sha256": _sha256_bytes(
            _canonical_bytes(decisions)
        ),
        "label_join": "after_exact_quiescent_score_blind_grid_audit_only",
        "evidence_scope": "opened_development_datasets_only_not_confirmation",
        "common_track_comparability": (
            "separately_labelled_author_recipe_results_never_common_recipe_rows"
        ),
    }


def _validate_split_and_cache_contract(
    datasets: Mapping[str, Any],
    cache_identity: Mapping[str, Any],
    split_identity: Mapping[str, Any],
) -> None:
    expected_cache_keys = {
        _subject_key(dataset, subject)
        for dataset, contract in datasets.items()
        for subject in contract["subjects"]
    }
    expected_split_keys = {
        _split_key(dataset, subject, fold)
        for dataset, contract in datasets.items()
        for subject in contract["subjects"]
        for fold in contract["folds"]
    }
    if set(cache_identity) != expected_cache_keys:
        raise ValueError("cache identity keys differ from the dataset subjects")
    if set(split_identity) != expected_split_keys:
        raise ValueError("split identity keys differ from the subject/fold grid")
    for key, value in cache_identity.items():
        digest = value.get("array_sha256") if isinstance(value, Mapping) else None
        if not isinstance(digest, str) or not HEX_64_RE.fullmatch(digest):
            raise ValueError(f"cache identity {key} has no valid array digest")
    for key, value in split_identity.items():
        if not isinstance(value, Mapping):
            raise ValueError(f"split identity {key} is not an object")
        partitions = value.get("partitions")
        if not isinstance(partitions, Mapping) or set(partitions) != {
            "train",
            "validation",
            "source",
            "test",
        }:
            raise ValueError(f"split identity {key} has invalid partitions")
        expected_key = _split_key(
            str(value.get("dataset")),
            int(value.get("subject", -1)),
            int(value.get("fold", -1)),
        )
        subject_key = _subject_key(
            str(value.get("dataset")), int(value.get("subject", -1))
        )
        if (
            key != expected_key
            or subject_key not in cache_identity
            or value.get("cache_array_sha256")
            != cache_identity[subject_key].get("array_sha256")
        ):
            raise ValueError(f"split identity {key} is not bound to its cache")
        try:
            trial_count = int(value["trial_count"])
            counts = {
                name: int(partitions[name]["count"])
                for name in ("train", "validation", "source", "test")
            }
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"split identity {key} has invalid counts") from error
        for name, partition in partitions.items():
            digest = partition.get("rows_sha256")
            if (
                counts[name] <= 0
                or not isinstance(digest, str)
                or not HEX_64_RE.fullmatch(digest)
            ):
                raise ValueError(f"split identity {key}/{name} is invalid")
        if (
            counts["source"] != counts["train"] + counts["validation"]
            or counts["source"] + counts["test"] != trial_count
        ):
            raise ValueError(f"split identity {key} does not cover the cache")


def assemble_plan(
    *,
    dataset_contracts: Mapping[str, Any],
    references: Sequence[str],
    seeds: Sequence[int],
    cache_identity: Mapping[str, Any],
    split_identity: Mapping[str, Any],
    source_identity: Mapping[str, str],
    environment_identity: Mapping[str, Any],
    reference_provenance: Mapping[str, Any],
    recipe_contracts: Mapping[str, Any],
    analysis_contract: Mapping[str, Any] | None = None,
    executor: str = DEFAULT_EXECUTOR,
    worker_cpu_threads: int = 4,
) -> dict[str, Any]:
    """Construct a canonical plan from explicit, test-injectable components."""

    reference_values = tuple(str(value) for value in references)
    seed_values = tuple(int(value) for value in seeds)
    if reference_values != REFERENCES:
        raise ValueError(f"references must be exactly {REFERENCES}")
    if set(reference_values).intersection(full_grid.COMMON_ARCHITECTURES):
        raise ValueError(
            "author-recipe stable IDs may not alias common architecture names"
        )
    if not seed_values or len(seed_values) != len(set(seed_values)):
        raise ValueError("seeds must be nonempty and unique")
    worker_cpu_threads = full_grid._validate_cpu_threads(worker_cpu_threads)
    normalized_datasets: dict[str, Any] = {}
    subject_fold_count = 0
    for dataset, raw_contract in dataset_contracts.items():
        contract = copy.deepcopy(dict(raw_contract))
        subjects = [int(value) for value in contract["subjects"]]
        folds = [int(value) for value in contract["folds"]]
        if (
            not subjects
            or not folds
            or len(subjects) != len(set(subjects))
            or len(folds) != len(set(folds))
        ):
            raise ValueError(f"{dataset} subjects/folds must be nonempty and unique")
        contract["subjects"] = subjects
        contract["folds"] = folds
        contract["n_classes"] = int(contract["n_classes"])
        normalized_datasets[str(dataset)] = json.loads(
            _canonical_bytes(contract)
        )
        subject_fold_count += len(subjects) * len(folds)
    if not normalized_datasets:
        raise ValueError("at least one dataset contract is required")
    caches = copy.deepcopy(dict(cache_identity))
    splits = copy.deepcopy(dict(split_identity))
    _validate_split_and_cache_contract(normalized_datasets, caches, splits)
    sources = copy.deepcopy(dict(source_identity))
    if not sources or any(
        not isinstance(value, str) or not HEX_64_RE.fullmatch(value)
        for value in sources.values()
    ):
        raise ValueError("source identity must contain lowercase SHA-256 values")
    if set(reference_provenance) != set(REFERENCES):
        raise ValueError("reference provenance differs from the exact roster")
    if set(recipe_contracts) != set(REFERENCES):
        raise ValueError("recipe contracts differ from the exact roster")
    formal_roster = (
        tuple(normalized_datasets) == OPENED_DATASETS
        and seed_values == FORMAL_SEEDS
        and reference_values == REFERENCES
    )
    if not isinstance(executor, str) or not executor:
        raise ValueError("executor must be a nonempty import path")
    if formal_roster and executor != DEFAULT_EXECUTOR:
        raise AuthorRecipeError(
            "formal plans require the audited default executor"
        )

    payload: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "purpose": "author_recipe_adapted_reference_prediction_grid",
        "track": TRACK,
        "evidence_scope": "opened_development_datasets_only_not_confirmation",
        "confirmation_evidence": False,
        "score_blind": True,
        "common_recipe_track": False,
        "identity_contract": {
            "namespace": "registry_stable_id_author_recipe_adapted",
            "stable_ids": list(reference_values),
            "bare_common_architecture_aliases_forbidden": [
                "tcformer",
                "fbcnet",
            ],
        },
        "dataset_order": list(normalized_datasets),
        "datasets": normalized_datasets,
        "references": list(reference_values),
        "seeds": list(seed_values),
        "cache_identity": caches,
        "split_identity": splits,
        "source_identity": sources,
        "environment_identity": copy.deepcopy(dict(environment_identity)),
        "reference_provenance": copy.deepcopy(dict(reference_provenance)),
        "recipe_contracts": copy.deepcopy(dict(recipe_contracts)),
        "analysis_contract": (
            None
            if analysis_contract is None
            else copy.deepcopy(dict(analysis_contract))
        ),
        "executor": executor,
        "execution_config": {"worker_cpu_threads": worker_cpu_threads},
        "job_order": ["dataset", "reference", "subject", "fold", "seed"],
        "n_jobs": subject_fold_count * len(reference_values) * len(seed_values),
        "output_contract": {
            "test_labels_present": False,
            "test_scores_present": False,
            "prediction_arrays": ["rows", "probabilities"],
            "atomic_unit": "dataset/reference/subject/fold/seed",
            "test_inference": "one_predict_probabilities_call_after_source_fit",
            "analysis_track": TRACK,
            "common_track_join_forbidden": True,
        },
    }
    payload["plan_sha256"] = plan_sha256(payload)
    return payload


def plan_sha256(plan: Mapping[str, Any]) -> str:
    payload = copy.deepcopy(dict(plan))
    payload.pop("plan_sha256", None)
    return _sha256_bytes(_canonical_bytes(payload))


def _validate_plan_shape(plan: Mapping[str, Any]) -> None:
    _require_exact_keys(
        plan,
        {
            "schema",
            "purpose",
            "track",
            "evidence_scope",
            "confirmation_evidence",
            "score_blind",
            "common_recipe_track",
            "identity_contract",
            "dataset_order",
            "datasets",
            "references",
            "seeds",
            "cache_identity",
            "split_identity",
            "source_identity",
            "environment_identity",
            "reference_provenance",
            "recipe_contracts",
            "analysis_contract",
            "executor",
            "execution_config",
            "job_order",
            "n_jobs",
            "output_contract",
            "plan_sha256",
        },
        "immutable plan",
    )
    if (
        plan["schema"] != PLAN_SCHEMA
        or plan["purpose"] != "author_recipe_adapted_reference_prediction_grid"
        or plan["track"] != TRACK
        or plan["evidence_scope"]
        != "opened_development_datasets_only_not_confirmation"
        or plan["confirmation_evidence"] is not False
        or plan["score_blind"] is not True
        or plan["common_recipe_track"] is not False
    ):
        raise AuthorRecipeError("immutable plan scope/track contract is invalid")
    identity = plan["identity_contract"]
    if not isinstance(identity, Mapping):
        raise AuthorRecipeError("immutable plan identity contract is absent")
    _require_exact_keys(
        identity,
        {
            "namespace",
            "stable_ids",
            "bare_common_architecture_aliases_forbidden",
        },
        "immutable plan identity contract",
    )
    if identity != {
        "namespace": "registry_stable_id_author_recipe_adapted",
        "stable_ids": list(REFERENCES),
        "bare_common_architecture_aliases_forbidden": ["tcformer", "fbcnet"],
    }:
        raise AuthorRecipeError("immutable plan identity separation is invalid")
    if (
        not isinstance(plan["dataset_order"], list)
        or not plan["dataset_order"]
        or len(plan["dataset_order"]) != len(set(plan["dataset_order"]))
        or not isinstance(plan["datasets"], Mapping)
        or set(plan["datasets"]) != set(plan["dataset_order"])
        or plan["references"] != list(REFERENCES)
        or not isinstance(plan["seeds"], list)
        or not plan["seeds"]
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in plan["seeds"]
        )
        or len(plan["seeds"]) != len(set(plan["seeds"]))
        or plan["job_order"]
        != ["dataset", "reference", "subject", "fold", "seed"]
    ):
        raise AuthorRecipeError("immutable plan Cartesian roster is invalid")
    executor = plan["executor"]
    if not isinstance(executor, str) or not executor:
        raise AuthorRecipeError("immutable plan executor is invalid")
    if (
        tuple(plan["dataset_order"]) == OPENED_DATASETS
        and tuple(plan["references"]) == REFERENCES
        and tuple(plan["seeds"]) == FORMAL_SEEDS
        and executor != DEFAULT_EXECUTOR
    ):
        raise AuthorRecipeError(
            "formal immutable plans require the audited default executor"
        )
    for dataset in plan["dataset_order"]:
        if (
            not isinstance(dataset, str)
            or not dataset
            or dataset in {".", ".."}
            or "/" in dataset
            or (os.altsep is not None and os.altsep in dataset)
        ):
            raise AuthorRecipeError("immutable plan has an unsafe dataset name")
    execution = plan["execution_config"]
    if not isinstance(execution, Mapping):
        raise AuthorRecipeError("immutable plan execution config is absent")
    _require_exact_keys(
        execution, {"worker_cpu_threads"}, "immutable plan execution config"
    )
    threads = execution["worker_cpu_threads"]
    if isinstance(threads, bool) or not isinstance(threads, int) or threads <= 0:
        raise AuthorRecipeError("immutable plan CPU thread count is invalid")
    output = plan["output_contract"]
    if not isinstance(output, Mapping):
        raise AuthorRecipeError("immutable plan output contract is absent")
    _require_exact_keys(
        output,
        {
            "test_labels_present",
            "test_scores_present",
            "prediction_arrays",
            "atomic_unit",
            "test_inference",
            "analysis_track",
            "common_track_join_forbidden",
        },
        "immutable plan output contract",
    )
    if output != {
        "test_labels_present": False,
        "test_scores_present": False,
        "prediction_arrays": ["rows", "probabilities"],
        "atomic_unit": "dataset/reference/subject/fold/seed",
        "test_inference": "one_predict_probabilities_call_after_source_fit",
        "analysis_track": TRACK,
        "common_track_join_forbidden": True,
    }:
        raise AuthorRecipeError("immutable plan output contract is invalid")
    expected_jobs = sum(
        len(plan["datasets"][dataset]["subjects"])
        * len(plan["datasets"][dataset]["folds"])
        for dataset in plan["dataset_order"]
    ) * len(REFERENCES) * len(plan["seeds"])
    if (
        isinstance(plan["n_jobs"], bool)
        or not isinstance(plan["n_jobs"], int)
        or plan["n_jobs"] != expected_jobs
    ):
        raise AuthorRecipeError("immutable plan job count is invalid")
    digest = plan.get("plan_sha256")
    if (
        not isinstance(digest, str)
        or not HEX_64_RE.fullmatch(digest)
        or digest != plan_sha256(plan)
    ):
        raise AuthorRecipeError("immutable plan digest is invalid")


def build_plan(
    *,
    cache_root: Path,
    executor: str = DEFAULT_EXECUTOR,
    worker_cpu_threads: int = 4,
) -> dict[str, Any]:
    if executor != DEFAULT_EXECUTOR:
        raise AuthorRecipeError(
            "formal plans require the audited default executor; injected "
            "executors are test-only"
        )
    _validate_registry_contract()
    contracts = _dataset_contracts()
    caches, splits = _cache_and_split_identity(cache_root, contracts)
    environment = _environment_identity()
    plan = assemble_plan(
        dataset_contracts=contracts,
        references=REFERENCES,
        seeds=FORMAL_SEEDS,
        cache_identity=caches,
        split_identity=splits,
        source_identity=_source_identity(),
        environment_identity=environment,
        reference_provenance=_reference_provenance(),
        recipe_contracts=_recipe_contracts(),
        analysis_contract=_analysis_contract(environment),
        executor=executor,
        worker_cpu_threads=worker_cpu_threads,
    )
    if int(plan["n_jobs"]) != EXPECTED_JOB_COUNT:
        raise AuthorRecipeError(
            f"formal author grid has {plan['n_jobs']} jobs, expected "
            f"{EXPECTED_JOB_COUNT}"
        )
    return plan


def write_or_validate_plan(
    run_root: Path, expected: Mapping[str, Any]
) -> dict[str, Any]:
    run_root = _absolute_path(run_root)
    _safe_mkdir(run_root)
    path = run_root / "plan.json"
    digest_path = run_root / "plan.sha256"
    expected_value = copy.deepcopy(dict(expected))
    _validate_plan_shape(expected_value)
    digest = plan_sha256(expected_value)
    if expected_value.get("plan_sha256") != digest:
        raise AuthorRecipeError("expected plan carries an invalid checksum")
    if os.path.lexists(path) and os.path.lexists(digest_path):
        observed = load_plan(run_root)
        if observed != expected_value:
            raise AuthorRecipeError(
                "existing immutable plan differs from the requested plan"
            )
        return observed
    if os.path.lexists(path):
        observed = strict_load(path, root=run_root, readonly=True)
        if (
            observed != expected_value
            or observed.get("schema") != PLAN_SCHEMA
            or observed.get("plan_sha256") != digest
            or _read_regular_bytes(
                path, root=run_root, readonly=True, context="immutable plan"
            )
            != _canonical_bytes(observed) + b"\n"
        ):
            raise AuthorRecipeError("orphaned plan.json is not the expected plan")
        try:
            _write_bytes_exclusive(
                digest_path,
                f"{digest}\n".encode("ascii"),
                root=run_root,
            )
        except FileExistsError:
            pass
        return load_plan(run_root)
    if os.path.lexists(digest_path):
        if (
            _read_regular_bytes(
                digest_path,
                root=run_root,
                readonly=True,
                context="immutable plan digest",
            )
            .decode("ascii")
            .strip()
            != digest
        ):
            raise AuthorRecipeError("orphaned plan digest differs from expected")
        try:
            _write_json_exclusive(path, expected_value, root=run_root)
        except FileExistsError:
            pass
        return load_plan(run_root)
    try:
        _write_json_exclusive(path, expected_value, root=run_root)
        _write_bytes_exclusive(
            digest_path, f"{digest}\n".encode("ascii"), root=run_root
        )
    except FileExistsError:
        observed = load_plan(run_root)
        if observed != expected_value:
            raise AuthorRecipeError("concurrent plan creation produced drift")
        return observed
    return load_plan(run_root)


def load_plan(run_root: Path) -> dict[str, Any]:
    run_root = _absolute_path(run_root)
    _require_safe_ancestors(
        run_root,
        include_leaf=True,
        allow_missing_leaf=False,
        context="run root",
    )
    path = run_root / "plan.json"
    digest_path = run_root / "plan.sha256"
    if not os.path.lexists(path) or not os.path.lexists(digest_path):
        raise FileNotFoundError(f"plan.json/plan.sha256 absent under {run_root}")
    plan = strict_load(path, root=run_root, readonly=True)
    _validate_plan_shape(plan)
    digest = plan_sha256(plan)
    digest_bytes = _read_regular_bytes(
        digest_path,
        root=run_root,
        readonly=True,
        context="immutable plan digest",
    )
    plan_bytes = _read_regular_bytes(
        path, root=run_root, readonly=True, context="immutable plan"
    )
    if (
        plan.get("plan_sha256") != digest
        or digest_bytes.decode("ascii").strip() != digest
        or plan_bytes != _canonical_bytes(plan) + b"\n"
    ):
        raise AuthorRecipeError("immutable plan checksum/schema is invalid")
    return plan


def load_or_repair_plan(run_root: Path) -> dict[str, Any]:
    run_root = _absolute_path(run_root)
    path = run_root / "plan.json"
    digest_path = run_root / "plan.sha256"
    if os.path.lexists(digest_path):
        return load_plan(run_root)
    if not os.path.lexists(path):
        raise FileNotFoundError(f"plan.json/plan.sha256 absent under {run_root}")
    observed = strict_load(path, root=run_root, readonly=True)
    _validate_plan_shape(observed)
    digest = plan_sha256(observed)
    if (
        observed.get("schema") != PLAN_SCHEMA
        or observed.get("plan_sha256") != digest
        or _read_regular_bytes(
            path, root=run_root, readonly=True, context="orphaned plan"
        )
        != _canonical_bytes(observed) + b"\n"
    ):
        raise AuthorRecipeError("orphaned plan.json is invalid")
    return write_or_validate_plan(run_root, observed)


def _verify_static_runtime_identity(
    plan: Mapping[str, Any],
    *,
    worker_cuda_visibility: bool = False,
) -> None:
    _validate_plan_shape(plan)
    if plan.get("executor") != DEFAULT_EXECUTOR:
        raise AuthorRecipeError(
            "formal runtime requires the audited default executor"
        )
    if worker_cuda_visibility:
        for module_name in (
            "reference_training",
            "baselines",
            "models",
            "training",
            "data",
            "config",
            "author_recipe_analysis",
        ):
            importlib.import_module(f"{__package__}.{module_name}")
    if tuple(plan.get("dataset_order", ())) != OPENED_DATASETS:
        raise AuthorRecipeError("plan does not contain the five opened datasets")
    if plan.get("datasets") != _dataset_contracts():
        raise AuthorRecipeError("dataset contracts differ from the immutable plan")
    if tuple(plan.get("references", ())) != REFERENCES:
        raise AuthorRecipeError("reference roster differs from the immutable plan")
    expected_identity_contract = {
        "namespace": "registry_stable_id_author_recipe_adapted",
        "stable_ids": list(REFERENCES),
        "bare_common_architecture_aliases_forbidden": [
            "tcformer",
            "fbcnet",
        ],
    }
    if plan.get("identity_contract") != expected_identity_contract:
        raise AuthorRecipeError(
            "author/common track identity separation has drifted"
        )
    _validate_registry_contract()
    if tuple(plan.get("seeds", ())) != FORMAL_SEEDS:
        raise AuthorRecipeError("seed roster differs from the immutable plan")
    if plan.get("recipe_contracts") != _recipe_contracts():
        raise AuthorRecipeError("reference recipe contracts have drifted")
    if _source_identity() != plan.get("source_identity"):
        raise AuthorRecipeError("source closure differs from the immutable plan")
    if _reference_provenance() != plan.get("reference_provenance"):
        raise AuthorRecipeError("official reference provenance has drifted")
    observed_environment = _environment_identity()
    expected_environment: Mapping[str, Any] = plan["environment_identity"]
    if worker_cuda_visibility:
        observed_environment = _normalized_worker_environment(observed_environment)
        expected_environment = _normalized_worker_environment(expected_environment)
    if observed_environment != expected_environment:
        raise AuthorRecipeError("full UV runtime differs from the immutable plan")
    expected_analysis = _analysis_contract(plan["environment_identity"])
    if plan.get("analysis_contract") != expected_analysis:
        raise AuthorRecipeError("frozen label-joining analysis contract has drifted")


def verify_runtime_identity(
    plan: Mapping[str, Any],
    *,
    cache_root: Path,
    worker_cuda_visibility: bool = False,
) -> None:
    """Verify the complete formal runtime, every cache, and every split."""

    _verify_static_runtime_identity(
        plan, worker_cuda_visibility=worker_cuda_visibility
    )
    caches, splits = _cache_and_split_identity(cache_root, plan["datasets"])
    if caches != plan.get("cache_identity"):
        raise AuthorRecipeError("cache identities differ from the immutable plan")
    if splits != plan.get("split_identity"):
        raise AuthorRecipeError("split identities differ from the immutable plan")


def _rebind_authoritative_inputs(
    *,
    run_root: Path,
    plan: Mapping[str, Any],
    cache_root: Path | None,
    job: Job,
    claim: Claim,
    metadata: Mapping[str, Any],
    requested_gpu: str | None,
) -> Mapping[str, Any]:
    """Rebind every frozen input and held worker capability at commit time."""

    root = _absolute_path(run_root)
    live_plan = load_plan(root)
    if _canonical_bytes(live_plan) != _canonical_bytes(plan):
        raise AuthorRecipeError(
            "immutable plan changed at the publication boundary"
        )
    _validate_plan_shape(live_plan)
    if live_plan.get("executor") == DEFAULT_EXECUTOR:
        if cache_root is None:
            raise AuthorRecipeError(
                "formal publication requires the exact cache-root identity"
            )
        _verify_static_runtime_identity(
            live_plan,
            worker_cuda_visibility=True,
        )
        job_contract = copy.deepcopy(
            dict(live_plan["datasets"][job.dataset])
        )
        job_contract["subjects"] = [job.subject]
        job_contract["folds"] = [job.fold]
        caches, splits = _cache_and_split_identity(
            _absolute_path(cache_root),
            {job.dataset: job_contract},
        )
        if caches != {
            _subject_key(job.dataset, job.subject): _planned_cache(
                live_plan, job
            )
        }:
            raise AuthorRecipeError(
                "live job cache differs at publication boundary"
            )
        if splits != {
            _split_key(job.dataset, job.subject, job.fold): _planned_split(
                live_plan, job
            )
        }:
            raise AuthorRecipeError(
                "live job split differs at publication boundary"
            )
    _validate_job_identity_mapping(job.identity(), job, "commit job")
    _validate_job_metadata(live_plan, job, metadata)
    if (
        metadata.get("cache_array_sha256")
        != _planned_cache(live_plan, job).get("array_sha256")
        or metadata.get("split") != _split_metadata(
            _planned_split(live_plan, job)
        )
    ):
        raise AuthorRecipeError(
            "record cache/split identity changed at publication"
        )
    claim_payload = _validate_claim_payload(root, live_plan, claim)
    if not _claim_is_live(claim_payload):
        raise ClaimUnavailable("job claim is no longer live at publication")
    if requested_gpu is not None and (
        claim.resource_guard.get("gpu", {}).get("gpu") != requested_gpu
    ):
        raise ClaimUnavailable(
            "requested GPU differs from the device bound by the claim"
        )
    try:
        gate_opened = os.fstat(claim.publication_gate_descriptor)
        gate_visible = _anchored_lstat(_publication_gate_path(root))
        gate_payload = os.pread(
            claim.publication_gate_descriptor,
            len(PUBLICATION_GATE_BYTES) + 1,
            0,
        )
    except (FileNotFoundError, OSError) as error:
        raise ClaimUnavailable(
            "job lost its shared publication-gate lease"
        ) from error
    if (
        (int(gate_opened.st_dev), int(gate_opened.st_ino))
        != (int(gate_visible.st_dev), int(gate_visible.st_ino))
        or stat.S_ISLNK(gate_visible.st_mode)
        or not stat.S_ISREG(gate_visible.st_mode)
        or gate_visible.st_nlink != 1
        or gate_visible.st_mode & 0o222
        or gate_payload != PUBLICATION_GATE_BYTES
        or os.path.lexists(_publication_fence_path(root))
    ):
        raise ClaimUnavailable(
            "publication-gate lease or fence identity changed"
        )
    return live_plan


def iter_jobs(plan: Mapping[str, Any]) -> Iterator[Job]:
    count = 0
    seen: set[str] = set()
    for dataset in plan["dataset_order"]:
        contract = plan["datasets"][dataset]
        for reference, subject, fold, seed in product(
            plan["references"],
            contract["subjects"],
            contract["folds"],
            plan["seeds"],
        ):
            job = Job(
                dataset=str(dataset),
                reference=str(reference),
                subject=int(subject),
                fold=int(fold),
                seed=int(seed),
            )
            if job.job_id in seen:
                raise AuthorRecipeError(f"job-id collision for {job.identity()}")
            seen.add(job.job_id)
            count += 1
            yield job
    if count != int(plan["n_jobs"]):
        raise AuthorRecipeError(
            f"plan declares {plan['n_jobs']} jobs but generated {count}"
        )


def _record_directory(run_root: Path, job: Job) -> Path:
    for value in (job.dataset, job.reference):
        if (
            not value
            or value in {".", ".."}
            or "/" in value
            or os.sep in value
            or (os.altsep is not None and os.altsep in value)
        ):
            raise AuthorRecipeError("job identity contains an unsafe path component")
    name = (
        f"s{job.subject:03d}_f{job.fold:02d}_seed{job.seed}_"
        f"{job.job_id.removeprefix('author-job-')[:10]}"
    )
    result, _ = _require_contained(
        _absolute_path(run_root) / "records" / job.dataset / job.reference / name,
        _absolute_path(run_root),
        "record directory",
    )
    return result


def _claim_path(run_root: Path, job: Job) -> Path:
    result, _ = _require_contained(
        _absolute_path(run_root)
        / "claims"
        / job.job_id[-2:]
        / f"{job.job_id}.json",
        _absolute_path(run_root),
        "claim path",
    )
    return result


def _failure_root(run_root: Path, job: Job) -> Path:
    result, _ = _require_contained(
        _absolute_path(run_root) / "failures" / job.job_id[-2:],
        _absolute_path(run_root),
        "failure path",
    )
    return result


def _quarantine(
    run_root: Path,
    path: Path,
    *,
    category: str,
    reason: str,
    expected_identity: tuple[int, int],
) -> Path:
    root = _absolute_path(run_root)
    source, _ = _require_contained(path, root, "quarantine source")
    _require_safe_ancestors(source, root=root, context="quarantine source")
    source_stat = _anchored_lstat(source)
    source_identity = (
        int(source_stat.st_dev),
        int(source_stat.st_ino),
    )
    if source_identity != tuple(map(int, expected_identity)):
        raise ClaimUnavailable(
            "quarantine source identity changed before invalidation"
        )
    if (
        not re.fullmatch(r"[A-Za-z0-9_.-]+", category)
        or not re.fullmatch(r"[A-Za-z0-9_.-]+", reason)
        or category in {"", ".", ".."}
        or reason in {"", ".", ".."}
    ):
        raise AuthorRecipeError("unsafe quarantine category/reason")
    destination_root = _safe_mkdir(
        root / "quarantine" / category, root=root
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    destination = destination_root / (
        f"{source.name}.{reason}.{stamp}.{uuid.uuid4().hex[:10]}"
    )
    source_is_directory = stat.S_ISDIR(source_stat.st_mode)
    if source_is_directory:
        descriptor = _open_directory_anchored(source)
        try:
            opened = os.fstat(descriptor)
            if (
                int(opened.st_dev),
                int(opened.st_ino),
            ) != source_identity:
                raise ClaimUnavailable(
                    "quarantine directory changed before invalidation"
                )
            os.fchmod(descriptor, 0o700)
            os.fsync(descriptor)
        except BaseException:
            # Never let a failed preparation leave a writable canonical
            # directory. fchmod(0555) happens before the seal fsync.
            try:
                os.fchmod(descriptor, 0o555)
                os.fsync(descriptor)
            finally:
                raise
        finally:
            os.close(descriptor)
        try:
            _fsync_directory(source.parent, root=root)
        except BaseException:
            _seal_directory_read_only(
                source,
                root=root,
                expected_identity=source_identity,
            )
            raise
    try:
        _rename_leaf_nofollow(
            source,
            destination,
            root=root,
            expected_source_identity=source_identity,
        )
    except BaseException:
        # A native no-replace helper can move the exact inode, fsync, then
        # raise. Treat only that exact canonical binding as a completed move.
        if not _path_binds_identity(destination, source_identity):
            if (
                source_is_directory
                and _path_binds_identity(source, source_identity)
            ):
                _seal_directory_read_only(
                    source,
                    root=root,
                    expected_identity=source_identity,
                )
            raise
    destination_stat = _anchored_lstat(destination)
    if source_identity != (
        int(destination_stat.st_dev),
        int(destination_stat.st_ino),
    ):
        raise AuthorRecipeError("quarantined path changed during rename")
    if source_is_directory:
        _seal_directory_read_only(
            destination,
            root=root,
            expected_identity=source_identity,
        )
    if source.parent != destination_root:
        _fsync_directory(source.parent, root=root)
    _fsync_directory(destination_root, root=root)
    return destination


def _process_identity() -> dict[str, Any]:
    return full_grid._process_identity()


def _claim_is_live(
    claim: Mapping[str, Any], *, foreign_claim_timeout_seconds: float = 0.0
) -> bool:
    return full_grid._claim_is_live(
        claim,
        foreign_claim_timeout_seconds=foreign_claim_timeout_seconds,
    )


def _validate_owner_mapping(value: Any, context: str) -> None:
    if not isinstance(value, Mapping):
        raise AuthorRecipeError(f"{context} owner is absent")
    _require_exact_keys(
        value, {"host", "pid", "boot_id", "start_ticks"}, f"{context} owner"
    )
    if (
        not isinstance(value["host"], str)
        or not value["host"]
        or isinstance(value["pid"], bool)
        or not isinstance(value["pid"], int)
        or value["pid"] <= 0
        or any(
            value[name] is not None and not isinstance(value[name], str)
            for name in ("boot_id", "start_ticks")
        )
    ):
        raise AuthorRecipeError(f"{context} owner is invalid")


def _publication_gate_path(run_root: Path) -> Path:
    return _absolute_path(run_root) / PUBLICATION_GATE_FILENAME


def _publication_fence_path(run_root: Path) -> Path:
    return _absolute_path(run_root) / PUBLICATION_FENCE_FILENAME


def _open_publication_gate(
    run_root: Path, *, exclusive: bool
) -> int:
    """Open and non-blockingly lock the durable run-root publication gate."""

    root = _absolute_path(run_root)
    _require_safe_ancestors(
        root,
        include_leaf=True,
        allow_missing_leaf=False,
        context="publication gate run root",
    )
    path = _publication_gate_path(root)
    if not os.path.lexists(path):
        try:
            _write_bytes_exclusive(path, PUBLICATION_GATE_BYTES, root=root)
        except FileExistsError:
            pass
    before, descriptor = _regular_identity(
        path,
        root=root,
        readonly=True,
        context="publication gate",
    )
    try:
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 4096)
            if not chunk:
                break
            chunks.append(chunk)
        if b"".join(chunks) != PUBLICATION_GATE_BYTES:
            raise AuthorRecipeError("publication gate has invalid content")
        os.lseek(descriptor, 0, os.SEEK_SET)
    except BaseException:
        os.close(descriptor)
        raise
    operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    try:
        fcntl.flock(descriptor, operation | fcntl.LOCK_NB)
    except BlockingIOError as error:
        os.close(descriptor)
        raise ClaimUnavailable(
            "run root is fenced for publication"
            if not exclusive
            else "run root still has active workers or a publisher"
        ) from error
    visible = os.lstat(path)
    opened = os.fstat(descriptor)
    if (
        (visible.st_dev, visible.st_ino) != (before.st_dev, before.st_ino)
        or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
        or stat.S_ISLNK(visible.st_mode)
        or not stat.S_ISREG(visible.st_mode)
    ):
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
        raise AuthorRecipeError("publication gate changed while locking")
    return descriptor


def _validate_publication_fence_document(
    path: Path,
    *,
    run_root: Path,
    plan: Mapping[str, Any],
    expected_nonce: str | None = None,
    expected_owner: Mapping[str, Any] | None = None,
    expected_identity: tuple[int, int] | None = None,
    allow_plan_mismatch: bool = False,
) -> dict[str, Any]:
    root = _absolute_path(run_root)
    payload, fingerprint = _read_regular_snapshot(
        path,
        root=root,
        readonly=True,
        context="publication fence",
        expected_identity=expected_identity,
    )
    value = _decode_json_object(payload, context=str(path))
    _require_exact_keys(
        value,
        {
            "schema",
            "created_at",
            "plan_sha256",
            "nonce",
            "owner",
            "destination",
        },
        "publication fence",
    )
    _validate_timestamp(value["created_at"], "publication fence")
    _validate_owner_mapping(value["owner"], "publication fence")
    if (
        value["schema"] != PUBLICATION_FENCE_SCHEMA
        or not isinstance(value["plan_sha256"], str)
        or not HEX_64_RE.fullmatch(value["plan_sha256"])
        or (
            not allow_plan_mismatch
            and value["plan_sha256"] != plan["plan_sha256"]
        )
        or not isinstance(value["nonce"], str)
        or not re.fullmatch(r"[0-9a-f]{32}", value["nonce"])
        or not isinstance(value["destination"], str)
        or not value["destination"]
        or value["destination"]
        != str(_absolute_path(value["destination"]))
    ):
        raise AuthorRecipeError("publication fence identity is invalid")
    observed_identity = (fingerprint[0], fingerprint[1])
    if (
        (
            expected_identity is not None
            and observed_identity != tuple(map(int, expected_identity))
        )
        or (
            expected_nonce is not None
            and value["nonce"] != expected_nonce
        )
        or (
            expected_owner is not None
            and value["owner"] != dict(expected_owner)
        )
        or payload != _canonical_bytes(value) + b"\n"
    ):
        raise AuthorRecipeError("publication fence changed or is not canonical")
    return value


def _recover_stale_publication_fence(
    run_root: Path, plan: Mapping[str, Any]
) -> tuple[Path, ...]:
    """Recover a crash-left fence only while the caller owns the gate lock."""

    root = _absolute_path(run_root)
    path = _publication_fence_path(root)
    if not os.path.lexists(path):
        return ()
    fence_stat = _anchored_lstat(path)
    fence_identity = (int(fence_stat.st_dev), int(fence_stat.st_ino))
    minimally_loaded: Mapping[str, Any] = {}
    try:
        raw, _ = _read_regular_snapshot(
            path,
            root=root,
            readonly=True,
            context="publication fence recovery",
            expected_identity=fence_identity,
        )
        minimally_loaded = _decode_json_object(raw, context=str(path))
        value = _validate_publication_fence_document(
            path,
            run_root=root,
            plan=plan,
            expected_identity=fence_identity,
            allow_plan_mismatch=True,
        )
    except Exception:
        if _claim_is_live(minimally_loaded):
            raise ClaimUnavailable(
                "run root has a live but invalid publication fence"
            )
        reason = "invalid"
    else:
        if _claim_is_live(value):
            raise ClaimUnavailable("run root has a live publication fence")
        reason = "stale"
    try:
        return (
            _quarantine(
                root,
                path,
                category="publication-fences",
                reason=reason,
                expected_identity=fence_identity,
            ),
        )
    except FileNotFoundError:
        return ()


def _acquire_worker_publication_gate(
    run_root: Path, plan: Mapping[str, Any]
) -> int:
    descriptor = _open_publication_gate(run_root, exclusive=False)
    try:
        _recover_stale_publication_fence(run_root, plan)
        if os.path.lexists(_publication_fence_path(run_root)):
            raise ClaimUnavailable("run root is fenced for publication")
        return descriptor
    except BaseException:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
        raise


def acquire_publication_fence(
    run_root: Path,
    plan: Mapping[str, Any],
    *,
    destination: Path,
) -> PublicationFence:
    """Fence the run root against workers for the complete publish window."""

    root = _absolute_path(run_root)
    descriptor = _open_publication_gate(root, exclusive=True)
    try:
        _recover_stale_publication_fence(root, plan)
        path = _publication_fence_path(root)
        owner = _process_identity()
        nonce = uuid.uuid4().hex
        payload = {
            "schema": PUBLICATION_FENCE_SCHEMA,
            "created_at": _utc_now(),
            "plan_sha256": plan["plan_sha256"],
            "nonce": nonce,
            "owner": owner,
            "destination": str(_absolute_path(destination)),
        }
        _write_json_exclusive(path, payload, root=root)
        observed = os.lstat(path)
        fence = PublicationFence(
            run_root=root,
            path=path,
            path_identity=(int(observed.st_dev), int(observed.st_ino)),
            plan_sha256=str(plan["plan_sha256"]),
            nonce=nonce,
            owner=owner,
            destination=str(_absolute_path(destination)),
            gate_descriptor=descriptor,
        )
        validate_publication_fence(fence, plan)
        return fence
    except BaseException:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
        raise


def validate_publication_fence(
    fence: PublicationFence, plan: Mapping[str, Any]
) -> dict[str, Any]:
    root = _absolute_path(fence.run_root)
    if (
        fence.plan_sha256 != plan["plan_sha256"]
        or _absolute_path(fence.path) != _publication_fence_path(root)
        or not _claim_is_live(
            {"created_at": _utc_now(), "owner": dict(fence.owner)}
        )
    ):
        raise AuthorRecipeError("publication fence ownership is not live")
    value = _validate_publication_fence_document(
        fence.path,
        run_root=root,
        plan=plan,
        expected_nonce=fence.nonce,
        expected_owner=fence.owner,
        expected_identity=fence.path_identity,
    )
    if (
        fence.destination != str(_absolute_path(fence.destination))
        or value["destination"] != fence.destination
    ):
        raise AuthorRecipeError(
            "publication fence destination identity changed"
        )
    return value


def release_publication_fence(
    fence: PublicationFence, plan: Mapping[str, Any]
) -> None:
    root = _absolute_path(fence.run_root)
    try:
        validate_publication_fence(fence, plan)
        _quarantine(
            root,
            fence.path,
            category="publication-fences",
            reason="released",
            expected_identity=fence.path_identity,
        )
    finally:
        try:
            fcntl.flock(fence.gate_descriptor, fcntl.LOCK_UN)
        finally:
            os.close(fence.gate_descriptor)


def _quarantine_partials(run_root: Path, job: Job) -> tuple[Path, ...]:
    run = _absolute_path(run_root)
    root = run / "partials"
    if not os.path.lexists(root):
        return ()
    _require_safe_ancestors(
        root,
        root=run,
        include_leaf=True,
        allow_missing_leaf=False,
        context="partial root",
    )
    root_stat = os.lstat(root)
    if not stat.S_ISDIR(root_stat.st_mode) or stat.S_ISLNK(root_stat.st_mode):
        raise AuthorRecipeError("partial root is aliased or non-directory")
    recovered: list[Path] = []
    prefix = f"{job.job_id}."
    for entry in os.scandir(root):
        if not entry.name.startswith(prefix) or not entry.name.endswith(".partial"):
            continue
        path = Path(entry.path)
        observed = entry.stat(follow_symlinks=False)
        try:
            recovered.append(
                _quarantine(
                    run_root,
                    path,
                    category="partials",
                    reason="stale",
                    expected_identity=(
                        int(observed.st_dev),
                        int(observed.st_ino),
                    ),
                )
            )
        except FileNotFoundError:
            continue
    return tuple(recovered)


def _resource_mapping(
    value: full_grid.ResourceStatus | full_grid.GPUStatus,
) -> dict[str, Any]:
    if isinstance(value, full_grid.ResourceStatus):
        if not bool(value.gpu.get("safe")):
            raise GPUUnavailable(str(value.gpu.get("reason", value.reason)))
        if not bool(value.disk.get("safe")):
            raise DiskUnavailable(str(value.disk.get("reason", value.reason)))
        result = json.loads(_canonical_bytes(value.as_dict()))
        _validate_resource_guard(result, "claim resource guard")
        return result
    raise TypeError(
        "author-recipe claims require a combined GPU and disk ResourceStatus"
    )


def _validate_claim_payload(
    run_root: Path,
    plan: Mapping[str, Any],
    claim: Claim,
) -> dict[str, Any]:
    root = _absolute_path(run_root)
    expected_path = _claim_path(root, claim.job)
    if (
        _absolute_path(claim.run_root) != root
        or _absolute_path(claim.path) != expected_path
    ):
        raise AuthorRecipeError("claim path is outside its exact run-root identity")
    payload, _ = _read_regular_snapshot(
        expected_path,
        root=root,
        readonly=True,
        context="job claim",
        expected_identity=claim.path_identity,
    )
    observed = _decode_json_object(payload, context=str(expected_path))
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
    _validate_timestamp(observed["created_at"], "job claim")
    _validate_job_identity_mapping(observed["job"], claim.job, "job claim")
    expected = {
        "schema": CLAIM_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "job_id": claim.job.job_id,
        "nonce": claim.nonce,
        "owner": dict(claim.owner),
        "resource_guard": dict(claim.resource_guard),
    }
    for key, value in expected.items():
        if observed.get(key) != value:
            raise AuthorRecipeError(f"job claim has invalid {key}")
    _validate_owner_mapping(observed["owner"], "job claim")
    _validate_resource_guard(observed["resource_guard"], "job claim resource guard")
    if payload != _canonical_bytes(observed) + b"\n":
        raise AuthorRecipeError("job claim JSON is not canonical")
    return observed


def _validate_released_claim_tombstone(
    path: Path,
    *,
    run_root: Path,
    plan: Mapping[str, Any],
    job: Job,
    expected_identity: tuple[int, int],
) -> dict[str, Any]:
    prefix = f".{job.job_id}.json.released-"
    match = re.fullmatch(
        re.escape(prefix) + r"([0-9a-f]{32})-([0-9a-f]{32})",
        path.name,
    )
    if match is None:
        raise AuthorRecipeError("released claim tombstone name is invalid")
    payload, _ = _read_regular_snapshot(
        path,
        root=run_root,
        readonly=True,
        context="released claim tombstone",
        expected_identity=expected_identity,
    )
    value = _decode_json_object(payload, context=str(path))
    _require_exact_keys(
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
        },
        "released claim tombstone",
    )
    _validate_timestamp(value["created_at"], "released claim tombstone")
    _validate_job_identity_mapping(
        value["job"], job, "released claim tombstone"
    )
    _validate_owner_mapping(value["owner"], "released claim tombstone")
    _validate_resource_guard(
        value["resource_guard"], "released claim tombstone resource guard"
    )
    if (
        value["schema"] != CLAIM_SCHEMA
        or value["plan_sha256"] != plan["plan_sha256"]
        or value["job_id"] != job.job_id
        or value["nonce"] != match.group(1)
        or payload != _canonical_bytes(value) + b"\n"
    ):
        raise AuthorRecipeError(
            "released claim tombstone identity is invalid"
        )
    return value


def _recover_released_claim_tombstones(
    run_root: Path,
    plan: Mapping[str, Any],
    job: Job,
) -> tuple[Path, ...]:
    root = _absolute_path(run_root)
    claim_parent = _claim_path(root, job).parent
    if not os.path.lexists(claim_parent):
        return ()
    _require_safe_ancestors(
        claim_parent,
        root=root,
        include_leaf=True,
        allow_missing_leaf=False,
        context="claim tombstone directory",
    )
    parent_stat = os.lstat(claim_parent)
    if stat.S_ISLNK(parent_stat.st_mode) or not stat.S_ISDIR(
        parent_stat.st_mode
    ):
        raise AuthorRecipeError("claim tombstone directory is unsafe")
    prefix = f".{job.job_id}.json.released-"
    recovered: list[Path] = []
    for entry in os.scandir(claim_parent):
        if not entry.name.startswith(prefix):
            continue
        path = _absolute_path(entry.path)
        observed = entry.stat(follow_symlinks=False)
        identity = (int(observed.st_dev), int(observed.st_ino))
        try:
            if (
                stat.S_ISREG(observed.st_mode)
                and observed.st_nlink == 1
                and not (observed.st_mode & 0o222)
            ):
                _validate_released_claim_tombstone(
                    path,
                    run_root=root,
                    plan=plan,
                    job=job,
                    expected_identity=identity,
                )
                reason = "released"
            else:
                reason = "invalid-released"
        except Exception:
            reason = "invalid-released"
        try:
            recovered.append(
                _quarantine(
                    root,
                    path,
                    category="claims",
                    reason=reason,
                    expected_identity=identity,
                )
            )
        except FileNotFoundError:
            continue
    return tuple(recovered)


def _acquire_claim_with_gate_held(
    run_root: Path,
    plan: Mapping[str, Any],
    job: Job,
    *,
    publication_gate_descriptor: int,
    recover_stale: bool = True,
    foreign_claim_timeout_seconds: float = 0.0,
    before_claim: Callable[
        [], full_grid.ResourceStatus | full_grid.GPUStatus
    ]
    | None = None,
) -> Claim:
    run_root = _absolute_path(run_root)
    _require_safe_ancestors(
        run_root,
        include_leaf=True,
        allow_missing_leaf=False,
        context="run root",
    )
    _recover_released_claim_tombstones(run_root, plan, job)
    output = _record_directory(run_root, job)
    if os.path.lexists(output):
        output_stat = _anchored_lstat(output)
        output_identity = (
            int(output_stat.st_dev),
            int(output_stat.st_ino),
        )
        try:
            validate_completion(run_root, plan, job)
        except Exception:
            if not recover_stale:
                raise ClaimUnavailable(f"{job.job_id} has a corrupt completion")
            _quarantine(
                run_root,
                output,
                category="records",
                reason="corrupt",
                expected_identity=output_identity,
            )
        else:
            raise ClaimUnavailable(f"{job.job_id} is already complete")
    path = _claim_path(run_root, job)
    if os.path.lexists(path):
        claim_stat = _anchored_lstat(path)
        claim_identity = (
            int(claim_stat.st_dev),
            int(claim_stat.st_ino),
        )
        try:
            raw, _ = _read_regular_snapshot(
                path,
                root=run_root,
                readonly=True,
                context="stale claim recovery",
                expected_identity=claim_identity,
            )
            observed = _decode_json_object(raw, context=str(path))
        except Exception:
            observed = {}
        if _claim_is_live(
            observed,
            foreign_claim_timeout_seconds=foreign_claim_timeout_seconds,
        ):
            raise ClaimUnavailable(f"{job.job_id} has a live claim")
        if not recover_stale:
            raise ClaimUnavailable(f"{job.job_id} has a stale claim")
        try:
            _quarantine(
                run_root,
                path,
                category="claims",
                reason="stale",
                expected_identity=claim_identity,
            )
        except FileNotFoundError as error:
            raise ClaimUnavailable("lost stale-claim recovery race") from error
        _quarantine_partials(run_root, job)
    if before_claim is None:
        raise AuthorRecipeError(
            "every author-recipe claim requires cooperative GPU and disk probes"
        )
    resource_guard = _resource_mapping(before_claim())
    owner = _process_identity()
    nonce = uuid.uuid4().hex
    payload = {
        "schema": CLAIM_SCHEMA,
        "created_at": _utc_now(),
        "plan_sha256": plan["plan_sha256"],
        "job_id": job.job_id,
        "job": job.identity(),
        "nonce": nonce,
        "owner": owner,
        "resource_guard": resource_guard,
    }
    try:
        _write_json_exclusive(path, payload, root=run_root)
    except FileExistsError as error:
        raise ClaimUnavailable(f"{job.job_id} claim race lost") from error
    claim_stat = os.lstat(path)
    claim = Claim(
        job=job,
        run_root=run_root,
        path=path,
        path_identity=(int(claim_stat.st_dev), int(claim_stat.st_ino)),
        plan_sha256=str(plan["plan_sha256"]),
        nonce=nonce,
        owner=owner,
        resource_guard=resource_guard,
        publication_gate_descriptor=publication_gate_descriptor,
    )
    _validate_claim_payload(run_root, plan, claim)
    return claim


def acquire_claim(
    run_root: Path,
    plan: Mapping[str, Any],
    job: Job,
    *,
    recover_stale: bool = True,
    foreign_claim_timeout_seconds: float = 0.0,
    before_claim: Callable[
        [], full_grid.ResourceStatus | full_grid.GPUStatus
    ]
    | None = None,
) -> Claim:
    """Acquire a claim while holding a shared run-root publication gate."""

    root = _absolute_path(run_root)
    gate_descriptor = _acquire_worker_publication_gate(root, plan)
    try:
        return _acquire_claim_with_gate_held(
            root,
            plan,
            job,
            publication_gate_descriptor=gate_descriptor,
            recover_stale=recover_stale,
            foreign_claim_timeout_seconds=foreign_claim_timeout_seconds,
            before_claim=before_claim,
        )
    except BaseException:
        try:
            fcntl.flock(gate_descriptor, fcntl.LOCK_UN)
        finally:
            os.close(gate_descriptor)
        raise


def _release_claim_file(claim: Claim) -> None:
    root = _absolute_path(claim.run_root)
    expected = _claim_path(root, claim.job)
    output = _record_directory(root, claim.job)

    def invalidate_owned_record(*, reason: str) -> None:
        owned = claim.publication_state.record_identity
        if owned is None:
            return
        if not os.path.lexists(output):
            claim.publication_state.record_identity = None
            return
        try:
            _quarantine(
                root,
                output,
                category="records",
                reason=reason,
                expected_identity=owned,
            )
        except ClaimUnavailable as error:
            claim.publication_state.record_identity = None
            raise AuthorRecipeError(
                "owned record was replaced during claim release; "
                "the replacement was preserved"
            ) from error
        claim.publication_state.record_identity = None

    if _absolute_path(claim.path) != expected:
        raise AuthorRecipeError("refusing to release a claim outside its run root")
    if not os.path.lexists(expected):
        if claim.publication_state.record_identity is not None:
            invalidate_owned_record(reason="claimlostrelease")
            raise AuthorRecipeError(
                "owned claim vanished after publication; record quarantined"
            )
        return
    try:
        observed = _validate_claim_payload(
            root, {"plan_sha256": claim.plan_sha256}, claim
        )
    except BaseException:
        invalidate_owned_record(reason="claimreplacedrelease")
        raise
    if observed.get("nonce") != claim.nonce or observed.get("owner") != dict(
        claim.owner
    ):
        invalidate_owned_record(reason="claimreplacedrelease")
        raise AuthorRecipeError(
            "refusing to release a claim owned by another process"
        )
    claim_identity = tuple(map(int, claim.path_identity))
    tombstone = expected.parent / (
        f".{expected.name}.released-{claim.nonce}-{uuid.uuid4().hex}"
    )
    if os.path.lexists(tombstone):
        raise FileExistsError(tombstone)
    try:
        _rename_leaf_nofollow(
            expected,
            tombstone,
            root=root,
            expected_source_identity=claim_identity,
        )
    except BaseException:
        if not _path_binds_identity(tombstone, claim_identity):
            invalidate_owned_record(reason="claimreplacedrelease")
            raise
    moved = _anchored_lstat(tombstone)
    if (
        (int(moved.st_dev), int(moved.st_ino)) != claim_identity
        or stat.S_ISLNK(moved.st_mode)
        or not stat.S_ISREG(moved.st_mode)
    ):
        invalidate_owned_record(reason="claimreplacedrelease")
        raise AuthorRecipeError(
            "claim path was swapped during release; moved object was preserved"
        )
    # The exact claim is no longer canonical. A later power cut may leave the
    # tombstone, but it cannot invalidate an already durable owned record.
    claim.publication_state.record_identity = None
    _validate_released_claim_tombstone(
        tombstone,
        run_root=root,
        plan={"plan_sha256": claim.plan_sha256},
        job=claim.job,
        expected_identity=claim_identity,
    )
    _quarantine(
        root,
        tombstone,
        category="claims",
        reason="released",
        expected_identity=claim_identity,
    )
    _fsync_directory(expected.parent, root=root)


def release_claim(claim: Claim) -> None:
    """Release the exact claim and always drop its shared publication gate."""

    descriptor = claim.publication_gate_descriptor
    try:
        os.fstat(descriptor)
    except OSError:
        plan = load_plan(claim.run_root)
        if plan["plan_sha256"] != claim.plan_sha256:
            raise AuthorRecipeError(
                "cannot reacquire gate for a claim from another plan"
            )
        descriptor = _acquire_worker_publication_gate(
            claim.run_root, plan
        )
    try:
        _release_claim_file(claim)
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _recover_completed_job_auxiliary_state_with_gate_held(
    run_root: Path,
    plan: Mapping[str, Any],
    job: Job,
    *,
    foreign_claim_timeout_seconds: float = 0.0,
) -> tuple[Path, ...]:
    """Recover only stale claim/partial state around a valid durable result.

    A power cut may occur after the final directory rename but before claim
    release.  The completion is fully revalidated before any auxiliary state
    is moved.  A live claim is never reclaimed.
    """

    validate_completion(run_root, plan, job)
    recovered: list[Path] = list(
        _recover_released_claim_tombstones(run_root, plan, job)
    )
    path = _claim_path(run_root, job)
    if os.path.lexists(path):
        claim_stat = _anchored_lstat(path)
        claim_identity = (
            int(claim_stat.st_dev),
            int(claim_stat.st_ino),
        )
        try:
            raw, _ = _read_regular_snapshot(
                path,
                root=run_root,
                readonly=True,
                context="completed claim recovery",
                expected_identity=claim_identity,
            )
            value = _decode_json_object(raw, context=str(path))
        except Exception:
            value = {}
        if _claim_is_live(
            value,
            foreign_claim_timeout_seconds=foreign_claim_timeout_seconds,
        ):
            raise ClaimUnavailable(f"{job.job_id} has a live post-publication claim")
        try:
            recovered.append(
                _quarantine(
                    run_root,
                    path,
                    category="claims",
                    reason="completed-stale",
                    expected_identity=claim_identity,
                )
            )
        except FileNotFoundError as error:
            raise ClaimUnavailable(
                "lost completed-claim recovery race"
            ) from error
    recovered.extend(_quarantine_partials(run_root, job))
    return tuple(recovered)


def recover_completed_job_auxiliary_state(
    run_root: Path,
    plan: Mapping[str, Any],
    job: Job,
    *,
    foreign_claim_timeout_seconds: float = 0.0,
) -> tuple[Path, ...]:
    descriptor = _acquire_worker_publication_gate(run_root, plan)
    try:
        return _recover_completed_job_auxiliary_state_with_gate_held(
            run_root,
            plan,
            job,
            foreign_claim_timeout_seconds=foreign_claim_timeout_seconds,
        )
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _planned_cache(plan: Mapping[str, Any], job: Job) -> Mapping[str, Any]:
    try:
        result = plan["cache_identity"][_subject_key(job.dataset, job.subject)]
    except (KeyError, TypeError) as error:
        raise AuthorRecipeError("plan has no cache identity for job") from error
    return result


def _planned_split(plan: Mapping[str, Any], job: Job) -> Mapping[str, Any]:
    try:
        result = plan["split_identity"][
            _split_key(job.dataset, job.subject, job.fold)
        ]
    except (KeyError, TypeError) as error:
        raise AuthorRecipeError("plan has no split identity for job") from error
    return result


def _split_metadata(identity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "trial_count": int(identity["trial_count"]),
        "partitions": copy.deepcopy(dict(identity["partitions"])),
    }


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str] | frozenset[str],
    context: str,
) -> None:
    observed = set(value)
    if observed != set(expected):
        raise AuthorRecipeError(
            f"{context} violates its exact schema; "
            f"missing={sorted(set(expected) - observed)}, "
            f"extra={sorted(observed - set(expected))}"
        )


def _validate_timestamp(value: Any, context: str) -> None:
    if not isinstance(value, str):
        raise AuthorRecipeError(f"{context} timestamp is not a string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise AuthorRecipeError(f"{context} timestamp is invalid") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise AuthorRecipeError(f"{context} timestamp is not timezone-aware")


def _validate_job_identity_mapping(
    value: Any, job: Job, context: str
) -> None:
    if not isinstance(value, Mapping):
        raise AuthorRecipeError(f"{context} job identity is absent")
    _require_exact_keys(
        value,
        {"dataset", "reference", "subject", "fold", "seed"},
        f"{context} job identity",
    )
    if dict(value) != job.identity():
        raise AuthorRecipeError(f"{context} job identity differs from the plan")
    for key in ("subject", "fold", "seed"):
        if isinstance(value[key], bool) or not isinstance(value[key], int):
            raise AuthorRecipeError(f"{context} job identity has invalid {key}")


def _validate_split_metadata(
    value: Any, expected: Mapping[str, Any], context: str
) -> None:
    if not isinstance(value, Mapping):
        raise AuthorRecipeError(f"{context} split metadata is absent")
    _require_exact_keys(value, {"trial_count", "partitions"}, f"{context} split")
    trial_count = value["trial_count"]
    if (
        isinstance(trial_count, bool)
        or not isinstance(trial_count, int)
        or trial_count <= 0
    ):
        raise AuthorRecipeError(f"{context} split trial_count is invalid")
    partitions = value["partitions"]
    if not isinstance(partitions, Mapping):
        raise AuthorRecipeError(f"{context} split partitions are absent")
    _require_exact_keys(
        partitions,
        {"train", "validation", "source", "test"},
        f"{context} split partitions",
    )
    for name in ("train", "validation", "source", "test"):
        partition = partitions[name]
        if not isinstance(partition, Mapping):
            raise AuthorRecipeError(f"{context} split {name} is absent")
        _require_exact_keys(
            partition, {"count", "rows_sha256"}, f"{context} split {name}"
        )
        count = partition["count"]
        digest = partition["rows_sha256"]
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
            or not isinstance(digest, str)
            or not HEX_64_RE.fullmatch(digest)
        ):
            raise AuthorRecipeError(f"{context} split {name} is invalid")
    if value != expected:
        raise AuthorRecipeError(f"{context} split identity differs from the plan")


def _validate_resource_guard(value: Any, context: str) -> None:
    if not isinstance(value, Mapping):
        raise AuthorRecipeError(f"{context} resource guard is absent")
    _require_exact_keys(value, {"safe", "gpu", "disk", "reason"}, context)
    if value["safe"] is not True or not isinstance(value["reason"], str):
        raise AuthorRecipeError(f"{context} combined resource status is unsafe")
    gpu = value["gpu"]
    disk = value["disk"]
    if not isinstance(gpu, Mapping) or not isinstance(disk, Mapping):
        raise AuthorRecipeError(f"{context} GPU/disk guards are absent")
    _require_exact_keys(
        gpu,
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
        f"{context} GPU",
    )
    if (
        gpu["safe"] is not True
        or not isinstance(gpu["gpu"], str)
        or not gpu["gpu"]
        or not isinstance(gpu["reason"], str)
    ):
        raise AuthorRecipeError(f"{context} GPU guard is invalid")
    for name in (
        "utilization_percent",
        "memory_used_mib",
        "own_compute_memory_mib",
    ):
        raw = gpu[name]
        if raw is None and name != "own_compute_memory_mib":
            continue
        if (
            isinstance(raw, bool)
            or not isinstance(raw, (int, float))
            or not math.isfinite(float(raw))
            or float(raw) < 0.0
        ):
            raise AuthorRecipeError(f"{context} GPU {name} is invalid")
    for name in ("gpu_uuid", "pci_bus_id", "name"):
        if gpu[name] is not None and not isinstance(gpu[name], str):
            raise AuthorRecipeError(f"{context} GPU {name} is invalid")
    processes = gpu["foreign_processes"]
    if not isinstance(processes, list):
        raise AuthorRecipeError(f"{context} GPU process roster is invalid")
    for index, process in enumerate(processes):
        if not isinstance(process, Mapping):
            raise AuthorRecipeError(f"{context} GPU process {index} is invalid")
        _require_exact_keys(
            process,
            {"pid", "process_name", "used_gpu_memory_mib"},
            f"{context} GPU process {index}",
        )
        if (
            isinstance(process["pid"], bool)
            or not isinstance(process["pid"], int)
            or process["pid"] <= 0
            or not isinstance(process["process_name"], str)
            or isinstance(process["used_gpu_memory_mib"], bool)
            or not isinstance(process["used_gpu_memory_mib"], (int, float))
            or not math.isfinite(float(process["used_gpu_memory_mib"]))
            or float(process["used_gpu_memory_mib"]) < 0.0
        ):
            raise AuthorRecipeError(f"{context} GPU process {index} is invalid")
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
        f"{context} disk",
    )
    if (
        disk["safe"] is not True
        or not isinstance(disk["path"], str)
        or not disk["path"]
        or disk["path"] != str(_absolute_path(disk["path"]))
        or not isinstance(disk["reason"], str)
        or "allow-low-disk" in disk["reason"]
    ):
        raise AuthorRecipeError(f"{context} disk guard is invalid")
    for name in ("free_bytes", "total_bytes"):
        raw = disk[name]
        if (
            isinstance(raw, bool)
            or not isinstance(raw, int)
            or raw < 0
        ):
            raise AuthorRecipeError(f"{context} disk {name} is invalid")
    if disk["total_bytes"] < disk["free_bytes"]:
        raise AuthorRecipeError(f"{context} disk capacity is inconsistent")
    for name in ("free_gib", "minimum_free_gib"):
        raw = disk[name]
        if (
            isinstance(raw, bool)
            or not isinstance(raw, (int, float))
            or not math.isfinite(float(raw))
            or float(raw) < 0.0
        ):
            raise AuthorRecipeError(f"{context} disk {name} is invalid")
    free_gib = float(disk["free_gib"])
    minimum_free_gib = float(disk["minimum_free_gib"])
    free_bytes = int(disk["free_bytes"])
    frozen_floor_bytes = math.ceil(DEFAULT_MIN_FREE_GIB * (1024**3))
    if (
        minimum_free_gib < DEFAULT_MIN_FREE_GIB
        or free_gib < DEFAULT_MIN_FREE_GIB
        or free_bytes < frozen_floor_bytes
        or free_gib < minimum_free_gib
        or not math.isclose(
            free_gib,
            free_bytes / float(1024**3),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        raise AuthorRecipeError(f"{context} disk guard is below its floor")


def _recursive_forbidden_keys(
    value: Any, path: str = "record"
) -> list[str]:
    violations: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_SCORE_BLIND_KEYS:
                violations.append(f"{path}.{key}")
            violations.extend(
                _recursive_forbidden_keys(child, f"{path}.{key}")
            )
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            violations.extend(
                _recursive_forbidden_keys(child, f"{path}[{index}]")
            )
    return violations


def _expected_config(plan: Mapping[str, Any], job: Job) -> dict[str, Any]:
    if job.reference == "reference.tcformer":
        result = copy.deepcopy(
            plan["recipe_contracts"][job.reference]["per_dataset_config"][
                job.dataset
            ]
        )
    elif job.reference == "reference.fbcnet":
        result = copy.deepcopy(plan["recipe_contracts"][job.reference]["config"])
    else:
        raise AuthorRecipeError(f"unknown author reference {job.reference}")
    result["seed"] = job.seed
    result["device"] = "cuda"
    return result


def _expected_recipe_metadata(
    plan: Mapping[str, Any], job: Job
) -> dict[str, Any]:
    config = _expected_config(plan, job)
    if job.reference == "reference.tcformer":
        value = tcformer_reference_metadata(TCFormerReferenceConfig(**config))
    else:
        value = fbcnet_reference_metadata(FBCNetReferenceConfig(**config))
    # Recipe metadata contains tuples. Normalize it before both pre-publication
    # validation and post-JSON validation so an immutable round trip cannot
    # create a false provenance drift.
    return json.loads(_canonical_bytes(value))


def _expected_protocol(reference: str) -> dict[str, Any]:
    if reference == "reference.tcformer":
        return {
            "duration_selection": (
                "plan_prespecified_dataset_horizon_no_outcome_selection"
            ),
            "scaler_rows": "train_plus_validation",
            "optimization_rows": "train_plus_validation",
            "validation_outcomes_used_for_selection": False,
            "test_use": "single_predict_probabilities_call_only",
            "test_performance_computed": False,
        }
    if reference == "reference.fbcnet":
        return {
            "selection_scaler_rows": "train_only",
            "stage1_optimization_rows": "train_only",
            "stage1_selection_rows": "validation_only",
            "stage2_optimization_rows": "train_plus_validation",
            "stage2_stopping_rows": (
                "original_validation_subset_released_loss_threshold"
            ),
            "test_scaler_rows": "train_only_same_transform_as_fitted_model",
            "test_use": "single_predict_probabilities_call_only",
            "test_performance_computed": False,
        }
    raise AuthorRecipeError(f"unknown author reference {reference}")


def _validate_job_metadata(
    plan: Mapping[str, Any], job: Job, metadata: Mapping[str, Any]
) -> None:
    _require_exact_keys(
        metadata,
        {
            "track",
            "cache_array_sha256",
            "channels",
            "split",
            "scalers",
            "recipe",
            "fit",
            "protocol",
            "timing_seconds",
            "cuda_peak_memory_bytes",
            "runtime",
        },
        "executor metadata",
    )
    if metadata.get("track") != TRACK:
        raise AuthorRecipeError("executor metadata has the wrong result track")
    if metadata.get("cache_array_sha256") != _planned_cache(plan, job).get(
        "array_sha256"
    ):
        raise AuthorRecipeError("executor metadata cache identity differs from plan")
    _validate_split_metadata(
        metadata.get("split"),
        _split_metadata(_planned_split(plan, job)),
        "executor metadata",
    )
    channels = metadata.get("channels")
    if (
        not isinstance(channels, list)
        or not channels
        or any(not isinstance(value, str) or not value for value in channels)
        or len(set(channels)) != len(channels)
    ):
        raise AuthorRecipeError("executor metadata has invalid channels")
    scalers = metadata.get("scalers")
    expected_scaler_keys = (
        {"source_mean_sha256", "source_std_sha256"}
        if job.reference == "reference.tcformer"
        else {"selection_mean_sha256", "selection_std_sha256"}
    )
    if (
        not isinstance(scalers, Mapping)
        or set(scalers) != expected_scaler_keys
        or any(
            not isinstance(value, str) or not HEX_64_RE.fullmatch(value)
            for value in scalers.values()
        )
    ):
        raise AuthorRecipeError("executor metadata has invalid scaler provenance")
    if metadata.get("recipe") != _expected_recipe_metadata(plan, job):
        raise AuthorRecipeError("executor metadata recipe differs from plan")
    if metadata.get("protocol") != _expected_protocol(job.reference):
        raise AuthorRecipeError("executor metadata has invalid source/test protocol")
    fit = metadata.get("fit")
    if not isinstance(fit, Mapping):
        raise AuthorRecipeError("executor metadata has no fit contract")
    common_hashes = (
        "initial_state_sha256",
        "final_state_sha256",
        "final_optimizer_state_sha256",
    )
    if any(
        not isinstance(fit.get(name), str)
        or not HEX_64_RE.fullmatch(str(fit.get(name)))
        for name in common_hashes
    ):
        raise AuthorRecipeError("executor metadata has invalid state hashes")
    if fit.get("seed_installed_before_construction") is not True:
        raise AuthorRecipeError("executor metadata does not prove construction seeding")
    parameter_count = fit.get("parameter_count")
    source_count = fit.get("source_count")
    if (
        isinstance(parameter_count, bool)
        or not isinstance(parameter_count, int)
        or isinstance(source_count, bool)
        or not isinstance(source_count, int)
    ):
        raise AuthorRecipeError("executor metadata has invalid fit counters")
    expected_source_count = int(
        _planned_split(plan, job)["partitions"]["source"]["count"]
    )
    if parameter_count <= 0 or source_count != expected_source_count:
        raise AuthorRecipeError("executor metadata fit counters violate the plan")
    if job.reference == "reference.tcformer":
        if set(fit) != {
            "seed_installed_before_construction",
            "initial_state_sha256",
            "final_state_sha256",
            "final_optimizer_state_sha256",
            "epochs_run",
            "source_count",
            "parameter_count",
        }:
            raise AuthorRecipeError("TCFormer fit metadata has an invalid exact schema")
        if (
            isinstance(fit["epochs_run"], bool)
            or not isinstance(fit["epochs_run"], int)
            or fit["epochs_run"] != int(_expected_config(plan, job)["epochs"])
        ):
            raise AuthorRecipeError("TCFormer fixed horizon differs from the plan")
    else:
        expected_keys = {
            "seed_installed_before_construction",
            "initial_state_sha256",
            "final_state_sha256",
            "final_optimizer_state_sha256",
            "stage1_best_epoch",
            "stage1_stop",
            "stage1_epochs_run",
            "stage2_epochs_run",
            "stage2_stop",
            "stage1_train_loss_threshold",
            "stage1_threshold_origin",
            "stage1_threshold_origin_epoch",
            "stage1_best_model_state_sha256",
            "stage1_best_optimizer_state_sha256",
            "stage2_start_model_state_sha256",
            "stage2_start_optimizer_state_sha256",
            "source_count",
            "original_validation_count",
            "parameter_count",
        }
        if set(fit) != expected_keys:
            raise AuthorRecipeError("FBCNet fit metadata has an invalid exact schema")
        for name in (
            "stage1_best_model_state_sha256",
            "stage1_best_optimizer_state_sha256",
            "stage2_start_model_state_sha256",
            "stage2_start_optimizer_state_sha256",
        ):
            if (
                not isinstance(fit[name], str)
                or not HEX_64_RE.fullmatch(fit[name])
            ):
                raise AuthorRecipeError("FBCNet transition state hash is invalid")
        integer_fields = (
            "stage1_best_epoch",
            "stage1_epochs_run",
            "stage2_epochs_run",
            "stage1_threshold_origin_epoch",
            "original_validation_count",
        )
        if any(
            isinstance(fit[name], bool) or not isinstance(fit[name], int)
            for name in integer_fields
        ) or any(
            not isinstance(fit[name], str) or not fit[name]
            for name in ("stage1_stop", "stage2_stop")
        ):
            raise AuthorRecipeError("FBCNet fit counters/stops are invalid")
        threshold = fit["stage1_train_loss_threshold"]
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold))
            or float(threshold) < 0.0
            or fit["stage1_threshold_origin"]
            != "last_stage1_epoch_before_best_restore"
            or isinstance(fit["stage1_threshold_origin_epoch"], bool)
            or not isinstance(fit["stage1_threshold_origin_epoch"], int)
            or fit["stage1_threshold_origin_epoch"]
            != int(fit["stage1_epochs_run"]) - 1
        ):
            raise AuthorRecipeError(
                "FBCNet stage transition threshold provenance is invalid"
            )
        if (
            fit["stage1_best_model_state_sha256"]
            != fit["stage2_start_model_state_sha256"]
            or fit["stage1_best_optimizer_state_sha256"]
            != fit["stage2_start_optimizer_state_sha256"]
            or fit["stage1_best_epoch"] < 0
            or fit["stage1_best_epoch"] >= fit["stage1_epochs_run"]
            or int(fit["stage1_epochs_run"]) < 1
            or int(fit["stage1_epochs_run"])
            > int(_expected_config(plan, job)["stage1_max_epochs"])
            or int(fit["stage2_epochs_run"]) < 1
            or int(fit["stage2_epochs_run"])
            > int(_expected_config(plan, job)["stage2_max_epochs"])
            or int(fit["original_validation_count"])
            != int(
                _planned_split(plan, job)["partitions"]["validation"]["count"]
            )
        ):
            raise AuthorRecipeError("FBCNet transition/refit metadata is invalid")
    timing = metadata.get("timing_seconds")
    if not isinstance(timing, Mapping) or set(timing) != {
        "source_fit",
        "test_inference",
        "job_total",
    }:
        raise AuthorRecipeError("executor metadata has invalid timing")
    for name, raw in timing.items():
        if (
            isinstance(raw, bool)
            or not isinstance(raw, (int, float))
            or not math.isfinite(float(raw))
            or float(raw) < 0.0
        ):
            raise AuthorRecipeError(f"invalid timing field {name}")
    if float(timing["job_total"]) + 1e-9 < (
        float(timing["source_fit"]) + float(timing["test_inference"])
    ):
        raise AuthorRecipeError("executor job_total is shorter than its components")
    peak = metadata.get("cuda_peak_memory_bytes")
    if isinstance(peak, bool) or not isinstance(peak, int) or peak < 0:
        raise AuthorRecipeError("executor metadata has invalid peak memory")
    runtime = metadata.get("runtime")
    threads = int(plan["execution_config"]["worker_cpu_threads"])
    if not isinstance(runtime, Mapping):
        raise AuthorRecipeError("executor metadata runtime provenance is absent")
    _require_exact_keys(
        runtime,
        {
            "worker_cpu_threads",
            "worker_interop_threads",
            "thread_environment",
            "nvidia_driver_versions",
        },
        "executor metadata runtime",
    )
    if (
        runtime.get("worker_cpu_threads") != threads
        or runtime.get("worker_interop_threads") != 1
        or runtime.get("thread_environment")
        != {name: str(threads) for name in THREAD_ENVIRONMENT_VARIABLES}
        or runtime.get("nvidia_driver_versions")
        != plan["environment_identity"].get("nvidia_driver_versions")
    ):
        raise AuthorRecipeError("executor metadata has invalid runtime provenance")


def commit_job_output(
    run_root: Path,
    plan: Mapping[str, Any],
    claim: Claim,
    *,
    metadata: Mapping[str, Any],
    test_rows: np.ndarray,
    probabilities: np.ndarray,
    cache_root: Path | None = None,
    requested_gpu: str | None = None,
) -> Path:
    run_root = _absolute_path(run_root)
    job = claim.job
    claim_payload = _validate_claim_payload(run_root, plan, claim)
    guard = claim_payload["resource_guard"]
    _validate_resource_guard(guard, "claim resource guard")
    rows = np.asarray(test_rows)
    values = np.asarray(probabilities)
    split = _planned_split(plan, job)
    test_identity = split["partitions"]["test"]
    n_classes = int(plan["datasets"][job.dataset]["n_classes"])
    if (
        rows.dtype != np.dtype(np.int64)
        or rows.ndim != 1
        or len(rows) != int(test_identity["count"])
        or np.any(rows < 0)
        or np.any(rows >= int(split["trial_count"]))
        or len(np.unique(rows)) != len(rows)
        or _rows_sha256(rows) != test_identity["rows_sha256"]
        or values.dtype != np.dtype(np.float64)
        or values.shape != (len(rows), n_classes)
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(values > 1.0)
        or not np.allclose(
            values.sum(axis=1), 1.0, rtol=0.0, atol=1e-6
        )
    ):
        raise AuthorRecipeError("executor returned invalid prediction arrays")
    metadata_violations = _recursive_forbidden_keys(metadata, "metadata")
    if metadata_violations:
        raise AuthorRecipeError(
            "score-blind metadata contains forbidden fields: "
            f"{metadata_violations[:5]}"
        )
    _validate_job_metadata(plan, job, metadata)
    record = {
        "schema": RECORD_SCHEMA,
        "created_at": _utc_now(),
        "plan_sha256": plan["plan_sha256"],
        "track": TRACK,
        "common_recipe_track": False,
        "job_id": job.job_id,
        "job": job.identity(),
        "score_blind": True,
        "claim_resource_guard": copy.deepcopy(dict(guard)),
        "test_count": len(rows),
        "test_rows_sha256": _rows_sha256(rows),
        "metadata": copy.deepcopy(dict(metadata)),
    }
    violations = _recursive_forbidden_keys(record)
    if violations:
        raise AuthorRecipeError(
            f"score-blind record contains forbidden fields: {violations[:5]}"
        )
    partial_root = _safe_mkdir(run_root / "partials", root=run_root)
    partial = partial_root / f"{job.job_id}.{claim.nonce}.partial"
    _safe_mkdir(partial, root=run_root, mode=0o700, exist_ok=False)
    partial_stat = _anchored_lstat(partial)
    partial_identity = (
        int(partial_stat.st_dev),
        int(partial_stat.st_ino),
    )
    record_path = partial / "record.json"
    prediction_path = partial / "predictions.npz"
    completion_path = partial / "completion.json"
    destination = _record_directory(run_root, job)
    published_identity: tuple[int, int] | None = None
    try:
        _write_json_exclusive(record_path, record, root=run_root)
        archive_buffer = io.BytesIO()
        np.savez_compressed(
            archive_buffer,
            rows=np.ascontiguousarray(rows),
            probabilities=np.ascontiguousarray(values),
        )
        _write_bytes_exclusive(
            prediction_path, archive_buffer.getvalue(), root=run_root
        )
        completion = {
            "schema": COMPLETION_SCHEMA,
            "created_at": _utc_now(),
            "plan_sha256": plan["plan_sha256"],
            "track": TRACK,
            "job_id": job.job_id,
            "job": job.identity(),
            "files": {
                "record.json": _sha256_file(record_path),
                "predictions.npz": _sha256_file(prediction_path),
            },
        }
        _write_json_exclusive(completion_path, completion, root=run_root)
        _fsync_directory(partial, root=run_root)
        _safe_mkdir(destination.parent, root=run_root)
        _rebind_authoritative_inputs(
            run_root=run_root,
            plan=plan,
            cache_root=cache_root,
            job=job,
            claim=claim,
            metadata=metadata,
            requested_gpu=requested_gpu,
        )
        try:
            _rename_leaf_nofollow(
                partial,
                destination,
                root=run_root,
                expected_source_identity=partial_identity,
            )
        except BaseException as publication_error:
            if _path_binds_identity(destination, partial_identity):
                published_identity = partial_identity
                claim.publication_state.record_identity = published_identity
                _seal_directory_read_only(
                    destination,
                    root=run_root,
                    expected_identity=partial_identity,
                )
                _rebind_authoritative_inputs(
                    run_root=run_root,
                    plan=plan,
                    cache_root=cache_root,
                    job=job,
                    claim=claim,
                    metadata=metadata,
                    requested_gpu=requested_gpu,
                )
                raise
            if isinstance(publication_error, FileExistsError):
                validate_completion(run_root, plan, job)
                _quarantine(
                    run_root,
                    partial,
                    category="partials",
                    reason="duplicate",
                    expected_identity=partial_identity,
                )
                return destination
            raise
        if not _path_binds_identity(destination, partial_identity):
            raise AuthorRecipeError(
                "published record no longer binds the staged inode"
            )
        published_identity = partial_identity
        claim.publication_state.record_identity = published_identity
        _seal_directory_read_only(
            destination,
            root=run_root,
            expected_identity=partial_identity,
        )
        _fsync_directory(partial_root, root=run_root)
        _fsync_directory(destination.parent, root=run_root)
        validate_completion(run_root, plan, job)
        _rebind_authoritative_inputs(
            run_root=run_root,
            plan=plan,
            cache_root=cache_root,
            job=job,
            claim=claim,
            metadata=metadata,
            requested_gpu=requested_gpu,
        )
        return destination
    except BaseException:
        if (
            published_identity is not None
            and os.path.lexists(destination)
        ):
            try:
                _quarantine(
                    run_root,
                    destination,
                    category="records",
                    reason="failed",
                    expected_identity=published_identity,
                )
            except (FileNotFoundError, ClaimUnavailable):
                pass
            finally:
                published_identity = None
                claim.publication_state.record_identity = None
        if os.path.lexists(partial):
            try:
                _quarantine(
                    run_root,
                    partial,
                    category="partials",
                    reason="failed",
                    expected_identity=partial_identity,
                )
            except (FileNotFoundError, ClaimUnavailable):
                pass
        raise


def _capture_completion_snapshot(
    run_root: Path, plan: Mapping[str, Any], job: Job
) -> CompletionSnapshot:
    """Validate one package from one coherent directory and all-child FD set."""

    run_root = _absolute_path(run_root)
    directory = _record_directory(run_root, job)
    _require_safe_ancestors(
        directory,
        root=run_root,
        include_leaf=True,
        allow_missing_leaf=False,
        context="record directory",
    )
    directory_descriptor = _open_directory_anchored(directory)
    descriptors: dict[str, int] = {}
    artifact_bytes: dict[str, bytes] = {}
    entry_fingerprints: dict[str, tuple[int, ...]] = {}
    try:
        directory_before = os.fstat(directory_descriptor)
        directory_fingerprint = _stat_fingerprint(directory_before)
        if (
            not stat.S_ISDIR(directory_before.st_mode)
            or stat.S_IMODE(directory_before.st_mode) != 0o555
        ):
            raise AuthorRecipeError(
                f"{directory} is not an immutable 0555 completion directory"
            )
        initial_names = tuple(sorted(os.listdir(directory_descriptor)))
        if set(initial_names) != FINAL_FILENAMES:
            raise AuthorRecipeError(
                f"{directory} has an invalid exact file set"
            )
        for name in sorted(FINAL_FILENAMES):
            observed = os.stat(
                name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            fingerprint = _stat_fingerprint(observed)
            if (
                stat.S_ISLNK(observed.st_mode)
                or not stat.S_ISREG(observed.st_mode)
                or observed.st_nlink != 1
                or observed.st_mode & 0o222
            ):
                raise AuthorRecipeError(
                    f"{directory / name} is an aliased or writable file"
                )
            descriptor = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=directory_descriptor,
            )
            descriptors[name] = descriptor
            if _stat_fingerprint(os.fstat(descriptor)) != fingerprint:
                raise AuthorRecipeError(
                    f"{directory / name} changed while opening package"
                )
            entry_fingerprints[name] = fingerprint
        expected_entries = tuple(sorted(entry_fingerprints.items()))
        before_read_entries = tuple(
            (
                name,
                _stat_fingerprint(
                    os.stat(
                        name,
                        dir_fd=directory_descriptor,
                        follow_symlinks=False,
                    )
                ),
            )
            for name in sorted(os.listdir(directory_descriptor))
        )
        if (
            before_read_entries != expected_entries
            or _stat_fingerprint(os.fstat(directory_descriptor))
            != directory_fingerprint
        ):
            raise AuthorRecipeError(
                f"{directory} changed before package read"
            )
        for name in sorted(FINAL_FILENAMES):
            size = entry_fingerprints[name][
                _SNAPSHOT_STABLE_FIELDS.index("st_size")
            ]
            chunks: list[bytes] = []
            offset = 0
            while offset < size:
                chunk = os.pread(
                    descriptors[name],
                    min(1024 * 1024, size - offset),
                    offset,
                )
                if not chunk:
                    raise AuthorRecipeError(
                        f"{directory / name} package read was incomplete"
                    )
                chunks.append(chunk)
                offset += len(chunk)
            if os.pread(descriptors[name], 1, offset):
                raise AuthorRecipeError(
                    f"{directory / name} grew during package read"
                )
            artifact_bytes[name] = b"".join(chunks)
        after_read_entries: list[tuple[str, tuple[int, ...]]] = []
        for name in sorted(os.listdir(directory_descriptor)):
            path_fingerprint = _stat_fingerprint(
                os.stat(
                    name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
            )
            after_read_entries.append((name, path_fingerprint))
            if (
                name not in descriptors
                or _stat_fingerprint(os.fstat(descriptors[name]))
                != entry_fingerprints.get(name)
                or path_fingerprint != entry_fingerprints.get(name)
            ):
                raise AuthorRecipeError(
                    f"{directory / name} changed during package read"
                )
        canonical_after = _anchored_lstat(directory)
        if (
            tuple(after_read_entries) != expected_entries
            or _stat_fingerprint(os.fstat(directory_descriptor))
            != directory_fingerprint
            or _stat_fingerprint(canonical_after)
            != directory_fingerprint
        ):
            raise AuthorRecipeError(
                f"{directory} changed while package was validated"
            )
    finally:
        for descriptor in descriptors.values():
            os.close(descriptor)
        os.close(directory_descriptor)
    completion_path = directory / "completion.json"
    record_path = directory / "record.json"
    prediction_path = directory / "predictions.npz"
    completion_bytes = artifact_bytes["completion.json"]
    record_bytes = artifact_bytes["record.json"]
    prediction_bytes = artifact_bytes["predictions.npz"]
    completion = _decode_json_object(
        completion_bytes, context=str(completion_path)
    )
    _require_exact_keys(
        completion,
        {
            "schema",
            "created_at",
            "plan_sha256",
            "track",
            "job_id",
            "job",
            "files",
        },
        "completion",
    )
    _validate_timestamp(completion["created_at"], "completion")
    _validate_job_identity_mapping(completion["job"], job, "completion")
    if completion_bytes != _canonical_bytes(completion) + b"\n":
        raise AuthorRecipeError("completion JSON is not canonical")
    expected_completion = {
        "schema": COMPLETION_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "track": TRACK,
        "job_id": job.job_id,
        "job": job.identity(),
    }
    for key, value in expected_completion.items():
        if completion.get(key) != value:
            raise AuthorRecipeError(f"completion has invalid {key}")
    files = completion["files"]
    if not isinstance(files, Mapping):
        raise AuthorRecipeError("completion file manifest is absent")
    _require_exact_keys(
        files, {"record.json", "predictions.npz"}, "completion file manifest"
    )
    expected_files = {
        "record.json": _sha256_bytes(record_bytes),
        "predictions.npz": _sha256_bytes(prediction_bytes),
    }
    if any(
        not isinstance(value, str) or not HEX_64_RE.fullmatch(value)
        for value in files.values()
    ) or files != expected_files:
        raise AuthorRecipeError("completion file checksums are invalid")
    record = _decode_json_object(record_bytes, context=str(record_path))
    _require_exact_keys(
        record,
        {
            "schema",
            "created_at",
            "plan_sha256",
            "track",
            "common_recipe_track",
            "job_id",
            "job",
            "score_blind",
            "claim_resource_guard",
            "test_count",
            "test_rows_sha256",
            "metadata",
        },
        "score-blind record",
    )
    _validate_timestamp(record["created_at"], "score-blind record")
    _validate_job_identity_mapping(record["job"], job, "score-blind record")
    if record_bytes != _canonical_bytes(record) + b"\n":
        raise AuthorRecipeError("score-blind record JSON is not canonical")
    expected_record = {
        "schema": RECORD_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "track": TRACK,
        "common_recipe_track": False,
        "job_id": job.job_id,
        "job": job.identity(),
        "score_blind": True,
    }
    for key, value in expected_record.items():
        if record.get(key) != value:
            raise AuthorRecipeError(f"record has invalid {key}")
    violations = _recursive_forbidden_keys(record)
    if violations:
        raise AuthorRecipeError(
            f"record violates score-blind schema: {violations[:5]}"
        )
    _validate_resource_guard(
        record["claim_resource_guard"], "record claim resource guard"
    )
    test_count = record["test_count"]
    test_rows_sha256 = record["test_rows_sha256"]
    if (
        isinstance(test_count, bool)
        or not isinstance(test_count, int)
        or test_count <= 0
        or not isinstance(test_rows_sha256, str)
        or not HEX_64_RE.fullmatch(test_rows_sha256)
    ):
        raise AuthorRecipeError("record test-row identity is invalid")
    metadata = record.get("metadata")
    if not isinstance(metadata, Mapping):
        raise AuthorRecipeError("record has no metadata")
    _validate_job_metadata(plan, job, metadata)
    try:
        with zipfile.ZipFile(
            io.BytesIO(prediction_bytes), mode="r"
        ) as raw_archive:
            raw_members = [
                member.filename for member in raw_archive.infolist()
            ]
            if raw_members != ["rows.npy", "probabilities.npy"]:
                raise AuthorRecipeError(
                    "prediction archive has invalid or duplicate raw members"
                )
        with np.load(io.BytesIO(prediction_bytes), allow_pickle=False) as archive:
            if archive.files != ["rows", "probabilities"]:
                raise AuthorRecipeError(
                    "prediction archive has an invalid exact array order"
                )
            rows = archive["rows"].copy()
            probabilities = archive["probabilities"].copy()
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise AuthorRecipeError(f"prediction archive cannot be loaded: {error}") from error
    split = _planned_split(plan, job)
    expected_test = split["partitions"]["test"]
    n_classes = int(plan["datasets"][job.dataset]["n_classes"])
    if (
        rows.dtype != np.int64
        or rows.ndim != 1
        or len(rows) != int(record["test_count"])
        or len(rows) != int(expected_test["count"])
        or np.any(rows < 0)
        or np.any(rows >= int(split["trial_count"]))
        or len(np.unique(rows)) != len(rows)
        or _rows_sha256(rows) != record["test_rows_sha256"]
        or _rows_sha256(rows) != expected_test["rows_sha256"]
        or probabilities.dtype != np.float64
        or probabilities.shape != (len(rows), n_classes)
        or not np.all(np.isfinite(probabilities))
        or np.any(probabilities < 0.0)
        or np.any(probabilities > 1.0)
        or not np.allclose(
            probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-6
        )
    ):
        raise AuthorRecipeError("published prediction arrays are invalid")
    rows = np.ascontiguousarray(rows, dtype=np.int64)
    probabilities = np.ascontiguousarray(
        probabilities, dtype=np.float64
    )
    rows.setflags(write=False)
    probabilities.setflags(write=False)
    artifact_sha256 = {
        name: _sha256_bytes(payload)
        for name, payload in sorted(artifact_bytes.items())
    }
    return CompletionSnapshot(
        record=record,
        completion=completion,
        rows=rows,
        probabilities=probabilities,
        artifact_bytes=dict(artifact_bytes),
        artifact_sha256=artifact_sha256,
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


def _model_factory(
    job: Job,
    *,
    n_channels: int,
    n_outputs: int,
    n_times: int,
    sfreq: float,
    channel_names: tuple[str, ...],
    positions: np.ndarray,
) -> Callable[[], Any]:
    def factory() -> Any:
        import torch

        from .baselines import make_model

        base = make_model(
            "tcformer" if job.reference == "reference.tcformer" else "fbcnet",
            n_channels=n_channels,
            n_outputs=n_outputs,
            n_times=n_times,
            sfreq=sfreq,
            channel_names=channel_names,
            channel_positions=torch.as_tensor(positions, dtype=torch.float32),
        )
        if job.reference == "reference.fbcnet":
            from .reference_training import PrefilteredFBCNetAdapter

            return PrefilteredFBCNetAdapter(base)
        return base

    return factory


def execute_author_recipe_job(
    *,
    job: Job,
    plan: Mapping[str, Any],
    cache_root: Path,
    device: str,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Fit one frozen reference procedure and return score-blind predictions."""

    import torch

    from .data import (
        apply_channel_scaler,
        fit_channel_scaler,
        split_indices,
    )
    from .models import parameter_count
    from .reference_training import (
        PrefilteredFBCNetAdapter,
        apply_fbcnet_cheby2_filterbank,
        fit_fbcnet_reference,
        fit_tcformer_reference,
        optimizer_state_sha256,
    )
    from .training import predict_probabilities

    started = time.perf_counter()
    data = _load_subject_cache_secure(
        job.dataset, job.subject, cache_root=cache_root
    )
    if data["identity"] != _planned_cache(plan, job):
        raise AuthorRecipeError("runtime cache differs from the immutable plan")
    train_rows, validation_rows, test_rows = split_indices(
        job.dataset,
        data["y"],
        data["sessions"],
        data["runs"],
        fold=job.fold,
        subject=job.subject,
    )
    observed_split = full_grid._one_split_identity(
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
        raise AuthorRecipeError("runtime split differs from the immutable plan")
    source_rows = np.sort(
        np.concatenate((train_rows, validation_rows))
    ).astype(np.int64, copy=False)
    channel_names = tuple(str(value) for value in data["channel_names"].tolist())
    n_classes = int(plan["datasets"][job.dataset]["n_classes"])
    sfreq = float(
        plan["datasets"][job.dataset]["preprocessing"]["sfreq_hz"]
    )
    target = torch.device(device)
    if target.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target)
    factory = _model_factory(
        job,
        n_channels=int(data["x"].shape[1]),
        n_outputs=n_classes,
        n_times=int(data["x"].shape[2]),
        sfreq=sfreq,
        channel_names=channel_names,
        positions=data["positions"],
    )

    if job.reference == "reference.tcformer":
        mean, std = fit_channel_scaler(data["x"][source_rows], channel_names)
        x_source = apply_channel_scaler(data["x"][source_rows], mean, std)
        x_test = apply_channel_scaler(data["x"][test_rows], mean, std)
        config = TCFormerReferenceConfig(**_expected_config(plan, job))
        fit = fit_tcformer_reference(
            factory,
            x_source,
            data["y"][source_rows],
            data["positions"],
            config=config,
        )
        fit_metadata = {
            "seed_installed_before_construction": True,
            "initial_state_sha256": fit["initial_state_sha256"],
            "final_state_sha256": fit["final_state_sha256"],
            "final_optimizer_state_sha256": optimizer_state_sha256(
                fit["optimizer"]
            ),
            "epochs_run": int(fit["epochs_run"]),
            "source_count": len(source_rows),
            "parameter_count": parameter_count(fit["model"]),
        }
        scalers = {
            "source_mean_sha256": _array_sha256(mean),
            "source_std_sha256": _array_sha256(std),
        }
    elif job.reference == "reference.fbcnet":
        mean, std = fit_channel_scaler(data["x"][train_rows], channel_names)
        x_train = apply_fbcnet_cheby2_filterbank(
            apply_channel_scaler(data["x"][train_rows], mean, std), sfreq
        )
        x_validation = apply_fbcnet_cheby2_filterbank(
            apply_channel_scaler(data["x"][validation_rows], mean, std), sfreq
        )
        x_test = apply_fbcnet_cheby2_filterbank(
            apply_channel_scaler(data["x"][test_rows], mean, std), sfreq
        )
        config = FBCNetReferenceConfig(**_expected_config(plan, job))
        fit = fit_fbcnet_reference(
            factory,
            x_train,
            data["y"][train_rows],
            x_validation,
            data["y"][validation_rows],
            data["positions"],
            config=config,
        )
        if not isinstance(fit["model"], PrefilteredFBCNetAdapter):
            raise AuthorRecipeError("FBCNet reference fitter returned wrong adapter")
        fit_metadata = {
            "seed_installed_before_construction": True,
            "initial_state_sha256": fit["initial_state_sha256"],
            "final_state_sha256": fit["final_state_sha256"],
            "final_optimizer_state_sha256": fit[
                "final_optimizer_state_sha256"
            ],
            "stage1_best_epoch": int(fit["stage1_best_epoch"]),
            "stage1_stop": str(fit["stage1_stop"]),
            "stage1_epochs_run": int(fit["stage1_epochs_run"]),
            "stage2_epochs_run": int(fit["stage2_epochs_run"]),
            "stage2_stop": str(fit["stage2_stop"]),
            "stage1_train_loss_threshold": float(
                fit["stage1_train_loss_threshold"]
            ),
            "stage1_threshold_origin": str(
                fit["stage1_threshold_origin"]
            ),
            "stage1_threshold_origin_epoch": int(
                fit["stage1_threshold_origin_epoch"]
            ),
            "stage1_best_model_state_sha256": fit[
                "stage1_best_model_state_sha256"
            ],
            "stage1_best_optimizer_state_sha256": fit[
                "stage1_best_optimizer_state_sha256"
            ],
            "stage2_start_model_state_sha256": fit[
                "stage2_start_model_state_sha256"
            ],
            "stage2_start_optimizer_state_sha256": fit[
                "stage2_start_optimizer_state_sha256"
            ],
            "source_count": int(fit["stage2_source_count"]),
            "original_validation_count": int(
                fit["stage2_original_validation_count"]
            ),
            "parameter_count": parameter_count(fit["model"]),
        }
        scalers = {
            "selection_mean_sha256": _array_sha256(mean),
            "selection_std_sha256": _array_sha256(std),
        }
    else:
        raise AuthorRecipeError(f"unknown author reference {job.reference}")

    if target.type == "cuda":
        torch.cuda.synchronize(target)
    inference_started = time.perf_counter()
    # This is deliberately the only test-prediction call in the executor.
    probabilities = predict_probabilities(
        fit["model"],
        x_test,
        data["positions"],
        device=device,
    )
    if target.type == "cuda":
        torch.cuda.synchronize(target)
    inference_seconds = time.perf_counter() - inference_started
    peak_memory = (
        int(torch.cuda.max_memory_allocated(target))
        if target.type == "cuda"
        else 0
    )
    observed_recipe = json.loads(_canonical_bytes(fit["recipe"]))
    if observed_recipe != _expected_recipe_metadata(plan, job):
        raise AuthorRecipeError("reference fitter returned an unexpected recipe")
    metadata = {
        "track": TRACK,
        "cache_array_sha256": str(data["identity"]["array_sha256"]),
        "channels": list(channel_names),
        "split": _split_metadata(observed_split),
        "scalers": scalers,
        "recipe": observed_recipe,
        "fit": fit_metadata,
        "protocol": _expected_protocol(job.reference),
        "timing_seconds": {
            "source_fit": float(fit["fit_seconds"]),
            "test_inference": inference_seconds,
            "job_total": time.perf_counter() - started,
        },
        "cuda_peak_memory_bytes": peak_memory,
        "runtime": {
            "worker_cpu_threads": int(torch.get_num_threads()),
            "worker_interop_threads": int(torch.get_num_interop_threads()),
            "thread_environment": {
                name: os.environ.get(name)
                for name in THREAD_ENVIRONMENT_VARIABLES
            },
            "nvidia_driver_versions": full_grid._nvidia_driver_versions(),
        },
    }
    del fit, factory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return (
        metadata,
        np.asarray(test_rows, dtype=np.int64),
        np.asarray(probabilities, dtype=np.float64),
    )


def _failure_files(run_root: Path, job: Job) -> tuple[Path, ...]:
    root = _failure_root(run_root, job)
    return tuple(sorted(root.glob(f"{job.job_id}.attempt-*.json"))) if root.exists() else ()


def _record_failure(
    run_root: Path,
    plan: Mapping[str, Any],
    claim: Claim,
    error: BaseException,
) -> Path:
    root = _failure_root(run_root, claim.job)
    attempt = len(_failure_files(run_root, claim.job)) + 1
    path = root / f"{claim.job.job_id}.attempt-{attempt:04d}.json"
    _write_json_exclusive(
        path,
        {
            "schema": FAILURE_SCHEMA,
            "created_at": _utc_now(),
            "plan_sha256": plan["plan_sha256"],
            "job_id": claim.job.job_id,
            "job": claim.job.identity(),
            "owner": dict(claim.owner),
            "error_type": type(error).__qualname__,
            "error_message": str(error),
        },
    )
    return path


def _record_tree_extras(
    run_root: Path, jobs: Sequence[Job]
) -> tuple[list[str], list[str]]:
    run_root = _absolute_path(run_root)
    records_root = run_root / "records"
    if not os.path.lexists(records_root):
        return [], []
    try:
        root_stat = os.lstat(records_root)
    except FileNotFoundError:
        return [], []
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        return ["records"], []
    _require_safe_ancestors(records_root, root=run_root, context="records root")
    expected_dirs = {_record_directory(run_root, job) for job in jobs}
    expected_files = {
        directory / filename
        for directory in expected_dirs
        for filename in FINAL_FILENAMES
    }
    allowed_dirs = {records_root}
    for directory in expected_dirs:
        current = directory
        while current != records_root:
            allowed_dirs.add(current)
            current = current.parent
    extra_files: list[str] = []
    extra_dirs: list[str] = []
    for path, kind in _walk_tree_no_follow(records_root, run_root=run_root):
        if kind == "directory":
            if path not in allowed_dirs:
                extra_dirs.append(str(path.relative_to(run_root)))
        elif kind == "regular":
            if path not in expected_files:
                extra_files.append(str(path.relative_to(run_root)))
        else:
            extra_files.append(str(path.relative_to(run_root)))
    return sorted(extra_files), sorted(extra_dirs)


def _walk_tree_no_follow(
    root: Path, *, run_root: Path
) -> list[tuple[Path, str]]:
    """Walk a coherent tree through held no-follow directory descriptors."""

    root = _absolute_path(root)
    run_root = _absolute_path(run_root)
    _require_contained(root, run_root, "audit tree")
    if not os.path.lexists(root):
        return []
    root_stat = _anchored_lstat(root)
    if stat.S_ISLNK(root_stat.st_mode):
        return [(root, "symlink")]
    if not stat.S_ISDIR(root_stat.st_mode):
        return [(root, "special")]
    _require_safe_ancestors(root, root=run_root, context="audit tree")
    result: list[tuple[Path, str]] = []
    root_descriptor = _open_directory_anchored(root)

    def walk(directory: Path, descriptor: int) -> None:
        before = _stat_fingerprint(os.fstat(descriptor))
        for name in sorted(os.listdir(descriptor)):
            if name in {"", ".", ".."} or "/" in name:
                raise AuthorRecipeError("audit tree contains an unsafe name")
            path = directory / name
            observed = os.stat(
                name,
                dir_fd=descriptor,
                follow_symlinks=False,
            )
            if stat.S_ISLNK(observed.st_mode):
                kind = "symlink"
            elif stat.S_ISDIR(observed.st_mode):
                kind = "directory"
            elif stat.S_ISREG(observed.st_mode):
                kind = (
                    "regular"
                    if int(observed.st_nlink) == 1
                    else "hardlink"
                )
            else:
                kind = "special"
            result.append((path, kind))
            if kind == "directory":
                child = os.open(
                    name,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_NONBLOCK", 0)
                    | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=descriptor,
                )
                try:
                    opened = os.fstat(child)
                    if _stat_fingerprint(opened) != _stat_fingerprint(
                        observed
                    ):
                        raise AuthorRecipeError(
                            f"audit directory changed while opening: {path}"
                        )
                    walk(path, child)
                finally:
                    os.close(child)
        if _stat_fingerprint(os.fstat(descriptor)) != before:
            raise AuthorRecipeError(
                f"audit directory changed while enumerating: {directory}"
            )

    try:
        walk(root, root_descriptor)
        if _stat_fingerprint(_anchored_lstat(root)) != _stat_fingerprint(
            root_stat
        ):
            raise AuthorRecipeError(
                f"audit tree root changed while enumerating: {root}"
            )
    finally:
        os.close(root_descriptor)
    result.sort(key=lambda item: str(item[0]))
    return result


def audit_grid(
    run_root: Path,
    plan: Mapping[str, Any],
    *,
    publication_fence: PublicationFence | None = None,
) -> dict[str, Any]:
    run_root = _absolute_path(run_root)
    _require_safe_ancestors(
        run_root,
        include_leaf=True,
        allow_missing_leaf=False,
        context="run root",
    )
    run_root_fingerprint = _stat_fingerprint(
        _anchored_lstat(run_root)
    )
    expected_fence_path: Path | None = None
    if publication_fence is not None:
        validate_publication_fence(publication_fence, plan)
        expected_fence_path = _publication_fence_path(run_root)
    jobs = tuple(iter_jobs(plan))
    complete: list[str] = []
    missing: list[str] = []
    corrupt: dict[str, str] = {}
    for job in jobs:
        directory = _record_directory(run_root, job)
        if not os.path.lexists(directory):
            missing.append(job.job_id)
            continue
        try:
            validate_completion(run_root, plan, job)
        except Exception as error:
            corrupt[job.job_id] = str(error)
        else:
            complete.append(job.job_id)
    extra_files, extra_dirs = _record_tree_extras(run_root, jobs)
    expected_claims = {_claim_path(run_root, job): job for job in jobs}
    allowed_claim_dirs = {run_root / "claims"}
    for path in expected_claims:
        current = path.parent
        while current != run_root / "claims":
            allowed_claim_dirs.add(current)
            current = current.parent
    live: list[str] = []
    stale: list[str] = []
    unknown_claims: list[str] = []
    claims_root = run_root / "claims"
    if os.path.lexists(claims_root):
        for path, kind in _walk_tree_no_follow(
            claims_root, run_root=run_root
        ):
            if kind == "directory" and path in allowed_claim_dirs:
                continue
            job = expected_claims.get(path)
            if job is None or kind != "regular":
                unknown_claims.append(str(path.relative_to(run_root)))
                continue
            try:
                value = strict_load(path, root=run_root, readonly=True)
            except Exception:
                stale.append(job.job_id)
            else:
                (live if _claim_is_live(value) else stale).append(job.job_id)
    partials: list[str] = []
    partial_root = run_root / "partials"
    if os.path.lexists(partial_root):
        partials = sorted(
            str(path.relative_to(run_root))
            for path, _ in _walk_tree_no_follow(
                partial_root, run_root=run_root
            )
        )
        if not partials:
            root_stat = os.lstat(partial_root)
            if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(
                root_stat.st_mode
            ):
                partials = ["partials"]
    allowed_root_types = {
        "plan.json": "regular",
        "plan.sha256": "regular",
        "records": "directory",
        "claims": "directory",
        "partials": "directory",
        "failures": "directory",
        "quarantine": "directory",
        "worker_logs": "directory",
        "audit.json": "regular",
        PUBLICATION_GATE_FILENAME: "regular",
    }
    if expected_fence_path is not None:
        allowed_root_types[PUBLICATION_FENCE_FILENAME] = "regular"
    unknown_filesystem_paths: list[str] = []
    for entry in os.scandir(run_root):
        observed = entry.stat(follow_symlinks=False)
        if stat.S_ISLNK(observed.st_mode):
            kind = "symlink"
        elif stat.S_ISDIR(observed.st_mode):
            kind = "directory"
        elif stat.S_ISREG(observed.st_mode):
            kind = (
                "regular"
                if int(observed.st_nlink) == 1
                else "hardlink"
            )
        else:
            kind = "special"
        expected_kind = allowed_root_types.get(entry.name)
        if expected_kind is None or kind != expected_kind:
            unknown_filesystem_paths.append(entry.name)
            continue
        if entry.name == PUBLICATION_GATE_FILENAME:
            try:
                if (
                    _read_regular_bytes(
                        Path(entry.path),
                        root=run_root,
                        readonly=True,
                        context="publication gate",
                    )
                    != PUBLICATION_GATE_BYTES
                ):
                    raise AuthorRecipeError("publication gate content drifted")
            except Exception:
                unknown_filesystem_paths.append(entry.name)
    for auxiliary_name in ("failures", "quarantine", "worker_logs"):
        auxiliary_root = run_root / auxiliary_name
        if not os.path.lexists(auxiliary_root):
            continue
        for path, kind in _walk_tree_no_follow(
            auxiliary_root, run_root=run_root
        ):
            if kind in {"symlink", "hardlink", "special"}:
                unknown_filesystem_paths.append(
                    str(path.relative_to(run_root))
                )
    unknown_filesystem_paths = sorted(set(unknown_filesystem_paths))
    failures = {
        job.job_id: len(_failure_files(run_root, job))
        for job in jobs
        if _failure_files(run_root, job)
    }
    exact = (
        len(complete) == len(jobs)
        and not missing
        and not corrupt
        and not extra_files
        and not extra_dirs
        and not live
        and not stale
        and not unknown_claims
        and not partials
        and not unknown_filesystem_paths
    )
    result = {
        "schema": AUDIT_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "track": TRACK,
        "expected_jobs": len(jobs),
        "complete_jobs": len(complete),
        "missing_jobs": len(missing),
        "corrupt_jobs": len(corrupt),
        "extra_files": len(extra_files),
        "extra_directories": len(extra_dirs),
        "live_claims": len(live),
        "stale_claims": len(stale),
        "unknown_claims": len(unknown_claims),
        "partials": len(partials),
        "unknown_filesystem_paths": len(unknown_filesystem_paths),
        "failed_jobs": len(failures),
        "exact_cartesian_complete": exact,
        "complete_job_ids": complete,
        "details": {
            "missing_job_ids": missing,
            "corrupt": corrupt,
            "extra_file_paths": extra_files,
            "extra_directory_paths": extra_dirs,
            "live_claim_job_ids": live,
            "stale_claim_job_ids": stale,
            "unknown_claim_paths": unknown_claims,
            "partial_paths": partials,
            "unknown_filesystem_paths": unknown_filesystem_paths,
            "failure_attempts": failures,
        },
    }
    if _stat_fingerprint(_anchored_lstat(run_root)) != run_root_fingerprint:
        raise AuthorRecipeError(
            "run root changed while the exact audit was captured"
        )
    return result


def _rotated_indices(length: int, start: int) -> Iterator[int]:
    for offset in range(length):
        yield (start + offset) % length


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
    allow_busy_gpu: bool,
    foreign_claim_timeout_seconds: float,
    executor: Callable[..., tuple[dict[str, Any], np.ndarray, np.ndarray]]
    | None = None,
    gpu_probe: Callable[[], full_grid.GPUStatus] | None = None,
    disk_probe: Callable[[], full_grid.DiskStatus] | None = None,
    space_check_path: Path | None = None,
    minimum_free_gib: float = DEFAULT_MIN_FREE_GIB,
) -> int:
    # Repeat the floor at the programmatic worker boundary: this publication
    # runner has no low-disk bypass, including for callers that skip ``main``.
    full_grid._validate_disk_threshold(
        minimum_free_gib, allow_low_disk=False
    )
    plan = load_plan(run_root)
    formal_roster = (
        tuple(plan.get("dataset_order", ())) == OPENED_DATASETS
        and tuple(plan.get("references", ())) == REFERENCES
        and tuple(plan.get("seeds", ())) == FORMAL_SEEDS
    )
    if formal_roster and executor is not None:
        raise AuthorRecipeError(
            "formal workers may not inject an executor callable"
        )
    jobs = tuple(iter_jobs(plan))
    if not jobs:
        return 0
    selected_executor = (
        full_grid._resolve_callable(str(plan["executor"]))
        if executor is None
        else executor
    )
    selected_gpu_probe = gpu_probe or (
        lambda: full_grid.probe_gpu(
            gpu,
            allowed_pids=(os.getpid(),),
            max_utilization_percent=max_utilization_percent,
            max_foreign_memory_mib=max_foreign_memory_mib,
            allow_busy=allow_busy_gpu,
        )
    )
    checked_path = (
        None
        if space_check_path is None
        else _absolute_path(space_check_path)
    )

    def default_disk_probe() -> full_grid.DiskStatus:
        primary = full_grid.probe_disk(
            run_root,
            minimum_free_gib=minimum_free_gib,
            allow_low_disk=False,
        )
        if (
            primary.safe
            and checked_path is not None
            and checked_path != _absolute_path(run_root)
        ):
            return full_grid.probe_disk(
                checked_path,
                minimum_free_gib=minimum_free_gib,
                allow_low_disk=False,
            )
        return primary

    selected_disk_probe = disk_probe or default_disk_probe

    def resource_probe() -> full_grid.ResourceStatus:
        return full_grid.combine_resource_status(
            selected_gpu_probe(), selected_disk_probe()
        )

    cursor = (
        worker_index * math.ceil(len(jobs) / MAX_GPU_WORKERS)
    ) % len(jobs)
    invocation_failures: dict[str, int] = {}
    while True:
        ran = False
        gpu_busy = False
        unblocked = False
        for index in _rotated_indices(len(jobs), cursor):
            job = jobs[index]
            output = _record_directory(run_root, job)
            if os.path.lexists(output):
                try:
                    validate_completion(run_root, plan, job)
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
                                foreign_claim_timeout_seconds=(
                                    foreign_claim_timeout_seconds
                                ),
                            )
                        except ClaimUnavailable:
                            pass
                    continue
            historical = len(_failure_files(run_root, job))
            if (
                not retry_failed and historical >= max_attempts_per_job
            ) or invocation_failures.get(job.job_id, 0) >= max_attempts_per_job:
                continue
            unblocked = True
            try:
                claim = acquire_claim(
                    run_root,
                    plan,
                    job,
                    recover_stale=recover_stale,
                    foreign_claim_timeout_seconds=foreign_claim_timeout_seconds,
                    before_claim=resource_probe,
                )
            except full_grid.GPUUnavailable as error:
                gpu_busy = True
                break
            except GPUUnavailable:
                gpu_busy = True
                break
            except (full_grid.DiskUnavailable, DiskUnavailable) as error:
                print(f"[{_utc_now()}] worker stop: {error}", file=sys.stderr)
                return 2
            except ClaimUnavailable:
                continue
            try:
                metadata, rows, probabilities = selected_executor(
                    job=job,
                    plan=plan,
                    cache_root=cache_root,
                    device="cuda:0",
                )
                commit_job_output(
                    run_root,
                    plan,
                    claim,
                    metadata=metadata,
                    test_rows=rows,
                    probabilities=probabilities,
                    cache_root=cache_root,
                    requested_gpu=gpu,
                )
            except Exception as error:
                invocation_failures[job.job_id] = (
                    invocation_failures.get(job.job_id, 0) + 1
                )
                _record_failure(run_root, plan, claim, error)
            finally:
                release_claim(claim)
            cursor = (index + 1) % len(jobs)
            ran = True
            break
        if ran:
            continue
        if gpu_busy:
            time.sleep(max(poll_seconds, 0.1))
            continue
        audit = audit_grid(run_root, plan)
        if audit["exact_cartesian_complete"]:
            return 0
        if audit["live_claims"]:
            time.sleep(max(poll_seconds, 0.1))
            continue
        if not unblocked or audit["failed_jobs"]:
            return 1
        time.sleep(max(poll_seconds, 0.1))


def _require_uv_virtual_environment() -> None:
    if Path(sys.prefix).resolve() == Path(sys.base_prefix).resolve():
        raise AuthorRecipeError(
            "author-recipe execution requires an isolated UV virtual "
            "environment; run `uv sync --frozen` then `uv run ...`"
        )
    try:
        subprocess.run(
            ["uv", "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise AuthorRecipeError("uv executable is unavailable") from error
    _validate_uv_project_contract(Path(__file__).resolve().parents[2])


def _require_cublas_environment() -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != REQUIRED_CUBLAS_WORKSPACE_CONFIG:
        raise AuthorRecipeError(
            "export CUBLAS_WORKSPACE_CONFIG=:4096:8 before Python starts"
        )


def _spawn_workers(args: argparse.Namespace, plan: Mapping[str, Any]) -> int:
    identities = full_grid._resolve_gpus(args.gpus)
    threads = full_grid._validate_cpu_threads(
        int(plan["execution_config"]["worker_cpu_threads"])
    )
    run_root = _absolute_path(args.run_root)
    cache_root = _absolute_path(args.cache_root)
    log_root = _safe_mkdir(run_root / "worker_logs", root=run_root)
    processes: list[tuple[subprocess.Popen[bytes], Any]] = []
    code = 0
    try:
        for index, identity in enumerate(identities):
            command = [
                sys.executable,
                "-m",
                "benchmark.author_recipe_grid",
                "worker",
                "--run-root",
                str(run_root),
                "--cache-root",
                str(cache_root),
                "--gpu",
                identity.uuid,
                "--worker-index",
                str(index),
                "--max-attempts-per-job",
                str(args.max_attempts_per_job),
                "--poll-seconds",
                str(args.poll_seconds),
                "--max-idle-utilization",
                str(args.max_idle_utilization),
                "--max-idle-foreign-memory-mib",
                str(args.max_idle_foreign_memory_mib),
                "--foreign-claim-timeout-seconds",
                str(args.foreign_claim_timeout_seconds),
                "--space-check-path",
                str(
                    _absolute_path(args.space_check_path)
                    if args.space_check_path
                    else run_root
                ),
                "--min-free-gib",
                str(args.min_free_gib),
                "--cpu-threads",
                str(threads),
            ]
            if args.allow_busy_gpu:
                command.append("--allow-busy-gpu")
            if args.retry_failed:
                command.append("--retry-failed")
            if not args.recover_stale:
                command.append("--no-recover-stale")
            log_path = log_root / (
                f"worker-{index:02d}-gpu-{identity.index}-"
                f"{identity.uuid.removeprefix('GPU-')[:8]}.log"
            )
            _require_safe_ancestors(
                log_path, root=run_root, context="worker log"
            )
            log_descriptor = os.open(
                log_path,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_APPEND
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            log_stat = os.fstat(log_descriptor)
            if not stat.S_ISREG(log_stat.st_mode):
                os.close(log_descriptor)
                raise AuthorRecipeError("worker log is not a regular file")
            handle = os.fdopen(log_descriptor, "ab", buffering=0)
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = identity.uuid
            environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            environment["CUBLAS_WORKSPACE_CONFIG"] = REQUIRED_CUBLAS_WORKSPACE_CONFIG
            environment["AUTHOR_RECIPE_PLAN_SHA256"] = str(plan["plan_sha256"])
            for variable in THREAD_ENVIRONMENT_VARIABLES:
                environment[variable] = str(threads)
            try:
                process = subprocess.Popen(
                    command,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    env=environment,
                )
            except BaseException:
                handle.close()
                raise
            processes.append((process, handle))
        for process, _ in processes:
            code = max(code, process.wait())
    except BaseException:
        for process, _ in processes:
            if process.poll() is None:
                process.terminate()
        for process, _ in processes:
            process.wait()
        raise
    finally:
        for _, handle in processes:
            handle.close()
    audit = audit_grid(run_root, plan)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if audit["exact_cartesian_complete"] else max(code, 1)


def _add_safety_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--recover-stale",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--max-attempts-per-job", type=int, default=3)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--max-idle-utilization", type=float, default=10.0)
    parser.add_argument(
        "--max-idle-foreign-memory-mib", type=float, default=1024.0
    )
    parser.add_argument(
        "--foreign-claim-timeout-seconds", type=float, default=0.0
    )
    parser.add_argument("--allow-busy-gpu", action="store_true")
    parser.add_argument("--space-check-path")
    parser.add_argument(
        "--min-free-gib", type=float, default=DEFAULT_MIN_FREE_GIB
    )
    parser.add_argument("--cpu-threads", type=int, default=4)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="initialize or resume the grid")
    run.add_argument("--run-root", required=True)
    run.add_argument("--cache-root", required=True)
    run.add_argument("--gpus", default="0,1,2")
    _add_safety_options(run)
    worker = subparsers.add_parser("worker", help=argparse.SUPPRESS)
    worker.add_argument("--run-root", required=True)
    worker.add_argument("--cache-root", required=True)
    worker.add_argument("--gpu", required=True)
    worker.add_argument("--worker-index", type=int, required=True)
    _add_safety_options(worker)
    status = subparsers.add_parser("status")
    status.add_argument("--run-root", required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--run-root", required=True)
    audit.add_argument("--output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_root = _absolute_path(args.run_root)
    if args.command in {"run", "worker"}:
        _require_uv_virtual_environment()
        _require_cublas_environment()
        full_grid._validate_disk_threshold(
            args.min_free_gib, allow_low_disk=False
        )
        if args.max_attempts_per_job <= 0:
            raise ValueError("max attempts must be positive")
    if args.command == "run":
        cache_root = _absolute_path(args.cache_root)
        expected = build_plan(
            cache_root=cache_root,
            executor=DEFAULT_EXECUTOR,
            worker_cpu_threads=args.cpu_threads,
        )
        plan = write_or_validate_plan(run_root, expected)
        verify_runtime_identity(plan, cache_root=cache_root)
        return _spawn_workers(args, plan)
    if args.command == "worker":
        plan = load_or_repair_plan(run_root)
        expected_digest = os.environ.get("AUTHOR_RECIPE_PLAN_SHA256")
        if expected_digest != plan["plan_sha256"]:
            raise AuthorRecipeError("worker plan digest environment is absent/drifted")
        full_grid._verify_visible_cuda_device(args.gpu)
        planned_threads = int(
            plan["execution_config"]["worker_cpu_threads"]
        )
        if int(args.cpu_threads) != planned_threads:
            raise AuthorRecipeError(
                f"worker CPU threads {args.cpu_threads} differ from planned "
                f"value {planned_threads}"
            )
        full_grid._configure_torch_cpu_threads(args.cpu_threads)
        cache_root = _absolute_path(args.cache_root)
        verify_runtime_identity(
            plan,
            cache_root=cache_root,
            worker_cuda_visibility=True,
        )
        return worker_loop(
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
            allow_busy_gpu=args.allow_busy_gpu,
            foreign_claim_timeout_seconds=args.foreign_claim_timeout_seconds,
            space_check_path=(
                None
                if args.space_check_path is None
                else Path(args.space_check_path)
            ),
            minimum_free_gib=args.min_free_gib,
        )
    plan = load_or_repair_plan(run_root)
    result = audit_grid(run_root, plan)
    if args.command == "status":
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "audit":
        if args.output:
            _write_json_exclusive(_absolute_path(args.output), result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["exact_cartesian_complete"] else 1
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
