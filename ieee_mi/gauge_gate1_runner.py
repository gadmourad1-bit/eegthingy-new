"""Audit-grade runner for the 17-subject Gauge opened-development gate.

This module is intentionally candidate-only.  It trains exactly the 17 jobs
returned by :func:`ieee_mi.gauge_quotient_screen.candidate_jobs` for
``OPENED17_STAGE`` and treats the already completed CHSD-v2 comparator records
as immutable, read-only evidence.  It cannot schedule or execute a comparator.

The persistent protocol is deliberately stricter than the earlier scratch
screens:

* one UV-managed virtual environment and one exact source closure are bound in
  the immutable plan;
* all cache, split, decision-rule, and comparator identities are plan-bound;
* every canonical plan/result is a sealed directory published with an atomic
  no-replace rename;
* readers hold the directory and all children open simultaneously and verify
  exact name/inode bindings plus a complete before/after directory
  fingerprint;
* claims, recovery, quarantine, and commits are inode-bound and serialized by
  both a persistent run fence and the project-wide physical-GPU lease guard;
* a failure after a rename hides the exact newly published inode before
  returning an error, so a failed commit never remains authoritative.

The scope is opened development validation only.  A result from this runner is
not confirmation evidence, a state-of-the-art claim, or clinical evidence.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import errno
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import stat
import subprocess
import sys
import time
import uuid
import zipfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

# Shared-workstation limits must be fixed before NumPy, sklearn, or torch is
# imported by any of the bound project modules.
CPU_THREADS: Final = 4
for _thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_thread_variable] = str(CPU_THREADS)

import numpy as np

from . import project_gpu_leases
from .config import (
    CHANNEL_SCALING,
    DEFAULT_MONTAGE_PROFILE,
    PREPROCESSING,
    dataset_spec,
    preprocessing_for_dataset,
)
from .data import SUBJECT_CACHE_NPZ_MEMBERS
from .development_screen import (
    CURATED_SPLITS,
    METRIC_NAMES,
    ScreenJob,
    _array_sha256,
    _json_safe,
    _model_identity,
    _rows_sha256,
    _state_sha256,
    development_rows,
)

PLAN_SCHEMA: Final = "ieee-mi-gauge-gate1-plan-v1"
RECORD_SCHEMA: Final = "ieee-mi-gauge-gate1-record-v1"
COMPLETION_SCHEMA: Final = "ieee-mi-gauge-gate1-completion-v1"
STATUS_SCHEMA: Final = "ieee-mi-gauge-gate1-status-v1"
AUDIT_SCHEMA: Final = "ieee-mi-gauge-gate1-audit-v1"
CLAIM_SCHEMA: Final = "ieee-mi-gauge-gate1-claim-v1"
PLAN_MARKER: Final = b"ieee-mi-gauge-gate1-plan-committed-v1\n"
RESULT_MARKER: Final = b"ieee-mi-gauge-gate1-result-committed-v1\n"
EVALUATION_SCOPE: Final = "opened_development_validation_only"
PURPOSE: Final = "gauge_candidate_opened_17_subject_gate"
TRACK_SCOPE: Final = "gauge-gate1-opened17"
OPENED17_STAGE: Final = "opened_17_subject_gate"
MODEL_NAME: Final = "gauge_quotient_crossmoment_v1"
MODEL_FACTORY: Final = "ieee_mi.gauge_quotient:make_gauge_quotient_model"
MAXIMUM_PARAMETERS: Final = 30_000
TRAIN_CONFIG_JSON: Final = (
    '{"batch_size":64,"device":"cuda:0","epochs":200,"gradient_clip":5.0,'
    '"label_smoothing":0.05,"learning_rate":0.0008,"min_delta":0.0001,'
    '"noise_std":0.01,"patience":35,"reflection_probability":0.0,'
    '"reflection_weight":0.0,"seed":7,"segment_count":8,'
    '"segment_probability":0.5,"time_shift_samples":8,"weight_decay":0.0005}'
)
TRAIN_CONFIG_SHA256: Final = (
    "b88746eb01b60a2d2c314f6865113a40b8a64b3d90d60f1265e8c9ec83778364"
)
OPENED17_RULE_JSON: Final = (
    '{"candidate_must_strictly_exceed_primary_reference":true,'
    '"failure_action":"kill_candidate_before_disjoint_gate",'
    '"maximum_dataset_deficit":0.03,"minimum_nonnegative_dataset_deltas":3,'
    '"minimum_strict_subject_win_rate":0.5294117647058824,'
    '"primary_metric":"equal_dataset_balanced_accuracy",'
    '"primary_reference":"cardinal_fbc_micro_extended"}'
)
OPENED17_RULE_SHA256: Final = (
    "88402bb4974c699f896f0d3815f727436d2d1a954be6787fcac500e19ad26055"
)
WORKER_COUNT: Final = 3
EXPECTED_JOB_COUNT: Final = 17
MINIMUM_FREE_GIB: Final = 50.0
REQUIRED_CUBLAS_WORKSPACE_CONFIG: Final = ":4096:8"
UV_EXECUTABLE_ENVIRONMENT_VARIABLE: Final = "IEEE_MI_UV_EXECUTABLE"
ENVIRONMENT_SCHEMA: Final = "ieee-mi-uv-environment-v2"
REFERENCE_PLAN_SCHEMA: Final = "ieee-mi-opened-development-screen-plan-v1"
REFERENCE_RECORD_SCHEMA: Final = "ieee-mi-opened-development-validation-record-v1"
EXPECTED_REFERENCE_PLAN_SHA256: Final = (
    "57298f7c18b8841e741d72194cb601faf3ac71cb81b1209b1c3964ddf43c0fc6"
)
EXPECTED_REFERENCE_MODELS: Final = (
    "cardinal_fbc_micro_extended",
    "fbcnet",
    "tcformer",
)
EXPECTED_REFERENCE_RECORD_COUNT: Final = EXPECTED_JOB_COUNT * len(
    EXPECTED_REFERENCE_MODELS
)
PLAN_DIRECTORY: Final = "plan"
AUTHORITY_FENCE: Final = ".gauge-gate1.authority.lock"
CLAIMS_DIRECTORY: Final = "claims"
STAGING_DIRECTORY: Final = "staging"
RECORDS_DIRECTORY: Final = "records"
FORENSICS_DIRECTORY: Final = "forensics"
ANALYSIS_ROOT_DIRECTORY: Final = "analysis"
MAX_JSON_BYTES: Final = 16 * 1024 * 1024

GPU_UUID_RE = re.compile(
    r"^GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
NONCE_RE = re.compile(r"^[0-9a-f]{32}$")

# Explicit transitive runtime closure.  The approved screen imports the
# conditioned/robustness specification modules to obtain its frozen constants,
# while the Gauge factory imports the native Cardinal implementation.
SOURCE_FILES: Final = (
    "ieee_mi/__init__.py",
    "ieee_mi/gauge_gate1_runner.py",
    "ieee_mi/gauge_gate1_analysis.py",
    "ieee_mi/gauge_quotient_screen.py",
    "ieee_mi/gauge_quotient.py",
    "ieee_mi/project_gpu_leases.py",
    "ieee_mi/conditioned_screen.py",
    "ieee_mi/development_screen.py",
    "ieee_mi/robustness_screen.py",
    "ieee_mi/config.py",
    "ieee_mi/data.py",
    "ieee_mi/training.py",
    "ieee_mi/baselines.py",
    "ieee_mi/models.py",
    "ieee_mi/tcformer_source.py",
    "docs/GAUGE_QUOTIENT_NOVELTY_PLAN.md",
)
REQUIRED_PACKAGES: Final = (
    "braindecode",
    "einops",
    "mne",
    "moabb",
    "numpy",
    "pandas",
    "pyriemann",
    "scikit-learn",
    "scipy",
    "torch",
)
FORBIDDEN_ARTIFACT_KEYS = frozenset(
    {
        "labels",
        "targets",
        "ground_truth",
        "true_labels",
        "validation_labels",
        "y_true",
        "y_pred",
        "outcomes",
        "raw_outcomes",
        "samples",
        "predictions",
        "probabilities",
        "logits",
        "history",
        "checkpoint",
        "checkpoints",
    }
)
FORBIDDEN_SCOPE_TOKENS = frozenset({"test", "heldout"})


class GaugeGate1Error(RuntimeError):
    """A fail-closed protocol, identity, or publication violation."""


class ClaimUnavailable(GaugeGate1Error):
    """Another live worker owns the exact job."""


class ResourceUnavailable(GaugeGate1Error):
    """A shared-workstation resource guard refused new work."""


@dataclass(frozen=True)
class DirectorySnapshot:
    path: Path
    descriptor: int
    observed: os.stat_result


@dataclass
class NamespaceSnapshot:
    """Pinned exact ancestor chain from one declared stable root to a target."""

    root: Path
    directories: tuple[DirectorySnapshot, ...]
    target_path: Path
    target_observed: os.stat_result
    target_directory: bool
    closed: bool = False

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for item in reversed(self.directories):
            try:
                os.close(item.descriptor)
            except OSError:
                pass

    def revalidate(self) -> None:
        if self.closed:
            raise GaugeGate1Error("namespace snapshot is already closed")
        if not self.directories:
            raise GaugeGate1Error("namespace snapshot has no stable root")
        root = self.directories[0]
        if root.path != self.root:
            raise GaugeGate1Error("namespace stable-root binding changed")
        reopened_root = _open_absolute_directory(self.root)
        try:
            reopened = os.fstat(reopened_root)
        finally:
            os.close(reopened_root)
        if (
            _stat_fingerprint(reopened)
            != _stat_fingerprint(root.observed)
        ):
            raise GaugeGate1Error(
                f"namespace stable root changed: {self.root}"
            )
        for index, item in enumerate(self.directories):
            current = os.fstat(item.descriptor)
            if (
                _stat_fingerprint(current)
                != _stat_fingerprint(item.observed)
            ):
                raise GaugeGate1Error(
                    f"namespace ancestor changed: {item.path}"
                )
            if index:
                parent = self.directories[index - 1]
                named = os.stat(
                    item.path.name,
                    dir_fd=parent.descriptor,
                    follow_symlinks=False,
                )
                if (
                    not stat.S_ISDIR(named.st_mode)
                    or _stat_fingerprint(named)
                    != _stat_fingerprint(item.observed)
                ):
                    raise GaugeGate1Error(
                        f"namespace ancestor binding changed: {item.path}"
                    )
        parent = self.directories[-1]
        named_target = os.stat(
            self.target_path.name,
            dir_fd=parent.descriptor,
            follow_symlinks=False,
        )
        kind_matches = (
            stat.S_ISDIR(named_target.st_mode)
            if self.target_directory
            else stat.S_ISREG(named_target.st_mode)
        )
        if (
            not kind_matches
            or _stat_fingerprint(named_target)
            != _stat_fingerprint(self.target_observed)
        ):
            raise GaugeGate1Error(
                f"namespace target binding changed: {self.target_path}"
            )


@dataclass
class FileSnapshot:
    path: Path
    descriptor: int
    observed: os.stat_result
    payload: bytes
    namespace: NamespaceSnapshot | None = None

    def close(self) -> None:
        descriptor = self.descriptor
        self.descriptor = -1
        namespace = self.namespace
        self.namespace = None
        try:
            if descriptor >= 0:
                os.close(descriptor)
        except OSError:
            pass
        finally:
            if namespace is not None:
                namespace.close()


@dataclass
class MultiFileSnapshot:
    files: tuple[FileSnapshot, ...]

    def close(self) -> None:
        for item in self.files:
            item.close()

    def revalidate(self) -> None:
        for item in self.files:
            _assert_file_snapshot(item)

    def by_path(self) -> dict[Path, FileSnapshot]:
        return {item.path: item for item in self.files}


@dataclass
class PackageSnapshot:
    path: Path
    descriptor: int
    observed: os.stat_result
    fingerprint: tuple[Any, ...]
    children: dict[str, FileSnapshot]
    namespace: NamespaceSnapshot

    def close(self) -> None:
        for child in self.children.values():
            child.close()
        descriptor = self.descriptor
        self.descriptor = -1
        try:
            if descriptor >= 0:
                os.close(descriptor)
        except OSError:
            pass
        finally:
            self.namespace.close()

    def revalidate(self) -> None:
        if self.descriptor < 0:
            raise GaugeGate1Error("package snapshot is already closed")
        current = os.fstat(self.descriptor)
        if _stat_fingerprint(current) != _stat_fingerprint(self.observed):
            raise GaugeGate1Error(f"sealed package directory changed: {self.path}")
        self.namespace.revalidate()
        if _directory_fingerprint(self.descriptor) != self.fingerprint:
            raise GaugeGate1Error(f"sealed package entries changed: {self.path}")
        for child in self.children.values():
            _assert_file_snapshot_at(self.descriptor, child)


@dataclass(frozen=True)
class ClaimHandle:
    job: ScreenJob
    path: Path
    descriptor: int
    observed: os.stat_result
    value: Mapping[str, Any]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _mapping_sha256(value: Mapping[str, Any]) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _payload_sha256(value: Mapping[str, Any], key: str) -> str:
    detached = copy.deepcopy(dict(value))
    detached.pop(key, None)
    return _mapping_sha256(detached)


def _exact_keys(
    value: Any,
    expected: set[str] | frozenset[str],
    *,
    path: str,
) -> None:
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise GaugeGate1Error(f"{path} fields differ from the fixed schema")


def _require_int(value: Any, *, path: str, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise GaugeGate1Error(f"{path} must be an integer")
    if minimum is not None and value < minimum:
        raise GaugeGate1Error(f"{path} is below {minimum}")
    return value


def _require_number(
    value: Any,
    *,
    path: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        raise GaugeGate1Error(f"{path} must be a finite number")
    result = float(value)
    if minimum is not None and result < minimum:
        raise GaugeGate1Error(f"{path} is below {minimum}")
    if maximum is not None and result > maximum:
        raise GaugeGate1Error(f"{path} exceeds {maximum}")
    return result


def _strict_json_bytes(payload: bytes, *, source: str) -> dict[str, Any]:
    if len(payload) > MAX_JSON_BYTES:
        raise GaugeGate1Error(f"JSON artifact is too large: {source}")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise GaugeGate1Error(
                    f"duplicate JSON key {key!r} in {source}"
                )
            result[key] = value
        return result

    try:
        decoded = payload.decode("utf-8")
        value = json.loads(
            decoded,
            object_pairs_hook=unique,
            parse_constant=lambda token: (_ for _ in ()).throw(
                GaugeGate1Error(
                    f"non-finite JSON constant {token!r} in {source}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GaugeGate1Error(f"invalid JSON in {source}: {error}") from error
    if not isinstance(value, dict):
        raise GaugeGate1Error(f"{source} is not a JSON object")
    return value


def _assert_no_excluded_evidence(value: Any, path: str = "record") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = re.sub(
                r"[^a-z0-9]+", "_", str(key).lower()
            ).strip("_")
            tokens = frozenset(token for token in normalized.split("_") if token)
            if normalized in FORBIDDEN_ARTIFACT_KEYS or (
                tokens & FORBIDDEN_SCOPE_TOKENS
            ):
                raise GaugeGate1Error(
                    f"forbidden outcome/evaluation field at {path}.{key}"
                )
            _assert_no_excluded_evidence(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_no_excluded_evidence(child, f"{path}.{index}")


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _relative_beneath(root: Path, child: Path) -> Path:
    root_absolute = _absolute(root)
    child_absolute = _absolute(child)
    try:
        relative = child_absolute.relative_to(root_absolute)
    except ValueError as error:
        raise GaugeGate1Error(
            f"{child_absolute} is outside project root {root_absolute}"
        ) from error
    if not relative.parts or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise GaugeGate1Error("path must be a non-root project descendant")
    return relative


def _validate_absolute_nonroot(path: Path, *, label: str) -> Path:
    absolute = _absolute(path)
    if absolute == Path(absolute.anchor) or absolute.name in {"", ".", ".."}:
        raise GaugeGate1Error(f"{label} must be a non-root absolute path")
    return absolute


def _directory_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )


def _file_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )


def _open_absolute_directory(path: Path) -> int:
    absolute = _absolute(path)
    descriptor = os.open(absolute.anchor, _directory_flags())
    try:
        for component in absolute.parts[1:]:
            child = os.open(
                component,
                _directory_flags(),
                dir_fd=descriptor,
            )
            observed = os.fstat(child)
            if not stat.S_ISDIR(observed.st_mode):
                os.close(child)
                raise GaugeGate1Error(
                    f"path component is not a real directory: {absolute}"
                )
            os.close(descriptor)
            descriptor = child
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _open_absolute_file(path: Path) -> tuple[int, int]:
    absolute = _absolute(path)
    parent = _open_absolute_directory(absolute.parent)
    try:
        descriptor = os.open(absolute.name, _file_flags(), dir_fd=parent)
    except Exception:
        os.close(parent)
        raise
    return parent, descriptor


def _stat_fingerprint(observed: os.stat_result) -> tuple[int, ...]:
    return (
        int(observed.st_mode),
        int(observed.st_dev),
        int(observed.st_ino),
        int(observed.st_nlink),
        int(observed.st_uid),
        int(observed.st_gid),
        int(observed.st_size),
        int(observed.st_mtime_ns),
        int(observed.st_ctime_ns),
    )


def _open_namespaced_target(
    path: Path,
    *,
    namespace_root: Path,
    directory: bool,
) -> tuple[int, os.stat_result, NamespaceSnapshot]:
    """Open a target and retain its exact ancestor namespace to a stable root.

    Only directory components on the target's own relative path are pinned.
    Their full mutation fingerprints detect A->B->A restoration, while
    changes inside unrelated sibling subtrees do not alter this chain.
    """

    target = _absolute(path)
    root = _validate_absolute_nonroot(
        namespace_root,
        label="namespace stable root",
    )
    try:
        relative_parent = target.parent.relative_to(root)
    except ValueError as error:
        raise GaugeGate1Error(
            f"snapshot target {target} is outside stable root {root}"
        ) from error
    if any(part in {"", ".", ".."} for part in relative_parent.parts):
        raise GaugeGate1Error("snapshot ancestor chain is unsafe")

    directories: list[DirectorySnapshot] = []
    descriptor = _open_absolute_directory(root)
    target_descriptor: int | None = None
    try:
        current_path = root
        current_stat = os.fstat(descriptor)
        directories.append(
            DirectorySnapshot(
                path=current_path,
                descriptor=descriptor,
                observed=current_stat,
            )
        )
        descriptor = -1
        for component in relative_parent.parts:
            parent = directories[-1]
            child = os.open(
                component,
                _directory_flags(),
                dir_fd=parent.descriptor,
            )
            child_stat = os.fstat(child)
            if not stat.S_ISDIR(child_stat.st_mode):
                os.close(child)
                raise GaugeGate1Error(
                    f"snapshot ancestor is not a directory: {current_path / component}"
                )
            current_path /= component
            directories.append(
                DirectorySnapshot(
                    path=current_path,
                    descriptor=child,
                    observed=child_stat,
                )
            )
        parent = directories[-1]
        flags = _directory_flags() if directory else _file_flags()
        target_descriptor = os.open(
            target.name,
            flags,
            dir_fd=parent.descriptor,
        )
        target_stat = os.fstat(target_descriptor)
        kind_matches = (
            stat.S_ISDIR(target_stat.st_mode)
            if directory
            else stat.S_ISREG(target_stat.st_mode)
        )
        if not kind_matches:
            raise GaugeGate1Error(
                f"snapshot target has the wrong type: {target}"
            )
        namespace = NamespaceSnapshot(
            root=root,
            directories=tuple(directories),
            target_path=target,
            target_observed=target_stat,
            target_directory=directory,
        )
        namespace.revalidate()
        return target_descriptor, target_stat, namespace
    except Exception:
        if target_descriptor is not None:
            os.close(target_descriptor)
        for item in reversed(directories):
            try:
                os.close(item.descriptor)
            except OSError:
                pass
        if descriptor >= 0:
            os.close(descriptor)
        raise


def _read_descriptor(descriptor: int, *, maximum: int | None = None) -> bytes:
    observed = os.fstat(descriptor)
    size = int(observed.st_size)
    if maximum is not None and size > maximum:
        raise GaugeGate1Error("file exceeds the permitted byte limit")
    blocks: list[bytes] = []
    offset = 0
    while offset < size:
        block = os.pread(descriptor, min(1024 * 1024, size - offset), offset)
        if not block:
            raise GaugeGate1Error("file ended before its stat-reported size")
        blocks.append(block)
        offset += len(block)
    if os.fstat(descriptor).st_size != size:
        raise GaugeGate1Error("file size changed while it was read")
    return b"".join(blocks)


def _assert_path_binds_file(path: Path, observed: os.stat_result) -> None:
    parent = _open_absolute_directory(path.parent)
    try:
        current = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
    finally:
        os.close(parent)
    if (
        not stat.S_ISREG(current.st_mode)
        or (current.st_dev, current.st_ino)
        != (observed.st_dev, observed.st_ino)
    ):
        raise GaugeGate1Error(f"file name/inode binding changed: {path}")


def _assert_path_binds_directory(path: Path, observed: os.stat_result) -> None:
    parent = _open_absolute_directory(path.parent)
    try:
        current = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
    finally:
        os.close(parent)
    if (
        not stat.S_ISDIR(current.st_mode)
        or (current.st_dev, current.st_ino)
        != (observed.st_dev, observed.st_ino)
    ):
        raise GaugeGate1Error(f"directory name/inode binding changed: {path}")


def _open_file_snapshot(
    path: Path,
    *,
    maximum: int | None = None,
    namespace_root: Path | None = None,
    track_namespace: bool = True,
) -> FileSnapshot:
    absolute = _absolute(path)
    parent: int | None = None
    namespace: NamespaceSnapshot | None = None
    if track_namespace:
        descriptor, before, namespace = _open_namespaced_target(
            absolute,
            namespace_root=(
                absolute.parent if namespace_root is None else namespace_root
            ),
            directory=False,
        )
    else:
        parent, descriptor = _open_absolute_file(absolute)
        before = os.fstat(descriptor)
    try:
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise GaugeGate1Error(
                f"artifact must be a unique regular file: {path}"
            )
        payload = _read_descriptor(descriptor, maximum=maximum)
        after = os.fstat(descriptor)
        if _stat_fingerprint(after) != _stat_fingerprint(before):
            raise GaugeGate1Error(f"file changed while read: {path}")
        if namespace is not None:
            namespace.revalidate()
        else:
            assert parent is not None
            current = os.stat(
                absolute.name,
                dir_fd=parent,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISREG(current.st_mode)
                or _stat_fingerprint(current)
                != _stat_fingerprint(before)
            ):
                raise GaugeGate1Error(f"file name changed while read: {path}")
        return FileSnapshot(
            path=absolute,
            descriptor=descriptor,
            observed=before,
            payload=payload,
            namespace=namespace,
        )
    except Exception:
        os.close(descriptor)
        if namespace is not None:
            namespace.close()
        raise
    finally:
        if parent is not None:
            os.close(parent)


def _assert_file_snapshot(snapshot: FileSnapshot) -> None:
    if snapshot.descriptor < 0:
        raise GaugeGate1Error(f"file snapshot is already closed: {snapshot.path}")
    current = os.fstat(snapshot.descriptor)
    if _stat_fingerprint(current) != _stat_fingerprint(snapshot.observed):
        raise GaugeGate1Error(f"opened file changed: {snapshot.path}")
    if snapshot.namespace is not None:
        snapshot.namespace.revalidate()
    else:
        _assert_path_binds_file(snapshot.path, snapshot.observed)
    if _read_descriptor(snapshot.descriptor) != snapshot.payload:
        raise GaugeGate1Error(f"opened file bytes changed: {snapshot.path}")


def _assert_file_snapshot_at(parent: int, snapshot: FileSnapshot) -> None:
    current = os.fstat(snapshot.descriptor)
    if _stat_fingerprint(current) != _stat_fingerprint(snapshot.observed):
        raise GaugeGate1Error(f"sealed child changed: {snapshot.path.name}")
    named = os.stat(snapshot.path.name, dir_fd=parent, follow_symlinks=False)
    if (
        not stat.S_ISREG(named.st_mode)
        or (named.st_dev, named.st_ino)
        != (snapshot.observed.st_dev, snapshot.observed.st_ino)
    ):
        raise GaugeGate1Error(
            f"sealed child name/inode changed: {snapshot.path.name}"
        )
    if _read_descriptor(snapshot.descriptor) != snapshot.payload:
        raise GaugeGate1Error(f"sealed child bytes changed: {snapshot.path.name}")


def _open_multi_file_snapshot(
    paths: Sequence[Path],
    *,
    json_limit: bool = False,
    namespace_root: Path | None = None,
) -> MultiFileSnapshot:
    absolute = tuple(_absolute(path) for path in paths)
    if not absolute or len(absolute) != len(set(absolute)):
        raise GaugeGate1Error("snapshot path set is empty or duplicated")
    opened: list[FileSnapshot] = []
    try:
        # All descriptors remain open before any snapshot is accepted.  A
        # later exact path-binding pass makes this one coherent file set.
        for path in absolute:
            opened.append(
                _open_file_snapshot(
                    path,
                    maximum=MAX_JSON_BYTES if json_limit else None,
                    namespace_root=namespace_root,
                )
            )
        snapshot = MultiFileSnapshot(tuple(opened))
        snapshot.revalidate()
        return snapshot
    except Exception:
        for item in opened:
            item.close()
        raise


def _directory_fingerprint(descriptor: int) -> tuple[Any, ...]:
    directory = os.fstat(descriptor)
    names = sorted(os.listdir(descriptor))
    entries: list[tuple[Any, ...]] = []
    for name in names:
        observed = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        entries.append((name, *_stat_fingerprint(observed)))
    return (_stat_fingerprint(directory), tuple(entries))


def _open_package(
    path: Path,
    *,
    expected_names: frozenset[str],
    namespace_root: Path | None = None,
) -> PackageSnapshot:
    absolute = _absolute(path)
    descriptor, observed, namespace = _open_namespaced_target(
        absolute,
        namespace_root=(
            absolute.parent if namespace_root is None else namespace_root
        ),
        directory=True,
    )
    children: dict[str, FileSnapshot] = {}
    try:
        if not stat.S_ISDIR(observed.st_mode) or observed.st_mode & 0o222:
            raise GaugeGate1Error(f"package is not a sealed directory: {path}")
        first = _directory_fingerprint(descriptor)
        names = frozenset(os.listdir(descriptor))
        if names != expected_names:
            raise GaugeGate1Error(f"package member set differs: {path}")
        for name in sorted(expected_names):
            child_descriptor = os.open(name, _file_flags(), dir_fd=descriptor)
            child_stat = os.fstat(child_descriptor)
            if (
                not stat.S_ISREG(child_stat.st_mode)
                or child_stat.st_nlink != 1
                or child_stat.st_mode & 0o222
            ):
                os.close(child_descriptor)
                raise GaugeGate1Error(
                    f"package member is not sealed and unique: {path / name}"
                )
            payload = _read_descriptor(
                child_descriptor,
                maximum=MAX_JSON_BYTES,
            )
            children[name] = FileSnapshot(
                path=_absolute(absolute / name),
                descriptor=child_descriptor,
                observed=child_stat,
                payload=payload,
            )
        second = _directory_fingerprint(descriptor)
        if second != first:
            raise GaugeGate1Error(f"package changed while opened: {path}")
        package = PackageSnapshot(
            path=absolute,
            descriptor=descriptor,
            observed=observed,
            fingerprint=first,
            children=children,
            namespace=namespace,
        )
        package.revalidate()
        return package
    except Exception:
        for child in children.values():
            child.close()
        os.close(descriptor)
        namespace.close()
        raise


def _fsync_directory(path: Path) -> None:
    descriptor = _open_absolute_directory(path)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _linux_cross_parent_sealed_stage_authority(
    source: Path,
    destination: Path,
) -> Iterator[None]:
    """Temporarily make one private sealed stage movable on Linux.

    Linux requires write permission on a directory when a cross-parent rename
    updates its ``..`` entry.  Gate 1 seals private package stages to ``0555``
    before publication, so ext4 correctly rejects the otherwise valid
    ``renameat2(RENAME_NOREPLACE)`` with ``EACCES``.  Hold the exact stage
    inode open, add only owner-write for the syscall, and restore the original
    sealed mode through that descriptor after success or failure.

    The exception is deliberately unavailable to arbitrary directories: it
    applies only to an exact ``.stage.*`` child of an owner-only ``staging``
    directory, owned by this effective user, with mode exactly ``0555``.
    Same-parent renames and non-directory entries never need this authority.
    """

    source_absolute = _absolute(source)
    destination_absolute = _absolute(destination)
    source_parent = _open_absolute_directory(source_absolute.parent)
    destination_parent = _open_absolute_directory(destination_absolute.parent)
    stage_descriptor: int | None = None
    original_mode: int | None = None
    permission_widened = False
    try:
        source_parent_stat = os.fstat(source_parent)
        destination_parent_stat = os.fstat(destination_parent)
        if (
            source_parent_stat.st_dev,
            source_parent_stat.st_ino,
        ) == (
            destination_parent_stat.st_dev,
            destination_parent_stat.st_ino,
        ):
            yield
            return

        source_stat = os.stat(
            source_absolute.name,
            dir_fd=source_parent,
            follow_symlinks=False,
        )
        if not stat.S_ISDIR(source_stat.st_mode) or source_stat.st_mode & 0o222:
            yield
            return
        if (
            source_absolute.parent.name != STAGING_DIRECTORY
            or not source_absolute.name.startswith(".stage.")
        ):
            raise GaugeGate1Error(
                "cross-parent sealed directory is not a private package stage"
            )
        if (
            source_parent_stat.st_uid != os.geteuid()
            or stat.S_IMODE(source_parent_stat.st_mode) & 0o077
        ):
            raise GaugeGate1Error(
                "cross-parent sealed stage parent is not owner-private"
            )

        stage_descriptor = os.open(
            source_absolute.name,
            _directory_flags(),
            dir_fd=source_parent,
        )
        pinned = os.fstat(stage_descriptor)
        if (
            not stat.S_ISDIR(pinned.st_mode)
            or (pinned.st_dev, pinned.st_ino)
            != (source_stat.st_dev, source_stat.st_ino)
        ):
            raise GaugeGate1Error(
                "cross-parent sealed stage name/inode binding changed"
            )
        original_mode = stat.S_IMODE(pinned.st_mode)
        if pinned.st_uid != os.geteuid() or original_mode != 0o555:
            raise GaugeGate1Error(
                "cross-parent package stage is not an owned 0555 directory"
            )

        os.fchmod(stage_descriptor, original_mode | stat.S_IWUSR)
        permission_widened = True
        widened = os.fstat(stage_descriptor)
        rebound = os.stat(
            source_absolute.name,
            dir_fd=source_parent,
            follow_symlinks=False,
        )
        if (
            stat.S_IMODE(widened.st_mode) != original_mode | stat.S_IWUSR
            or not stat.S_ISDIR(rebound.st_mode)
            or (rebound.st_dev, rebound.st_ino)
            != (pinned.st_dev, pinned.st_ino)
        ):
            raise GaugeGate1Error(
                "cross-parent sealed stage changed while granting move authority"
            )
        yield
    finally:
        try:
            if (
                stage_descriptor is not None
                and permission_widened
                and original_mode is not None
            ):
                os.fchmod(stage_descriptor, original_mode)
                os.fsync(stage_descriptor)
                restored = os.fstat(stage_descriptor)
                if stat.S_IMODE(restored.st_mode) != original_mode:
                    raise GaugeGate1Error(
                        "cross-parent package stage could not be resealed"
                    )
        finally:
            if stage_descriptor is not None:
                os.close(stage_descriptor)
            os.close(destination_parent)
            os.close(source_parent)


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically rename without replacement on Linux or macOS."""

    source_bytes = os.fsencode(_absolute(source))
    destination_bytes = os.fsencode(_absolute(destination))
    libc = ctypes.CDLL(None, use_errno=True)
    error_code = 0
    if sys.platform.startswith("linux"):
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise GaugeGate1Error("renameat2 is unavailable")
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        with _linux_cross_parent_sealed_stage_authority(source, destination):
            result = renameat2(
                -100,
                source_bytes,
                -100,
                destination_bytes,
                1,  # RENAME_NOREPLACE
            )
            if result != 0:
                error_code = ctypes.get_errno()
    elif sys.platform == "darwin":
        # APFS refuses to rename a non-empty directory after its own write
        # bits are removed, even though POSIX rename authority normally comes
        # from the parent.  The formal runner is CUDA/NVIDIA/Linux-only.  This
        # narrowly named opt-in exists solely so the Mac can exercise crash
        # and concurrency semantics with synthetic fixtures; it is rejected
        # by every formal preflight.
        synthetic = os.environ.get("IEEE_MI_DARWIN_SYNTHETIC_TEST") == "1"
        if not synthetic:
            raise GaugeGate1Error(
                "Darwin directory publication is synthetic-test-only"
            )
        renamex = getattr(libc, "renamex_np", None)
        if renamex is None:
            raise GaugeGate1Error("renamex_np is unavailable")
        renamex.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        renamex.restype = ctypes.c_int
        source_stat = os.lstat(source)
        directory_descriptor: int | None = None
        original_mode = stat.S_IMODE(source_stat.st_mode)
        if stat.S_ISDIR(source_stat.st_mode) and not original_mode & 0o200:
            directory_descriptor = os.open(source, _directory_flags())
            os.fchmod(directory_descriptor, original_mode | 0o200)
        try:
            result = renamex(
                source_bytes,
                destination_bytes,
                0x00000004,  # RENAME_EXCL
            )
            if result != 0:
                error_code = ctypes.get_errno()
        finally:
            if directory_descriptor is not None:
                os.fchmod(directory_descriptor, original_mode)
                os.fsync(directory_descriptor)
                os.close(directory_descriptor)
    else:
        raise GaugeGate1Error(
            "formal no-replace directory publication needs Linux or macOS"
        )
    if result != 0:
        if error_code in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(
                error_code,
                os.strerror(error_code),
                str(destination),
            )
        raise OSError(
            error_code,
            os.strerror(error_code),
            str(destination),
        )


def _secure_mkdir_beneath(
    root: Path,
    relative: Path,
    *,
    mode: int = 0o700,
) -> Path:
    root_absolute = _absolute(root)
    root_descriptor = _open_absolute_directory(root_absolute)
    descriptor = root_descriptor
    current = root_absolute
    try:
        if relative.is_absolute() or not relative.parts or any(
            part in {"", ".", ".."} for part in relative.parts
        ):
            raise GaugeGate1Error("invalid project-relative directory")
        for component in relative.parts:
            created = False
            try:
                child = os.open(
                    component,
                    _directory_flags(),
                    dir_fd=descriptor,
                )
            except FileNotFoundError:
                try:
                    os.mkdir(component, mode, dir_fd=descriptor)
                    created = True
                except FileExistsError:
                    pass
                child = os.open(
                    component,
                    _directory_flags(),
                    dir_fd=descriptor,
                )
            if created:
                try:
                    # Make every nested parent durable before any immutable
                    # plan can attest that workers will only validate it.
                    os.fsync(child)
                    os.fsync(descriptor)
                except Exception:
                    os.close(child)
                    raise
            observed = os.fstat(child)
            if not stat.S_ISDIR(observed.st_mode):
                os.close(child)
                raise GaugeGate1Error("directory component is not real")
            if descriptor != root_descriptor:
                os.close(descriptor)
            descriptor = child
            current /= component
        return current
    finally:
        if descriptor != root_descriptor:
            os.close(descriptor)
        os.close(root_descriptor)


def _validate_directory_beneath(root: Path, relative: Path) -> Path:
    """Open an existing real directory chain without creating any component."""

    root_absolute = _absolute(root)
    root_descriptor = _open_absolute_directory(root_absolute)
    descriptor = root_descriptor
    current = root_absolute
    try:
        if relative.is_absolute() or not relative.parts or any(
            part in {"", ".", ".."} for part in relative.parts
        ):
            raise GaugeGate1Error("invalid project-relative directory")
        for component in relative.parts:
            try:
                child = os.open(
                    component,
                    _directory_flags(),
                    dir_fd=descriptor,
                )
            except OSError as error:
                raise GaugeGate1Error(
                    f"required directory component is absent or unsafe: "
                    f"{current / component}"
                ) from error
            observed = os.fstat(child)
            if not stat.S_ISDIR(observed.st_mode):
                os.close(child)
                raise GaugeGate1Error("directory component is not real")
            if descriptor != root_descriptor:
                os.close(descriptor)
            descriptor = child
            current /= component
        return current
    finally:
        if descriptor != root_descriptor:
            os.close(descriptor)
        os.close(root_descriptor)


def _ensure_run_layout(project_root: Path, run_root: Path) -> None:
    """Create only the requested run leaf and its private subdirectories.

    The real lab layout intentionally keeps ``source``, ``runs``, and
    ``data_cache`` as siblings.  The project root remains the GPU-lease/source
    authority, while the run root is an independently verified absolute tree.
    """

    del project_root
    run_root = _validate_absolute_nonroot(run_root, label="run root")
    parent = _open_absolute_directory(run_root.parent)
    try:
        try:
            os.mkdir(run_root.name, 0o700, dir_fd=parent)
        except FileExistsError:
            pass
        child = os.open(run_root.name, _directory_flags(), dir_fd=parent)
        try:
            if not stat.S_ISDIR(os.fstat(child).st_mode):
                raise GaugeGate1Error("run root is not a real directory")
        finally:
            os.close(child)
        os.fsync(parent)
    finally:
        os.close(parent)
    for name in (
        CLAIMS_DIRECTORY,
        STAGING_DIRECTORY,
        RECORDS_DIRECTORY,
        FORENSICS_DIRECTORY,
        ANALYSIS_ROOT_DIRECTORY,
    ):
        _secure_mkdir_beneath(run_root, Path(name))


def preflight_launch_layout(
    *,
    project_root: Path,
    run_root: Path,
    reference_root: Path,
    cache_root: Path | None = None,
) -> dict[str, Any]:
    """Read-only verification of the sibling-root lab launch layout."""

    roots = {
        "project_root": _validate_absolute_nonroot(
            project_root, label="project root"
        ),
        "reference_root": _validate_absolute_nonroot(
            reference_root, label="reference root"
        ),
    }
    if cache_root is not None:
        roots["cache_root"] = _validate_absolute_nonroot(
            cache_root, label="cache root"
        )
    run = _validate_absolute_nonroot(run_root, label="run root")
    if len(set(roots.values()) | {run}) != len(roots) + 1:
        raise GaugeGate1Error("source, reference, cache, and run roots must differ")
    identity: dict[str, Any] = {}
    for name, path in roots.items():
        descriptor = _open_absolute_directory(path)
        try:
            observed = os.fstat(descriptor)
            identity[name] = {
                "path": str(path),
                "st_dev": int(observed.st_dev),
                "st_ino": int(observed.st_ino),
            }
        finally:
            os.close(descriptor)
    parent = _open_absolute_directory(run.parent)
    try:
        parent_stat = os.fstat(parent)
        identity["run_parent"] = {
            "path": str(run.parent),
            "st_dev": int(parent_stat.st_dev),
            "st_ino": int(parent_stat.st_ino),
        }
        try:
            observed_run = os.stat(
                run.name,
                dir_fd=parent,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            identity["run_root"] = {
                "path": str(run),
                "exists": False,
            }
        else:
            if not stat.S_ISDIR(observed_run.st_mode):
                raise GaugeGate1Error("existing run root is not a real directory")
            identity["run_root"] = {
                "path": str(run),
                "exists": True,
                "st_dev": int(observed_run.st_dev),
                "st_ino": int(observed_run.st_ino),
            }
    finally:
        os.close(parent)
    return identity


def _open_or_create_authority_fence(run_root: Path) -> tuple[int, os.stat_result]:
    parent = _open_absolute_directory(run_root)
    try:
        flags = (
            os.O_RDWR
            | os.O_CREAT
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        descriptor = os.open(AUTHORITY_FENCE, flags, 0o600, dir_fd=parent)
        observed = os.fstat(descriptor)
        if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
            os.close(descriptor)
            raise GaugeGate1Error("authority fence is not a unique regular file")
        current = os.stat(
            AUTHORITY_FENCE,
            dir_fd=parent,
            follow_symlinks=False,
        )
        if (current.st_dev, current.st_ino) != (
            observed.st_dev,
            observed.st_ino,
        ):
            os.close(descriptor)
            raise GaugeGate1Error("authority fence name/inode changed")
        return descriptor, observed
    finally:
        os.close(parent)


def _assert_authority_fence(
    run_root: Path,
    expected: Mapping[str, Any],
) -> None:
    parent = _open_absolute_directory(run_root)
    try:
        descriptor = os.open(
            AUTHORITY_FENCE,
            _file_flags(),
            dir_fd=parent,
        )
        try:
            observed = os.fstat(descriptor)
            named = os.stat(
                AUTHORITY_FENCE,
                dir_fd=parent,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISREG(observed.st_mode)
                or observed.st_nlink != 1
                or (observed.st_dev, observed.st_ino)
                != (named.st_dev, named.st_ino)
                or dict(expected)
                != {
                    "st_dev": int(observed.st_dev),
                    "st_ino": int(observed.st_ino),
                }
            ):
                raise GaugeGate1Error(
                    "authority fence differs from immutable plan"
                )
        finally:
            os.close(descriptor)
    finally:
        os.close(parent)


@contextmanager
def _authority_lock(
    run_root: Path,
    *,
    expected: Mapping[str, Any] | None = None,
) -> Iterator[tuple[int, os.stat_result]]:
    descriptor, observed = _open_or_create_authority_fence(run_root)
    key = {
        "st_dev": int(observed.st_dev),
        "st_ino": int(observed.st_ino),
    }
    if expected is not None and dict(expected) != key:
        os.close(descriptor)
        raise GaugeGate1Error("authority fence differs from immutable plan")
    fcntl.flock(descriptor, fcntl.LOCK_EX)
    try:
        parent = _open_absolute_directory(run_root)
        try:
            current = os.stat(
                AUTHORITY_FENCE,
                dir_fd=parent,
                follow_symlinks=False,
            )
        finally:
            os.close(parent)
        if (current.st_dev, current.st_ino) != (
            observed.st_dev,
            observed.st_ino,
        ):
            raise GaugeGate1Error("locked authority fence was replaced")
        yield descriptor, observed
        current_fd = os.fstat(descriptor)
        if _stat_fingerprint(current_fd) != _stat_fingerprint(observed):
            raise GaugeGate1Error("locked authority fence changed")
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _write_file_exclusive(path: Path, payload: bytes, mode: int) -> os.stat_result:
    parent = _open_absolute_directory(path.parent)
    try:
        descriptor = os.open(
            path.name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            mode,
            dir_fd=parent,
        )
        try:
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise GaugeGate1Error("short write")
                view = view[written:]
            os.fsync(descriptor)
            os.fchmod(descriptor, mode)
            observed = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        os.fsync(parent)
        return observed
    finally:
        os.close(parent)


def _seal_tree(path: Path) -> None:
    """Seal a private stage after all writes, before authoritative rename."""

    for root, directories, files in os.walk(path, topdown=False):
        root_path = Path(root)
        for filename in files:
            target = root_path / filename
            if target.is_symlink():
                raise GaugeGate1Error("cannot seal a symlink")
            os.chmod(target, 0o444)
        for directory in directories:
            target = root_path / directory
            if target.is_symlink():
                raise GaugeGate1Error("cannot seal a symlink")
            os.chmod(target, 0o555)
        os.chmod(root_path, 0o555)


def _stage_package(
    *,
    staging_parent: Path,
    basename: str,
    members: Mapping[str, bytes],
) -> tuple[Path, os.stat_result]:
    nonce = uuid.uuid4().hex
    stage = staging_parent / f".stage.{basename}.{os.getpid()}.{nonce}"
    parent = _open_absolute_directory(staging_parent)
    try:
        os.mkdir(stage.name, 0o700, dir_fd=parent)
        os.fsync(parent)
    finally:
        os.close(parent)
    for name, payload in sorted(members.items()):
        if "/" in name or name in {"", ".", ".."}:
            raise GaugeGate1Error("invalid package member name")
        _write_file_exclusive(stage / name, payload, 0o600)
    _fsync_directory(stage)
    _seal_tree(stage)
    _fsync_directory(stage)
    _fsync_directory(staging_parent)
    observed = stage.lstat()
    if not stat.S_ISDIR(observed.st_mode):
        raise GaugeGate1Error("stage stopped being a directory")
    return stage, observed


def _entry_matches(path: Path, expected: os.stat_result, *, directory: bool) -> bool:
    try:
        observed = path.lstat()
    except FileNotFoundError:
        return False
    kind = stat.S_ISDIR(observed.st_mode) if directory else stat.S_ISREG(
        observed.st_mode
    )
    return kind and (observed.st_dev, observed.st_ino) == (
        expected.st_dev,
        expected.st_ino,
    )


def _hide_exact_entry(
    path: Path,
    expected: os.stat_result,
    *,
    label: str,
    directory: bool,
) -> Path:
    if not _entry_matches(path, expected, directory=directory):
        raise GaugeGate1Error(
            f"refusing to quarantine an entry without its exact inode: {path}"
        )
    hidden = path.parent / (
        f".{label}.{path.name}.{os.getpid()}.{uuid.uuid4().hex}"
    )
    _rename_noreplace(path, hidden)
    if not _entry_matches(hidden, expected, directory=directory):
        raise GaugeGate1Error("quarantine rename lost exact inode binding")
    _fsync_directory(path.parent)
    return hidden


def _publish_package(
    *,
    stage: Path,
    stage_stat: os.stat_result,
    destination: Path,
    expected_names: frozenset[str],
    validator: Any,
    namespace_root: Path | None = None,
) -> PackageSnapshot:
    published = False
    try:
        try:
            _rename_noreplace(stage, destination)
            published = True
        except Exception:
            # Covers a rename implementation/fault injector that moved the
            # inode and then raised.
            published = _entry_matches(
                destination, stage_stat, directory=True
            )
            raise
        _fsync_directory(destination.parent)
        if stage.parent != destination.parent:
            _fsync_directory(stage.parent)
        snapshot = _open_package(
            destination,
            expected_names=expected_names,
            namespace_root=namespace_root,
        )
        try:
            if (
                snapshot.observed.st_dev,
                snapshot.observed.st_ino,
            ) != (stage_stat.st_dev, stage_stat.st_ino):
                raise GaugeGate1Error("published package inode differs from stage")
            validator(snapshot)
            snapshot.revalidate()
            return snapshot
        except Exception:
            snapshot.close()
            raise
    except Exception:
        if published or _entry_matches(destination, stage_stat, directory=True):
            _hide_exact_entry(
                destination,
                stage_stat,
                label="invalid-postrename",
                directory=True,
            )
        raise


def _job_key(job: ScreenJob) -> str:
    return f"{job.dataset}:s{job.subject:03d}:f{job.fold:02d}"


def _subject_key(job: ScreenJob) -> str:
    return f"{job.dataset}:s{job.subject:03d}"


def _job_from_mapping(value: Mapping[str, Any]) -> ScreenJob:
    _exact_keys(
        value,
        {"dataset", "model", "subject", "fold", "seed"},
        path="job",
    )
    if type(value["dataset"]) is not str or type(value["model"]) is not str:
        raise GaugeGate1Error("job string fields have invalid types")
    return ScreenJob(
        dataset=value["dataset"],
        model=value["model"],
        subject=_require_int(value["subject"], path="job.subject", minimum=1),
        fold=_require_int(value["fold"], path="job.fold", minimum=0),
        seed=_require_int(value["seed"], path="job.seed", minimum=0),
    )


def _candidate_manifest() -> tuple[ScreenJob, ...]:
    jobs = tuple(
        ScreenJob(
            dataset=dataset,
            model=MODEL_NAME,
            subject=subject,
            fold=0,
            seed=7,
        )
        for dataset, subjects in CURATED_SPLITS
        for subject in subjects
    )
    if (
        len(jobs) != EXPECTED_JOB_COUNT
        or len({job.job_id for job in jobs}) != EXPECTED_JOB_COUNT
        or any(job.model != MODEL_NAME for job in jobs)
        or any(job.fold != 0 or job.seed != 7 for job in jobs)
    ):
        raise GaugeGate1Error("approved Gauge candidate job set changed")
    return jobs


def _approved_reference_record_paths(
    reference_root: Path,
) -> tuple[Path, ...]:
    return tuple(
        reference_root
        / _reference_record_relative(
            ScreenJob(
                dataset=job.dataset,
                model=model,
                subject=job.subject,
                fold=job.fold,
                seed=job.seed,
            )
        )
        for model in EXPECTED_REFERENCE_MODELS
        for job in _candidate_manifest()
    )


def _assert_approved_screen_dependency() -> None:
    """Cross-check local frozen copies against the approved specification.

    This is called on formal plan creation and worker startup.  Keeping imports
    lazy lets the persistence/decision tests run on a CPU-only Mac while the
    formal lab path still proves byte-level semantic agreement with the
    torch-backed approved Gauge modules.
    """

    from .gauge_quotient import (
        MAXIMUM_PARAMETERS as APPROVED_MAXIMUM_PARAMETERS,
    )
    from .gauge_quotient import MODEL_NAME as APPROVED_MODEL_NAME
    from .gauge_quotient_screen import (
        MODEL_FACTORY as APPROVED_MODEL_FACTORY,
        OPENED17_RULE_JSON as APPROVED_RULE_JSON,
        OPENED17_RULE_SHA256 as APPROVED_RULE_SHA256,
        OPENED17_STAGE as APPROVED_STAGE,
        STAGE_REFERENCE_PLAN_SHA256,
        TRAIN_CONFIG_JSON as APPROVED_TRAIN_JSON,
        TRAIN_CONFIG_SHA256 as APPROVED_TRAIN_SHA256,
        candidate_jobs as approved_candidate_jobs,
        reference_record_paths as approved_reference_paths,
    )

    checks = (
        APPROVED_MAXIMUM_PARAMETERS == MAXIMUM_PARAMETERS,
        APPROVED_MODEL_NAME == MODEL_NAME,
        APPROVED_MODEL_FACTORY == MODEL_FACTORY,
        APPROVED_RULE_JSON == OPENED17_RULE_JSON,
        APPROVED_RULE_SHA256 == OPENED17_RULE_SHA256,
        APPROVED_STAGE == OPENED17_STAGE,
        STAGE_REFERENCE_PLAN_SHA256[APPROVED_STAGE]
        == EXPECTED_REFERENCE_PLAN_SHA256,
        APPROVED_TRAIN_JSON == TRAIN_CONFIG_JSON,
        APPROVED_TRAIN_SHA256 == TRAIN_CONFIG_SHA256,
        approved_candidate_jobs(APPROVED_STAGE) == _candidate_manifest(),
    )
    if not all(checks):
        raise GaugeGate1Error(
            "local Gate 1 constants differ from approved Gauge specification"
        )
    sentinel = Path("/approved-reference-root")
    if approved_reference_paths(APPROVED_STAGE, sentinel) != (
        _approved_reference_record_paths(sentinel)
    ):
        raise GaugeGate1Error(
            "local comparator mapping differs from approved Gauge specification"
        )


def _validate_sha(value: Any, *, path: str) -> str:
    if type(value) is not str or not SHA256_RE.fullmatch(value):
        raise GaugeGate1Error(f"{path} is not a lowercase SHA-256")
    return value


def _validate_cache_identity_maps(
    cache_identity: Any,
    split_identity: Any,
) -> None:
    jobs = _candidate_manifest()
    expected_subjects = {_subject_key(job) for job in jobs}
    expected_splits = {_job_key(job) for job in jobs}
    if not isinstance(cache_identity, Mapping) or set(cache_identity) != expected_subjects:
        raise GaugeGate1Error("cache identity does not cover exact Gate 1 subjects")
    if not isinstance(split_identity, Mapping) or set(split_identity) != expected_splits:
        raise GaugeGate1Error("split identity does not cover exact Gate 1 jobs")
    for key, value in cache_identity.items():
        _exact_keys(
            value,
            {
                "relative_path",
                "file_size_bytes",
                "file_sha256",
                "array_sha256",
            },
            path=f"cache_identity.{key}",
        )
        relative = value["relative_path"]
        if (
            type(relative) is not str
            or Path(relative).is_absolute()
            or any(part in {"", ".", ".."} for part in Path(relative).parts)
            or not relative.endswith(".npz")
        ):
            raise GaugeGate1Error("cache relative path is unsafe")
        _require_int(
            value["file_size_bytes"],
            path=f"cache_identity.{key}.file_size_bytes",
            minimum=1,
        )
        _validate_sha(value["file_sha256"], path=f"cache_identity.{key}.file_sha256")
        _validate_sha(value["array_sha256"], path=f"cache_identity.{key}.array_sha256")
    for key, value in split_identity.items():
        _exact_keys(
            value,
            {"train", "validation"},
            path=f"split_identity.{key}",
        )
        for partition in ("train", "validation"):
            item = value[partition]
            _exact_keys(
                item,
                {"count", "rows_sha256"},
                path=f"split_identity.{key}.{partition}",
            )
            _require_int(
                item["count"],
                path=f"split_identity.{key}.{partition}.count",
                minimum=1,
            )
            _validate_sha(
                item["rows_sha256"],
                path=f"split_identity.{key}.{partition}.rows_sha256",
            )


def _validate_source_identity(value: Any) -> None:
    if not isinstance(value, Mapping) or set(value) != set(SOURCE_FILES):
        raise GaugeGate1Error("source identity differs from explicit closure")
    for path, identity in value.items():
        _exact_keys(
            identity,
            {"sha256", "size_bytes"},
            path=f"source_identity.{path}",
        )
        _validate_sha(identity["sha256"], path=f"source_identity.{path}.sha256")
        _require_int(
            identity["size_bytes"],
            path=f"source_identity.{path}.size_bytes",
            minimum=1,
        )


def _validate_environment_identity(value: Any) -> None:
    _exact_keys(
        value,
        {
            "schema",
            "python",
            "python_executable",
            "python_prefix",
            "platform",
            "uv_binding",
            "uv_executable",
            "uv_executable_sha256",
            "uv_version",
            "pyvenv_cfg_sha256",
            "uv_pip_freeze",
            "uv_pip_freeze_sha256",
            "required_packages",
            "torch_cuda_version",
            "torch_cudnn_version",
            "nvidia_driver_versions",
            "required_cublas_workspace_config",
            "cpu_threads_per_worker",
            "minimum_free_gib",
        },
        path="environment_identity",
    )
    if value["schema"] != ENVIRONMENT_SCHEMA:
        raise GaugeGate1Error("environment schema changed")
    if value["uv_binding"] != {
        "kind": "required_absolute_environment_variable",
        "name": UV_EXECUTABLE_ENVIRONMENT_VARIABLE,
    }:
        raise GaugeGate1Error("UV executable binding contract changed")
    for name in (
        "python",
        "python_executable",
        "python_prefix",
        "platform",
        "uv_executable",
        "uv_version",
    ):
        if type(value[name]) is not str or not value[name]:
            raise GaugeGate1Error(f"environment {name} is invalid")
    uv_path = Path(value["uv_executable"])
    if (
        not uv_path.is_absolute()
        or str(uv_path) != str(_absolute(uv_path))
    ):
        raise GaugeGate1Error("environment UV executable is not canonical absolute")
    for name in (
        "uv_executable_sha256",
        "pyvenv_cfg_sha256",
        "uv_pip_freeze_sha256",
    ):
        _validate_sha(value[name], path=f"environment_identity.{name}")
    freeze = value["uv_pip_freeze"]
    if (
        not isinstance(freeze, list)
        or not freeze
        or any(type(item) is not str or not item for item in freeze)
        or freeze != sorted(freeze, key=str.casefold)
        or len(freeze) != len(set(freeze))
    ):
        raise GaugeGate1Error("UV freeze is empty, unsorted, or duplicated")
    if value["uv_pip_freeze_sha256"] != _sha256_bytes(
        "\n".join(freeze).encode("utf-8")
    ):
        raise GaugeGate1Error("UV freeze digest differs")
    packages = value["required_packages"]
    if (
        not isinstance(packages, Mapping)
        or set(packages) != set(REQUIRED_PACKAGES)
        or any(type(item) is not str or not item for item in packages.values())
    ):
        raise GaugeGate1Error("required package identity changed")
    drivers = value["nvidia_driver_versions"]
    if (
        not isinstance(drivers, list)
        or not drivers
        or drivers != sorted(set(drivers))
        or any(type(item) is not str or not item for item in drivers)
    ):
        raise GaugeGate1Error("NVIDIA driver identity is invalid")
    if (
        value["required_cublas_workspace_config"]
        != REQUIRED_CUBLAS_WORKSPACE_CONFIG
        or value["cpu_threads_per_worker"] != CPU_THREADS
        or value["minimum_free_gib"] != MINIMUM_FREE_GIB
    ):
        raise GaugeGate1Error("environment safety constants changed")
    if type(value["torch_cuda_version"]) is not str:
        raise GaugeGate1Error("torch CUDA version is invalid")
    if type(value["torch_cudnn_version"]) not in {int, type(None)}:
        raise GaugeGate1Error("torch cuDNN version is invalid")


def _validate_reference_identity(value: Any) -> None:
    _exact_keys(
        value,
        {
            "schema",
            "access",
            "root",
            "plan",
            "records",
            "record_count",
            "snapshot_sha256",
        },
        path="reference_identity",
    )
    if (
        value["schema"] != "ieee-mi-gauge-gate1-reference-snapshot-v1"
        or value["access"] != "read_only_reuse_no_rerun_no_mutation"
        or type(value["root"]) is not str
        or not Path(value["root"]).is_absolute()
        or value["record_count"] != EXPECTED_REFERENCE_RECORD_COUNT
    ):
        raise GaugeGate1Error("reference identity header changed")
    plan = value["plan"]
    _exact_keys(
        plan,
        {
            "relative_path",
            "file_sha256",
            "file_size_bytes",
            "schema",
            "embedded_plan_sha256",
        },
        path="reference_identity.plan",
    )
    if (
        plan["relative_path"] != "plan.json"
        or plan["schema"] != REFERENCE_PLAN_SCHEMA
        or plan["embedded_plan_sha256"] != EXPECTED_REFERENCE_PLAN_SHA256
    ):
        raise GaugeGate1Error("reference plan binding changed")
    _validate_sha(plan["file_sha256"], path="reference_identity.plan.file_sha256")
    _require_int(
        plan["file_size_bytes"],
        path="reference_identity.plan.file_size_bytes",
        minimum=1,
    )
    records = value["records"]
    if not isinstance(records, list) or len(records) != EXPECTED_REFERENCE_RECORD_COUNT:
        raise GaugeGate1Error("reference record identity count changed")
    paths: set[str] = set()
    expected_jobs = {
        (
            job.dataset,
            model,
            job.subject,
            job.fold,
            job.seed,
        )
        for job in _candidate_manifest()
        for model in EXPECTED_REFERENCE_MODELS
    }
    expected_sequence = [
        ScreenJob(
            dataset=job.dataset,
            model=model,
            subject=job.subject,
            fold=job.fold,
            seed=job.seed,
        )
        for model in EXPECTED_REFERENCE_MODELS
        for job in _candidate_manifest()
    ]
    observed_jobs: set[tuple[Any, ...]] = set()
    for index, (item, expected_job) in enumerate(
        zip(records, expected_sequence, strict=True)
    ):
        _exact_keys(
            item,
            {
                "relative_path",
                "file_sha256",
                "file_size_bytes",
                "job_id",
                "dataset",
                "model",
                "subject",
                "fold",
                "seed",
            },
            path=f"reference_identity.records.{index}",
        )
        relative = item["relative_path"]
        if (
            type(relative) is not str
            or Path(relative).is_absolute()
            or any(part in {"", ".", ".."} for part in Path(relative).parts)
            or not relative.endswith(".json")
            or relative in paths
        ):
            raise GaugeGate1Error("reference record path is unsafe or duplicated")
        paths.add(relative)
        if item != {
            "relative_path": str(_reference_record_relative(expected_job)),
            "file_sha256": item["file_sha256"],
            "file_size_bytes": item["file_size_bytes"],
            "job_id": expected_job.job_id,
            **expected_job.identity(),
        }:
            raise GaugeGate1Error(
                "reference record sequence or job/path binding changed"
            )
        _validate_sha(
            item["file_sha256"],
            path=f"reference_identity.records.{index}.file_sha256",
        )
        _require_int(
            item["file_size_bytes"],
            path=f"reference_identity.records.{index}.file_size_bytes",
            minimum=1,
        )
        observed_jobs.add(
            (
                item["dataset"],
                item["model"],
                item["subject"],
                item["fold"],
                item["seed"],
            )
        )
    if observed_jobs != expected_jobs:
        raise GaugeGate1Error("reference records do not cover the exact comparator grid")
    expected_snapshot = _mapping_sha256(
        {
            "plan": plan,
            "records": records,
        }
    )
    if value["snapshot_sha256"] != expected_snapshot:
        raise GaugeGate1Error("reference snapshot digest differs")


def _validate_gpu_roster(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) != WORKER_COUNT:
        raise GaugeGate1Error("GPU roster must contain exactly three devices")
    uuids: list[str] = []
    for index, item in enumerate(value):
        _exact_keys(
            item,
            {"worker_index", "gpu_uuid", "pci_bus_id", "name"},
            path=f"gpu_roster.{index}",
        )
        if (
            item["worker_index"] != index
            or type(item["gpu_uuid"]) is not str
            or not GPU_UUID_RE.fullmatch(item["gpu_uuid"])
            or type(item["pci_bus_id"]) is not str
            or not item["pci_bus_id"]
            or type(item["name"]) is not str
            or not item["name"]
        ):
            raise GaugeGate1Error("GPU roster entry is invalid")
        uuids.append(item["gpu_uuid"])
    if len(set(uuids)) != WORKER_COUNT:
        raise GaugeGate1Error("GPU roster contains a duplicate physical UUID")
    return tuple(uuids)


def assemble_plan(
    *,
    project_root: Path,
    run_root: Path,
    cache_root: Path,
    authority_fence: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    environment_identity: Mapping[str, Any],
    cache_identity: Mapping[str, Any],
    split_identity: Mapping[str, Any],
    reference_identity: Mapping[str, Any],
    gpu_roster: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Assemble, but do not publish, the one exact Gate 1 plan."""

    jobs = _candidate_manifest()
    train_config = json.loads(TRAIN_CONFIG_JSON)
    decision_rule = json.loads(OPENED17_RULE_JSON)
    payload: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "purpose": PURPOSE,
        "stage": OPENED17_STAGE,
        "evaluation_scope": EVALUATION_SCOPE,
        "confirmation_evidence": False,
        "opened_development_only": True,
        "candidate": MODEL_NAME,
        "candidate_factory": MODEL_FACTORY,
        "project_root": str(_absolute(project_root)),
        "run_root": str(_absolute(run_root)),
        "cache_root": str(_absolute(cache_root)),
        "authority_fence": copy.deepcopy(dict(authority_fence)),
        "worker_count": WORKER_COUNT,
        "job_count": EXPECTED_JOB_COUNT,
        "jobs": [
            {
                **job.identity(),
                "job_id": job.job_id,
                "worker_index": index % WORKER_COUNT,
            }
            for index, job in enumerate(jobs)
        ],
        "train_config": train_config,
        "train_config_json": TRAIN_CONFIG_JSON,
        "train_config_sha256": TRAIN_CONFIG_SHA256,
        "decision_rule": decision_rule,
        "decision_rule_json": OPENED17_RULE_JSON,
        "decision_rule_sha256": OPENED17_RULE_SHA256,
        "source_identity": copy.deepcopy(dict(source_identity)),
        "source_identity_sha256": _mapping_sha256(source_identity),
        "environment_identity": copy.deepcopy(dict(environment_identity)),
        "environment_identity_sha256": _mapping_sha256(environment_identity),
        "cache_identity": copy.deepcopy(dict(cache_identity)),
        "split_identity": copy.deepcopy(dict(split_identity)),
        "reference_identity": copy.deepcopy(dict(reference_identity)),
        "gpu_roster": [copy.deepcopy(dict(item)) for item in gpu_roster],
        "execution_contract": {
            "uv_managed_virtualenv_required": True,
            "maximum_project_gpu_workers": 3,
            "worker_count": WORKER_COUNT,
            "cpu_threads_per_worker": CPU_THREADS,
            "minimum_free_gib": MINIMUM_FREE_GIB,
            "cublas_workspace_config": REQUIRED_CUBLAS_WORKSPACE_CONFIG,
            "candidate_only": True,
            "comparator_training_allowed": False,
        },
        "artifact_contract": {
            "aggregate_validation_metrics_only": True,
            "raw_outcomes_persisted": False,
            "raw_trials_persisted": False,
            "model_state_persisted": False,
            "epoch_trace_persisted": False,
            "sealed_directory_publication": True,
            "atomic_no_replace": True,
        },
    }
    payload["plan_sha256"] = _payload_sha256(payload, "plan_sha256")
    validate_plan(payload)
    return payload


def validate_plan(plan: Mapping[str, Any]) -> None:
    expected_top = {
        "schema",
        "purpose",
        "stage",
        "evaluation_scope",
        "confirmation_evidence",
        "opened_development_only",
        "candidate",
        "candidate_factory",
        "project_root",
        "run_root",
        "cache_root",
        "authority_fence",
        "worker_count",
        "job_count",
        "jobs",
        "train_config",
        "train_config_json",
        "train_config_sha256",
        "decision_rule",
        "decision_rule_json",
        "decision_rule_sha256",
        "source_identity",
        "source_identity_sha256",
        "environment_identity",
        "environment_identity_sha256",
        "cache_identity",
        "split_identity",
        "reference_identity",
        "gpu_roster",
        "execution_contract",
        "artifact_contract",
        "plan_sha256",
    }
    _exact_keys(plan, expected_top, path="plan")
    if (
        plan["schema"] != PLAN_SCHEMA
        or plan["purpose"] != PURPOSE
        or plan["stage"] != OPENED17_STAGE
        or plan["evaluation_scope"] != EVALUATION_SCOPE
        or plan["confirmation_evidence"] is not False
        or plan["opened_development_only"] is not True
        or plan["candidate"] != MODEL_NAME
        or plan["candidate_factory"] != MODEL_FACTORY
        or plan["worker_count"] != WORKER_COUNT
        or plan["job_count"] != EXPECTED_JOB_COUNT
    ):
        raise GaugeGate1Error("fixed Gate 1 plan identity changed")
    for path_name in ("project_root", "run_root", "cache_root"):
        if type(plan[path_name]) is not str or not Path(plan[path_name]).is_absolute():
            raise GaugeGate1Error(f"plan {path_name} is not absolute")
        _validate_absolute_nonroot(Path(plan[path_name]), label=path_name)
    _validate_absolute_nonroot(
        Path(plan["reference_identity"]["root"]),
        label="reference root",
    )
    _exact_keys(
        plan["authority_fence"],
        {"st_dev", "st_ino"},
        path="plan.authority_fence",
    )
    _require_int(
        plan["authority_fence"]["st_dev"],
        path="plan.authority_fence.st_dev",
        minimum=0,
    )
    _require_int(
        plan["authority_fence"]["st_ino"],
        path="plan.authority_fence.st_ino",
        minimum=1,
    )
    if (
        plan["train_config"] != json.loads(TRAIN_CONFIG_JSON)
        or plan["train_config_json"] != TRAIN_CONFIG_JSON
        or plan["train_config_sha256"] != TRAIN_CONFIG_SHA256
        or _sha256_bytes(plan["train_config_json"].encode("ascii"))
        != plan["train_config_sha256"]
    ):
        raise GaugeGate1Error("frozen training configuration changed")
    if (
        plan["decision_rule"] != json.loads(OPENED17_RULE_JSON)
        or plan["decision_rule_json"] != OPENED17_RULE_JSON
        or plan["decision_rule_sha256"] != OPENED17_RULE_SHA256
        or _sha256_bytes(plan["decision_rule_json"].encode("ascii"))
        != plan["decision_rule_sha256"]
    ):
        raise GaugeGate1Error("frozen Gate 1 rule changed")
    _validate_source_identity(plan["source_identity"])
    if plan["source_identity_sha256"] != _mapping_sha256(
        plan["source_identity"]
    ):
        raise GaugeGate1Error("source closure digest differs")
    _validate_environment_identity(plan["environment_identity"])
    if plan["environment_identity_sha256"] != _mapping_sha256(
        plan["environment_identity"]
    ):
        raise GaugeGate1Error("environment digest differs")
    _validate_cache_identity_maps(
        plan["cache_identity"],
        plan["split_identity"],
    )
    _validate_reference_identity(plan["reference_identity"])
    _validate_gpu_roster(plan["gpu_roster"])
    expected_jobs = _candidate_manifest()
    if not isinstance(plan["jobs"], list) or len(plan["jobs"]) != len(expected_jobs):
        raise GaugeGate1Error("Gate 1 job list changed")
    for index, (expected, value) in enumerate(
        zip(expected_jobs, plan["jobs"], strict=True)
    ):
        if value != {
            **expected.identity(),
            "job_id": expected.job_id,
            "worker_index": index % WORKER_COUNT,
        }:
            raise GaugeGate1Error("Gate 1 job order or worker partition changed")
    expected_execution = {
        "uv_managed_virtualenv_required": True,
        "maximum_project_gpu_workers": 3,
        "worker_count": WORKER_COUNT,
        "cpu_threads_per_worker": CPU_THREADS,
        "minimum_free_gib": MINIMUM_FREE_GIB,
        "cublas_workspace_config": REQUIRED_CUBLAS_WORKSPACE_CONFIG,
        "candidate_only": True,
        "comparator_training_allowed": False,
    }
    if plan["execution_contract"] != expected_execution:
        raise GaugeGate1Error("execution contract changed")
    expected_artifact = {
        "aggregate_validation_metrics_only": True,
        "raw_outcomes_persisted": False,
        "raw_trials_persisted": False,
        "model_state_persisted": False,
        "epoch_trace_persisted": False,
        "sealed_directory_publication": True,
        "atomic_no_replace": True,
    }
    if plan["artifact_contract"] != expected_artifact:
        raise GaugeGate1Error("artifact contract changed")
    _validate_sha(plan["plan_sha256"], path="plan.plan_sha256")
    if plan["plan_sha256"] != _payload_sha256(plan, "plan_sha256"):
        raise GaugeGate1Error("plan SHA-256 differs")


def _source_snapshot(
    project_root: Path,
    *,
    expected: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], MultiFileSnapshot]:
    paths = tuple(_absolute(project_root / name) for name in SOURCE_FILES)
    snapshot = _open_multi_file_snapshot(
        paths,
        namespace_root=project_root,
    )
    identity = {
        name: {
            "sha256": _sha256_bytes(item.payload),
            "size_bytes": len(item.payload),
        }
        for name, item in zip(SOURCE_FILES, snapshot.files, strict=True)
    }
    _validate_source_identity(identity)
    if expected is not None and identity != dict(expected):
        snapshot.close()
        raise GaugeGate1Error("source closure differs from immutable plan")
    return identity, snapshot


def _require_uv_virtualenv() -> Path:
    if sys.prefix == sys.base_prefix:
        raise GaugeGate1Error("a UV-managed virtual environment is required")
    configuration = Path(sys.prefix) / "pyvenv.cfg"
    snapshot = _open_file_snapshot(
        configuration,
        maximum=1024 * 1024,
        namespace_root=Path(sys.prefix),
    )
    try:
        text = snapshot.payload.decode("utf-8")
        _assert_file_snapshot(snapshot)
    except Exception:
        snapshot.close()
        raise
    snapshot.close()
    if not any(
        line.strip().lower().startswith("uv =")
        for line in text.splitlines()
    ):
        raise GaugeGate1Error("active virtual environment is not UV-managed")
    return configuration


def _bound_uv_executable() -> tuple[Path, FileSnapshot]:
    """Open the operator-bound UV executable without consulting ``PATH``."""

    raw = os.environ.get(UV_EXECUTABLE_ENVIRONMENT_VARIABLE)
    if raw is None or not raw:
        raise GaugeGate1Error(
            f"{UV_EXECUTABLE_ENVIRONMENT_VARIABLE} must bind the exact "
            "absolute UV executable"
        )
    try:
        candidate = Path(raw)
        absolute = _absolute(candidate)
    except (OSError, ValueError) as error:
        raise GaugeGate1Error("bound UV path is invalid") from error
    if not candidate.is_absolute() or str(candidate) != str(absolute):
        raise GaugeGate1Error(
            f"{UV_EXECUTABLE_ENVIRONMENT_VARIABLE} must be canonical absolute"
        )
    try:
        snapshot = _open_file_snapshot(
            absolute,
            namespace_root=absolute.parent,
        )
    except (OSError, ValueError) as error:
        raise GaugeGate1Error(
            f"cannot open bound UV executable: {absolute}"
        ) from error
    if not stat.S_IMODE(snapshot.observed.st_mode) & 0o111:
        snapshot.close()
        raise GaugeGate1Error("bound UV file is not executable")
    return absolute, snapshot


def environment_identity() -> dict[str, Any]:
    """Fingerprint only the active UV environment; never install anything."""

    configuration = _require_uv_virtualenv()
    uv_path, uv_snapshot = _bound_uv_executable()
    try:
        configuration_snapshot = _open_file_snapshot(
            configuration,
            maximum=1024 * 1024,
            namespace_root=Path(sys.prefix),
        )
    except Exception:
        uv_snapshot.close()
        raise
    try:
        uv_sha = _sha256_bytes(uv_snapshot.payload)
        version = subprocess.run(
            [str(uv_path), "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
        freeze_result = subprocess.run(
            [
                str(uv_path),
                "pip",
                "freeze",
                "--python",
                sys.executable,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        driver_result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise GaugeGate1Error(
            f"cannot fingerprint UV/NVIDIA environment: {error}"
        ) from error
    finally:
        # Kept open across both UV executions and the driver query.  Exact
        # path/inode/content checks close the ordinary A/B and A->B->A swap
        # window around environment fingerprinting.
        try:
            _assert_file_snapshot(uv_snapshot)
            _assert_file_snapshot(configuration_snapshot)
        finally:
            uv_snapshot.close()
            configuration_snapshot.close()
    freeze_raw = [
        line.strip()
        for line in freeze_result.stdout.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not freeze_raw or len(freeze_raw) != len(set(freeze_raw)):
        raise GaugeGate1Error("UV package freeze is empty or duplicated")
    freeze = sorted(freeze_raw, key=str.casefold)
    drivers = sorted(
        {
            line.strip()
            for line in driver_result.stdout.splitlines()
            if line.strip()
        }
    )
    packages: dict[str, str] = {}
    for package in REQUIRED_PACKAGES:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as error:
            raise GaugeGate1Error(
                f"required package is absent from UV environment: {package}"
            ) from error
    import torch

    value = {
        "schema": ENVIRONMENT_SCHEMA,
        "python": sys.version,
        "python_executable": str(_absolute(Path(sys.executable))),
        "python_prefix": str(_absolute(Path(sys.prefix))),
        "platform": platform.platform(),
        "uv_binding": {
            "kind": "required_absolute_environment_variable",
            "name": UV_EXECUTABLE_ENVIRONMENT_VARIABLE,
        },
        "uv_executable": str(uv_path),
        "uv_executable_sha256": uv_sha,
        "uv_version": version,
        "pyvenv_cfg_sha256": _sha256_bytes(configuration_snapshot.payload),
        "uv_pip_freeze": freeze,
        "uv_pip_freeze_sha256": _sha256_bytes(
            "\n".join(freeze).encode("utf-8")
        ),
        "required_packages": packages,
        "torch_cuda_version": str(torch.version.cuda),
        "torch_cudnn_version": torch.backends.cudnn.version(),
        "nvidia_driver_versions": drivers,
        "required_cublas_workspace_config": REQUIRED_CUBLAS_WORKSPACE_CONFIG,
        "cpu_threads_per_worker": CPU_THREADS,
        "minimum_free_gib": MINIMUM_FREE_GIB,
    }
    _validate_environment_identity(value)
    return value


def _reference_record_relative(job: ScreenJob) -> Path:
    return (
        Path(RECORDS_DIRECTORY)
        / job.dataset
        / job.model
        / f"subject_{job.subject:03d}"
        / f"fold_{job.fold:02d}"
        / f"seed_{job.seed:03d}.json"
    )


def _validate_reference_plan(plan: Mapping[str, Any]) -> None:
    if (
        plan.get("schema") != REFERENCE_PLAN_SCHEMA
        or plan.get("plan_sha256") != EXPECTED_REFERENCE_PLAN_SHA256
        or plan.get("evaluation_scope") != EVALUATION_SCOPE
        or plan.get("confirmation_evidence") is not False
        or plan.get("train_config") != json.loads(TRAIN_CONFIG_JSON)
        or plan.get("n_jobs") != 136
    ):
        raise GaugeGate1Error("reference plan identity or common recipe changed")
    if plan["plan_sha256"] != _payload_sha256(plan, "plan_sha256"):
        raise GaugeGate1Error("reference plan embedded digest differs")
    if (
        type(plan.get("cache_root")) is not str
        or not Path(plan["cache_root"]).is_absolute()
        or not isinstance(plan.get("source_identity"), Mapping)
        or not plan["source_identity"]
        or any(
            type(name) is not str
            or type(digest) is not str
            or not SHA256_RE.fullmatch(digest)
            for name, digest in plan["source_identity"].items()
        )
    ):
        raise GaugeGate1Error("reference plan path/source identity is invalid")
    planned_jobs = plan.get("jobs")
    if not isinstance(planned_jobs, list) or len(planned_jobs) != 136:
        raise GaugeGate1Error("reference plan job manifest changed")
    selected: dict[tuple[str, str, int, int, int], Mapping[str, Any]] = {}
    for item in planned_jobs:
        if not isinstance(item, Mapping):
            raise GaugeGate1Error("reference plan job entry is invalid")
        try:
            key = (
                item["dataset"],
                item["model"],
                item["subject"],
                item["fold"],
                item["seed"],
            )
        except KeyError as error:
            raise GaugeGate1Error("reference plan job entry is incomplete") from error
        if key in selected:
            raise GaugeGate1Error("reference plan job entry is duplicated")
        selected[key] = item
    for model in EXPECTED_REFERENCE_MODELS:
        for candidate in _candidate_manifest():
            expected = ScreenJob(
                dataset=candidate.dataset,
                model=model,
                subject=candidate.subject,
                fold=candidate.fold,
                seed=candidate.seed,
            )
            key = (
                expected.dataset,
                expected.model,
                expected.subject,
                expected.fold,
                expected.seed,
            )
            item = selected.get(key)
            if (
                item is None
                or item.get("job_id") != expected.job_id
                or type(item.get("worker_index")) is not int
            ):
                raise GaugeGate1Error(
                    "reference plan omits an exact comparator cell"
                )
    _validate_cache_identity_maps(
        {
            key: value
            for key, value in plan.get("cache_identity", {}).items()
            if key in {_subject_key(job) for job in _candidate_manifest()}
        },
        {
            key: value
            for key, value in plan.get("split_identity", {}).items()
            if key in {_job_key(job) for job in _candidate_manifest()}
        },
    )


def _validate_reference_record(
    record: Mapping[str, Any],
    *,
    job: ScreenJob,
    reference_plan: Mapping[str, Any],
) -> None:
    expected_top = {
        "cache_identity",
        "confirmation_evidence",
        "created_at",
        "evaluation_scope",
        "fit",
        "job",
        "job_id",
        "model",
        "plan_sha256",
        "resource_preflight",
        "scaler",
        "schema",
        "source_identity",
        "split",
        "timing",
        "validation_metrics",
    }
    _exact_keys(record, expected_top, path="reference_record")
    if (
        record["schema"] != REFERENCE_RECORD_SCHEMA
        or record["plan_sha256"] != EXPECTED_REFERENCE_PLAN_SHA256
        or record["evaluation_scope"] != EVALUATION_SCOPE
        or record["confirmation_evidence"] is not False
        or record["job_id"] != job.job_id
        or record["job"] != job.identity()
    ):
        raise GaugeGate1Error("reference record identity changed")
    if record["cache_identity"] != {
        name: reference_plan["cache_identity"][_subject_key(job)][name]
        for name in ("array_sha256", "file_sha256")
    }:
        raise GaugeGate1Error("reference cache binding differs")
    if record["split"] != reference_plan["split_identity"][_job_key(job)]:
        raise GaugeGate1Error("reference split binding differs")
    if record["source_identity"] != reference_plan["source_identity"]:
        raise GaugeGate1Error("reference record source binding differs")
    _validate_timestamp(record["created_at"], path="reference_record.created_at")
    scaler = record["scaler"]
    _exact_keys(
        scaler,
        {"fit_scope", "mean_sha256", "std_sha256", "contract"},
        path="reference_record.scaler",
    )
    if (
        scaler["fit_scope"] != "training_only"
        or scaler["contract"] != dict(CHANNEL_SCALING)
    ):
        raise GaugeGate1Error("reference scaler contract differs")
    _validate_sha(
        scaler["mean_sha256"],
        path="reference_record.scaler.mean_sha256",
    )
    _validate_sha(
        scaler["std_sha256"],
        path="reference_record.scaler.std_sha256",
    )
    fit = record["fit"]
    _exact_keys(
        fit,
        {
            "best_epoch",
            "best_validation_loss",
            "config",
            "early_stopping",
            "epochs_run",
        },
        path="reference_record.fit",
    )
    best_epoch = _require_int(
        fit["best_epoch"],
        path="reference_record.fit.best_epoch",
        minimum=0,
    )
    epochs_run = _require_int(
        fit["epochs_run"],
        path="reference_record.fit.epochs_run",
        minimum=1,
    )
    if (
        best_epoch >= epochs_run
        or epochs_run > 200
        or fit["config"] != json.loads(TRAIN_CONFIG_JSON)
        or fit["early_stopping"] is not True
    ):
        raise GaugeGate1Error("reference fit contract differs")
    _require_number(
        fit["best_validation_loss"],
        path="reference_record.fit.best_validation_loss",
        minimum=0,
    )
    model = record["model"]
    _exact_keys(
        model,
        {
            "constructed_state_sha256",
            "identity",
            "initial_state_sha256",
            "parameter_count",
            "selected_state_sha256",
            "source_statistics",
        },
        path="reference_record.model",
    )
    identity = model["identity"]
    if (
        not isinstance(identity, Mapping)
        or identity.get("requested_name") != job.model
        or type(identity.get("class")) is not str
        or not identity["class"]
        or type(identity.get("uses_positions")) is not bool
    ):
        raise GaugeGate1Error("reference model identity differs")
    _require_int(
        model["parameter_count"],
        path="reference_record.model.parameter_count",
        minimum=1,
    )
    for name in (
        "constructed_state_sha256",
        "initial_state_sha256",
        "selected_state_sha256",
    ):
        _validate_sha(model[name], path=f"reference_record.model.{name}")
    if model["constructed_state_sha256"] != model["initial_state_sha256"]:
        raise GaugeGate1Error("reference model changed before fit")
    if model["source_statistics"] != {
        "available": False,
        "ran": False,
        "fit_scope": "training_rows_only",
        "outcome_aware": False,
    }:
        raise GaugeGate1Error("reference source-statistics contract differs")
    metrics = record["validation_metrics"]
    if not isinstance(metrics, Mapping) or set(metrics) != set(METRIC_NAMES):
        raise GaugeGate1Error("reference metric set changed")
    for name, value in metrics.items():
        if name in {"accuracy", "balanced_accuracy", "roc_auc"}:
            _require_number(value, path=f"reference.{name}", minimum=0, maximum=1)
        elif name == "cohen_kappa":
            _require_number(value, path=f"reference.{name}", minimum=-1, maximum=1)
        else:
            _require_number(value, path=f"reference.{name}", minimum=0)
    timing = record["timing"]
    _exact_keys(
        timing,
        {
            "construction_seconds",
            "evaluation_seconds",
            "fit_seconds",
            "total_seconds",
        },
        path="reference_record.timing",
    )
    components = [
        _require_number(
            timing[name],
            path=f"reference_record.timing.{name}",
            minimum=0,
        )
        for name in (
            "construction_seconds",
            "fit_seconds",
            "evaluation_seconds",
        )
    ]
    total = _require_number(
        timing["total_seconds"],
        path="reference_record.timing.total_seconds",
        minimum=0,
    )
    if total + 1e-9 < sum(components):
        raise GaugeGate1Error("reference timing aggregate is impossible")
    if not isinstance(record["resource_preflight"], Mapping):
        raise GaugeGate1Error("reference resource preflight is invalid")
    _assert_no_excluded_evidence(record)


def _reference_snapshot(
    reference_root: Path,
    *,
    expected: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], MultiFileSnapshot, dict[str, Any]]:
    jobs = _candidate_manifest()
    reference_jobs = tuple(
        ScreenJob(
            dataset=job.dataset,
            model=model,
            subject=job.subject,
            fold=job.fold,
            seed=job.seed,
        )
        for model in EXPECTED_REFERENCE_MODELS
        for job in jobs
    )
    mapped_paths = _approved_reference_record_paths(reference_root)
    expected_mapped_paths = tuple(
        reference_root / _reference_record_relative(job)
        for job in reference_jobs
    )
    if mapped_paths != expected_mapped_paths:
        raise GaugeGate1Error(
            "reference paths differ from approved screen specification"
        )
    paths = (reference_root / "plan.json",) + mapped_paths
    snapshot = _open_multi_file_snapshot(
        paths,
        json_limit=True,
        namespace_root=reference_root,
    )
    try:
        plan_item = snapshot.files[0]
        plan = _strict_json_bytes(plan_item.payload, source=str(plan_item.path))
        _validate_reference_plan(plan)
        records: list[dict[str, Any]] = []
        for job, item in zip(reference_jobs, snapshot.files[1:], strict=True):
            record = _strict_json_bytes(item.payload, source=str(item.path))
            _validate_reference_record(
                record,
                job=job,
                reference_plan=plan,
            )
            records.append(
                {
                    "relative_path": str(item.path.relative_to(_absolute(reference_root))),
                    "file_sha256": _sha256_bytes(item.payload),
                    "file_size_bytes": len(item.payload),
                    "job_id": job.job_id,
                    **job.identity(),
                }
            )
        plan_identity = {
            "relative_path": "plan.json",
            "file_sha256": _sha256_bytes(plan_item.payload),
            "file_size_bytes": len(plan_item.payload),
            "schema": REFERENCE_PLAN_SCHEMA,
            "embedded_plan_sha256": EXPECTED_REFERENCE_PLAN_SHA256,
        }
        identity = {
            "schema": "ieee-mi-gauge-gate1-reference-snapshot-v1",
            "access": "read_only_reuse_no_rerun_no_mutation",
            "root": str(_absolute(reference_root)),
            "plan": plan_identity,
            "records": records,
            "record_count": len(records),
            "snapshot_sha256": _mapping_sha256(
                {"plan": plan_identity, "records": records}
            ),
        }
        _validate_reference_identity(identity)
        if expected is not None and identity != dict(expected):
            raise GaugeGate1Error(
                "read-only reference evidence differs from immutable plan"
            )
        snapshot.revalidate()
        return identity, snapshot, plan
    except Exception:
        snapshot.close()
        raise


@contextmanager
def _independent_snapshot_reader(
    snapshot: FileSnapshot,
) -> Iterator[Any]:
    """Yield a byte-zero reader with an independent open-file description."""

    _assert_file_snapshot(snapshot)
    parent, descriptor = _open_absolute_file(snapshot.path)
    handle: Any | None = None
    try:
        opened = os.fstat(descriptor)
        named = os.stat(
            snapshot.path.name,
            dir_fd=parent,
            follow_symlinks=False,
        )
        if (
            _stat_fingerprint(opened)
            != _stat_fingerprint(snapshot.observed)
            or not stat.S_ISREG(named.st_mode)
            or (named.st_dev, named.st_ino)
            != (snapshot.observed.st_dev, snapshot.observed.st_ino)
        ):
            raise GaugeGate1Error(
                f"independent reader differs from snapshot: {snapshot.path}"
            )
        if os.lseek(descriptor, 0, os.SEEK_SET) != 0:
            raise GaugeGate1Error("independent snapshot reader did not rewind")
        handle = os.fdopen(descriptor, "rb", closefd=True)
        descriptor = -1
        if handle.seek(0, os.SEEK_SET) != 0:
            raise GaugeGate1Error("independent snapshot handle did not rewind")
        yield handle
        if (
            _stat_fingerprint(os.fstat(handle.fileno()))
            != _stat_fingerprint(snapshot.observed)
        ):
            raise GaugeGate1Error(
                f"independent snapshot reader changed: {snapshot.path}"
            )
    finally:
        try:
            if handle is not None:
                handle.close()
            elif descriptor >= 0:
                os.close(descriptor)
        finally:
            os.close(parent)
            _assert_file_snapshot(snapshot)


def _validate_npz_members(snapshot: FileSnapshot) -> None:
    try:
        with _independent_snapshot_reader(snapshot) as handle:
            with zipfile.ZipFile(handle, "r") as archive:
                names = [item.filename for item in archive.infolist()]
                expected = [f"{name}.npy" for name in SUBJECT_CACHE_NPZ_MEMBERS]
                if (
                    len(names) != len(set(names))
                    or set(names) != set(expected)
                    or len(names) != len(expected)
                    or any(
                        "/" in name
                        or name.startswith(".")
                        or item.flag_bits & 0x1
                        for name, item in zip(names, archive.infolist(), strict=True)
                    )
                ):
                    raise GaugeGate1Error(
                        "NPZ member set is incomplete, duplicated, or unsafe: "
                        f"{snapshot.path}"
                    )
    except (EOFError, OSError, ValueError, zipfile.BadZipFile) as error:
        raise GaugeGate1Error(
            f"invalid NPZ cache {snapshot.path}: {error}"
        ) from error


def _cache_snapshot(
    cache_root: Path,
    contract: Mapping[str, Any],
) -> FileSnapshot:
    path = _absolute(cache_root / str(contract["relative_path"]))
    if path.parent == _absolute(cache_root):
        raise GaugeGate1Error("cache path layout is unexpectedly shallow")
    snapshot = _open_file_snapshot(
        path,
        namespace_root=cache_root,
    )
    try:
        if (
            len(snapshot.payload) != contract["file_size_bytes"]
            or _sha256_bytes(snapshot.payload) != contract["file_sha256"]
        ):
            raise GaugeGate1Error(f"cache differs from immutable plan: {path}")
        _validate_npz_members(snapshot)
        _assert_file_snapshot(snapshot)
        return snapshot
    except Exception:
        snapshot.close()
        raise


def _npz_metadata(snapshot: FileSnapshot) -> dict[str, Any]:
    try:
        with _independent_snapshot_reader(snapshot) as handle:
            with np.load(handle, allow_pickle=False) as archive:
                result = {
                    "sessions": np.asarray(archive["sessions"]).astype(str),
                    "runs": np.asarray(archive["runs"]).astype(str),
                    "positions": np.asarray(
                        archive["positions"], dtype=np.float32
                    ).copy(),
                    "channel_names": tuple(
                        str(value) for value in archive["channel_names"].tolist()
                    ),
                    "identity_raw": str(archive["identity"].item()),
                }
    except (
        EOFError,
        KeyError,
        OSError,
        ValueError,
        zipfile.BadZipFile,
    ) as error:
        raise GaugeGate1Error(f"cannot read cache metadata: {error}") from error
    return result


def _validate_cache_metadata_identity(
    identity: Mapping[str, Any],
    *,
    job: ScreenJob,
    contract: Mapping[str, Any],
    row_count: int,
) -> None:
    dataset_identity = identity.get("dataset")
    shape = identity.get("shape")
    if (
        not isinstance(dataset_identity, Mapping)
        or dataset_identity.get("key") != job.dataset
        or type(identity.get("subject")) is not int
        or identity["subject"] != job.subject
        or identity.get("montage_profile") != DEFAULT_MONTAGE_PROFILE
        or identity.get("preprocessing") != preprocessing_for_dataset(job.dataset)
        or not isinstance(shape, list)
        or len(shape) != 3
        or any(type(value) is not int or value <= 0 for value in shape)
        or shape[0] != row_count
        or identity.get("array_sha256") != contract["array_sha256"]
    ):
        raise GaugeGate1Error(
            f"cache metadata identity differs for {job.dataset} S{job.subject}"
        )


def _read_selected_npy_rows(
    snapshot: FileSnapshot,
    member: str,
    rows: np.ndarray,
    *,
    expected_dtype: np.dtype[Any],
    expected_ndim: int,
) -> tuple[np.ndarray, tuple[int, ...]]:
    selected = np.asarray(rows, dtype=np.int64)
    if (
        selected.ndim != 1
        or not len(selected)
        or len(np.unique(selected)) != len(selected)
    ):
        raise GaugeGate1Error("selected rows are empty, malformed, or duplicated")
    order = np.argsort(selected)
    sorted_rows = selected[order]
    try:
        with _independent_snapshot_reader(snapshot) as handle:
            with zipfile.ZipFile(handle, "r") as archive:
                if sum(
                    info.filename == f"{member}.npy"
                    for info in archive.infolist()
                ) != 1:
                    raise GaugeGate1Error("NPZ member is absent or duplicated")
                with archive.open(f"{member}.npy", "r") as stream:
                    version = np.lib.format.read_magic(stream)
                    if version == (1, 0):
                        shape, fortran, dtype = np.lib.format.read_array_header_1_0(
                            stream
                        )
                    elif version == (2, 0):
                        shape, fortran, dtype = np.lib.format.read_array_header_2_0(
                            stream
                        )
                    elif version == (3, 0):
                        shape, fortran, dtype = np.lib.format._read_array_header(  # type: ignore[attr-defined]  # noqa: SLF001
                            stream, version
                        )
                    else:
                        raise GaugeGate1Error("unsupported NPY version")
                    shape = tuple(int(value) for value in shape)
                    dtype = np.dtype(dtype)
                    if (
                        len(shape) != expected_ndim
                        or fortran
                        or dtype != np.dtype(expected_dtype)
                        or dtype.hasobject
                        or sorted_rows[0] < 0
                        or sorted_rows[-1] >= shape[0]
                    ):
                        raise GaugeGate1Error(
                            f"{member} NPY layout differs from contract"
                        )
                    row_shape = shape[1:]
                    row_elements = math.prod(row_shape) if row_shape else 1
                    row_bytes = row_elements * dtype.itemsize
                    sorted_result = np.empty(
                        (len(sorted_rows), *row_shape), dtype=dtype
                    )
                    cursor = 0
                    for output_index, row in enumerate(sorted_rows.tolist()):
                        skip = (row - cursor) * row_bytes
                        if skip:
                            stream.seek(skip, os.SEEK_CUR)
                        payload = stream.read(row_bytes)
                        if len(payload) != row_bytes:
                            raise GaugeGate1Error("NPY payload is truncated")
                        sorted_result[output_index] = np.frombuffer(
                            payload,
                            dtype=dtype,
                            count=row_elements,
                        ).reshape(row_shape)
                        cursor = row + 1
    except (
        EOFError,
        KeyError,
        OSError,
        ValueError,
        zipfile.BadZipFile,
    ) as error:
        raise GaugeGate1Error(f"cannot stream {member} rows: {error}") from error
    result = np.empty_like(sorted_result)
    result[order] = sorted_result
    return result, shape


def _load_validation_data(
    *,
    job: ScreenJob,
    plan: Mapping[str, Any],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[str, ...],
]:
    contract = plan["cache_identity"][_subject_key(job)]
    snapshot = _cache_snapshot(Path(plan["cache_root"]), contract)
    try:
        metadata = _npz_metadata(snapshot)
        train_rows, validation_rows = development_rows(
            job.dataset,
            job.subject,
            metadata["sessions"],
            metadata["runs"],
        )
        split = plan["split_identity"][_job_key(job)]
        if (
            split["train"]
            != {
                "count": len(train_rows),
                "rows_sha256": _rows_sha256(train_rows),
            }
            or split["validation"]
            != {
                "count": len(validation_rows),
                "rows_sha256": _rows_sha256(validation_rows),
            }
        ):
            raise GaugeGate1Error("derived training/validation rows differ")
        selected = np.concatenate((train_rows, validation_rows))
        x, x_shape = _read_selected_npy_rows(
            snapshot,
            "x",
            selected,
            expected_dtype=np.dtype(np.float32),
            expected_ndim=3,
        )
        y, y_shape = _read_selected_npy_rows(
            snapshot,
            "y",
            selected,
            expected_dtype=np.dtype(np.int64),
            expected_ndim=1,
        )
        identity = _strict_json_bytes(
            metadata["identity_raw"].encode("utf-8"),
            source=f"{snapshot.path}:identity",
        )
        _validate_cache_metadata_identity(
            identity,
            job=job,
            contract=contract,
            row_count=len(metadata["sessions"]),
        )
        if (
            x_shape[0] != len(metadata["sessions"])
            or y_shape != (len(metadata["sessions"]),)
        ):
            raise GaugeGate1Error("cache array identity or row count differs")
        count = len(train_rows)
        x_train = x[:count]
        x_validation = x[count:]
        y_train = y[:count]
        y_validation = y[count:]
        positions = metadata["positions"]
        channel_names = metadata["channel_names"]
        classes = set(range(dataset_spec(job.dataset).n_classes))
        if (
            not np.isfinite(x_train).all()
            or not np.isfinite(x_validation).all()
            or not np.isfinite(positions).all()
            or set(y_train.tolist()) != classes
            or set(y_validation.tolist()) != classes
            or positions.shape != (x_train.shape[1], 3)
            or len(channel_names) != x_train.shape[1]
        ):
            raise GaugeGate1Error("cache selected data contract differs")
        _assert_file_snapshot(snapshot)
        return (
            x_train,
            y_train,
            x_validation,
            y_validation,
            positions,
            channel_names,
        )
    finally:
        snapshot.close()


def _cache_and_split_from_reference(
    *,
    cache_root: Path,
    reference_plan: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    cache_identity = {
        _subject_key(job): copy.deepcopy(
            reference_plan["cache_identity"][_subject_key(job)]
        )
        for job in _candidate_manifest()
    }
    split_identity = {
        _job_key(job): copy.deepcopy(
            reference_plan["split_identity"][_job_key(job)]
        )
        for job in _candidate_manifest()
    }
    _validate_cache_identity_maps(cache_identity, split_identity)
    for job in _candidate_manifest():
        contract = cache_identity[_subject_key(job)]
        snapshot = _cache_snapshot(cache_root, contract)
        try:
            metadata = _npz_metadata(snapshot)
            train, validation = development_rows(
                job.dataset,
                job.subject,
                metadata["sessions"],
                metadata["runs"],
            )
            expected = split_identity[_job_key(job)]
            if expected != {
                "train": {
                    "count": len(train),
                    "rows_sha256": _rows_sha256(train),
                },
                "validation": {
                    "count": len(validation),
                    "rows_sha256": _rows_sha256(validation),
                },
            }:
                raise GaugeGate1Error("reference split differs from cache metadata")
            identity = _strict_json_bytes(
                metadata["identity_raw"].encode("utf-8"),
                source=f"{snapshot.path}:identity",
            )
            _validate_cache_metadata_identity(
                identity,
                job=job,
                contract=contract,
                row_count=len(metadata["sessions"]),
            )
        finally:
            snapshot.close()
    return cache_identity, split_identity


def _gpu_inventory() -> list[dict[str, str]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=uuid,pci.bus_id,name",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ResourceUnavailable(f"cannot inventory NVIDIA GPUs: {error}") from error
    rows: list[dict[str, str]] = []
    for line in result.stdout.splitlines():
        fields = [value.strip() for value in line.split(",")]
        if (
            len(fields) != 3
            or not GPU_UUID_RE.fullmatch(fields[0])
            or not fields[1]
            or not fields[2]
        ):
            raise ResourceUnavailable("NVIDIA inventory row is malformed")
        rows.append(
            {
                "gpu_uuid": fields[0],
                "pci_bus_id": fields[1],
                "name": fields[2],
            }
        )
    if not rows or len({row["gpu_uuid"] for row in rows}) != len(rows):
        raise ResourceUnavailable("NVIDIA inventory is empty or duplicated")
    return rows


def _build_gpu_roster(gpu_uuids: Sequence[str]) -> list[dict[str, Any]]:
    if (
        len(gpu_uuids) != WORKER_COUNT
        or len(set(gpu_uuids)) != WORKER_COUNT
        or any(not GPU_UUID_RE.fullmatch(value) for value in gpu_uuids)
    ):
        raise GaugeGate1Error("exactly three unique full GPU UUIDs are required")
    inventory = {row["gpu_uuid"]: row for row in _gpu_inventory()}
    roster = []
    for index, gpu_uuid in enumerate(gpu_uuids):
        try:
            row = inventory[gpu_uuid]
        except KeyError as error:
            raise ResourceUnavailable(
                f"planned GPU is absent: {gpu_uuid}"
            ) from error
        roster.append({"worker_index": index, **row})
    _validate_gpu_roster(roster)
    return roster


def build_plan(
    *,
    project_root: Path,
    run_root: Path,
    reference_root: Path,
    gpu_uuids: Sequence[str],
) -> dict[str, Any]:
    """Read immutable inputs and assemble a plan without publishing it."""

    _assert_approved_screen_dependency()
    project_root = _absolute(project_root)
    run_root = _absolute(run_root)
    reference_root = _absolute(reference_root)
    _validate_absolute_nonroot(project_root, label="project root")
    _validate_absolute_nonroot(run_root, label="run root")
    _validate_absolute_nonroot(reference_root, label="reference root")
    preflight_launch_layout(
        project_root=project_root,
        run_root=run_root,
        reference_root=reference_root,
    )
    _ensure_run_layout(project_root, run_root)
    _probe_disk(run_root)
    fence_descriptor, fence_stat = _open_or_create_authority_fence(run_root)
    os.close(fence_descriptor)
    source, source_snapshot = _source_snapshot(project_root)
    try:
        reference, reference_snapshot, reference_plan = _reference_snapshot(
            reference_root
        )
        try:
            cache_root = _absolute(Path(reference_plan["cache_root"]))
            _validate_absolute_nonroot(cache_root, label="cache root")
            preflight_launch_layout(
                project_root=project_root,
                run_root=run_root,
                reference_root=reference_root,
                cache_root=cache_root,
            )
            _probe_disk(cache_root)
            cache, split = _cache_and_split_from_reference(
                cache_root=cache_root,
                reference_plan=reference_plan,
            )
            environment = environment_identity()
            roster = _build_gpu_roster(gpu_uuids)
            source_snapshot.revalidate()
            reference_snapshot.revalidate()
            return assemble_plan(
                project_root=project_root,
                run_root=run_root,
                cache_root=cache_root,
                authority_fence={
                    "st_dev": int(fence_stat.st_dev),
                    "st_ino": int(fence_stat.st_ino),
                },
                source_identity=source,
                environment_identity=environment,
                cache_identity=cache,
                split_identity=split,
                reference_identity=reference,
                gpu_roster=roster,
            )
        finally:
            reference_snapshot.close()
    finally:
        source_snapshot.close()


def _plan_members(plan: Mapping[str, Any]) -> dict[str, bytes]:
    payload = _canonical_bytes(plan) + b"\n"
    return {
        "plan.json": payload,
        "plan.sha256": (_sha256_bytes(payload) + "\n").encode("ascii"),
        "COMMITTED": PLAN_MARKER,
    }


def _validate_plan_package(snapshot: PackageSnapshot) -> dict[str, Any]:
    plan_bytes = snapshot.children["plan.json"].payload
    expected_hash = snapshot.children["plan.sha256"].payload
    if expected_hash != (_sha256_bytes(plan_bytes) + "\n").encode("ascii"):
        raise GaugeGate1Error("plan package file digest differs")
    if snapshot.children["COMMITTED"].payload != PLAN_MARKER:
        raise GaugeGate1Error("plan package commit marker differs")
    plan = _strict_json_bytes(plan_bytes, source=str(snapshot.path / "plan.json"))
    validate_plan(plan)
    return plan


def publish_or_validate_plan(
    *,
    project_root: Path,
    run_root: Path,
    plan: Mapping[str, Any],
) -> bool:
    """Publish one immutable plan package, or validate the exact existing one."""

    validate_plan(plan)
    if str(_absolute(project_root)) != plan["project_root"]:
        raise GaugeGate1Error("project root differs from plan")
    if str(_absolute(run_root)) != plan["run_root"]:
        raise GaugeGate1Error("run root differs from plan")
    with _authority_lock(
        run_root,
        expected=plan["authority_fence"],
    ):
        jobs = _candidate_manifest()
        _quarantine_orphan_private_entries(
            parent=run_root / STAGING_DIRECTORY,
            prefix=".stage.plan.",
            label="orphan-plan-stage",
            directory=True,
        )
        destination = _absolute(run_root / PLAN_DIRECTORY)
        if destination.exists():
            _validate_record_parents(run_root, jobs)
            snapshot = _open_package(
                destination,
                expected_names=frozenset({"plan.json", "plan.sha256", "COMMITTED"}),
                namespace_root=run_root,
            )
            try:
                existing = _validate_plan_package(snapshot)
                if existing != dict(plan):
                    raise GaugeGate1Error(
                        "run root contains a different immutable plan"
                    )
                return False
            finally:
                snapshot.close()
        # Every result snapshot pins its complete ancestor chain.  Creating a
        # sibling job directory after any result exists would therefore look
        # like a namespace-history attack.  Materialize the exact 17 immutable
        # parents under authority before the plan can become visible.
        _prepare_record_parents(run_root, jobs)
        _validate_record_parents(run_root, jobs)
        stage, stage_stat = _stage_package(
            staging_parent=run_root / STAGING_DIRECTORY,
            basename="plan",
            members=_plan_members(plan),
        )
        snapshot = _publish_package(
            stage=stage,
            stage_stat=stage_stat,
            destination=destination,
            expected_names=frozenset({"plan.json", "plan.sha256", "COMMITTED"}),
            validator=_validate_plan_package,
            namespace_root=run_root,
        )
        snapshot.close()
        return True


def load_plan_snapshot(run_root: Path) -> tuple[dict[str, Any], PackageSnapshot]:
    snapshot = _open_package(
        _absolute(run_root / PLAN_DIRECTORY),
        expected_names=frozenset({"plan.json", "plan.sha256", "COMMITTED"}),
        namespace_root=run_root,
    )
    try:
        plan = _validate_plan_package(snapshot)
        if plan["run_root"] != str(_absolute(run_root)):
            raise GaugeGate1Error("plan is bound to a different run root")
        _assert_authority_fence(run_root, plan["authority_fence"])
        return plan, snapshot
    except Exception:
        snapshot.close()
        raise


def _record_path(run_root: Path, job: ScreenJob) -> Path:
    return (
        _absolute(run_root)
        / _record_parent_relative(job)
        / f"seed_{job.seed:03d}"
    )


def _record_parent_relative(job: ScreenJob) -> Path:
    return (
        Path(RECORDS_DIRECTORY)
        / job.dataset
        / job.model
        / f"subject_{job.subject:03d}"
        / f"fold_{job.fold:02d}"
    )


def _claim_path(run_root: Path, job: ScreenJob) -> Path:
    return _absolute(run_root / CLAIMS_DIRECTORY / f"{job.job_id}.json")


def _stage_prefix(job: ScreenJob, claim_nonce: str) -> str:
    return f".stage.result.{job.job_id}.{claim_nonce}."


def _prepare_record_parents(run_root: Path, jobs: Sequence[ScreenJob]) -> None:
    """Create every exact job parent before the immutable plan is published."""

    if not jobs or len({job.job_id for job in jobs}) != len(jobs):
        raise GaugeGate1Error("record-parent job set is empty or duplicated")
    for job in jobs:
        _secure_mkdir_beneath(run_root, _record_parent_relative(job))


def _validate_record_parent(run_root: Path, job: ScreenJob) -> None:
    """Validate a precreated job parent; workers never mutate shared layout."""

    _validate_directory_beneath(run_root, _record_parent_relative(job))


def _validate_record_parents(
    run_root: Path,
    jobs: Sequence[ScreenJob],
) -> None:
    if not jobs or len({job.job_id for job in jobs}) != len(jobs):
        raise GaugeGate1Error("record-parent job set is empty or duplicated")
    for job in jobs:
        _validate_record_parent(run_root, job)


def _validate_timestamp(value: Any, *, path: str) -> None:
    if type(value) is not str or not value:
        raise GaugeGate1Error(f"{path} is invalid")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise GaugeGate1Error(f"{path} is invalid") from error
    if parsed.tzinfo is None:
        raise GaugeGate1Error(f"{path} lacks a timezone")


def _validate_resource_guard(
    value: Any,
    *,
    plan: Mapping[str, Any],
    expected_gpu_uuid: str,
) -> None:
    _exact_keys(value, {"gpu", "disk"}, path="resource_guard")
    gpu = value["gpu"]
    _exact_keys(
        gpu,
        {
            "gpu_uuid",
            "pci_bus_id",
            "name",
            "utilization_percent",
            "memory_used_mib",
            "own_pid",
            "own_compute_memory_mib",
            "foreign_compute_processes",
        },
        path="resource_guard.gpu",
    )
    roster = {
        item["gpu_uuid"]: item for item in plan["gpu_roster"]
    }
    expected = roster.get(expected_gpu_uuid)
    if (
        expected is None
        or gpu["gpu_uuid"] != expected_gpu_uuid
        or gpu["pci_bus_id"] != expected["pci_bus_id"]
        or gpu["name"] != expected["name"]
        or gpu["foreign_compute_processes"] != []
    ):
        raise GaugeGate1Error("resource guard is not bound to planned GPU")
    _require_int(gpu["own_pid"], path="resource_guard.gpu.own_pid", minimum=1)
    _require_number(
        gpu["utilization_percent"],
        path="resource_guard.gpu.utilization_percent",
        minimum=0,
        maximum=100,
    )
    for name in ("memory_used_mib", "own_compute_memory_mib"):
        _require_number(gpu[name], path=f"resource_guard.gpu.{name}", minimum=0)
    disk = value["disk"]
    if not isinstance(disk, list) or len(disk) != 2:
        raise GaugeGate1Error("resource disk guard count changed")
    expected_paths = [plan["run_root"], plan["cache_root"]]
    observed_paths: list[str] = []
    for item in disk:
        _exact_keys(
            item,
            {"path", "free_bytes", "free_gib", "minimum_free_gib"},
            path="resource_guard.disk",
        )
        if (
            type(item["path"]) is not str
            or item["minimum_free_gib"] != MINIMUM_FREE_GIB
        ):
            raise GaugeGate1Error("disk guard identity changed")
        observed_paths.append(item["path"])
        free_bytes = _require_int(
            item["free_bytes"], path="disk.free_bytes", minimum=1
        )
        free_gib = _require_number(
            item["free_gib"],
            path="disk.free_gib",
            minimum=MINIMUM_FREE_GIB,
        )
        if not math.isclose(
            free_gib,
            free_bytes / float(1024**3),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise GaugeGate1Error("disk byte/GiB values are inconsistent")
    if observed_paths != expected_paths:
        raise GaugeGate1Error("disk guard paths differ from plan")


def _validate_claim(
    value: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    job: ScreenJob,
    run_root: Path,
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
        },
        path="claim",
    )
    if (
        value["schema"] != CLAIM_SCHEMA
        or value["plan_sha256"] != plan["plan_sha256"]
        or value["job"] != job.identity()
        or value["job_id"] != job.job_id
        or type(value["nonce"]) is not str
        or not NONCE_RE.fullmatch(value["nonce"])
    ):
        raise GaugeGate1Error("claim identity differs")
    _validate_timestamp(value["created_at"], path="claim.created_at")
    owner = value["owner"]
    _exact_keys(
        owner,
        {"host", "pid", "boot_id", "start_ticks"},
        path="claim.owner",
    )
    if (
        type(owner["host"]) is not str
        or not owner["host"]
        or type(owner["pid"]) is not int
        or owner["pid"] <= 0
        or type(owner["boot_id"]) is not str
        or not project_gpu_leases.BOOT_ID_RE.fullmatch(owner["boot_id"])
        or type(owner["start_ticks"]) is not int
        or owner["start_ticks"] <= 0
    ):
        raise GaugeGate1Error("claim owner identity is invalid")
    lease = value["gpu_lease_receipt"]
    try:
        gpu_uuid = lease["lease"]["gpu_uuid"]
        project_gpu_leases.validate_gpu_lease_receipt(
            lease,
            project_root=Path(plan["project_root"]),
            run_root=run_root,
            plan_sha256=plan["plan_sha256"],
            gpu_uuid=gpu_uuid,
            track_scope=TRACK_SCOPE,
        )
    except (KeyError, TypeError, project_gpu_leases.ProjectGPULeaseError) as error:
        raise GaugeGate1Error(f"claim GPU lease receipt is invalid: {error}") from error
    _validate_resource_guard(
        value["resource_guard"],
        plan=plan,
        expected_gpu_uuid=gpu_uuid,
    )
    if (
        lease["lease"]["owner"] != owner
        or value["resource_guard"]["gpu"]["own_pid"] != owner["pid"]
    ):
        raise GaugeGate1Error(
            "claim owner, GPU process, and lease owner differ"
        )


def _owner_is_live(owner: Mapping[str, Any]) -> bool:
    # The shared helper validates the complete boot-id/PID/start-tick tuple and
    # deliberately treats an unknown remote host as live (never steal).
    return project_gpu_leases._owner_is_live(owner)  # noqa: SLF001


def _open_claim(
    path: Path,
    *,
    plan: Mapping[str, Any],
    job: ScreenJob,
) -> ClaimHandle:
    # Claims are mutable coordination records shared by three workers, not
    # persisted scientific evidence. Their directory is intentionally not a
    # quiescent namespace; authority-lock and exact-inode checks protect them.
    snapshot = _open_file_snapshot(
        path,
        maximum=MAX_JSON_BYTES,
        track_namespace=False,
    )
    try:
        value = _strict_json_bytes(snapshot.payload, source=str(path))
        _validate_claim(value, plan=plan, job=job, run_root=Path(plan["run_root"]))
        return ClaimHandle(
            job=job,
            path=snapshot.path,
            descriptor=snapshot.descriptor,
            observed=snapshot.observed,
            value=value,
        )
    except Exception:
        snapshot.close()
        raise


def _assert_claim_handle(
    claim: ClaimHandle,
    *,
    plan: Mapping[str, Any],
) -> None:
    snapshot = FileSnapshot(
        path=claim.path,
        descriptor=claim.descriptor,
        observed=claim.observed,
        payload=_canonical_bytes(claim.value) + b"\n",
    )
    _assert_file_snapshot(snapshot)
    parsed = _strict_json_bytes(snapshot.payload, source=str(claim.path))
    _validate_claim(
        parsed,
        plan=plan,
        job=claim.job,
        run_root=Path(plan["run_root"]),
    )
    if parsed != dict(claim.value):
        raise GaugeGate1Error("claim content differs from handle")


def _publish_claim(
    *,
    run_root: Path,
    plan: Mapping[str, Any],
    job: ScreenJob,
    value: Mapping[str, Any],
) -> ClaimHandle:
    path = _claim_path(run_root, job)
    payload = _canonical_bytes(value) + b"\n"
    stage = path.parent / (
        f".claimstage.{job.job_id}.{os.getpid()}.{uuid.uuid4().hex}"
    )
    stage_stat = _write_file_exclusive(stage, payload, 0o444)
    published = False
    try:
        try:
            _rename_noreplace(stage, path)
            published = True
        except Exception:
            published = _entry_matches(path, stage_stat, directory=False)
            raise
        _fsync_directory(path.parent)
        claim = _open_claim(path, plan=plan, job=job)
        if (
            claim.observed.st_dev,
            claim.observed.st_ino,
        ) != (stage_stat.st_dev, stage_stat.st_ino):
            os.close(claim.descriptor)
            raise GaugeGate1Error("claim inode differs from staged claim")
        return claim
    except Exception:
        if published or _entry_matches(path, stage_stat, directory=False):
            _hide_exact_entry(
                path,
                stage_stat,
                label="invalid-claim-postrename",
                directory=False,
            )
        raise


def _hide_claim(
    claim: ClaimHandle,
    *,
    label: str,
) -> Path:
    _assert_path_binds_file(claim.path, claim.observed)
    hidden = _hide_exact_entry(
        claim.path,
        claim.observed,
        label=label,
        directory=False,
    )
    return hidden


def _runtime_binding_stack(
    *,
    plan: Mapping[str, Any],
    job: ScreenJob,
) -> tuple[ExitStack, PackageSnapshot, MultiFileSnapshot, FileSnapshot, MultiFileSnapshot]:
    stack = ExitStack()
    try:
        loaded, plan_snapshot = load_plan_snapshot(Path(plan["run_root"]))
        stack.callback(plan_snapshot.close)
        if loaded != dict(plan):
            raise GaugeGate1Error("live plan differs from worker plan")
        _, source = _source_snapshot(
            Path(plan["project_root"]),
            expected=plan["source_identity"],
        )
        stack.callback(source.close)
        cache = _cache_snapshot(
            Path(plan["cache_root"]),
            plan["cache_identity"][_subject_key(job)],
        )
        stack.callback(cache.close)
        _, reference, _ = _reference_snapshot(
            Path(plan["reference_identity"]["root"]),
            expected=plan["reference_identity"],
        )
        stack.callback(reference.close)
        observed_environment = environment_identity()
        if observed_environment != plan["environment_identity"]:
            raise GaugeGate1Error("UV environment differs from immutable plan")
        validate_plan(plan)
        plan_snapshot.revalidate()
        source.revalidate()
        _assert_file_snapshot(cache)
        reference.revalidate()
        return stack, plan_snapshot, source, cache, reference
    except Exception:
        stack.close()
        raise


def _runtime_binding_revalidate(
    *,
    plan: Mapping[str, Any],
    plan_snapshot: PackageSnapshot,
    source: MultiFileSnapshot,
    cache: FileSnapshot,
    reference: MultiFileSnapshot,
) -> None:
    validate_plan(plan)
    plan_snapshot.revalidate()
    source.revalidate()
    _assert_file_snapshot(cache)
    reference.revalidate()
    if environment_identity() != plan["environment_identity"]:
        raise GaugeGate1Error("UV environment changed across publication")


def _probe_disk(path: Path) -> dict[str, Any]:
    absolute = _absolute(path)
    descriptor = _open_absolute_directory(absolute)
    try:
        observed = os.fstatvfs(descriptor)
    finally:
        os.close(descriptor)
    free_bytes = int(observed.f_bavail) * int(observed.f_frsize)
    free_gib = free_bytes / float(1024**3)
    if free_gib < MINIMUM_FREE_GIB:
        raise ResourceUnavailable(
            f"{free_gib:.2f} GiB at {absolute} is below "
            f"{MINIMUM_FREE_GIB:.0f} GiB"
        )
    return {
        "path": str(absolute),
        "free_bytes": free_bytes,
        "free_gib": free_gib,
        "minimum_free_gib": MINIMUM_FREE_GIB,
    }


def _probe_gpu(gpu_uuid: str, *, plan: Mapping[str, Any]) -> dict[str, Any]:
    if not GPU_UUID_RE.fullmatch(gpu_uuid):
        raise GaugeGate1Error("GPU UUID is invalid")
    try:
        gpu_result = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                gpu_uuid,
                "--query-gpu=uuid,pci.bus_id,name,utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        process_result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ResourceUnavailable(f"NVIDIA guard failed: {error}") from error
    rows = [
        [value.strip() for value in line.split(",")]
        for line in gpu_result.stdout.splitlines()
        if line.strip()
    ]
    if len(rows) != 1 or len(rows[0]) != 5 or rows[0][0] != gpu_uuid:
        raise ResourceUnavailable("NVIDIA query did not bind exact GPU UUID")
    own_pid = os.getpid()
    own_memory = 0.0
    foreign: list[dict[str, Any]] = []
    for line in process_result.stdout.splitlines():
        if not line.strip():
            continue
        fields = [value.strip() for value in line.split(",")]
        if len(fields) != 4 or fields[1] != gpu_uuid:
            continue
        try:
            pid = int(fields[0])
            memory = float(fields[3])
        except ValueError as error:
            raise ResourceUnavailable("NVIDIA process row is malformed") from error
        if pid == own_pid:
            own_memory += memory
        else:
            foreign.append(
                {
                    "pid": pid,
                    "process_name": fields[2],
                    "used_gpu_memory_mib": memory,
                }
            )
    if foreign:
        raise ResourceUnavailable(
            f"GPU {gpu_uuid} has foreign compute PIDs "
            f"{[item['pid'] for item in foreign]}"
        )
    value = {
        "gpu_uuid": gpu_uuid,
        "pci_bus_id": rows[0][1],
        "name": rows[0][2],
        "utilization_percent": float(rows[0][3]),
        "memory_used_mib": float(rows[0][4]),
        "own_pid": own_pid,
        "own_compute_memory_mib": own_memory,
        "foreign_compute_processes": [],
    }
    expected = next(
        item for item in plan["gpu_roster"] if item["gpu_uuid"] == gpu_uuid
    )
    if (
        value["pci_bus_id"] != expected["pci_bus_id"]
        or value["name"] != expected["name"]
    ):
        raise ResourceUnavailable("GPU PCI/name identity differs from plan")
    return value


def _resource_guard(gpu_uuid: str, *, plan: Mapping[str, Any]) -> dict[str, Any]:
    value = {
        "gpu": _probe_gpu(gpu_uuid, plan=plan),
        "disk": [
            _probe_disk(Path(plan["run_root"])),
            _probe_disk(Path(plan["cache_root"])),
        ],
    }
    _validate_resource_guard(
        value,
        plan=plan,
        expected_gpu_uuid=gpu_uuid,
    )
    return value


def _claim_for_job(
    *,
    plan: Mapping[str, Any],
    job: ScreenJob,
    gpu_lease: project_gpu_leases.GPULease,
    resource_guard: Mapping[str, Any],
) -> ClaimHandle:
    run_root = Path(plan["run_root"])
    path = _claim_path(run_root, job)
    published_claim: ClaimHandle | None = None
    try:
        with project_gpu_leases.guard_gpu_lease(gpu_lease) as receipt:
            with _authority_lock(
                run_root,
                expected=plan["authority_fence"],
            ):
                stack, plan_snapshot, source, cache, reference = (
                    _runtime_binding_stack(
                        plan=plan,
                        job=job,
                    )
                )
                with stack:
                    if _record_path(run_root, job).exists():
                        snapshot = load_result_snapshot(run_root, job, plan)
                        snapshot.close()
                        raise ClaimUnavailable("job already has a valid result")
                    if path.exists():
                        existing = _open_claim(path, plan=plan, job=job)
                        try:
                            if _owner_is_live(existing.value["owner"]):
                                raise ClaimUnavailable("job has a live claim")
                            _hide_claim(existing, label="stale-claim")
                            _quarantine_stages_for_claim(
                                run_root=run_root,
                                job=job,
                                claim_nonce=existing.value["nonce"],
                            )
                        finally:
                            os.close(existing.descriptor)
                    value = {
                        "schema": CLAIM_SCHEMA,
                        "created_at": _utc_now(),
                        "plan_sha256": plan["plan_sha256"],
                        "job": job.identity(),
                        "job_id": job.job_id,
                        "nonce": uuid.uuid4().hex,
                        "owner": project_gpu_leases.owner_identity(),
                        "resource_guard": copy.deepcopy(dict(resource_guard)),
                        "gpu_lease_receipt": copy.deepcopy(receipt),
                    }
                    _validate_claim(value, plan=plan, job=job, run_root=run_root)
                    _quarantine_orphan_private_entries(
                        parent=run_root / CLAIMS_DIRECTORY,
                        prefix=f".claimstage.{job.job_id}.",
                        label="orphan-claim-stage",
                        directory=False,
                    )
                    published_claim = _publish_claim(
                        run_root=run_root,
                        plan=plan,
                        job=job,
                        value=value,
                    )
                    _runtime_binding_revalidate(
                        plan=plan,
                        plan_snapshot=plan_snapshot,
                        source=source,
                        cache=cache,
                        reference=reference,
                    )
        if published_claim is None:
            raise AssertionError("claim guard exited without a claim")
        return published_claim
    except Exception:
        if published_claim is not None:
            try:
                with _authority_lock(
                    run_root,
                    expected=plan["authority_fence"],
                ):
                    if _entry_matches(
                        published_claim.path,
                        published_claim.observed,
                        directory=False,
                    ):
                        _hide_claim(
                            published_claim,
                            label="invalid-claim-guard-exit",
                        )
            finally:
                os.close(published_claim.descriptor)
        raise


def _quarantine_stages_for_claim(
    *,
    run_root: Path,
    job: ScreenJob,
    claim_nonce: str,
) -> tuple[Path, ...]:
    stage_root = _absolute(run_root / STAGING_DIRECTORY)
    descriptor = _open_absolute_directory(stage_root)
    hidden: list[Path] = []
    try:
        prefix = _stage_prefix(job, claim_nonce)
        names = sorted(name for name in os.listdir(descriptor) if name.startswith(prefix))
        for name in names:
            observed = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            if not stat.S_ISDIR(observed.st_mode):
                raise GaugeGate1Error("claim-linked stage is not a directory")
            path = stage_root / name
            hidden_path = _hide_exact_entry(
                path,
                observed,
                label="stale-stage",
                directory=True,
            )
            # Only after the authoritative name is hidden may permissions be
            # changed for forensic retention.
            _seal_tree(hidden_path)
            hidden.append(hidden_path)
        return tuple(hidden)
    finally:
        os.close(descriptor)


def _quarantine_orphan_private_entries(
    *,
    parent: Path,
    prefix: str,
    label: str,
    directory: bool,
) -> tuple[Path, ...]:
    """Hide exact non-authoritative leftovers while caller holds run fence."""

    descriptor = _open_absolute_directory(parent)
    hidden: list[Path] = []
    try:
        names = sorted(name for name in os.listdir(descriptor) if name.startswith(prefix))
        for name in names:
            observed = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            expected_kind = (
                stat.S_ISDIR(observed.st_mode)
                if directory
                else stat.S_ISREG(observed.st_mode)
            )
            if not expected_kind:
                raise GaugeGate1Error("orphan private entry has an unsafe type")
            destination = _hide_exact_entry(
                parent / name,
                observed,
                label=label,
                directory=directory,
            )
            if directory:
                _seal_tree(destination)
            else:
                os.chmod(destination, 0o444)
            hidden.append(destination)
        return tuple(hidden)
    finally:
        os.close(descriptor)


def _recover_completed_auxiliary(
    *,
    plan: Mapping[str, Any],
    job: ScreenJob,
    gpu_lease: project_gpu_leases.GPULease,
) -> None:
    """Archive only a stale exact claim left after a valid durable result."""

    run_root = Path(plan["run_root"])
    claim_path = _claim_path(run_root, job)
    if not claim_path.exists():
        return
    with project_gpu_leases.guard_gpu_lease(gpu_lease):
        with _authority_lock(
            run_root,
            expected=plan["authority_fence"],
        ):
            stack, plan_snapshot, source, cache, reference = (
                _runtime_binding_stack(plan=plan, job=job)
            )
            with stack:
                result = load_result_snapshot(run_root, job, plan)
                try:
                    claim = _open_claim(claim_path, plan=plan, job=job)
                    try:
                        if _owner_is_live(claim.value["owner"]):
                            return
                        _hide_claim(claim, label="completed-stale-claim")
                        _quarantine_stages_for_claim(
                            run_root=run_root,
                            job=job,
                            claim_nonce=claim.value["nonce"],
                        )
                    finally:
                        os.close(claim.descriptor)
                    _runtime_binding_revalidate(
                        plan=plan,
                        plan_snapshot=plan_snapshot,
                        source=source,
                        cache=cache,
                        reference=reference,
                    )
                    result.revalidate()
                finally:
                    result.close()


def validate_record(
    record: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    job: ScreenJob,
    run_root: Path,
) -> None:
    expected_top = {
        "schema",
        "created_at",
        "plan_sha256",
        "job",
        "job_id",
        "evaluation_scope",
        "confirmation_evidence",
        "opened_development_only",
        "candidate_only",
        "cache_identity",
        "split",
        "source_identity_sha256",
        "environment_identity_sha256",
        "reference_identity_sha256",
        "scaler",
        "model",
        "fit",
        "validation_metrics",
        "timing",
        "resource_guard",
        "gpu_lease_receipt",
    }
    _exact_keys(record, expected_top, path="record")
    if (
        record["schema"] != RECORD_SCHEMA
        or record["plan_sha256"] != plan["plan_sha256"]
        or record["job"] != job.identity()
        or record["job_id"] != job.job_id
        or record["evaluation_scope"] != EVALUATION_SCOPE
        or record["confirmation_evidence"] is not False
        or record["opened_development_only"] is not True
        or record["candidate_only"] is not True
    ):
        raise GaugeGate1Error("record identity or evidence scope changed")
    _validate_timestamp(record["created_at"], path="record.created_at")
    if record["cache_identity"] != plan["cache_identity"][_subject_key(job)]:
        raise GaugeGate1Error("record cache binding differs")
    if record["split"] != plan["split_identity"][_job_key(job)]:
        raise GaugeGate1Error("record split binding differs")
    if (
        record["source_identity_sha256"] != plan["source_identity_sha256"]
        or record["environment_identity_sha256"]
        != plan["environment_identity_sha256"]
        or record["reference_identity_sha256"]
        != plan["reference_identity"]["snapshot_sha256"]
    ):
        raise GaugeGate1Error("record immutable identity digest differs")
    scaler = record["scaler"]
    _exact_keys(
        scaler,
        {"fit_scope", "mean_sha256", "std_sha256", "contract"},
        path="record.scaler",
    )
    if scaler["fit_scope"] != "training_only" or scaler["contract"] != dict(
        CHANNEL_SCALING
    ):
        raise GaugeGate1Error("scaler scope/contract differs")
    _validate_sha(scaler["mean_sha256"], path="record.scaler.mean_sha256")
    _validate_sha(scaler["std_sha256"], path="record.scaler.std_sha256")
    model = record["model"]
    _exact_keys(
        model,
        {
            "identity",
            "parameter_count",
            "constructed_state_sha256",
            "initial_state_sha256",
            "selected_state_sha256",
            "source_statistics",
        },
        path="record.model",
    )
    identity = model["identity"]
    if (
        not isinstance(identity, Mapping)
        or identity.get("requested_name") != MODEL_NAME
        or identity.get("class")
        != "ieee_mi.gauge_quotient.GaugeQuotientCrossMomentNet"
        or identity.get("uses_positions") is not True
        or not set(identity).issubset(
            {"requested_name", "class", "uses_positions", "config"}
        )
    ):
        raise GaugeGate1Error("Gauge model identity differs")
    count = _require_int(
        model["parameter_count"],
        path="record.model.parameter_count",
        minimum=1,
    )
    if count > MAXIMUM_PARAMETERS:
        raise GaugeGate1Error("Gauge parameter count exceeds frozen ceiling")
    for name in (
        "constructed_state_sha256",
        "initial_state_sha256",
        "selected_state_sha256",
    ):
        _validate_sha(model[name], path=f"record.model.{name}")
    if model["constructed_state_sha256"] != model["initial_state_sha256"]:
        raise GaugeGate1Error("Gauge initial state changed before fitting")
    if model["source_statistics"] != {
        "available": False,
        "ran": False,
        "fit_scope": "training_rows_only",
        "outcome_aware": False,
    }:
        raise GaugeGate1Error("Gauge source-statistics contract differs")
    fit = record["fit"]
    _exact_keys(
        fit,
        {
            "best_epoch",
            "epochs_run",
            "best_validation_loss",
            "early_stopping",
            "config",
        },
        path="record.fit",
    )
    epochs = plan["train_config"]["epochs"]
    best_epoch = _require_int(
        fit["best_epoch"], path="record.fit.best_epoch", minimum=0
    )
    epochs_run = _require_int(
        fit["epochs_run"], path="record.fit.epochs_run", minimum=1
    )
    if (
        best_epoch >= epochs_run
        or epochs_run > epochs
        or fit["early_stopping"] is not True
        or fit["config"] != plan["train_config"]
    ):
        raise GaugeGate1Error("fit epoch/config contract differs")
    _require_number(
        fit["best_validation_loss"],
        path="record.fit.best_validation_loss",
        minimum=0,
    )
    metrics = record["validation_metrics"]
    if not isinstance(metrics, Mapping) or set(metrics) != set(METRIC_NAMES):
        raise GaugeGate1Error("validation metric set differs")
    for name, value in metrics.items():
        if name in {"accuracy", "balanced_accuracy", "roc_auc"}:
            _require_number(
                value, path=f"record.validation_metrics.{name}", minimum=0, maximum=1
            )
        elif name == "cohen_kappa":
            _require_number(
                value, path=f"record.validation_metrics.{name}", minimum=-1, maximum=1
            )
        else:
            _require_number(
                value, path=f"record.validation_metrics.{name}", minimum=0
            )
    timing = record["timing"]
    _exact_keys(
        timing,
        {
            "construction_seconds",
            "fit_seconds",
            "evaluation_seconds",
            "total_seconds",
        },
        path="record.timing",
    )
    components = [
        _require_number(timing[name], path=f"record.timing.{name}", minimum=0)
        for name in (
            "construction_seconds",
            "fit_seconds",
            "evaluation_seconds",
        )
    ]
    total = _require_number(
        timing["total_seconds"],
        path="record.timing.total_seconds",
        minimum=0,
    )
    if total + 1e-9 < sum(components):
        raise GaugeGate1Error("total timing is smaller than timed components")
    try:
        gpu_uuid = record["gpu_lease_receipt"]["lease"]["gpu_uuid"]
        project_gpu_leases.validate_gpu_lease_receipt(
            record["gpu_lease_receipt"],
            project_root=Path(plan["project_root"]),
            run_root=run_root,
            plan_sha256=plan["plan_sha256"],
            gpu_uuid=gpu_uuid,
            track_scope=TRACK_SCOPE,
        )
    except (KeyError, TypeError, project_gpu_leases.ProjectGPULeaseError) as error:
        raise GaugeGate1Error(f"record GPU lease receipt is invalid: {error}") from error
    _validate_resource_guard(
        record["resource_guard"],
        plan=plan,
        expected_gpu_uuid=gpu_uuid,
    )
    try:
        lease_owner_pid = record["gpu_lease_receipt"]["lease"]["owner"]["pid"]
    except (KeyError, TypeError) as error:
        raise GaugeGate1Error("record lease owner is incomplete") from error
    if record["resource_guard"]["gpu"]["own_pid"] != lease_owner_pid:
        raise GaugeGate1Error(
            "record GPU process and persisted lease owner differ"
        )
    _assert_no_excluded_evidence(record)


def _result_members(
    record: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    job: ScreenJob,
    claim_nonce: str,
) -> dict[str, bytes]:
    record_bytes = _canonical_bytes(record) + b"\n"
    completion = {
        "schema": COMPLETION_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "job": job.identity(),
        "job_id": job.job_id,
        "claim_nonce": claim_nonce,
        "record_sha256": _sha256_bytes(record_bytes),
        "record_size_bytes": len(record_bytes),
        "evaluation_scope": EVALUATION_SCOPE,
        "confirmation_evidence": False,
    }
    return {
        "record.json": record_bytes,
        "completion.json": _canonical_bytes(completion) + b"\n",
        "COMMITTED": RESULT_MARKER,
    }


def _validate_result_package(
    snapshot: PackageSnapshot,
    *,
    plan: Mapping[str, Any],
    job: ScreenJob,
) -> dict[str, Any]:
    if snapshot.children["COMMITTED"].payload != RESULT_MARKER:
        raise GaugeGate1Error("result commit marker differs")
    record_bytes = snapshot.children["record.json"].payload
    record = _strict_json_bytes(record_bytes, source=str(snapshot.path / "record.json"))
    validate_record(
        record,
        plan=plan,
        job=job,
        run_root=Path(plan["run_root"]),
    )
    completion = _strict_json_bytes(
        snapshot.children["completion.json"].payload,
        source=str(snapshot.path / "completion.json"),
    )
    _exact_keys(
        completion,
        {
            "schema",
            "plan_sha256",
            "job",
            "job_id",
            "claim_nonce",
            "record_sha256",
            "record_size_bytes",
            "evaluation_scope",
            "confirmation_evidence",
        },
        path="completion",
    )
    if (
        completion["schema"] != COMPLETION_SCHEMA
        or completion["plan_sha256"] != plan["plan_sha256"]
        or completion["job"] != job.identity()
        or completion["job_id"] != job.job_id
        or type(completion["claim_nonce"]) is not str
        or not NONCE_RE.fullmatch(completion["claim_nonce"])
        or completion["record_sha256"] != _sha256_bytes(record_bytes)
        or completion["record_size_bytes"] != len(record_bytes)
        or completion["evaluation_scope"] != EVALUATION_SCOPE
        or completion["confirmation_evidence"] is not False
    ):
        raise GaugeGate1Error("completion binding differs")
    snapshot.revalidate()
    return record


def load_result_snapshot(
    run_root: Path,
    job: ScreenJob,
    plan: Mapping[str, Any],
) -> PackageSnapshot:
    snapshot = _open_package(
        _record_path(run_root, job),
        expected_names=frozenset({"record.json", "completion.json", "COMMITTED"}),
        namespace_root=run_root,
    )
    try:
        _validate_result_package(snapshot, plan=plan, job=job)
        return snapshot
    except Exception:
        snapshot.close()
        raise


def _execute_job(
    *,
    job: ScreenJob,
    plan: Mapping[str, Any],
    resource_guard: Mapping[str, Any],
    gpu_lease_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    from .data import apply_channel_scaler, fit_channel_scaler
    from .gauge_quotient import make_gauge_quotient_model
    from .training import (
        TrainConfig,
        classification_metrics,
        configure_determinism,
        fit_model,
        predict_probabilities,
    )

    started = time.perf_counter()
    (
        x_train_raw,
        y_train,
        x_validation_raw,
        y_validation,
        positions,
        channel_names,
    ) = _load_validation_data(job=job, plan=plan)
    scaler_mean, scaler_std = fit_channel_scaler(x_train_raw, channel_names)
    x_train = apply_channel_scaler(x_train_raw, scaler_mean, scaler_std)
    x_validation = apply_channel_scaler(
        x_validation_raw, scaler_mean, scaler_std
    )
    configure_determinism(job.seed)
    construction_started = time.perf_counter()
    preprocessing = preprocessing_for_dataset(job.dataset)
    model = make_gauge_quotient_model(
        requested_model=job.model,
        n_channels=x_train.shape[1],
        n_outputs=dataset_spec(job.dataset).n_classes,
        n_times=x_train.shape[2],
        sfreq=float(preprocessing["sfreq_hz"]),
        channel_names=channel_names,
        channel_positions=torch.as_tensor(positions, dtype=torch.float32),
    )
    construction_seconds = time.perf_counter() - construction_started
    constructed_state = _state_sha256(model)
    initial_state = _state_sha256(model)
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    identity = _model_identity(model, job.model)
    if callable(getattr(model, "fit_source_statistics", None)):
        raise GaugeGate1Error("Gauge unexpectedly exposes source-statistics fitting")
    config = TrainConfig(**dict(plan["train_config"]))
    fit = fit_model(
        model,
        x_train,
        y_train,
        x_validation,
        y_validation,
        positions,
        config=config,
        mirror_index=None,
    )
    evaluation_started = time.perf_counter()
    probabilities = predict_probabilities(
        fit["model"],
        x_validation,
        positions,
        device="cuda:0",
    )
    metrics = classification_metrics(y_validation, probabilities)
    evaluation_seconds = time.perf_counter() - evaluation_started
    selected_state = _state_sha256(fit["model"])
    total_seconds = time.perf_counter() - started
    record = {
        "schema": RECORD_SCHEMA,
        "created_at": _utc_now(),
        "plan_sha256": plan["plan_sha256"],
        "job": job.identity(),
        "job_id": job.job_id,
        "evaluation_scope": EVALUATION_SCOPE,
        "confirmation_evidence": False,
        "opened_development_only": True,
        "candidate_only": True,
        "cache_identity": copy.deepcopy(
            plan["cache_identity"][_subject_key(job)]
        ),
        "split": copy.deepcopy(plan["split_identity"][_job_key(job)]),
        "source_identity_sha256": plan["source_identity_sha256"],
        "environment_identity_sha256": plan["environment_identity_sha256"],
        "reference_identity_sha256": plan["reference_identity"]["snapshot_sha256"],
        "scaler": {
            "fit_scope": "training_only",
            "mean_sha256": _array_sha256(scaler_mean),
            "std_sha256": _array_sha256(scaler_std),
            "contract": dict(CHANNEL_SCALING),
        },
        "model": {
            "identity": _json_safe(identity),
            "parameter_count": int(parameter_count),
            "constructed_state_sha256": constructed_state,
            "initial_state_sha256": initial_state,
            "selected_state_sha256": selected_state,
            "source_statistics": {
                "available": False,
                "ran": False,
                "fit_scope": "training_rows_only",
                "outcome_aware": False,
            },
        },
        "fit": {
            "best_epoch": int(fit["best_epoch"]),
            "epochs_run": int(fit["epochs_run"]),
            "best_validation_loss": float(fit["best_validation_loss"]),
            "early_stopping": True,
            "config": dict(fit["config"]),
        },
        "validation_metrics": {
            name: float(metrics[name]) for name in sorted(METRIC_NAMES)
        },
        "timing": {
            "construction_seconds": construction_seconds,
            "fit_seconds": float(fit["fit_seconds"]),
            "evaluation_seconds": evaluation_seconds,
            "total_seconds": total_seconds,
        },
        "resource_guard": copy.deepcopy(dict(resource_guard)),
        "gpu_lease_receipt": copy.deepcopy(dict(gpu_lease_receipt)),
    }
    validate_record(
        record,
        plan=plan,
        job=job,
        run_root=Path(plan["run_root"]),
    )
    del probabilities, fit, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return record


def _commit_result(
    *,
    plan: Mapping[str, Any],
    job: ScreenJob,
    claim: ClaimHandle,
    record: Mapping[str, Any],
    gpu_lease: project_gpu_leases.GPULease,
) -> None:
    run_root = Path(plan["run_root"])
    _validate_record_parent(run_root, job)
    members = _result_members(
        record,
        plan=plan,
        job=job,
        claim_nonce=claim.value["nonce"],
    )
    stage, stage_stat = _stage_package(
        staging_parent=run_root / STAGING_DIRECTORY,
        basename=f"result.{job.job_id}.{claim.value['nonce']}",
        members=members,
    )
    # Rename the generic private name into the claim-bound prefix.  This is
    # still non-authoritative and gives recovery an exact nonce namespace.
    claim_stage = stage.parent / (
        _stage_prefix(job, claim.value["nonce"]) + uuid.uuid4().hex
    )
    _rename_noreplace(stage, claim_stage)
    _fsync_directory(stage.parent)
    stage = claim_stage
    stage_stat = stage.lstat()
    destination = _record_path(run_root, job)
    result_was_published = False
    try:
        with project_gpu_leases.guard_gpu_lease(gpu_lease) as receipt:
            if receipt != claim.value["gpu_lease_receipt"]:
                raise GaugeGate1Error("commit lease receipt differs from claim")
            with _authority_lock(
                run_root,
                expected=plan["authority_fence"],
            ):
                _assert_claim_handle(claim, plan=plan)
                stack, plan_snapshot, source, cache, reference = (
                    _runtime_binding_stack(
                        plan=plan,
                        job=job,
                    )
                )
                with stack:
                    if destination.exists():
                        existing = load_result_snapshot(run_root, job, plan)
                        existing.close()
                        raise GaugeGate1Error("valid result appeared before commit")
                    commit_gpu_uuid = receipt["lease"]["gpu_uuid"]
                    _probe_gpu(commit_gpu_uuid, plan=plan)
                    _probe_disk(Path(plan["run_root"]))
                    _probe_disk(Path(plan["cache_root"]))

                    def validator(snapshot: PackageSnapshot) -> None:
                        parsed = _validate_result_package(
                            snapshot,
                            plan=plan,
                            job=job,
                        )
                        if parsed != dict(record):
                            raise GaugeGate1Error(
                                "published record differs from executor"
                            )

                    published = _publish_package(
                        stage=stage,
                        stage_stat=stage_stat,
                        destination=destination,
                        expected_names=frozenset(
                            {"record.json", "completion.json", "COMMITTED"}
                        ),
                        validator=validator,
                        namespace_root=run_root,
                    )
                    result_was_published = True
                    try:
                        _runtime_binding_revalidate(
                            plan=plan,
                            plan_snapshot=plan_snapshot,
                            source=source,
                            cache=cache,
                            reference=reference,
                        )
                        _probe_gpu(commit_gpu_uuid, plan=plan)
                        _probe_disk(Path(plan["run_root"]))
                        _probe_disk(Path(plan["cache_root"]))
                        published.revalidate()
                    finally:
                        published.close()
                    _hide_claim(claim, label="completed-claim")
        # This assignment is deliberately after guard_gpu_lease.__exit__.
        # Until that post-yield lease validation succeeds, the result remains
        # rollback-owned by this invocation.
        result_was_published = False
    except Exception:
        if result_was_published or _entry_matches(
            destination,
            stage_stat,
            directory=True,
        ):
            with _authority_lock(
                run_root,
                expected=plan["authority_fence"],
            ):
                if _entry_matches(
                    destination,
                    stage_stat,
                    directory=True,
                ):
                    _hide_exact_entry(
                        destination,
                        stage_stat,
                        label="invalid-result-guard-exit",
                        directory=True,
                    )
        raise


def _worker_preflight(gpu_uuid: str, *, plan: Mapping[str, Any]) -> int:
    _assert_approved_screen_dependency()
    if not sys.platform.startswith("linux"):
        raise GaugeGate1Error("formal Gauge Gate 1 workers require Linux")
    if os.environ.get("IEEE_MI_DARWIN_SYNTHETIC_TEST") is not None:
        raise GaugeGate1Error("synthetic Darwin test mode is forbidden formally")
    _require_uv_virtualenv()
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != REQUIRED_CUBLAS_WORKSPACE_CONFIG:
        raise GaugeGate1Error(
            f"CUBLAS_WORKSPACE_CONFIG must be {REQUIRED_CUBLAS_WORKSPACE_CONFIG!r}"
        )
    if os.environ.get("CUDA_VISIBLE_DEVICES") != gpu_uuid:
        raise GaugeGate1Error(
            "CUDA_VISIBLE_DEVICES must contain exactly the full worker GPU UUID"
        )
    roster = _validate_gpu_roster(plan["gpu_roster"])
    if gpu_uuid not in roster:
        raise GaugeGate1Error("worker GPU is absent from immutable roster")
    import torch

    torch.set_num_threads(CPU_THREADS)
    if torch.cuda.device_count() != 1:
        raise GaugeGate1Error("worker requires exactly one visible CUDA GPU")
    observed = str(getattr(torch.cuda.get_device_properties(0), "uuid", ""))
    if observed and observed.lower().replace("gpu-", "") != gpu_uuid.lower().replace(
        "gpu-", ""
    ):
        raise GaugeGate1Error("cuda:0 UUID differs from planned physical GPU")
    return roster.index(gpu_uuid)


def run_worker(
    *,
    project_root: Path,
    run_root: Path,
    gpu_uuid: str,
) -> dict[str, int]:
    plan, plan_snapshot = load_plan_snapshot(run_root)
    try:
        if plan["project_root"] != str(_absolute(project_root)):
            raise GaugeGate1Error("worker project root differs from plan")
        worker_index = _worker_preflight(gpu_uuid, plan=plan)
        if environment_identity() != plan["environment_identity"]:
            raise GaugeGate1Error("worker UV environment differs from plan")
        source, source_snapshot = _source_snapshot(
            project_root,
            expected=plan["source_identity"],
        )
        del source
        source_snapshot.close()
    finally:
        plan_snapshot.close()
    lease = project_gpu_leases.acquire_gpu_lease(
        project_root=project_root,
        run_root=run_root,
        plan_sha256=plan["plan_sha256"],
        gpu_uuid=gpu_uuid,
        track_scope=TRACK_SCOPE,
    )
    completed = 0
    resumed = 0
    busy = 0
    try:
        jobs = [
            _job_from_mapping(
                {
                    key: item[key]
                    for key in ("dataset", "model", "subject", "fold", "seed")
                }
            )
            for item in plan["jobs"]
            if item["worker_index"] == worker_index
        ]
        for job in jobs:
            if _record_path(run_root, job).exists():
                snapshot = load_result_snapshot(run_root, job, plan)
                snapshot.close()
                _recover_completed_auxiliary(
                    plan=plan,
                    job=job,
                    gpu_lease=lease,
                )
                resumed += 1
                continue
            resource = _resource_guard(gpu_uuid, plan=plan)
            try:
                claim = _claim_for_job(
                    plan=plan,
                    job=job,
                    gpu_lease=lease,
                    resource_guard=resource,
                )
            except ClaimUnavailable:
                busy += 1
                continue
            try:
                record = _execute_job(
                    job=job,
                    plan=plan,
                    resource_guard=resource,
                    gpu_lease_receipt=claim.value["gpu_lease_receipt"],
                )
                _commit_result(
                    plan=plan,
                    job=job,
                    claim=claim,
                    record=record,
                    gpu_lease=lease,
                )
                completed += 1
            finally:
                os.close(claim.descriptor)
    finally:
        project_gpu_leases.release_gpu_lease(lease)
    return {"completed": completed, "resumed": resumed, "busy": busy}


def status(run_root: Path) -> dict[str, Any]:
    plan, plan_snapshot = load_plan_snapshot(run_root)
    try:
        complete: list[str] = []
        missing: list[str] = []
        corrupt: dict[str, str] = {}
        claims: dict[str, str] = {}
        workers = {
            str(index): {"expected": 0, "complete": 0}
            for index in range(WORKER_COUNT)
        }
        for item in plan["jobs"]:
            job = _job_from_mapping(
                {
                    key: item[key]
                    for key in ("dataset", "model", "subject", "fold", "seed")
                }
            )
            worker = str(item["worker_index"])
            workers[worker]["expected"] += 1
            path = _record_path(run_root, job)
            if path.exists():
                try:
                    snapshot = load_result_snapshot(run_root, job, plan)
                except Exception as error:
                    corrupt[job.job_id] = str(error)
                else:
                    snapshot.close()
                    complete.append(job.job_id)
                    workers[worker]["complete"] += 1
                continue
            claim_path = _claim_path(run_root, job)
            if claim_path.exists():
                try:
                    claim = _open_claim(claim_path, plan=plan, job=job)
                except Exception as error:
                    claims[job.job_id] = f"corrupt:{error}"
                else:
                    try:
                        claims[job.job_id] = (
                            "live" if _owner_is_live(claim.value["owner"]) else "stale"
                        )
                    finally:
                        os.close(claim.descriptor)
            missing.append(job.job_id)
        plan_snapshot.revalidate()
        return {
            "schema": STATUS_SCHEMA,
            "evaluation_scope": EVALUATION_SCOPE,
            "confirmation_evidence": False,
            "plan_sha256": plan["plan_sha256"],
            "expected": EXPECTED_JOB_COUNT,
            "complete": len(complete),
            "missing": len(missing),
            "corrupt": len(corrupt),
            "workers": workers,
            "missing_job_ids": missing,
            "corrupt_results": corrupt,
            "claims": claims,
        }
    finally:
        plan_snapshot.close()


def audit(*, project_root: Path, run_root: Path) -> dict[str, Any]:
    plan, plan_snapshot = load_plan_snapshot(run_root)
    errors: list[str] = []
    try:
        if plan["project_root"] != str(_absolute(project_root)):
            errors.append("project_root_differs")
        try:
            source, source_snapshot = _source_snapshot(
                project_root,
                expected=plan["source_identity"],
            )
            del source
            source_snapshot.close()
        except Exception as error:
            errors.append(f"source:{error}")
        try:
            if environment_identity() != plan["environment_identity"]:
                errors.append("environment_differs")
        except Exception as error:
            errors.append(f"environment:{error}")
        try:
            _, reference_snapshot, _ = _reference_snapshot(
                Path(plan["reference_identity"]["root"]),
                expected=plan["reference_identity"],
            )
            reference_snapshot.close()
        except Exception as error:
            errors.append(f"reference:{error}")
        for job in _candidate_manifest():
            try:
                cache = _cache_snapshot(
                    Path(plan["cache_root"]),
                    plan["cache_identity"][_subject_key(job)],
                )
                cache.close()
            except Exception as error:
                errors.append(f"cache:{job.job_id}:{error}")
        current_status = status(run_root)
        if current_status["corrupt"]:
            errors.append("corrupt_results")
        if any(
            state.startswith("corrupt:")
            for state in current_status["claims"].values()
        ):
            errors.append("corrupt_claims")
        plan_snapshot.revalidate()
        return {
            "schema": AUDIT_SCHEMA,
            "evaluation_scope": EVALUATION_SCOPE,
            "confirmation_evidence": False,
            "plan_sha256": plan["plan_sha256"],
            "approved_for_execution": not errors,
            "error_count": len(errors),
            "errors": errors,
            "status": current_status,
            "reference_access": "read_only_reuse_no_rerun_no_mutation",
            "candidate_only": True,
        }
    finally:
        plan_snapshot.close()


def _parse_gpu_uuids(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if (
        len(values) != WORKER_COUNT
        or len(set(values)) != WORKER_COUNT
        or any(not GPU_UUID_RE.fullmatch(item) for item in values)
    ):
        raise argparse.ArgumentTypeError(
            "provide exactly three comma-separated full NVIDIA GPU UUIDs"
        )
    return values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    initialize = commands.add_parser(
        "init",
        help="assemble and atomically publish the immutable Gate 1 plan",
    )
    initialize.add_argument("--project-root", required=True, type=Path)
    initialize.add_argument("--run-root", required=True, type=Path)
    initialize.add_argument("--reference-run-root", required=True, type=Path)
    initialize.add_argument("--gpu-uuids", required=True, type=_parse_gpu_uuids)
    layout = commands.add_parser(
        "layout-check",
        help="verify source/run/reference/cache roots without writing",
    )
    layout.add_argument("--project-root", required=True, type=Path)
    layout.add_argument("--run-root", required=True, type=Path)
    layout.add_argument("--reference-run-root", required=True, type=Path)
    layout.add_argument("--cache-root", required=True, type=Path)
    worker = commands.add_parser("worker")
    worker.add_argument("--project-root", required=True, type=Path)
    worker.add_argument("--run-root", required=True, type=Path)
    worker.add_argument("--gpu-uuid", required=True)
    status_parser = commands.add_parser("status")
    status_parser.add_argument("--run-root", required=True, type=Path)
    audit_parser = commands.add_parser("audit")
    audit_parser.add_argument("--project-root", required=True, type=Path)
    audit_parser.add_argument("--run-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "init":
        plan = build_plan(
            project_root=arguments.project_root,
            run_root=arguments.run_root,
            reference_root=arguments.reference_run_root,
            gpu_uuids=arguments.gpu_uuids,
        )
        created = publish_or_validate_plan(
            project_root=arguments.project_root,
            run_root=arguments.run_root,
            plan=plan,
        )
        print(
            json.dumps(
                {
                    "created": created,
                    "plan_sha256": plan["plan_sha256"],
                    "job_count": EXPECTED_JOB_COUNT,
                    "candidate_only": True,
                    "evaluation_scope": EVALUATION_SCOPE,
                },
                sort_keys=True,
            )
        )
        return 0
    if arguments.command == "worker":
        print(
            json.dumps(
                run_worker(
                    project_root=arguments.project_root,
                    run_root=arguments.run_root,
                    gpu_uuid=arguments.gpu_uuid,
                ),
                sort_keys=True,
            )
        )
        return 0
    if arguments.command == "layout-check":
        print(
            json.dumps(
                preflight_launch_layout(
                    project_root=arguments.project_root,
                    run_root=arguments.run_root,
                    reference_root=arguments.reference_run_root,
                    cache_root=arguments.cache_root,
                ),
                sort_keys=True,
                indent=2,
            )
        )
        return 0
    if arguments.command == "status":
        print(json.dumps(status(arguments.run_root), sort_keys=True, indent=2))
        return 0
    if arguments.command == "audit":
        result = audit(
            project_root=arguments.project_root,
            run_root=arguments.run_root,
        )
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0 if result["approved_for_execution"] else 2
    raise AssertionError(arguments.command)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "AUDIT_SCHEMA",
    "CLAIM_SCHEMA",
    "COMPLETION_SCHEMA",
    "EXPECTED_JOB_COUNT",
    "GaugeGate1Error",
    "PLAN_SCHEMA",
    "RECORD_SCHEMA",
    "STATUS_SCHEMA",
    "assemble_plan",
    "audit",
    "build_plan",
    "load_plan_snapshot",
    "load_result_snapshot",
    "main",
    "preflight_launch_layout",
    "publish_or_validate_plan",
    "run_worker",
    "status",
    "validate_plan",
    "validate_record",
]
