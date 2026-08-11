"""Project-wide, power-cut-recoverable physical GPU worker leases.

Every formal experiment track uses the same registry below the verified project
root.  A lease names a physical NVIDIA UUID, not a process-local CUDA ordinal.
The registry therefore enforces both one worker per physical device and a
project-wide maximum of three concurrent GPU workers.

This module is intentionally standalone.  It does not import any experiment
runner, so procedure, common-recipe, and GeoAdapt tracks can share it without a
circular dependency.
"""

from __future__ import annotations

import copy
import ctypes
import errno
import fcntl
import hashlib
import json
import os
import re
import socket
import stat
import sys
import threading
import time
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

LEASE_SCHEMA = "ieee-mi-project-gpu-lease-v1"
LEASE_RECEIPT_SCHEMA = "ieee-mi-project-gpu-lease-receipt-v1"
REGISTRY_FENCE_SCHEMA = "ieee-mi-project-gpu-registry-fence-v1"
REGISTRY_DIRECTORY = ".ieee-mi-project-gpu-leases"
FORENSIC_DIRECTORY = ".ieee-mi-project-gpu-lease-forensics"
REGISTRY_FENCE_FILENAME = ".registry.fence"
MAX_ACTIVE_LEASES = 3
GPU_UUID_RE = re.compile(
    r"^GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
NONCE_RE = re.compile(r"^[0-9a-f]{32}$")
BOOT_ID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12}$"
)
TRACK_SCOPE_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_NONLINUX_BOOT_ID = str(uuid.uuid4())
_NONLINUX_START_TICKS = max(1, time.monotonic_ns())
_REGISTRY_THREAD_LOCKS: dict[tuple[int, str], threading.RLock] = {}


class ProjectGPULeaseError(RuntimeError):
    """Raised when the shared registry cannot preserve its safety contract."""


class GPUWorkerUnavailable(ProjectGPULeaseError):
    """Raised when a requested UUID or the global three-worker cap is busy."""


class RegistryFenceLost(ProjectGPULeaseError):
    """Raised when the locked persistent registry fence is replaced."""


@dataclass(frozen=True)
class GPULease:
    """An inode- and nonce-bound receipt for one physical GPU lease."""

    project_root: Path
    registry_root: Path
    path: Path
    nonce: str
    owner: Mapping[str, Any]
    value: Mapping[str, Any]
    st_dev: int
    st_ino: int
    fence_st_dev: int
    fence_st_ino: int


@dataclass(frozen=True)
class _LeaseSnapshot:
    path: Path
    descriptor: int
    value: Mapping[str, Any]
    st_dev: int
    st_ino: int


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


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


def _project_root_digest(project_root: Path) -> str:
    return _sha256_bytes(os.fsencode(_absolute(project_root)))


def _run_root_digest(run_root: Path) -> str:
    return _sha256_bytes(os.fsencode(_absolute(run_root)))


def _assert_no_symlink_tree(
    path: Path,
    *,
    include_leaf: bool = True,
    allow_missing_tail: bool = False,
) -> Path:
    absolute = _absolute(path)
    current = Path(absolute.anchor)
    parts = absolute.parts[1:] if include_leaf else absolute.parts[1:-1]
    missing = False
    for component in parts:
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
        if stat.S_ISLNK(observed.st_mode):
            raise ProjectGPULeaseError(
                f"GPU lease path contains a symlink component: {current}"
            )
        if current != absolute and not stat.S_ISDIR(observed.st_mode):
            raise ProjectGPULeaseError(
                f"GPU lease path ancestor is not a directory: {current}"
            )
    return absolute


def _require_real_directory(path: Path) -> os.stat_result:
    absolute = _assert_no_symlink_tree(path)
    descriptor = _open_directory_absolute(absolute)
    try:
        observed = os.fstat(descriptor)
        if not stat.S_ISDIR(observed.st_mode):
            raise ProjectGPULeaseError(f"{absolute} is not a real directory")
        return observed
    finally:
        os.close(descriptor)


def _secure_mkdir_beneath(project_root: Path, relative: Path) -> Path:
    project = _assert_no_symlink_tree(project_root)
    _require_real_directory(project)
    if relative.is_absolute() or any(
        component in {"", ".", ".."} for component in relative.parts
    ):
        raise ProjectGPULeaseError("invalid project-relative lease directory")
    descriptor = _open_directory_absolute(project)
    current = project
    try:
        for component in relative.parts:
            try:
                child = os.open(
                    component,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_NONBLOCK", 0)
                    | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=descriptor,
                )
            except FileNotFoundError:
                try:
                    os.mkdir(component, 0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
                child = os.open(
                    component,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_NONBLOCK", 0)
                    | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=descriptor,
                )
            except OSError as error:
                raise ProjectGPULeaseError(
                    "lease directory stopped being a no-follow real tree"
                ) from error
            observed = os.fstat(child)
            if not stat.S_ISDIR(observed.st_mode):
                os.close(child)
                raise ProjectGPULeaseError(
                    "lease directory component is not a directory"
                )
            os.close(descriptor)
            descriptor = child
            current /= component
    finally:
        os.close(descriptor)
    _assert_no_symlink_tree(current)
    return current


def _open_directory_absolute(path: Path) -> int:
    absolute = _absolute(path)
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
                raise ProjectGPULeaseError(
                    f"path component is not a directory: {absolute}"
                )
            os.close(descriptor)
            descriptor = child
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _anchored_lstat(path: Path) -> os.stat_result:
    absolute = _absolute(path)
    parent = _open_directory_absolute(absolute.parent)
    try:
        return os.stat(
            absolute.name,
            dir_fd=parent,
            follow_symlinks=False,
        )
    finally:
        os.close(parent)


def _forensic_directory(project_root: Path, category: str) -> Path:
    if category not in {"partials", "released", "stale"}:
        raise ProjectGPULeaseError("unknown GPU lease forensic category")
    return _secure_mkdir_beneath(
        project_root, Path(FORENSIC_DIRECTORY) / category
    )


def _read_regular_file(path: Path) -> bytes:
    absolute = _assert_no_symlink_tree(path, include_leaf=False)
    parent_descriptor = _open_directory_absolute(absolute.parent)
    try:
        path_stat = os.stat(
            absolute.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if not stat.S_ISREG(path_stat.st_mode) or path_stat.st_nlink != 1:
            raise ProjectGPULeaseError(
                f"{absolute} is not a unique regular file"
            )
        descriptor = os.open(
            absolute.name,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_descriptor,
        )
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or (before.st_dev, before.st_ino)
                != (path_stat.st_dev, path_stat.st_ino)
            ):
                raise ProjectGPULeaseError(f"{absolute} changed while opening")
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            after = os.fstat(descriptor)
            fields = (
                "st_dev",
                "st_ino",
                "st_nlink",
                "st_size",
                "st_mtime_ns",
                "st_ctime_ns",
            )
            if any(getattr(before, key) != getattr(after, key) for key in fields):
                raise ProjectGPULeaseError(
                    f"{absolute} changed during verified read"
                )
            final_path = os.stat(
                absolute.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if (
                (final_path.st_dev, final_path.st_ino)
                != (before.st_dev, before.st_ino)
                or final_path.st_nlink != 1
            ):
                raise ProjectGPULeaseError(
                    f"{absolute} was replaced during read"
                )
            payload = b"".join(chunks)
            if len(payload) != before.st_size:
                raise ProjectGPULeaseError(
                    f"{absolute} produced an incomplete read"
                )
            return payload
        finally:
            os.close(descriptor)
    finally:
        os.close(parent_descriptor)


def _unique_pairs(
    pairs: list[tuple[str, Any]], *, source: str
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProjectGPULeaseError(
                f"{source} has duplicate JSON key {key!r}"
            )
        result[key] = value
    return result


def _strict_json_bytes(payload: bytes, *, source: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=lambda pairs: _unique_pairs(pairs, source=source),
            parse_constant=lambda raw: (_ for _ in ()).throw(
                ProjectGPULeaseError(
                    f"{source} has non-finite JSON constant {raw!r}"
                )
            ),
        )
    except UnicodeDecodeError as error:
        raise ProjectGPULeaseError(f"{source} is not UTF-8 JSON") from error
    except json.JSONDecodeError as error:
        raise ProjectGPULeaseError(f"{source} is not valid JSON") from error
    if not isinstance(value, dict):
        raise ProjectGPULeaseError(f"{source} is not a JSON object")
    if payload != _canonical_bytes(value) + b"\n":
        raise ProjectGPULeaseError(f"{source} is not canonical JSON")
    return value


def _strict_json_file(path: Path) -> dict[str, Any]:
    return _strict_json_bytes(_read_regular_file(path), source=str(path))


def _strict_json_descriptor(descriptor: int, *, source: str) -> dict[str, Any]:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise ProjectGPULeaseError(f"{source} is not a unique regular file")
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    after = os.fstat(descriptor)
    fields = (
        "st_dev",
        "st_ino",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(getattr(before, key) != getattr(after, key) for key in fields):
        raise ProjectGPULeaseError(f"{source} changed during descriptor read")
    return _strict_json_bytes(b"".join(chunks), source=source)


def _fsync_directory(path: Path) -> None:
    _require_real_directory(path)
    descriptor = _open_directory_absolute(path)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
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
            payload = _canonical_bytes(value) + b"\n"
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


def _stage_json(
    project_root: Path,
    *,
    basename: str,
    value: Mapping[str, Any],
) -> Path:
    """Write a non-authoritative receipt before no-replace publication."""

    if (
        not basename
        or "/" in basename
        or basename in {".", ".."}
        or not re.fullmatch(r"[A-Za-z0-9_.-]+", basename)
    ):
        raise ProjectGPULeaseError("invalid staged lease basename")
    root = _forensic_directory(project_root, "partials")
    path = root / f"{basename}.{uuid.uuid4().hex}.partial"
    _write_json_exclusive(path, value)
    if _strict_json_file(path) != dict(value):
        raise ProjectGPULeaseError("staged GPU receipt changed after creation")
    return path


def _atomic_rename_noreplace(source: Path, destination: Path) -> None:
    source = _assert_no_symlink_tree(source, include_leaf=False)
    destination = _assert_no_symlink_tree(destination, include_leaf=False)
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
            raise ProjectGPULeaseError(
                "anchored atomic no-replace lease recovery is unavailable"
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


def _boot_id() -> str | None:
    if not sys.platform.startswith("linux"):
        return _NONLINUX_BOOT_ID
    path = Path("/proc/sys/kernel/random/boot_id")
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
            if (
                not stat.S_ISREG(observed_stat.st_mode)
                or observed_stat.st_nlink != 1
            ):
                return None
            observed = os.read(descriptor, 128).decode("ascii").strip().lower()
        finally:
            os.close(descriptor)
    except (OSError, UnicodeDecodeError):
        return None
    return observed if BOOT_ID_RE.fullmatch(observed) else None


def _process_start_ticks(pid: int) -> int | None:
    if not sys.platform.startswith("linux"):
        return _NONLINUX_START_TICKS if pid == os.getpid() else None
    path = Path(f"/proc/{pid}/stat")
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
            if (
                not stat.S_ISREG(observed_stat.st_mode)
                or observed_stat.st_nlink != 1
            ):
                return None
            text = os.read(descriptor, 64 * 1024).decode("ascii")
        finally:
            os.close(descriptor)
        value = int(text.rsplit(")", 1)[1].split()[19])
    except (OSError, UnicodeDecodeError, ValueError, IndexError):
        return None
    return value if value > 0 else None


def owner_identity() -> dict[str, Any]:
    """Return an exact, durable identity for the current process."""

    value = {
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "boot_id": _boot_id(),
        "start_ticks": _process_start_ticks(os.getpid()),
    }
    if not _owner_is_valid(value):
        raise ProjectGPULeaseError(
            "cannot establish a non-null boot/process-start identity"
        )
    return value


def _owner_is_valid(value: Any) -> bool:
    return bool(
        isinstance(value, Mapping)
        and set(value) == {"host", "pid", "boot_id", "start_ticks"}
        and isinstance(value["host"], str)
        and value["host"]
        and type(value["pid"]) is int
        and value["pid"] > 0
        and isinstance(value["boot_id"], str)
        and BOOT_ID_RE.fullmatch(value["boot_id"])
        and type(value["start_ticks"]) is int
        and value["start_ticks"] > 0
    )


def _owner_is_live(value: Mapping[str, Any]) -> bool:
    """Fail closed unless a valid local identity can be proven dead."""

    if not _owner_is_valid(value):
        raise ProjectGPULeaseError("malformed GPU lease owner")
    if value["host"] != socket.gethostname():
        return True
    current_boot = _boot_id()
    if current_boot is None:
        return True
    if value["boot_id"] != current_boot:
        return False
    try:
        os.kill(int(value["pid"]), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    observed_start = _process_start_ticks(int(value["pid"]))
    if observed_start is None:
        return True
    return observed_start == value["start_ticks"]


def _canonical_gpu_uuid(value: str) -> str:
    if not isinstance(value, str) or not GPU_UUID_RE.fullmatch(value):
        raise ProjectGPULeaseError("invalid physical NVIDIA GPU UUID")
    return "GPU-" + value[4:].lower()


def canonical_gpu_uuid(value: str) -> str:
    """Return the one registry spelling for a physical NVIDIA UUID."""

    return _canonical_gpu_uuid(value)


def _registry_value(project_root: Path) -> dict[str, str]:
    digest = _project_root_digest(project_root)
    return {
        "schema": REGISTRY_FENCE_SCHEMA,
        "project_root_sha256": digest,
        "token": _sha256_bytes(
            b"ieee-mi-project-gpu-registry-fence-v1\0"
            + digest.encode("ascii")
        ),
    }


def _registry_root(project_root: Path) -> Path:
    project = _assert_no_symlink_tree(project_root)
    _require_real_directory(project)
    return _secure_mkdir_beneath(project, Path(REGISTRY_DIRECTORY))


def _assert_registry_descriptor(
    project_root: Path,
    path: Path,
    descriptor: int,
) -> None:
    descriptor_stat = os.fstat(descriptor)
    try:
        path_stat = _anchored_lstat(path)
    except FileNotFoundError as error:
        raise RegistryFenceLost("GPU registry fence disappeared") from error
    if (
        not stat.S_ISREG(descriptor_stat.st_mode)
        or descriptor_stat.st_nlink != 1
        or (descriptor_stat.st_dev, descriptor_stat.st_ino)
        != (path_stat.st_dev, path_stat.st_ino)
    ):
        raise RegistryFenceLost("GPU registry fence inode was replaced")
    try:
        observed = _strict_json_descriptor(descriptor, source=str(path))
    except ProjectGPULeaseError as error:
        raise RegistryFenceLost("GPU registry fence token is invalid") from error
    if observed != _registry_value(project_root):
        raise RegistryFenceLost("GPU registry fence token changed")


@contextmanager
def _registry_file_lock(project_root: Path) -> Iterator[tuple[Path, int]]:
    project = _assert_no_symlink_tree(project_root)
    root = _registry_root(project)
    path = root / REGISTRY_FENCE_FILENAME
    if not path.exists() and not path.is_symlink():
        staged = _stage_json(
            project,
            basename="registry-fence",
            value=_registry_value(project),
        )
        try:
            _atomic_rename_noreplace(staged, path)
            _fsync_directory(root)
        except FileExistsError:
            # Another process atomically installed the authoritative fence.
            # Our complete, non-authoritative receipt remains in the external
            # forensic partial tree and cannot affect registry interpretation.
            pass
    parent = _open_directory_absolute(path.parent)
    try:
        descriptor = os.open(
            path.name,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent,
        )
    finally:
        os.close(parent)
    fence_error: Exception | None = None
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        _assert_registry_descriptor(project, path, descriptor)
        yield root, descriptor
    finally:
        try:
            _assert_registry_descriptor(project, path, descriptor)
        except Exception as error:
            fence_error = error
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
        if fence_error is not None:
            raise fence_error


@contextmanager
def _registry_lock(project_root: Path) -> Iterator[tuple[Path, int]]:
    """Serialize both threads and cooperating processes for one registry."""

    project = _assert_no_symlink_tree(project_root)
    key = (os.getpid(), os.fspath(project))
    thread_lock = _REGISTRY_THREAD_LOCKS.setdefault(key, threading.RLock())
    with thread_lock:
        with _registry_file_lock(project) as locked:
            yield locked


def _validate_lease(
    value: Any,
    *,
    project_root: Path,
    expected_gpu_uuid: str | None = None,
) -> None:
    expected_keys = {
        "schema",
        "created_at",
        "project_root_sha256",
        "track_scope",
        "run_root_sha256",
        "plan_sha256",
        "gpu_uuid",
        "nonce",
        "owner",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise ProjectGPULeaseError("GPU lease has an unknown schema")
    try:
        created = datetime.fromisoformat(str(value["created_at"]))
    except (TypeError, ValueError) as error:
        raise ProjectGPULeaseError("GPU lease timestamp is invalid") from error
    canonical_uuid = _canonical_gpu_uuid(str(value["gpu_uuid"]))
    if (
        value["schema"] != LEASE_SCHEMA
        or created.tzinfo is None
        or value["project_root_sha256"] != _project_root_digest(project_root)
        or not isinstance(value["track_scope"], str)
        or not TRACK_SCOPE_RE.fullmatch(value["track_scope"])
        or not isinstance(value["run_root_sha256"], str)
        or not SHA256_RE.fullmatch(value["run_root_sha256"])
        or not isinstance(value["plan_sha256"], str)
        or not SHA256_RE.fullmatch(value["plan_sha256"])
        or value["gpu_uuid"] != canonical_uuid
        or (
            expected_gpu_uuid is not None
            and canonical_uuid != _canonical_gpu_uuid(expected_gpu_uuid)
        )
        or not isinstance(value["nonce"], str)
        or not NONCE_RE.fullmatch(value["nonce"])
        or not _owner_is_valid(value["owner"])
    ):
        raise ProjectGPULeaseError("GPU lease identity is invalid")


def _active_lease_paths(root: Path) -> list[Path]:
    result: list[Path] = []
    for path in sorted(root.iterdir()):
        observed = _anchored_lstat(path)
        if path.name == REGISTRY_FENCE_FILENAME:
            if (
                not stat.S_ISREG(observed.st_mode)
                or observed.st_nlink != 1
            ):
                raise ProjectGPULeaseError("registry fence is not a unique file")
            continue
        if (
            stat.S_ISLNK(observed.st_mode)
            or not stat.S_ISREG(observed.st_mode)
            or observed.st_nlink != 1
            or not re.fullmatch(r"GPU-[0-9a-f-]{36}\.json", path.name)
        ):
            raise ProjectGPULeaseError(
                f"unknown or unsafe active GPU lease node: {path.name}"
            )
        result.append(path)
    return result


def _open_lease_snapshot(
    path: Path,
    *,
    project_root: Path,
    expected_gpu_uuid: str,
) -> _LeaseSnapshot:
    parent = _open_directory_absolute(path.parent)
    try:
        descriptor = os.open(
            path.name,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent,
        )
        path_stat = os.stat(
            path.name,
            dir_fd=parent,
            follow_symlinks=False,
        )
    except Exception:
        os.close(parent)
        raise
    os.close(parent)
    try:
        descriptor_stat = os.fstat(descriptor)
        if (
            not stat.S_ISREG(descriptor_stat.st_mode)
            or descriptor_stat.st_nlink != 1
            or (descriptor_stat.st_dev, descriptor_stat.st_ino)
            != (path_stat.st_dev, path_stat.st_ino)
        ):
            raise ProjectGPULeaseError("active GPU lease inode changed")
        value = _strict_json_descriptor(descriptor, source=str(path))
        _validate_lease(
            value,
            project_root=project_root,
            expected_gpu_uuid=expected_gpu_uuid,
        )
        return _LeaseSnapshot(
            path=path,
            descriptor=descriptor,
            value=dict(value),
            st_dev=int(descriptor_stat.st_dev),
            st_ino=int(descriptor_stat.st_ino),
        )
    except Exception:
        os.close(descriptor)
        raise


def _assert_snapshot_path(snapshot: _LeaseSnapshot, path: Path) -> None:
    descriptor_stat = os.fstat(snapshot.descriptor)
    try:
        path_stat = _anchored_lstat(path)
    except FileNotFoundError as error:
        raise ProjectGPULeaseError("GPU lease snapshot path disappeared") from error
    if (
        not stat.S_ISREG(descriptor_stat.st_mode)
        or descriptor_stat.st_nlink != 1
        or (descriptor_stat.st_dev, descriptor_stat.st_ino)
        != (snapshot.st_dev, snapshot.st_ino)
        or (path_stat.st_dev, path_stat.st_ino)
        != (snapshot.st_dev, snapshot.st_ino)
        or _strict_json_descriptor(
            snapshot.descriptor, source=str(snapshot.path)
        )
        != dict(snapshot.value)
    ):
        raise ProjectGPULeaseError("GPU lease snapshot ownership was lost")


def _assert_gpu_lease_locked(lease: GPULease, root: Path) -> None:
    """Revalidate one acquired lease while the registry fence is locked."""

    if not isinstance(lease, GPULease):
        raise ProjectGPULeaseError("GPU lease handle has an unknown type")
    if not isinstance(lease.value, Mapping):
        raise ProjectGPULeaseError("GPU lease handle value is invalid")
    project = _assert_no_symlink_tree(Path(lease.project_root))
    expected_root = project / REGISTRY_DIRECTORY
    if (
        _absolute(Path(lease.registry_root)) != expected_root
        or _absolute(root) != expected_root
    ):
        raise ProjectGPULeaseError("GPU lease registry identity changed")
    _validate_lease(
        lease.value,
        project_root=project,
        expected_gpu_uuid=str(lease.value.get("gpu_uuid", "")),
    )
    expected_path = expected_root / f"{lease.value['gpu_uuid']}.json"
    if (
        _absolute(Path(lease.path)) != expected_path
        or lease.nonce != lease.value["nonce"]
        or dict(lease.owner) != lease.value["owner"]
        or type(lease.st_dev) is not int
        or lease.st_dev < 0
        or type(lease.st_ino) is not int
        or lease.st_ino <= 0
        or type(lease.fence_st_dev) is not int
        or lease.fence_st_dev < 0
        or type(lease.fence_st_ino) is not int
        or lease.fence_st_ino <= 0
    ):
        raise ProjectGPULeaseError("GPU lease handle identity is invalid")
    fence_path = expected_root / REGISTRY_FENCE_FILENAME
    fence_stat = _anchored_lstat(fence_path)
    if (
        not stat.S_ISREG(fence_stat.st_mode)
        or fence_stat.st_nlink != 1
        or (fence_stat.st_dev, fence_stat.st_ino)
        != (lease.fence_st_dev, lease.fence_st_ino)
    ):
        raise RegistryFenceLost("GPU registry fence differs from acquired lease")
    snapshot = _open_lease_snapshot(
        expected_path,
        project_root=project,
        expected_gpu_uuid=str(lease.value["gpu_uuid"]),
    )
    try:
        if (
            (snapshot.st_dev, snapshot.st_ino) != (lease.st_dev, lease.st_ino)
            or snapshot.value != dict(lease.value)
            or snapshot.value["nonce"] != lease.nonce
            or snapshot.value["owner"] != dict(lease.owner)
        ):
            raise ProjectGPULeaseError("GPU lease ownership was lost")
        if not _owner_is_live(snapshot.value["owner"]):
            raise ProjectGPULeaseError("GPU lease owner is no longer live")
        if owner_identity() != dict(lease.owner):
            raise ProjectGPULeaseError(
                "GPU lease is being asserted by a different process"
            )
        _assert_snapshot_path(snapshot, expected_path)
    finally:
        os.close(snapshot.descriptor)


def assert_gpu_lease(lease: GPULease) -> None:
    """Fail closed unless the exact acquired lease is still authoritative.

    The registry fence, lease path, inode, nonce, owner, and full immutable
    value are revalidated under the project-wide registry lock.  Formal
    runners call this immediately before every job claim and commit.
    """

    if not isinstance(lease, GPULease):
        raise ProjectGPULeaseError("GPU lease handle has an unknown type")
    project = _assert_no_symlink_tree(Path(lease.project_root))
    _require_real_directory(project)
    with _registry_lock(project) as (root, _registry_descriptor):
        _assert_gpu_lease_locked(lease, root)


@contextmanager
def guard_gpu_lease(lease: GPULease) -> Iterator[dict[str, Any]]:
    """Hold the authoritative registry lock across one publication boundary.

    The exact lease and persistent fence are checked before the caller may
    publish and again after the caller returns, while the same registry lock
    remains held.  The yielded receipt is a detached canonical value suitable
    for binding into the caller's immutable artifact.
    """

    if not isinstance(lease, GPULease):
        raise ProjectGPULeaseError("GPU lease handle has an unknown type")
    project = _assert_no_symlink_tree(Path(lease.project_root))
    _require_real_directory(project)
    with _registry_lock(project) as (root, _registry_descriptor):
        _assert_gpu_lease_locked(lease, root)
        receipt = _gpu_lease_receipt_unchecked(lease)
        try:
            yield copy.deepcopy(receipt)
        finally:
            _assert_gpu_lease_locked(lease, root)


def _gpu_lease_receipt_unchecked(lease: GPULease) -> dict[str, Any]:
    value = copy.deepcopy(dict(lease.value))
    return {
        "schema": LEASE_RECEIPT_SCHEMA,
        "lease": value,
        "lease_sha256": _sha256_bytes(_canonical_bytes(value)),
        "lease_st_dev": int(lease.st_dev),
        "lease_st_ino": int(lease.st_ino),
        "registry_fence_st_dev": int(lease.fence_st_dev),
        "registry_fence_st_ino": int(lease.fence_st_ino),
    }


def gpu_lease_receipt(lease: GPULease) -> dict[str, Any]:
    """Return a canonical artifact receipt for a currently active lease."""

    assert_gpu_lease(lease)
    return _gpu_lease_receipt_unchecked(lease)


def validate_gpu_lease_receipt(
    receipt: Any,
    *,
    project_root: Path,
    run_root: Path,
    plan_sha256: str,
    gpu_uuid: str,
    track_scope: str,
) -> None:
    """Validate a persisted lease receipt against one exact run identity."""

    expected_keys = {
        "schema",
        "lease",
        "lease_sha256",
        "lease_st_dev",
        "lease_st_ino",
        "registry_fence_st_dev",
        "registry_fence_st_ino",
    }
    if not isinstance(receipt, Mapping) or set(receipt) != expected_keys:
        raise ProjectGPULeaseError("GPU lease receipt has an unknown schema")
    project = _assert_no_symlink_tree(project_root)
    run = _assert_no_symlink_tree(run_root)
    _require_real_directory(project)
    _require_real_directory(run)
    _validate_lease(
        receipt["lease"],
        project_root=project,
        expected_gpu_uuid=gpu_uuid,
    )
    lease_value = receipt["lease"]
    if (
        receipt["schema"] != LEASE_RECEIPT_SCHEMA
        or lease_value["run_root_sha256"] != _run_root_digest(run)
        or lease_value["plan_sha256"] != plan_sha256
        or lease_value["track_scope"] != track_scope
        or lease_value["gpu_uuid"] != _canonical_gpu_uuid(gpu_uuid)
        or receipt["lease_sha256"]
        != _sha256_bytes(_canonical_bytes(lease_value))
        or type(receipt["lease_st_dev"]) is not int
        or receipt["lease_st_dev"] < 0
        or type(receipt["lease_st_ino"]) is not int
        or receipt["lease_st_ino"] <= 0
        or type(receipt["registry_fence_st_dev"]) is not int
        or receipt["registry_fence_st_dev"] < 0
        or type(receipt["registry_fence_st_ino"]) is not int
        or receipt["registry_fence_st_ino"] <= 0
    ):
        raise ProjectGPULeaseError("GPU lease receipt identity is invalid")


def _archive_snapshot(
    snapshot: _LeaseSnapshot,
    *,
    project_root: Path,
    category: str,
    registry_descriptor: int,
) -> Path:
    destination_root = _forensic_directory(project_root, category)
    destination = (
        destination_root
        / f"{snapshot.value['gpu_uuid']}.{snapshot.value['nonce']}.json"
    )
    _assert_snapshot_path(snapshot, snapshot.path)
    _assert_registry_descriptor(
        project_root,
        snapshot.path.parent / REGISTRY_FENCE_FILENAME,
        registry_descriptor,
    )
    _atomic_rename_noreplace(snapshot.path, destination)
    _assert_registry_descriptor(
        project_root,
        snapshot.path.parent / REGISTRY_FENCE_FILENAME,
        registry_descriptor,
    )
    _assert_snapshot_path(snapshot, destination)
    _fsync_directory(destination_root)
    _fsync_directory(snapshot.path.parent)
    return destination


def acquire_gpu_lease(
    *,
    project_root: Path,
    run_root: Path,
    plan_sha256: str,
    gpu_uuid: str,
    track_scope: str,
) -> GPULease:
    """Atomically acquire one physical GPU under the global three-worker cap."""

    project = _assert_no_symlink_tree(project_root)
    _require_real_directory(project)
    run = _assert_no_symlink_tree(run_root)
    _require_real_directory(run)
    canonical_uuid = _canonical_gpu_uuid(gpu_uuid)
    if not isinstance(plan_sha256, str) or not SHA256_RE.fullmatch(plan_sha256):
        raise ProjectGPULeaseError("invalid plan SHA-256 for GPU lease")
    if not isinstance(track_scope, str) or not TRACK_SCOPE_RE.fullmatch(track_scope):
        raise ProjectGPULeaseError("invalid track scope for GPU lease")
    with _registry_lock(project) as (root, registry_descriptor):
        live_snapshots: list[_LeaseSnapshot] = []
        try:
            for path in _active_lease_paths(root):
                expected_uuid = path.name.removesuffix(".json")
                # Malformed content fails before liveness is considered and is
                # never stolen automatically.
                snapshot = _open_lease_snapshot(
                    path,
                    project_root=project,
                    expected_gpu_uuid=expected_uuid,
                )
                if _owner_is_live(snapshot.value["owner"]):
                    live_snapshots.append(snapshot)
                    if expected_uuid == canonical_uuid:
                        raise GPUWorkerUnavailable(
                            f"physical GPU {canonical_uuid} already has a lease"
                        )
                else:
                    try:
                        _archive_snapshot(
                            snapshot,
                            project_root=project,
                            category="stale",
                            registry_descriptor=registry_descriptor,
                        )
                    finally:
                        os.close(snapshot.descriptor)
            if len(live_snapshots) >= MAX_ACTIVE_LEASES:
                raise GPUWorkerUnavailable(
                    f"project GPU-worker cap is {MAX_ACTIVE_LEASES}"
                )
            for snapshot in live_snapshots:
                _assert_snapshot_path(snapshot, snapshot.path)
            owner = owner_identity()
            nonce = uuid.uuid4().hex
            value = {
                "schema": LEASE_SCHEMA,
                "created_at": datetime.now(UTC).isoformat(),
                "project_root_sha256": _project_root_digest(project),
                "track_scope": track_scope,
                "run_root_sha256": _run_root_digest(run),
                "plan_sha256": plan_sha256,
                "gpu_uuid": canonical_uuid,
                "nonce": nonce,
                "owner": owner,
            }
            _validate_lease(
                value,
                project_root=project,
                expected_gpu_uuid=canonical_uuid,
            )
            staged = _stage_json(
                project,
                basename=f"{canonical_uuid}.{nonce}",
                value=value,
            )
            for snapshot in live_snapshots:
                _assert_snapshot_path(snapshot, snapshot.path)
            path = root / f"{canonical_uuid}.json"
            fence_path = root / REGISTRY_FENCE_FILENAME
            try:
                _assert_registry_descriptor(
                    project, fence_path, registry_descriptor
                )
                _atomic_rename_noreplace(staged, path)
                _assert_registry_descriptor(
                    project, fence_path, registry_descriptor
                )
                _fsync_directory(root)
                _assert_registry_descriptor(
                    project, fence_path, registry_descriptor
                )
            except FileExistsError as error:
                raise GPUWorkerUnavailable(
                    f"physical GPU {canonical_uuid} lease race was lost"
                ) from error
            created = _open_lease_snapshot(
                path,
                project_root=project,
                expected_gpu_uuid=canonical_uuid,
            )
            try:
                if created.value != value:
                    raise ProjectGPULeaseError(
                        "GPU lease changed immediately after publication"
                    )
                fence_stat = _anchored_lstat(
                    root / REGISTRY_FENCE_FILENAME
                )
                return GPULease(
                    project_root=project,
                    registry_root=root,
                    path=path,
                    nonce=nonce,
                    owner=dict(owner),
                    value=dict(value),
                    st_dev=created.st_dev,
                    st_ino=created.st_ino,
                    fence_st_dev=int(fence_stat.st_dev),
                    fence_st_ino=int(fence_stat.st_ino),
                )
            finally:
                os.close(created.descriptor)
        finally:
            for snapshot in live_snapshots:
                os.close(snapshot.descriptor)


def release_gpu_lease(lease: GPULease) -> None:
    """Archive only the exact inode, nonce, owner, and receipt acquired."""

    with _registry_lock(lease.project_root) as (root, registry_descriptor):
        _assert_gpu_lease_locked(lease, root)
        snapshot = _open_lease_snapshot(
            lease.path,
            project_root=lease.project_root,
            expected_gpu_uuid=str(lease.value["gpu_uuid"]),
        )
        try:
            if (
                (snapshot.st_dev, snapshot.st_ino)
                != (lease.st_dev, lease.st_ino)
                or snapshot.value != dict(lease.value)
                or snapshot.value["nonce"] != lease.nonce
                or snapshot.value["owner"] != dict(lease.owner)
            ):
                raise ProjectGPULeaseError("GPU lease ownership was lost")
            _archive_snapshot(
                snapshot,
                project_root=lease.project_root,
                category="released",
                registry_descriptor=registry_descriptor,
            )
        finally:
            os.close(snapshot.descriptor)
