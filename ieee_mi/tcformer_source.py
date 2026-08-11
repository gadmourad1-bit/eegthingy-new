"""Pinned, compact provenance contract for the official TCFormer runtime.

The upstream repository contains hundreds of megabytes of non-runtime Git
objects.  The clean benchmark vendors only the five files it actually needs
and verifies them against hashes taken from the pinned upstream commit.  The
vendored files remain unmodified and retain the upstream MIT license.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any, Mapping


TCFORMER_REPOSITORY = "https://github.com/Altaheri/TCFormer"
TCFORMER_COMMIT = "74c89b7ab8c64e4eb51e0f748dd87dd4c94e68c5"
TCFORMER_LICENSE = "MIT"
TCFORMER_MANIFEST_SCHEMA = "ieee-mi-vendored-tcformer-source-v1"
TCFORMER_MANIFEST_FILENAME = "SOURCE.json"
TCFORMER_PINNED_FILES: Mapping[str, Mapping[str, Any]] = {
    "LICENSE": {
        "size_bytes": 1071,
        "sha256": "daea0e9f8596568cf5aae39b69f45ffc9f1b00d600d5315c09e90668cd99ee97",
    },
    "models/channel_group_attention.py": {
        "size_bytes": 5418,
        "sha256": "5623b68c9e1524faad00a8450a8f2fb909d3f39c7d4d20bef7ae769b8de8aabb",
    },
    "models/modules.py": {
        "size_bytes": 1866,
        "sha256": "cb24a800c47cf3e417864b9928ba8da61017b1e928ae53706879c6af8a910c14",
    },
    "models/tcformer.py": {
        "size_bytes": 22281,
        "sha256": "755cba76838ad325ffd35a75c2e555235f1541a4bb409634105f10f3537acd66",
    },
    "utils/weight_initialization.py": {
        "size_bytes": 377,
        "sha256": "e92148c28b1ae39dc7c7a0d52704594545fb09cd9eaa6b1b819c4a3836ad01e3",
    },
}


class TCFormerSourceError(RuntimeError):
    """The vendored TCFormer source does not match the pinned contract."""


def canonical_manifest() -> dict[str, Any]:
    """Return the exact SOURCE.json payload expected in a clean release."""

    return {
        "schema": TCFORMER_MANIFEST_SCHEMA,
        "name": "TCFormer",
        "upstream_repository": TCFORMER_REPOSITORY,
        "upstream_commit": TCFORMER_COMMIT,
        "license": TCFORMER_LICENSE,
        "files": {
            name: dict(contract)
            for name, contract in sorted(TCFORMER_PINNED_FILES.items())
        },
    }


def canonical_manifest_bytes() -> bytes:
    return (
        json.dumps(
            canonical_manifest(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        + b"\n"
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TCFormerSourceError(
                f"TCFormer SOURCE.json contains duplicate key {key!r}"
            )
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise TCFormerSourceError(
        f"TCFormer SOURCE.json contains invalid numeric constant {value!r}"
    )


def _require_real_ancestors(path: Path, *, include_leaf: bool) -> None:
    absolute = Path(os.path.abspath(path))
    limit = len(absolute.parts) if include_leaf else len(absolute.parts) - 1
    current = Path(absolute.anchor)
    for component in absolute.parts[1:limit]:
        current /= component
        try:
            observed = os.lstat(current)
        except FileNotFoundError as error:
            raise TCFormerSourceError(f"TCFormer source path is absent: {current}") from error
        if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
            raise TCFormerSourceError(
                f"TCFormer source ancestor is not a real directory: {current}"
            )


def _read_unique_regular_bytes(path: Path, *, root: Path) -> bytes:
    absolute = Path(os.path.abspath(path))
    source_root = Path(os.path.abspath(root))
    try:
        relative = absolute.relative_to(source_root)
    except ValueError as error:
        raise TCFormerSourceError(
            f"TCFormer source escapes the vendored root: {absolute}"
        ) from error
    if relative.name in {"", ".", ".."}:
        raise TCFormerSourceError("TCFormer source has no safe file leaf")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    parent_descriptor = os.open(source_root.anchor, directory_flags)
    try:
        for component in source_root.parts[1:]:
            try:
                child = os.open(
                    component,
                    directory_flags,
                    dir_fd=parent_descriptor,
                )
            except OSError as error:
                raise TCFormerSourceError(
                    f"TCFormer source ancestor is not a real directory: {source_root}"
                ) from error
            os.close(parent_descriptor)
            parent_descriptor = child
        for component in relative.parts[:-1]:
            if component in {"", ".", ".."}:
                raise TCFormerSourceError("TCFormer source has an unsafe relative path")
            try:
                child = os.open(
                    component,
                    directory_flags,
                    dir_fd=parent_descriptor,
                )
            except OSError as error:
                raise TCFormerSourceError(
                    f"TCFormer source ancestor is not a real directory: {absolute}"
                ) from error
            os.close(parent_descriptor)
            parent_descriptor = child
        pathname_stat = os.stat(
            relative.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        descriptor = os.open(relative.name, flags, dir_fd=parent_descriptor)
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
            stat.S_ISLNK(pathname_stat.st_mode)
            or
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or any(
                getattr(before, name) != getattr(pathname_stat, name)
                for name in stable_fields
            )
        ):
            raise TCFormerSourceError(
                "TCFormer source is not a stable single-link regular file: "
                f"{absolute}"
            )
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
                relative.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if any(
                getattr(before, name) != getattr(observed, name)
                for observed in (after, final_path_stat)
                for name in stable_fields
            ):
                raise TCFormerSourceError(
                    f"TCFormer source changed while it was read: {absolute}"
                )
            payload = b"".join(chunks)
            if len(payload) != before.st_size:
                raise TCFormerSourceError(
                    f"TCFormer source read was incomplete: {absolute}"
                )
            return payload
        finally:
            os.close(descriptor)
    finally:
        os.close(parent_descriptor)


def default_tcformer_root() -> Path:
    configured = os.environ.get("IEEE_MI_TCFORMER_ROOT")
    value = (
        Path(configured)
        if configured
        else Path.cwd() / "third_party" / "TCFormer"
    )
    return Path(os.path.abspath(value))


def verify_tcformer_source(root: str | Path | None = None) -> dict[str, Any]:
    """Verify and identify the exact compact upstream runtime snapshot."""

    source_root = default_tcformer_root() if root is None else Path(os.path.abspath(root))
    _require_real_ancestors(source_root, include_leaf=True)
    manifest_path = source_root / TCFORMER_MANIFEST_FILENAME
    manifest_bytes = _read_unique_regular_bytes(manifest_path, root=source_root)
    try:
        manifest = json.loads(
            manifest_bytes,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TCFormerSourceError("TCFormer SOURCE.json is not canonical JSON") from error
    if not isinstance(manifest, dict) or manifest != canonical_manifest():
        raise TCFormerSourceError(
            "TCFormer SOURCE.json differs from the pinned exact manifest"
        )
    if manifest_bytes != canonical_manifest_bytes():
        raise TCFormerSourceError("TCFormer SOURCE.json bytes are not canonical")

    observed_files: dict[str, dict[str, Any]] = {}
    for relative, expected in sorted(TCFORMER_PINNED_FILES.items()):
        payload = _read_unique_regular_bytes(source_root / relative, root=source_root)
        observed = {
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        if observed != dict(expected):
            raise TCFormerSourceError(
                f"TCFormer vendored file differs from commit {TCFORMER_COMMIT}: "
                f"{relative}"
            )
        observed_files[relative] = observed

    return {
        "source_mode": "vendored_runtime_snapshot",
        "repository": TCFORMER_REPOSITORY,
        "source_root": str(source_root),
        "commit": TCFORMER_COMMIT,
        "license": TCFORMER_LICENSE,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "file_identity": observed_files,
    }
