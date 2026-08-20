"""Create the compact, pinned TCFormer runtime snapshot used by the release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path
from typing import Sequence

from benchmark.tcformer_source import (
    TCFORMER_COMMIT,
    TCFORMER_MANIFEST_FILENAME,
    TCFORMER_PINNED_FILES,
    TCFORMER_REPOSITORY,
    _read_unique_regular_bytes,
    _require_real_ancestors,
    canonical_manifest_bytes,
    verify_tcformer_source,
)


class VendorError(RuntimeError):
    pass


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_exclusive(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o444,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise VendorError(f"short write while creating {path}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _git_output(checkout: Path, arguments: Sequence[str]) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(checkout), *arguments],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        raise VendorError(
            f"cannot verify the TCFormer source checkout at {checkout}"
        ) from error


def vendor_tcformer(source_checkout: Path, destination: Path) -> dict[str, object]:
    source = Path(os.path.abspath(source_checkout))
    target = Path(os.path.abspath(destination))
    _require_real_ancestors(source, include_leaf=True)
    observed_source = os.lstat(source)
    if stat.S_ISLNK(observed_source.st_mode) or not stat.S_ISDIR(
        observed_source.st_mode
    ):
        raise VendorError("TCFormer source checkout must be a real directory")
    if os.path.lexists(target):
        raise VendorError(f"destination already exists; refusing to overwrite: {target}")
    _require_real_ancestors(target, include_leaf=False)

    commit = _git_output(source, ("rev-parse", "HEAD"))
    if commit != TCFORMER_COMMIT:
        raise VendorError(
            f"TCFormer checkout is {commit}, expected {TCFORMER_COMMIT}"
        )
    if _git_output(source, ("status", "--porcelain", "--untracked-files=no")):
        raise VendorError("TCFormer checkout has tracked modifications")
    origin = _git_output(source, ("remote", "get-url", "origin"))
    if origin.rstrip("/") not in {
        TCFORMER_REPOSITORY,
        f"{TCFORMER_REPOSITORY}.git",
    }:
        raise VendorError(f"unexpected TCFormer origin: {origin}")

    payloads: dict[str, bytes] = {}
    for relative, expected in sorted(TCFORMER_PINNED_FILES.items()):
        payload = _read_unique_regular_bytes(source / relative, root=source)
        observed = {
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        if observed != dict(expected):
            raise VendorError(f"upstream runtime file differs from pin: {relative}")
        payloads[relative] = payload

    target.mkdir(mode=0o755)
    _fsync_directory(target.parent)
    for relative, payload in payloads.items():
        output = target / relative
        output.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
        _write_exclusive(output, payload)
        _fsync_directory(output.parent)
    _write_exclusive(
        target / TCFORMER_MANIFEST_FILENAME,
        canonical_manifest_bytes(),
    )
    _fsync_directory(target)
    identity = verify_tcformer_source(target)
    return identity


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a compact hash-pinned TCFormer runtime snapshot"
    )
    parser.add_argument("--source-checkout", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    identity = vendor_tcformer(
        arguments.source_checkout,
        arguments.destination,
    )
    print(json.dumps(identity, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
