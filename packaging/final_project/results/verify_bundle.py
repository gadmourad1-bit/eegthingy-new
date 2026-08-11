#!/usr/bin/env python3
"""Verify imported bytes, sealed closures, checksums, and release boundaries."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import stat
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parent
HEX64 = re.compile(r"[0-9a-f]{64}")
IMPORT_AUTHORITY = "authorized-lab-artifact-authority"
IMPORT_LOCATOR_PREFIX = "sealed-artifact://"
IMPORT_FIELDS = (
    "source_authority",
    "source_locator",
    "destination",
    "sha256",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or not value or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe relative path: {value!r}")
    return path


def regular_files() -> set[str]:
    files: set[str] = set()
    for directory, dirnames, filenames in os.walk(ROOT, followlinks=False):
        base = Path(directory)
        for name in dirnames:
            if (base / name).is_symlink():
                raise ValueError(f"symlinked directory is forbidden: {base / name}")
        for name in filenames:
            path = base / name
            if path.is_symlink():
                raise ValueError(f"symlink is forbidden: {path}")
            if not stat.S_ISREG(path.stat(follow_symlinks=False).st_mode):
                raise ValueError(f"non-regular file is forbidden: {path}")
            relative = path.relative_to(ROOT).as_posix()
            if relative != "SHA256SUMS":
                files.add(relative)
    return files


def verify_checksums() -> None:
    lines = (ROOT / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    entries: list[tuple[str, str]] = []
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match:
            raise ValueError(f"malformed SHA256SUMS line: {line!r}")
        digest, value = match.groups()
        safe_relative(value)
        entries.append((value, digest))
    paths = [path for path, _ in entries]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("SHA256SUMS paths are not sorted and unique")
    actual = regular_files()
    if set(paths) != actual:
        raise ValueError(
            f"SHA256SUMS coverage mismatch; missing={sorted(actual-set(paths))}, "
            f"extra={sorted(set(paths)-actual)}"
        )
    for relative, expected in entries:
        observed = sha256(ROOT / relative)
        if observed != expected:
            raise ValueError(f"checksum mismatch for {relative}: {observed} != {expected}")


def verify_imports() -> None:
    with (ROOT / "IMPORT_MANIFEST.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != IMPORT_FIELDS:
            raise ValueError("unexpected import-manifest columns")
        rows = list(reader)
    if len(rows) != 27:
        raise ValueError(f"expected 27 imported files, found {len(rows)}")
    destinations = [row["destination"] for row in rows]
    if len(destinations) != len(set(destinations)):
        raise ValueError("duplicate import destination")
    for row in rows:
        destination = safe_relative(row["destination"])
        expected_locator = IMPORT_LOCATOR_PREFIX + destination.as_posix()
        if row["source_authority"] != IMPORT_AUTHORITY:
            raise ValueError(f"unexpected import authority for {destination}")
        if row["source_locator"] != expected_locator:
            raise ValueError(f"unexpected logical source locator for {destination}")
        expected = row["sha256"]
        if not HEX64.fullmatch(expected):
            raise ValueError(f"invalid import checksum for {destination}")
        if sha256(ROOT / Path(*destination.parts)) != expected:
            raise ValueError(f"imported bytes differ for {destination}")


def verify_scientific_closure() -> None:
    common = ROOT / "common_grid_v6"
    analysis = common / "analysis"
    manifest = json.loads((analysis / "manifest.json").read_text(encoding="utf-8"))
    sealed_files = manifest["files"]
    observed_names = {path.name for path in analysis.iterdir() if path.is_file()}
    if observed_names != set(sealed_files) | {"manifest.json"} or len(observed_names) != 17:
        raise ValueError("sealed common-grid analysis is not the exact 17-file package")
    for name, expected in sealed_files.items():
        safe_relative(name)
        if sha256(analysis / name) != expected:
            raise ValueError(f"sealed analysis manifest mismatch for {name}")

    plan = json.loads((common / "plan.json").read_text(encoding="utf-8"))
    audit = json.loads((common / "final_audit.json").read_text(encoding="utf-8"))
    receipt = json.loads((common / "preflight" / "receipt.json").read_text(encoding="utf-8"))
    canonical_plan_sha = (common / "plan.sha256").read_text(encoding="utf-8").strip()
    if not HEX64.fullmatch(canonical_plan_sha):
        raise ValueError("invalid canonical plan digest")
    if not all(
        value == canonical_plan_sha
        for value in (
            plan["plan_sha256"],
            audit["plan_sha256"],
            receipt["plan_sha256"],
            manifest["plan_sha256"],
        )
    ):
        raise ValueError("common-grid plan identity is not closed")
    report_sha = sha256(common / "preflight" / "report.json")
    if report_sha != receipt["report_sha256"] or report_sha != audit["preflight_report_sha256"]:
        raise ValueError("preflight report identity is not closed")
    required_audit = {
        "exact_cartesian_complete": True,
        "expected": 96320,
        "complete": 96320,
        "missing": 0,
        "extra": 0,
        "failed_jobs": 0,
        "live_claims": 0,
        "stale_claims": 0,
        "partials": 0,
        "unexpected_root_entries": 0,
        "unsafe_paths": 0,
        "preflight_attestation_valid": True,
    }
    for key, expected in required_audit.items():
        if audit.get(key) != expected:
            raise ValueError(f"common-grid audit field {key} is not {expected!r}")
    if receipt.get("completed_checks") != 172 or receipt.get("expected_checks") != 172:
        raise ValueError("common-grid preflight is not 172/172")

    gauge = ROOT / "gauge_gate1"
    if sha256(gauge / "plan.json") != (gauge / "plan.sha256").read_text().strip():
        raise ValueError("Gauge plan checksum mismatch")
    if sha256(gauge / "analysis.json") != (gauge / "analysis.sha256").read_text().strip():
        raise ValueError("Gauge analysis checksum mismatch")
    gauge_analysis = json.loads((gauge / "analysis.json").read_text(encoding="utf-8"))
    if gauge_analysis.get("passed") is not False or gauge_analysis.get("job_count") != 17:
        raise ValueError("Gauge Gate 1 status was misrepresented")

    cardinal = json.loads(
        (ROOT / "cardinal_fbms_transfer" / "independent_verification_lab.json").read_text(
            encoding="utf-8"
        )
    )
    if cardinal.get("status") != "PASS" or cardinal.get("validated_record_count") != 8675:
        raise ValueError("CardinalFBMS verification status was misrepresented")

    for name in ("all_models_balanced_accuracy.csv", "all_models_accuracy.csv"):
        with (common / name).open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 43 or len({row["model"] for row in rows}) != 43:
            raise ValueError(f"{name} is not a complete 43-model view")

    forbidden_suffixes = {".npz", ".npy", ".pt", ".pth", ".ckpt", ".edf", ".gdf", ".fif"}
    forbidden_parts = {"predictions", "datasets", "claims", "checkpoints", "worker_logs", "cache"}
    for relative in regular_files():
        path = PurePosixPath(relative)
        if path.suffix.lower() in forbidden_suffixes or forbidden_parts.intersection(path.parts):
            raise ValueError(f"forbidden raw/runtime artifact in results bundle: {relative}")


def main() -> None:
    verify_checksums()
    verify_imports()
    verify_scientific_closure()
    print("PASS: results bundle checksums, imports, sealed closures, and boundaries verified")


if __name__ == "__main__":
    main()
