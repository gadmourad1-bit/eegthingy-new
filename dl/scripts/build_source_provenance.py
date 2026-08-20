#!/usr/bin/env python3
"""Build the fail-closed source inventory for the sanitized release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "SOURCE_PROVENANCE.json"
EXCLUDED_ROOTS = frozenset(
    {"results", "output", ".venv", ".venv-test", ".venv-docs"}
)
EXCLUDED_FILES = frozenset({"SOURCE_PROVENANCE.json"})
BANNED_DIRECTORY_NAMES = frozenset(
    {"__pycache__", ".pytest_cache", ".ruff_cache", "uv-cache", "data_cache"}
)

# Construct the retired spelling so this verifier does not contain it itself.
RETIRED_TOKENS = (
    ("i" + "eee").encode("ascii"),
)

PREDECESSOR_ATTESTATIONS = {
    "source_provenance_sha256": (
        "1653210ce2edbd792548273e5358079843c50f6c0b914dcffb0a8bc4d9827830"
    ),
    "results_ledger_sha256": (
        "059710591fb9a5c444a6e79082c291659439b9b1f718596cda23d33966a4ee1e"
    ),
    "canonical_plan_sha256": (
        "d4dd852fc9e7b80c8c10c54df03d63ce0ae7e7555ee1b7abdee8d79dca28b61a"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> dict[str, object]:
    def object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate key in {path}: {key}")
            value[key] = item
        return value

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=object_pairs)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def safe_relative(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != value
    ):
        raise ValueError(f"unsafe relative path: {value!r}")
    return value


def has_retired_token(value: bytes) -> bool:
    lowered = value.lower()
    return any(token in lowered for token in RETIRED_TOKENS)


def validate_sanitized_release() -> None:
    """Reject the retired namespace in every distributed path or raw file."""

    for current, directories, files in os.walk(ROOT):
        current_path = Path(current)
        relative_directory = current_path.relative_to(ROOT)
        kept: list[str] = []
        for name in sorted(directories):
            child_relative = relative_directory / name
            if child_relative.parts and child_relative.parts[0] in {
                ".venv",
                ".venv-test",
                ".venv-docs",
            }:
                continue
            if name in BANNED_DIRECTORY_NAMES:
                raise ValueError(f"banned generated directory: {child_relative.as_posix()}")
            child = current_path / name
            observed = child.lstat()
            if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
                raise ValueError(f"unsafe directory node: {child_relative.as_posix()}")
            if has_retired_token(child_relative.as_posix().encode("utf-8")):
                raise ValueError(f"retired namespace in path: {child_relative.as_posix()}")
            kept.append(name)
        directories[:] = kept

        for name in sorted(files):
            path = current_path / name
            relative = path.relative_to(ROOT).as_posix()
            observed = path.lstat()
            if stat.S_ISLNK(observed.st_mode) or not stat.S_ISREG(observed.st_mode):
                raise ValueError(f"unsafe file node: {relative}")
            if has_retired_token(relative.encode("utf-8")):
                raise ValueError(f"retired namespace in path: {relative}")
            if has_retired_token(path.read_bytes()):
                raise ValueError(f"retired namespace in file: {relative}")


def inventory() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for current, directories, files in os.walk(ROOT):
        current_path = Path(current)
        relative_directory = current_path.relative_to(ROOT)
        kept: list[str] = []
        for name in sorted(directories):
            child = current_path / name
            relative = child.relative_to(ROOT)
            if relative.parts and relative.parts[0] in EXCLUDED_ROOTS:
                continue
            if name in BANNED_DIRECTORY_NAMES:
                raise ValueError(f"banned generated directory: {relative.as_posix()}")
            observed = child.lstat()
            if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
                raise ValueError(f"unsafe directory node: {relative.as_posix()}")
            kept.append(name)
        directories[:] = kept
        if relative_directory.parts and relative_directory.parts[0] in EXCLUDED_ROOTS:
            continue
        for name in sorted(files):
            path = current_path / name
            relative = path.relative_to(ROOT).as_posix()
            if relative in EXCLUDED_FILES:
                continue
            safe_relative(relative)
            observed = path.lstat()
            if (
                stat.S_ISLNK(observed.st_mode)
                or not stat.S_ISREG(observed.st_mode)
                or observed.st_nlink != 1
            ):
                raise ValueError(f"unsafe or multiply linked file: {relative}")
            rows.append(
                {"path": relative, "sha256": sha256(path), "size": observed.st_size}
            )
    rows.sort(key=lambda item: str(item["path"]))
    return rows


def build(generated_at_utc: str) -> dict[str, object]:
    if not generated_at_utc.endswith("Z") or "T" not in generated_at_utc:
        raise ValueError("--generated-at-utc must be an explicit UTC timestamp ending in Z")
    validate_sanitized_release()
    rows = inventory()
    plan = strict_json(ROOT / "results/common_grid_v6/plan.json")
    return {
        "schema": "benchmark-source-provenance-v4",
        "generated_at_utc": generated_at_utc,
        "destination_root": ".",
        "destination_root_semantics": "directory containing SOURCE_PROVENANCE.json",
        "project": {
            "python_namespace": "benchmark",
            "distribution_name": "benchmark",
        },
        "sanitation": {
            "status": "complete",
            "policy": "retired project namespace absent from distributed path names and raw bytes",
            "scientific_values_modified": False,
            "identity_metadata_reissued": True,
            "predecessor_attestations": PREDECESSOR_ATTESTATIONS,
        },
        "integrity": {
            "canonical_plan_sha256": plan.get("plan_sha256"),
            "results_ledger_sha256": sha256(ROOT / "results/SHA256SUMS"),
            "reports_ledger_sha256": sha256(ROOT / "output/pdf/SHA256SUMS"),
            "reviewer_catalog_sha256": sha256(
                ROOT / "reviewer/reviewer_score_cells.csv"
            ),
        },
        "inventory": {
            "scope": "all distributed regular files except independently sealed results and generated PDF outputs",
            "excluded_roots": sorted(EXCLUDED_ROOTS),
            "excluded_files": sorted(EXCLUDED_FILES),
            "file_count": len(rows),
            "files": rows,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-at-utc", required=True)
    arguments = parser.parse_args()
    payload = build(arguments.generated_at_utc)
    temporary = OUTPUT.with_name(OUTPUT.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
