#!/usr/bin/env python3
"""Fail-closed structural, namespace, provenance, and checksum verification."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import tomllib


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]

REQUIRED = (
    ".gitattributes",
    ".python-version",
    "README.md",
    "SOURCE_PROVENANCE.json",
    "pyproject.toml",
    "uv.lock",
    "src/benchmark/full_grid.py",
    "src/benchmark/full_grid_analysis.py",
    "src/benchmark/reviewer_replay.py",
    "tests/core/test_full_grid.py",
    "tests/core/test_reviewer_replay.py",
    "historical/common_grid_v6/source/eeg_mi_v6/full_grid.py",
    "historical/common_grid_v6/source/eeg_mi_v6/full_grid_analysis.py",
    "historical/native_fbms_full_grid_v1/source/eeg_mi_native_v1/native_pretraining.py",
    "third_party/TCFormer/LICENSE",
    "third_party/TCFormer/SOURCE.json",
    "docs/PROJECT_STRUCTURE.md",
    "docs/REPRODUCIBILITY.md",
    "docs/REVIEWER_REPLAY.md",
    "reviewer/README.md",
    "reviewer/reviewer_score_cells.csv",
    "results/common_grid_v6/plan.json",
    "results/common_grid_v6/plan.sha256",
    "results/common_grid_v6/analysis/manifest.json",
    "results/SHA256SUMS",
    "output/pdf/SHA256SUMS",
    "scripts/build_source_provenance.py",
    "scripts/generate_reviewer_score_cells.py",
    "scripts/reproducibility_contract.py",
    "scripts/reproduce.sh",
    "scripts/verify_release.py",
)
BANNED_DIRECTORY_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "checkpoints",
    "data_cache",
    "uv-cache",
}
IGNORED_ROOT_ENVIRONMENTS = {".venv", ".venv-test", ".venv-docs"}
BANNED_FILE_SUFFIXES = (".pyc", ".pyo", ".pt", ".pth", ".ckpt", ".log")
PROVENANCE_EXCLUDED_ROOTS = sorted(
    {"results", "output", ".venv", ".venv-test", ".venv-docs"}
)
RETIRED_TOKENS = (
    ("i" + "eee").encode("ascii"),
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

EXPECTED_PREDECESSOR_ATTESTATIONS = {
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

EXPECTED_PYTORCH_INDEX_NAME = "pytorch-cu124"
EXPECTED_PYTORCH_INDEX_URL = "https://download.pytorch.org/whl/cu124"
EXPECTED_TORCH_VERSION = "2.6.0+cu124"
EXPECTED_TORCHAUDIO_VERSION = "2.6.0+cu124"
EXPECTED_TORCHAUDIO_WHEEL_URL = (
    "https://download-r2.pytorch.org/whl/cu124/"
    "torchaudio-2.6.0%2Bcu124-cp312-cp312-linux_x86_64.whl"
)
EXPECTED_TORCHAUDIO_WHEEL_HASH = (
    "sha256:3e5ffa69606171c74f3e2b969785ead50b782ca657e746aaee1ee7cc88dcfc08"
)
EXPECTED_LOCK_PACKAGE_COUNT = 109
EXPECTED_REPORTS = {
    "Benchmark_Results_and_Statistics.pdf",
    "Methodology_and_Architectures.pdf",
    "Reproducibility_Handbook.pdf",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> object:
    def object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON constant: {value}")

    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=object_pairs,
        parse_constant=reject_constant,
    )


def safe_relative_path(value: object) -> str | None:
    if not isinstance(value, str) or not value or "\\" in value:
        return None
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != value
    ):
        return None
    return value


def has_retired_token(value: bytes) -> bool:
    lowered = value.lower()
    return any(token in lowered for token in RETIRED_TOKENS)


def scan_release(errors: list[str]) -> None:
    """Reject unsafe nodes, generated debris, and the retired namespace."""

    for current, directories, files in os.walk(ROOT):
        current_path = Path(current)
        relative_directory = current_path.relative_to(ROOT)
        kept: list[str] = []
        for name in sorted(directories):
            relative = relative_directory / name
            if len(relative.parts) == 1 and name in IGNORED_ROOT_ENVIRONMENTS:
                continue
            child = current_path / name
            try:
                observed = child.lstat()
            except OSError as error:
                errors.append(f"cannot inspect directory {relative.as_posix()}: {error}")
                continue
            if name in BANNED_DIRECTORY_NAMES:
                errors.append(f"banned generated directory present: {relative.as_posix()}")
                continue
            if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
                errors.append(f"unsafe directory node: {relative.as_posix()}")
                continue
            if has_retired_token(relative.as_posix().encode("utf-8")):
                errors.append(f"retired namespace in path: {relative.as_posix()}")
            kept.append(name)
        directories[:] = kept

        for name in sorted(files):
            path = current_path / name
            relative = path.relative_to(ROOT).as_posix()
            try:
                observed = path.lstat()
            except OSError as error:
                errors.append(f"cannot inspect file {relative}: {error}")
                continue
            if stat.S_ISLNK(observed.st_mode) or not stat.S_ISREG(observed.st_mode):
                errors.append(f"unsafe file node: {relative}")
                continue
            if path.suffix.lower() in BANNED_FILE_SUFFIXES:
                errors.append(f"banned generated file present: {relative}")
            if has_retired_token(relative.encode("utf-8")):
                errors.append(f"retired namespace in path: {relative}")
            try:
                if has_retired_token(path.read_bytes()):
                    errors.append(f"retired namespace in file: {relative}")
            except OSError as error:
                errors.append(f"cannot scan file {relative}: {error}")


def expected_inventory(errors: list[str]) -> dict[str, tuple[str, int]]:
    rows: dict[str, tuple[str, int]] = {}
    for current, directories, files in os.walk(ROOT):
        current_path = Path(current)
        relative_directory = current_path.relative_to(ROOT)
        kept: list[str] = []
        for name in sorted(directories):
            relative = relative_directory / name
            if relative.parts and relative.parts[0] in PROVENANCE_EXCLUDED_ROOTS:
                continue
            if name in BANNED_DIRECTORY_NAMES:
                continue
            kept.append(name)
        directories[:] = kept
        if relative_directory.parts and relative_directory.parts[0] in PROVENANCE_EXCLUDED_ROOTS:
            continue
        for name in sorted(files):
            path = current_path / name
            relative = path.relative_to(ROOT).as_posix()
            if relative == "SOURCE_PROVENANCE.json":
                continue
            try:
                observed = path.lstat()
                if (
                    stat.S_ISLNK(observed.st_mode)
                    or not stat.S_ISREG(observed.st_mode)
                    or observed.st_nlink != 1
                ):
                    continue
                rows[relative] = (sha256(path), observed.st_size)
            except OSError as error:
                errors.append(f"cannot inventory {relative}: {error}")
    return rows


def verify_source_provenance(errors: list[str]) -> None:
    path = ROOT / "SOURCE_PROVENANCE.json"
    try:
        payload = strict_json(path)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        errors.append(f"invalid SOURCE_PROVENANCE.json: {error}")
        return
    if not isinstance(payload, dict):
        errors.append("SOURCE_PROVENANCE.json root must be an object")
        return
    if payload.get("schema") != "benchmark-source-provenance-v4":
        errors.append("source provenance schema drifted")
    if payload.get("destination_root") != ".":
        errors.append("source provenance root semantics drifted")
    project = payload.get("project")
    if project != {
        "python_namespace": "benchmark",
        "distribution_name": "benchmark",
    }:
        errors.append("source provenance project identity drifted")
    sanitation = payload.get("sanitation")
    if not isinstance(sanitation, dict):
        errors.append("source provenance sanitation contract is missing")
    else:
        if sanitation.get("status") != "complete":
            errors.append("source sanitation is not marked complete")
        if sanitation.get("scientific_values_modified") is not False:
            errors.append("source provenance does not preserve scientific values")
        if sanitation.get("identity_metadata_reissued") is not True:
            errors.append("source provenance does not disclose the metadata reissue")
        if sanitation.get("predecessor_attestations") != EXPECTED_PREDECESSOR_ATTESTATIONS:
            errors.append("predecessor attestation anchors drifted")

    integrity = payload.get("integrity")
    if not isinstance(integrity, dict):
        errors.append("source provenance integrity block is missing")
    else:
        try:
            plan = strict_json(ROOT / "results/common_grid_v6/plan.json")
            expected_integrity = {
                "canonical_plan_sha256": plan.get("plan_sha256") if isinstance(plan, dict) else None,
                "results_ledger_sha256": sha256(ROOT / "results/SHA256SUMS"),
                "reports_ledger_sha256": sha256(ROOT / "output/pdf/SHA256SUMS"),
                "reviewer_catalog_sha256": sha256(
                    ROOT / "reviewer/reviewer_score_cells.csv"
                ),
            }
            if integrity != expected_integrity:
                errors.append("source provenance integrity anchors drifted")
        except (OSError, UnicodeError, ValueError) as error:
            errors.append(f"cannot validate provenance integrity anchors: {error}")

    inventory = payload.get("inventory")
    if not isinstance(inventory, dict):
        errors.append("source provenance inventory is missing")
        return
    if inventory.get("excluded_roots") != PROVENANCE_EXCLUDED_ROOTS:
        errors.append("source provenance excluded-root policy drifted")
    if inventory.get("excluded_files") != ["SOURCE_PROVENANCE.json"]:
        errors.append("source provenance excluded-file policy drifted")
    files = inventory.get("files")
    if not isinstance(files, list):
        errors.append("source provenance file ledger is missing")
        return
    observed: dict[str, tuple[str, int]] = {}
    for index, row in enumerate(files):
        if not isinstance(row, dict):
            errors.append(f"source provenance row {index} is not an object")
            continue
        relative = safe_relative_path(row.get("path"))
        digest = row.get("sha256")
        size = row.get("size")
        if relative is None or not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
            errors.append(f"invalid source provenance row {index}")
            continue
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            errors.append(f"invalid source provenance size at row {index}")
            continue
        if relative in observed:
            errors.append(f"duplicate source provenance path: {relative}")
            continue
        observed[relative] = (digest, size)
    if inventory.get("file_count") != len(files):
        errors.append("source provenance file count drifted")
    expected = expected_inventory(errors)
    if observed != expected:
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        mismatched = sorted(
            key for key in set(expected) & set(observed) if expected[key] != observed[key]
        )
        if missing:
            errors.append(f"source provenance missing files: {missing[:8]}")
        if extra:
            errors.append(f"source provenance has extra files: {extra[:8]}")
        if mismatched:
            errors.append(f"source provenance checksum/size mismatch: {mismatched[:8]}")


def verify_uv_runtime_contract(errors: list[str]) -> None:
    try:
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
        errors.append(f"invalid UV project metadata: {error}")
        return
    project_table = project.get("project")
    if not isinstance(project_table, dict):
        errors.append("pyproject has no project table")
        return
    if project_table.get("name") != "benchmark":
        errors.append("pyproject project name drifted")
    if project_table.get("requires-python") != "==3.12.13":
        errors.append("pyproject Python pin drifted")
    dependencies = project_table.get("dependencies")
    if not isinstance(dependencies, list):
        errors.append("pyproject runtime dependencies are not a list")
        dependencies = []
    for name, expected in {
        "torch": EXPECTED_TORCH_VERSION,
        "torchaudio": EXPECTED_TORCHAUDIO_VERSION,
    }.items():
        if dependencies.count(f"{name}=={expected}") != 1:
            errors.append(f"pyproject {name} pin drifted")
    tool = project.get("tool")
    uv = tool.get("uv") if isinstance(tool, dict) else None
    sources = uv.get("sources") if isinstance(uv, dict) else None
    for name in ("torch", "torchaudio"):
        if not isinstance(sources, dict) or sources.get(name) != {
            "index": EXPECTED_PYTORCH_INDEX_NAME
        }:
            errors.append(f"pyproject {name} source drifted")
    indexes = uv.get("index") if isinstance(uv, dict) else None
    matching_indexes = [
        row
        for row in indexes or []
        if isinstance(row, dict) and row.get("name") == EXPECTED_PYTORCH_INDEX_NAME
    ]
    if matching_indexes != [
        {
            "name": EXPECTED_PYTORCH_INDEX_NAME,
            "url": EXPECTED_PYTORCH_INDEX_URL,
            "explicit": True,
        }
    ]:
        errors.append("explicit PyTorch index contract drifted")

    packages = lock.get("package")
    if not isinstance(packages, list) or len(packages) != EXPECTED_LOCK_PACKAGE_COUNT:
        errors.append("uv.lock package count drifted")
        return
    by_name: dict[str, list[dict[str, object]]] = {}
    for row in packages:
        if isinstance(row, dict) and isinstance(row.get("name"), str):
            by_name.setdefault(str(row["name"]), []).append(row)
    for name, expected in {
        "torch": EXPECTED_TORCH_VERSION,
        "torchaudio": EXPECTED_TORCHAUDIO_VERSION,
    }.items():
        rows = by_name.get(name, [])
        if len(rows) != 1 or rows[0].get("version") != expected:
            errors.append(f"uv.lock {name} version drifted")
            continue
        source = rows[0].get("source")
        if not isinstance(source, dict) or not str(source.get("registry", "")).startswith(
            EXPECTED_PYTORCH_INDEX_URL
        ):
            errors.append(f"uv.lock {name} registry drifted")
    torchaudio = by_name.get("torchaudio", [{}])[0]
    wheels = torchaudio.get("wheels") if isinstance(torchaudio, dict) else None
    if (
        not isinstance(wheels, list)
        or len(wheels) != 1
        or not isinstance(wheels[0], dict)
        or wheels[0].get("url") != EXPECTED_TORCHAUDIO_WHEEL_URL
        or wheels[0].get("hash") != EXPECTED_TORCHAUDIO_WHEEL_HASH
    ):
        errors.append("uv.lock TorchAudio wheel contract drifted")
    roots = by_name.get("benchmark", [])
    if len(roots) != 1:
        errors.append("uv.lock root project package drifted")


def read_checksum_ledger(path: Path, errors: list[str]) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as error:
        errors.append(f"cannot read checksum ledger {path}: {error}")
        return {}
    rows: dict[str, str] = {}
    for line in lines:
        parts = line.split("  ", 1)
        if len(parts) != 2 or SHA256_RE.fullmatch(parts[0]) is None:
            errors.append(f"malformed checksum row in {path}: {line!r}")
            continue
        digest, relative = parts
        if relative in rows or safe_relative_path(relative) != relative:
            errors.append(f"unsafe or duplicate checksum path in {path}: {relative!r}")
            continue
        rows[relative] = digest
    return rows


def verify_publication_outputs(errors: list[str]) -> None:
    directory = ROOT / "output/pdf"
    rows = read_checksum_ledger(directory / "SHA256SUMS", errors)
    if set(rows) != EXPECTED_REPORTS:
        errors.append("publication report checksum roster drifted")
        return
    actual = {
        path.name for path in directory.iterdir() if path.is_file() and path.name != "SHA256SUMS"
    }
    if actual != EXPECTED_REPORTS:
        errors.append("publication output directory has missing or extra files")
    for relative, expected in rows.items():
        path = directory / relative
        if path.is_symlink() or not path.is_file() or sha256(path) != expected:
            errors.append(f"publication report checksum mismatch: {relative}")


def isolated_child(script: Path, arguments: list[str]) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    return subprocess.run(
        [sys.executable, "-I", "-B", "-S", str(script), *arguments],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def verify_child_checks(errors: list[str]) -> None:
    for label, script, arguments in (
        (
            "reproducibility identity",
            ROOT / "scripts/reproducibility_contract.py",
            [],
        ),
        (
            "reviewer catalog",
            ROOT / "scripts/generate_reviewer_score_cells.py",
            ["--check"],
        ),
        (
            "results bundle",
            ROOT / "results/verify_bundle.py",
            [],
        ),
    ):
        completed = isolated_child(script, arguments)
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            errors.append(f"{label} verification failed: {detail}")


def main() -> int:
    if len(sys.argv) != 1:
        print("Usage: verify_release.py", file=sys.stderr)
        return 2
    errors: list[str] = []
    for relative in REQUIRED:
        path = ROOT / relative
        if not path.is_file() or path.is_symlink():
            errors.append(f"required regular file is missing or unsafe: {relative}")
    scan_release(errors)
    verify_source_provenance(errors)
    verify_uv_runtime_contract(errors)
    verify_publication_outputs(errors)
    if not errors:
        verify_child_checks(errors)
    if errors:
        print("RELEASE VERIFICATION FAILED", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    inventory = strict_json(ROOT / "SOURCE_PROVENANCE.json")
    count = inventory.get("inventory", {}).get("file_count") if isinstance(inventory, dict) else None
    print(f"RELEASE VERIFICATION PASSED (files={count})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
