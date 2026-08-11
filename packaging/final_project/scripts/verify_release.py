#!/usr/bin/env python3
"""Fail-closed structural and checksum checks for the clean release bundle."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]

REQUIRED = (
    "README.md",
    ".python-version",
    "pyproject.toml",
    "uv.lock",
    "SOURCE_PROVENANCE.json",
    "ieee_mi/full_grid.py",
    "ieee_mi/full_grid_analysis.py",
    "test_full_grid.py",
    "third_party/TCFormer/LICENSE",
    "third_party/TCFormer/SOURCE.json",
    "docs/REPRODUCIBILITY.md",
    "docs/RESULTS_INTERPRETATION.md",
    "docs/METHODOLOGY_AND_ARCHITECTURES.md",
    "results/common_grid_v6/analysis/manifest.json",
    "results/SHA256SUMS",
)

BANNED_DIRECTORY_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "checkpoints",
    "data_cache",
    "uv-cache",
}

# These three generated environments are an intentional project-root runtime
# boundary. They are never part of the release checksum surface, but their
# presence after setup must not make the release verifier self-contradictory.
# Names matching these anywhere below the root remain forbidden.
IGNORED_ROOT_ENVIRONMENTS = {".venv", ".venv-test", ".venv-docs"}

BANNED_FILE_SUFFIXES = (".pyc", ".pyo", ".pt", ".pth", ".ckpt", ".log")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

FROZEN_CORE_HASHES = {
    "ieee_mi/full_grid.py": "5e687a5257754da301d2d42984948dc315de1f979fd1e4bd7c27c507857e67f5",
    "ieee_mi/full_grid_analysis.py": "bcacbf8c8b36e0ea353e690bece38eabc8712c288b0d553a940dbf68d8a22e40",
    "test_full_grid.py": "731c82eb55bba37202da6a9e21b75f00c5c237d4517592cf2d22e8f9def9d006",
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> object:
    """Load JSON while rejecting duplicate keys and non-finite constants."""

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
    """Return one canonical POSIX-relative path, never an escape path."""

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


def canonical_requirement_name(value: object) -> str | None:
    """Return the normalized leading requirement name from a direct pin."""

    if not isinstance(value, str):
        return None
    match = re.match(r"^[A-Za-z0-9_.-]+", value.strip())
    if match is None:
        return None
    return match.group(0).lower().replace("_", "-")


def verify_uv_runtime_contract(errors: list[str]) -> None:
    """Verify the exact mutually compatible Torch/TorchAudio CUDA contract."""

    pyproject_path = ROOT / "pyproject.toml"
    lock_path = ROOT / "uv.lock"
    if (
        not pyproject_path.is_file()
        or pyproject_path.is_symlink()
        or not lock_path.is_file()
        or lock_path.is_symlink()
    ):
        return
    try:
        project = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
        errors.append(f"invalid UV manifest TOML: {error}")
        return

    project_table = project.get("project")
    dependencies = (
        project_table.get("dependencies")
        if isinstance(project_table, dict)
        else None
    )
    expected_pins = {
        "torch": f"torch=={EXPECTED_TORCH_VERSION}",
        "torchaudio": f"torchaudio=={EXPECTED_TORCHAUDIO_VERSION}",
    }
    if not isinstance(dependencies, list):
        errors.append("pyproject runtime dependencies are not a list")
    else:
        for name, expected_pin in expected_pins.items():
            observed = [
                value
                for value in dependencies
                if canonical_requirement_name(value) == name
            ]
            if observed != [expected_pin]:
                errors.append(
                    f"pyproject {name} pin must be exactly {expected_pin}: "
                    f"observed={observed!r}"
                )

    tool = project.get("tool")
    uv = tool.get("uv") if isinstance(tool, dict) else None
    sources = uv.get("sources") if isinstance(uv, dict) else None
    expected_source = {"index": EXPECTED_PYTORCH_INDEX_NAME}
    if not isinstance(sources, dict):
        errors.append("pyproject has no [tool.uv.sources] mapping")
    else:
        for name in expected_pins:
            if sources.get(name) != expected_source:
                errors.append(
                    f"pyproject {name} source must be the explicit "
                    f"{EXPECTED_PYTORCH_INDEX_NAME} index"
                )

    indexes = uv.get("index") if isinstance(uv, dict) else None
    expected_index = {
        "name": EXPECTED_PYTORCH_INDEX_NAME,
        "url": EXPECTED_PYTORCH_INDEX_URL,
        "explicit": True,
    }
    matching_indexes = (
        [
            value
            for value in indexes
            if isinstance(value, dict)
            and value.get("name") == EXPECTED_PYTORCH_INDEX_NAME
        ]
        if isinstance(indexes, list)
        else []
    )
    if matching_indexes != [expected_index]:
        errors.append(
            "pyproject pytorch-cu124 index must have the exact reviewed URL "
            "and explicit=true"
        )

    packages = lock.get("package")
    if not isinstance(packages, list):
        errors.append("uv.lock package ledger is not a list")
        return
    if len(packages) != EXPECTED_LOCK_PACKAGE_COUNT:
        errors.append(
            f"uv.lock package count must be {EXPECTED_LOCK_PACKAGE_COUNT}: "
            f"observed={len(packages)}"
        )
    named_packages: dict[str, list[dict[str, object]]] = {}
    for value in packages:
        if not isinstance(value, dict):
            errors.append("uv.lock contains a non-table package entry")
            continue
        name = canonical_requirement_name(value.get("name"))
        if name is not None:
            named_packages.setdefault(name, []).append(value)

    expected_registry = {"registry": EXPECTED_PYTORCH_INDEX_URL}
    for name, expected_version in (
        ("torch", EXPECTED_TORCH_VERSION),
        ("torchaudio", EXPECTED_TORCHAUDIO_VERSION),
    ):
        matches = named_packages.get(name, [])
        if len(matches) != 1:
            errors.append(f"uv.lock must contain exactly one {name} package")
            continue
        package = matches[0]
        if package.get("version") != expected_version:
            errors.append(
                f"uv.lock {name} version must be {expected_version}: "
                f"observed={package.get('version')!r}"
            )
        if package.get("source") != expected_registry:
            errors.append(
                f"uv.lock {name} must resolve from {EXPECTED_PYTORCH_INDEX_URL}"
            )

    torchaudio_packages = named_packages.get("torchaudio", [])
    if len(torchaudio_packages) == 1:
        torchaudio = torchaudio_packages[0]
        if torchaudio.get("dependencies") != [{"name": "torch"}]:
            errors.append("uv.lock torchaudio must depend exactly on torch")
        wheels = torchaudio.get("wheels")
        if not isinstance(wheels, list) or len(wheels) != 1:
            errors.append("uv.lock torchaudio must contain exactly one wheel")
        else:
            wheel = wheels[0]
            if (
                not isinstance(wheel, dict)
                or wheel.get("url") != EXPECTED_TORCHAUDIO_WHEEL_URL
                or wheel.get("hash") != EXPECTED_TORCHAUDIO_WHEEL_HASH
            ):
                errors.append(
                    "uv.lock torchaudio wheel URL/hash does not match the "
                    "reviewed CPython 3.12 Linux x86-64 cu124 artifact"
                )

    root_packages = named_packages.get("ieee-mi-benchmark", [])
    if len(root_packages) != 1:
        errors.append("uv.lock must contain exactly one root project package")
        return
    root_package = root_packages[0]
    root_dependencies = root_package.get("dependencies")
    if not isinstance(root_dependencies, list) or {"name": "torchaudio"} not in root_dependencies:
        errors.append("uv.lock root package does not directly depend on torchaudio")
    metadata = root_package.get("metadata")
    requires_dist = metadata.get("requires-dist") if isinstance(metadata, dict) else None
    expected_metadata = {
        "name": "torchaudio",
        "specifier": f"=={EXPECTED_TORCHAUDIO_VERSION}",
        "index": EXPECTED_PYTORCH_INDEX_URL,
    }
    observed_metadata = (
        [
            value
            for value in requires_dist
            if isinstance(value, dict) and value.get("name") == "torchaudio"
        ]
        if isinstance(requires_dist, list)
        else []
    )
    if observed_metadata != [expected_metadata]:
        errors.append(
            "uv.lock root metadata must pin torchaudio==2.6.0+cu124 to the "
            "explicit cu124 index"
        )


def verify_source_provenance(errors: list[str]) -> None:
    path = ROOT / "SOURCE_PROVENANCE.json"
    if not path.is_file() or path.is_symlink():
        return
    try:
        provenance = strict_json(path)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        errors.append(f"invalid SOURCE_PROVENANCE.json: {error}")
        return
    if not isinstance(provenance, dict):
        errors.append("SOURCE_PROVENANCE.json root must be an object")
        return
    if (
        provenance.get("destination_root") != "."
        or provenance.get("destination_root_semantics")
        != "directory containing SOURCE_PROVENANCE.json"
    ):
        errors.append("source provenance destination root is not portable")
    seen: set[str] = set()
    for section_name in ("formal_v6", "local_research_extensions"):
        section = provenance.get(section_name)
        if not isinstance(section, dict):
            errors.append(f"source provenance section missing: {section_name}")
            continue
        files = section.get("files")
        count = section.get("file_count")
        if not isinstance(files, list) or type(count) is not int or count != len(files):
            errors.append(f"source provenance file count is invalid: {section_name}")
            continue
        for index, item in enumerate(files):
            if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
                errors.append(
                    f"source provenance entry is invalid: {section_name}[{index}]"
                )
                continue
            relative = safe_relative_path(item.get("path"))
            expected = item.get("sha256")
            if relative is None or not isinstance(expected, str) or not SHA256_RE.fullmatch(expected):
                errors.append(
                    f"source provenance path/hash is invalid: {section_name}[{index}]"
                )
                continue
            if relative in seen:
                errors.append(f"duplicate source provenance path: {relative}")
                continue
            seen.add(relative)
            target = ROOT / relative
            if not target.is_file() or target.is_symlink():
                errors.append(f"source provenance target missing or unsafe: {relative}")
            elif sha256(target) != expected:
                errors.append(f"source provenance checksum mismatch: {relative}")


def verify_analysis_manifest(errors: list[str]) -> None:
    analysis = ROOT / "results/common_grid_v6/analysis"
    manifest_path = analysis / "manifest.json"
    if not manifest_path.is_file():
        return
    try:
        manifest = strict_json(manifest_path)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        errors.append(f"invalid analysis manifest: {error}")
        return
    if not isinstance(manifest, dict):
        errors.append("analysis manifest root must be an object")
        return
    declared = manifest.get("files")
    if not isinstance(declared, dict):
        errors.append("analysis manifest has no files mapping")
        return
    safe_declared: dict[str, str] = {}
    for raw_name, raw_hash in declared.items():
        name = safe_relative_path(raw_name)
        if (
            name is None
            or "/" in name
            or not isinstance(raw_hash, str)
            or not SHA256_RE.fullmatch(raw_hash)
        ):
            errors.append(f"analysis manifest contains an unsafe entry: {raw_name!r}")
            continue
        safe_declared[name] = raw_hash
    actual = {
        path.name
        for path in analysis.iterdir()
        if path.is_file() and not path.is_symlink()
    }
    expected = {"manifest.json", *safe_declared}
    if actual != expected:
        errors.append(
            "analysis files differ from manifest: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    for name, expected_hash in safe_declared.items():
        path = analysis / name
        if not path.is_file() or path.is_symlink():
            continue
        observed = sha256(path)
        if observed != expected_hash:
            errors.append(f"analysis checksum mismatch: {name}")


def verify_sha256sums(errors: list[str]) -> None:
    sums = ROOT / "results/SHA256SUMS"
    if not sums.is_file():
        return
    declared: dict[str, str] = {}
    try:
        lines = sums.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        errors.append(f"cannot read results/SHA256SUMS: {error}")
        return
    for number, line in enumerate(lines, 1):
        if not line.strip():
            errors.append(f"blank results/SHA256SUMS line {number}")
            continue
        try:
            expected, relative = line.split(maxsplit=1)
        except ValueError:
            errors.append(f"malformed results/SHA256SUMS line {number}")
            continue
        if relative.startswith("*"):
            relative = relative[1:]
        safe_relative = safe_relative_path(relative)
        if (
            not SHA256_RE.fullmatch(expected)
            or safe_relative is None
            or safe_relative == "SHA256SUMS"
        ):
            errors.append(f"unsafe results/SHA256SUMS line {number}")
            continue
        if safe_relative in declared:
            errors.append(f"duplicate results checksum target: {safe_relative}")
            continue
        declared[safe_relative] = expected
        path = ROOT / "results" / relative
        if not path.is_file() or path.is_symlink():
            errors.append(f"results checksum target missing: {relative}")
        elif sha256(path) != expected:
            errors.append(f"results checksum mismatch: {relative}")
    actual = {
        path.relative_to(ROOT / "results").as_posix()
        for path in (ROOT / "results").rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and path != sums
    }
    if set(declared) != actual:
        errors.append(
            "results checksum coverage differs from release files: "
            f"missing={sorted(actual - set(declared))}, "
            f"extra={sorted(set(declared) - actual)}"
        )


def main() -> int:
    if len(sys.argv) != 1:
        print("Usage: scripts/verify_release.py", file=sys.stderr)
        return 2
    errors: list[str] = []
    for relative in REQUIRED:
        if not (ROOT / relative).is_file():
            errors.append(f"required file missing: {relative}")

    for relative, expected in FROZEN_CORE_HASHES.items():
        path = ROOT / relative
        if path.is_file() and sha256(path) != expected:
            errors.append(f"frozen core checksum mismatch: {relative}")

    verify_uv_runtime_contract(errors)

    for current, directories, files in os.walk(ROOT):
        current_path = Path(current)
        for name in tuple(directories):
            path = current_path / name
            try:
                observed = path.lstat()
            except OSError as error:
                errors.append(f"cannot inspect directory {path.relative_to(ROOT)}: {error}")
                continue
            if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
                errors.append(f"unsafe directory node present: {path.relative_to(ROOT)}")
                directories.remove(name)
                continue
            if current_path == ROOT and name in IGNORED_ROOT_ENVIRONMENTS:
                # Do not traverse or checksum project-private environments.
                # Their isolation is validated by setup_uv.sh/_common.sh.
                directories.remove(name)
                continue
            if name in IGNORED_ROOT_ENVIRONMENTS:
                errors.append(
                    f"project environment is outside the allowed root boundary: "
                    f"{path.relative_to(ROOT)}"
                )
                directories.remove(name)
                continue
            if name in BANNED_DIRECTORY_NAMES:
                errors.append(
                    f"banned generated directory present: "
                    f"{(current_path / name).relative_to(ROOT)}"
                )
        for name in files:
            path = current_path / name
            try:
                observed = path.lstat()
            except OSError as error:
                errors.append(f"cannot inspect file {path.relative_to(ROOT)}: {error}")
                continue
            if (
                not stat.S_ISREG(observed.st_mode)
                or stat.S_ISLNK(observed.st_mode)
                or observed.st_nlink != 1
            ):
                errors.append(f"unsafe file node present: {path.relative_to(ROOT)}")
            if name.endswith(BANNED_FILE_SUFFIXES) or name == ".DS_Store":
                errors.append(
                    f"banned generated file present: "
                    f"{path.relative_to(ROOT)}"
                )

    if (ROOT / "full_grid.py").exists():
        errors.append("duplicate root full_grid.py is forbidden")

    verify_source_provenance(errors)
    verify_analysis_manifest(errors)
    verify_sha256sums(errors)

    if errors:
        print("RELEASE VERIFICATION FAILED", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("RELEASE VERIFICATION PASSED")
    print(f"root={ROOT}")
    print(f"files={sum(1 for path in ROOT.rglob('*') if path.is_file())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
