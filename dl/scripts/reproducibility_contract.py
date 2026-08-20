#!/usr/bin/env python3
"""Validate the fixed scientific identity of the sanitized benchmark release."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PLAN_SHA256 = (
    "65b93b7e5d09cfc30fb6ce28156368e66aec1503f4efa37faa3189e25ebbc3b1"
)
EXPECTED_FORMAL_SEEDS = (7, 17, 27, 37, 47)
EXPECTED_TEMPLATE_SEED = 7
EXPECTED_JOBS = 96_320
EXPECTED_ARCHITECTURES = 43
EXPECTED_BOOTSTRAP_SEED = 20_260_729
EXPECTED_BOOTSTRAP_RESAMPLES = 100_000
EXPECTED_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
EXPECTED_SOURCE_PATHS = frozenset(
    {
        "eeg_mi/baselines.py",
        "eeg_mi/benchmark.py",
        "eeg_mi/config.py",
        "eeg_mi/data.py",
        "eeg_mi/full_grid.py",
        "eeg_mi/models.py",
        "eeg_mi/project_gpu_leases.py",
        "eeg_mi/tcformer_source.py",
        "eeg_mi/training.py",
        "pyproject.toml",
        "uv.lock",
    }
)
EXPECTED_ANALYSIS_SOURCE = "eeg_mi/full_grid_analysis.py"
EXPECTED_REPORT_NAMES = frozenset(
    {
        "Benchmark_Results_and_Statistics.pdf",
        "Methodology_and_Architectures.pdf",
        "Reproducibility_Handbook.pdf",
    }
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ContractError(RuntimeError):
    """The release no longer matches its reproducibility contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    def object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON constant: {value}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise ContractError(f"JSON root is not an object: {path}")
    return value


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _literal_assignments(path: Path, names: set[str]) -> dict[str, object]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as error:
        raise ContractError(f"cannot parse {path}: {error}") from error
    values: dict[str, object] = {}
    for node in tree.body:
        target: ast.expr | None = None
        expression: ast.expr | None = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, expression = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign):
            target, expression = node.target, node.value
        if not isinstance(target, ast.Name) or target.id not in names:
            continue
        if expression is None:
            continue
        try:
            values[target.id] = ast.literal_eval(expression)
        except (ValueError, TypeError) as error:
            raise ContractError(f"{path}:{target.id} is not a literal") from error
    missing = names - set(values)
    if missing:
        raise ContractError(f"{path} lacks contract constants: {sorted(missing)}")
    return values


def _read_checksum_ledger(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as error:
        raise ContractError(f"cannot read checksum ledger {path}: {error}") from error
    rows: dict[str, str] = {}
    for line in lines:
        if not line:
            raise ContractError(f"blank checksum row in {path}")
        parts = line.split("  ", 1)
        if len(parts) != 2 or SHA256_RE.fullmatch(parts[0]) is None:
            raise ContractError(f"malformed checksum row in {path}: {line!r}")
        digest, relative = parts
        if relative in rows or not relative or "/" in relative or "\\" in relative:
            raise ContractError(f"unsafe or duplicate checksum path in {path}: {relative!r}")
        rows[relative] = digest
    return rows


def _verify_reports() -> dict[str, str]:
    directory = ROOT / "output/pdf"
    ledger = directory / "SHA256SUMS"
    rows = _read_checksum_ledger(ledger)
    if frozenset(rows) != EXPECTED_REPORT_NAMES:
        raise ContractError("publication report roster drifted")
    actual_names = frozenset(
        item.name for item in directory.iterdir() if item.is_file() and item.name != "SHA256SUMS"
    )
    if actual_names != EXPECTED_REPORT_NAMES:
        raise ContractError("publication output directory has missing or extra files")
    for name, expected in rows.items():
        path = directory / name
        if path.is_symlink() or not path.is_file() or _sha256(path) != expected:
            raise ContractError(f"publication report checksum mismatch: {name}")
    return dict(sorted(rows.items()))


def validate_release_identity(root: Path = ROOT) -> dict[str, object]:
    global ROOT
    original_root = ROOT
    ROOT = root
    try:
        plan_path = root / "results/common_grid_v6/plan.json"
        sidecar_path = root / "results/common_grid_v6/plan.sha256"
        analysis_path = root / "results/common_grid_v6/analysis/analysis.json"
        source_root = root / "historical/common_grid_v6/source"
        package_root = source_root / "eeg_mi_v6"
        plan = _strict_json(plan_path)
        analysis = _strict_json(analysis_path)

        canonical = copy.deepcopy(plan)
        canonical.pop("plan_sha256", None)
        computed_plan = hashlib.sha256(_canonical_bytes(canonical)).hexdigest()
        if plan.get("plan_sha256") != computed_plan:
            raise ContractError("plan internal canonical digest drifted")
        if plan_path.read_bytes() != _canonical_bytes(plan) + b"\n":
            raise ContractError("plan is not canonically serialized")
        try:
            sidecar = sidecar_path.read_text(encoding="ascii")
        except (OSError, UnicodeError) as error:
            raise ContractError(f"cannot read plan sidecar: {error}") from error
        if sidecar != computed_plan + "\n":
            raise ContractError("plan digest sidecar drifted")
        if EXPECTED_PLAN_SHA256 is not None and computed_plan != EXPECTED_PLAN_SHA256:
            raise ContractError("published plan identity drifted")

        if tuple(plan.get("seeds", ())) != EXPECTED_FORMAL_SEEDS:
            raise ContractError("formal seed roster drifted")
        if plan.get("n_jobs") != EXPECTED_JOBS:
            raise ContractError("formal job count drifted")
        architectures = plan.get("architectures")
        if not isinstance(architectures, list) or len(architectures) != EXPECTED_ARCHITECTURES:
            raise ContractError("formal architecture roster drifted")
        train_config = plan.get("train_config")
        if not isinstance(train_config, dict) or train_config.get("seed") != EXPECTED_TEMPLATE_SEED:
            raise ContractError("training template seed drifted")

        analysis_contract = plan.get("analysis_contract")
        if not isinstance(analysis_contract, dict):
            raise ContractError("plan analysis contract is missing")
        if analysis_contract.get("bootstrap_seed") != EXPECTED_BOOTSTRAP_SEED:
            raise ContractError("bootstrap seed drifted")
        if analysis_contract.get("bootstrap_resamples") != EXPECTED_BOOTSTRAP_RESAMPLES:
            raise ContractError("bootstrap resample count drifted")
        if analysis.get("plan_sha256") != computed_plan:
            raise ContractError("analysis references a different plan")

        source_identity = plan.get("source_identity")
        if not isinstance(source_identity, dict) or frozenset(source_identity) != EXPECTED_SOURCE_PATHS:
            raise ContractError("formal source roster drifted")
        for logical, expected in source_identity.items():
            if not isinstance(expected, str) or SHA256_RE.fullmatch(expected) is None:
                raise ContractError(f"invalid formal source digest: {logical}")
            if logical.startswith("eeg_mi/"):
                relative = logical.removeprefix("eeg_mi/")
                archive_path = package_root / relative
                if _sha256(archive_path) != expected:
                    raise ContractError(f"formal source mismatch: {logical}")
            else:
                archive_path = source_root / logical
                if _sha256(archive_path) != expected:
                    raise ContractError(f"formal source mismatch: {logical}")

        analysis_identity = analysis_contract.get("source_identity")
        if not isinstance(analysis_identity, dict) or set(analysis_identity) != {EXPECTED_ANALYSIS_SOURCE}:
            raise ContractError("analysis source roster drifted")
        expected_analysis_hash = analysis_identity[EXPECTED_ANALYSIS_SOURCE]
        if (
            _sha256(package_root / "full_grid_analysis.py") != expected_analysis_hash
        ):
            raise ContractError("analysis source mismatch")

        source_constants = _literal_assignments(
            root / "src/benchmark/full_grid.py",
            {
                "FORMAL_SEEDS",
                "FORMAL_ANALYSIS_BOOTSTRAP_SEED",
                "FORMAL_ANALYSIS_BOOTSTRAP_RESAMPLES",
                "FORMAL_EXPECTED_JOBS",
                "REQUIRED_CUBLAS_WORKSPACE_CONFIG",
            },
        )
        expected_constants = {
            "FORMAL_SEEDS": EXPECTED_FORMAL_SEEDS,
            "FORMAL_ANALYSIS_BOOTSTRAP_SEED": EXPECTED_BOOTSTRAP_SEED,
            "FORMAL_ANALYSIS_BOOTSTRAP_RESAMPLES": EXPECTED_BOOTSTRAP_RESAMPLES,
            "FORMAL_EXPECTED_JOBS": EXPECTED_JOBS,
            "REQUIRED_CUBLAS_WORKSPACE_CONFIG": EXPECTED_CUBLAS_WORKSPACE_CONFIG,
        }
        if source_constants != expected_constants:
            raise ContractError("live formal constants drifted")

        tcformer = plan.get("tcformer_source_identity")
        tcformer_files = tcformer.get("file_identity") if isinstance(tcformer, dict) else None
        if not isinstance(tcformer_files, dict) or not tcformer_files:
            raise ContractError("TCFormer source identity is missing")
        for relative, entry in tcformer_files.items():
            if not isinstance(relative, str) or not isinstance(entry, dict):
                raise ContractError("invalid TCFormer source entry")
            expected = entry.get("sha256")
            path = root / "third_party/TCFormer" / relative
            if not isinstance(expected, str) or _sha256(path) != expected:
                raise ContractError(f"TCFormer source mismatch: {relative}")

        reports = _verify_reports()
        return {
            "schema": "eeg-mi-reproducibility-contract-v3",
            "plan_sha256": computed_plan,
            "formal_seeds": list(EXPECTED_FORMAL_SEEDS),
            "training_template_seed": EXPECTED_TEMPLATE_SEED,
            "bootstrap_seed": EXPECTED_BOOTSTRAP_SEED,
            "bootstrap_resamples": EXPECTED_BOOTSTRAP_RESAMPLES,
            "jobs": EXPECTED_JOBS,
            "architectures": EXPECTED_ARCHITECTURES,
            "cublas_workspace_config": EXPECTED_CUBLAS_WORKSPACE_CONFIG,
            "formal_source_files_verified": len(source_identity) + 1,
            "reports": reports,
            "scientific_replay_guarantee": (
                "fixed seeds, fixed source, fixed data identity, fixed software, and "
                "matching CUDA/GPU identity are required for exact scientific replay"
            ),
            "fresh_run_byte_identity": False,
        }
    finally:
        ROOT = original_root


def main() -> int:
    if len(sys.argv) != 1:
        print("Usage: reproducibility_contract.py", file=sys.stderr)
        return 2
    try:
        payload = validate_release_identity()
    except (ContractError, OSError, UnicodeError, ValueError, TypeError) as error:
        print(f"REPRODUCIBILITY CONTRACT FAILED: {error}", file=sys.stderr)
        return 1
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
