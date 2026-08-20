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
    "source_sha256",
    "release_sha256",
    "reissue_mode",
)

SOURCE_RESULTS_LEDGER_SHA256 = (
    "059710591fb9a5c444a6e79082c291659439b9b1f718596cda23d33966a4ee1e"
)
SOURCE_COMMON_PLAN_SHA256 = (
    "d4dd852fc9e7b80c8c10c54df03d63ce0ae7e7555ee1b7abdee8d79dca28b61a"
)
SOURCE_REISSUE_HASHES = {
    "IMPORT_MANIFEST.csv": "a91f36c68e849e403b2a12db0b671c2d4d5b91040de0b1f43632afa87aa494ae",
    "cardinal_fbms_transfer/independent_verification_lab.json": "7c1302f17f22a4e895c7e42257f5f0a8ab1badf0d59b93ea6b4ee314489f2562",
    "common_grid_v6/analysis/RESULTS.md": "8ad1871dc74899021b29488ed2dc24666e1c636b2e7d70e943bfbe3500bfac09",
    "common_grid_v6/analysis/analysis.json": "007eca1574a44b52c68a2770ed0d56a2301a13c637c0f72fce1c469168ce3961",
    "common_grid_v6/analysis/manifest.json": "960ca912d058a2ad3b60def4116dbba436f8ae9162916cde42a4ab265bf01f5d",
    "common_grid_v6/final_audit.json": "010b44542157efb6ba5a95376198b9b691e9803ce3fe605e77cb5cdb9739d063",
    "common_grid_v6/plan.json": "a04c8f7ff9a8233c0dbdca986e0998a0e6c121e909eaaffc69187d7c0cf93ef8",
    "common_grid_v6/plan.sha256": "8784af9c1a92ab4cf338a8a6b0bea6feebf3596bd2ef6d5679302d6a7c496f95",
    "common_grid_v6/preflight/receipt.json": "a3a3428ba6dc7a5f6167347ece90cd267aebae1d646ef8c3d066e8c5047e96bc",
    "common_grid_v6/preflight/report.json": "39bef71293d33f343eb2d908f0c9bdc6cef06c9e457b98575ca06e86c80af4ef",
    "gauge_gate1/analysis.json": "d89480531b4f37f98aa1faae27f976038ec5e09322368d02a9626bfec9d29eff",
    "gauge_gate1/analysis.sha256": "4b59cb2f59c53c301587165826df82d6c053d45d0227d59a658f59b92a74d343",
    "gauge_gate1/plan.json": "f5f12cdc4200ed60e417e71a2260ecb8a3085d5bb59b7749b6f5854614acfeb2",
    "gauge_gate1/plan.sha256": "753548e45113e8c11a427db2a353f85d9e83674aaea6c264e21db1e0364bf969",
}
UNCHANGED_SCIENTIFIC_CSV_HASHES = {
    "calibration_summary.csv": "e1ba9e5934606ca7e4df91dd67c4a59aec7e32eea041fcaba7a1c8626f1bafeb",
    "complexity_summary.csv": "c34c0360331ebbd27377b20fbd00f6e8438975a0cf1b60efd3e1a95871f148f2",
    "dataset_summary.csv": "0f24934e47c275ffb66b06e78266cb7fc1883f3046da876a5ed50089c09686cf",
    "input_checksum_ledger.csv": "4a176c06ec13659a218ff4b93bfda79f6562e2e4e12e772ce25313614d4c09c2",
    "job_metrics.csv": "fda26e3a545811d1f9d4b4ed2b9885cff0bab3505f6be8e6eb97491bb3355e1d",
    "model_ranking.csv": "adddfd4687e09710b2867252fc873687f48ef073c555a9c0b604af9dd31e4e32",
    "overall_summary.csv": "d45dd54801f4ba9074bba808cd7076bd04c9a5267944ec4f7af274de96a11c6f",
    "resource_timing_summary.csv": "854567e4bb4d1bfbb1d1636011d4a962f198f33c67788d68f19ac1f26791071c",
    "subject_metrics.csv": "58103676e66416a5cf22c6de3d230dccff4411e8c3333d81f4490cb408c06660",
    "subject_seed_metrics.csv": "8db25a6deff6128c87dbd0c1bf87da5b06cc46e1947ffab04ae7f5dd5dc64ae8",
    "tcformer_dataset_context.csv": "f16f4e65d5282334ecd450b8189e42cdff292ac197c3c67e1f6e324c6f5379d9",
    "tcformer_overall_context.csv": "94dd7743bdefafa64a6874c132482c33e5278016369f5c69a06f4b1fc199c391",
    "tcformer_seed_context.csv": "b41e7b5b90acae044d1fffb4139c00522fe1d9a1cd274ffcef22f71ef7dde33e",
    "trial_micro_summary.csv": "9178e6a7e4a5c7a26f40cf39f522b4efb933c3095c194099b5d8d031e428f8c7",
}
JSON_NUMERIC_PROJECTIONS = {
    "cardinal_fbms_transfer/independent_verification_lab.json": (46, "29807ce042e14afb93649e45d4cd542ddeecad95f79b15a5391d5a2a930a8ab4"),
    "common_grid_v6/analysis/analysis.json": (14399, "c947b10a23300a34b229cf72b08e559d483000edd9c2513cbfafd65a348b8eda"),
    "common_grid_v6/analysis/manifest.json": (2, "6804012f5916907e36f9672f288141d6759281839de8ec7cbd1c3bd5e1b1067c"),
    "common_grid_v6/final_audit.json": (18, "c710a29162fa6e7ef8be5f43c217ac72b20a3d402dbb7ddedf82ed6b8ec215dd"),
    "common_grid_v6/plan.json": (23471, "7de4eb2a136c832aa9cd8d1d71d7a06fecead271003c270317ee7b375e11422b"),
    "common_grid_v6/preflight/receipt.json": (8, "379886a41bd9af932f6e82081f84a1ed6e53fd94266c4bf3c47ae7070997713a"),
    "common_grid_v6/preflight/report.json": (729, "04b90203118b33134d8ec48fccca7b5529121be35a59bb2927c68fd4c0da3db6"),
    "gauge_gate1/analysis.json": (63, "da1434823b2373df0667b35e1fc607952002b5f356aba169a829ee72a76b402f"),
    "gauge_gate1/plan.json": (386, "c8cc72a96a10203972a668625f330148b33c9e722efccdf3be8020d76ae7d7f8"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def numeric_projection(value: object) -> tuple[int, str]:
    """Hash ordered, type-tagged numeric leaves independently of string metadata."""

    rows: list[list[object]] = []

    def walk(item: object) -> None:
        if isinstance(item, dict):
            for child in item.values():
                walk(child)
        elif isinstance(item, list):
            for child in item:
                walk(child)
        elif isinstance(item, bool):
            rows.append(["bool", item])
        elif isinstance(item, int):
            rows.append(["int", item])
        elif isinstance(item, float):
            rows.append(["float", repr(item)])

    walk(value)
    payload = json.dumps(
        rows, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return len(rows), hashlib.sha256(payload).hexdigest()


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
        source_sha256 = row["source_sha256"]
        release_sha256 = row["release_sha256"]
        mode = row["reissue_mode"]
        if not HEX64.fullmatch(source_sha256) or not HEX64.fullmatch(release_sha256):
            raise ValueError(f"invalid source/release checksum for {destination}")
        relative = destination.as_posix()
        scientific_prefix = "common_grid_v6/analysis/"
        scientific_name = (
            relative.removeprefix(scientific_prefix)
            if relative.startswith(scientific_prefix)
            else ""
        )
        if relative in SOURCE_REISSUE_HASHES:
            expected_source = SOURCE_REISSUE_HASHES[relative]
            expected_mode = "metadata_reissue"
        elif scientific_name in UNCHANGED_SCIENTIFIC_CSV_HASHES:
            expected_source = UNCHANGED_SCIENTIFIC_CSV_HASHES[scientific_name]
            expected_mode = "byte_identical"
        else:
            raise ValueError(f"unclassified imported artifact: {destination}")
        if source_sha256 != expected_source or mode != expected_mode:
            raise ValueError(f"incorrect reissue lineage for {destination}")
        observed = sha256(ROOT / Path(*destination.parts))
        if observed != release_sha256:
            raise ValueError(f"released bytes differ for {destination}")
        if mode == "byte_identical" and source_sha256 != release_sha256:
            raise ValueError(f"byte-identical import changed: {destination}")


def verify_reissue_invariance() -> None:
    """Verify the reissue lineage and exact preservation of scientific values."""

    provenance_path = ROOT / "REISSUE_PROVENANCE.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("schema") != "eeg-mi-sanitized-results-reissue-v1":
        raise ValueError("unexpected result-reissue provenance schema")
    if provenance.get("source_identity") != {
        "results_checksum_ledger_sha256": SOURCE_RESULTS_LEDGER_SHA256,
        "common_plan_sha256": SOURCE_COMMON_PLAN_SHA256,
    }:
        raise ValueError("source result-bundle identity changed")

    expected_csvs = dict(sorted(UNCHANGED_SCIENTIFIC_CSV_HASHES.items()))
    if provenance.get("unchanged_scientific_csvs") != expected_csvs:
        raise ValueError("scientific CSV preservation declaration changed")
    analysis = ROOT / "common_grid_v6" / "analysis"
    for name, expected in expected_csvs.items():
        if sha256(analysis / name) != expected:
            raise ValueError(f"scientific CSV is not byte-identical: {name}")

    expected_numeric = {
        relative: {"leaf_count": count, "sha256": digest}
        for relative, (count, digest) in sorted(JSON_NUMERIC_PROJECTIONS.items())
    }
    if provenance.get("json_numeric_projections") != expected_numeric:
        raise ValueError("JSON numeric-preservation declaration changed")
    for relative, expected in JSON_NUMERIC_PROJECTIONS.items():
        value = json.loads((ROOT / relative).read_text(encoding="utf-8"))
        if numeric_projection(value) != expected:
            raise ValueError(f"JSON numeric leaves changed: {relative}")

    changed = provenance.get("changed_artifacts")
    if not isinstance(changed, dict) or set(changed) != set(SOURCE_REISSUE_HASHES):
        raise ValueError("metadata-reissue artifact roster changed")
    for relative, source_digest in SOURCE_REISSUE_HASHES.items():
        row = changed.get(relative)
        if not isinstance(row, dict) or set(row) != {
            "source_sha256",
            "reissued_sha256",
        }:
            raise ValueError(f"invalid metadata-reissue row: {relative}")
        if row["source_sha256"] != source_digest:
            raise ValueError(f"source checksum changed in reissue row: {relative}")
        if row["reissued_sha256"] != sha256(ROOT / relative):
            raise ValueError(f"reissued checksum changed: {relative}")

    common = ROOT / "common_grid_v6"
    plan_path = common / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    detached_plan = dict(plan)
    detached_plan.pop("plan_sha256", None)
    plan_digest = hashlib.sha256(canonical_bytes(detached_plan)).hexdigest()
    if (
        plan.get("plan_sha256") != plan_digest
        or (common / "plan.sha256").read_text(encoding="ascii") != plan_digest + "\n"
    ):
        raise ValueError("reissued common-grid plan digest is invalid")

    analysis_contract = plan.get("analysis_contract")
    if not isinstance(analysis_contract, dict):
        raise ValueError("reissued analysis contract is invalid")
    analysis_contract_digest = hashlib.sha256(
        canonical_bytes(analysis_contract)
    ).hexdigest()
    analysis_json = json.loads((analysis / "analysis.json").read_text(encoding="utf-8"))
    audit = json.loads((common / "final_audit.json").read_text(encoding="utf-8"))
    manifest = json.loads((analysis / "manifest.json").read_text(encoding="utf-8"))
    if (
        analysis_json.get("analysis_contract") != analysis_contract
        or analysis_json.get("analysis_contract_sha256") != analysis_contract_digest
        or manifest.get("analysis_contract_sha256") != analysis_contract_digest
        or manifest.get("analysis_source_identity")
        != analysis_contract.get("source_identity")
        or analysis_json.get("audit") != audit
    ):
        raise ValueError("reissued analysis identity is not closed")

    preflight = common / "preflight"
    report_path = preflight / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    receipt = json.loads((preflight / "receipt.json").read_text(encoding="utf-8"))
    report_digest = sha256(report_path)
    gpu_receipt = report.get("gpu_lease_receipt")
    if not isinstance(gpu_receipt, dict) or not isinstance(gpu_receipt.get("lease"), dict):
        raise ValueError("reissued preflight GPU receipt is invalid")
    lease_digest = hashlib.sha256(canonical_bytes(gpu_receipt["lease"])).hexdigest()
    if (
        gpu_receipt.get("lease_sha256") != lease_digest
        or receipt.get("publication_gpu_lease_receipt") != gpu_receipt
        or receipt.get("report_sha256") != report_digest
    ):
        raise ValueError("reissued preflight identity is not closed")

    declared_reissue = provenance.get("reissued_identity")
    expected_reissue = {
        "common_plan_sha256": plan_digest,
        "common_plan_file_sha256": sha256(plan_path),
        "analysis_manifest_sha256": sha256(analysis / "manifest.json"),
    }
    if declared_reissue != expected_reissue:
        raise ValueError("reissued result identity declaration changed")

    forbidden = bytes((105, 101, 101, 101))
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT).as_posix().encode("utf-8").lower()
        if forbidden in relative:
            raise ValueError(f"reserved token remains in result path: {path}")
        if path.is_file() and forbidden in path.read_bytes().lower():
            raise ValueError(f"reserved token remains in result bytes: {path}")


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
    verify_reissue_invariance()
    verify_scientific_closure()
    print("PASS: results bundle checksums, imports, sealed closures, and boundaries verified")


if __name__ == "__main__":
    main()
