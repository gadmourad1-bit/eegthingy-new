#!/usr/bin/env python3
"""Build deterministic publication PDFs from the sealed benchmark bundle.

The builder deliberately uses only the Python standard library and ReportLab.
It validates the checksum inventory, sealed analysis manifest, exact audit,
and cross-table scientific identities before creating any output.  It does
not open EEG, predictions, caches, or external result roots.

Visual rendering/inspection is a separate release step.  A successful build
proves input and structural checks, not that every rendered page has passed
human visual QA.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import textwrap
import unicodedata
from typing import Any, Iterable, Mapping, Sequence

try:
    from reportlab import rl_config
    from reportlab.graphics.shapes import Circle, Drawing, Line, Rect, String
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.lib.pagesizes import LETTER, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.pdfgen import canvas
    from reportlab.platypus import (
        BaseDocTemplate,
        CondPageBreak,
        Frame,
        KeepTogether,
        ListFlowable,
        ListItem,
        LongTable,
        PageBreak,
        PageTemplate,
        Paragraph,
        Preformatted,
        Spacer,
        Table,
        TableStyle,
    )
    from reportlab.platypus.tableofcontents import TableOfContents
except ImportError as error:  # pragma: no cover - exercised in runtime setup
    raise SystemExit(
        "ReportLab is required. Build the isolated docs environment with "
        "'scripts/reproduce.sh setup docs', then run this script through it."
    ) from error


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ANALYSIS = RESULTS / "common_grid_v6" / "analysis"
DEFAULT_OUTPUT = ROOT / "output" / "pdf"

EXPECTED_DATASETS = (
    "local_exp4",
    "bnci2014_001",
    "bnci2014_004",
    "cho2017",
    "physionet_mi",
)
EXPECTED_SEEDS = (7, 17, 27, 37, 47)
EXPECTED_MODELS = 43
EXPECTED_SPLITS = 448
EXPECTED_JOBS = 96_320
EXPECTED_PREFLIGHT_CHECKS = 172
EXPECTED_PARTICIPANTS = 132
EXPECTED_ANALYSIS_MANIFEST_SHA256 = (
    "960ca912d058a2ad3b60def4116dbba436f8ae9162916cde42a4ab265bf01f5d"
)
EXPECTED_RESULTS_INVENTORY_SHA256 = (
    "059710591fb9a5c444a6e79082c291659439b9b1f718596cda23d33966a4ee1e"
)
TCFORMER = "tcformer"

PDF_FILENAMES = (
    "Benchmark_Results_and_Statistics.pdf",
    "Methodology_and_Architectures.pdf",
    "Reproducibility_Handbook.pdf",
)

NAVY = colors.HexColor("#102A43")
BLUE = colors.HexColor("#2F6B9A")
TEAL = colors.HexColor("#2A9D8F")
GOLD = colors.HexColor("#D69E2E")
RED = colors.HexColor("#C44536")
INK = colors.HexColor("#263238")
MUTED = colors.HexColor("#5F6C7B")
PALE_BLUE = colors.HexColor("#EAF2F8")
PALE_TEAL = colors.HexColor("#E8F5F2")
PALE_GOLD = colors.HexColor("#FFF7DF")
PALE_RED = colors.HexColor("#FCEDEA")
GRID = colors.HexColor("#CBD5E1")
ROW_ALT = colors.HexColor("#F7FAFC")
WHITE = colors.white


class ReportInputError(RuntimeError):
    """A required sealed input is absent or inconsistent."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise ReportInputError(f"required file is missing: {path.relative_to(ROOT)}")
    if path.stat().st_size <= 0:
        raise ReportInputError(f"required file is empty: {path.relative_to(ROOT)}")
    return path


def load_json(path: Path) -> dict[str, Any]:
    require_file(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReportInputError(f"invalid JSON: {path.relative_to(ROOT)}") from error
    if not isinstance(value, dict):
        raise ReportInputError(f"JSON root must be an object: {path.relative_to(ROOT)}")
    return value


def read_csv_exact(path: Path, columns: Sequence[str]) -> list[dict[str, str]]:
    require_file(path)
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(columns):
            raise ReportInputError(
                f"unexpected CSV columns in {path.relative_to(ROOT)}: "
                f"{reader.fieldnames!r}"
            )
        rows = list(reader)
    if not rows:
        raise ReportInputError(f"CSV has no data rows: {path.relative_to(ROOT)}")
    return rows


def exact_int(value: str | int, *, label: str) -> int:
    if isinstance(value, bool):
        raise ReportInputError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ReportInputError(f"{label} must be an integer") from error
    if str(result) != str(value) and not isinstance(value, int):
        raise ReportInputError(f"{label} is not canonically encoded")
    return result


def finite_float(
    value: str | float | int,
    *,
    label: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise ReportInputError(f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ReportInputError(f"{label} must be numeric") from error
    if not math.isfinite(result):
        raise ReportInputError(f"{label} must be finite")
    if minimum is not None and result < minimum:
        raise ReportInputError(f"{label} is below {minimum}")
    if maximum is not None and result > maximum:
        raise ReportInputError(f"{label} is above {maximum}")
    return result


def assert_close(left: float, right: float, *, label: str) -> None:
    if not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12):
        raise ReportInputError(f"inconsistent value for {label}: {left!r} != {right!r}")


def validate_results_checksums() -> None:
    sums_path = require_file(RESULTS / "SHA256SUMS")
    if sha256_file(sums_path) != EXPECTED_RESULTS_INVENTORY_SHA256:
        raise ReportInputError("results checksum inventory is not the sealed release inventory")
    declared: dict[str, str] = {}
    for line_number, line in enumerate(
        sums_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or re.fullmatch(r"[0-9a-f]{64}", parts[0]) is None:
            raise ReportInputError(f"malformed results/SHA256SUMS line {line_number}")
        relative = parts[1].lstrip("*")
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts or relative in declared:
            raise ReportInputError(f"unsafe or duplicate checksum path: {relative!r}")
        declared[relative] = parts[0]

    actual = {
        path.relative_to(RESULTS).as_posix()
        for path in RESULTS.rglob("*")
        if path.is_file() and path != sums_path
    }
    if set(declared) != actual:
        raise ReportInputError(
            "results checksum inventory is not exact: "
            f"missing={sorted(actual - set(declared))}, "
            f"extra={sorted(set(declared) - actual)}"
        )
    for relative, expected in sorted(declared.items()):
        path = RESULTS / relative
        if sha256_file(path) != expected:
            raise ReportInputError(f"results checksum mismatch: {relative}")


def validate_analysis_manifest() -> dict[str, Any]:
    manifest_path = ANALYSIS / "manifest.json"
    if sha256_file(require_file(manifest_path)) != EXPECTED_ANALYSIS_MANIFEST_SHA256:
        raise ReportInputError("analysis manifest identity differs from the sealed publication")
    manifest = load_json(manifest_path)
    if manifest.get("schema") != "ieee-mi-common-grid-analysis-manifest-v4":
        raise ReportInputError("unexpected common analysis manifest schema")
    files = manifest.get("files")
    if not isinstance(files, dict) or len(files) != 16:
        raise ReportInputError("analysis manifest must hash exactly 16 sibling files")
    actual = {
        path.name for path in ANALYSIS.iterdir() if path.is_file() and path.name != "manifest.json"
    }
    if set(files) != actual:
        raise ReportInputError(
            "sealed analysis roster differs from manifest: "
            f"missing={sorted(actual - set(files))}, "
            f"extra={sorted(set(files) - actual)}"
        )
    for name, expected in sorted(files.items()):
        if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise ReportInputError(f"invalid analysis digest for {name}")
        if sha256_file(ANALYSIS / name) != expected:
            raise ReportInputError(f"sealed analysis checksum mismatch: {name}")
    return manifest


@dataclass(frozen=True)
class ValidatedInputs:
    plan: dict[str, Any]
    audit: dict[str, Any]
    manifest: dict[str, Any]
    models: tuple[str, ...]
    ranking: tuple[dict[str, str], ...]
    dataset_summary: tuple[dict[str, str], ...]
    overall_summary: tuple[dict[str, str], ...]
    balanced_wide: tuple[dict[str, str], ...]
    accuracy_wide: tuple[dict[str, str], ...]
    top_ten: tuple[dict[str, str], ...]
    dataset_winners: tuple[dict[str, str], ...]
    tcformer_context: tuple[dict[str, str], ...]
    calibration: tuple[dict[str, str], ...]
    complexity: tuple[dict[str, str], ...]
    protocols: tuple[dict[str, str], ...]
    gauge: dict[str, Any]
    cardinal_fbms: dict[str, Any]

    @property
    def leader(self) -> str:
        return self.ranking[0]["model"]

    @property
    def leader_ba(self) -> float:
        return float(self.ranking[0]["value"])


RANKING_COLUMNS = (
    "rank",
    "model",
    "metric",
    "value",
    "tie_break",
    "status",
)
DATASET_COLUMNS = (
    "dataset",
    "model",
    "subjects_averaged",
    "aggregation",
    "accuracy",
    "accuracy_defined_count",
    "balanced_accuracy",
    "balanced_accuracy_defined_count",
    "chance_normalized_balanced_accuracy",
    "chance_normalized_balanced_accuracy_defined_count",
    "macro_f1",
    "macro_f1_defined_count",
    "cohen_kappa",
    "cohen_kappa_defined_count",
    "ovr_macro_auroc",
    "ovr_macro_auroc_defined_count",
    "nll",
    "nll_defined_count",
    "multiclass_brier",
    "multiclass_brier_defined_count",
    "ece",
    "ece_defined_count",
)
OVERALL_COLUMNS = (
    "model",
    "datasets_equal_weighted",
    "aggregation",
    "accuracy",
    "accuracy_defined_count",
    "balanced_accuracy",
    "balanced_accuracy_defined_count",
    "chance_normalized_balanced_accuracy",
    "chance_normalized_balanced_accuracy_defined_count",
    "macro_f1",
    "macro_f1_defined_count",
    "cohen_kappa",
    "cohen_kappa_defined_count",
    "ovr_macro_auroc",
    "ovr_macro_auroc_defined_count",
    "nll",
    "nll_defined_count",
    "multiclass_brier",
    "multiclass_brier_defined_count",
    "ece",
    "ece_defined_count",
)
BALANCED_WIDE_COLUMNS = (
    "rank",
    "model",
    "local_exp4_balanced_accuracy",
    "bnci2014_001_balanced_accuracy",
    "bnci2014_004_balanced_accuracy",
    "cho2017_balanced_accuracy",
    "physionet_mi_balanced_accuracy",
    "equal_dataset_macro_balanced_accuracy",
    "status",
)
ACCURACY_WIDE_COLUMNS = (
    "descriptive_accuracy_rank",
    "primary_balanced_accuracy_rank",
    "model",
    "local_exp4_accuracy",
    "bnci2014_001_accuracy",
    "bnci2014_004_accuracy",
    "cho2017_accuracy",
    "physionet_mi_accuracy",
    "equal_dataset_macro_accuracy",
    "status",
)
TOP_COLUMNS = (
    "rank",
    "model",
    "equal_dataset_macro_balanced_accuracy",
    "status",
)
WINNER_COLUMNS = (
    "dataset",
    "model",
    "balanced_accuracy",
    "subjects_averaged",
    "aggregation",
)
TCFORMER_CONTEXT_COLUMNS = (
    "model",
    "comparator",
    "datasets",
    "subjects_total",
    "equal_dataset_macro_balanced_accuracy_difference",
    "descriptive_fixed_suite_bootstrap_interval95_low",
    "descriptive_fixed_suite_bootstrap_interval95_high",
    "fixed_suite_bootstrap_resamples",
    "fixed_suite_bootstrap_seed",
    "descriptive_dataset_superpopulation_interval95_low",
    "descriptive_dataset_superpopulation_interval95_high",
    "dataset_superpopulation_bootstrap_resamples",
    "dataset_superpopulation_bootstrap_seed",
    "status",
)
CALIBRATION_COLUMNS = (
    "level",
    "dataset",
    "model",
    "nll",
    "multiclass_brier",
    "ece",
    "ece_bins",
)
COMPLEXITY_COLUMNS = (
    "model",
    "parameter_count_median",
    "parameter_count_min",
    "parameter_count_max",
    "inference_ms_per_trial_median",
    "cuda_peak_memory_bytes_median",
    "job_total_seconds_median",
)
PROTOCOL_COLUMNS = (
    "track_id",
    "protocol_label",
    "status",
    "completed_units",
    "planned_units",
    "datasets",
    "primary_model_or_condition",
    "primary_metric",
    "primary_value",
    "comparison_boundary",
    "artifact",
)


def validate_inputs() -> ValidatedInputs:
    required_docs = (
        ROOT / "docs" / "METHODOLOGY_AND_ARCHITECTURES.md",
        ROOT / "docs" / "REPRODUCIBILITY.md",
        ROOT / "docs" / "DATA_ACCESS.md",
        ROOT / "docs" / "ETHICS_AND_PRIVACY.md",
        ROOT / "docs" / "PROTOCOL_BOUNDARIES.md",
    )
    for path in required_docs:
        require_file(path)

    validate_results_checksums()
    manifest = validate_analysis_manifest()
    plan = load_json(RESULTS / "common_grid_v6" / "plan.json")
    audit = load_json(RESULTS / "common_grid_v6" / "final_audit.json")

    plan_sidecar = require_file(RESULTS / "common_grid_v6" / "plan.sha256").read_text(
        encoding="ascii"
    ).strip()
    if plan.get("plan_sha256") != plan_sidecar or manifest.get("plan_sha256") != plan_sidecar:
        raise ReportInputError("plan identity differs across plan, sidecar, and analysis")
    if audit.get("plan_sha256") != plan_sidecar:
        raise ReportInputError("final audit is not bound to the distributed plan")
    if plan.get("schema") != "ieee-mi-score-blind-common-grid-plan-v5":
        raise ReportInputError("unexpected common-grid plan schema")
    if tuple(plan.get("dataset_order", ())) != EXPECTED_DATASETS:
        raise ReportInputError("plan dataset order is not the fixed five-dataset suite")
    if tuple(plan.get("seeds", ())) != EXPECTED_SEEDS:
        raise ReportInputError("plan seed roster is not frozen")
    models_value = plan.get("architectures")
    if not isinstance(models_value, list) or len(models_value) != EXPECTED_MODELS:
        raise ReportInputError("plan must contain exactly 43 architectures")
    models = tuple(models_value)
    if any(not isinstance(model, str) or not model for model in models) or len(set(models)) != len(models):
        raise ReportInputError("plan architecture roster is malformed or duplicated")
    if plan.get("common_architecture_count") != EXPECTED_MODELS or plan.get("n_jobs") != EXPECTED_JOBS:
        raise ReportInputError("plan architecture/job cardinality is inconsistent")

    exact_audit = {
        "expected": EXPECTED_JOBS,
        "complete": EXPECTED_JOBS,
        "missing": 0,
        "extra": 0,
        "failed_jobs": 0,
        "live_claims": 0,
        "stale_claims": 0,
        "partials": 0,
        "unexpected_root_entries": 0,
        "unsafe_paths": 0,
        "invalid_forensic_artifacts": 0,
    }
    if audit.get("schema") != "ieee-mi-score-blind-common-grid-audit-v3":
        raise ReportInputError("unexpected common-grid audit schema")
    if audit.get("exact_cartesian_complete") is not True or audit.get("preflight_attestation_valid") is not True:
        raise ReportInputError("common-grid final audit is not exact and preflight-valid")
    for key, expected in exact_audit.items():
        if audit.get(key) != expected:
            raise ReportInputError(f"final audit field {key!r} is not {expected!r}")
    if audit.get("quarantine_artifacts") != 6 or audit.get("resolved_forensic_artifacts") != 6:
        raise ReportInputError("expected six resolved forensic claim artifacts")

    preflight = load_json(RESULTS / "common_grid_v6" / "preflight" / "report.json")
    preflight_path = RESULTS / "common_grid_v6" / "preflight" / "report.json"
    receipt = load_json(RESULTS / "common_grid_v6" / "preflight" / "receipt.json")
    checks = preflight.get("results")
    if (
        preflight.get("schema") != "ieee-mi-common-track-cuda-preflight-v3"
        or preflight.get("plan_sha256") != plan_sidecar
        or preflight.get("score_blind") is not True
        or preflight.get("accuracy_computed") is not False
        or preflight.get("test_splits_opened") is not False
        or preflight.get("label_or_split_members_opened") is not False
        or preflight.get("trial_feature_members_opened") is not False
    ):
        raise ReportInputError("preflight report does not retain the frozen score-blind boundary")
    if (
        preflight.get("expected_checks") != EXPECTED_PREFLIGHT_CHECKS
        or preflight.get("completed_checks") != EXPECTED_PREFLIGHT_CHECKS
        or not isinstance(checks, list)
        or len(checks) != EXPECTED_PREFLIGHT_CHECKS
    ):
        raise ReportInputError("preflight report must contain exactly 172 completed checks")
    if any(
        not isinstance(row, dict)
        or row.get("forward_backward_optimizer_step") != "passed"
        for row in checks
    ):
        raise ReportInputError("every preflight forward/backward optimizer check must pass")
    if (
        receipt.get("schema") != "ieee-mi-common-track-cuda-preflight-receipt-v1"
        or receipt.get("report_schema") != preflight.get("schema")
        or receipt.get("plan_sha256") != plan_sidecar
        or receipt.get("expected_checks") != EXPECTED_PREFLIGHT_CHECKS
        or receipt.get("completed_checks") != EXPECTED_PREFLIGHT_CHECKS
        or receipt.get("report_sha256") != sha256_file(preflight_path)
    ):
        raise ReportInputError("preflight receipt is missing or inconsistent with its report")

    ranking = read_csv_exact(ANALYSIS / "model_ranking.csv", RANKING_COLUMNS)
    dataset_summary = read_csv_exact(ANALYSIS / "dataset_summary.csv", DATASET_COLUMNS)
    overall_summary = read_csv_exact(ANALYSIS / "overall_summary.csv", OVERALL_COLUMNS)
    balanced_wide = read_csv_exact(
        RESULTS / "common_grid_v6" / "all_models_balanced_accuracy.csv",
        BALANCED_WIDE_COLUMNS,
    )
    accuracy_wide = read_csv_exact(
        RESULTS / "common_grid_v6" / "all_models_accuracy.csv", ACCURACY_WIDE_COLUMNS
    )
    top_ten = read_csv_exact(
        RESULTS / "common_grid_v6" / "top_10_balanced_accuracy.csv", TOP_COLUMNS
    )
    dataset_winners = read_csv_exact(
        RESULTS / "common_grid_v6" / "per_dataset_balanced_accuracy_winners.csv",
        WINNER_COLUMNS,
    )
    tcformer_context = read_csv_exact(
        ANALYSIS / "tcformer_overall_context.csv", TCFORMER_CONTEXT_COLUMNS
    )
    calibration_all = read_csv_exact(ANALYSIS / "calibration_summary.csv", CALIBRATION_COLUMNS)
    calibration = [
        row
        for row in calibration_all
        if row["level"] == "equal_dataset_macro" and row["dataset"] == "ALL_DATASETS"
    ]
    complexity = read_csv_exact(ANALYSIS / "complexity_summary.csv", COMPLEXITY_COLUMNS)
    protocols = read_csv_exact(RESULTS / "protocol_summary.csv", PROTOCOL_COLUMNS)

    if len(ranking) != EXPECTED_MODELS or tuple(row["model"] for row in ranking) != tuple(
        row["model"] for row in balanced_wide
    ):
        raise ReportInputError("ranking and balanced-accuracy view do not share exact order")
    if {row["model"] for row in ranking} != set(models):
        raise ReportInputError("ranking model roster differs from the plan")
    for index, row in enumerate(ranking, 1):
        if exact_int(row["rank"], label="ranking rank") != index:
            raise ReportInputError("model ranking is not consecutive 1..43")
        if row["metric"] != "equal_dataset_macro_balanced_accuracy":
            raise ReportInputError("model ranking uses an unexpected primary metric")
        value = finite_float(row["value"], label="ranked balanced accuracy", minimum=0.0, maximum=1.0)
        wide = balanced_wide[index - 1]
        if exact_int(wide["rank"], label="balanced view rank") != index:
            raise ReportInputError("balanced view rank differs from sealed ranking")
        assert_close(
            value,
            finite_float(
                wide["equal_dataset_macro_balanced_accuracy"],
                label="balanced view overall",
                minimum=0.0,
                maximum=1.0,
            ),
            label=f"overall balanced accuracy for {row['model']}",
        )

    model_set = set(models)
    if len(overall_summary) != EXPECTED_MODELS or {row["model"] for row in overall_summary} != model_set:
        raise ReportInputError("overall summary must contain the exact 43-model roster")
    overall_by_model = {row["model"]: row for row in overall_summary}
    for row in ranking:
        overall = overall_by_model[row["model"]]
        if exact_int(overall["datasets_equal_weighted"], label="dataset count") != len(EXPECTED_DATASETS):
            raise ReportInputError("overall summary does not equal-weight five datasets")
        assert_close(
            float(row["value"]),
            finite_float(overall["balanced_accuracy"], label="overall BA", minimum=0.0, maximum=1.0),
            label=f"sealed ranking/summary BA for {row['model']}",
        )

    expected_dataset_pairs = {(dataset, model) for dataset in EXPECTED_DATASETS for model in models}
    observed_dataset_pairs = {(row["dataset"], row["model"]) for row in dataset_summary}
    if len(dataset_summary) != len(expected_dataset_pairs) or observed_dataset_pairs != expected_dataset_pairs:
        raise ReportInputError("dataset summary is not the exact 5x43 product")
    dataset_by_pair = {(row["dataset"], row["model"]): row for row in dataset_summary}
    for row in balanced_wide:
        model = row["model"]
        for dataset in EXPECTED_DATASETS:
            assert_close(
                finite_float(
                    row[f"{dataset}_balanced_accuracy"],
                    label=f"wide BA {dataset}/{model}",
                    minimum=0.0,
                    maximum=1.0,
                ),
                finite_float(
                    dataset_by_pair[(dataset, model)]["balanced_accuracy"],
                    label=f"sealed BA {dataset}/{model}",
                    minimum=0.0,
                    maximum=1.0,
                ),
                label=f"dataset balanced accuracy {dataset}/{model}",
            )

    if len(accuracy_wide) != EXPECTED_MODELS or {row["model"] for row in accuracy_wide} != model_set:
        raise ReportInputError("accuracy view must contain the exact 43-model roster")
    accuracy_ranks = {exact_int(row["descriptive_accuracy_rank"], label="accuracy rank") for row in accuracy_wide}
    if accuracy_ranks != set(range(1, EXPECTED_MODELS + 1)):
        raise ReportInputError("accuracy ranks are not a permutation of 1..43")
    for row in accuracy_wide:
        model = row["model"]
        assert_close(
            finite_float(row["equal_dataset_macro_accuracy"], label="wide accuracy", minimum=0.0, maximum=1.0),
            finite_float(overall_by_model[model]["accuracy"], label="sealed accuracy", minimum=0.0, maximum=1.0),
            label=f"overall accuracy for {model}",
        )
        for dataset in EXPECTED_DATASETS:
            assert_close(
                finite_float(row[f"{dataset}_accuracy"], label="wide dataset accuracy", minimum=0.0, maximum=1.0),
                finite_float(dataset_by_pair[(dataset, model)]["accuracy"], label="sealed dataset accuracy", minimum=0.0, maximum=1.0),
                label=f"dataset accuracy {dataset}/{model}",
            )

    if len(top_ten) != 10:
        raise ReportInputError("top-ten table must contain exactly ten rows")
    for expected, observed in zip(ranking[:10], top_ten, strict=True):
        if expected["rank"] != observed["rank"] or expected["model"] != observed["model"]:
            raise ReportInputError("top-ten view differs from sealed ranking")
        assert_close(
            float(expected["value"]),
            float(observed["equal_dataset_macro_balanced_accuracy"]),
            label=f"top-ten BA for {expected['model']}",
        )

    if len(dataset_winners) != len(EXPECTED_DATASETS) or tuple(
        row["dataset"] for row in dataset_winners
    ) != EXPECTED_DATASETS:
        raise ReportInputError("per-dataset winner table has an unexpected order or count")
    for winner in dataset_winners:
        dataset = winner["dataset"]
        sealed_rows = [row for row in dataset_summary if row["dataset"] == dataset]
        actual_winner = max(sealed_rows, key=lambda row: float(row["balanced_accuracy"]))
        if winner["model"] != actual_winner["model"]:
            raise ReportInputError(f"per-dataset winner is wrong for {dataset}")
        assert_close(
            float(winner["balanced_accuracy"]),
            float(actual_winner["balanced_accuracy"]),
            label=f"winner BA for {dataset}",
        )

    if len(calibration) != EXPECTED_MODELS or {row["model"] for row in calibration} != model_set:
        raise ReportInputError("equal-dataset calibration table must contain 43 models")
    for row in calibration:
        if exact_int(row["ece_bins"], label="ECE bins") != 15:
            raise ReportInputError("calibration summary must use exactly 15 ECE bins")
        for key in ("nll", "multiclass_brier", "ece"):
            finite_float(row[key], label=f"{key}/{row['model']}", minimum=0.0)

    if len(complexity) != EXPECTED_MODELS or {row["model"] for row in complexity} != model_set:
        raise ReportInputError("complexity table must contain the exact 43-model roster")
    for row in complexity:
        for key in COMPLEXITY_COLUMNS[1:]:
            finite_float(row[key], label=f"{key}/{row['model']}", minimum=0.0)

    if len(tcformer_context) != EXPECTED_MODELS - 1 or {
        row["model"] for row in tcformer_context
    } != model_set - {TCFORMER}:
        raise ReportInputError("TCFormer context table must cover all 42 other models")
    for row in tcformer_context:
        if row["comparator"] != TCFORMER:
            raise ReportInputError("unexpected uncertainty comparator")
        if exact_int(row["datasets"], label="context datasets") != 5 or exact_int(
            row["subjects_total"], label="context participants"
        ) != EXPECTED_PARTICIPANTS:
            raise ReportInputError("TCFormer context has unexpected scope")
        if exact_int(row["fixed_suite_bootstrap_resamples"], label="bootstrap resamples") != 100_000:
            raise ReportInputError("fixed-suite bootstrap must use 100,000 resamples")
        finite_float(row["descriptive_fixed_suite_bootstrap_interval95_low"], label="CI low")
        finite_float(row["descriptive_fixed_suite_bootstrap_interval95_high"], label="CI high")

    protocol_by_id = {row["track_id"]: row for row in protocols}
    if len(protocol_by_id) != len(protocols):
        raise ReportInputError("protocol summary contains duplicate track IDs")
    expected_protocols: dict[str, tuple[int, int | None, str]] = {
        "common_grid_v6": (EXPECTED_JOBS, EXPECTED_JOBS, "complete_exact_audit_pass"),
        "gauge_gate1": (17, 17, "complete_gate_failed"),
        "cardinal_fbms_transfer": (8_675, 8_675, "complete_post_outcome_verification_pass"),
        "author_recipe": (0, 4_480, "pending_unrun"),
        "geoadapt_v2": (0, 6_495, "pending_unrun"),
        "local_procedures": (0, 8_780, "pending_unrun"),
        "deterministic_controls": (0, 1_344, "pending_unrun"),
        "hemiq_v2": (0, 45, "pending_unrun"),
        "cardinal_fbc_transfer": (364, 6_940, "incomplete_not_imported"),
        "gauge_gate2": (0, None, "not_run_by_prespecified_rule"),
    }
    if set(protocol_by_id) != set(expected_protocols):
        raise ReportInputError("protocol summary track roster is unexpected")
    for key, (complete, planned, status) in expected_protocols.items():
        row = protocol_by_id[key]
        planned_is_valid = (
            row["planned_units"] == ""
            if planned is None
            else exact_int(row["planned_units"], label=f"{key} planned") == planned
        )
        if (
            exact_int(row["completed_units"], label=f"{key} completed") != complete
            or not planned_is_valid
            or row["status"] != status
        ):
            raise ReportInputError(f"protocol count differs for {key}")

    gauge = load_json(RESULTS / "gauge_gate1" / "analysis.json")
    if gauge.get("passed") is not False or gauge.get("failure_action") != "kill_candidate_before_disjoint_gate":
        raise ReportInputError("Gauge artifact is not the frozen failed Gate-1 decision")
    if gauge.get("job_count") != 17 or gauge.get("nonnegative_dataset_delta_count") != 2:
        raise ReportInputError("Gauge Gate-1 scope/result is inconsistent")
    gauge_scores = gauge.get("model_equal_dataset_balanced_accuracy")
    gauge_candidate = gauge.get("candidate")
    if (
        gauge.get("schema") != "ieee-mi-gauge-gate1-decision-v1"
        or gauge.get("opened_development_only") is not True
        or gauge.get("confirmation_evidence") is not False
        or gauge.get("dataset_order") != list(EXPECTED_DATASETS)
        or not isinstance(gauge_candidate, str)
        or not isinstance(gauge_scores, dict)
        or gauge_candidate not in gauge_scores
    ):
        raise ReportInputError("Gauge Gate-1 decision boundary or score payload is malformed")
    gauge_score = finite_float(
        gauge_scores[gauge_candidate], label="Gauge Gate-1 candidate BA", minimum=0.0, maximum=1.0
    )
    gauge_protocol = protocol_by_id["gauge_gate1"]
    if gauge_protocol["primary_model_or_condition"] != gauge_candidate:
        raise ReportInputError("Gauge protocol summary names a different candidate")
    assert_close(
        gauge_score,
        finite_float(gauge_protocol["primary_value"], label="Gauge protocol BA", minimum=0.0, maximum=1.0),
        label="Gauge artifact/protocol summary BA",
    )

    cardinal_fbms = load_json(
        RESULTS / "cardinal_fbms_transfer" / "independent_verification_lab.json"
    )
    if cardinal_fbms.get("status") != "PASS" or cardinal_fbms.get("validated_record_count") != 8_675:
        raise ReportInputError("CardinalFBMS independent verification did not pass 8,675 records")
    if cardinal_fbms.get("post_outcome_verification") is not True or cardinal_fbms.get("not_the_frozen_gate") is not True:
        raise ReportInputError("CardinalFBMS verification boundary is missing")
    transfer_scores = cardinal_fbms.get("equal_dataset_condition_mean_balanced_accuracy")
    transfer_condition = "pretrained_cardinal_fbms"
    if (
        cardinal_fbms.get("schema")
        != "ieee-mi-cardinal-fbms-post-outcome-independent-verification-v1"
        or not isinstance(transfer_scores, dict)
        or transfer_condition not in transfer_scores
    ):
        raise ReportInputError("CardinalFBMS verification score payload is malformed")
    transfer_score = finite_float(
        transfer_scores[transfer_condition],
        label="CardinalFBMS verified BA",
        minimum=0.0,
        maximum=1.0,
    )
    transfer_protocol = protocol_by_id["cardinal_fbms_transfer"]
    if transfer_protocol["primary_model_or_condition"] != transfer_condition:
        raise ReportInputError("CardinalFBMS protocol summary names a different condition")
    assert_close(
        transfer_score,
        finite_float(
            transfer_protocol["primary_value"],
            label="CardinalFBMS protocol BA",
            minimum=0.0,
            maximum=1.0,
        ),
        label="CardinalFBMS artifact/protocol summary BA",
    )

    return ValidatedInputs(
        plan=plan,
        audit=audit,
        manifest=manifest,
        models=models,
        ranking=tuple(ranking),
        dataset_summary=tuple(dataset_summary),
        overall_summary=tuple(overall_summary),
        balanced_wide=tuple(balanced_wide),
        accuracy_wide=tuple(accuracy_wide),
        top_ten=tuple(top_ten),
        dataset_winners=tuple(dataset_winners),
        tcformer_context=tuple(tcformer_context),
        calibration=tuple(calibration),
        complexity=tuple(complexity),
        protocols=tuple(protocols),
        gauge=gauge,
        cardinal_fbms=cardinal_fbms,
    )


UNICODE_REPLACEMENTS = str.maketrans(
    {
        "\u00a0": " ",
        "\u00ad": "",
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2026": "...",
        "\u00d7": "x",
        "\u00b1": "+/-",
        "\u2264": "<=",
        "\u2265": ">=",
        "\u2192": "->",
        "\u2190": "<-",
        "\u21d2": "=>",
    }
)


def ascii_text(value: Any) -> str:
    text = str(value).translate(UNICODE_REPLACEMENTS)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return text


def inline_markup(value: str) -> str:
    text = ascii_text(value).strip()
    tokens: dict[str, str] = {}

    def token(markup: str) -> str:
        key = f"@@INLINE{len(tokens):05d}@@"
        tokens[key] = markup
        return key

    def replace_link(match: re.Match[str]) -> str:
        label = html.escape(ascii_text(match.group(1)), quote=False)
        url = ascii_text(match.group(2)).strip()
        safe_url = html.escape(url, quote=True)
        visible_url = html.escape(url, quote=False)
        return token(
            f'<link href="{safe_url}" color="#2F6B9A"><u>{label}</u></link> '
            f'<font size="7" color="#5F6C7B">[{visible_url}]</font>'
        )

    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", replace_link, text)

    def replace_code(match: re.Match[str]) -> str:
        code = html.escape(ascii_text(match.group(1)), quote=False)
        return token(f'<font name="Courier" color="#174A5B">{code}</font>')

    text = re.sub(r"`([^`]+)`", replace_code, text)
    rendered = html.escape(text, quote=False)
    rendered = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", rendered)
    rendered = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", rendered)
    for key, markup in tokens.items():
        rendered = rendered.replace(key, markup)
    return rendered


def make_styles(*, compact: bool = False) -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    body_size = 8.2 if compact else 9.0
    leading = 10.4 if compact else 12.0
    return {
        "Title": ParagraphStyle(
            "ReportTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=28 if not compact else 25,
            leading=32 if not compact else 29,
            textColor=NAVY,
            alignment=TA_LEFT,
            spaceAfter=14,
        ),
        "Subtitle": ParagraphStyle(
            "ReportSubtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=12,
            leading=16,
            textColor=MUTED,
            spaceAfter=12,
        ),
        "H1": ParagraphStyle(
            "ReportH1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=18 if not compact else 16,
            leading=22 if not compact else 19,
            textColor=NAVY,
            spaceBefore=13,
            spaceAfter=8,
            keepWithNext=True,
        ),
        "H2": ParagraphStyle(
            "ReportH2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13 if not compact else 12,
            leading=16 if not compact else 14,
            textColor=BLUE,
            spaceBefore=10,
            spaceAfter=5,
            keepWithNext=True,
        ),
        "H3": ParagraphStyle(
            "ReportH3",
            parent=base["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=10.5 if not compact else 9.5,
            leading=13,
            textColor=TEAL,
            spaceBefore=8,
            spaceAfter=4,
            keepWithNext=True,
        ),
        "H4": ParagraphStyle(
            "ReportH4",
            parent=base["Heading4"],
            fontName="Helvetica-Bold",
            fontSize=9.2,
            leading=11,
            textColor=INK,
            spaceBefore=6,
            spaceAfter=3,
            keepWithNext=True,
        ),
        "Body": ParagraphStyle(
            "ReportBody",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=body_size,
            leading=leading,
            textColor=INK,
            alignment=TA_LEFT,
            spaceAfter=5,
            splitLongWords=True,
        ),
        "Small": ParagraphStyle(
            "ReportSmall",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.2,
            leading=9.2,
            textColor=MUTED,
            spaceAfter=3,
            splitLongWords=True,
        ),
        "Cell": ParagraphStyle(
            "ReportCell",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=6.5 if compact else 7.0,
            leading=8.0 if compact else 8.6,
            textColor=INK,
            spaceAfter=0,
            splitLongWords=True,
        ),
        "CellHeader": ParagraphStyle(
            "ReportCellHeader",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=6.4 if compact else 6.9,
            leading=7.8 if compact else 8.4,
            textColor=WHITE,
            alignment=TA_CENTER,
            spaceAfter=0,
            splitLongWords=True,
        ),
        "Code": ParagraphStyle(
            "ReportCode",
            parent=base["Code"],
            fontName="Courier",
            fontSize=6.4 if compact else 7.0,
            leading=8.2 if compact else 9.0,
            textColor=colors.HexColor("#173F4F"),
            backColor=colors.HexColor("#F1F5F8"),
            borderColor=GRID,
            borderWidth=0.5,
            borderPadding=6,
            spaceBefore=3,
            spaceAfter=7,
        ),
        "Callout": ParagraphStyle(
            "ReportCallout",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.4 if compact else 9.0,
            leading=11 if compact else 12,
            textColor=NAVY,
            spaceAfter=0,
        ),
        "TOC0": ParagraphStyle(
            "TOC0", fontName="Helvetica-Bold", fontSize=10, leading=14, leftIndent=0, textColor=NAVY
        ),
        "TOC1": ParagraphStyle(
            "TOC1", fontName="Helvetica", fontSize=8.5, leading=12, leftIndent=14, textColor=INK
        ),
        "TOC2": ParagraphStyle(
            "TOC2", fontName="Helvetica", fontSize=7.5, leading=10, leftIndent=28, textColor=MUTED
        ),
    }


class InvariantCanvas(canvas.Canvas):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["invariant"] = 1
        kwargs["pageCompression"] = 1
        super().__init__(*args, **kwargs)


class ReportDocTemplate(BaseDocTemplate):
    def __init__(
        self,
        filename: str,
        *,
        title: str,
        subject: str,
        pagesize: tuple[float, float],
        landscape_report: bool,
    ) -> None:
        margin_x = 0.42 * inch if landscape_report else 0.62 * inch
        super().__init__(
            filename,
            pagesize=pagesize,
            leftMargin=margin_x,
            rightMargin=margin_x,
            topMargin=0.58 * inch,
            bottomMargin=0.50 * inch,
            title=ascii_text(title),
            author="",
            subject=ascii_text(subject),
            creator="Deterministic EEG MI benchmark report builder",
            keywords="EEG, motor imagery, reproducibility, development benchmark",
        )
        self.report_title = ascii_text(title)
        self.report_subject = ascii_text(subject)
        self._heading_sequence = 0
        frame = Frame(
            self.leftMargin,
            self.bottomMargin,
            self.width,
            self.height,
            leftPadding=0,
            rightPadding=0,
            topPadding=4,
            bottomPadding=2,
            id="report-frame",
        )
        self.addPageTemplates(PageTemplate(id="report", frames=[frame], onPage=self._decorate_page))

    def beforeDocument(self) -> None:
        self._heading_sequence = 0

    def _decorate_page(self, canv: canvas.Canvas, doc: BaseDocTemplate) -> None:
        width, height = self.pagesize
        canv.saveState()
        canv.setTitle(self.report_title)
        canv.setAuthor("")
        canv.setSubject(self.report_subject)
        canv.setCreator("Deterministic EEG MI benchmark report builder")
        canv.setFillColor(NAVY)
        canv.rect(0, height - 0.16 * inch, width, 0.16 * inch, stroke=0, fill=1)
        canv.setStrokeColor(GRID)
        canv.setLineWidth(0.45)
        canv.line(self.leftMargin, 0.34 * inch, width - self.rightMargin, 0.34 * inch)
        canv.setFont("Helvetica", 6.8)
        canv.setFillColor(MUTED)
        header = self.report_title
        if len(header) > 76:
            header = header[:73] + "..."
        canv.drawString(self.leftMargin, height - 0.34 * inch, header)
        canv.drawRightString(width - self.rightMargin, height - 0.34 * inch, "OPENED DEVELOPMENT EVIDENCE")
        canv.drawString(self.leftMargin, 0.20 * inch, "No confirmation, clinical, novelty, or global SOTA claim")
        canv.drawRightString(width - self.rightMargin, 0.20 * inch, f"Page {doc.page}")
        canv.restoreState()

    def afterFlowable(self, flowable: Any) -> None:
        if not isinstance(flowable, Paragraph) or not hasattr(flowable, "_toc_level"):
            return
        level = int(flowable._toc_level)
        text = ascii_text(flowable.getPlainText())
        key = f"heading-{self._heading_sequence:04d}"
        self._heading_sequence += 1
        self.canv.bookmarkPage(key)
        self.canv.addOutlineEntry(text, key, level=level, closed=False)
        self.notify("TOCEntry", (level, text, self.page, key))


def heading(text: str, level: int, styles: Mapping[str, ParagraphStyle]) -> Paragraph:
    style_name = f"H{min(max(level, 1), 4)}"
    value = Paragraph(inline_markup(text), styles[style_name])
    if level <= 3:
        value._toc_level = level - 1
    return value


def callout(
    text: str,
    styles: Mapping[str, ParagraphStyle],
    *,
    background: colors.Color = PALE_BLUE,
    accent: colors.Color = BLUE,
) -> Table:
    body = Paragraph(inline_markup(text), styles["Callout"])
    table = Table([[body]], colWidths=[None], hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), background),
                ("BOX", (0, 0), (-1, -1), 0.5, accent),
                ("LINEBEFORE", (0, 0), (0, -1), 3.2, accent),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return table


def table_widths(column_count: int, available_width: float) -> list[float]:
    if column_count == 2:
        ratios = (0.35, 0.65)
    elif column_count == 3:
        ratios = (0.23, 0.35, 0.42)
    elif column_count == 4:
        ratios = (0.17, 0.24, 0.27, 0.32)
    elif column_count == 5:
        ratios = (0.16, 0.21, 0.21, 0.21, 0.21)
    else:
        ratios = tuple(1.0 / column_count for _ in range(column_count))
    return [available_width * ratio for ratio in ratios]


def styled_table(
    rows: Sequence[Sequence[Any]],
    *,
    styles: Mapping[str, ParagraphStyle],
    col_widths: Sequence[float] | None = None,
    header_rows: int = 1,
    compact: bool = False,
    alignments: Mapping[int, str] | None = None,
) -> LongTable:
    if not rows or not rows[0]:
        raise ValueError("table rows must be nonempty")
    normalized: list[list[Paragraph]] = []
    for row_index, row in enumerate(rows):
        normalized.append(
            [
                Paragraph(
                    inline_markup(str(cell)),
                    styles["CellHeader"] if row_index < header_rows else styles["Cell"],
                )
                for cell in row
            ]
        )
    table = LongTable(
        normalized,
        colWidths=list(col_widths) if col_widths is not None else None,
        repeatRows=header_rows,
        splitByRow=1,
        hAlign="LEFT",
    )
    commands: list[tuple[Any, ...]] = [
        ("BACKGROUND", (0, 0), (-1, header_rows - 1), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, header_rows - 1), WHITE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.35, GRID),
        ("LEFTPADDING", (0, 0), (-1, -1), 3 if compact else 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3 if compact else 4),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5 if compact else 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5 if compact else 3.5),
    ]
    for row_index in range(header_rows, len(rows)):
        if (row_index - header_rows) % 2 == 1:
            commands.append(("BACKGROUND", (0, row_index), (-1, row_index), ROW_ALT))
    if alignments:
        for column, alignment in alignments.items():
            commands.append(("ALIGN", (column, header_rows), (column, -1), alignment))
    table.setStyle(TableStyle(commands))
    return table


def markdown_table_rows(lines: Sequence[str]) -> list[list[str]]:
    def cells(line: str) -> list[str]:
        stripped = line.strip().strip("|")
        return [part.strip() for part in stripped.split("|")]

    rows = [cells(line) for line in lines]
    if len(rows) < 2 or len({len(row) for row in rows}) != 1:
        raise ReportInputError("malformed Markdown table in documentation")
    separator = rows[1]
    if not all(re.fullmatch(r":?-{3,}:?", value.replace(" ", "")) for value in separator):
        raise ReportInputError("Markdown table is missing its separator row")
    return [rows[0], *rows[2:]]


def wrap_code(code: str, *, width: int) -> str:
    lines: list[str] = []
    for original in ascii_text(code).splitlines():
        if len(original) <= width:
            lines.append(original)
            continue
        indent = original[: len(original) - len(original.lstrip())]
        lines.extend(
            textwrap.wrap(
                original,
                width=width,
                subsequent_indent=indent + "  ",
                break_long_words=True,
                break_on_hyphens=False,
                replace_whitespace=False,
                drop_whitespace=False,
            )
        )
    return "\n".join(lines)


def markdown_flowables(
    markdown: str,
    *,
    styles: Mapping[str, ParagraphStyle],
    available_width: float,
    code_width: int,
) -> list[Any]:
    lines = ascii_text(markdown).splitlines()
    flowables: list[Any] = []
    index = 0
    paragraph_parts: list[str] = []

    def flush_paragraph() -> None:
        if not paragraph_parts:
            return
        text = " ".join(part.strip() for part in paragraph_parts if part.strip())
        paragraph_parts.clear()
        if text:
            flowables.append(Paragraph(inline_markup(text), styles["Body"]))

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            index += 1
            continue

        if stripped.startswith("```"):
            flush_paragraph()
            index += 1
            code_lines: list[str] = []
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code_lines.append(lines[index])
                index += 1
            if index >= len(lines):
                raise ReportInputError("unterminated fenced code block in documentation")
            index += 1
            flowables.append(Preformatted(wrap_code("\n".join(code_lines), width=code_width), styles["Code"]))
            continue

        heading_match = re.match(r"^(#{1,4})\s+(.+)$", stripped)
        if heading_match:
            flush_paragraph()
            flowables.append(heading(heading_match.group(2), len(heading_match.group(1)), styles))
            index += 1
            continue

        if stripped.startswith("|") and index + 1 < len(lines) and re.match(
            r"^\s*\|?\s*:?-{3,}", lines[index + 1]
        ):
            flush_paragraph()
            table_lines = [line, lines[index + 1]]
            index += 2
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index])
                index += 1
            parsed = markdown_table_rows(table_lines)
            flowables.append(
                styled_table(
                    parsed,
                    styles=styles,
                    col_widths=table_widths(len(parsed[0]), available_width),
                    compact=len(parsed[0]) >= 4,
                )
            )
            flowables.append(Spacer(1, 7))
            continue

        if stripped.startswith(">"):
            flush_paragraph()
            quote_parts: list[str] = []
            while index < len(lines) and lines[index].strip().startswith(">"):
                quote_parts.append(lines[index].strip().lstrip("> "))
                index += 1
            flowables.append(callout(" ".join(quote_parts), styles, background=PALE_GOLD, accent=GOLD))
            flowables.append(Spacer(1, 6))
            continue

        list_match = re.match(r"^\s*(-|\d+\.)\s+(.+)$", line)
        if list_match:
            flush_paragraph()
            ordered = list_match.group(1) != "-"
            items: list[str] = []
            while index < len(lines):
                current = re.match(r"^\s*(-|\d+\.)\s+(.+)$", lines[index])
                if current is None or (current.group(1) != "-") != ordered:
                    break
                item_parts = [current.group(2).strip()]
                index += 1
                while index < len(lines):
                    continuation = lines[index]
                    if not continuation.strip():
                        break
                    if re.match(r"^\s*(-|\d+\.)\s+", continuation) or re.match(
                        r"^(#{1,4})\s+", continuation.strip()
                    ):
                        break
                    if continuation.strip().startswith(("|", "```", ">")):
                        break
                    item_parts.append(continuation.strip())
                    index += 1
                items.append(" ".join(item_parts))
                if index < len(lines) and not lines[index].strip():
                    break
            list_items = [
                ListItem(Paragraph(inline_markup(item), styles["Body"]), leftIndent=13)
                for item in items
            ]
            list_flowable = ListFlowable(
                list_items,
                bulletType="1" if ordered else "bullet",
                start="1" if ordered else None,
                leftIndent=18,
                bulletFontName="Helvetica",
                bulletFontSize=8,
                spaceAfter=5,
            )
            # Short lists should move as a unit instead of leaving one or two
            # continuation bullets on an otherwise empty page.  Long lists
            # remain splittable so they cannot exceed a full frame.
            if len(items) <= 6:
                keep_group: list[Any] = [list_flowable]
                if (
                    flowables
                    and isinstance(flowables[-1], Paragraph)
                    and hasattr(flowables[-1], "_toc_level")
                ):
                    keep_group.insert(0, flowables.pop())
                flowables.append(KeepTogether(keep_group))
            else:
                flowables.append(list_flowable)
            continue

        paragraph_parts.append(stripped)
        index += 1

    flush_paragraph()
    return flowables


def toc(styles: Mapping[str, ParagraphStyle]) -> TableOfContents:
    value = TableOfContents()
    value.levelStyles = [styles["TOC0"], styles["TOC1"], styles["TOC2"]]
    value.dotsMinLevel = 0
    return value


def cover(
    *,
    title: str,
    subtitle: str,
    caveat: str,
    metadata: Sequence[tuple[str, str]],
    styles: Mapping[str, ParagraphStyle],
) -> list[Any]:
    rows = [[Paragraph(inline_markup(key), styles["CellHeader"]), Paragraph(inline_markup(value), styles["Cell"])] for key, value in metadata]
    metadata_table = Table(rows, colWidths=[1.55 * inch, 4.9 * inch], hAlign="LEFT")
    metadata_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), NAVY),
                ("TEXTCOLOR", (0, 0), (0, -1), WHITE),
                ("BACKGROUND", (1, 0), (1, -1), ROW_ALT),
                ("GRID", (0, 0), (-1, -1), 0.4, GRID),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return [
        Spacer(1, 0.72 * inch),
        RectFlowable(width=1.15 * inch, height=0.07 * inch, color=TEAL),
        Spacer(1, 0.18 * inch),
        Paragraph(inline_markup(title), styles["Title"]),
        Paragraph(inline_markup(subtitle), styles["Subtitle"]),
        Spacer(1, 0.20 * inch),
        callout(caveat, styles, background=PALE_RED, accent=RED),
        Spacer(1, 0.30 * inch),
        metadata_table,
        Spacer(1, 0.35 * inch),
        Paragraph(
            "Generated deterministically from the checksum-verified release bundle. "
            "No author identity was supplied or inferred.",
            styles["Small"],
        ),
        PageBreak(),
        # Keep the visible contents heading out of its own TOC.  Besides being
        # redundant, that self-entry can force a one-line second TOC page.
        Paragraph(inline_markup("Contents"), styles["H1"]),
        toc(styles),
        PageBreak(),
    ]


class RectFlowable(Spacer):
    def __init__(self, *, width: float, height: float, color: colors.Color) -> None:
        super().__init__(width, height)
        self.bar_width = width
        self.bar_height = height
        self.bar_color = color

    def draw(self) -> None:
        self.canv.saveState()
        self.canv.setFillColor(self.bar_color)
        self.canv.rect(0, 0, self.bar_width, self.bar_height, stroke=0, fill=1)
        self.canv.restoreState()


def percent(value: float, digits: int = 2) -> str:
    return f"{100.0 * value:.{digits}f}%"


def short_dataset(dataset: str) -> str:
    return {
        "local_exp4": "Local",
        "bnci2014_001": "BNCI-2a",
        "bnci2014_004": "BNCI-2b",
        "cho2017": "Cho",
        "physionet_mi": "PhysioNet",
    }[dataset]


def top_ten_chart(inputs: ValidatedInputs, *, width: float, height: float) -> Drawing:
    drawing = Drawing(width, height)
    drawing.add(String(0, height - 12, "Top 10 equal-dataset balanced accuracy", fontName="Helvetica-Bold", fontSize=10, fillColor=NAVY))
    left = 178
    right = 42
    top = height - 29
    bottom = 25
    chart_width = width - left - right
    x_min, x_max = 0.70, 0.745
    rows = list(inputs.ranking[:10])
    row_height = (top - bottom) / len(rows)
    for tick in (0.70, 0.71, 0.72, 0.73, 0.74):
        x = left + (tick - x_min) / (x_max - x_min) * chart_width
        drawing.add(Line(x, bottom - 2, x, top + 2, strokeColor=GRID, strokeWidth=0.4))
        drawing.add(String(x, 9, f"{tick*100:.0f}%", fontName="Helvetica", fontSize=6.5, textAnchor="middle", fillColor=MUTED))
    for index, row in enumerate(rows):
        value = float(row["value"])
        y = top - (index + 0.72) * row_height
        label = ascii_text(row["model"])
        drawing.add(String(0, y + 1, f"{index+1:>2}. {label}", fontName="Helvetica", fontSize=6.6, fillColor=INK))
        bar_width = max(0.0, (value - x_min) / (x_max - x_min) * chart_width)
        color = TEAL if index == 0 else BLUE
        drawing.add(Rect(left, y - 1, bar_width, row_height * 0.58, strokeColor=None, fillColor=color))
        drawing.add(String(left + bar_width + 3, y + 1, percent(value, 3), fontName="Helvetica-Bold", fontSize=6.4, fillColor=INK))
    drawing.add(String(left, height - 23, "Truncated horizontal scale: 70.0%-74.5%", fontName="Helvetica-Oblique", fontSize=6.4, fillColor=MUTED))
    return drawing


def dataset_group_chart(inputs: ValidatedInputs, *, width: float, height: float) -> Drawing:
    drawing = Drawing(width, height)
    drawing.add(String(0, height - 12, "Dataset behavior: winner, overall leader, TCFormer", fontName="Helvetica-Bold", fontSize=10, fillColor=NAVY))
    left = 36
    right = 8
    top = height - 32
    bottom = 44
    chart_width = width - left - right
    chart_height = top - bottom
    dataset_map = {(row["dataset"], row["model"]): row for row in inputs.dataset_summary}
    winners = {row["dataset"]: row for row in inputs.dataset_winners}
    for tick in (0.0, 0.25, 0.50, 0.75, 1.0):
        y = bottom + tick * chart_height
        drawing.add(Line(left, y, left + chart_width, y, strokeColor=GRID, strokeWidth=0.4))
        drawing.add(String(left - 4, y - 2, f"{tick*100:.0f}", fontName="Helvetica", fontSize=6.2, textAnchor="end", fillColor=MUTED))
    group_width = chart_width / len(EXPECTED_DATASETS)
    colors_by = (TEAL, BLUE, GOLD)
    for index, dataset in enumerate(EXPECTED_DATASETS):
        center = left + (index + 0.5) * group_width
        values = (
            float(winners[dataset]["balanced_accuracy"]),
            float(dataset_map[(dataset, inputs.leader)]["balanced_accuracy"]),
            float(dataset_map[(dataset, TCFORMER)]["balanced_accuracy"]),
        )
        bar_width = min(11.5, group_width / 4.7)
        for offset, (value, color) in enumerate(zip(values, colors_by, strict=True)):
            x = center + (offset - 1) * (bar_width + 2) - bar_width / 2
            drawing.add(Rect(x, bottom, bar_width, value * chart_height, strokeColor=None, fillColor=color))
        drawing.add(String(center, 25, short_dataset(dataset), fontName="Helvetica", fontSize=6.3, textAnchor="middle", fillColor=INK))
    legend_x = left + 3
    for label, color in zip(("Dataset winner", "Overall leader", "TCFormer"), colors_by, strict=True):
        drawing.add(Rect(legend_x, height - 26, 8, 6, strokeColor=None, fillColor=color))
        drawing.add(String(legend_x + 11, height - 25, label, fontName="Helvetica", fontSize=6.2, fillColor=INK))
        legend_x += 87
    drawing.add(String(2, bottom + chart_height / 2, "BA (%)", fontName="Helvetica", fontSize=6.5, fillColor=MUTED, angle=90))
    return drawing


def complexity_scatter(inputs: ValidatedInputs, *, width: float, height: float) -> Drawing:
    drawing = Drawing(width, height)
    drawing.add(String(0, height - 12, "Performance-complexity context (all 43 models)", fontName="Helvetica-Bold", fontSize=10, fillColor=NAVY))
    left, right, bottom, top_pad = 46, 22, 33, 28
    chart_width = width - left - right
    chart_height = height - bottom - top_pad
    rank_by_model = {row["model"]: float(row["value"]) for row in inputs.ranking}
    points = [
        (row["model"], math.log10(float(row["parameter_count_median"])), rank_by_model[row["model"]])
        for row in inputs.complexity
    ]
    x_min = math.floor(min(point[1] for point in points) * 2) / 2
    x_max = math.ceil(max(point[1] for point in points) * 2) / 2
    y_min, y_max = 0.50, 0.75
    for tick in range(int(x_min * 2), int(x_max * 2) + 1):
        value = tick / 2
        x = left + (value - x_min) / (x_max - x_min) * chart_width
        drawing.add(Line(x, bottom, x, bottom + chart_height, strokeColor=GRID, strokeWidth=0.35))
        drawing.add(String(x, 17, f"10^{value:g}", fontName="Helvetica", fontSize=6.2, textAnchor="middle", fillColor=MUTED))
    for tick in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75):
        y = bottom + (tick - y_min) / (y_max - y_min) * chart_height
        drawing.add(Line(left, y, left + chart_width, y, strokeColor=GRID, strokeWidth=0.35))
        drawing.add(String(left - 5, y - 2, percent(tick, 0), fontName="Helvetica", fontSize=6.2, textAnchor="end", fillColor=MUTED))
    for model, log_parameters, balanced_accuracy in points:
        x = left + (log_parameters - x_min) / (x_max - x_min) * chart_width
        y = bottom + (balanced_accuracy - y_min) / (y_max - y_min) * chart_height
        color = RED if model == inputs.leader else GOLD if model == TCFORMER else BLUE
        radius = 4 if model in {inputs.leader, TCFORMER} else 2.1
        drawing.add(Circle(x, y, radius, strokeColor=WHITE, strokeWidth=0.4, fillColor=color))
        if model in {inputs.leader, TCFORMER}:
            label = "Leader" if model == inputs.leader else "TCFormer"
            drawing.add(String(x + 5, y + 3, label, fontName="Helvetica-Bold", fontSize=6.5, fillColor=color))
    drawing.add(String(left + chart_width / 2, 4, "Median trainable parameter count (log10 scale)", fontName="Helvetica", fontSize=7, textAnchor="middle", fillColor=MUTED))
    drawing.add(String(5, bottom + chart_height / 2, "Equal-dataset BA", fontName="Helvetica", fontSize=7, fillColor=MUTED, angle=90))
    return drawing


def result_metric_cards(inputs: ValidatedInputs, styles: Mapping[str, ParagraphStyle]) -> Table:
    tcformer_row = next(row for row in inputs.ranking if row["model"] == TCFORMER)
    context = next(row for row in inputs.tcformer_context if row["model"] == inputs.leader)
    cards = (
        ("Exact jobs", f"{EXPECTED_JOBS:,} / {EXPECTED_JOBS:,}", "zero missing or failed"),
        ("Observed leader", percent(inputs.leader_ba, 3), inputs.leader),
        ("TCFormer", percent(float(tcformer_row["value"]), 3), f"rank {tcformer_row['rank']} of 43"),
        (
            "Leader - TCFormer",
            f"{100*float(context['equal_dataset_macro_balanced_accuracy_difference']):+.3f} pp",
            (
                f"95% descriptive interval "
                f"[{100*float(context['descriptive_fixed_suite_bootstrap_interval95_low']):+.3f}, "
                f"{100*float(context['descriptive_fixed_suite_bootstrap_interval95_high']):+.3f}]"
            ),
        ),
    )
    row: list[Table] = []
    for title, value, detail in cards:
        card = Table(
            [[Paragraph(inline_markup(title), styles["Small"])], [Paragraph(f"<b>{inline_markup(value)}</b>", styles["H2"])], [Paragraph(inline_markup(detail), styles["Small"])]]
        )
        card.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), PALE_BLUE),
                    ("BOX", (0, 0), (-1, -1), 0.5, BLUE),
                    ("LEFTPADDING", (0, 0), (-1, -1), 7),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )
        row.append(card)
    outer = Table([row], colWidths=[1.98 * inch] * 4, hAlign="LEFT")
    outer.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 2), ("RIGHTPADDING", (0, 0), (-1, -1), 2)]))
    return outer


def build_results_story(inputs: ValidatedInputs, doc: ReportDocTemplate) -> list[Any]:
    styles = make_styles(compact=True)
    gauge_score = float(
        inputs.gauge["model_equal_dataset_balanced_accuracy"][inputs.gauge["candidate"]]
    )
    transfer_score = float(
        inputs.cardinal_fbms["equal_dataset_condition_mean_balanced_accuracy"][
            "pretrained_cardinal_fbms"
        ]
    )
    story = cover(
        title="Benchmark Results and Statistics",
        subtitle="Exact 43-model common-recipe motor-imagery EEG development benchmark",
        caveat=(
            "Exploratory opened-development-cohort evidence only. The observed leader was "
            "outcome-selected; this report is not confirmation, clinical validation, proof "
            "of novelty, or a global state-of-the-art claim."
        ),
        metadata=(
            ("Common grid", "43 models x 448 participant/fold units x 5 seeds"),
            ("Exact completion", "96,320 / 96,320 jobs; exact audit passed"),
            ("Primary outcome", "Equal-dataset macro balanced accuracy"),
            ("Plan identity", str(inputs.plan["plan_sha256"])),
            ("Analysis manifest", sha256_file(ANALYSIS / "manifest.json")),
        ),
        styles=styles,
    )

    story.extend(
        [
            heading("Executive result", 1, styles),
            result_metric_cards(inputs, styles),
            Spacer(1, 8),
            callout(
                "The selected leader's descriptive interval versus TCFormer includes zero. "
                "The same leader also had worse aggregate NLL, Brier score, and 15-bin ECE "
                "than TCFormer. Rank is not a clinical or statistical separation for every pair.",
                styles,
                background=PALE_GOLD,
                accent=GOLD,
            ),
            Spacer(1, 8),
            Table(
                [[top_ten_chart(inputs, width=4.95 * inch, height=2.75 * inch), dataset_group_chart(inputs, width=4.35 * inch, height=2.75 * inch)]],
                colWidths=[5.05 * inch, 4.45 * inch],
                hAlign="LEFT",
                style=TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 2)]),
            ),
            Spacer(1, 6),
            complexity_scatter(inputs, width=9.45 * inch, height=2.35 * inch),
            PageBreak(),
            heading("Top 10 common-recipe configurations", 1, styles),
        ]
    )

    top_rows: list[list[str]] = [["Rank", "Model", "Equal-dataset BA", "Evidence status"]]
    for row in inputs.ranking[:10]:
        top_rows.append([row["rank"], row["model"], percent(float(row["value"]), 4), "descriptive opened development"])
    story.append(
        styled_table(
            top_rows,
            styles=styles,
            col_widths=[0.45 * inch, 4.6 * inch, 1.35 * inch, 2.55 * inch],
            compact=True,
            alignments={0: "RIGHT", 2: "RIGHT"},
        )
    )
    story.extend([Spacer(1, 8), heading("Per-dataset winners", 2, styles)])
    winner_rows: list[list[str]] = [["Dataset", "Observed winner", "Balanced accuracy", "Participants"]]
    for row in inputs.dataset_winners:
        winner_rows.append(
            [short_dataset(row["dataset"]), row["model"], percent(float(row["balanced_accuracy"]), 4), row["subjects_averaged"]]
        )
    story.append(
        styled_table(
            winner_rows,
            styles=styles,
            col_widths=[1.1 * inch, 4.6 * inch, 1.45 * inch, 1.0 * inch],
            compact=True,
            alignments={2: "RIGHT", 3: "RIGHT"},
        )
    )

    story.extend([Spacer(1, 9), heading("Descriptive uncertainty versus TCFormer", 2, styles)])
    context_by_model = {row["model"]: row for row in inputs.tcformer_context}
    uncertainty_rows: list[list[str]] = [["Rank", "Model", "Delta (pp)", "Fixed-suite descriptive 95% interval (pp)", "Resamples"]]
    for row in inputs.ranking[:10]:
        context = context_by_model[row["model"]]
        uncertainty_rows.append(
            [
                row["rank"],
                row["model"],
                f"{100*float(context['equal_dataset_macro_balanced_accuracy_difference']):+.3f}",
                (
                    f"[{100*float(context['descriptive_fixed_suite_bootstrap_interval95_low']):+.3f}, "
                    f"{100*float(context['descriptive_fixed_suite_bootstrap_interval95_high']):+.3f}]"
                ),
                f"{int(context['fixed_suite_bootstrap_resamples']):,}",
            ]
        )
    uncertainty_table = styled_table(
        uncertainty_rows,
        styles=styles,
        col_widths=[0.42 * inch, 3.65 * inch, 0.9 * inch, 2.5 * inch, 1.0 * inch],
        compact=True,
        alignments={0: "RIGHT", 2: "RIGHT", 4: "RIGHT"},
    )
    uncertainty_note = Paragraph(
        "Intervals preserve participant clustering across this fixed opened suite. "
        "They are not selection-adjusted confirmatory confidence intervals; no "
        "p-values or null-hypothesis decisions were computed.",
        styles["Small"],
    )
    story.append(KeepTogether([uncertainty_table, Spacer(1, 3), uncertainty_note]))

    story.extend([PageBreak(), heading("Calibration and probabilistic quality", 1, styles)])
    calibration_by_model = {row["model"]: row for row in inputs.calibration}
    overall_by_model = {row["model"]: row for row in inputs.overall_summary}
    selected_models = []
    for model in [inputs.leader, inputs.ranking[1]["model"], inputs.ranking[2]["model"], "fbmsnet", "fbcnet", TCFORMER]:
        if model not in selected_models:
            selected_models.append(model)
    calibration_rows: list[list[str]] = [["Model", "BA", "Accuracy", "NLL (lower)", "Brier (lower)", "ECE-15 (lower)"]]
    for model in selected_models:
        cal = calibration_by_model[model]
        summary = overall_by_model[model]
        calibration_rows.append(
            [
                model,
                percent(float(summary["balanced_accuracy"]), 2),
                percent(float(summary["accuracy"]), 2),
                f"{float(cal['nll']):.4f}",
                f"{float(cal['multiclass_brier']):.4f}",
                percent(float(cal["ece"]), 2),
            ]
        )
    story.append(
        styled_table(
            calibration_rows,
            styles=styles,
            col_widths=[3.65 * inch, 0.85 * inch, 0.85 * inch, 1.1 * inch, 1.1 * inch, 1.1 * inch],
            alignments={1: "RIGHT", 2: "RIGHT", 3: "RIGHT", 4: "RIGHT", 5: "RIGHT"},
        )
    )
    story.extend(
        [
            Spacer(1, 7),
            callout(
                "The balanced-accuracy leader is not the calibration leader. Confidence "
                "thresholds, abstention, and command activation require prospective "
                "calibration in the intended population and operating stream.",
                styles,
                background=PALE_GOLD,
                accent=GOLD,
            ),
            Spacer(1, 10),
            heading("Complexity and runtime context", 2, styles),
        ]
    )
    complexity_by_model = {row["model"]: row for row in inputs.complexity}
    complexity_rows: list[list[str]] = [["Model", "Median parameters", "Median inference (ms/trial)", "Median peak CUDA (MiB)", "Median job time (s)"]]
    for model in selected_models:
        row = complexity_by_model[model]
        complexity_rows.append(
            [
                model,
                f"{float(row['parameter_count_median']):,.0f}",
                f"{float(row['inference_ms_per_trial_median']):.4f}",
                f"{float(row['cuda_peak_memory_bytes_median']) / 1024**2:,.1f}",
                f"{float(row['job_total_seconds_median']):,.2f}",
            ]
        )
    story.append(
        styled_table(
            complexity_rows,
            styles=styles,
            col_widths=[3.45 * inch, 1.25 * inch, 1.55 * inch, 1.45 * inch, 1.25 * inch],
            alignments={1: "RIGHT", 2: "RIGHT", 3: "RIGHT", 4: "RIGHT"},
        )
    )
    story.append(Paragraph("Parameter count varies with channel count and class count; the table reports medians across jobs. Shared-workstation load affects timing, so runtime is an engineering outcome, not a scientific tie-breaker.", styles["Small"]))

    story.extend([PageBreak(), heading("Full 43-model balanced-accuracy table", 1, styles)])
    story.append(Paragraph("Percentages are rounded to two decimals for presentation. The sealed CSV retains full float precision. Overall is an equal average of five dataset means.", styles["Small"]))
    ba_rows: list[list[str]] = [["Rank", "Model", "Local", "BNCI-2a", "BNCI-2b", "Cho", "PhysioNet", "Overall"]]
    for row in inputs.balanced_wide:
        ba_rows.append(
            [
                row["rank"],
                row["model"],
                percent(float(row["local_exp4_balanced_accuracy"])),
                percent(float(row["bnci2014_001_balanced_accuracy"])),
                percent(float(row["bnci2014_004_balanced_accuracy"])),
                percent(float(row["cho2017_balanced_accuracy"])),
                percent(float(row["physionet_mi_balanced_accuracy"])),
                percent(float(row["equal_dataset_macro_balanced_accuracy"])),
            ]
        )
    story.append(
        styled_table(
            ba_rows,
            styles=styles,
            col_widths=[0.38 * inch, 3.25 * inch, 0.78 * inch, 0.78 * inch, 0.78 * inch, 0.78 * inch, 0.84 * inch, 0.82 * inch],
            compact=True,
            alignments={0: "RIGHT", 2: "RIGHT", 3: "RIGHT", 4: "RIGHT", 5: "RIGHT", 6: "RIGHT", 7: "RIGHT"},
        )
    )

    story.extend([PageBreak(), heading("Full 43-model standard-accuracy table", 1, styles)])
    story.append(Paragraph("Accuracy rank is a descriptive secondary ordering. The primary balanced-accuracy rank is retained beside it.", styles["Small"]))
    accuracy_rows: list[list[str]] = [["Acc. rank", "BA rank", "Model", "Local", "BNCI-2a", "BNCI-2b", "Cho", "PhysioNet", "Overall"]]
    for row in inputs.accuracy_wide:
        accuracy_rows.append(
            [
                row["descriptive_accuracy_rank"],
                row["primary_balanced_accuracy_rank"],
                row["model"],
                percent(float(row["local_exp4_accuracy"])),
                percent(float(row["bnci2014_001_accuracy"])),
                percent(float(row["bnci2014_004_accuracy"])),
                percent(float(row["cho2017_accuracy"])),
                percent(float(row["physionet_mi_accuracy"])),
                percent(float(row["equal_dataset_macro_accuracy"])),
            ]
        )
    story.append(
        styled_table(
            accuracy_rows,
            styles=styles,
            col_widths=[0.48 * inch, 0.45 * inch, 2.85 * inch, 0.72 * inch, 0.72 * inch, 0.72 * inch, 0.72 * inch, 0.80 * inch, 0.78 * inch],
            compact=True,
            alignments={0: "RIGHT", 1: "RIGHT", 3: "RIGHT", 4: "RIGHT", 5: "RIGHT", 6: "RIGHT", 7: "RIGHT", 8: "RIGHT"},
        )
    )

    story.extend([PageBreak(), heading("Separate protocols and negative results", 1, styles)])
    protocol_rows: list[list[str]] = [["Track", "Status", "Completed / planned", "Primary result", "Comparison boundary"]]
    for row in inputs.protocols:
        primary = "-"
        if row["primary_model_or_condition"] and row["primary_value"]:
            primary = f"{row['primary_model_or_condition']}: {percent(float(row['primary_value']), 3)}"
        planned = f"{int(row['planned_units']):,}" if row["planned_units"] else "not specified"
        protocol_rows.append(
            [
                row["protocol_label"],
                row["status"],
                f"{int(row['completed_units']):,} / {planned}",
                primary,
                row["comparison_boundary"],
            ]
        )
    story.append(
        styled_table(
            protocol_rows,
            styles=styles,
            col_widths=[2.25 * inch, 1.65 * inch, 1.15 * inch, 2.25 * inch, 2.35 * inch],
            compact=True,
        )
    )
    story.extend(
        [
            Spacer(1, 9),
            callout(
                f"Gauge Gate 1 reached {percent(gauge_score, 4)} on its separate "
                "17-subject, one-seed screen "
                "but failed the frozen breadth rule: only two of five dataset deltas were "
                "nonnegative. Gate 2 was not run, and Gauge is not a 44th common-grid row.",
                styles,
                background=PALE_RED,
                accent=RED,
            ),
            Spacer(1, 9),
            callout(
                "CardinalFBMS transfer independently verified 8,675 records and a "
                f"{percent(transfer_score, 3)} "
                "equal-dataset condition-mean BA for pretrained CardinalFBMS. It is a "
                "separate two-dataset native-transfer protocol; the post-outcome verifier "
                "is explicitly not the frozen gate.",
                styles,
                background=PALE_TEAL,
                accent=TEAL,
            ),
            Spacer(1, 10),
            heading("Interpretation caveats", 2, styles),
        ]
    )
    caveats = (
        "All five common cohorts were previously opened development cohorts.",
        "The winner was outcome-selected from 43 configurations; its interval is descriptive, not confirmatory.",
        "The five dataset environments do not represent the global population of montages, users, or care settings.",
        "Most recordings are from healthy volunteers; no patient-facing efficacy or safety claim is supported.",
        "Pseudonymous job-, subject-seed-, and subject-level metrics remain participant-derived and require governance review before public release.",
        "Raw EEG, trial predictions, direct identifiers, embeddings, and checkpoints are not included.",
        "Extended/compact/scale variants are configurations or ablations, not automatically separate novel architectures.",
        "Calibration, false activation, abstention, fatigue, latency, drift, and failure modes require prospective evaluation.",
    )
    story.append(
        ListFlowable(
            [ListItem(Paragraph(inline_markup(item), styles["Body"]), leftIndent=13) for item in caveats],
            bulletType="bullet",
            leftIndent=18,
            bulletFontSize=8,
        )
    )
    story.extend([Spacer(1, 9), heading("Provenance", 2, styles)])
    provenance_rows = [
        ["Artifact", "SHA-256 or identity"],
        ["Plan identity", str(inputs.plan["plan_sha256"])],
        ["Preflight report", sha256_file(RESULTS / "common_grid_v6" / "preflight" / "report.json")],
        ["Final audit", sha256_file(RESULTS / "common_grid_v6" / "final_audit.json")],
        ["Analysis manifest", sha256_file(ANALYSIS / "manifest.json")],
        ["Analysis contract", str(inputs.manifest["analysis_contract_sha256"])],
        ["Analysis input ledger", str(inputs.manifest["input_ledger_sha256"])],
        ["Forensic ledger", str(inputs.manifest["forensic_ledger_sha256"])],
    ]
    story.append(
        styled_table(
            provenance_rows,
            styles=styles,
            col_widths=[2.0 * inch, 7.0 * inch],
            compact=True,
        )
    )
    return story


def build_methodology_story(inputs: ValidatedInputs, doc: ReportDocTemplate) -> list[Any]:
    styles = make_styles(compact=False)
    story = cover(
        title="Methodology and Architectures",
        subtitle="Common-recipe protocol, in-house families, external references, and negative candidates",
        caveat=(
            "'In-house' identifies repository ownership or a project-specific derivative/procedure. "
            "It does not establish legal ownership, literature novelty, patentability, clinical "
            "validity, or state of the art."
        ),
        metadata=(
            ("Verified common scope", "43 models; five opened datasets; 96,320 exact jobs"),
            ("Primary outcome", "Equal-dataset macro balanced accuracy"),
            ("Observed leader", f"{inputs.leader} ({percent(inputs.leader_ba, 4)})"),
            ("Experimental decisions", "Gauge Gate 1 failed; CHSD promotion stopped"),
            ("Source document", "docs/METHODOLOGY_AND_ARCHITECTURES.md"),
        ),
        styles=styles,
    )
    markdown = require_file(ROOT / "docs" / "METHODOLOGY_AND_ARCHITECTURES.md").read_text(encoding="utf-8")
    story.extend(
        markdown_flowables(
            markdown,
            styles=styles,
            available_width=doc.width,
            code_width=82,
        )
    )
    return story


def build_reproducibility_story(inputs: ValidatedInputs, doc: ReportDocTemplate) -> list[Any]:
    styles = make_styles(compact=False)
    story = cover(
        title="Reproducibility Handbook",
        subtitle="Environment, exact rerun, data governance, ethics, and protocol boundaries",
        caveat=(
            "Exact reproduction requires lawful access to all five datasets, including private "
            "Local Exp4, and the frozen Linux/CUDA environment. Bundle verification is not a "
            "substitute for scientific rerunning or governance approval."
        ),
        metadata=(
            (
                "Runtime",
                "Linux x86-64; CPython 3.12.13; UV; PyTorch 2.6.0+cu124; "
                "TorchAudio 2.6.0+cu124",
            ),
            ("Formal scope", "43 x 448 x 5 = 96,320 score-blind jobs"),
            ("Preflight", "172 / 172 CUDA construction and backpropagation checks"),
            ("Data release", "No raw EEG or trial predictions; pseudonymous derived metrics require review"),
            ("License/authorship", "Unresolved; no author or IRB identity inferred"),
        ),
        styles=styles,
    )
    sources = (
        ROOT / "docs" / "REPRODUCIBILITY.md",
        ROOT / "docs" / "DATA_ACCESS.md",
        ROOT / "docs" / "ETHICS_AND_PRIVACY.md",
        ROOT / "docs" / "PROTOCOL_BOUNDARIES.md",
    )
    for source_index, path in enumerate(sources):
        if source_index == 1:
            # Let the data-access section use otherwise empty space after the
            # short reproduction-limitations list, but require enough room for
            # its source banner, heading, and opening paragraph.
            story.extend([Spacer(1, 12), CondPageBreak(2.15 * inch)])
        elif source_index:
            story.append(PageBreak())
        story.append(
            callout(
                f"Source section: {path.relative_to(ROOT).as_posix()}",
                styles,
                background=PALE_TEAL,
                accent=TEAL,
            )
        )
        story.append(Spacer(1, 5))
        story.extend(
            markdown_flowables(
                require_file(path).read_text(encoding="utf-8"),
                styles=styles,
                available_width=doc.width,
                code_width=82,
            )
        )
    return story


def validate_pdf_bytes(path: Path) -> None:
    require_file(path)
    payload = path.read_bytes()
    if not payload.startswith(b"%PDF-") or b"%%EOF" not in payload[-2048:]:
        raise ReportInputError(f"generated file is not a complete PDF: {path.name}")
    if len(payload) < 12_000:
        raise ReportInputError(f"generated PDF is unexpectedly small: {path.name}")
    page_objects = payload.count(b"/Type /Page") - payload.count(b"/Type /Pages")
    if page_objects < 2:
        raise ReportInputError(f"generated PDF has too few pages: {path.name}")


def build_one(
    path: Path,
    *,
    title: str,
    subject: str,
    pagesize: tuple[float, float],
    landscape_report: bool,
    story_builder: Any,
    inputs: ValidatedInputs,
) -> None:
    document = ReportDocTemplate(
        str(path),
        title=title,
        subject=subject,
        pagesize=pagesize,
        landscape_report=landscape_report,
    )
    story = story_builder(inputs, document)
    document.multiBuild(story, canvasmaker=InvariantCanvas)
    validate_pdf_bytes(path)


def build_reports(inputs: ValidatedInputs, output_dir: Path) -> tuple[Path, ...]:
    output_dir = output_dir.resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".build-reports-", dir=output_dir.parent))
    try:
        specifications = (
            (
                PDF_FILENAMES[0],
                "Benchmark Results and Statistics",
                "Audited common-grid development results and protocol-separated context",
                landscape(LETTER),
                True,
                build_results_story,
            ),
            (
                PDF_FILENAMES[1],
                "Methodology and Architectures",
                "Motor-imagery EEG benchmark methods and architecture provenance",
                LETTER,
                False,
                build_methodology_story,
            ),
            (
                PDF_FILENAMES[2],
                "Reproducibility Handbook",
                "Exact environment, rerun, governance, ethics, and protocol boundaries",
                LETTER,
                False,
                build_reproducibility_story,
            ),
        )
        staged_paths: list[Path] = []
        for filename, title, subject, pagesize, is_landscape, builder in specifications:
            staged = staging / filename
            build_one(
                staged,
                title=title,
                subject=subject,
                pagesize=pagesize,
                landscape_report=is_landscape,
                story_builder=builder,
                inputs=inputs,
            )
            staged_paths.append(staged)

        output_dir.mkdir(parents=True, exist_ok=True)
        final_paths: list[Path] = []
        for staged in staged_paths:
            destination = output_dir / staged.name
            os.replace(staged, destination)
            destination.chmod(0o644)
            validate_pdf_bytes(destination)
            final_paths.append(destination)
        return tuple(final_paths)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="destination directory (default: output/pdf below the project root)",
    )
    value.add_argument(
        "--validate-only",
        action="store_true",
        help="validate sealed inputs and cross-table consistency without creating PDFs",
    )
    return value


def main(argv: Sequence[str] | None = None) -> int:
    rl_config.invariant = 1
    rl_config.pageCompression = 1
    try:
        inputs = validate_inputs()
        if args := parser().parse_args(argv):
            if args.validate_only:
                print("PASS: sealed inputs and cross-table identities validated")
                return 0
            outputs = build_reports(inputs, args.output_dir)
    except (OSError, ReportInputError, ValueError) as error:
        print(f"REPORT BUILD FAILED: {error}", file=os.sys.stderr)
        return 1

    for path in outputs:
        print(f"WROTE {path}")
        print(f"SHA256 {sha256_file(path)}")
    print("Visual rendering and page-by-page inspection are still required before release.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
