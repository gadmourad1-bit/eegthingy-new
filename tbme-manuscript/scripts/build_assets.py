#!/usr/bin/env python3
"""Build deterministic manuscript statistics and vector figures.

The authoritative inputs are the sealed common-grid analysis tables in
``dl/results/common_grid_v6/analysis``.  This script never reads trial-level
predictions and never modifies the benchmark release.  It derives a compact
set of manuscript-facing statistics, then creates four vector PDF figures.

Run from any directory::

    python tbme-manuscript/scripts/build_assets.py
    python tbme-manuscript/scripts/build_assets.py --check

``--check`` rebuilds every asset in a temporary directory and requires exact
byte equality with the committed outputs.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence


# Matplotlib must see a writable cache before it is imported.  Keep the cache
# outside the manuscript tree so generated/ contains only declared artifacts.
MPL_CACHE = Path(tempfile.gettempdir()) / "tbme-manuscript-mpl-cache-v1"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPL_CACHE)
os.environ.setdefault("SOURCE_DATE_EPOCH", "1787184000")

try:
    import matplotlib

    matplotlib.use("pdf")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyBboxPatch
    import numpy as np
except ImportError as error:  # pragma: no cover - environment diagnosis
    raise SystemExit(
        "build_assets.py requires NumPy and Matplotlib in an isolated environment"
    ) from error


SCRIPT_PATH = Path(__file__).resolve()
MANUSCRIPT_ROOT = SCRIPT_PATH.parent.parent
REPOSITORY_ROOT = MANUSCRIPT_ROOT.parent
ANALYSIS_ROOT = REPOSITORY_ROOT / "dl" / "results" / "common_grid_v6" / "analysis"

FIGURE_NAMES = (
    "platform.pdf",
    "workflow.pdf",
    "architecture.pdf",
    "efficiency_scatter.pdf",
    "dataset_deltas.pdf",
)
GENERATED_NAMES = ("statistics.json", "selected_metrics.csv")
ASSET_PATHS = tuple(f"figures/{name}" for name in FIGURE_NAMES) + tuple(
    f"generated/{name}" for name in GENERATED_NAMES
)

DATASET_ORDER = (
    "local_exp4",
    "bnci2014_001",
    "bnci2014_004",
    "cho2017",
    "physionet_mi",
)
DATASET_LABELS = {
    "local_exp4": "Local Exp4",
    "bnci2014_001": "BNCI 2014-001",
    "bnci2014_004": "BNCI 2014-004",
    "cho2017": "Cho 2017",
    "physionet_mi": "PhysioNet MI",
}

LEADER = "cardinal_fbc_compactdyn_scale025_extended"
SELECTED_MODELS = (
    LEADER,
    "cardinal_dynamics_sinc_extended",
    "fbcnet",
    "cardinal_fbc",
    "tcformer",
)
COMPARATORS = SELECTED_MODELS[1:]
TCFORMER_CONTEXT_MODELS = SELECTED_MODELS[:-1]
CALIBRATION_SELECTED_MODELS = (
    LEADER,
    "cardinal_dynamics_sinc_extended",
    "tcformer",
)
SHARED_BOOTSTRAP_METRIC_MODELS = {
    "balanced_accuracy": SELECTED_MODELS,
    "accuracy": SELECTED_MODELS,
    "ece": SELECTED_MODELS,
    "nll": CALIBRATION_SELECTED_MODELS,
    "multiclass_brier": CALIBRATION_SELECTED_MODELS,
}
MODEL_ROLES = {
    LEADER: "leader",
    "cardinal_dynamics_sinc_extended": "compact",
    "fbcnet": "fbcnet",
    "cardinal_fbc": "cardinal_fbc",
    "tcformer": "tcformer",
}
MODEL_LABELS = {
    LEADER: "Leader (31-anchor, x0.25)",
    "cardinal_dynamics_sinc_extended": "Compact Sinc dynamics (31 anchors)",
    "fbcnet": "FBCNet",
    "cardinal_fbc": "CardinalFBC",
    "tcformer": "TCFormer",
}

BOOTSTRAP_RESAMPLES = 100_000
BOOTSTRAP_SEED = 20_260_729
SIGN_FLIP_RESAMPLES = 100_000
SIGN_FLIP_SEED = 20_260_820
SIGN_FLIP_BATCH_SIZE = 2_000
EXPECTED_LEADER_RAW_EXCEEDANCES = 15_247
EXPECTED_LEADER_MAX_EXCEEDANCES = 77_230

EXPECTED_RUNTIME = {
    "python": "3.12.13",
    "numpy": "2.4.4",
    "matplotlib": "3.10.9",
}
TCFORMER_CONTEXT_SOURCE = "tcformer_overall_context.csv"
MANUSCRIPT_TABLE_LABELS = {
    LEADER: "Coordinate-field FBC + 0.25 compact dynamics, 31 anchors",
    "cardinal_dynamics_sinc_extended": "Sinc coordinate dynamics, 31 anchors",
    "fbcnet": "FBCNet",
    "cardinal_fbc": "Coordinate-field FBC floor",
    "tcformer": "TCFormer",
}

COLOR_NAVY = "#17324D"
COLOR_TEAL = "#007C83"
COLOR_GOLD = "#D89A2B"
COLOR_CORAL = "#C6513E"
COLOR_PURPLE = "#67507A"
COLOR_BLUE = "#3B6FA1"
COLOR_GRAY = "#9AA4AD"
COLOR_LIGHT = "#EEF3F5"
COLOR_INK = "#17212B"
SELECTED_COLORS = {
    LEADER: COLOR_CORAL,
    "cardinal_dynamics_sinc_extended": COLOR_GOLD,
    "fbcnet": COLOR_BLUE,
    "cardinal_fbc": COLOR_TEAL,
    "tcformer": COLOR_PURPLE,
}
SELECTED_MARKERS = {
    LEADER: "*",
    "cardinal_dynamics_sinc_extended": "D",
    "fbcnet": "s",
    "cardinal_fbc": "o",
    "tcformer": "^",
}

RC_PARAMS = {
    "font.family": "DejaVu Sans",
    "font.size": 8.0,
    "axes.titlesize": 10.0,
    "axes.labelsize": 8.5,
    "axes.linewidth": 0.7,
    "axes.edgecolor": COLOR_INK,
    "axes.labelcolor": COLOR_INK,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "xtick.color": COLOR_INK,
    "ytick.color": COLOR_INK,
    "legend.fontsize": 7.2,
    "text.color": COLOR_INK,
    "pdf.fonttype": 42,
    "pdf.compression": 6,
    "savefig.transparent": False,
}

FIXED_PDF_TIME = dt.datetime(2026, 8, 20, 0, 0, 0, tzinfo=dt.timezone.utc)


class AssetError(RuntimeError):
    """An input contract or deterministic asset check failed."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _read_csv(name: str) -> list[dict[str, str]]:
    path = ANALYSIS_ROOT / name
    if not path.is_file():
        raise AssetError(f"required analysis table is missing: {path}")
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    if not reader.fieldnames or not rows:
        raise AssetError(f"analysis table is empty: {path}")
    return rows


def _unique_by(rows: Iterable[Mapping[str, str]], *keys: str) -> dict[tuple[str, ...], Mapping[str, str]]:
    result: dict[tuple[str, ...], Mapping[str, str]] = {}
    for row in rows:
        key = tuple(row[name] for name in keys)
        if key in result:
            raise AssetError(f"duplicate analysis key: {key}")
        result[key] = row
    return result


def _finite(value: str, *, label: str) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise AssetError(f"{label} is not numeric: {value!r}") from error
    if not math.isfinite(result):
        raise AssetError(f"{label} is not finite")
    return result


def _validate_runtime() -> None:
    observed = {
        "python": (
            f"{sys.version_info.major}.{sys.version_info.minor}."
            f"{sys.version_info.micro}"
        ),
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
    }
    if observed != EXPECTED_RUNTIME:
        raise AssetError(
            "asset runtime contract mismatch: "
            f"expected {EXPECTED_RUNTIME!r}, observed {observed!r}; "
            "use the repository .venv"
        )


def _load_inputs() -> dict[str, Any]:
    manifest_path = ANALYSIS_ROOT / "manifest.json"
    analysis_path = ANALYSIS_ROOT / "analysis.json"
    if not manifest_path.is_file() or not analysis_path.is_file():
        raise AssetError("analysis manifest or analysis summary is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    required = (
        "overall_summary.csv",
        "dataset_summary.csv",
        "complexity_summary.csv",
        "model_ranking.csv",
        "subject_metrics.csv",
        "calibration_summary.csv",
        TCFORMER_CONTEXT_SOURCE,
    )
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise AssetError("analysis manifest has no file ledger")
    for name in required:
        expected = files.get(name)
        observed = _sha256_file(ANALYSIS_ROOT / name)
        if expected != observed:
            raise AssetError(f"authoritative checksum mismatch: {name}")

    overall_rows = _read_csv("overall_summary.csv")
    dataset_rows = _read_csv("dataset_summary.csv")
    complexity_rows = _read_csv("complexity_summary.csv")
    ranking_rows = _read_csv("model_ranking.csv")
    subject_rows = _read_csv("subject_metrics.csv")
    calibration_rows = _read_csv("calibration_summary.csv")
    tcformer_context_rows = _read_csv(TCFORMER_CONTEXT_SOURCE)

    overall = _unique_by(overall_rows, "model")
    complexity = _unique_by(complexity_rows, "model")
    ranking = _unique_by(ranking_rows, "model")
    by_dataset = _unique_by(dataset_rows, "dataset", "model")
    subjects = _unique_by(subject_rows, "dataset", "model", "subject")
    calibration = _unique_by(
        calibration_rows, "level", "dataset", "model"
    )
    tcformer_context = _unique_by(
        tcformer_context_rows, "model", "comparator"
    )

    model_sets = (set(overall), set(complexity), set(ranking))
    if any(len(values) != 43 for values in model_sets) or not (
        model_sets[0] == model_sets[1] == model_sets[2]
    ):
        raise AssetError("the 43-model table contract is not exact")
    if len(by_dataset) != 43 * len(DATASET_ORDER):
        raise AssetError("dataset summary is not the complete 43 x 5 table")
    expected_calibration_keys = {
        ("dataset", dataset, model_tuple[0])
        for dataset in DATASET_ORDER
        for model_tuple in overall
    } | {
        ("equal_dataset_macro", "ALL_DATASETS", model_tuple[0])
        for model_tuple in overall
    }
    if set(calibration) != expected_calibration_keys:
        raise AssetError("calibration summary is not the complete 43 x (5 + 1) table")
    if any(row["ece_bins"] != "15" for row in calibration_rows):
        raise AssetError("calibration summary ECE-bin contract differs")
    if any((model,) not in overall for model in SELECTED_MODELS):
        raise AssetError("one or more selected manuscript models are absent")
    ranks = sorted(int(row["rank"]) for row in ranking_rows)
    if ranks != list(range(1, 44)):
        raise AssetError("model ranking is not a unique 1..43 ordering")
    if ranking[(LEADER,)]["rank"] != "1":
        raise AssetError("the declared manuscript leader is not rank 1")
    expected_subject_rows = 43 * 132
    if len(subjects) != expected_subject_rows:
        raise AssetError(
            f"subject table has {len(subjects)} rows; expected {expected_subject_rows}"
        )
    if analysis.get("descriptive_leader") != LEADER:
        raise AssetError("analysis summary and manuscript leader differ")
    if analysis.get("plan_sha256") != manifest.get("plan_sha256"):
        raise AssetError("analysis and manifest plan identities differ")
    expected_context_keys = {
        (model, "tcformer") for model_tuple in overall for model in model_tuple
        if model != "tcformer"
    }
    if set(tcformer_context) != expected_context_keys:
        raise AssetError(
            "TCFormer context is not the exact 42 non-TCFormer model family"
        )

    return {
        "manifest": manifest,
        "analysis": analysis,
        "overall": overall,
        "complexity": complexity,
        "ranking": ranking,
        "by_dataset": by_dataset,
        "subjects": subjects,
        "calibration": calibration,
        "tcformer_context": tcformer_context,
        "input_hashes": {
            name: _sha256_file(ANALYSIS_ROOT / name)
            for name in (*required, "analysis.json", "manifest.json")
        },
    }


def _sealed_tcformer_fixed_suite_context(
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    rows: Mapping[tuple[str, ...], Mapping[str, str]] = inputs[
        "tcformer_context"
    ]
    overall: Mapping[tuple[str, ...], Mapping[str, str]] = inputs["overall"]
    tcformer_ba = _finite(
        overall[("tcformer",)]["balanced_accuracy"],
        label="tcformer/balanced_accuracy",
    )
    results: list[dict[str, Any]] = []
    seeds: set[int] = set()
    for model in TCFORMER_CONTEXT_MODELS:
        row = rows[(model, "tcformer")]
        if row["datasets"] != "5" or row["subjects_total"] != "132":
            raise AssetError(f"sealed TCFormer context scope differs: {model}")
        try:
            resamples = int(row["fixed_suite_bootstrap_resamples"])
            seed = int(row["fixed_suite_bootstrap_seed"])
        except ValueError as error:
            raise AssetError(
                f"sealed TCFormer context resamples/seed is invalid: {model}"
            ) from error
        if resamples != 100_000 or seed <= 0 or seed in seeds:
            raise AssetError(
                f"sealed TCFormer context resampling contract differs: {model}"
            )
        seeds.add(seed)
        estimate = _finite(
            row["equal_dataset_macro_balanced_accuracy_difference"],
            label=f"{model}/sealed TCFormer difference",
        )
        low = _finite(
            row["descriptive_fixed_suite_bootstrap_interval95_low"],
            label=f"{model}/sealed TCFormer interval low",
        )
        high = _finite(
            row["descriptive_fixed_suite_bootstrap_interval95_high"],
            label=f"{model}/sealed TCFormer interval high",
        )
        model_ba = _finite(
            overall[(model,)]["balanced_accuracy"],
            label=f"{model}/balanced_accuracy",
        )
        if not math.isclose(
            estimate, model_ba - tcformer_ba, rel_tol=0.0, abs_tol=1e-14
        ):
            raise AssetError(
                f"sealed TCFormer point estimate differs from summary: {model}"
            )
        if not low <= estimate <= high:
            raise AssetError(f"sealed TCFormer interval excludes estimate: {model}")
        status = row["status"].strip()
        if not status:
            raise AssetError(f"sealed TCFormer context status is empty: {model}")
        results.append(
            {
                "model": model,
                "comparator": "tcformer",
                "balanced_accuracy_difference": estimate,
                "descriptive_fixed_suite_bootstrap_interval95": [low, high],
                "bootstrap_resamples": resamples,
                "bootstrap_seed": seed,
                "status": status,
            }
        )
    return {
        "source_file": f"dl/results/common_grid_v6/analysis/{TCFORMER_CONTEXT_SOURCE}",
        "source_sha256": inputs["input_hashes"][TCFORMER_CONTEXT_SOURCE],
        "scope": (
            "sealed descriptive model-minus-prespecified-TCFormer context for "
            "the fixed opened five-dataset suite"
        ),
        "results": results,
    }


def _validate_manuscript_reconciliation(
    sealed_context: Mapping[str, Any],
    model_rows: Sequence[Mapping[str, Any]],
) -> None:
    manuscript_path = MANUSCRIPT_ROOT / "main.tex"
    if not manuscript_path.is_file():
        raise AssetError(f"manuscript table source is missing: {manuscript_path}")
    lines = manuscript_path.read_text(encoding="utf-8").splitlines()
    manuscript_text = " ".join(line.strip() for line in lines)
    sealed_by_model = {
        row["model"]: row for row in sealed_context["results"]
    }
    metrics_by_model = {row["model"]: row for row in model_rows}

    def shortstack_values(cell: str, *, label: str) -> tuple[str, str]:
        text = cell.strip()
        if text.endswith(r"\\"):
            text = text[:-2].rstrip()
        prefix = r"\shortstack[r]{"
        if not text.startswith(prefix) or not text.endswith("}"):
            raise AssetError(f"manuscript metric cell is invalid: {label}")
        parts = text[len(prefix) : -1].split(r"\\")
        if (
            len(parts) != 2
            or not parts[1].startswith("(")
            or not parts[1].endswith(")")
        ):
            raise AssetError(f"manuscript metric cell is invalid: {label}")
        return parts[0], parts[1][1:-1]

    for model in SELECTED_MODELS:
        label = MANUSCRIPT_TABLE_LABELS[model]
        matches = [line for line in lines if line.startswith(f"{label} &")]
        if len(matches) != 1:
            raise AssetError(
                f"manuscript selected-model row count differs for {model}: "
                f"{len(matches)}"
            )
        cells = [cell.strip() for cell in matches[0].split("&")]
        if len(cells) != 9:
            raise AssetError(f"manuscript selected-model row shape differs: {model}")
        metric_row = metrics_by_model[model]
        if cells[1] != str(metric_row["rank"]):
            raise AssetError(f"manuscript selected-model rank differs: {model}")
        for metric, cell_index in (
            ("balanced_accuracy", 2),
            ("accuracy", 3),
            ("ece", 8),
        ):
            observed = shortstack_values(
                cells[cell_index], label=f"{model}/{metric}"
            )
            expected = (
                f"{100.0 * metric_row[metric]:.3f}",
                f"{100.0 * metric_row[f'{metric}_shared_bootstrap_se']:.3f}",
            )
            if observed != expected:
                raise AssetError(
                    f"manuscript point/SE differs for {model}/{metric}: "
                    f"expected {expected}, observed {observed}"
                )
        if model == "tcformer":
            if cells[4] != "0" or cells[5] != "Reference":
                raise AssetError("manuscript TCFormer reference cells differ")
            continue
        sealed_row = sealed_by_model[model]
        expected_delta = (
            f"{100.0 * sealed_row['balanced_accuracy_difference']:.3f}"
        )
        observed_delta = cells[4].replace("$", "").strip()
        if observed_delta != expected_delta:
            raise AssetError(
                f"manuscript TCFormer difference differs for {model}: "
                f"expected {expected_delta}, observed {observed_delta}"
            )
        interval_text = cells[5].replace("$", "").strip()
        interval_match = re.fullmatch(
            r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]",
            interval_text,
        )
        if interval_match is None:
            raise AssetError(f"manuscript TCFormer interval is invalid: {model}")
        expected_interval = tuple(
            f"{100.0 * value:.3f}"
            for value in sealed_row[
                "descriptive_fixed_suite_bootstrap_interval95"
            ]
        )
        if interval_match.groups() != expected_interval:
            raise AssetError(
                f"manuscript TCFormer interval differs for {model}: expected "
                f"{expected_interval}, observed {interval_match.groups()}"
            )

    def calibration_values(model: str) -> tuple[str, ...]:
        row = metrics_by_model[model]
        return (
            f"{100.0 * row['ece']:.3f}",
            f"{100.0 * row['ece_shared_bootstrap_se']:.3f}",
            f"{row['nll']:.3f}",
            f"{row['nll_shared_bootstrap_se']:.3f}",
            f"{row['multiclass_brier']:.3f}",
            f"{row['multiclass_brier_shared_bootstrap_se']:.3f}",
        )

    tc_ece, tc_ece_se, tc_nll, tc_nll_se, tc_brier, tc_brier_se = (
        calibration_values("tcformer")
    )
    leader_ece, leader_ece_se, leader_nll, leader_nll_se, leader_brier, leader_brier_se = (
        calibration_values(LEADER)
    )
    compact_ece, compact_ece_se, compact_nll, compact_nll_se, compact_brier, compact_brier_se = (
        calibration_values("cardinal_dynamics_sinc_extended")
    )
    expected_fragments = (
        f"expected calibration error ({tc_ece}\\%; SE {tc_ece_se} points), "
        f"negative log likelihood ({tc_nll}; SE {tc_nll_se}), and Brier score "
        f"({tc_brier}; SE {tc_brier_se})",
        f"The leader's corresponding values were {leader_ece}\\% "
        f"(SE {leader_ece_se} points), {leader_nll} (SE {leader_nll_se}), and "
        f"{leader_brier} (SE {leader_brier_se})",
        f"The compact Sinc alternative was intermediate at {compact_ece}\\% "
        f"(SE {compact_ece_se} points), {compact_nll} (SE {compact_nll_se}), "
        f"and {compact_brier} (SE {compact_brier_se})",
    )
    for fragment in expected_fragments:
        if manuscript_text.count(fragment) != 1:
            raise AssetError(
                "manuscript calibration point/SE prose differs from generated values"
            )


def _subject_matrices(inputs: Mapping[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    subjects: Mapping[tuple[str, ...], Mapping[str, str]] = inputs["subjects"]
    model_order = tuple(
        row["model"]
        for row in sorted(
            inputs["ranking"].values(), key=lambda row: int(row["rank"])
        )
    )
    result: dict[str, dict[str, np.ndarray]] = {}
    for dataset in DATASET_ORDER:
        identifiers = sorted(
            {
                int(key[2])
                for key in subjects
                if key[0] == dataset and key[1] == LEADER
            }
        )
        if not identifiers:
            raise AssetError(f"no subjects found for {dataset}")
        values: dict[str, np.ndarray] = {}
        for model in model_order:
            vector = np.asarray(
                [
                    _finite(
                        subjects[(dataset, model, str(subject))]["balanced_accuracy"],
                        label=f"{dataset}/{model}/S{subject} balanced accuracy",
                    )
                    for subject in identifiers
                ],
                dtype=np.float64,
            )
            values[model] = vector
        result[dataset] = values
    if sum(len(result[dataset][LEADER]) for dataset in DATASET_ORDER) != 132:
        raise AssetError("selected-model subject count is not 132")
    return result


def _selected_additional_metric_matrices(
    inputs: Mapping[str, Any],
    balanced_accuracy_matrices: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    subjects: Mapping[tuple[str, ...], Mapping[str, str]] = inputs["subjects"]
    result: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for metric, models in SHARED_BOOTSTRAP_METRIC_MODELS.items():
        if metric == "balanced_accuracy":
            continue
        metric_datasets: dict[str, dict[str, np.ndarray]] = {}
        for dataset in DATASET_ORDER:
            identifiers = sorted(
                {
                    int(key[2])
                    for key in subjects
                    if key[0] == dataset and key[1] == LEADER
                }
            )
            if len(identifiers) != len(balanced_accuracy_matrices[dataset][LEADER]):
                raise AssetError(f"subject ordering contract differs: {dataset}")
            metric_datasets[dataset] = {
                model: np.asarray(
                    [
                        _finite(
                            subjects[(dataset, model, str(subject))][metric],
                            label=f"{dataset}/{model}/S{subject} {metric}",
                        )
                        for subject in identifiers
                    ],
                    dtype=np.float64,
                )
                for model in models
            }
        result[metric] = metric_datasets
    return result


def _percentile_interval(samples: np.ndarray) -> tuple[float, float]:
    interval = np.quantile(samples, (0.025, 0.975), method="linear")
    return float(interval[0]), float(interval[1])


def _bootstrap_statistics(
    matrices: Mapping[str, Mapping[str, np.ndarray]],
    additional_metric_matrices: Mapping[
        str, Mapping[str, Mapping[str, np.ndarray]]
    ],
) -> dict[str, Any]:
    """Shared subject-stratified bootstrap for equal-dataset estimands.

    Exactly one subject-index matrix is drawn per dataset and reused across
    every requested metric and model, preserving paired resampling dependence.
    """

    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    model_count = len(SELECTED_MODELS)
    overall_samples = np.zeros((BOOTSTRAP_RESAMPLES, model_count), dtype=np.float64)
    dataset_samples: dict[str, np.ndarray] = {}
    dataset_points: dict[str, np.ndarray] = {}
    additional_overall_samples = {
        metric: np.zeros(
            (BOOTSTRAP_RESAMPLES, len(SHARED_BOOTSTRAP_METRIC_MODELS[metric])),
            dtype=np.float64,
        )
        for metric in additional_metric_matrices
    }
    additional_dataset_points: dict[
        str, dict[str, dict[str, float]]
    ] = {metric: {} for metric in additional_metric_matrices}

    for dataset in DATASET_ORDER:
        matrix = np.column_stack([matrices[dataset][model] for model in SELECTED_MODELS])
        indices = rng.integers(
            0,
            matrix.shape[0],
            size=(BOOTSTRAP_RESAMPLES, matrix.shape[0]),
            dtype=np.int32,
        )
        samples = matrix[indices].mean(axis=1, dtype=np.float64)
        dataset_samples[dataset] = samples
        dataset_points[dataset] = matrix.mean(axis=0, dtype=np.float64)
        overall_samples += samples / len(DATASET_ORDER)
        for metric, metric_datasets in additional_metric_matrices.items():
            metric_models = SHARED_BOOTSTRAP_METRIC_MODELS[metric]
            metric_matrix = np.column_stack(
                [metric_datasets[dataset][model] for model in metric_models]
            )
            if metric_matrix.shape[0] != matrix.shape[0]:
                raise AssetError(
                    f"shared bootstrap subject count differs: {dataset}/{metric}"
                )
            metric_samples = metric_matrix[indices].mean(
                axis=1, dtype=np.float64
            )
            additional_overall_samples[metric] += (
                metric_samples / len(DATASET_ORDER)
            )
            metric_points = metric_matrix.mean(axis=0, dtype=np.float64)
            additional_dataset_points[metric][dataset] = {
                model: float(metric_points[index])
                for index, model in enumerate(metric_models)
            }

    overall_points = np.mean(
        np.vstack([dataset_points[dataset] for dataset in DATASET_ORDER]), axis=0
    )
    absolute: dict[str, dict[str, float | list[float]]] = {}
    for index, model in enumerate(SELECTED_MODELS):
        low, high = _percentile_interval(overall_samples[:, index])
        absolute[model] = {
            "estimate": float(overall_points[index]),
            "bootstrap_se": float(np.std(overall_samples[:, index], ddof=1)),
            "bootstrap_ci95": [low, high],
        }

    metric_absolute: dict[
        str, dict[str, dict[str, float | list[float]]]
    ] = {"balanced_accuracy": absolute}
    for metric, metric_samples in additional_overall_samples.items():
        metric_absolute[metric] = {}
        metric_models = SHARED_BOOTSTRAP_METRIC_MODELS[metric]
        for index, model in enumerate(metric_models):
            low, high = _percentile_interval(metric_samples[:, index])
            point = np.mean(
                [
                    additional_dataset_points[metric][dataset][model]
                    for dataset in DATASET_ORDER
                ]
            )
            metric_absolute[metric][model] = {
                "estimate": float(point),
                "bootstrap_se": float(np.std(metric_samples[:, index], ddof=1)),
                "bootstrap_ci95": [low, high],
            }

    pairwise: dict[str, dict[str, float | list[float]]] = {}
    dataset_pairwise: dict[str, dict[str, dict[str, float | list[float]]]] = {}
    for comparator_index, comparator in enumerate(COMPARATORS, start=1):
        samples = overall_samples[:, 0] - overall_samples[:, comparator_index]
        low, high = _percentile_interval(samples)
        pairwise[comparator] = {
            "estimate": float(overall_points[0] - overall_points[comparator_index]),
            "bootstrap_se": float(np.std(samples, ddof=1)),
            "bootstrap_ci95": [low, high],
        }
        dataset_pairwise[comparator] = {}
        for dataset in DATASET_ORDER:
            dataset_delta = (
                dataset_samples[dataset][:, 0]
                - dataset_samples[dataset][:, comparator_index]
            )
            dataset_low, dataset_high = _percentile_interval(dataset_delta)
            dataset_pairwise[comparator][dataset] = {
                "estimate": float(
                    dataset_points[dataset][0]
                    - dataset_points[dataset][comparator_index]
                ),
                "bootstrap_se": float(np.std(dataset_delta, ddof=1)),
                "bootstrap_ci95": [dataset_low, dataset_high],
                "subjects": int(len(matrices[dataset][LEADER])),
            }

    return {
        "absolute": absolute,
        "metric_absolute": metric_absolute,
        "metric_dataset_points": additional_dataset_points,
        "pairwise": pairwise,
        "by_dataset": dataset_pairwise,
    }


def _stratified_standard_error(
    differences: Mapping[str, np.ndarray],
) -> float:
    variance = 0.0
    dataset_count = len(DATASET_ORDER)
    for dataset in DATASET_ORDER:
        values = differences[dataset]
        if len(values) < 2:
            raise AssetError("stratified SE requires at least two subjects")
        variance += float(np.var(values, ddof=1)) / len(values)
    return math.sqrt(variance / (dataset_count * dataset_count))


def _sign_flip_max_t(
    matrices: Mapping[str, Mapping[str, np.ndarray]],
    inputs: Mapping[str, Any],
) -> dict[str, dict[str, float | int]]:
    """Exploratory one-sided maxT test for all 42 model-minus-TC comparisons.

    Shared subject-level signs preserve dependence among comparisons.  The
    statistic is the unstudentized equal-dataset model-minus-TC mean
    difference.  The maximum statistic controls the complete 42-comparison
    family under the exploratory sign-symmetry null.  This selection-aware
    family includes the eventual descriptive leader.
    """

    comparison_models = tuple(
        row["model"]
        for row in sorted(
            inputs["ranking"].values(), key=lambda row: int(row["rank"])
        )
        if row["model"] != "tcformer"
    )
    if len(comparison_models) != 42 or LEADER not in comparison_models:
        raise AssetError("selection-aware maxT family is not the expected 42 models")

    difference_by_model: dict[str, dict[str, np.ndarray]] = {}
    observed_delta = np.empty(len(comparison_models), dtype=np.float64)
    observed_se = np.empty(len(comparison_models), dtype=np.float64)
    for model_index, model in enumerate(comparison_models):
        differences = {
            dataset: matrices[dataset][model] - matrices[dataset]["tcformer"]
            for dataset in DATASET_ORDER
        }
        difference_by_model[model] = differences
        observed_delta[model_index] = np.mean(
            [float(np.mean(differences[dataset])) for dataset in DATASET_ORDER]
        )
        observed_se[model_index] = _stratified_standard_error(differences)
        if observed_se[model_index] <= 0.0:
            raise AssetError(f"maxT standard error is not positive: {model}")

    flattened = np.column_stack(
        [
            np.concatenate(
                [difference_by_model[model][dataset] for dataset in DATASET_ORDER]
            )
            for model in comparison_models
        ]
    )
    dataset_slices: list[slice] = []
    offset = 0
    for dataset in DATASET_ORDER:
        count = len(matrices[dataset][LEADER])
        dataset_slices.append(slice(offset, offset + count))
        offset += count
    if flattened.shape != (132, 42):
        raise AssetError("maxT paired-difference matrix is not 132 x 42")

    rng = np.random.Generator(np.random.PCG64(SIGN_FLIP_SEED))
    exceed_max = np.zeros(len(comparison_models), dtype=np.int64)
    exceed_raw = np.zeros(len(comparison_models), dtype=np.int64)
    completed = 0
    while completed < SIGN_FLIP_RESAMPLES:
        batch = min(SIGN_FLIP_BATCH_SIZE, SIGN_FLIP_RESAMPLES - completed)
        signs = rng.integers(0, 2, size=(batch, 132), dtype=np.int8)
        signs = signs.astype(np.float64) * 2.0 - 1.0
        permuted_dataset_means: list[np.ndarray] = []
        for subject_slice in dataset_slices:
            values = flattened[subject_slice]
            count = values.shape[0]
            means = signs[:, subject_slice] @ values / count
            permuted_dataset_means.append(means)
        permuted_delta = np.mean(
            np.stack(permuted_dataset_means, axis=0), axis=0
        )
        maximum = np.max(permuted_delta, axis=1)
        exceed_max += np.sum(
            maximum[:, None] >= observed_delta[None, :], axis=0
        )
        exceed_raw += np.sum(
            permuted_delta >= observed_delta[None, :], axis=0
        )
        completed += batch

    result: dict[str, dict[str, float | int]] = {}
    denominator = SIGN_FLIP_RESAMPLES + 1
    for index, model in enumerate(comparison_models):
        result[model] = {
            "estimate": float(observed_delta[index]),
            "stratified_se": float(observed_se[index]),
            "difference_statistic": float(observed_delta[index]),
            "one_sided_unadjusted_p_value": float(
                (int(exceed_raw[index]) + 1) / denominator
            ),
            "one_sided_maxT_adjusted_p_value": float(
                (int(exceed_max[index]) + 1) / denominator
            ),
            "unadjusted_exceedances": int(exceed_raw[index]),
            "maxT_exceedances": int(exceed_max[index]),
        }
    leader_result = result[LEADER]
    if (
        leader_result["unadjusted_exceedances"]
        != EXPECTED_LEADER_RAW_EXCEEDANCES
        or leader_result["maxT_exceedances"]
        != EXPECTED_LEADER_MAX_EXCEEDANCES
    ):
        raise AssetError(
            "selection-aware sign-flip regression differs from the sealed audit"
        )
    return result


def _validate_point_estimates(
    inputs: Mapping[str, Any],
    matrices: Mapping[str, Mapping[str, np.ndarray]],
    bootstrap: Mapping[str, Any],
) -> None:
    overall = inputs["overall"]
    by_dataset = inputs["by_dataset"]
    calibration = inputs["calibration"]
    calibration_metrics = {"ece", "nll", "multiclass_brier"}
    for metric, model_results in bootstrap["metric_absolute"].items():
        for model, result in model_results.items():
            derived = float(result["estimate"])
            sealed = _finite(
                overall[(model,)][metric], label=f"{model}/{metric}"
            )
            if not math.isclose(
                derived, sealed, rel_tol=0.0, abs_tol=1e-14
            ):
                raise AssetError(
                    f"derived and sealed overall estimates differ: {model}/{metric}"
                )
            if metric in calibration_metrics:
                calibration_sealed = _finite(
                    calibration[("equal_dataset_macro", "ALL_DATASETS", model)][
                        metric
                    ],
                    label=f"calibration/{model}/{metric}",
                )
                if not math.isclose(
                    derived,
                    calibration_sealed,
                    rel_tol=0.0,
                    abs_tol=1e-14,
                ):
                    raise AssetError(
                        "derived and calibration-summary estimates differ: "
                        f"{model}/{metric}"
                    )
            for dataset in DATASET_ORDER:
                if metric == "balanced_accuracy":
                    subject_mean = float(np.mean(matrices[dataset][model]))
                else:
                    subject_mean = float(
                        bootstrap["metric_dataset_points"][metric][dataset][model]
                    )
                dataset_sealed = _finite(
                    by_dataset[(dataset, model)][metric],
                    label=f"{dataset}/{model}/{metric}",
                )
                if not math.isclose(
                    subject_mean,
                    dataset_sealed,
                    rel_tol=0.0,
                    abs_tol=1e-14,
                ):
                    raise AssetError(
                        "derived and dataset-summary estimates differ: "
                        f"{dataset}/{model}/{metric}"
                    )
                if metric in calibration_metrics:
                    dataset_calibration_sealed = _finite(
                        calibration[("dataset", dataset, model)][metric],
                        label=f"calibration/{dataset}/{model}/{metric}",
                    )
                    if not math.isclose(
                        subject_mean,
                        dataset_calibration_sealed,
                        rel_tol=0.0,
                        abs_tol=1e-14,
                    ):
                        raise AssetError(
                            "derived and calibration dataset estimates differ: "
                            f"{dataset}/{model}/{metric}"
                        )


def _model_rows(
    inputs: Mapping[str, Any], bootstrap: Mapping[str, Any]
) -> list[dict[str, Any]]:
    overall = inputs["overall"]
    complexity = inputs["complexity"]
    ranking = inputs["ranking"]
    rows: list[dict[str, Any]] = []
    for model in SELECTED_MODELS:
        summary = overall[(model,)]
        resources = complexity[(model,)]
        row: dict[str, Any] = {
            "role": MODEL_ROLES[model],
            "model": model,
            "label": MODEL_LABELS[model],
            "rank": int(ranking[(model,)]["rank"]),
            "accuracy": _finite(summary["accuracy"], label=f"{model}/accuracy"),
            "balanced_accuracy": _finite(
                summary["balanced_accuracy"], label=f"{model}/balanced_accuracy"
            ),
            "parameter_count_median": int(
                round(
                    _finite(
                        resources["parameter_count_median"],
                        label=f"{model}/parameter_count_median",
                    )
                )
            ),
            "inference_ms_per_trial_median": _finite(
                resources["inference_ms_per_trial_median"],
                label=f"{model}/inference_ms_per_trial_median",
            ),
            "job_total_seconds_median": _finite(
                resources["job_total_seconds_median"],
                label=f"{model}/job_total_seconds_median",
            ),
        }
        for metric, model_results in bootstrap["metric_absolute"].items():
            if model not in model_results:
                continue
            metric_result = model_results[model]
            interval = metric_result["bootstrap_ci95"]
            if metric not in row:
                row[metric] = _finite(
                    summary[metric], label=f"{model}/{metric}"
                )
            row[f"{metric}_shared_bootstrap_se"] = float(
                metric_result["bootstrap_se"]
            )
            row[f"{metric}_shared_bootstrap_ci95"] = [
                float(interval[0]),
                float(interval[1]),
            ]
        rows.append(row)
    return rows


def _statistics_payload(
    inputs: Mapping[str, Any],
    model_rows: Sequence[Mapping[str, Any]],
    bootstrap: Mapping[str, Any],
    max_t: Mapping[str, Mapping[str, float | int]],
    sealed_context: Mapping[str, Any],
) -> dict[str, Any]:
    pairwise = []
    for comparator in COMPARATORS:
        boot = bootstrap["pairwise"][comparator]
        row: dict[str, Any] = {
            "leader": LEADER,
            "comparator": comparator,
            "balanced_accuracy_difference": float(boot["estimate"]),
            "shared_bootstrap_se": float(boot["bootstrap_se"]),
            "shared_bootstrap_ci95": [
                float(value) for value in boot["bootstrap_ci95"]
            ],
            "sign_flip_stratified_se": None,
            "sign_flip_difference_statistic": None,
            "sign_flip_one_sided_raw_p_value": None,
            "sign_flip_all42_maxT_adjusted_p_value": None,
        }
        if comparator == "tcformer":
            test = max_t[LEADER]
            if not math.isclose(
                float(boot["estimate"]),
                float(test["estimate"]),
                rel_tol=0.0,
                abs_tol=1e-14,
            ):
                raise AssetError("bootstrap and maxT leader-minus-TC deltas differ")
            row.update(
                {
                    "sign_flip_stratified_se": float(test["stratified_se"]),
                    "sign_flip_difference_statistic": float(
                        test["difference_statistic"]
                    ),
                    "sign_flip_one_sided_raw_p_value": float(
                        test["one_sided_unadjusted_p_value"]
                    ),
                    "sign_flip_all42_maxT_adjusted_p_value": float(
                        test["one_sided_maxT_adjusted_p_value"]
                    ),
                }
            )
        pairwise.append(row)

    max_t_rows = [
        {
            "model": model,
            "comparator": "tcformer",
            "balanced_accuracy_difference": float(row["estimate"]),
            "stratified_se": float(row["stratified_se"]),
            "difference_statistic": float(row["difference_statistic"]),
            "one_sided_raw_p_value": float(row["one_sided_unadjusted_p_value"]),
            "one_sided_maxT_adjusted_p_value": float(
                row["one_sided_maxT_adjusted_p_value"]
            ),
        }
        for model, row in max_t.items()
    ]
    return {
        "schema": "tbme-manuscript-statistics-v4",
        "evidence_scope": "opened-development common-grid results only",
        "confirmation_evidence": False,
        "source": {
            "analysis_directory": "dl/results/common_grid_v6/analysis",
            "plan_sha256": inputs["manifest"]["plan_sha256"],
            "analysis_contract_sha256": inputs["manifest"][
                "analysis_contract_sha256"
            ],
            "input_sha256": dict(sorted(inputs["input_hashes"].items())),
        },
        "estimand": {
            "metric": "balanced_accuracy",
            "aggregation": (
                "folds concatenated within subject-seed; seeds averaged within "
                "subject; subjects averaged within dataset; five datasets weighted equally"
            ),
            "datasets": list(DATASET_ORDER),
            "subjects_total": 132,
            "models_in_common_grid": 43,
        },
        "shared_seed_subject_stratified_bootstrap": {
            "method": "subject-stratified nonparametric percentile bootstrap",
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "generator": "NumPy PCG64",
            "metrics_and_models": {
                metric: list(models)
                for metric, models in SHARED_BOOTSTRAP_METRIC_MODELS.items()
            },
            "source_sha256": {
                name: inputs["input_hashes"][name]
                for name in (
                    "subject_metrics.csv",
                    "overall_summary.csv",
                    "calibration_summary.csv",
                )
            },
            "resampling": (
                "subjects sampled with replacement independently inside each dataset; "
                "the exact per-dataset index arrays are shared across every metric "
                "and model; datasets weighted equally"
            ),
            "se_definition": "sample standard deviation of bootstrap estimates (ddof=1)",
            "interval": "2.5th and 97.5th percentiles using linear quantiles",
            "purpose": (
                "selected-model metric standard errors and absolute intervals, "
                "balanced-accuracy leader pairwise intervals, and dataset-delta "
                "figure intervals"
            ),
        },
        "sealed_tcformer_fixed_suite_context": dict(sealed_context),
        "exploratory_sign_flip_maxT": {
            "method": "paired subject-level shared-sign maximum-difference test",
            "resamples": SIGN_FLIP_RESAMPLES,
            "seed": SIGN_FLIP_SEED,
            "batch_size": SIGN_FLIP_BATCH_SIZE,
            "generator": "NumPy PCG64",
            "family_size": 42,
            "family": "all non-TCFormer model-minus-TCFormer comparisons",
            "test_statistic": (
                "unstudentized equal-dataset balanced-accuracy difference"
            ),
            "tail": "one-sided greater",
            "p_value_correction": "add-one Monte Carlo correction",
            "status": (
                "exploratory and post hoc only; sign symmetry is assumed and all "
                "model selection used the same opened development suite"
            ),
            "results": max_t_rows,
        },
        "selected_models": list(model_rows),
        "leader_pairwise_shared_bootstrap_comparisons": pairwise,
        "leader_pairwise_by_dataset_shared_bootstrap": {
            comparator: {
                dataset: {
                    "balanced_accuracy_difference": float(row["estimate"]),
                    "shared_bootstrap_se": float(row["bootstrap_se"]),
                    "shared_bootstrap_ci95": [
                        float(value) for value in row["bootstrap_ci95"]
                    ],
                    "subjects": int(row["subjects"]),
                }
                for dataset, row in dataset_rows.items()
            }
            for comparator, dataset_rows in bootstrap["by_dataset"].items()
        },
        "interpretation": (
            "Descriptive fixed-protocol evidence; not independent confirmation, "
            "a clinical-performance claim, or a global state-of-the-art claim."
        ),
        "runtime": {
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }


def _float_text(value: float | int | None) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    return format(float(value), ".12g")


def _selected_metrics_csv(
    model_rows: Sequence[Mapping[str, Any]],
    statistics: Mapping[str, Any],
) -> bytes:
    fields = (
        "record_type",
        "role",
        "model",
        "comparator",
        "rank",
        "accuracy",
        "accuracy_shared_bootstrap_se",
        "accuracy_shared_bootstrap_ci95_low",
        "accuracy_shared_bootstrap_ci95_high",
        "balanced_accuracy",
        "balanced_accuracy_shared_bootstrap_se",
        "balanced_accuracy_shared_bootstrap_ci95_low",
        "balanced_accuracy_shared_bootstrap_ci95_high",
        "ece",
        "ece_shared_bootstrap_se",
        "ece_shared_bootstrap_ci95_low",
        "ece_shared_bootstrap_ci95_high",
        "nll",
        "nll_shared_bootstrap_se",
        "nll_shared_bootstrap_ci95_low",
        "nll_shared_bootstrap_ci95_high",
        "multiclass_brier",
        "multiclass_brier_shared_bootstrap_se",
        "multiclass_brier_shared_bootstrap_ci95_low",
        "multiclass_brier_shared_bootstrap_ci95_high",
        "parameter_count_median",
        "inference_ms_per_trial_median",
        "job_total_seconds_median",
        "balanced_accuracy_difference",
        "balanced_accuracy_difference_shared_bootstrap_se",
        "balanced_accuracy_difference_shared_bootstrap_ci95_low",
        "balanced_accuracy_difference_shared_bootstrap_ci95_high",
        "sealed_tcformer_balanced_accuracy_difference",
        "sealed_tcformer_fixed_suite_ci95_low",
        "sealed_tcformer_fixed_suite_ci95_high",
        "sealed_tcformer_bootstrap_resamples",
        "sealed_tcformer_bootstrap_seed",
        "sealed_tcformer_source_sha256",
        "sign_flip_difference_statistic",
        "sign_flip_one_sided_raw_p_value",
        "sign_flip_all42_maxT_adjusted_p_value",
        "datasets",
        "subjects_total",
    )
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    sealed_context = statistics["sealed_tcformer_fixed_suite_context"]
    sealed_by_model = {
        row["model"]: row for row in sealed_context["results"]
    }
    for row in model_rows:
        output_row = {
            "record_type": "model",
            "role": row["role"],
            "model": row["model"],
            "rank": row["rank"],
            "accuracy": _float_text(row["accuracy"]),
            "balanced_accuracy": _float_text(row["balanced_accuracy"]),
            "parameter_count_median": row["parameter_count_median"],
            "inference_ms_per_trial_median": _float_text(
                row["inference_ms_per_trial_median"]
            ),
            "job_total_seconds_median": _float_text(
                row["job_total_seconds_median"]
            ),
            "datasets": 5,
            "subjects_total": 132,
        }
        for metric in SHARED_BOOTSTRAP_METRIC_MODELS:
            se_key = f"{metric}_shared_bootstrap_se"
            interval_key = f"{metric}_shared_bootstrap_ci95"
            if se_key not in row:
                continue
            interval = row[interval_key]
            output_row.update(
                {
                    metric: _float_text(row[metric]),
                    se_key: _float_text(row[se_key]),
                    f"{metric}_shared_bootstrap_ci95_low": _float_text(
                        interval[0]
                    ),
                    f"{metric}_shared_bootstrap_ci95_high": _float_text(
                        interval[1]
                    ),
                }
            )
        sealed = sealed_by_model.get(row["model"])
        if sealed is not None:
            sealed_interval = sealed[
                "descriptive_fixed_suite_bootstrap_interval95"
            ]
            output_row.update(
                {
                    "sealed_tcformer_balanced_accuracy_difference": _float_text(
                        sealed["balanced_accuracy_difference"]
                    ),
                    "sealed_tcformer_fixed_suite_ci95_low": _float_text(
                        sealed_interval[0]
                    ),
                    "sealed_tcformer_fixed_suite_ci95_high": _float_text(
                        sealed_interval[1]
                    ),
                    "sealed_tcformer_bootstrap_resamples": sealed[
                        "bootstrap_resamples"
                    ],
                    "sealed_tcformer_bootstrap_seed": sealed["bootstrap_seed"],
                    "sealed_tcformer_source_sha256": sealed_context[
                        "source_sha256"
                    ],
                }
            )
        writer.writerow(output_row)
    for row in statistics["leader_pairwise_shared_bootstrap_comparisons"]:
        interval = row["shared_bootstrap_ci95"]
        writer.writerow(
            {
                "record_type": "leader_comparison",
                "role": "leader_minus_comparator",
                "model": row["leader"],
                "comparator": row["comparator"],
                "balanced_accuracy_difference": _float_text(
                    row["balanced_accuracy_difference"]
                ),
                "balanced_accuracy_difference_shared_bootstrap_se": _float_text(
                    row["shared_bootstrap_se"]
                ),
                "balanced_accuracy_difference_shared_bootstrap_ci95_low": (
                    _float_text(interval[0])
                ),
                "balanced_accuracy_difference_shared_bootstrap_ci95_high": (
                    _float_text(interval[1])
                ),
                "sign_flip_difference_statistic": _float_text(
                    row["sign_flip_difference_statistic"]
                ),
                "sign_flip_one_sided_raw_p_value": _float_text(
                    row["sign_flip_one_sided_raw_p_value"]
                ),
                "sign_flip_all42_maxT_adjusted_p_value": _float_text(
                    row["sign_flip_all42_maxT_adjusted_p_value"]
                ),
                "datasets": 5,
                "subjects_total": 132,
            }
        )
    return output.getvalue().encode("utf-8")


def _pdf_metadata(title: str, subject: str) -> dict[str, Any]:
    return {
        "Title": title,
        "Author": "Motor-imagery EEG research team",
        "Subject": subject,
        "Keywords": "motor imagery, EEG, reproducible benchmark",
        "Creator": "tbme-manuscript/scripts/build_assets.py",
        "Producer": f"Matplotlib {matplotlib.__version__} PDF backend",
        "CreationDate": FIXED_PDF_TIME,
        "ModDate": FIXED_PDF_TIME,
    }


def _save_pdf(fig: Any, path: Path, *, title: str, subject: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.04,
        metadata=_pdf_metadata(title, subject),
    )
    plt.close(fig)


def _rounded_box(
    ax: Any,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str,
    edgecolor: str = COLOR_NAVY,
    fontsize: float = 7.4,
    linewidth: float = 1.0,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        transform=ax.transAxes,
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
        clip_on=False,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        linespacing=1.25,
    )


def _arrow(
    ax: Any,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = COLOR_NAVY,
    style: str = "-|>",
    connectionstyle: str = "arc3",
    linewidth: float = 1.1,
) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        xycoords=ax.transAxes,
        textcoords=ax.transAxes,
        arrowprops={
            "arrowstyle": style,
            "color": color,
            "linewidth": linewidth,
            "shrinkA": 2,
            "shrinkB": 2,
            "connectionstyle": connectionstyle,
        },
    )


def _platform_figure(path: Path) -> None:
    with plt.rc_context(RC_PARAMS):
        fig, ax = plt.subplots(figsize=(7.16, 3.35))
        ax.set_axis_off()
        ax.text(
            0.5,
            0.955,
            "End-to-end motor-imagery BCI platform",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=11,
            fontweight="bold",
            color=COLOR_NAVY,
        )
        blocks = (
            (0.045, 0.61, "1  Acquisition\nOpenBCI Cyton+Daisy, 125 Hz\n15-site 10-20 montage", "#EAF2F8"),
            (0.365, 0.61, "2  Cue presentation\n60 cues/run, 50/50 left/right\nplan 2 s | task 2 s | rest 2 s", "#E9F5F3"),
            (0.685, 0.61, "3  Recordings\n4 runs/participant\nannotated FIF + markers", "#F8F0DE"),
            (0.685, 0.245, "4  Decoder training\nEA-FBCSP / Riemannian TS\n45-s in-context calibration", "#F8E9E5"),
            (0.365, 0.245, "5  Online inference\n2.0-s window at 5 Hz\nrecenter; M-of-N + dwell gate", "#F0ECF6"),
            (0.045, 0.245, "6  Assistive control\nnetworked turn commands\ndecision-point navigation", "#EAF2F8"),
        )
        width, height = 0.27, 0.235
        for x, y, text, color in blocks:
            _rounded_box(ax, x, y, width, height, text, facecolor=color)
        _arrow(ax, (0.315, 0.728), (0.365, 0.728))
        _arrow(ax, (0.635, 0.728), (0.685, 0.728))
        _arrow(ax, (0.82, 0.61), (0.82, 0.48))
        _arrow(ax, (0.685, 0.363), (0.635, 0.363))
        _arrow(ax, (0.365, 0.363), (0.315, 0.363))
        ax.text(
            0.5,
            0.105,
            "The same annotated recordings feed the frozen 43-model offline benchmark (Fig. 2)",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=7.2,
            color="#4E5B66",
        )
        _save_pdf(
            fig,
            path,
            title="End-to-end motor-imagery BCI platform",
            subject="Acquisition, cue presentation, online decoding, and assistive control",
        )


def _workflow_figure(path: Path) -> None:
    with plt.rc_context(RC_PARAMS):
        fig, ax = plt.subplots(figsize=(7.16, 3.35))
        ax.set_axis_off()
        ax.text(
            0.5,
            0.955,
            "Score-blind common-grid workflow",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=11,
            fontweight="bold",
            color=COLOR_NAVY,
        )
        blocks = (
            (0.045, 0.61, "1  Inputs\n5 datasets; 132 participants\nfrozen source identities", "#EAF2F8"),
            (0.365, 0.61, "2  Preprocessing\n4-40 Hz; 128 Hz\ndataset-locked epochs", "#E9F5F3"),
            (0.685, 0.61, "3  Split contract\nfixed train/validation/test\nparticipant and fold isolation", "#F8F0DE"),
            (0.685, 0.245, "4  Common recipe\n43 architectures\n5 deterministic seeds", "#F8E9E5"),
            (0.365, 0.245, "5  Formal execution\n96,320 GPU jobs\nimmutable receipts", "#F0ECF6"),
            (0.045, 0.245, "6  Primary aggregation\nfold concat; seed mean\nparticipant mean; dataset mean", "#EAF2F8"),
        )
        width, height = 0.27, 0.235
        for x, y, text, color in blocks:
            _rounded_box(ax, x, y, width, height, text, facecolor=color)
        _arrow(ax, (0.315, 0.728), (0.365, 0.728))
        _arrow(ax, (0.635, 0.728), (0.685, 0.728))
        _arrow(ax, (0.82, 0.61), (0.82, 0.48))
        _arrow(ax, (0.685, 0.363), (0.635, 0.363))
        _arrow(ax, (0.365, 0.363), (0.315, 0.363))
        ax.text(
            0.5,
            0.105,
            "Opened-development evidence only  |  bounded source-bound reviewer replay  |  no independent confirmation",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=7.2,
            color="#4E5B66",
        )
        _save_pdf(
            fig,
            path,
            title="Score-blind common-grid workflow",
            subject="Reproducible five-dataset motor-imagery EEG benchmark workflow",
        )


def _architecture_figure(path: Path) -> None:
    with plt.rc_context(RC_PARAMS):
        fig, ax = plt.subplots(figsize=(7.16, 4.05))
        ax.set_axis_off()
        ax.text(
            0.5,
            0.965,
            "CardinalFBC compact-dynamics continuation",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=11,
            fontweight="bold",
            color=COLOR_NAVY,
        )
        ax.text(
            0.5,
            0.915,
            "Leader configuration: 31-anchor atlas and continuation scale 0.25",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=7.4,
            color="#4E5B66",
        )

        _rounded_box(
            ax,
            0.015,
            0.40,
            0.13,
            0.19,
            "Raw EEG x(t)\n+ electrode\ncoordinates r",
            facecolor="#EAF2F8",
            fontsize=7.2,
        )
        upper = (
            (0.20, "FBC floor\n9 fixed FIR bands"),
            (0.39, "Cardinal field\n32 spatial sources\n31 anchors"),
            (0.58, "BN + SiLU\n4 log-variance\nviews"),
        )
        lower = (
            (0.20, "Compact dynamics\n24 FIRs; length 25"),
            (0.39, "Cardinal energy\n24 sources\npool 75 / stride 15"),
            (0.58, "Signed dynamics\n12 channels; kernel 15\nscale x 0.25"),
        )
        for x, text in upper:
            _rounded_box(
                ax, x, 0.655, 0.155, 0.16, text, facecolor="#E9F5F3", fontsize=7.0
            )
        for x, text in lower:
            _rounded_box(
                ax, x, 0.245, 0.155, 0.18, text, facecolor="#F8F0DE", fontsize=6.9
            )
        _rounded_box(
            ax,
            0.775,
            0.43,
            0.085,
            0.13,
            "Feature\nconcat",
            facecolor="#F0ECF6",
            fontsize=7.0,
        )
        _rounded_box(
            ax,
            0.89,
            0.40,
            0.095,
            0.19,
            "Single expanded\nmax-norm head\nclass logits",
            facecolor="#F8E9E5",
            fontsize=6.8,
            edgecolor=COLOR_CORAL,
            linewidth=1.2,
        )

        _arrow(ax, (0.145, 0.495), (0.20, 0.735), connectionstyle="arc3,rad=-0.12")
        _arrow(ax, (0.145, 0.495), (0.20, 0.335), connectionstyle="arc3,rad=0.12")
        _arrow(ax, (0.355, 0.735), (0.39, 0.735))
        _arrow(ax, (0.545, 0.735), (0.58, 0.735))
        _arrow(ax, (0.355, 0.335), (0.39, 0.335))
        _arrow(ax, (0.545, 0.335), (0.58, 0.335))
        _arrow(ax, (0.735, 0.735), (0.775, 0.52), connectionstyle="arc3,rad=0.10")
        _arrow(ax, (0.735, 0.335), (0.775, 0.47), connectionstyle="arc3,rad=-0.10")
        _arrow(ax, (0.86, 0.495), (0.89, 0.495))

        note = FancyBboxPatch(
            (0.19, 0.065),
            0.62,
            0.095,
            boxstyle="round,pad=0.01,rounding_size=0.015",
            transform=ax.transAxes,
            linewidth=0.9,
            linestyle="--",
            edgecolor=COLOR_CORAL,
            facecolor="#FFF9F7",
        )
        ax.add_patch(note)
        ax.text(
            0.50,
            0.112,
            "New head columns start at zero: the paired FBC/Cardinal floor mapping is exact at initialization.",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=7.0,
            color="#703426",
        )
        _save_pdf(
            fig,
            path,
            title="CardinalFBC compact-dynamics architecture",
            subject="Parallel FBC floor and compact dynamics continuation with a shared head",
        )


def _efficiency_figure(path: Path, inputs: Mapping[str, Any]) -> None:
    overall = inputs["overall"]
    complexity = inputs["complexity"]
    ranking = inputs["ranking"]
    models = [
        row["model"]
        for row in sorted(
            (value for value in ranking.values()), key=lambda row: int(row["rank"])
        )
    ]
    parameters = np.asarray(
        [
            _finite(complexity[(model,)]["parameter_count_median"], label=model)
            for model in models
        ]
    )
    balanced = np.asarray(
        [
            100.0
            * _finite(overall[(model,)]["balanced_accuracy"], label=model)
            for model in models
        ]
    )

    with plt.rc_context(RC_PARAMS):
        fig, ax = plt.subplots(figsize=(7.16, 4.35))
        ax.scatter(
            parameters,
            balanced,
            s=23,
            color=COLOR_GRAY,
            alpha=0.72,
            edgecolors="white",
            linewidths=0.35,
            zorder=2,
        )
        offsets = {
            LEADER: (8, 9),
            "cardinal_dynamics_sinc_extended": (8, -17),
            "fbcnet": (-58, 12),
            "cardinal_fbc": (-60, -18),
            "tcformer": (8, -15),
        }
        for model in SELECTED_MODELS:
            x = _finite(complexity[(model,)]["parameter_count_median"], label=model)
            y = 100.0 * _finite(overall[(model,)]["balanced_accuracy"], label=model)
            marker = SELECTED_MARKERS[model]
            size = 105 if marker == "*" else 48
            ax.scatter(
                [x],
                [y],
                s=size,
                marker=marker,
                color=SELECTED_COLORS[model],
                edgecolors="white",
                linewidths=0.75,
                zorder=4,
            )
            ax.annotate(
                MODEL_LABELS[model],
                xy=(x, y),
                xytext=offsets[model],
                textcoords="offset points",
                ha="left" if offsets[model][0] >= 0 else "right",
                va="bottom" if offsets[model][1] >= 0 else "top",
                fontsize=7.0,
                color=SELECTED_COLORS[model],
                fontweight="bold" if model == LEADER else "normal",
                arrowprops={
                    "arrowstyle": "-",
                    "linewidth": 0.55,
                    "color": SELECTED_COLORS[model],
                    "shrinkA": 1,
                    "shrinkB": 3,
                },
                zorder=5,
            )
        ax.set_xscale("log")
        ax.set_xlabel("Median trainable parameters (log scale)")
        ax.set_ylabel("Equal-dataset balanced accuracy (%)")
        ax.set_title("Accuracy-efficiency landscape across all 43 common-recipe models")
        ax.grid(True, which="major", color="#D8DEE3", linewidth=0.55, alpha=0.8)
        ax.grid(True, which="minor", axis="x", color="#E9EDF0", linewidth=0.35)
        ax.set_axisbelow(True)
        ax.margins(x=0.13, y=0.10)
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=COLOR_GRAY,
                markeredgecolor="white",
                markersize=5.5,
                label="Other common-grid models",
            )
        ]
        ax.legend(handles=handles, loc="lower right", frameon=False)
        fig.text(
            0.5,
            0.012,
            "Point estimates concatenate folds, then average seeds, subjects, and five datasets equally.",
            ha="center",
            va="bottom",
            fontsize=6.8,
            color="#4E5B66",
        )
        fig.subplots_adjust(bottom=0.15, top=0.90, left=0.11, right=0.98)
        _save_pdf(
            fig,
            path,
            title="Accuracy-efficiency landscape",
            subject="Balanced accuracy versus model size for all 43 common-grid models",
        )


def _dataset_delta_figure(path: Path, bootstrap: Mapping[str, Any]) -> None:
    colors = [COLOR_GOLD, COLOR_BLUE, COLOR_TEAL, COLOR_PURPLE]
    markers = ["D", "s", "o", "^"]
    offsets = np.asarray((-0.24, -0.08, 0.08, 0.24))
    base_y = np.arange(len(DATASET_ORDER), dtype=np.float64)[::-1]

    with plt.rc_context(RC_PARAMS):
        fig, ax = plt.subplots(figsize=(7.16, 4.35))
        all_bounds: list[float] = []
        for index, comparator in enumerate(COMPARATORS):
            estimates = []
            lows = []
            highs = []
            for dataset in DATASET_ORDER:
                row = bootstrap["by_dataset"][comparator][dataset]
                estimates.append(100.0 * float(row["estimate"]))
                lows.append(100.0 * float(row["bootstrap_ci95"][0]))
                highs.append(100.0 * float(row["bootstrap_ci95"][1]))
            estimates_array = np.asarray(estimates)
            lows_array = np.asarray(lows)
            highs_array = np.asarray(highs)
            all_bounds.extend(lows_array.tolist())
            all_bounds.extend(highs_array.tolist())
            ax.errorbar(
                estimates_array,
                base_y + offsets[index],
                xerr=np.vstack(
                    (estimates_array - lows_array, highs_array - estimates_array)
                ),
                fmt=markers[index],
                markersize=5.1,
                markerfacecolor=colors[index],
                markeredgecolor="white",
                markeredgewidth=0.55,
                ecolor=colors[index],
                elinewidth=1.0,
                capsize=2.0,
                label=MODEL_LABELS[comparator],
                zorder=3,
            )
        ax.axvline(0.0, color=COLOR_INK, linewidth=0.9, linestyle="--", zorder=1)
        ax.set_yticks(base_y, [DATASET_LABELS[name] for name in DATASET_ORDER])
        ax.set_ylim(-0.65, len(DATASET_ORDER) - 0.35)
        bound = max(abs(min(all_bounds)), abs(max(all_bounds)))
        ax.set_xlim(-bound * 1.12, bound * 1.12)
        ax.set_xlabel("Leader minus comparator balanced accuracy (percentage points)")
        ax.set_title("Leader advantage by dataset")
        ax.grid(True, axis="x", color="#D8DEE3", linewidth=0.55, alpha=0.8)
        ax.set_axisbelow(True)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=2,
            frameon=False,
            columnspacing=1.5,
            handletextpad=0.4,
        )
        fig.text(
            0.5,
            0.012,
            "Intervals: 100,000 subject-stratified percentile bootstrap resamples; seed 20260729.",
            ha="center",
            va="bottom",
            fontsize=6.8,
            color="#4E5B66",
        )
        fig.subplots_adjust(bottom=0.15, top=0.83, left=0.17, right=0.98)
        _save_pdf(
            fig,
            path,
            title="Leader advantage by dataset",
            subject="Dataset-specific balanced-accuracy differences with bootstrap intervals",
        )


def _build_into(root: Path) -> None:
    _validate_runtime()
    inputs = _load_inputs()
    sealed_context = _sealed_tcformer_fixed_suite_context(inputs)
    matrices = _subject_matrices(inputs)
    additional_metric_matrices = _selected_additional_metric_matrices(
        inputs, matrices
    )
    bootstrap = _bootstrap_statistics(matrices, additional_metric_matrices)
    _validate_point_estimates(inputs, matrices, bootstrap)
    max_t = _sign_flip_max_t(matrices, inputs)
    model_rows = _model_rows(inputs, bootstrap)
    _validate_manuscript_reconciliation(sealed_context, model_rows)
    statistics = _statistics_payload(
        inputs, model_rows, bootstrap, max_t, sealed_context
    )

    generated = root / "generated"
    figures = root / "figures"
    generated.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    statistics_bytes = (
        json.dumps(
            statistics,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    (generated / "statistics.json").write_bytes(statistics_bytes)
    (generated / "selected_metrics.csv").write_bytes(
        _selected_metrics_csv(model_rows, statistics)
    )

    _platform_figure(figures / "platform.pdf")
    _workflow_figure(figures / "workflow.pdf")
    _architecture_figure(figures / "architecture.pdf")
    _efficiency_figure(figures / "efficiency_scatter.pdf", inputs)
    _dataset_delta_figure(figures / "dataset_deltas.pdf", bootstrap)


def _asset_hashes(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in ASSET_PATHS:
        path = root / relative
        if not path.is_file():
            raise AssetError(f"asset is missing: {path}")
        hashes[relative] = _sha256_file(path)
    return hashes


def _publish(staging: Path) -> None:
    for relative in ASSET_PATHS:
        source = staging / relative
        destination = MANUSCRIPT_ROOT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.stage")
        temporary.write_bytes(source.read_bytes())
        os.replace(temporary, destination)


def _check(staging: Path) -> None:
    failures: list[str] = []
    for relative in ASSET_PATHS:
        expected = MANUSCRIPT_ROOT / relative
        observed = staging / relative
        if not expected.is_file():
            failures.append(f"missing committed asset: {relative}")
            continue
        expected_bytes = expected.read_bytes()
        observed_bytes = observed.read_bytes()
        if expected_bytes != observed_bytes:
            failures.append(
                f"{relative}: expected {_sha256_bytes(expected_bytes)}, "
                f"rebuilt {_sha256_bytes(observed_bytes)}"
            )
    if failures:
        raise AssetError("deterministic asset check failed:\n" + "\n".join(failures))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="rebuild in temporary storage and require byte-identical outputs",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    with tempfile.TemporaryDirectory(
        prefix="tbme-manuscript-assets-", dir=REPOSITORY_ROOT
    ) as temporary:
        staging = Path(temporary)
        _build_into(staging)
        if arguments.check:
            _check(staging)
            status = "verified"
        else:
            _publish(staging)
            status = "built"
    hashes = _asset_hashes(MANUSCRIPT_ROOT)
    print(f"{status} {len(hashes)} deterministic manuscript assets")
    for relative, digest in sorted(hashes.items()):
        print(f"{digest}  {relative}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssetError as error:
        raise SystemExit(str(error)) from error
