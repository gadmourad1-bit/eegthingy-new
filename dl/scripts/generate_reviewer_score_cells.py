#!/usr/bin/env python3
"""Generate the deterministic reviewer-replay score-cell CSV.

The generator reads only the manifest-bound common-grid v6 dataset and
overall summary tables.  It does not recompute a metric, inspect a prediction,
or launch a replay.  Numeric score strings are copied byte-for-byte from the
authoritative CSV cells and paired with exact bounded-replay command templates.

By default the CSV is written under ``reviewer/``. ``--output -`` writes to
standard output. The sealed inputs and the ``results/`` tree can never be
output targets. ``--check`` verifies the existing catalog without rewriting it.
"""

from __future__ import annotations

import argparse
import copy
import csv
from decimal import Decimal, InvalidOperation, localcontext
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMMON_RELATIVE = Path("results/common_grid_v6")
ANALYSIS_RELATIVE = COMMON_RELATIVE / "analysis"
PLAN_RELATIVE = COMMON_RELATIVE / "plan.json"
PLAN_SIDECAR_RELATIVE = COMMON_RELATIVE / "plan.sha256"
AUDIT_RELATIVE = COMMON_RELATIVE / "final_audit.json"
MANIFEST_RELATIVE = ANALYSIS_RELATIVE / "manifest.json"
DATASET_TABLE_RELATIVE = ANALYSIS_RELATIVE / "dataset_summary.csv"
OVERALL_TABLE_RELATIVE = ANALYSIS_RELATIVE / "overall_summary.csv"
RANKING_TABLE_RELATIVE = ANALYSIS_RELATIVE / "model_ranking.csv"
JOB_METRICS_RELATIVE = ANALYSIS_RELATIVE / "job_metrics.csv"
DEFAULT_OUTPUT_RELATIVE = Path("reviewer/reviewer_score_cells.csv")

EXPECTED_PLAN_SHA256 = (
    "65b93b7e5d09cfc30fb6ce28156368e66aec1503f4efa37faa3189e25ebbc3b1"
)
EXPECTED_RAW_SHA256 = {
    PLAN_RELATIVE.as_posix(): (
        "548c0bb707ac99debfafbc9082dd322aec7eb799e69f1ad69e25231eeb75873f"
    ),
    AUDIT_RELATIVE.as_posix(): (
        "21908c268e15907806271245a1d14857b4e6704383b9bd6aadd5d253824af212"
    ),
    MANIFEST_RELATIVE.as_posix(): (
        "ed1070f69367e36109c473c193b4f898de40ad689a8d367ac3d9e7471a82a1cd"
    ),
    DATASET_TABLE_RELATIVE.as_posix(): (
        "0f24934e47c275ffb66b06e78266cb7fc1883f3046da876a5ed50089c09686cf"
    ),
    OVERALL_TABLE_RELATIVE.as_posix(): (
        "d45dd54801f4ba9074bba808cd7076bd04c9a5267944ec4f7af274de96a11c6f"
    ),
    RANKING_TABLE_RELATIVE.as_posix(): (
        "adddfd4687e09710b2867252fc873687f48ef073c555a9c0b604af9dd31e4e32"
    ),
    JOB_METRICS_RELATIVE.as_posix(): (
        "fda26e3a545811d1f9d4b4ed2b9885cff0bab3505f6be8e6eb97491bb3355e1d"
    ),
}

EXPECTED_MODEL_COUNT = 43
EXPECTED_DATASET_COUNT = 5
EXPECTED_GRID_JOB_COUNT = 96_320
SAFE_ID = re.compile(r"[a-z0-9_]+")
PLACEHOLDERS = {
    "cache_root": "${CACHE_ROOT:?set CACHE_ROOT}",
    "reviewer_root": "${REVIEWER_ROOT:?set REVIEWER_ROOT}",
    "gpu": "${GPU:?set GPU}",
}
DATA_ACCESS = {
    "local_exp4": (
        "private_institutional_authorization_required",
        "false",
    ),
    "bnci2014_001": (
        "public_official_record_cc_by_nd_4_0_subject_to_terms",
        "true",
    ),
    "bnci2014_004": (
        "public_official_record_cc_by_nd_4_0_subject_to_terms",
        "true",
    ),
    "cho2017": (
        "public_record_exact_acquired_byte_terms_unresolved",
        "true",
    ),
    "physionet_mi": (
        "public_official_record_odc_by_1_0_subject_to_terms",
        "true",
    ),
    "overall": (
        "mixed_includes_private_local_exp4_authorization_required",
        "false",
    ),
}
TIMING_CAVEAT = (
    "sum of sealed job_total_seconds measured on the formal four-RTX-A5000 "
    "workstation; not a wall-clock estimate or cross-hardware promise"
)
EVIDENCE_CAVEAT = (
    "opened-development evidence only; not independent confirmation, a clinical "
    "result, or a global SOTA claim"
)
STRICT_SCORE_CAVEAT = (
    "strict score match requires the exact selected Cartesian replay and checks "
    "accuracy plus balanced accuracy; it does not imply bit-exact predictions"
)
RANK_CAVEAT = (
    "rank is sealed-table context, not established by replaying one cell or one "
    "model; rank confirmation requires all 43 complete model rows"
)
IDENTITY_CAVEAT = (
    "fixed-protocol replication is not exact execution identity unless source, "
    "cache, split, lock, CUDA, driver, and physical GPU identities all match"
)
ACCURACY_RANK_DERIVATION = (
    "sealed overall accuracy descending; frozen plan architecture order breaks exact ties"
)

DATASET_FIELDS = (
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
OVERALL_FIELDS = (
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
RANKING_FIELDS = ("rank", "model", "metric", "value", "tie_break", "status")
JOB_METRIC_FIELDS = (
    "dataset",
    "model",
    "subject",
    "fold",
    "seed",
    "job_id",
    "test_count",
    "accuracy",
    "balanced_accuracy",
    "chance_normalized_balanced_accuracy",
    "macro_f1",
    "cohen_kappa",
    "ovr_macro_auroc",
    "nll",
    "multiclass_brier",
    "ece",
    "selected_epoch_index",
    "selected_epoch_count",
    "selection_epochs_run",
    "refit_epochs_run",
    "parameter_count",
    "selection_fit_seconds",
    "refit_fit_seconds",
    "test_inference_seconds",
    "job_total_seconds",
    "cuda_peak_memory_bytes",
    "inference_ms_per_trial",
)
OUTPUT_FIELDS = (
    "row_id",
    "score_cell_id",
    "model",
    "primary_ba_rank",
    "primary_ba_rank_metric",
    "primary_ba_rank_status",
    "descriptive_accuracy_rank",
    "descriptive_accuracy_rank_derivation",
    "score_level",
    "dataset",
    "replay_scope",
    "subject_count",
    "folds_per_subject",
    "subject_fold_count",
    "seed_count",
    "job_count",
    "accuracy",
    "balanced_accuracy",
    "accuracy_percent_2dp",
    "accuracy_percent_3dp",
    "balanced_accuracy_percent_2dp",
    "balanced_accuracy_percent_3dp",
    "aggregation",
    "historical_a5000_summed_hours",
    "historical_timing_caveat",
    "data_access",
    "general_access",
    "cache_redistributable",
    "evidence_caveat",
    "strict_score_caveat",
    "rank_caveat",
    "execution_identity_caveat",
    "reference_plan_sha256",
    "raw_plan_file_sha256",
    "sealed_analysis_manifest_sha256",
    "score_source",
    "score_source_sha256",
    "rank_source",
    "rank_source_sha256",
    "timing_source",
    "timing_source_sha256",
    "covered_by_model_run_row_id",
    "recommended_for_complete_table_run",
    "cache_root_placeholder",
    "reviewer_root_placeholder",
    "resolved_run_root_template",
    "gpu_placeholder",
    "estimate_command",
    "run_command",
    "status_command",
    "compare_command",
)


class ScoreCellError(RuntimeError):
    """A sealed input or deterministic output contract was violated."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_regular(path: Path) -> bytes:
    """Read one non-symlink regular file while checking stable identity."""

    absolute = Path(os.path.abspath(os.fspath(path)))
    before_path = absolute.lstat()
    if not absolute.is_file() or absolute.is_symlink():
        raise ScoreCellError(f"sealed input is not a regular file: {absolute}")
    descriptor = os.open(
        absolute,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before = os.fstat(descriptor)
        identity = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns")
        if any(getattr(before, key) != getattr(before_path, key) for key in identity):
            raise ScoreCellError(f"sealed input changed while opening: {absolute}")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
        if any(getattr(before, key) != getattr(after, key) for key in identity):
            raise ScoreCellError(f"sealed input changed while reading: {absolute}")
        payload = b"".join(chunks)
        if len(payload) != before.st_size:
            raise ScoreCellError(f"sealed input read was incomplete: {absolute}")
        return payload
    except OSError as error:
        raise ScoreCellError(f"could not read sealed input: {absolute}") from error
    finally:
        os.close(descriptor)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ScoreCellError(f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise ScoreCellError(f"non-finite JSON value: {value}")


def _load_json_bytes(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ScoreCellError(f"invalid JSON in {label}") from error
    if not isinstance(value, dict):
        raise ScoreCellError(f"JSON root is not an object in {label}")
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _plan_sha256(plan: Mapping[str, Any]) -> str:
    value = copy.deepcopy(dict(plan))
    value.pop("plan_sha256", None)
    return _sha256(_canonical_bytes(value))


def _load_csv_bytes(
    payload: bytes,
    *,
    label: str,
    expected_fields: tuple[str, ...],
) -> list[dict[str, str]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ScoreCellError(f"invalid UTF-8 in {label}") from error
    with io.StringIO(text, newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != expected_fields:
            raise ScoreCellError(f"unexpected columns in {label}")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ScoreCellError(f"malformed row in {label}")
    return rows


def _checked_score(value: str, *, cell: str, metric: str) -> str:
    try:
        number = Decimal(value)
    except InvalidOperation as error:
        raise ScoreCellError(f"invalid {metric} for {cell}") from error
    if not number.is_finite() or number < 0 or number > 1:
        raise ScoreCellError(f"out-of-range {metric} for {cell}")
    return value


def _percent(value: str, places: int) -> str:
    number = Decimal(value) * Decimal(100)
    return f"{number:.{places}f}%"


def _hours(total_seconds: Decimal) -> str:
    if not total_seconds.is_finite() or total_seconds < 0:
        raise ScoreCellError("historical timing total is invalid")
    with localcontext() as context:
        context.prec = 50
        value = total_seconds / Decimal(3600)
        return format(value.quantize(Decimal("0.000000000001")), "f")


def _safe_identifier(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or SAFE_ID.fullmatch(value) is None:
        raise ScoreCellError(f"unsafe {label}: {value!r}")
    return value


def _commands(*, model: str, dataset: str | None) -> dict[str, str]:
    scope = "model" if dataset is None else "dataset"
    selection = f"--scope {scope} --model {model}"
    if dataset is not None:
        selection += f" --dataset {dataset}"
    prefix = "scripts/reproduce.sh reviewer"
    run_suffix = f"{model}__{dataset if dataset is not None else 'overall'}"
    run_root = f"{PLACEHOLDERS['reviewer_root']}/{run_suffix}"
    return {
        "resolved_run_root_template": run_root,
        "estimate_command": f"{prefix} estimate {selection}",
        "run_command": (
            f"{prefix} run {selection} "
            f"--cache-root \"{PLACEHOLDERS['cache_root']}\" "
            f"--run-root \"{run_root}\" "
            f"--gpu \"{PLACEHOLDERS['gpu']}\""
        ),
        "status_command": (
            f"{prefix} status --run-root \"{run_root}\""
        ),
        "compare_command": (
            f"{prefix} compare --run-root \"{run_root}\" "
            f"--cache-root \"{PLACEHOLDERS['cache_root']}\""
        ),
    }


def _read_and_verify_inputs(project_root: Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    payloads: dict[str, bytes] = {}
    for relative, expected in EXPECTED_RAW_SHA256.items():
        payload = _read_regular(root / relative)
        observed = _sha256(payload)
        if observed != expected:
            raise ScoreCellError(
                f"sealed input checksum mismatch for {relative}: {observed}"
            )
        payloads[relative] = payload

    manifest = _load_json_bytes(
        payloads[MANIFEST_RELATIVE.as_posix()], label=MANIFEST_RELATIVE.as_posix()
    )
    manifest_files = manifest.get("files")
    if not isinstance(manifest_files, Mapping):
        raise ScoreCellError("sealed analysis manifest has no file map")
    for relative in (
        DATASET_TABLE_RELATIVE,
        OVERALL_TABLE_RELATIVE,
        RANKING_TABLE_RELATIVE,
        JOB_METRICS_RELATIVE,
    ):
        name = relative.name
        if manifest_files.get(name) != EXPECTED_RAW_SHA256[relative.as_posix()]:
            raise ScoreCellError(f"manifest checksum differs for {name}")

    plan = _load_json_bytes(
        payloads[PLAN_RELATIVE.as_posix()], label=PLAN_RELATIVE.as_posix()
    )
    sidecar = _read_regular(root / PLAN_SIDECAR_RELATIVE).decode("ascii").strip()
    computed_plan_sha = _plan_sha256(plan)
    if not all(
        value == EXPECTED_PLAN_SHA256
        for value in (
            computed_plan_sha,
            plan.get("plan_sha256"),
            manifest.get("plan_sha256"),
            sidecar,
        )
    ):
        raise ScoreCellError("sealed plan identity is inconsistent")

    audit = _load_json_bytes(
        payloads[AUDIT_RELATIVE.as_posix()], label=AUDIT_RELATIVE.as_posix()
    )
    expected_audit = {
        "plan_sha256": EXPECTED_PLAN_SHA256,
        "exact_cartesian_complete": True,
        "expected": EXPECTED_GRID_JOB_COUNT,
        "complete": EXPECTED_GRID_JOB_COUNT,
        "missing": 0,
        "extra": 0,
        "failed_jobs": 0,
    }
    if any(audit.get(key) != value for key, value in expected_audit.items()):
        raise ScoreCellError("formal grid audit is not complete and exact")

    return {"root": root, "payloads": payloads, "plan": plan}


def build_rows(project_root: Path = PROJECT_ROOT) -> list[dict[str, str]]:
    """Return all 43 x (five datasets + overall) reviewer score cells."""

    inputs = _read_and_verify_inputs(project_root)
    plan = inputs["plan"]
    payloads = inputs["payloads"]
    models_value = plan.get("architectures")
    datasets_value = plan.get("dataset_order")
    seeds_value = plan.get("seeds")
    if not isinstance(models_value, list) or not isinstance(datasets_value, list):
        raise ScoreCellError("plan model/dataset order is malformed")
    if not isinstance(seeds_value, list) or not seeds_value:
        raise ScoreCellError("plan seed roster is malformed")
    models = tuple(_safe_identifier(value, label="model ID") for value in models_value)
    datasets = tuple(
        _safe_identifier(value, label="dataset ID") for value in datasets_value
    )
    if len(models) != EXPECTED_MODEL_COUNT or len(set(models)) != len(models):
        raise ScoreCellError("plan does not contain 43 unique models")
    if len(datasets) != EXPECTED_DATASET_COUNT or len(set(datasets)) != len(datasets):
        raise ScoreCellError("plan does not contain five unique datasets")

    plan_datasets = plan.get("datasets")
    if not isinstance(plan_datasets, Mapping) or set(plan_datasets) != set(datasets):
        raise ScoreCellError("plan dataset definitions do not match dataset order")
    jobs_by_dataset: dict[str, int] = {}
    for dataset in datasets:
        definition = plan_datasets[dataset]
        if not isinstance(definition, Mapping):
            raise ScoreCellError(f"malformed plan dataset definition: {dataset}")
        subjects = definition.get("subjects")
        folds = definition.get("folds")
        if not isinstance(subjects, list) or not isinstance(folds, list):
            raise ScoreCellError(f"malformed plan dimensions: {dataset}")
        if not subjects or not folds or len(subjects) != len(set(subjects)):
            raise ScoreCellError(f"invalid plan dimensions: {dataset}")
        jobs_by_dataset[dataset] = len(subjects) * len(folds) * len(seeds_value)
    model_job_count = sum(jobs_by_dataset.values())
    if model_job_count * len(models) != plan.get("n_jobs"):
        raise ScoreCellError("plan job cardinality is inconsistent")
    if plan.get("n_jobs") != EXPECTED_GRID_JOB_COUNT:
        raise ScoreCellError("plan is not the exact 96,320-job publication")

    dataset_rows = _load_csv_bytes(
        payloads[DATASET_TABLE_RELATIVE.as_posix()],
        label=DATASET_TABLE_RELATIVE.as_posix(),
        expected_fields=DATASET_FIELDS,
    )
    overall_rows = _load_csv_bytes(
        payloads[OVERALL_TABLE_RELATIVE.as_posix()],
        label=OVERALL_TABLE_RELATIVE.as_posix(),
        expected_fields=OVERALL_FIELDS,
    )
    ranking_rows = _load_csv_bytes(
        payloads[RANKING_TABLE_RELATIVE.as_posix()],
        label=RANKING_TABLE_RELATIVE.as_posix(),
        expected_fields=RANKING_FIELDS,
    )
    job_metric_rows = _load_csv_bytes(
        payloads[JOB_METRICS_RELATIVE.as_posix()],
        label=JOB_METRICS_RELATIVE.as_posix(),
        expected_fields=JOB_METRIC_FIELDS,
    )
    dataset_index: dict[tuple[str, str], dict[str, str]] = {}
    for row in dataset_rows:
        key = (row["dataset"], row["model"])
        if key in dataset_index:
            raise ScoreCellError(f"duplicate sealed dataset cell: {key}")
        dataset_index[key] = row
    overall_index: dict[str, dict[str, str]] = {}
    for row in overall_rows:
        model = row["model"]
        if model in overall_index:
            raise ScoreCellError(f"duplicate sealed overall row: {model}")
        overall_index[model] = row
    expected_dataset_keys = {
        (dataset, model) for dataset in datasets for model in models
    }
    if set(dataset_index) != expected_dataset_keys:
        raise ScoreCellError("sealed dataset table is not the exact 43 x 5 grid")
    if set(overall_index) != set(models):
        raise ScoreCellError("sealed overall table is not the exact 43-model roster")

    ranking_index: dict[str, dict[str, str]] = {}
    for row in ranking_rows:
        model = row["model"]
        if model in ranking_index:
            raise ScoreCellError(f"duplicate sealed rank row: {model}")
        ranking_index[model] = row
    if set(ranking_index) != set(models):
        raise ScoreCellError("sealed ranking table is not the exact 43-model roster")
    if {int(row["rank"]) for row in ranking_rows} != set(
        range(1, len(models) + 1)
    ):
        raise ScoreCellError("sealed primary ranks are not exactly 1 through 43")
    for model, row in ranking_index.items():
        if (
            row["metric"] != "equal_dataset_macro_balanced_accuracy"
            or row["tie_break"] != "frozen_architecture_order"
            or Decimal(row["value"])
            != Decimal(overall_index[model]["balanced_accuracy"])
        ):
            raise ScoreCellError(f"sealed rank/overall join differs for {model}")

    architecture_index = {model: index for index, model in enumerate(models)}
    accuracy_order = sorted(
        models,
        key=lambda model: (
            -Decimal(overall_index[model]["accuracy"]),
            architecture_index[model],
        ),
    )
    accuracy_rank = {
        model: index + 1 for index, model in enumerate(accuracy_order)
    }
    ranked_models = tuple(
        sorted(models, key=lambda model: int(ranking_index[model]["rank"]))
    )

    expected_job_keys = {
        (dataset, model, int(subject), int(fold), int(seed))
        for dataset in datasets
        for model in models
        for subject in plan_datasets[dataset]["subjects"]
        for fold in plan_datasets[dataset]["folds"]
        for seed in seeds_value
    }
    observed_job_keys: set[tuple[str, str, int, int, int]] = set()
    observed_job_ids: set[str] = set()
    timing_seconds: dict[tuple[str, str], Decimal] = {
        (dataset, model): Decimal(0)
        for dataset in datasets
        for model in models
    }
    for row in job_metric_rows:
        try:
            key = (
                row["dataset"],
                row["model"],
                int(row["subject"]),
                int(row["fold"]),
                int(row["seed"]),
            )
            seconds = Decimal(row["job_total_seconds"])
        except (InvalidOperation, ValueError) as error:
            raise ScoreCellError("invalid sealed job-metric identity/timing") from error
        if key in observed_job_keys or row["job_id"] in observed_job_ids:
            raise ScoreCellError("duplicate sealed job-metric identity")
        if key not in expected_job_keys:
            raise ScoreCellError(f"unexpected sealed job-metric key: {key}")
        if not seconds.is_finite() or seconds < 0:
            raise ScoreCellError(f"invalid sealed job timing: {row['job_id']}")
        observed_job_keys.add(key)
        observed_job_ids.add(row["job_id"])
        timing_seconds[(row["dataset"], row["model"])] += seconds
    if observed_job_keys != expected_job_keys:
        raise ScoreCellError("sealed job metrics do not cover the exact plan")

    output: list[dict[str, str]] = []
    total_subjects = sum(len(plan_datasets[dataset]["subjects"]) for dataset in datasets)
    total_subject_folds = sum(
        len(plan_datasets[dataset]["subjects"])
        * len(plan_datasets[dataset]["folds"])
        for dataset in datasets
    )
    overall_fold_profile = "|".join(
        str(len(plan_datasets[dataset]["folds"])) for dataset in datasets
    )
    for model in ranked_models:
        primary_rank = ranking_index[model]["rank"]
        descriptive_accuracy_rank = str(accuracy_rank[model])
        for dataset in datasets:
            sealed = dataset_index[(dataset, model)]
            subject_count = len(plan_datasets[dataset]["subjects"])
            folds_per_subject = len(plan_datasets[dataset]["folds"])
            subject_fold_count = subject_count * folds_per_subject
            if sealed["subjects_averaged"] != str(subject_count):
                raise ScoreCellError(f"subject count differs for {model}/{dataset}")
            if any(
                sealed[field] != str(subject_count)
                for field in ("accuracy_defined_count", "balanced_accuracy_defined_count")
            ):
                raise ScoreCellError(f"defined score count differs for {model}/{dataset}")
            cell = f"{model}::{dataset}"
            model_run_row_id = f"model::{model}"
            accuracy = _checked_score(
                sealed["accuracy"], cell=cell, metric="accuracy"
            )
            balanced_accuracy = _checked_score(
                sealed["balanced_accuracy"],
                cell=cell,
                metric="balanced accuracy",
            )
            data_access, general_access = DATA_ACCESS[dataset]
            row = {
                "row_id": f"dataset::{model}::{dataset}",
                "score_cell_id": cell,
                "model": model,
                "primary_ba_rank": primary_rank,
                "primary_ba_rank_metric": ranking_index[model]["metric"],
                "primary_ba_rank_status": ranking_index[model]["status"],
                "descriptive_accuracy_rank": descriptive_accuracy_rank,
                "descriptive_accuracy_rank_derivation": ACCURACY_RANK_DERIVATION,
                "score_level": "dataset",
                "dataset": dataset,
                "replay_scope": "dataset",
                "subject_count": str(subject_count),
                "folds_per_subject": str(folds_per_subject),
                "subject_fold_count": str(subject_fold_count),
                "seed_count": str(len(seeds_value)),
                "job_count": str(jobs_by_dataset[dataset]),
                "accuracy": accuracy,
                "balanced_accuracy": balanced_accuracy,
                "accuracy_percent_2dp": _percent(accuracy, 2),
                "accuracy_percent_3dp": _percent(accuracy, 3),
                "balanced_accuracy_percent_2dp": _percent(
                    balanced_accuracy, 2
                ),
                "balanced_accuracy_percent_3dp": _percent(
                    balanced_accuracy, 3
                ),
                "aggregation": sealed["aggregation"],
                "historical_a5000_summed_hours": _hours(
                    timing_seconds[(dataset, model)]
                ),
                "historical_timing_caveat": TIMING_CAVEAT,
                "data_access": data_access,
                "general_access": general_access,
                "cache_redistributable": "false",
                "evidence_caveat": EVIDENCE_CAVEAT,
                "strict_score_caveat": STRICT_SCORE_CAVEAT,
                "rank_caveat": RANK_CAVEAT,
                "execution_identity_caveat": IDENTITY_CAVEAT,
                "reference_plan_sha256": EXPECTED_PLAN_SHA256,
                "raw_plan_file_sha256": EXPECTED_RAW_SHA256[
                    PLAN_RELATIVE.as_posix()
                ],
                "sealed_analysis_manifest_sha256": EXPECTED_RAW_SHA256[
                    MANIFEST_RELATIVE.as_posix()
                ],
                "score_source": DATASET_TABLE_RELATIVE.as_posix(),
                "score_source_sha256": EXPECTED_RAW_SHA256[
                    DATASET_TABLE_RELATIVE.as_posix()
                ],
                "rank_source": RANKING_TABLE_RELATIVE.as_posix(),
                "rank_source_sha256": EXPECTED_RAW_SHA256[
                    RANKING_TABLE_RELATIVE.as_posix()
                ],
                "timing_source": JOB_METRICS_RELATIVE.as_posix(),
                "timing_source_sha256": EXPECTED_RAW_SHA256[
                    JOB_METRICS_RELATIVE.as_posix()
                ],
                "covered_by_model_run_row_id": model_run_row_id,
                "recommended_for_complete_table_run": "false",
                "cache_root_placeholder": PLACEHOLDERS["cache_root"],
                "reviewer_root_placeholder": PLACEHOLDERS["reviewer_root"],
                "gpu_placeholder": PLACEHOLDERS["gpu"],
                **_commands(model=model, dataset=dataset),
            }
            output.append(row)

        sealed = overall_index[model]
        if sealed["datasets_equal_weighted"] != str(len(datasets)):
            raise ScoreCellError(f"overall dataset count differs for {model}")
        if any(
            sealed[field] != str(len(datasets))
            for field in ("accuracy_defined_count", "balanced_accuracy_defined_count")
        ):
            raise ScoreCellError(f"overall defined score count differs for {model}")
        cell = f"{model}::overall"
        accuracy = _checked_score(
            sealed["accuracy"], cell=cell, metric="accuracy"
        )
        balanced_accuracy = _checked_score(
            sealed["balanced_accuracy"],
            cell=cell,
            metric="balanced accuracy",
        )
        data_access, general_access = DATA_ACCESS["overall"]
        overall_seconds = sum(
            (timing_seconds[(dataset, model)] for dataset in datasets),
            Decimal(0),
        )
        output.append(
            {
                "row_id": f"model::{model}",
                "score_cell_id": cell,
                "model": model,
                "primary_ba_rank": primary_rank,
                "primary_ba_rank_metric": ranking_index[model]["metric"],
                "primary_ba_rank_status": ranking_index[model]["status"],
                "descriptive_accuracy_rank": descriptive_accuracy_rank,
                "descriptive_accuracy_rank_derivation": ACCURACY_RANK_DERIVATION,
                "score_level": "overall",
                "dataset": "overall",
                "replay_scope": "model",
                "subject_count": str(total_subjects),
                "folds_per_subject": f"by_dataset:{overall_fold_profile}",
                "subject_fold_count": str(total_subject_folds),
                "seed_count": str(len(seeds_value)),
                "job_count": str(model_job_count),
                "accuracy": accuracy,
                "balanced_accuracy": balanced_accuracy,
                "accuracy_percent_2dp": _percent(accuracy, 2),
                "accuracy_percent_3dp": _percent(accuracy, 3),
                "balanced_accuracy_percent_2dp": _percent(
                    balanced_accuracy, 2
                ),
                "balanced_accuracy_percent_3dp": _percent(
                    balanced_accuracy, 3
                ),
                "aggregation": sealed["aggregation"],
                "historical_a5000_summed_hours": _hours(overall_seconds),
                "historical_timing_caveat": TIMING_CAVEAT,
                "data_access": data_access,
                "general_access": general_access,
                "cache_redistributable": "false",
                "evidence_caveat": EVIDENCE_CAVEAT,
                "strict_score_caveat": STRICT_SCORE_CAVEAT,
                "rank_caveat": RANK_CAVEAT,
                "execution_identity_caveat": IDENTITY_CAVEAT,
                "reference_plan_sha256": EXPECTED_PLAN_SHA256,
                "raw_plan_file_sha256": EXPECTED_RAW_SHA256[
                    PLAN_RELATIVE.as_posix()
                ],
                "sealed_analysis_manifest_sha256": EXPECTED_RAW_SHA256[
                    MANIFEST_RELATIVE.as_posix()
                ],
                "score_source": OVERALL_TABLE_RELATIVE.as_posix(),
                "score_source_sha256": EXPECTED_RAW_SHA256[
                    OVERALL_TABLE_RELATIVE.as_posix()
                ],
                "rank_source": RANKING_TABLE_RELATIVE.as_posix(),
                "rank_source_sha256": EXPECTED_RAW_SHA256[
                    RANKING_TABLE_RELATIVE.as_posix()
                ],
                "timing_source": JOB_METRICS_RELATIVE.as_posix(),
                "timing_source_sha256": EXPECTED_RAW_SHA256[
                    JOB_METRICS_RELATIVE.as_posix()
                ],
                "covered_by_model_run_row_id": f"model::{model}",
                "recommended_for_complete_table_run": "true",
                "cache_root_placeholder": PLACEHOLDERS["cache_root"],
                "reviewer_root_placeholder": PLACEHOLDERS["reviewer_root"],
                "gpu_placeholder": PLACEHOLDERS["gpu"],
                **_commands(model=model, dataset=None),
            }
        )

    expected_count = len(models) * (len(datasets) + 1)
    if len(output) != expected_count:
        raise ScoreCellError("generated score-cell count is inconsistent")
    if len({row["score_cell_id"] for row in output}) != expected_count:
        raise ScoreCellError("generated score-cell IDs are not unique")
    if len({row["row_id"] for row in output}) != expected_count:
        raise ScoreCellError("generated row IDs are not unique")
    recommended = [
        row for row in output if row["recommended_for_complete_table_run"] == "true"
    ]
    if (
        len(recommended) != len(models)
        or sum(int(row["job_count"]) for row in recommended)
        != EXPECTED_GRID_JOB_COUNT
    ):
        raise ScoreCellError("recommended model-row execution set is not exact")
    return output


def render_csv(rows: Sequence[Mapping[str, str]]) -> bytes:
    """Render rows with a frozen field order and LF-only line endings."""

    with io.StringIO(newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=OUTPUT_FIELDS,
            lineterminator="\n",
            extrasaction="raise",
        )
        writer.writeheader()
        for row in rows:
            if set(row) != set(OUTPUT_FIELDS):
                raise ScoreCellError("generated row does not match output schema")
            writer.writerow(row)
        return handle.getvalue().encode("utf-8")


def _write_atomic(path: Path, payload: bytes, *, project_root: Path) -> None:
    destination = Path(os.path.abspath(os.fspath(path)))
    root = Path(project_root).resolve()
    canonical_destination = destination.parent.resolve() / destination.name
    protected = {
        (Path(project_root).resolve() / relative).resolve()
        for relative in (
            PLAN_RELATIVE,
            PLAN_SIDECAR_RELATIVE,
            AUDIT_RELATIVE,
            MANIFEST_RELATIVE,
            DATASET_TABLE_RELATIVE,
            OVERALL_TABLE_RELATIVE,
            RANKING_TABLE_RELATIVE,
            JOB_METRICS_RELATIVE,
        )
    }
    if canonical_destination in protected:
        raise ScoreCellError("refusing to overwrite an authoritative sealed input")
    try:
        canonical_destination.relative_to(root / "results")
    except ValueError:
        pass
    else:
        raise ScoreCellError("reviewer catalog output must not be written under results/")
    default_destination = root / DEFAULT_OUTPUT_RELATIVE
    if not destination.parent.exists() and destination == default_destination:
        destination.parent.mkdir(mode=0o755)
    if not destination.parent.is_dir():
        raise ScoreCellError(f"output parent does not exist: {destination.parent}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            os.fchmod(handle.fileno(), 0o644)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="release root containing results/common_grid_v6",
    )
    parser.add_argument(
        "--output",
        help=(
            "output CSV path, or '-' for standard output; defaults to "
            "PROJECT_ROOT/reviewer/reviewer_score_cells.csv"
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify that the output already equals the deterministic rendering",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        rows = build_rows(arguments.project_root)
        payload = render_csv(rows)
        output = (
            Path(arguments.project_root).resolve() / DEFAULT_OUTPUT_RELATIVE
            if arguments.output is None
            else arguments.output
        )
        if arguments.check:
            if output == "-":
                raise ScoreCellError("--check requires a file output, not standard output")
            observed = _read_regular(Path(output))
            if observed != payload:
                raise ScoreCellError("reviewer score-cell CSV is stale or noncanonical")
            print(f"PASS: {len(rows)} deterministic reviewer score cells")
        elif output == "-":
            sys.stdout.buffer.write(payload)
        else:
            _write_atomic(
                Path(output), payload, project_root=arguments.project_root
            )
        return 0
    except (ScoreCellError, OSError, KeyError, TypeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
