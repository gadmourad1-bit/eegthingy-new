"""Read-only analysis of the fixed opened-development validation screen.

This module reads only ``plan.json`` and the plan's atomic validation-record
JSON files.  It never opens an EEG cache, imports a cache loader, or indexes
trial rows.  Every expected record is checked with the screen runner's
validator and is additionally bound back to the immutable cache, source,
split, and training identities stored in the plan.

The resulting bundle is deliberately descriptive.  Its freeze/kill signals
are conservative model-development gates for the next prespecified stage;
they are not claims about novelty, untouched-cohort generalization, or global
standing.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .development_screen import (
    CANDIDATE_MODELS,
    CURATED_SPLITS,
    EVALUATION_SCOPE,
    METRIC_NAMES,
    WORKER_COUNT,
    ScreenJob,
    _record_path,
    _strict_json_load,
    curated_jobs,
    load_plan,
    validate_record_payload,
)


ANALYSIS_SCHEMA = "eeg-mi-opened-development-screen-analysis-v1"
NEW_CANDIDATES: tuple[str, ...] = tuple(
    name for name in CANDIDATE_MODELS if name.startswith("chsdnet")
)
REFERENCE_MODELS: tuple[str, ...] = (
    "cardinal_fbc_micro_extended",
    "tcformer",
)
DATASETS: tuple[str, ...] = tuple(dataset for dataset, _ in CURATED_SPLITS)
SUBJECTS_BY_DATASET: Mapping[str, tuple[int, ...]] = dict(CURATED_SPLITS)

FREEZE_MIN_NONNEGATIVE_DATASETS = 4
FREEZE_MAX_DATASET_DEFICIT = 0.02
FREEZE_MIN_PAIRED_WIN_RATE = 0.50
KILL_MIN_OVERALL_DEFICIT = 0.01
KILL_MAX_POSITIVE_DATASETS = 1
PAIR_TIE_TOLERANCE = 1e-12

_METRICS = tuple(sorted(METRIC_NAMES))
_MEAN_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "cohen_kappa",
    "negative_log_likelihood",
    "roc_auc",
)


class DevelopmentAnalysisError(RuntimeError):
    """Raised when an immutable plan cannot support a valid analysis."""


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _sample_sd(values: Sequence[float]) -> float | None:
    return statistics.stdev(values) if len(values) >= 2 else None


def _subject_key(job: ScreenJob) -> str:
    return f"{job.dataset}:s{job.subject:03d}"


def _split_key(job: ScreenJob) -> str:
    return f"{job.dataset}:s{job.subject:03d}:f{job.fold:02d}"


def _expected_jobs_from_plan(plan: Mapping[str, Any]) -> tuple[ScreenJob, ...]:
    """Require the plan to retain the exact fixed cartesian job contract."""

    items = plan.get("jobs")
    if not isinstance(items, list):
        raise DevelopmentAnalysisError("immutable plan jobs are not a list")
    expected = curated_jobs()
    if len(items) != len(expected):
        raise DevelopmentAnalysisError("immutable plan job count changed")

    observed: list[ScreenJob] = []
    seen: set[str] = set()
    required_keys = {
        "dataset",
        "model",
        "subject",
        "fold",
        "seed",
        "job_id",
        "worker_index",
    }
    for index, (item, expected_job) in enumerate(zip(items, expected, strict=True)):
        if not isinstance(item, Mapping) or set(item) != required_keys:
            raise DevelopmentAnalysisError(
                f"plan job {index} differs from the fixed job schema"
            )
        job = ScreenJob.from_mapping(item)
        if (
            job != expected_job
            or item.get("job_id") != expected_job.job_id
            or int(item.get("worker_index", -1)) != index % WORKER_COUNT
        ):
            raise DevelopmentAnalysisError(
                f"plan job {index} differs from the fixed manifest order"
            )
        if job.job_id in seen:
            raise DevelopmentAnalysisError(f"duplicate plan job {job.job_id}")
        seen.add(job.job_id)
        observed.append(job)
    return tuple(observed)


def _validate_record_bindings(
    payload: Mapping[str, Any],
    *,
    job: ScreenJob,
    plan: Mapping[str, Any],
) -> None:
    """Validate identities that the base record schema intentionally leaves open."""

    expected_cache = {
        name: plan["cache_identity"][_subject_key(job)][name]
        for name in ("array_sha256", "file_sha256")
    }
    if payload.get("cache_identity") != expected_cache:
        raise ValueError("record cache identity differs from immutable plan")
    if payload.get("source_identity") != plan["source_identity"]:
        raise ValueError("record source identity differs from immutable plan")
    if payload.get("split") != plan["split_identity"][_split_key(job)]:
        raise ValueError("record split identity differs from immutable plan")

    fit = payload.get("fit")
    if not isinstance(fit, Mapping) or fit.get("config") != plan["train_config"]:
        raise ValueError("record training configuration differs from immutable plan")
    model = payload.get("model")
    identity = model.get("identity") if isinstance(model, Mapping) else None
    if not isinstance(identity, Mapping) or identity.get("requested_name") != job.model:
        raise ValueError("record requested model differs from job identity")

    metrics = payload["validation_metrics"]
    bounded = ("accuracy", "balanced_accuracy", "roc_auc")
    if any(not 0.0 <= float(metrics[name]) <= 1.0 for name in bounded):
        raise ValueError("record contains an out-of-range bounded metric")
    if not -1.0 <= float(metrics["cohen_kappa"]) <= 1.0:
        raise ValueError("record contains an out-of-range Cohen kappa")
    if float(metrics["negative_log_likelihood"]) < 0.0:
        raise ValueError("record contains a negative log loss")

    timing = payload.get("timing")
    if not isinstance(timing, Mapping):
        raise ValueError("record timing identity is missing")
    for name in (
        "construction_seconds",
        "fit_seconds",
        "evaluation_seconds",
        "total_seconds",
    ):
        value = timing.get(name)
        if (
            not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"record timing {name} is invalid")


def _empty_job_row(job: ScreenJob, status: str, error: str | None) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        **job.identity(),
        "record_status": status,
        "record_error": error,
        "validation_count": None,
        **{name: None for name in _METRICS},
        "parameter_count": None,
        "best_epoch": None,
        "epochs_run": None,
        "construction_seconds": None,
        "fit_seconds": None,
        "evaluation_seconds": None,
        "total_seconds": None,
    }


def _job_row(job: ScreenJob, payload: Mapping[str, Any]) -> dict[str, Any]:
    metrics = payload["validation_metrics"]
    timing = payload["timing"]
    return {
        "job_id": job.job_id,
        **job.identity(),
        "record_status": "valid",
        "record_error": None,
        "validation_count": int(payload["split"]["validation"]["count"]),
        **{name: float(metrics[name]) for name in _METRICS},
        "parameter_count": int(payload["model"]["parameter_count"]),
        "best_epoch": int(payload["fit"]["best_epoch"]),
        "epochs_run": int(payload["fit"]["epochs_run"]),
        "construction_seconds": float(timing["construction_seconds"]),
        "fit_seconds": float(timing["fit_seconds"]),
        "evaluation_seconds": float(timing["evaluation_seconds"]),
        "total_seconds": float(timing["total_seconds"]),
    }


def _average_ranks(
    rows: Sequence[dict[str, Any]],
    *,
    score_key: str,
) -> dict[str, float]:
    """Return descending average ranks, assigning equal scores equal ranks."""

    eligible = [
        row
        for row in rows
        if row.get(score_key) is not None and bool(row.get("complete_coverage"))
    ]
    eligible.sort(key=lambda row: (-float(row[score_key]), str(row["model"])))
    ranks: dict[str, float] = {}
    index = 0
    while index < len(eligible):
        stop = index + 1
        score = float(eligible[index][score_key])
        while stop < len(eligible) and math.isclose(
            float(eligible[stop][score_key]),
            score,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            stop += 1
        average_rank = ((index + 1) + stop) / 2.0
        for row in eligible[index:stop]:
            ranks[str(row["model"])] = average_rank
        index = stop
    return ranks


def _aggregate_subject_models(
    jobs: Sequence[ScreenJob],
    valid_rows: Mapping[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    expected: dict[tuple[str, int, str], list[ScreenJob]] = defaultdict(list)
    for job in jobs:
        expected[(job.dataset, job.subject, job.model)].append(job)

    result: list[dict[str, Any]] = []
    for (dataset, subject, model), group_jobs in expected.items():
        rows = [
            valid_rows[job.job_id]
            for job in group_jobs
            if job.job_id in valid_rows
        ]
        row: dict[str, Any] = {
            "dataset": dataset,
            "subject": subject,
            "model": model,
            "expected_jobs": len(group_jobs),
            "valid_jobs": len(rows),
            "complete_coverage": len(rows) == len(group_jobs),
            "n_folds": len({item["fold"] for item in rows}),
            "n_seeds": len({item["seed"] for item in rows}),
        }
        for metric in _MEAN_METRICS:
            values = [float(item[metric]) for item in rows]
            row[f"{metric}_mean"] = _mean(values)
            row[f"{metric}_sd"] = _sample_sd(values)
        row["total_seconds_mean"] = _mean(
            [float(item["total_seconds"]) for item in rows]
        )
        result.append(row)
    return result


def _aggregate_dataset_models(
    subject_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    lookup = {
        (str(row["dataset"]), int(row["subject"]), str(row["model"])): row
        for row in subject_rows
    }
    result: list[dict[str, Any]] = []
    for dataset, subjects in CURATED_SPLITS:
        for model in CANDIDATE_MODELS:
            rows = [
                lookup[(dataset, subject, model)]
                for subject in subjects
                if lookup[(dataset, subject, model)]["complete_coverage"]
            ]
            complete = len(rows) == len(subjects)
            row: dict[str, Any] = {
                "dataset": dataset,
                "model": model,
                "expected_subjects": len(subjects),
                "valid_subjects": len(rows),
                "complete_coverage": complete,
            }
            for metric in _MEAN_METRICS:
                values = [
                    float(item[f"{metric}_mean"])
                    for item in rows
                    if item[f"{metric}_mean"] is not None
                ]
                row[f"{metric}_macro_mean"] = _mean(values)
                row[f"{metric}_subject_sd"] = _sample_sd(values)
            result.append(row)

    for dataset in DATASETS:
        dataset_rows = [row for row in result if row["dataset"] == dataset]
        ranks = _average_ranks(
            dataset_rows, score_key="balanced_accuracy_macro_mean"
        )
        for row in dataset_rows:
            row["balanced_accuracy_rank"] = ranks.get(str(row["model"]))
    return result


def _aggregate_overall(
    dataset_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    lookup = {
        (str(row["dataset"]), str(row["model"])): row for row in dataset_rows
    }
    result: list[dict[str, Any]] = []
    for model in CANDIDATE_MODELS:
        rows = [
            lookup[(dataset, model)]
            for dataset in DATASETS
            if lookup[(dataset, model)]["complete_coverage"]
        ]
        complete = len(rows) == len(DATASETS)
        row: dict[str, Any] = {
            "model": model,
            "expected_datasets": len(DATASETS),
            "valid_datasets": len(rows),
            "valid_subjects": sum(int(item["valid_subjects"]) for item in rows),
            "complete_coverage": complete,
        }
        for metric in _MEAN_METRICS:
            values = [
                float(item[f"{metric}_macro_mean"])
                for item in rows
                if item[f"{metric}_macro_mean"] is not None
            ]
            # An equal-dataset macro is emitted only for complete coverage.
            row[f"equal_dataset_macro_{metric}"] = (
                _mean(values) if complete else None
            )
            row[f"dataset_{metric}_sd"] = (
                _sample_sd(values) if complete else None
            )
        ranks = [
            float(item["balanced_accuracy_rank"])
            for item in rows
            if item["balanced_accuracy_rank"] is not None
        ]
        row["mean_dataset_rank"] = _mean(ranks) if complete else None
        result.append(row)

    ranks = _average_ranks(
        result, score_key="equal_dataset_macro_balanced_accuracy"
    )
    for row in result:
        row["balanced_accuracy_rank"] = ranks.get(str(row["model"]))
    return result


def _rank_rows(
    dataset_rows: Sequence[dict[str, Any]],
    overall_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in dataset_rows:
        result.append(
            {
                "scope": "dataset",
                "dataset": row["dataset"],
                "model": row["model"],
                "metric": "macro_balanced_accuracy",
                "score": row["balanced_accuracy_macro_mean"],
                "rank": row["balanced_accuracy_rank"],
                "complete_coverage": row["complete_coverage"],
            }
        )
    for row in overall_rows:
        result.append(
            {
                "scope": "equal_dataset",
                "dataset": None,
                "model": row["model"],
                "metric": "equal_dataset_macro_balanced_accuracy",
                "score": row["equal_dataset_macro_balanced_accuracy"],
                "rank": row["balanced_accuracy_rank"],
                "complete_coverage": row["complete_coverage"],
            }
        )
    return result


def _delta_summary(
    *,
    candidate: str,
    reference: str,
    dataset: str,
    expected_pairs: int,
    pairs: Sequence[tuple[float, float]],
) -> dict[str, Any]:
    ba_deltas = [pair[0] for pair in pairs]
    accuracy_deltas = [pair[1] for pair in pairs]
    wins = sum(delta > PAIR_TIE_TOLERANCE for delta in ba_deltas)
    losses = sum(delta < -PAIR_TIE_TOLERANCE for delta in ba_deltas)
    ties = len(ba_deltas) - wins - losses
    return {
        "scope": "dataset",
        "dataset": dataset,
        "candidate": candidate,
        "reference": reference,
        "aggregation": "paired_subject_fold_seed_mean",
        "expected_pairs": expected_pairs,
        "valid_pairs": len(pairs),
        "complete_pairing": len(pairs) == expected_pairs,
        "balanced_accuracy_delta": _mean(ba_deltas),
        "balanced_accuracy_delta_median": (
            statistics.median(ba_deltas) if ba_deltas else None
        ),
        "balanced_accuracy_delta_sd": _sample_sd(ba_deltas),
        "accuracy_delta": _mean(accuracy_deltas),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "win_rate": wins / len(pairs) if pairs else None,
    }


def _paired_deltas(
    jobs: Sequence[ScreenJob],
    valid_rows: Mapping[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    by_identity = {
        (
            str(row["dataset"]),
            int(row["subject"]),
            int(row["fold"]),
            int(row["seed"]),
            str(row["model"]),
        ): row
        for row in valid_rows.values()
    }
    expected_by_dataset: dict[str, list[ScreenJob]] = defaultdict(list)
    for job in jobs:
        if job.model == CANDIDATE_MODELS[0]:
            expected_by_dataset[job.dataset].append(job)

    result: list[dict[str, Any]] = []
    for candidate in NEW_CANDIDATES:
        for reference in REFERENCE_MODELS:
            dataset_summaries: list[dict[str, Any]] = []
            for dataset in DATASETS:
                pairs: list[tuple[float, float]] = []
                for identity_job in expected_by_dataset[dataset]:
                    key_prefix = (
                        dataset,
                        identity_job.subject,
                        identity_job.fold,
                        identity_job.seed,
                    )
                    candidate_row = by_identity.get((*key_prefix, candidate))
                    reference_row = by_identity.get((*key_prefix, reference))
                    if candidate_row is None or reference_row is None:
                        continue
                    pairs.append(
                        (
                            float(candidate_row["balanced_accuracy"])
                            - float(reference_row["balanced_accuracy"]),
                            float(candidate_row["accuracy"])
                            - float(reference_row["accuracy"]),
                        )
                    )
                summary = _delta_summary(
                    candidate=candidate,
                    reference=reference,
                    dataset=dataset,
                    expected_pairs=len(expected_by_dataset[dataset]),
                    pairs=pairs,
                )
                dataset_summaries.append(summary)
                result.append(summary)

            complete = all(item["complete_pairing"] for item in dataset_summaries)
            valid_ba = [
                float(item["balanced_accuracy_delta"])
                for item in dataset_summaries
                if item["balanced_accuracy_delta"] is not None
            ]
            valid_accuracy = [
                float(item["accuracy_delta"])
                for item in dataset_summaries
                if item["accuracy_delta"] is not None
            ]
            result.append(
                {
                    "scope": "equal_dataset",
                    "dataset": None,
                    "candidate": candidate,
                    "reference": reference,
                    "aggregation": "equal_dataset_mean_of_paired_means",
                    "expected_pairs": sum(
                        int(item["expected_pairs"]) for item in dataset_summaries
                    ),
                    "valid_pairs": sum(
                        int(item["valid_pairs"]) for item in dataset_summaries
                    ),
                    "complete_pairing": complete,
                    "balanced_accuracy_delta": (
                        _mean(valid_ba) if complete else None
                    ),
                    "balanced_accuracy_delta_median": None,
                    "balanced_accuracy_delta_sd": (
                        _sample_sd(valid_ba) if complete else None
                    ),
                    "accuracy_delta": (
                        _mean(valid_accuracy) if complete else None
                    ),
                    "wins": sum(int(item["wins"]) for item in dataset_summaries),
                    "ties": sum(int(item["ties"]) for item in dataset_summaries),
                    "losses": sum(int(item["losses"]) for item in dataset_summaries),
                    "win_rate": (
                        sum(int(item["wins"]) for item in dataset_summaries)
                        / sum(
                            int(item["valid_pairs"]) for item in dataset_summaries
                        )
                        if sum(
                            int(item["valid_pairs"]) for item in dataset_summaries
                        )
                        else None
                    ),
                }
            )
    return result


def _diagnostics(
    audit: Mapping[str, Any],
    overall_rows: Sequence[dict[str, Any]],
    pair_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    overall = {str(row["model"]): row for row in overall_rows}
    pair_lookup = {
        (str(row["candidate"]), str(row["reference"]), row["dataset"]): row
        for row in pair_rows
    }
    complete_audit = bool(audit["analysis_complete"])
    fully_covered = [
        candidate
        for candidate in NEW_CANDIDATES
        if overall[candidate]["equal_dataset_macro_balanced_accuracy"] is not None
    ]
    top_candidate = (
        max(
            fully_covered,
            key=lambda name: float(
                overall[name]["equal_dataset_macro_balanced_accuracy"]
            ),
        )
        if fully_covered
        else None
    )

    result: list[dict[str, Any]] = []
    for candidate in NEW_CANDIDATES:
        per_reference: dict[str, dict[str, Any]] = {}
        comparisons_complete = complete_audit
        for reference in REFERENCE_MODELS:
            overall_pair = pair_lookup[(candidate, reference, None)]
            dataset_pairs = [
                pair_lookup[(candidate, reference, dataset)]
                for dataset in DATASETS
            ]
            comparisons_complete = comparisons_complete and bool(
                overall_pair["complete_pairing"]
            )
            deltas = [
                float(item["balanced_accuracy_delta"])
                for item in dataset_pairs
                if item["balanced_accuracy_delta"] is not None
            ]
            per_reference[reference] = {
                "equal_dataset_ba_delta": overall_pair[
                    "balanced_accuracy_delta"
                ],
                "paired_win_rate": overall_pair["win_rate"],
                "nonnegative_datasets": sum(delta >= 0.0 for delta in deltas),
                "positive_datasets": sum(delta > 0.0 for delta in deltas),
                "minimum_dataset_delta": min(deltas) if deltas else None,
            }

        freeze_gate = comparisons_complete and all(
            values["equal_dataset_ba_delta"] is not None
            and float(values["equal_dataset_ba_delta"]) > 0.0
            and int(values["nonnegative_datasets"])
            >= FREEZE_MIN_NONNEGATIVE_DATASETS
            and values["minimum_dataset_delta"] is not None
            and float(values["minimum_dataset_delta"])
            >= -FREEZE_MAX_DATASET_DEFICIT
            and values["paired_win_rate"] is not None
            and float(values["paired_win_rate"]) >= FREEZE_MIN_PAIRED_WIN_RATE
            for values in per_reference.values()
        )
        kill_gate = comparisons_complete and all(
            values["equal_dataset_ba_delta"] is not None
            and float(values["equal_dataset_ba_delta"])
            <= -KILL_MIN_OVERALL_DEFICIT
            and int(values["positive_datasets"]) <= KILL_MAX_POSITIVE_DATASETS
            and values["paired_win_rate"] is not None
            and float(values["paired_win_rate"]) < 0.50
            for values in per_reference.values()
        )

        if not comparisons_complete:
            recommendation = "insufficient_valid_records"
        elif freeze_gate and candidate == top_candidate:
            recommendation = "freeze_for_next_prespecified_stage"
        elif kill_gate:
            recommendation = "kill_or_redesign_current_variant"
        elif freeze_gate:
            recommendation = "retain_as_ablation_not_frozen"
        else:
            recommendation = "continue_development_not_frozen"
        result.append(
            {
                "candidate": candidate,
                "development_rank_among_new_candidates": (
                    1
                    + sum(
                        float(overall[other]["equal_dataset_macro_balanced_accuracy"])
                        > float(
                            overall[candidate][
                                "equal_dataset_macro_balanced_accuracy"
                            ]
                        )
                        for other in fully_covered
                    )
                    if candidate in fully_covered
                    else None
                ),
                "is_top_new_candidate": candidate == top_candidate,
                "comparisons_complete": comparisons_complete,
                "freeze_gate": freeze_gate,
                "kill_gate": kill_gate,
                "recommendation": recommendation,
                "reference_diagnostics": per_reference,
            }
        )
    return result


def analyze_run(run_root: Path) -> dict[str, Any]:
    """Analyze a screen without modifying its plan, records, or cache tree."""

    root = run_root.resolve()
    plan = load_plan(root)
    jobs = _expected_jobs_from_plan(plan)
    expected_paths = {_record_path(root, job).resolve() for job in jobs}
    records_root = root / "records"
    observed_paths = (
        {path.resolve() for path in records_root.rglob("*.json")}
        if records_root.exists()
        else set()
    )
    unexpected = sorted(
        str(path.relative_to(root)) for path in observed_paths - expected_paths
    )

    valid_rows: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    corrupt: dict[str, str] = {}
    job_rows: list[dict[str, Any]] = []
    for job in jobs:
        path = _record_path(root, job)
        if not path.is_file():
            missing.append(job.job_id)
            job_rows.append(_empty_job_row(job, "missing", None))
            continue
        try:
            payload = _strict_json_load(path)
            validate_record_payload(
                payload,
                job=job,
                plan_sha256=str(plan["plan_sha256"]),
            )
            _validate_record_bindings(payload, job=job, plan=plan)
            row = _job_row(job, payload)
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            corrupt[job.job_id] = message
            job_rows.append(_empty_job_row(job, "corrupt", message))
        else:
            valid_rows[job.job_id] = row
            job_rows.append(row)

    audit = {
        "expected_records": len(jobs),
        "valid_records": len(valid_rows),
        "missing_records": len(missing),
        "corrupt_records": len(corrupt),
        "unexpected_json_records": len(unexpected),
        "analysis_complete": (
            len(valid_rows) == len(jobs)
            and not missing
            and not corrupt
            and not unexpected
        ),
        "missing_job_ids": missing,
        "corrupt_job_errors": corrupt,
        "unexpected_json_paths": unexpected,
    }
    subject_rows = _aggregate_subject_models(jobs, valid_rows)
    dataset_rows = _aggregate_dataset_models(subject_rows)
    overall_rows = _aggregate_overall(dataset_rows)
    pair_rows = _paired_deltas(jobs, valid_rows)
    rank_rows = _rank_rows(dataset_rows, overall_rows)
    diagnostics = _diagnostics(audit, overall_rows, pair_rows)
    return {
        "schema": ANALYSIS_SCHEMA,
        "evaluation_scope": EVALUATION_SCOPE,
        "plan_sha256": plan["plan_sha256"],
        "analysis_policy": {
            "inputs": ["plan.json", "expected atomic validation JSON records"],
            "cache_files_opened": False,
            "trial_arrays_opened": False,
            "raw_outcomes_consumed": False,
            "primary_metric": "equal_dataset_macro_balanced_accuracy",
            "screen_seed_count": 1,
        },
        "decision_thresholds": {
            "freeze_min_nonnegative_datasets_per_reference": (
                FREEZE_MIN_NONNEGATIVE_DATASETS
            ),
            "freeze_max_dataset_deficit": FREEZE_MAX_DATASET_DEFICIT,
            "freeze_min_paired_win_rate": FREEZE_MIN_PAIRED_WIN_RATE,
            "kill_min_overall_deficit_per_reference": KILL_MIN_OVERALL_DEFICIT,
            "kill_max_positive_datasets_per_reference": (
                KILL_MAX_POSITIVE_DATASETS
            ),
        },
        "audit": audit,
        "jobs": job_rows,
        "subject_model_means": subject_rows,
        "dataset_model_means": dataset_rows,
        "overall_model_means": overall_rows,
        "paired_deltas": pair_rows,
        "ranks": rank_rows,
        "freeze_kill_diagnostics": diagnostics,
        "interpretation_limits": [
            "Opened-development validation evidence is used for model selection only.",
            "One seed and a curated subject subset do not establish robustness.",
            "No architecture or scientific claim is established by this screen alone.",
        ],
    }


def _csv_text(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return ""
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        normalized: dict[str, Any] = {}
        for key in fieldnames:
            value = row.get(key)
            if isinstance(value, (dict, list, tuple)):
                normalized[key] = json.dumps(
                    value, sort_keys=True, separators=(",", ":"), allow_nan=False
                )
            elif value is None:
                normalized[key] = ""
            elif isinstance(value, bool):
                normalized[key] = str(value).lower()
            else:
                normalized[key] = value
        writer.writerow(normalized)
    return output.getvalue()


def _percent(value: Any) -> str:
    return "—" if value is None else f"{100.0 * float(value):.2f}%"


def _number(value: Any, digits: int = 3) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def render_markdown(analysis: Mapping[str, Any]) -> str:
    """Render a compact, cautious human-readable view of an analysis."""

    audit = analysis["audit"]
    lines = [
        "# Opened-development validation screen",
        "",
        "This report summarizes the immutable one-seed development screen. "
        "It is model-selection evidence only and does not establish novelty, "
        "untouched-cohort generalization, or global standing.",
        "",
        "## Integrity audit",
        "",
        "| Expected | Valid | Missing | Corrupt | Unexpected JSON | Complete |",
        "|---:|---:|---:|---:|---:|:---:|",
        (
            f"| {audit['expected_records']} | {audit['valid_records']} | "
            f"{audit['missing_records']} | {audit['corrupt_records']} | "
            f"{audit['unexpected_json_records']} | "
            f"{'yes' if audit['analysis_complete'] else 'no'} |"
        ),
        "",
        "## Equal-dataset macro results",
        "",
        "| Rank | Model | Balanced accuracy | Accuracy | Coverage |",
        "|---:|---|---:|---:|:---:|",
    ]
    overall = sorted(
        analysis["overall_model_means"],
        key=lambda row: (
            row["balanced_accuracy_rank"] is None,
            row["balanced_accuracy_rank"]
            if row["balanced_accuracy_rank"] is not None
            else float("inf"),
            row["model"],
        ),
    )
    for row in overall:
        lines.append(
            f"| {_number(row['balanced_accuracy_rank'], 1)} | "
            f"`{row['model']}` | "
            f"{_percent(row['equal_dataset_macro_balanced_accuracy'])} | "
            f"{_percent(row['equal_dataset_macro_accuracy'])} | "
            f"{'complete' if row['complete_coverage'] else 'incomplete'} |"
        )

    lines.extend(
        [
            "",
            "## Per-dataset macro results",
            "",
            "| Dataset | Rank | Model | Balanced accuracy | Accuracy | Coverage |",
            "|---|---:|---|---:|---:|:---:|",
        ]
    )
    for row in analysis["dataset_model_means"]:
        lines.append(
            f"| `{row['dataset']}` | "
            f"{_number(row['balanced_accuracy_rank'], 1)} | "
            f"`{row['model']}` | "
            f"{_percent(row['balanced_accuracy_macro_mean'])} | "
            f"{_percent(row['accuracy_macro_mean'])} | "
            f"{'complete' if row['complete_coverage'] else 'incomplete'} |"
        )

    lines.extend(
        [
            "",
            "## Paired equal-dataset deltas",
            "",
            "| Candidate | Reference | BA delta | Accuracy delta | Win rate | Complete |",
            "|---|---|---:|---:|---:|:---:|",
        ]
    )
    for row in analysis["paired_deltas"]:
        if row["scope"] != "equal_dataset":
            continue
        lines.append(
            f"| `{row['candidate']}` | `{row['reference']}` | "
            f"{_percent(row['balanced_accuracy_delta'])} | "
            f"{_percent(row['accuracy_delta'])} | "
            f"{_percent(row['win_rate'])} | "
            f"{'yes' if row['complete_pairing'] else 'no'} |"
        )

    lines.extend(
        [
            "",
            "## Conservative development gates",
            "",
            "| Candidate | New-model rank | Freeze gate | Kill signal | Recommendation |",
            "|---|---:|:---:|:---:|---|",
        ]
    )
    for row in analysis["freeze_kill_diagnostics"]:
        lines.append(
            f"| `{row['candidate']}` | "
            f"{_number(row['development_rank_among_new_candidates'], 0)} | "
            f"{'yes' if row['freeze_gate'] else 'no'} | "
            f"{'yes' if row['kill_gate'] else 'no'} | "
            f"`{row['recommendation']}` |"
        )
    lines.extend(
        [
            "",
            "The full JSON and CSV files retain job-level metrics, coverage "
            "flags, pairing counts, integrity errors, ranks, and thresholds.",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_analysis_bundle(
    analysis: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, str]:
    """Atomically write JSON, tabular CSV files, and the Markdown report."""

    output = output_dir.resolve()
    tables: Mapping[str, Sequence[Mapping[str, Any]]] = {
        "jobs.csv": analysis["jobs"],
        "subject_model_means.csv": analysis["subject_model_means"],
        "dataset_model_means.csv": analysis["dataset_model_means"],
        "overall_model_means.csv": analysis["overall_model_means"],
        "paired_deltas.csv": analysis["paired_deltas"],
        "ranks.csv": analysis["ranks"],
        "freeze_kill_diagnostics.csv": analysis["freeze_kill_diagnostics"],
        "audit.csv": [analysis["audit"]],
    }
    paths: dict[str, str] = {}
    json_path = output / "analysis.json"
    _atomic_write_text(
        json_path,
        json.dumps(analysis, sort_keys=True, indent=2, allow_nan=False) + "\n",
    )
    paths["analysis_json"] = str(json_path)
    for filename, rows in tables.items():
        path = output / filename
        _atomic_write_text(path, _csv_text(rows))
        paths[filename.removesuffix(".csv").replace("-", "_")] = str(path)
    markdown_path = output / "summary.md"
    _atomic_write_text(markdown_path, render_markdown(analysis))
    paths["summary_markdown"] = str(markdown_path)
    return paths


def analyze_and_write(
    run_root: Path,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    records_root = (run_root.resolve() / "records").resolve()
    resolved_output = output_dir.resolve()
    if resolved_output == records_root or records_root in resolved_output.parents:
        raise DevelopmentAnalysisError(
            "analysis output must not be placed inside the atomic records tree"
        )
    analysis = analyze_run(run_root)
    return analysis, write_analysis_bundle(analysis, resolved_output)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    analysis, paths = analyze_and_write(args.run_root, args.output_dir)
    print(
        json.dumps(
            {
                "analysis_complete": analysis["audit"]["analysis_complete"],
                "expected_records": analysis["audit"]["expected_records"],
                "valid_records": analysis["audit"]["valid_records"],
                "outputs": paths,
            },
            sort_keys=True,
        )
    )
    return 0 if analysis["audit"]["analysis_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ANALYSIS_SCHEMA",
    "NEW_CANDIDATES",
    "REFERENCE_MODELS",
    "analyze_and_write",
    "analyze_run",
    "render_markdown",
    "write_analysis_bundle",
]
