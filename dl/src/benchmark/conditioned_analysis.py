"""Read-only paired analysis for the prespecified conditioned v3 screen.

The analyzer first requires all 51 conditioned records to validate against
their immutable v3 plan.  It then validates and joins exactly 51 records for
CardinalFBC, TCFormer, and FBCNet from the plan-bound reference run.  EEG cache
files are never opened.  The output is a conservative development decision:
choose exactly one conditioning strength for the next prespecified stage, or
kill the conditioned family.
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

from .conditioned_screen import (
    CONDITIONED_MODELS,
    EXPECTED_JOB_COUNT,
    REFERENCE_MODELS,
    ScreenJob,
    _record_path,
    _strict_json_load,
    _subject_key,
    _split_key,
    _validate_record_bindings,
    conditioned_jobs,
    load_plan,
    validate_record_payload,
)
from .development_analysis import (
    _expected_jobs_from_plan as validate_reference_plan_jobs,
    _validate_record_bindings as validate_reference_record_bindings,
)
from .development_screen import (
    METRIC_NAMES,
    _record_path as reference_record_path,
    load_plan as load_reference_plan,
    validate_record_payload as validate_reference_record_payload,
)


ANALYSIS_SCHEMA = "eeg-mi-chsd-conditioned-screen-analysis-v3"
DATASETS: tuple[str, ...] = (
    "local_exp4",
    "bnci2014_001",
    "bnci2014_004",
    "cho2017",
    "physionet_mi",
)
EXPECTED_CONDITIONED_RECORDS = 51
EXPECTED_REFERENCE_RECORDS = 51
PAIR_TIE_TOLERANCE = 1e-12

# A candidate is chosen only if it is the unique top conditioned variant and
# clears every reference on the equal-dataset mean, is nonnegative on at least
# four datasets, never trails by more than two points on a dataset, and wins at
# least half of the 17 paired subject comparisons.
MIN_NONNEGATIVE_DATASETS_PER_REFERENCE = 4
MAX_DATASET_DEFICIT = 0.02
MIN_PAIRED_WIN_RATE = 0.50


class ConditionedAnalysisError(RuntimeError):
    """Raised when complete, bound paired evidence is unavailable."""


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ConditionedAnalysisError("cannot average an empty score sequence")
    return statistics.fmean(values)


def _sample_sd(values: Sequence[float]) -> float | None:
    return statistics.stdev(values) if len(values) >= 2 else None


def _reference_jobs() -> tuple[ScreenJob, ...]:
    return tuple(
        ScreenJob(dataset=dataset, model=model, subject=subject)
        for model in REFERENCE_MODELS
        for dataset, subjects in (
            ("local_exp4", (1, 4, 7)),
            ("bnci2014_001", (1, 5, 9)),
            ("bnci2014_004", (1, 5, 9)),
            ("cho2017", (1, 18, 35, 52)),
            ("physionet_mi", (1, 18, 36, 54)),
        )
        for subject in subjects
    )


def _validate_reference_binding(
    *,
    conditioned_plan: Mapping[str, Any],
    reference_root: Path,
    reference_plan: Mapping[str, Any],
) -> None:
    binding = conditioned_plan["reference_binding"]
    if str(reference_root.resolve()) != str(Path(binding["run_root"]).resolve()):
        raise ConditionedAnalysisError(
            "supplied reference root differs from the immutable v3 binding"
        )
    if (
        reference_plan["schema"] != binding["plan_schema"]
        or reference_plan["plan_sha256"] != binding["plan_sha256"]
        or reference_plan["evaluation_scope"] != binding["evaluation_scope"]
    ):
        raise ConditionedAnalysisError(
            "reference plan differs from the immutable v3 binding"
        )
    validate_reference_plan_jobs(reference_plan)
    for dataset, subjects in (
        ("local_exp4", (1, 4, 7)),
        ("bnci2014_001", (1, 5, 9)),
        ("bnci2014_004", (1, 5, 9)),
        ("cho2017", (1, 18, 35, 52)),
        ("physionet_mi", (1, 18, 36, 54)),
    ):
        for subject in subjects:
            subject_key = _subject_key(dataset, subject)
            split_key = _split_key(dataset, subject)
            if (
                conditioned_plan["cache_identity"][subject_key]
                != reference_plan["cache_identity"][subject_key]
            ):
                raise ConditionedAnalysisError(
                    f"reference cache binding differs for {subject_key}"
                )
            if (
                conditioned_plan["split_identity"][split_key]
                != reference_plan["split_identity"][split_key]
            ):
                raise ConditionedAnalysisError(
                    f"reference split binding differs for {split_key}"
                )


def _conditioned_row(
    job: ScreenJob, payload: Mapping[str, Any]
) -> dict[str, Any]:
    metrics = payload["validation_metrics"]
    return {
        "source": "conditioned_v3",
        "job_id": job.job_id,
        **job.identity(),
        "validation_count": int(payload["split"]["validation"]["count"]),
        **{name: float(metrics[name]) for name in sorted(METRIC_NAMES)},
        "parameter_count": int(payload["model"]["parameter_count"]),
        "best_epoch": int(payload["fit"]["best_epoch"]),
        "epochs_run": int(payload["fit"]["epochs_run"]),
        "fit_seconds": float(payload["timing"]["fit_seconds"]),
    }


def _reference_row(
    job: ScreenJob, payload: Mapping[str, Any]
) -> dict[str, Any]:
    metrics = payload["validation_metrics"]
    return {
        "source": "bound_reference",
        "job_id": job.job_id,
        **job.identity(),
        "validation_count": int(payload["split"]["validation"]["count"]),
        **{name: float(metrics[name]) for name in sorted(METRIC_NAMES)},
        "parameter_count": int(payload["model"]["parameter_count"]),
        "best_epoch": int(payload["fit"]["best_epoch"]),
        "epochs_run": int(payload["fit"]["epochs_run"]),
        "fit_seconds": float(payload["timing"]["fit_seconds"]),
    }


def _load_conditioned_rows(
    run_root: Path,
    plan: Mapping[str, Any],
) -> list[dict[str, Any]]:
    expected_jobs = conditioned_jobs()
    expected_paths = {
        _record_path(run_root, job).resolve() for job in expected_jobs
    }
    records_root = run_root / "records"
    observed_paths = (
        {path.resolve() for path in records_root.rglob("*.json")}
        if records_root.exists()
        else set()
    )
    unexpected = observed_paths - expected_paths
    if unexpected:
        raise ConditionedAnalysisError(
            f"conditioned run has {len(unexpected)} unexpected JSON records"
        )

    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for job in expected_jobs:
        path = _record_path(run_root, job)
        if not path.is_file():
            errors.append(f"missing {job.job_id}")
            continue
        try:
            payload = _strict_json_load(path)
            validate_record_payload(
                payload,
                job=job,
                plan_sha256=str(plan["plan_sha256"]),
            )
            _validate_record_bindings(payload, job=job, plan=plan)
            rows.append(_conditioned_row(job, payload))
        except Exception as error:
            errors.append(f"{job.job_id}: {type(error).__name__}: {error}")
    if errors or len(rows) != EXPECTED_CONDITIONED_RECORDS:
        preview = "; ".join(errors[:5])
        raise ConditionedAnalysisError(
            "all 51 conditioned records must validate before analysis"
            + (f": {preview}" if preview else "")
        )
    return rows


def _load_reference_rows(
    reference_root: Path,
    reference_plan: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for job in _reference_jobs():
        path = reference_record_path(reference_root, job)
        if not path.is_file():
            errors.append(f"missing {job.job_id}")
            continue
        try:
            payload = _strict_json_load(path)
            validate_reference_record_payload(
                payload,
                job=job,
                plan_sha256=str(reference_plan["plan_sha256"]),
            )
            validate_reference_record_bindings(
                payload, job=job, plan=reference_plan
            )
            rows.append(_reference_row(job, payload))
        except Exception as error:
            errors.append(f"{job.job_id}: {type(error).__name__}: {error}")
    if errors or len(rows) != EXPECTED_REFERENCE_RECORDS:
        preview = "; ".join(errors[:5])
        raise ConditionedAnalysisError(
            "all 51 bound reference records must validate before analysis"
            + (f": {preview}" if preview else "")
        )
    return rows


def _dataset_rows(
    job_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in job_rows:
        grouped[(str(row["dataset"]), str(row["model"]))].append(row)
    all_models = (*CONDITIONED_MODELS, *REFERENCE_MODELS)
    result: list[dict[str, Any]] = []
    for model in all_models:
        for dataset in DATASETS:
            rows = grouped[(dataset, model)]
            if not rows:
                raise ConditionedAnalysisError(
                    f"no validated rows for {dataset}/{model}"
                )
            result.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "n_subjects": len(rows),
                    **{
                        f"{metric}_mean": _mean(
                            [float(row[metric]) for row in rows]
                        )
                        for metric in sorted(METRIC_NAMES)
                    },
                    "balanced_accuracy_sd": _sample_sd(
                        [float(row["balanced_accuracy"]) for row in rows]
                    ),
                }
            )
    return result


def _overall_rows(
    dataset_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_model: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in dataset_rows:
        by_model[str(row["model"])].append(row)
    result: list[dict[str, Any]] = []
    for model in (*CONDITIONED_MODELS, *REFERENCE_MODELS):
        rows = by_model[model]
        if {str(row["dataset"]) for row in rows} != set(DATASETS):
            raise ConditionedAnalysisError(
                f"{model} lacks complete five-dataset coverage"
            )
        result.append(
            {
                "model": model,
                "n_datasets": len(DATASETS),
                **{
                    f"equal_dataset_{metric}": _mean(
                        [float(row[f"{metric}_mean"]) for row in rows]
                    )
                    for metric in sorted(METRIC_NAMES)
                },
            }
        )
    result.sort(
        key=lambda row: float(row["equal_dataset_balanced_accuracy"]),
        reverse=True,
    )
    for rank, row in enumerate(result, start=1):
        row["balanced_accuracy_rank"] = rank
    return result


def _paired_comparisons(
    job_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    lookup = {
        (
            str(row["dataset"]),
            int(row["subject"]),
            int(row["fold"]),
            int(row["seed"]),
            str(row["model"]),
        ): row
        for row in job_rows
    }
    subjects = {
        dataset: tuple(
            sorted(
                {
                    int(row["subject"])
                    for row in job_rows
                    if row["dataset"] == dataset
                }
            )
        )
        for dataset in DATASETS
    }
    result: list[dict[str, Any]] = []
    for candidate in CONDITIONED_MODELS:
        for reference in REFERENCE_MODELS:
            dataset_summaries: list[dict[str, Any]] = []
            for dataset in DATASETS:
                deltas: list[float] = []
                accuracy_deltas: list[float] = []
                for subject in subjects[dataset]:
                    identity = (dataset, subject, 0, 7)
                    candidate_row = lookup[(*identity, candidate)]
                    reference_row = lookup[(*identity, reference)]
                    deltas.append(
                        float(candidate_row["balanced_accuracy"])
                        - float(reference_row["balanced_accuracy"])
                    )
                    accuracy_deltas.append(
                        float(candidate_row["accuracy"])
                        - float(reference_row["accuracy"])
                    )
                wins = sum(delta > PAIR_TIE_TOLERANCE for delta in deltas)
                losses = sum(delta < -PAIR_TIE_TOLERANCE for delta in deltas)
                row = {
                    "scope": "dataset",
                    "dataset": dataset,
                    "candidate": candidate,
                    "reference": reference,
                    "n_pairs": len(deltas),
                    "balanced_accuracy_delta": _mean(deltas),
                    "accuracy_delta": _mean(accuracy_deltas),
                    "wins": wins,
                    "ties": len(deltas) - wins - losses,
                    "losses": losses,
                    "win_rate": wins / len(deltas),
                }
                dataset_summaries.append(row)
                result.append(row)

            all_dataset_deltas = [
                float(row["balanced_accuracy_delta"])
                for row in dataset_summaries
            ]
            total_pairs = sum(int(row["n_pairs"]) for row in dataset_summaries)
            total_wins = sum(int(row["wins"]) for row in dataset_summaries)
            total_losses = sum(int(row["losses"]) for row in dataset_summaries)
            result.append(
                {
                    "scope": "equal_dataset",
                    "dataset": None,
                    "candidate": candidate,
                    "reference": reference,
                    "n_pairs": total_pairs,
                    "balanced_accuracy_delta": _mean(all_dataset_deltas),
                    "accuracy_delta": _mean(
                        [
                            float(row["accuracy_delta"])
                            for row in dataset_summaries
                        ]
                    ),
                    "wins": total_wins,
                    "ties": total_pairs - total_wins - total_losses,
                    "losses": total_losses,
                    "win_rate": total_wins / total_pairs,
                    "nonnegative_datasets": sum(
                        delta >= 0.0 for delta in all_dataset_deltas
                    ),
                    "minimum_dataset_delta": min(all_dataset_deltas),
                }
            )
    return result


def _decision(
    overall_rows: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    overall = {
        str(row["model"]): float(row["equal_dataset_balanced_accuracy"])
        for row in overall_rows
    }
    candidate_scores = {
        model: overall[model] for model in CONDITIONED_MODELS
    }
    best_score = max(candidate_scores.values())
    best = [
        model
        for model, score in candidate_scores.items()
        if math.isclose(
            score, best_score, rel_tol=0.0, abs_tol=PAIR_TIE_TOLERANCE
        )
    ]
    pair_lookup = {
        (str(row["candidate"]), str(row["reference"])): row
        for row in pair_rows
        if row["scope"] == "equal_dataset"
    }

    diagnostics: dict[str, dict[str, Any]] = {}
    for candidate in CONDITIONED_MODELS:
        by_reference: dict[str, dict[str, Any]] = {}
        for reference in REFERENCE_MODELS:
            row = pair_lookup[(candidate, reference)]
            passed = (
                float(row["balanced_accuracy_delta"]) > 0.0
                and int(row["nonnegative_datasets"])
                >= MIN_NONNEGATIVE_DATASETS_PER_REFERENCE
                and float(row["minimum_dataset_delta"])
                >= -MAX_DATASET_DEFICIT
                and float(row["win_rate"]) >= MIN_PAIRED_WIN_RATE
            )
            by_reference[reference] = {
                "passed": passed,
                "equal_dataset_balanced_accuracy_delta": row[
                    "balanced_accuracy_delta"
                ],
                "nonnegative_datasets": row["nonnegative_datasets"],
                "minimum_dataset_delta": row["minimum_dataset_delta"],
                "paired_win_rate": row["win_rate"],
            }
        diagnostics[candidate] = {
            "unique_top_conditioned_variant": best == [candidate],
            "all_reference_gates_passed": all(
                item["passed"] for item in by_reference.values()
            ),
            "reference_gates": by_reference,
        }

    selected = (
        best[0]
        if len(best) == 1
        and diagnostics[best[0]]["all_reference_gates_passed"]
        else None
    )
    return {
        "decision": (
            "choose_one_for_next_prespecified_stage"
            if selected is not None
            else "kill_conditioned_family"
        ),
        "selected_model": selected,
        "rule": {
            "unique_top_conditioned_variant_required": True,
            "references": list(REFERENCE_MODELS),
            "minimum_equal_dataset_delta_per_reference": "strictly_positive",
            "minimum_nonnegative_datasets_per_reference": (
                MIN_NONNEGATIVE_DATASETS_PER_REFERENCE
            ),
            "maximum_dataset_deficit": MAX_DATASET_DEFICIT,
            "minimum_paired_win_rate": MIN_PAIRED_WIN_RATE,
        },
        "candidate_diagnostics": diagnostics,
    }


def analyze_run(
    run_root: Path,
    reference_run_root: Path,
) -> dict[str, Any]:
    """Validate, join, and summarize without opening an EEG cache."""

    root = run_root.resolve()
    reference_root = reference_run_root.resolve()
    conditioned_plan = load_plan(root)
    reference_plan = load_reference_plan(reference_root)
    _validate_reference_binding(
        conditioned_plan=conditioned_plan,
        reference_root=reference_root,
        reference_plan=reference_plan,
    )
    conditioned_rows = _load_conditioned_rows(root, conditioned_plan)
    reference_rows = _load_reference_rows(reference_root, reference_plan)
    all_rows = [*conditioned_rows, *reference_rows]
    dataset_rows = _dataset_rows(all_rows)
    overall_rows = _overall_rows(dataset_rows)
    paired_rows = _paired_comparisons(all_rows)
    decision = _decision(overall_rows, paired_rows)
    return {
        "schema": ANALYSIS_SCHEMA,
        "evaluation_scope": conditioned_plan["evaluation_scope"],
        "conditioned_plan_sha256": conditioned_plan["plan_sha256"],
        "reference_plan_sha256": reference_plan["plan_sha256"],
        "analysis_policy": {
            "cache_files_opened": False,
            "trial_arrays_opened": False,
            "conditioned_records_required": EXPECTED_CONDITIONED_RECORDS,
            "reference_records_required": EXPECTED_REFERENCE_RECORDS,
            "primary_metric": "equal_dataset_balanced_accuracy",
            "aggregation": (
                "subject_validation_score_then_dataset_mean_then_equal_dataset_mean"
            ),
        },
        "audit": {
            "conditioned_records_valid": len(conditioned_rows),
            "reference_records_valid": len(reference_rows),
            "paired_subject_records": EXPECTED_CONDITIONED_RECORDS,
            "complete": True,
        },
        "jobs": all_rows,
        "dataset_scores": dataset_rows,
        "overall_scores": overall_rows,
        "paired_comparisons": paired_rows,
        "selection": decision,
        "interpretation_limits": [
            "This is opened-development validation evidence used for selection.",
            "One seed and 17 curated subjects do not establish broad robustness.",
            "The output authorizes at most one candidate for a later prespecified stage.",
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
                    value,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            elif value is None:
                normalized[key] = ""
            else:
                normalized[key] = value
        writer.writerow(normalized)
    return output.getvalue()


def _markdown(analysis: Mapping[str, Any]) -> str:
    selection = analysis["selection"]
    lines = [
        "# Conditioned v3 development decision",
        "",
        f"- Decision: `{selection['decision']}`",
        f"- Selected model: `{selection['selected_model']}`",
        f"- Conditioned records validated: {analysis['audit']['conditioned_records_valid']}",
        f"- Reference records validated: {analysis['audit']['reference_records_valid']}",
        f"- Conditioned plan: `{analysis['conditioned_plan_sha256']}`",
        f"- Reference plan: `{analysis['reference_plan_sha256']}`",
        "",
        "## Equal-dataset balanced accuracy",
        "",
        "| Rank | Model | Score |",
        "|---:|---|---:|",
    ]
    for row in analysis["overall_scores"]:
        lines.append(
            f"| {row['balanced_accuracy_rank']} | `{row['model']}` | "
            f"{float(row['equal_dataset_balanced_accuracy']):.6f} |"
        )
    lines.extend(
        [
            "",
            "This report is limited to the fixed opened-development validation screen.",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_analysis_bundle(
    analysis: Mapping[str, Any],
    output_root: Path,
    *,
    reference_run_root: Path,
) -> tuple[Path, ...]:
    output_root = output_root.resolve()
    reference_root = reference_run_root.resolve()
    if output_root == reference_root or reference_root in output_root.parents:
        raise ConditionedAnalysisError(
            "analysis output must not modify the reference run"
        )
    artifacts = {
        output_root / "conditioned_analysis.json": (
            json.dumps(analysis, sort_keys=True, indent=2, allow_nan=False) + "\n"
        ),
        output_root / "conditioned_jobs.csv": _csv_text(analysis["jobs"]),
        output_root / "dataset_scores.csv": _csv_text(
            analysis["dataset_scores"]
        ),
        output_root / "paired_comparisons.csv": _csv_text(
            analysis["paired_comparisons"]
        ),
        output_root / "DECISION.md": _markdown(analysis),
    }
    for path, content in artifacts.items():
        _atomic_write_text(path, content)
    return tuple(artifacts)


def analyze_and_write(
    run_root: Path,
    reference_run_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    analysis = analyze_run(run_root, reference_run_root)
    write_analysis_bundle(
        analysis,
        output_root,
        reference_run_root=reference_run_root,
    )
    return analysis


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--reference-run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    analysis = analyze_and_write(
        args.run_root.resolve(),
        args.reference_run_root.resolve(),
        args.output_root.resolve(),
    )
    print(
        json.dumps(
            {
                "conditioned_records_valid": analysis["audit"][
                    "conditioned_records_valid"
                ],
                "reference_records_valid": analysis["audit"][
                    "reference_records_valid"
                ],
                "decision": analysis["selection"]["decision"],
                "selected_model": analysis["selection"]["selected_model"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ANALYSIS_SCHEMA",
    "ConditionedAnalysisError",
    "analyze_and_write",
    "analyze_run",
    "write_analysis_bundle",
]
