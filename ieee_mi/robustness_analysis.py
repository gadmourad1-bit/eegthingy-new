"""Read-only analysis for the disjoint-subject robustness amendment.

The analyzer opens only the immutable plan and aggregate JSON records.  It
rejects missing, unexpected, corrupt, or plan-inconsistent artifacts and must
validate all 460 jobs before constructing exactly one pass/stop decision.
EEG cache files are never opened.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import statistics
import uuid
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .development_screen import METRIC_NAMES, ScreenJob, _strict_json_load
from .robustness_screen import (
    AMENDMENT_LABEL,
    CANDIDATE_MODEL,
    DECISION_RULE,
    EVALUATION_SCOPE,
    EXPECTED_JOB_COUNT,
    EXPECTED_SUBJECT_COUNT,
    GATE_TOLERANCE,
    HISTORICAL_V3_DECISION,
    MODELS,
    REFERENCE_MODELS,
    ROBUSTNESS_COHORTS,
    _record_path,
    _validate_record_bindings,
    load_plan,
    robustness_environment_identity,
    robustness_jobs,
    source_identity,
    validate_record_payload,
)

ANALYSIS_SCHEMA = "ieee-mi-chsd-disjoint-robustness-analysis-v1"
DATASETS: tuple[str, ...] = tuple(dataset for dataset, _ in ROBUSTNESS_COHORTS)
PAIR_TIE_TOLERANCE = GATE_TOLERANCE


class RobustnessAnalysisError(RuntimeError):
    """Raised when complete plan-bound aggregate evidence is unavailable."""


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise RobustnessAnalysisError("cannot average an empty score sequence")
    return statistics.fmean(values)


def _sample_sd(values: Sequence[float]) -> float | None:
    return statistics.stdev(values) if len(values) >= 2 else None


def _job_row(job: ScreenJob, payload: Mapping[str, Any]) -> dict[str, Any]:
    metrics = payload["validation_metrics"]
    return {
        "job_id": job.job_id,
        **job.identity(),
        "validation_count": int(payload["split"]["validation"]["count"]),
        **{name: float(metrics[name]) for name in sorted(METRIC_NAMES)},
        "parameter_count": int(payload["model"]["parameter_count"]),
        "best_epoch": int(payload["fit"]["best_epoch"]),
        "epochs_run": int(payload["fit"]["epochs_run"]),
        "fit_seconds": float(payload["timing"]["fit_seconds"]),
    }


def _load_all_rows(
    run_root: Path,
    plan: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Validate every expected artifact before returning any analysis rows."""

    expected_jobs = robustness_jobs()
    expected_paths = {_record_path(run_root, job) for job in expected_jobs}
    records_root = run_root / "records"
    if records_root.is_symlink():
        raise RobustnessAnalysisError("robustness records root is a symlink")
    entries = tuple(records_root.rglob("*")) if records_root.exists() else ()
    symlinks = [path for path in entries if path.is_symlink()]
    if symlinks:
        raise RobustnessAnalysisError(
            f"robustness records contain {len(symlinks)} symlink artifacts"
        )
    observed_paths = {path for path in entries if path.is_file()}
    unexpected = observed_paths - expected_paths
    if unexpected:
        raise RobustnessAnalysisError(
            f"robustness run has {len(unexpected)} unexpected record artifacts"
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
            rows.append(_job_row(job, payload))
        except Exception as error:  # noqa: BLE001 - aggregate all record failures
            errors.append(f"{job.job_id}: {type(error).__name__}: {error}")
    if errors or len(rows) != EXPECTED_JOB_COUNT:
        preview = "; ".join(errors[:8])
        raise RobustnessAnalysisError(
            "all 460 robustness records must validate before analysis"
            + (f": {preview}" if preview else "")
        )
    if len({row["job_id"] for row in rows}) != EXPECTED_JOB_COUNT:
        raise RobustnessAnalysisError("validated robustness job IDs are duplicated")
    return rows


def _dataset_rows(
    job_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in job_rows:
        grouped[(str(row["dataset"]), str(row["model"]))].append(row)
    subject_counts = {
        dataset: len(subjects) for dataset, subjects in ROBUSTNESS_COHORTS
    }
    result: list[dict[str, Any]] = []
    for model in MODELS:
        for dataset in DATASETS:
            rows = grouped[(dataset, model)]
            if len(rows) != subject_counts[dataset]:
                raise RobustnessAnalysisError(
                    f"{dataset}/{model} has {len(rows)} subjects, expected "
                    f"{subject_counts[dataset]}"
                )
            if len({int(row["subject"]) for row in rows}) != len(rows):
                raise RobustnessAnalysisError(
                    f"{dataset}/{model} has duplicate subject rows"
                )
            result.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "n_subjects": len(rows),
                    **{
                        f"{metric}_mean": _mean([float(row[metric]) for row in rows])
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
    for model in MODELS:
        rows = by_model[model]
        if len(rows) != len(DATASETS) or {str(row["dataset"]) for row in rows} != set(
            DATASETS
        ):
            raise RobustnessAnalysisError(
                f"{model} lacks complete five-dataset coverage"
            )
        result.append(
            {
                "model": model,
                "n_datasets": len(DATASETS),
                "n_subjects": EXPECTED_SUBJECT_COUNT,
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
    if len(lookup) != EXPECTED_JOB_COUNT:
        raise RobustnessAnalysisError("paired lookup is not one-to-one")

    result: list[dict[str, Any]] = []
    for reference in REFERENCE_MODELS:
        dataset_summaries: list[dict[str, Any]] = []
        all_deltas: list[float] = []
        all_accuracy_deltas: list[float] = []
        for dataset, subjects in ROBUSTNESS_COHORTS:
            deltas: list[float] = []
            accuracy_deltas: list[float] = []
            for subject in subjects:
                identity = (dataset, subject, 0, 7)
                try:
                    candidate_row = lookup[(*identity, CANDIDATE_MODEL)]
                    reference_row = lookup[(*identity, reference)]
                except KeyError as error:
                    raise RobustnessAnalysisError(
                        f"paired row is absent for {identity}/{reference}"
                    ) from error
                deltas.append(
                    float(candidate_row["balanced_accuracy"])
                    - float(reference_row["balanced_accuracy"])
                )
                accuracy_deltas.append(
                    float(candidate_row["accuracy"]) - float(reference_row["accuracy"])
                )
            wins = sum(delta > PAIR_TIE_TOLERANCE for delta in deltas)
            losses = sum(delta < -PAIR_TIE_TOLERANCE for delta in deltas)
            row = {
                "scope": "dataset",
                "dataset": dataset,
                "candidate": CANDIDATE_MODEL,
                "reference": reference,
                "n_pairs": len(deltas),
                "balanced_accuracy_delta": _mean(deltas),
                "accuracy_delta": _mean(accuracy_deltas),
                "wins": wins,
                "ties": len(deltas) - wins - losses,
                "losses": losses,
                "win_rate": wins / len(deltas),
            }
            result.append(row)
            dataset_summaries.append(row)
            all_deltas.extend(deltas)
            all_accuracy_deltas.extend(accuracy_deltas)

        if len(all_deltas) != EXPECTED_SUBJECT_COUNT:
            raise RobustnessAnalysisError(
                f"{reference} paired subject cardinality changed"
            )
        wins = sum(delta > PAIR_TIE_TOLERANCE for delta in all_deltas)
        losses = sum(delta < -PAIR_TIE_TOLERANCE for delta in all_deltas)
        dataset_deltas = [
            float(row["balanced_accuracy_delta"]) for row in dataset_summaries
        ]
        result.append(
            {
                "scope": "all_subjects_and_equal_dataset",
                "dataset": None,
                "candidate": CANDIDATE_MODEL,
                "reference": reference,
                "n_pairs": len(all_deltas),
                "equal_dataset_balanced_accuracy_delta": _mean(dataset_deltas),
                "pooled_subject_balanced_accuracy_delta": _mean(all_deltas),
                "pooled_subject_accuracy_delta": _mean(all_accuracy_deltas),
                "wins": wins,
                "ties": len(all_deltas) - wins - losses,
                "losses": losses,
                "paired_subject_win_rate": wins / len(all_deltas),
                "nonnegative_datasets": sum(
                    delta >= -PAIR_TIE_TOLERANCE for delta in dataset_deltas
                ),
                "minimum_dataset_delta": min(dataset_deltas),
            }
        )
    return result


def _decision(
    overall_rows: Sequence[Mapping[str, Any]],
    paired_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply only the pre-encoded all-conditions robustness gate."""

    overall = {
        str(row["model"]): float(row["equal_dataset_balanced_accuracy"])
        for row in overall_rows
    }
    if set(overall) != set(MODELS):
        raise RobustnessAnalysisError("decision lacks one or more model scores")
    candidate_score = overall[CANDIDATE_MODEL]
    strongest_score = max(overall[reference] for reference in REFERENCE_MODELS)
    strongest_references = [
        reference
        for reference in REFERENCE_MODELS
        if math.isclose(
            overall[reference],
            strongest_score,
            rel_tol=0.0,
            abs_tol=PAIR_TIE_TOLERANCE,
        )
    ]
    gap = strongest_score - candidate_score
    within_strongest = gap <= (
        float(DECISION_RULE["maximum_gap_to_strongest_reference"])
        + PAIR_TIE_TOLERANCE
    )
    above_tcformer = (
        candidate_score - overall["tcformer"] > PAIR_TIE_TOLERANCE
    )

    pair_lookup = {
        str(row["reference"]): row
        for row in paired_rows
        if row["scope"] == "all_subjects_and_equal_dataset"
    }
    if set(pair_lookup) != set(REFERENCE_MODELS):
        raise RobustnessAnalysisError("decision lacks paired reference summaries")

    reference_diagnostics: dict[str, dict[str, Any]] = {}
    for reference in REFERENCE_MODELS:
        row = pair_lookup[reference]
        nonnegative = int(row["nonnegative_datasets"]) >= int(
            DECISION_RULE["minimum_nonnegative_datasets_per_reference"]
        )
        bounded_deficit = float(row["minimum_dataset_delta"]) >= (
            -float(DECISION_RULE["maximum_dataset_deficit_per_reference"])
            - PAIR_TIE_TOLERANCE
        )
        sufficient_wins = float(row["paired_subject_win_rate"]) >= (
            float(DECISION_RULE["minimum_paired_subject_win_rate_per_reference"])
            - PAIR_TIE_TOLERANCE
        )
        reference_diagnostics[reference] = {
            "passed": nonnegative and bounded_deficit and sufficient_wins,
            "nonnegative_dataset_gate_passed": nonnegative,
            "dataset_deficit_gate_passed": bounded_deficit,
            "paired_subject_win_rate_gate_passed": sufficient_wins,
            "nonnegative_datasets": int(row["nonnegative_datasets"]),
            "minimum_dataset_delta": float(row["minimum_dataset_delta"]),
            "paired_subject_win_rate": float(row["paired_subject_win_rate"]),
            "equal_dataset_balanced_accuracy_delta": float(
                row["equal_dataset_balanced_accuracy_delta"]
            ),
        }

    passed = (
        within_strongest
        and above_tcformer
        and all(diagnostic["passed"] for diagnostic in reference_diagnostics.values())
    )
    return {
        "decision": ("pass_robustness_gate" if passed else "stop_chsd_promotion"),
        "passed": passed,
        "candidate": CANDIDATE_MODEL,
        "rule": dict(DECISION_RULE),
        "candidate_equal_dataset_balanced_accuracy": candidate_score,
        "strongest_reference_models": strongest_references,
        "strongest_reference_equal_dataset_balanced_accuracy": strongest_score,
        "gap_to_strongest_reference": gap,
        "within_one_percentage_point_gate_passed": within_strongest,
        "strictly_above_tcformer_gate_passed": above_tcformer,
        "tcformer_equal_dataset_balanced_accuracy": overall["tcformer"],
        "reference_diagnostics": reference_diagnostics,
    }


def analyze_run(run_root: Path) -> dict[str, Any]:
    """Validate all records and then return one deterministic pass/stop result."""

    root = run_root.resolve()
    plan = load_plan(root)
    if source_identity() != plan["source_identity"]:
        raise RobustnessAnalysisError("analysis source differs from the immutable plan")
    if robustness_environment_identity() != plan["environment_identity"]:
        raise RobustnessAnalysisError(
            "analysis UV/runtime environment differs from the immutable plan"
        )
    rows = _load_all_rows(root, plan)
    dataset_rows = _dataset_rows(rows)
    overall_rows = _overall_rows(dataset_rows)
    paired_rows = _paired_comparisons(rows)
    decision = _decision(overall_rows, paired_rows)
    result = {
        "schema": ANALYSIS_SCHEMA,
        "amendment_label": AMENDMENT_LABEL,
        "evaluation_scope": EVALUATION_SCOPE,
        "confirmation_evidence": False,
        "plan_sha256": plan["plan_sha256"],
        "lineage": {
            **dict(plan["lineage"]),
            "reference_v2": dict(plan["lineage"]["reference_v2"]),
            "conditioned_v3": dict(plan["lineage"]["conditioned_v3"]),
            "conditioned_v3_analysis": dict(plan["lineage"]["conditioned_v3_analysis"]),
        },
        "analysis_policy": {
            "cache_files_opened": False,
            "trial_arrays_opened": False,
            "records_required_before_decision": EXPECTED_JOB_COUNT,
            "primary_metric": "equal_dataset_balanced_accuracy",
            "aggregation": (
                "subject_validation_score_then_dataset_mean_then_equal_dataset_mean"
            ),
            "paired_win_rate_aggregation": (
                "strict_subject_wins_divided_by_all_115_paired_subjects"
            ),
        },
        "audit": {
            "expected_records": EXPECTED_JOB_COUNT,
            "valid_records": len(rows),
            "expected_subjects": EXPECTED_SUBJECT_COUNT,
            "models": len(MODELS),
            "complete": True,
            "decision_count": 1,
        },
        "jobs": rows,
        "dataset_scores": dataset_rows,
        "overall_scores": overall_rows,
        "paired_comparisons": paired_rows,
        "robustness_decision": decision,
        "interpretation_limits": [
            (
                "This is a transparent post-v3 exploratory amendment using "
                "opened-development validation evidence."
            ),
            (
                "The historical v3 decision remains kill_conditioned_family "
                "and is not rewritten by this analysis."
            ),
            (
                "A pass permits only a later prespecified stage; it is not "
                "confirmation evidence and cannot establish state of the art."
            ),
            (
                "A failure requires stopping promotion of this CHSD candidate "
                "under the encoded rule."
            ),
        ],
    }
    validate_analysis_payload(result)
    return result


def validate_analysis_payload(result: Mapping[str, Any]) -> None:
    """Validate the one-decision, complete-record analysis artifact."""

    expected_top = {
        "schema",
        "amendment_label",
        "evaluation_scope",
        "confirmation_evidence",
        "plan_sha256",
        "lineage",
        "analysis_policy",
        "audit",
        "jobs",
        "dataset_scores",
        "overall_scores",
        "paired_comparisons",
        "robustness_decision",
        "interpretation_limits",
    }
    if set(result) != expected_top:
        raise RobustnessAnalysisError("analysis fields differ from v1")
    if (
        result["schema"] != ANALYSIS_SCHEMA
        or result["amendment_label"] != AMENDMENT_LABEL
        or result["evaluation_scope"] != EVALUATION_SCOPE
        or result["confirmation_evidence"] is not False
    ):
        raise RobustnessAnalysisError("analysis scope or schema changed")
    audit = result["audit"]
    if (
        not isinstance(audit, Mapping)
        or int(audit.get("expected_records", -1)) != EXPECTED_JOB_COUNT
        or int(audit.get("valid_records", -1)) != EXPECTED_JOB_COUNT
        or int(audit.get("expected_subjects", -1)) != EXPECTED_SUBJECT_COUNT
        or int(audit.get("models", -1)) != len(MODELS)
        or audit.get("complete") is not True
        or int(audit.get("decision_count", -1)) != 1
    ):
        raise RobustnessAnalysisError(
            "analysis was not built from all 460 validated records"
        )
    if (
        len(result["jobs"]) != EXPECTED_JOB_COUNT
        or len(result["dataset_scores"]) != len(DATASETS) * len(MODELS)
        or len(result["overall_scores"]) != len(MODELS)
        or len(result["paired_comparisons"])
        != len(REFERENCE_MODELS) * (len(DATASETS) + 1)
    ):
        raise RobustnessAnalysisError("analysis table cardinality changed")
    decision = result["robustness_decision"]
    if (
        not isinstance(decision, Mapping)
        or decision.get("decision")
        not in {"pass_robustness_gate", "stop_chsd_promotion"}
        or not isinstance(decision.get("passed"), bool)
        or decision.get("rule") != dict(DECISION_RULE)
        or (decision["passed"] != (decision["decision"] == "pass_robustness_gate"))
    ):
        raise RobustnessAnalysisError("analysis does not contain one valid decision")
    lineage_analysis = result["lineage"]["conditioned_v3_analysis"]
    if (
        lineage_analysis["decision"] != HISTORICAL_V3_DECISION
        or lineage_analysis["selected_model"] is not None
    ):
        raise RobustnessAnalysisError("analysis rewrites the historical v3 decision")


def _csv_text(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        raise RobustnessAnalysisError("cannot serialize an empty CSV table")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return stream.getvalue()


def _report_markdown(result: Mapping[str, Any]) -> str:
    decision = result["robustness_decision"]
    lines = [
        "# Disjoint-subject CHSD robustness amendment",
        "",
        f"- Decision: `{decision['decision']}`",
        f"- Passed: `{str(decision['passed']).lower()}`",
        f"- Valid aggregate records: {result['audit']['valid_records']}",
        f"- Candidate: `{CANDIDATE_MODEL}`",
        "",
        "## Equal-dataset balanced accuracy",
        "",
        "| Rank | Model | Balanced accuracy |",
        "|---:|---|---:|",
    ]
    for row in result["overall_scores"]:
        lines.append(
            f"| {row['balanced_accuracy_rank']} | {row['model']} | "
            f"{100.0 * float(row['equal_dataset_balanced_accuracy']):.3f}% |"
        )
    lines.extend(
        [
            "",
            "## Gate diagnostics",
            "",
            (
                f"- Gap to strongest reference: "
                f"{100.0 * float(decision['gap_to_strongest_reference']):.3f} "
                "percentage points."
            ),
            (
                "- Within 1.0-point gate: "
                f"`{str(decision['within_one_percentage_point_gate_passed']).lower()}`."
            ),
            (
                "- Strictly above TCFormer: "
                f"`{str(decision['strictly_above_tcformer_gate_passed']).lower()}`."
            ),
            "",
        ]
    )
    for reference, diagnostic in decision["reference_diagnostics"].items():
        lines.extend(
            [
                f"### {reference}",
                "",
                f"- Passed: `{str(diagnostic['passed']).lower()}`",
                (f"- Nonnegative datasets: {diagnostic['nonnegative_datasets']}/5"),
                (
                    f"- Worst dataset delta: "
                    f"{100.0 * float(diagnostic['minimum_dataset_delta']):.3f} "
                    "percentage points"
                ),
                (
                    f"- Paired-subject win rate: "
                    f"{100.0 * float(diagnostic['paired_subject_win_rate']):.2f}%"
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            (
                "This is exploratory opened-development validation evidence. "
                "It does not reverse the v3 kill decision and is not "
                "confirmation evidence."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_create_text(path: Path, text: str) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / (f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.partial")
    try:
        with temporary.open("x", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            return False
        return True
    finally:
        temporary.unlink(missing_ok=True)


def _write_or_validate_text(path: Path, text: str) -> None:
    if _atomic_create_text(path, text):
        return
    if path.is_symlink() or not path.is_file():
        raise RobustnessAnalysisError(
            f"existing analysis artifact is not a regular file: {path}"
        )
    if path.read_bytes() != text.encode("utf-8"):
        raise RobustnessAnalysisError(f"existing analysis artifact differs: {path}")


def analyze_and_write(run_root: Path, output_dir: Path) -> dict[str, Any]:
    """Validate all 460 records, then atomically publish deterministic outputs."""

    root = run_root.resolve()
    if output_dir.is_symlink():
        raise RobustnessAnalysisError("analysis output must not be a symlink")
    output = output_dir.resolve()
    if output == root or output.parent != root.parent:
        raise RobustnessAnalysisError(
            "analysis output must be a sibling of, not within, the immutable run"
        )
    result = analyze_run(root)
    expected_names = {
        "analysis.json",
        "dataset_scores.csv",
        "overall_scores.csv",
        "paired_comparisons.csv",
        "REPORT.md",
    }
    if output.exists():
        if not output.is_dir():
            raise RobustnessAnalysisError("analysis output is not a directory")
        entries = tuple(output.iterdir())
        if any(path.is_symlink() for path in entries):
            raise RobustnessAnalysisError("analysis output contains a symlink")
        unexpected = {path.name for path in entries} - expected_names
        if unexpected:
            raise RobustnessAnalysisError(
                f"analysis output contains unexpected artifacts: {sorted(unexpected)}"
            )
    output.mkdir(parents=True, exist_ok=True)
    json_text = (
        json.dumps(
            result,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    _write_or_validate_text(output / "analysis.json", json_text)
    _write_or_validate_text(
        output / "dataset_scores.csv",
        _csv_text(result["dataset_scores"]),
    )
    _write_or_validate_text(
        output / "overall_scores.csv",
        _csv_text(result["overall_scores"]),
    )
    _write_or_validate_text(
        output / "paired_comparisons.csv",
        _csv_text(result["paired_comparisons"]),
    )
    _write_or_validate_text(output / "REPORT.md", _report_markdown(result))
    if {path.name for path in output.iterdir()} != expected_names:
        raise RobustnessAnalysisError("analysis output bundle is incomplete or polluted")
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = analyze_and_write(args.run_root.resolve(), args.output_dir.resolve())
    print(
        json.dumps(
            {
                "plan_sha256": result["plan_sha256"],
                "valid_records": result["audit"]["valid_records"],
                "decision": result["robustness_decision"]["decision"],
                "output_dir": str(args.output_dir.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ANALYSIS_SCHEMA",
    "RobustnessAnalysisError",
    "analyze_and_write",
    "analyze_run",
    "validate_analysis_payload",
]
