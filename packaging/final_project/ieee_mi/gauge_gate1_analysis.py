"""Deterministic decision analysis for the Gauge opened-17 gate.

The analyzer consumes all 17 sealed candidate records and the 51 plan-bound,
read-only comparator records as one descriptor-held coherent snapshot.  It
applies only the frozen rule from :mod:`ieee_mi.gauge_quotient_screen`.
No threshold, cohort, metric, or comparator can be selected at analysis time.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from . import gauge_gate1_runner as runner


ANALYSIS_SCHEMA: Final = "ieee-mi-gauge-gate1-decision-v1"
ANALYSIS_DIRECTORY: Final = "analysis_gate1"
ANALYSIS_MARKER: Final = b"ieee-mi-gauge-gate1-analysis-committed-v1\n"
PRIMARY_REFERENCE: Final = "cardinal_fbc_micro_extended"


def _mean(values: Sequence[float]) -> float:
    if not values or any(not math.isfinite(value) for value in values):
        raise runner.GaugeGate1Error("analysis mean input is empty or nonfinite")
    return math.fsum(values) / len(values)


def _score_map(
    *,
    candidate: Mapping[str, float],
    references: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    jobs = runner._candidate_manifest()  # noqa: SLF001
    expected_ids = {job.job_id for job in jobs}
    if set(candidate) != expected_ids:
        raise runner.GaugeGate1Error("candidate analysis cells are incomplete")
    if set(references) != set(runner.EXPECTED_REFERENCE_MODELS):
        raise runner.GaugeGate1Error("reference analysis model set changed")
    for model, values in references.items():
        if set(values) != expected_ids:
            raise runner.GaugeGate1Error(
                f"reference analysis cells are incomplete for {model}"
            )
    for value in (
        list(candidate.values())
        + [
            score
            for model in references.values()
            for score in model.values()
        ]
    ):
        if (
            type(value) not in {int, float}
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise runner.GaugeGate1Error(
                "balanced-accuracy analysis cell is invalid"
            )

    dataset_order = tuple(dict.fromkeys(job.dataset for job in jobs))
    dataset_rows: list[dict[str, Any]] = []
    equal_dataset: dict[str, float] = {}
    models = (runner.MODEL_NAME, *runner.EXPECTED_REFERENCE_MODELS)
    score_sources: dict[str, Mapping[str, float]] = {
        runner.MODEL_NAME: candidate,
        **references,
    }
    per_model_dataset: dict[str, dict[str, float]] = {
        model: {} for model in models
    }
    for dataset in dataset_order:
        dataset_jobs = tuple(job for job in jobs if job.dataset == dataset)
        for model in models:
            per_model_dataset[model][dataset] = _mean(
                [
                    float(score_sources[model][job.job_id])
                    for job in dataset_jobs
                ]
            )
        candidate_mean = per_model_dataset[runner.MODEL_NAME][dataset]
        reference_mean = per_model_dataset[PRIMARY_REFERENCE][dataset]
        dataset_rows.append(
            {
                "dataset": dataset,
                "subject_count": len(dataset_jobs),
                "model_balanced_accuracy": {
                    model: per_model_dataset[model][dataset]
                    for model in models
                },
                "candidate_balanced_accuracy": candidate_mean,
                "primary_reference_balanced_accuracy": reference_mean,
                "candidate_minus_primary_reference": (
                    candidate_mean - reference_mean
                ),
            }
        )
    for model in models:
        equal_dataset[model] = _mean(
            [per_model_dataset[model][dataset] for dataset in dataset_order]
        )

    deltas = [
        row["candidate_minus_primary_reference"] for row in dataset_rows
    ]
    wins = sum(
        float(candidate[job.job_id])
        > float(references[PRIMARY_REFERENCE][job.job_id])
        for job in jobs
    )
    strict_win_rate = wins / len(jobs)
    equal_delta = (
        equal_dataset[runner.MODEL_NAME] - equal_dataset[PRIMARY_REFERENCE]
    )
    rule = json.loads(runner.OPENED17_RULE_JSON)
    conditions = {
        "strictly_exceeds_primary_equal_dataset_balanced_accuracy": (
            equal_delta > 0.0
        ),
        "minimum_nonnegative_dataset_deltas": (
            sum(delta >= 0.0 for delta in deltas)
            >= rule["minimum_nonnegative_dataset_deltas"]
        ),
        "maximum_dataset_deficit": (
            min(deltas) >= -float(rule["maximum_dataset_deficit"])
        ),
        "minimum_strict_subject_win_rate": (
            strict_win_rate >= float(rule["minimum_strict_subject_win_rate"])
        ),
    }
    return {
        "dataset_order": list(dataset_order),
        "dataset_balanced_accuracy": dataset_rows,
        "model_equal_dataset_balanced_accuracy": equal_dataset,
        "candidate_minus_primary_equal_dataset_balanced_accuracy": equal_delta,
        "nonnegative_dataset_delta_count": sum(
            delta >= 0.0 for delta in deltas
        ),
        "worst_dataset_delta": min(deltas),
        "strict_subject_wins": wins,
        "strict_subject_count": len(jobs),
        "strict_subject_win_rate": strict_win_rate,
        "conditions": conditions,
        "passed": all(conditions.values()),
    }


def decision_from_balanced_accuracy(
    *,
    candidate: Mapping[str, float],
    references: Mapping[str, Mapping[str, float]],
    plan_sha256: str,
    reference_snapshot_sha256: str,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Apply the four frozen Gate 1 conditions to an exact 17-cell map."""

    runner._validate_sha(plan_sha256, path="analysis.plan_sha256")  # noqa: SLF001
    runner._validate_sha(  # noqa: SLF001
        reference_snapshot_sha256,
        path="analysis.reference_snapshot_sha256",
    )
    scores = _score_map(candidate=candidate, references=references)
    value = {
        "schema": ANALYSIS_SCHEMA,
        "created_at": created_at or datetime.now(UTC).isoformat(),
        "plan_sha256": plan_sha256,
        "evaluation_scope": runner.EVALUATION_SCOPE,
        "confirmation_evidence": False,
        "opened_development_only": True,
        "candidate": runner.MODEL_NAME,
        "primary_reference": PRIMARY_REFERENCE,
        "job_count": runner.EXPECTED_JOB_COUNT,
        "reference_record_count": runner.EXPECTED_REFERENCE_RECORD_COUNT,
        "reference_snapshot_sha256": reference_snapshot_sha256,
        "decision_rule": json.loads(runner.OPENED17_RULE_JSON),
        "decision_rule_json": runner.OPENED17_RULE_JSON,
        "decision_rule_sha256": runner.OPENED17_RULE_SHA256,
        **scores,
        "failure_action": json.loads(runner.OPENED17_RULE_JSON)[
            "failure_action"
        ],
        "interpretation": (
            "opened_development_gate_only_not_confirmation_not_sota_not_clinical"
        ),
    }
    validate_analysis(value)
    return value


def validate_analysis(value: Mapping[str, Any]) -> None:
    expected = {
        "schema",
        "created_at",
        "plan_sha256",
        "evaluation_scope",
        "confirmation_evidence",
        "opened_development_only",
        "candidate",
        "primary_reference",
        "job_count",
        "reference_record_count",
        "reference_snapshot_sha256",
        "decision_rule",
        "decision_rule_json",
        "decision_rule_sha256",
        "dataset_order",
        "dataset_balanced_accuracy",
        "model_equal_dataset_balanced_accuracy",
        "candidate_minus_primary_equal_dataset_balanced_accuracy",
        "nonnegative_dataset_delta_count",
        "worst_dataset_delta",
        "strict_subject_wins",
        "strict_subject_count",
        "strict_subject_win_rate",
        "conditions",
        "passed",
        "failure_action",
        "interpretation",
    }
    runner._exact_keys(value, expected, path="analysis")  # noqa: SLF001
    if (
        value["schema"] != ANALYSIS_SCHEMA
        or value["evaluation_scope"] != runner.EVALUATION_SCOPE
        or value["confirmation_evidence"] is not False
        or value["opened_development_only"] is not True
        or value["candidate"] != runner.MODEL_NAME
        or value["primary_reference"] != PRIMARY_REFERENCE
        or value["job_count"] != runner.EXPECTED_JOB_COUNT
        or value["reference_record_count"]
        != runner.EXPECTED_REFERENCE_RECORD_COUNT
        or value["decision_rule"] != json.loads(runner.OPENED17_RULE_JSON)
        or value["decision_rule_json"] != runner.OPENED17_RULE_JSON
        or value["decision_rule_sha256"] != runner.OPENED17_RULE_SHA256
        or value["failure_action"]
        != json.loads(runner.OPENED17_RULE_JSON)["failure_action"]
        or value["interpretation"]
        != "opened_development_gate_only_not_confirmation_not_sota_not_clinical"
    ):
        raise runner.GaugeGate1Error("analysis fixed identity changed")
    runner._validate_timestamp(value["created_at"], path="analysis.created_at")  # noqa: SLF001
    runner._validate_sha(value["plan_sha256"], path="analysis.plan_sha256")  # noqa: SLF001
    runner._validate_sha(  # noqa: SLF001
        value["reference_snapshot_sha256"],
        path="analysis.reference_snapshot_sha256",
    )
    rows = value["dataset_balanced_accuracy"]
    expected_dataset_counts: dict[str, int] = {}
    for job in runner._candidate_manifest():  # noqa: SLF001
        expected_dataset_counts[job.dataset] = (
            expected_dataset_counts.get(job.dataset, 0) + 1
        )
    if (
        not isinstance(rows, list)
        or value["dataset_order"] != [
            row["dataset"] for row in rows if isinstance(row, Mapping)
        ]
        or len(rows) != 5
        or value["dataset_order"] != list(expected_dataset_counts)
    ):
        raise runner.GaugeGate1Error("analysis dataset table changed")
    for row in rows:
        runner._exact_keys(  # noqa: SLF001
            row,
            {
                "dataset",
                "subject_count",
                "model_balanced_accuracy",
                "candidate_balanced_accuracy",
                "primary_reference_balanced_accuracy",
                "candidate_minus_primary_reference",
            },
            path="analysis.dataset",
        )
        if type(row["dataset"]) is not str or not row["dataset"]:
            raise runner.GaugeGate1Error("analysis dataset name is invalid")
        runner._require_int(  # noqa: SLF001
            row["subject_count"], path="analysis.subject_count", minimum=1
        )
        if row["subject_count"] != expected_dataset_counts[row["dataset"]]:
            raise runner.GaugeGate1Error(
                "analysis per-dataset subject count changed"
            )
        for name in (
            "candidate_balanced_accuracy",
            "primary_reference_balanced_accuracy",
        ):
            runner._require_number(  # noqa: SLF001
                row[name], path=f"analysis.{name}", minimum=0, maximum=1
            )
        model_scores = row["model_balanced_accuracy"]
        if (
            not isinstance(model_scores, Mapping)
            or set(model_scores)
            != {runner.MODEL_NAME, *runner.EXPECTED_REFERENCE_MODELS}
        ):
            raise runner.GaugeGate1Error(
                "analysis per-dataset model score set changed"
            )
        for model, score in model_scores.items():
            runner._require_number(  # noqa: SLF001
                score,
                path=f"analysis.dataset.{model}",
                minimum=0,
                maximum=1,
            )
        if (
            model_scores[runner.MODEL_NAME]
            != row["candidate_balanced_accuracy"]
            or model_scores[PRIMARY_REFERENCE]
            != row["primary_reference_balanced_accuracy"]
        ):
            raise runner.GaugeGate1Error(
                "analysis candidate/reference dataset aliases differ"
            )
        delta = runner._require_number(  # noqa: SLF001
            row["candidate_minus_primary_reference"],
            path="analysis.dataset_delta",
            minimum=-1,
            maximum=1,
        )
        if not math.isclose(
            delta,
            row["candidate_balanced_accuracy"]
            - row["primary_reference_balanced_accuracy"],
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise runner.GaugeGate1Error("analysis dataset delta is inconsistent")
    summaries = value["model_equal_dataset_balanced_accuracy"]
    if (
        not isinstance(summaries, Mapping)
        or set(summaries)
        != {runner.MODEL_NAME, *runner.EXPECTED_REFERENCE_MODELS}
    ):
        raise runner.GaugeGate1Error("analysis model summary set changed")
    for model, score in summaries.items():
        runner._require_number(  # noqa: SLF001
            score, path=f"analysis.model.{model}", minimum=0, maximum=1
        )
        recomputed = _mean(
            [row["model_balanced_accuracy"][model] for row in rows]
        )
        if not math.isclose(
            score,
            recomputed,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise runner.GaugeGate1Error(
                "analysis equal-dataset model summary is inconsistent"
            )
    delta = runner._require_number(  # noqa: SLF001
        value["candidate_minus_primary_equal_dataset_balanced_accuracy"],
        path="analysis.equal_dataset_delta",
        minimum=-1,
        maximum=1,
    )
    if not math.isclose(
        delta,
        summaries[runner.MODEL_NAME] - summaries[PRIMARY_REFERENCE],
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise runner.GaugeGate1Error("analysis equal-dataset delta is inconsistent")
    nonnegative = runner._require_int(  # noqa: SLF001
        value["nonnegative_dataset_delta_count"],
        path="analysis.nonnegative_dataset_delta_count",
        minimum=0,
    )
    wins = runner._require_int(  # noqa: SLF001
        value["strict_subject_wins"],
        path="analysis.strict_subject_wins",
        minimum=0,
    )
    count = runner._require_int(  # noqa: SLF001
        value["strict_subject_count"],
        path="analysis.strict_subject_count",
        minimum=1,
    )
    if count != runner.EXPECTED_JOB_COUNT or wins > count:
        raise runner.GaugeGate1Error("analysis subject counts differ")
    rate = runner._require_number(  # noqa: SLF001
        value["strict_subject_win_rate"],
        path="analysis.strict_subject_win_rate",
        minimum=0,
        maximum=1,
    )
    if not math.isclose(rate, wins / count, rel_tol=0.0, abs_tol=1e-15):
        raise runner.GaugeGate1Error("analysis subject win rate is inconsistent")
    row_deltas = [row["candidate_minus_primary_reference"] for row in rows]
    if nonnegative != sum(item >= 0.0 for item in row_deltas):
        raise runner.GaugeGate1Error("analysis nonnegative count is inconsistent")
    worst = runner._require_number(  # noqa: SLF001
        value["worst_dataset_delta"],
        path="analysis.worst_dataset_delta",
        minimum=-1,
        maximum=1,
    )
    if not math.isclose(worst, min(row_deltas), rel_tol=0.0, abs_tol=1e-15):
        raise runner.GaugeGate1Error("analysis worst delta is inconsistent")
    conditions = value["conditions"]
    expected_conditions = {
        "strictly_exceeds_primary_equal_dataset_balanced_accuracy": delta > 0.0,
        "minimum_nonnegative_dataset_deltas": nonnegative >= 3,
        "maximum_dataset_deficit": worst >= -0.03,
        "minimum_strict_subject_win_rate": wins >= 9,
    }
    if conditions != expected_conditions or value["passed"] is not all(
        expected_conditions.values()
    ):
        raise runner.GaugeGate1Error("analysis decision conditions are inconsistent")


def _coherent_inputs(
    plan: Mapping[str, Any],
) -> tuple[
    runner.MultiFileSnapshot,
    list[runner.PackageSnapshot],
    dict[str, float],
    dict[str, dict[str, float]],
]:
    _, reference, _ = runner._reference_snapshot(  # noqa: SLF001
        Path(plan["reference_identity"]["root"]),
        expected=plan["reference_identity"],
    )
    results: list[runner.PackageSnapshot] = []
    try:
        candidate: dict[str, float] = {}
        for job in runner._candidate_manifest():  # noqa: SLF001
            result = runner.load_result_snapshot(
                Path(plan["run_root"]), job, plan
            )
            results.append(result)
            record = runner._validate_result_package(  # noqa: SLF001
                result, plan=plan, job=job
            )
            candidate[job.job_id] = float(
                record["validation_metrics"]["balanced_accuracy"]
            )
        references = {
            model: {} for model in runner.EXPECTED_REFERENCE_MODELS
        }
        reference_jobs = [
            runner.ScreenJob(
                dataset=job.dataset,
                model=model,
                subject=job.subject,
                fold=job.fold,
                seed=job.seed,
            )
            for model in runner.EXPECTED_REFERENCE_MODELS
            for job in runner._candidate_manifest()  # noqa: SLF001
        ]
        for job, file_snapshot in zip(
            reference_jobs, reference.files[1:], strict=True
        ):
            record = runner._strict_json_bytes(  # noqa: SLF001
                file_snapshot.payload,
                source=str(file_snapshot.path),
            )
            references[job.model][
                runner.ScreenJob(
                    dataset=job.dataset,
                    model=runner.MODEL_NAME,
                    subject=job.subject,
                    fold=job.fold,
                    seed=job.seed,
                ).job_id
            ] = float(record["validation_metrics"]["balanced_accuracy"])
        reference.revalidate()
        for result in results:
            result.revalidate()
        return reference, results, candidate, references
    except Exception:
        for result in results:
            result.close()
        reference.close()
        raise


def analyze(*, project_root: Path, run_root: Path) -> tuple[dict[str, Any], bool]:
    plan, plan_snapshot = runner.load_plan_snapshot(run_root)
    try:
        if plan["project_root"] != str(runner._absolute(project_root)):  # noqa: SLF001
            raise runner.GaugeGate1Error("analysis project root differs from plan")
        destination = runner._absolute(  # noqa: SLF001
            Path(plan["run_root"])
            / runner.ANALYSIS_ROOT_DIRECTORY
            / ANALYSIS_DIRECTORY
        )
        source_identity, source = runner._source_snapshot(  # noqa: SLF001
            project_root, expected=plan["source_identity"]
        )
        del source_identity
        try:
            if runner.environment_identity() != plan["environment_identity"]:
                raise runner.GaugeGate1Error(
                    "analysis UV environment differs from plan"
                )
            reference, results, candidate, references = _coherent_inputs(plan)
            try:
                value = decision_from_balanced_accuracy(
                    candidate=candidate,
                    references=references,
                    plan_sha256=plan["plan_sha256"],
                    reference_snapshot_sha256=plan["reference_identity"][
                        "snapshot_sha256"
                    ],
                )
                if destination.exists():
                    existing_snapshot = runner._open_package(  # noqa: SLF001
                        destination,
                        expected_names=frozenset(
                            {"analysis.json", "analysis.sha256", "COMMITTED"}
                        ),
                        namespace_root=Path(plan["run_root"]),
                    )
                    try:
                        existing = _validate_analysis_package(
                            existing_snapshot,
                            plan=plan,
                        )
                        _assert_analysis_equivalent(existing, value)
                        plan_snapshot.revalidate()
                        source.revalidate()
                        reference.revalidate()
                        for result in results:
                            result.revalidate()
                        if (
                            runner.environment_identity()
                            != plan["environment_identity"]
                        ):
                            raise runner.GaugeGate1Error(
                                "analysis environment differs on resume"
                            )
                        existing_snapshot.revalidate()
                        return existing, False
                    finally:
                        existing_snapshot.close()
                members = _analysis_members(value)
                stage, stage_stat = runner._stage_package(  # noqa: SLF001
                    staging_parent=Path(plan["run_root"])
                    / runner.STAGING_DIRECTORY,
                    basename="analysis-gate1",
                    members=members,
                )
                with runner._authority_lock(  # noqa: SLF001
                    Path(plan["run_root"]),
                    expected=plan["authority_fence"],
                ):
                    plan_snapshot.revalidate()
                    source.revalidate()
                    reference.revalidate()
                    for result in results:
                        result.revalidate()
                    if destination.exists():
                        raced_snapshot = runner._open_package(  # noqa: SLF001
                            destination,
                            expected_names=frozenset(
                                {
                                    "analysis.json",
                                    "analysis.sha256",
                                    "COMMITTED",
                                }
                            ),
                            namespace_root=Path(plan["run_root"]),
                        )
                        try:
                            raced = _validate_analysis_package(
                                raced_snapshot,
                                plan=plan,
                            )
                            _assert_analysis_equivalent(raced, value)
                        finally:
                            raced_snapshot.close()
                        if runner._entry_matches(  # noqa: SLF001
                            stage, stage_stat, directory=True
                        ):
                            runner._hide_exact_entry(  # noqa: SLF001
                                stage,
                                stage_stat,
                                label="raced-analysis-stage",
                                directory=True,
                            )
                        return raced, False
                    published = runner._publish_package(  # noqa: SLF001
                        stage=stage,
                        stage_stat=stage_stat,
                        destination=destination,
                        expected_names=frozenset(
                            {"analysis.json", "analysis.sha256", "COMMITTED"}
                        ),
                        validator=lambda snapshot: _validate_analysis_package(
                            snapshot, plan=plan
                        ),
                        namespace_root=Path(plan["run_root"]),
                    )
                    try:
                        plan_snapshot.revalidate()
                        source.revalidate()
                        reference.revalidate()
                        for result in results:
                            result.revalidate()
                        if (
                            runner.environment_identity()
                            != plan["environment_identity"]
                        ):
                            raise runner.GaugeGate1Error(
                                "analysis environment changed across publication"
                            )
                        published.revalidate()
                    except Exception:
                        published.close()
                        if runner._entry_matches(  # noqa: SLF001
                            destination, stage_stat, directory=True
                        ):
                            runner._hide_exact_entry(  # noqa: SLF001
                                destination,
                                stage_stat,
                                label="invalid-analysis-rebind",
                                directory=True,
                            )
                        raise
                    published.close()
                return value, True
            finally:
                for result in results:
                    result.close()
                reference.close()
        finally:
            source.close()
    finally:
        plan_snapshot.close()


def _assert_analysis_equivalent(
    existing: Mapping[str, Any],
    recomputed: Mapping[str, Any],
) -> None:
    left = dict(existing)
    right = dict(recomputed)
    left.pop("created_at", None)
    right.pop("created_at", None)
    if left != right:
        raise runner.GaugeGate1Error(
            "persisted analysis differs from coherent recomputation"
        )


def _analysis_members(value: Mapping[str, Any]) -> dict[str, bytes]:
    payload = runner._canonical_bytes(value) + b"\n"  # noqa: SLF001
    return {
        "analysis.json": payload,
        "analysis.sha256": (
            runner._sha256_bytes(payload) + "\n"  # noqa: SLF001
        ).encode("ascii"),
        "COMMITTED": ANALYSIS_MARKER,
    }


def _validate_analysis_package(
    snapshot: runner.PackageSnapshot,
    *,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    payload = snapshot.children["analysis.json"].payload
    if (
        snapshot.children["analysis.sha256"].payload
        != (runner._sha256_bytes(payload) + "\n").encode("ascii")  # noqa: SLF001
        or snapshot.children["COMMITTED"].payload != ANALYSIS_MARKER
    ):
        raise runner.GaugeGate1Error("analysis package binding differs")
    value = runner._strict_json_bytes(  # noqa: SLF001
        payload, source=str(snapshot.path / "analysis.json")
    )
    validate_analysis(value)
    if (
        value["plan_sha256"] != plan["plan_sha256"]
        or value["reference_snapshot_sha256"]
        != plan["reference_identity"]["snapshot_sha256"]
    ):
        raise runner.GaugeGate1Error("analysis package differs from plan")
    snapshot.revalidate()
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    value, created = analyze(
        project_root=arguments.project_root,
        run_root=arguments.run_root,
    )
    print(
        json.dumps(
            {
                "created": created,
                "plan_sha256": value["plan_sha256"],
                "passed": value["passed"],
                "evaluation_scope": value["evaluation_scope"],
                "analysis": value,
            },
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ANALYSIS_SCHEMA",
    "analyze",
    "decision_from_balanced_accuracy",
    "main",
    "validate_analysis",
]
