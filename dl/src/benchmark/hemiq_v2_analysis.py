"""Aggregate-only analysis for the formal HemiQ harmonized-v2 track.

This module has no training entry point.  It acquires the runner's exclusive
publication fence, completes and persists a label-free exact-grid audit, then
opens held-out labels.  It publishes only per-subject/seed, per-subject, and
equal-participant aggregate metrics; no trial-level labels are written.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import stat
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import hemiq_v2_grid as grid


ANALYSIS_SCHEMA = "eeg-mi-hemiq-harmonized-v2-analysis-v1"
PUBLICATION_MANIFEST_SCHEMA = (
    "eeg-mi-hemiq-harmonized-v2-analysis-publication-manifest-v1"
)
METRIC_NAMES = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "cohen_kappa",
    "negative_log_likelihood",
    "brier",
    "expected_calibration_error",
)
AGGREGATION_DESCRIPTION = (
    "fold concatenation (one official fold), arithmetic seed mean "
    "within participant, then equal participant mean"
)
SUBJECT_SEED_COLUMNS = (
    "dataset",
    "subject",
    "fold",
    "seed",
    "n_trials",
    *METRIC_NAMES,
)
SUBJECT_COLUMNS = (
    "dataset",
    "subject",
    "n_seeds",
    "n_trials_per_seed",
    *METRIC_NAMES,
)
OUTPUT_FILENAMES = (
    "analysis.json",
    "audit.json",
    "report.md",
    "subject_metrics.csv",
    "subject_seed_metrics.csv",
)


class HemiQAnalysisError(RuntimeError):
    """Raised when labels or analysis publication cross the frozen boundary."""


@dataclass
class _PublicationSnapshot:
    destination: Path
    parent_descriptor: int
    parent_fingerprint: tuple[int, ...]
    directory_descriptor: int
    directory_fingerprint: tuple[int, ...]
    member_descriptors: dict[str, int]
    member_fingerprints: dict[str, tuple[int, ...]]

    def close(self) -> None:
        for descriptor in self.member_descriptors.values():
            os.close(descriptor)
        self.member_descriptors.clear()
        if self.directory_descriptor >= 0:
            os.close(self.directory_descriptor)
            self.directory_descriptor = -1
        if self.parent_descriptor >= 0:
            os.close(self.parent_descriptor)
            self.parent_descriptor = -1


def _finite_float(value: Any) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not math.isfinite(float(value))
    ):
        raise HemiQAnalysisError("analysis metric is not finite")
    return float(value)


def _exact_finite_float(value: Any) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise HemiQAnalysisError(
            "persisted analysis metric is not an exact finite float"
        )
    return value


def classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    ece_bins: int = 15,
) -> dict[str, float]:
    """Compute the frozen binary metric suite for one aggregate prediction set."""

    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        cohen_kappa_score,
        f1_score,
        log_loss,
    )

    y = np.asarray(labels)
    probability = np.asarray(probabilities, dtype=np.float64)
    if (
        y.dtype != np.int64
        or y.ndim != 1
        or set(y.tolist()) != {0, 1}
        or probability.shape != (len(y), 2)
        or not np.all(np.isfinite(probability))
        or np.any(probability < 0.0)
        or np.any(probability > 1.0)
        or not np.allclose(
            probability.sum(axis=1), 1.0, rtol=0.0, atol=1e-7
        )
        or type(ece_bins) is not int
        or ece_bins <= 1
    ):
        raise HemiQAnalysisError("binary metric input is invalid")
    predicted = (probability[:, 1] >= 0.5).astype(np.int64)
    confidence = np.max(probability, axis=1)
    correctness = (predicted == y).astype(np.float64)
    boundaries = np.linspace(0.0, 1.0, ece_bins + 1)
    ece = 0.0
    for index in range(ece_bins):
        lower = boundaries[index]
        upper = boundaries[index + 1]
        mask = (
            (confidence >= lower)
            & (
                confidence <= upper
                if index == ece_bins - 1
                else confidence < upper
            )
        )
        if np.any(mask):
            ece += float(mask.mean()) * abs(
                float(correctness[mask].mean())
                - float(confidence[mask].mean())
            )
    values = {
        "accuracy": accuracy_score(y, predicted),
        "balanced_accuracy": balanced_accuracy_score(y, predicted),
        "brier": np.mean(np.square(probability[:, 1] - y)),
        "cohen_kappa": cohen_kappa_score(y, predicted),
        "expected_calibration_error": ece,
        "macro_f1": f1_score(y, predicted, average="macro", zero_division=0),
        "negative_log_likelihood": log_loss(
            y, probability, labels=[0, 1]
        ),
    }
    return {name: _finite_float(values[name]) for name in METRIC_NAMES}


def _mean_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    if not rows:
        raise HemiQAnalysisError("cannot aggregate an empty metric table")
    return {
        name: _finite_float(
            np.mean([_finite_float(row[name]) for row in rows], dtype=np.float64)
        )
        for name in METRIC_NAMES
    }


def _validate_audit(audit: Any, *, plan: Mapping[str, Any]) -> None:
    grid._exact_keys(
        audit,
        {
            "audited_at",
            "completed_jobs",
            "evidence_scope",
            "plan_sha256",
            "preflight_gpu_uuids",
            "roster",
            "roster_sha256",
            "schema",
            "status",
        },
        path="audit",
    )
    preflight_uuids = audit["preflight_gpu_uuids"]
    roster = audit["roster"]
    if (
        audit["schema"] != grid.AUDIT_SCHEMA
        or audit["status"] != "pass"
        or audit["plan_sha256"] != plan["plan_sha256"]
        or not grid._is_exact_int(audit["completed_jobs"], minimum=1)
        or audit["completed_jobs"] != grid.EXPECTED_JOBS
        or audit["evidence_scope"] != plan["evidence_scope"]
        or not isinstance(audit["audited_at"], str)
        or not audit["audited_at"]
        or not isinstance(preflight_uuids, list)
        or not preflight_uuids
        or any(
            not isinstance(value, str)
            or grid.GPU_UUID_RE.fullmatch(value) is None
            for value in preflight_uuids
        )
        or preflight_uuids != sorted(set(preflight_uuids))
        or not isinstance(roster, list)
        or len(roster)
        != 1 + grid.EXPECTED_JOBS * len(grid.RESULT_FILENAMES) + len(
            preflight_uuids
        )
        or audit["roster_sha256"]
        != grid._sha256_bytes(grid._canonical_bytes(roster))
    ):
        raise HemiQAnalysisError("quiescent audit receipt is invalid")
    paths: list[str] = []
    for index, entry in enumerate(roster):
        grid._exact_keys(
            entry,
            {"relative_path", "sha256", "size_bytes"},
            path=f"audit.roster[{index}]",
        )
        relative = entry["relative_path"]
        if (
            not isinstance(relative, str)
            or not relative
            or relative.startswith("/")
            or ".." in Path(relative).parts
            or not isinstance(entry["sha256"], str)
            or grid.SHA256_RE.fullmatch(entry["sha256"]) is None
            or not grid._is_exact_int(entry["size_bytes"], minimum=1)
        ):
            raise HemiQAnalysisError("quiescent audit roster is invalid")
        paths.append(relative)
    if len(paths) != len(set(paths)):
        raise HemiQAnalysisError("quiescent audit roster contains duplicates")


def _audit_roster_lookup(
    audit: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    _validate_audit(audit, plan=plan)
    return {
        str(entry["relative_path"]): entry
        for entry in audit["roster"]
    }


def _require_metric_snapshot_in_audit(
    *,
    validated: Mapping[str, Any],
    audit_lookup: Mapping[str, Mapping[str, Any]],
    job: grid.Job,
) -> None:
    roster = validated.get("artifact_roster")
    if not isinstance(roster, list) or len(roster) != len(
        grid.RESULT_FILENAMES
    ):
        raise HemiQAnalysisError(
            f"descriptor snapshot roster is invalid for {job.job_id}"
        )
    expected_paths = {
        f"records/{job.job_id}/{name}"
        for name in grid.RESULT_FILENAMES
    }
    observed_paths: set[str] = set()
    for entry in roster:
        grid._exact_keys(
            entry,
            {"relative_path", "sha256", "size_bytes"},
            path=f"metric_snapshot.{job.job_id}",
        )
        relative = entry["relative_path"]
        if not isinstance(relative, str):
            raise HemiQAnalysisError("metric snapshot path is invalid")
        observed_paths.add(relative)
        expected = audit_lookup.get(relative)
        if expected is None or not grid._same_typed_value(entry, expected):
            raise HemiQAnalysisError(
                f"metric snapshot differs from persisted audit: {relative}"
            )
    if observed_paths != expected_paths:
        raise HemiQAnalysisError(
            f"metric snapshot roster is incomplete for {job.job_id}"
        )


def _verify_analysis_execution_identity(
    *,
    plan: Mapping[str, Any],
    project_root: Path,
) -> None:
    """Rebind analysis code, frozen config, and metric environment to the plan."""

    grid.verify_source_identity(plan, project_root=project_root)
    grid.verify_uv_identity(plan, project_root=project_root)
    observed_config = grid._config_identity(project_root)
    if not grid._same_typed_value(
        observed_config, plan["configuration_identity"]
    ):
        raise HemiQAnalysisError(
            "analysis HemiQ configuration differs from the plan"
        )


def _compute_after_audit(
    *,
    plan: Mapping[str, Any],
    project_root: Path,
    cache_root: Path,
    run_root: Path,
    audit: Mapping[str, Any],
    fence: grid.AnalysisFence | None = None,
) -> dict[str, Any]:
    """Open labels only after the caller has persisted a valid audit."""

    _validate_audit(audit, plan=plan)
    if fence is not None:
        if grid._absolute(run_root) != fence.run_root:
            raise HemiQAnalysisError(
                "metric root differs from the fenced run root"
            )
        grid.assert_analysis_fence(fence, plan=plan)
    audit_lookup = _audit_roster_lookup(audit, plan=plan)
    subject_seed_rows: list[dict[str, Any]] = []
    cache_by_subject: dict[int, tuple[dict[str, Any], np.ndarray]] = {}
    for subject in grid.SUBJECTS:
        if fence is not None:
            grid.assert_analysis_fence(fence, plan=plan)
        cache, (_, _, test_rows) = grid._verify_loaded_cache(
            plan,
            cache_root=cache_root,
            subject=subject,
            allow_test_class_validation=True,
        )
        # This is the first held-out label indexing operation in the formal
        # analysis path.  The persisted audit receipt must already exist.
        labels = np.ascontiguousarray(
            np.asarray(cache["y"], dtype=np.int64)[test_rows],
            dtype=np.int64,
        )
        cache_by_subject[subject] = (cache, labels)
        if fence is not None:
            grid.assert_analysis_fence(fence, plan=plan)

    for job in grid.expected_jobs():
        if fence is not None:
            grid.assert_analysis_fence(fence, plan=plan)
        validated = grid.validate_completion(
            plan=plan,
            project_root=project_root,
            run_root=run_root,
            job=job,
            fence=fence,
        )
        _require_metric_snapshot_in_audit(
            validated=validated,
            audit_lookup=audit_lookup,
            job=job,
        )
        cache, labels = cache_by_subject[job.subject]
        expected_rows = grid._bnci004_rows(
            cache, validate_held_out_classes=True
        )[2]
        observed_rows = np.asarray(validated["test_rows"], dtype=np.int64)
        if not np.array_equal(observed_rows, expected_rows):
            raise HemiQAnalysisError(
                f"prediction rows differ for {job.job_id}"
            )
        metrics = classification_metrics(
            labels, np.asarray(validated["probabilities"], dtype=np.float64)
        )
        subject_seed_rows.append(
            {
                "dataset": grid.DATASET,
                "fold": job.fold,
                "n_trials": len(labels),
                "seed": job.seed,
                "subject": job.subject,
                **metrics,
            }
        )
        if fence is not None:
            grid.assert_analysis_fence(fence, plan=plan)

    if len(subject_seed_rows) != grid.EXPECTED_JOBS:
        raise HemiQAnalysisError("analysis subject/seed table is incomplete")
    subject_rows: list[dict[str, Any]] = []
    for subject in grid.SUBJECTS:
        rows = [
            row for row in subject_seed_rows if row["subject"] == subject
        ]
        if (
            len(rows) != len(grid.SEEDS)
            or [row["seed"] for row in rows] != list(grid.SEEDS)
            or len({row["n_trials"] for row in rows}) != 1
        ):
            raise HemiQAnalysisError(
                f"subject {subject} seed aggregation is incomplete"
            )
        subject_rows.append(
            {
                "dataset": grid.DATASET,
                "n_seeds": len(grid.SEEDS),
                "n_trials_per_seed": rows[0]["n_trials"],
                "subject": subject,
                **_mean_metrics(rows),
            }
        )
    dataset_summary = {
        "aggregation": AGGREGATION_DESCRIPTION,
        "dataset": grid.DATASET,
        "n_jobs": grid.EXPECTED_JOBS,
        "n_seeds": len(grid.SEEDS),
        "n_subjects": len(grid.SUBJECTS),
        **_mean_metrics(subject_rows),
    }
    result = {
        "audit_roster_sha256": audit["roster_sha256"],
        "dataset_summary": dataset_summary,
        "evidence_scope": plan["evidence_scope"],
        "metric_contract": {
            "class_order": [0, 1],
            "ece_bins": 15,
            "metrics": list(METRIC_NAMES),
            "prediction_threshold": 0.5,
        },
        "model_id": grid.MODEL_ID,
        "plan_sha256": plan["plan_sha256"],
        "schema": ANALYSIS_SCHEMA,
        "subject_metrics": subject_rows,
        "subject_seed_metrics": subject_seed_rows,
        "track": grid.TRACK,
    }
    _validate_analysis(result, plan=plan, audit=audit)
    return result


def _validate_metric_row(
    row: Any,
    *,
    expected_keys: Sequence[str],
    path: str,
) -> None:
    grid._exact_keys(row, set(expected_keys), path=path)
    for metric in METRIC_NAMES:
        observed = _exact_finite_float(row[metric])
        if metric in {
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "brier",
            "expected_calibration_error",
        } and not 0.0 <= observed <= 1.0:
            raise HemiQAnalysisError(
                f"{path}.{metric} lies outside [0, 1]"
            )
        if metric == "cohen_kappa" and not -1.0 <= observed <= 1.0:
            raise HemiQAnalysisError(
                f"{path}.{metric} lies outside [-1, 1]"
            )
        if metric == "negative_log_likelihood" and observed < 0.0:
            raise HemiQAnalysisError(
                f"{path}.{metric} is negative"
            )
    if (
        row["dataset"] != grid.DATASET
        or not grid._is_exact_int(row["subject"], minimum=1)
    ):
        raise HemiQAnalysisError(f"{path} identity is invalid")


def _validate_analysis(
    value: Any,
    *,
    plan: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> None:
    grid.validate_plan(plan)
    _validate_audit(audit, plan=plan)
    grid._exact_keys(
        value,
        {
            "audit_roster_sha256",
            "dataset_summary",
            "evidence_scope",
            "metric_contract",
            "model_id",
            "plan_sha256",
            "schema",
            "subject_metrics",
            "subject_seed_metrics",
            "track",
        },
        path="analysis",
    )
    if (
        value["schema"] != ANALYSIS_SCHEMA
        or value["plan_sha256"] != plan["plan_sha256"]
        or value["audit_roster_sha256"] != audit["roster_sha256"]
        or value["evidence_scope"] != plan["evidence_scope"]
        or value["model_id"] != grid.MODEL_ID
        or value["track"] != grid.TRACK
        or not grid._same_typed_value(
            value["metric_contract"],
            {
                "class_order": [0, 1],
                "ece_bins": 15,
                "metrics": list(METRIC_NAMES),
                "prediction_threshold": 0.5,
            },
        )
    ):
        raise HemiQAnalysisError("analysis identity is invalid")
    seed_rows = value["subject_seed_metrics"]
    subject_rows = value["subject_metrics"]
    if (
        not isinstance(seed_rows, list)
        or len(seed_rows) != grid.EXPECTED_JOBS
        or not isinstance(subject_rows, list)
        or len(subject_rows) != len(grid.SUBJECTS)
    ):
        raise HemiQAnalysisError("analysis table cardinality is invalid")
    for index, row in enumerate(seed_rows):
        _validate_metric_row(
            row,
            expected_keys=SUBJECT_SEED_COLUMNS,
            path=f"analysis.subject_seed_metrics[{index}]",
        )
        expected = grid.expected_jobs()[index]
        split_key = (
            f"{grid.DATASET}:s{expected.subject:03d}:f{expected.fold:02d}"
        )
        expected_trials = plan["split_identity"][split_key][
            "partitions"
        ]["test"]["shape"][0]
        if (
            row["subject"] != expected.subject
            or not grid._is_exact_int(row["fold"], minimum=0)
            or row["fold"] != expected.fold
            or not grid._is_exact_int(row["seed"], minimum=0)
            or row["seed"] != expected.seed
            or not grid._is_exact_int(row["n_trials"], minimum=1)
            or row["n_trials"] != expected_trials
        ):
            raise HemiQAnalysisError("subject/seed row ordering is invalid")
    for index, row in enumerate(subject_rows):
        _validate_metric_row(
            row,
            expected_keys=SUBJECT_COLUMNS,
            path=f"analysis.subject_metrics[{index}]",
        )
        subject_seed_rows = [
            seed_row
            for seed_row in seed_rows
            if seed_row["subject"] == grid.SUBJECTS[index]
        ]
        expected_trials = plan["split_identity"][
            f"{grid.DATASET}:s{grid.SUBJECTS[index]:03d}:f00"
        ]["partitions"]["test"]["shape"][0]
        if (
            row["subject"] != grid.SUBJECTS[index]
            or not grid._is_exact_int(row["n_seeds"], minimum=1)
            or row["n_seeds"] != len(grid.SEEDS)
            or not grid._is_exact_int(row["n_trials_per_seed"], minimum=1)
            or row["n_trials_per_seed"] != expected_trials
            or len(subject_seed_rows) != len(grid.SEEDS)
        ):
            raise HemiQAnalysisError("subject row ordering is invalid")
        recomputed = _mean_metrics(subject_seed_rows)
        for metric in METRIC_NAMES:
            if row[metric] != recomputed[metric]:
                raise HemiQAnalysisError(
                    f"subject aggregate differs for {row['subject']}:{metric}"
                )
    summary = value["dataset_summary"]
    grid._exact_keys(
        summary,
        {
            "aggregation",
            "dataset",
            "n_jobs",
            "n_seeds",
            "n_subjects",
            *METRIC_NAMES,
        },
        path="analysis.dataset_summary",
    )
    if (
        summary["dataset"] != grid.DATASET
        or not grid._is_exact_int(summary["n_jobs"], minimum=1)
        or summary["n_jobs"] != grid.EXPECTED_JOBS
        or not grid._is_exact_int(summary["n_seeds"], minimum=1)
        or summary["n_seeds"] != len(grid.SEEDS)
        or not grid._is_exact_int(summary["n_subjects"], minimum=1)
        or summary["n_subjects"] != len(grid.SUBJECTS)
        or summary["aggregation"] != AGGREGATION_DESCRIPTION
    ):
        raise HemiQAnalysisError("dataset summary identity is invalid")
    summary_row = {
        "dataset": grid.DATASET,
        "subject": grid.SUBJECTS[0],
        **{metric: summary[metric] for metric in METRIC_NAMES},
    }
    _validate_metric_row(
        summary_row,
        expected_keys=("dataset", "subject", *METRIC_NAMES),
        path="analysis.dataset_summary",
    )
    recomputed_summary = _mean_metrics(subject_rows)
    for metric in METRIC_NAMES:
        if summary[metric] != recomputed_summary[metric]:
            raise HemiQAnalysisError(
                f"equal-participant dataset aggregate differs: {metric}"
            )


def _csv_bytes(
    rows: Sequence[Mapping[str, Any]], columns: Sequence[str]
) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(columns),
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        if set(row) != set(columns):
            raise HemiQAnalysisError("CSV row schema differs")
        writer.writerow({name: row[name] for name in columns})
    return buffer.getvalue().encode("utf-8")


def _report_markdown(analysis: Mapping[str, Any]) -> bytes:
    summary = analysis["dataset_summary"]
    lines = [
        "# HemiQ-Field harmonized-v2 development analysis",
        "",
        "This is a separate development-only procedure result. It is not the "
        "historical HemiQ confirmation, a clinical result, or an SOTA claim.",
        "",
        "## Frozen scope",
        "",
        f"- Dataset: `{grid.DATASET}`",
        f"- Subjects: {len(grid.SUBJECTS)}",
        f"- Seeds per subject: {len(grid.SEEDS)}",
        f"- Atomic score-blind jobs: {grid.EXPECTED_JOBS}",
        "- Inputs: supplied bipolar C3/Cz/C4 derivations only",
        "- Aggregation: fold concatenation, seed mean within participant, "
        "equal participant mean",
        "",
        "## Equal-participant mean",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for name in METRIC_NAMES:
        lines.append(f"| {name.replace('_', ' ')} | {summary[name]:.6f} |")
    lines.extend(
        [
            "",
            "The label-free quiescent audit and exact artifact roster seal are "
            "embedded in this publication.",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def _stage_analysis_payloads(
    *,
    stage_path: Path,
    analysis: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> tuple[dict[str, Any], int, int]:
    stage_descriptor = grid._open_directory_absolute(stage_path)
    try:
        payloads = {
            "analysis.json": grid._canonical_bytes(dict(analysis)),
            "audit.json": grid._canonical_bytes(dict(audit)),
            "report.md": _report_markdown(analysis),
            "subject_metrics.csv": _csv_bytes(
                analysis["subject_metrics"], SUBJECT_COLUMNS
            ),
            "subject_seed_metrics.csv": _csv_bytes(
                analysis["subject_seed_metrics"], SUBJECT_SEED_COLUMNS
            ),
        }
        for name in OUTPUT_FILENAMES:
            if name == "audit.json" and grid._path_exists(stage_path / name):
                observed = grid._read_unique_regular_bytes(stage_path / name)
                if observed != payloads[name]:
                    raise HemiQAnalysisError(
                        "persisted pre-label audit differs before sealing"
                    )
                status = grid._anchored_lstat(stage_path / name)
                if status.st_mode & 0o222:
                    raise HemiQAnalysisError(
                        "persisted pre-label audit is writable"
                    )
            else:
                grid._write_exclusive_at(
                    stage_descriptor, name, payloads[name], mode=0o400
                )
        artifact_manifest = {
            name: {
                "sha256": grid._sha256_bytes(payloads[name]),
                "size_bytes": len(payloads[name]),
            }
            for name in OUTPUT_FILENAMES
        }
        manifest = {
            "analysis_sha256": artifact_manifest["analysis.json"]["sha256"],
            "artifacts": artifact_manifest,
            "audit_roster_sha256": audit["roster_sha256"],
            "plan_sha256": analysis["plan_sha256"],
            "schema": PUBLICATION_MANIFEST_SCHEMA,
            "seal_sha256": grid._sha256_bytes(
                grid._canonical_bytes(artifact_manifest)
            ),
        }
        grid._write_exclusive_at(
            stage_descriptor,
            "manifest.json",
            grid._canonical_bytes(manifest),
            mode=0o400,
        )
        os.fsync(stage_descriptor)
        status = os.fstat(stage_descriptor)
        os.fchmod(stage_descriptor, 0o555)
        os.fsync(stage_descriptor)
        readonly = os.fstat(stage_descriptor)
        if (
            (readonly.st_dev, readonly.st_ino)
            != (status.st_dev, status.st_ino)
            or (readonly.st_mode & 0o777) != 0o555
        ):
            raise HemiQAnalysisError(
                "analysis stage did not become descriptor-bound read-only"
            )
        return manifest, int(readonly.st_dev), int(readonly.st_ino)
    finally:
        os.close(stage_descriptor)


def _assert_publication_snapshot(
    snapshot: _PublicationSnapshot,
    *,
    expected_dev: int,
    expected_ino: int,
) -> None:
    """Rebind every held descriptor to the exact canonical publication."""

    if snapshot.parent_descriptor < 0 or snapshot.directory_descriptor < 0:
        raise HemiQAnalysisError("published analysis snapshot is closed")
    parent_after = os.fstat(snapshot.parent_descriptor)
    directory_after = os.fstat(snapshot.directory_descriptor)
    try:
        path_after = os.stat(
            snapshot.destination.name,
            dir_fd=snapshot.parent_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError as error:
        raise HemiQAnalysisError(
            "published analysis canonical name disappeared"
        ) from error
    if (
        grid._stat_fingerprint(parent_after)
        != snapshot.parent_fingerprint
        or grid._stat_fingerprint(directory_after)
        != snapshot.directory_fingerprint
        or grid._stat_fingerprint(path_after)
        != snapshot.directory_fingerprint
        or (int(directory_after.st_dev), int(directory_after.st_ino))
        != (expected_dev, expected_ino)
        or not stat.S_ISDIR(directory_after.st_mode)
        or stat.S_IMODE(directory_after.st_mode) != 0o555
    ):
        raise HemiQAnalysisError(
            "published analysis directory authority changed"
        )
    expected_names = sorted((*OUTPUT_FILENAMES, "manifest.json"))
    if sorted(os.listdir(snapshot.directory_descriptor)) != expected_names:
        raise HemiQAnalysisError("published analysis roster is not exact")
    for name in expected_names:
        descriptor = snapshot.member_descriptors.get(name, -1)
        fingerprint = snapshot.member_fingerprints.get(name)
        if descriptor < 0 or fingerprint is None:
            raise HemiQAnalysisError(
                "published analysis member snapshot is incomplete"
            )
        held = os.fstat(descriptor)
        try:
            named = os.stat(
                name,
                dir_fd=snapshot.directory_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError as error:
            raise HemiQAnalysisError(
                f"published analysis member disappeared: {name}"
            ) from error
        if (
            grid._stat_fingerprint(held) != fingerprint
            or grid._stat_fingerprint(named) != fingerprint
            or not stat.S_ISREG(held.st_mode)
            or held.st_nlink != 1
            or stat.S_IMODE(held.st_mode) != 0o400
        ):
            raise HemiQAnalysisError(
                f"published analysis member authority changed: {name}"
            )


def _open_publication_snapshot(
    destination: Path,
    *,
    expected_dev: int,
    expected_ino: int,
) -> _PublicationSnapshot:
    """Open the directory and every member once, then retain all descriptors."""

    destination = grid._absolute(destination)
    parent_descriptor = grid._open_directory_absolute(destination.parent)
    directory_descriptor = -1
    member_descriptors: dict[str, int] = {}
    try:
        parent_fingerprint = grid._stat_fingerprint(
            os.fstat(parent_descriptor)
        )
        path_status = os.stat(
            destination.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        directory_descriptor = os.open(
            destination.name,
            grid._directory_flags(),
            dir_fd=parent_descriptor,
        )
        directory_status = os.fstat(directory_descriptor)
        directory_fingerprint = grid._stat_fingerprint(directory_status)
        if (
            directory_fingerprint != grid._stat_fingerprint(path_status)
            or (int(directory_status.st_dev), int(directory_status.st_ino))
            != (expected_dev, expected_ino)
            or not stat.S_ISDIR(directory_status.st_mode)
            or stat.S_IMODE(directory_status.st_mode) != 0o555
        ):
            raise HemiQAnalysisError(
                "published analysis directory is writable or replaced"
            )
        expected_names = sorted((*OUTPUT_FILENAMES, "manifest.json"))
        if sorted(os.listdir(directory_descriptor)) != expected_names:
            raise HemiQAnalysisError(
                "published analysis roster is not exact"
            )
        member_fingerprints: dict[str, tuple[int, ...]] = {}
        for name in expected_names:
            descriptor = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0),
                dir_fd=directory_descriptor,
            )
            member_descriptors[name] = descriptor
            opened = os.fstat(descriptor)
            named = os.stat(
                name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            grid._assert_unique_regular(
                opened,
                source=str(destination / name),
            )
            fingerprint = grid._stat_fingerprint(opened)
            if (
                grid._stat_fingerprint(named) != fingerprint
                or stat.S_IMODE(opened.st_mode) != 0o400
            ):
                raise HemiQAnalysisError(
                    f"published analysis member is detached: {name}"
                )
            member_fingerprints[name] = fingerprint
        snapshot = _PublicationSnapshot(
            destination=destination,
            parent_descriptor=parent_descriptor,
            parent_fingerprint=parent_fingerprint,
            directory_descriptor=directory_descriptor,
            directory_fingerprint=directory_fingerprint,
            member_descriptors=member_descriptors,
            member_fingerprints=member_fingerprints,
        )
        _assert_publication_snapshot(
            snapshot,
            expected_dev=expected_dev,
            expected_ino=expected_ino,
        )
        return snapshot
    except Exception:
        for descriptor in member_descriptors.values():
            os.close(descriptor)
        if directory_descriptor >= 0:
            os.close(directory_descriptor)
        os.close(parent_descriptor)
        raise


def _verify_publication(
    destination: Path,
    *,
    manifest: Mapping[str, Any],
    expected_dev: int,
    expected_ino: int,
    snapshot: _PublicationSnapshot | None = None,
) -> None:
    """Verify one descriptor-held package, rebinding after every member read."""

    owned_snapshot = snapshot is None
    if snapshot is None:
        snapshot = _open_publication_snapshot(
            destination,
            expected_dev=expected_dev,
            expected_ino=expected_ino,
        )
    elif snapshot.destination != grid._absolute(destination):
        raise HemiQAnalysisError(
            "published analysis snapshot names a different destination"
        )
    try:
        _assert_publication_snapshot(
            snapshot,
            expected_dev=expected_dev,
            expected_ino=expected_ino,
        )
        payloads: dict[str, bytes] = {}
        for name in sorted((*OUTPUT_FILENAMES, "manifest.json")):
            payload = grid._read_descriptor_bytes(
                snapshot.member_descriptors[name],
                source=str(snapshot.destination / name),
            )
            expected_size = snapshot.member_fingerprints[name][
                grid.STAT_FINGERPRINT_FIELDS.index("st_size")
            ]
            if len(payload) != expected_size:
                raise HemiQAnalysisError(
                    f"published analysis member read was incomplete: {name}"
                )
            payloads[name] = payload
            _assert_publication_snapshot(
                snapshot,
                expected_dev=expected_dev,
                expected_ino=expected_ino,
            )
        observed_manifest = grid._strict_json_bytes(
            payloads["manifest.json"],
            source=str(snapshot.destination / "manifest.json"),
        )
        if observed_manifest != dict(manifest):
            raise HemiQAnalysisError("published analysis manifest changed")
        if set(manifest.get("artifacts", {})) != set(OUTPUT_FILENAMES):
            raise HemiQAnalysisError(
                "published analysis artifact manifest is not exact"
            )
        for name, identity in manifest["artifacts"].items():
            payload = payloads[name]
            if (
                len(payload) != identity["size_bytes"]
                or grid._sha256_bytes(payload) != identity["sha256"]
            ):
                raise HemiQAnalysisError(
                    f"published analysis artifact changed: {name}"
                )
        _assert_publication_snapshot(
            snapshot,
            expected_dev=expected_dev,
            expected_ino=expected_ino,
        )
    finally:
        if owned_snapshot:
            snapshot.close()


def _hide_failed_publication(
    destination: Path,
    *,
    expected_dev: int,
    expected_ino: int,
) -> Path | None:
    """Remove a post-rename failure from the visible destination atomically."""

    parent = grid._open_directory_absolute(destination.parent)
    try:
        try:
            observed = os.stat(
                destination.name,
                dir_fd=parent,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return None
        exact = (int(observed.st_dev), int(observed.st_ino)) == (
            expected_dev,
            expected_ino,
        )
        if not exact:
            raise HemiQAnalysisError(
                "published analysis inode was replaced before quarantine"
            )
        hidden_name = (
            f".{destination.name}.failed-{uuid.uuid4().hex}"
        )
        grid._rename_noreplace_at(
            parent,
            destination.name,
            parent,
            hidden_name,
        )
        hidden = os.stat(
            hidden_name,
            dir_fd=parent,
            follow_symlinks=False,
        )
        if (
            int(hidden.st_dev),
            int(hidden.st_ino),
        ) != (int(observed.st_dev), int(observed.st_ino)):
            raise HemiQAnalysisError(
                "failed analysis quarantine changed inode"
            )
        os.fsync(parent)
        return destination.parent / hidden_name
    finally:
        os.close(parent)


def analyze_and_publish(
    *,
    project_root: Path,
    cache_root: Path,
    run_root: Path,
    destination: Path,
) -> dict[str, Any]:
    """Audit, analyze, seal, and publish one fresh aggregate-only directory."""

    plan = grid.load_plan(run_root)
    destination = grid._absolute(destination)
    run = grid._absolute(run_root)
    try:
        destination.relative_to(run)
    except ValueError:
        pass
    else:
        raise HemiQAnalysisError(
            "analysis destination must be outside the immutable run root"
        )
    if grid._path_exists(destination):
        raise FileExistsError(str(destination))
    parent = grid._safe_mkdir(destination.parent)
    stage_name = f".{destination.name}.stage-{uuid.uuid4().hex}"
    stage_path = parent / stage_name
    parent_descriptor = grid._open_directory_absolute(parent)
    staged_identity: tuple[int, int] | None = None
    try:
        os.mkdir(stage_name, mode=0o700, dir_fd=parent_descriptor)
        staged_status = os.stat(
            stage_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        staged_identity = (
            int(staged_status.st_dev),
            int(staged_status.st_ino),
        )
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)

    published_identity: tuple[int, int] | None = None
    publication_snapshot: _PublicationSnapshot | None = None
    publication_result: dict[str, Any] | None = None
    try:
        with grid.analysis_fence(run_root=run_root, plan=plan) as fence:
            _verify_analysis_execution_identity(
                plan=plan,
                project_root=project_root,
            )
            audit = grid.audit_grid(
                plan=plan,
                project_root=project_root,
                run_root=run_root,
                fence=fence,
            )
            _validate_audit(audit, plan=plan)
            # Persist and fsync the exact label-free audit before any cache
            # label is indexed by _compute_after_audit.
            stage_descriptor = grid._open_directory_absolute(stage_path)
            try:
                grid._write_exclusive_at(
                    stage_descriptor,
                    "audit.json",
                    grid._canonical_bytes(audit),
                    mode=0o400,
                )
                os.fsync(stage_descriptor)
            finally:
                os.close(stage_descriptor)
            analysis = _compute_after_audit(
                plan=plan,
                project_root=project_root,
                cache_root=cache_root,
                run_root=run_root,
                audit=audit,
                fence=fence,
            )
            audit_payload = grid._read_unique_regular_bytes(
                stage_path / "audit.json"
            )
            if audit_payload != grid._canonical_bytes(audit):
                raise HemiQAnalysisError("persisted pre-label audit changed")
            manifest, stage_dev, stage_ino = _stage_analysis_payloads(
                stage_path=stage_path,
                analysis=analysis,
                audit=audit,
            )
            if staged_identity != (stage_dev, stage_ino):
                raise HemiQAnalysisError(
                    "analysis stage inode changed before publication"
                )
            grid.assert_analysis_fence(fence, plan=plan)
            repeated = grid.audit_grid(
                plan=plan,
                project_root=project_root,
                run_root=run_root,
                fence=fence,
            )
            if repeated["roster_sha256"] != audit["roster_sha256"]:
                raise HemiQAnalysisError(
                    "score-blind roster changed before publication"
                )
            _verify_analysis_execution_identity(
                plan=plan,
                project_root=project_root,
            )
            stage_parent = grid._open_directory_absolute(parent)
            destination_parent = grid._open_directory_absolute(destination.parent)
            try:
                stage_parent_status = os.fstat(stage_parent)
                destination_parent_status = os.fstat(destination_parent)
                if (
                    stage_parent_status.st_dev,
                    stage_parent_status.st_ino,
                ) != (
                    destination_parent_status.st_dev,
                    destination_parent_status.st_ino,
                ):
                    raise HemiQAnalysisError(
                        "analysis publication must use one same-parent rename"
                    )
                grid._rename_noreplace_at(
                    stage_parent,
                    stage_name,
                    destination_parent,
                    destination.name,
                )
                # This assignment is deliberately the first operation after a
                # successful rename.  Every subsequent failure, including
                # parent fsync and context-manager release, quarantines this
                # exact inode and leaves no visible destination.
                published_identity = (stage_dev, stage_ino)
                os.fsync(stage_parent)
                os.fsync(destination_parent)
            finally:
                os.close(stage_parent)
                os.close(destination_parent)
            publication_snapshot = _open_publication_snapshot(
                destination,
                expected_dev=stage_dev,
                expected_ino=stage_ino,
            )
            _verify_publication(
                destination,
                manifest=manifest,
                expected_dev=stage_dev,
                expected_ino=stage_ino,
                snapshot=publication_snapshot,
            )
            grid.assert_analysis_fence(fence, plan=plan)
            final_audit = grid.audit_grid(
                plan=plan,
                project_root=project_root,
                run_root=run_root,
                fence=fence,
            )
            if final_audit["roster_sha256"] != audit["roster_sha256"]:
                raise HemiQAnalysisError(
                    "run authority changed after analysis publication"
                )
            publication_result = {
                "analysis_sha256": manifest["analysis_sha256"],
                "audit_roster_sha256": audit["roster_sha256"],
                "destination": str(destination),
                "n_jobs": grid.EXPECTED_JOBS,
                "status": "published",
            }
        if (
            publication_result is None
            or publication_snapshot is None
            or published_identity is None
        ):
            raise HemiQAnalysisError(
                "analysis publication produced no retained snapshot"
            )
        # The same directory and member descriptors remain open across fence
        # release. Re-read and rebind the whole package as the final operation
        # before returning authority to the caller.
        _verify_publication(
            destination,
            manifest=manifest,
            expected_dev=published_identity[0],
            expected_ino=published_identity[1],
            snapshot=publication_snapshot,
        )
        return publication_result
    except Exception:
        if published_identity is not None:
            _hide_failed_publication(
                destination,
                expected_dev=published_identity[0],
                expected_ino=published_identity[1],
            )
        elif grid._path_exists(destination):
            # A no-replace helper can be fault-injected to complete the move
            # and then raise before control reaches the post-rename
            # assignment. Bind the visible name to the known stage inode
            # before hiding it; a replacement authority is contention and is
            # never moved.
            destination_status = grid._anchored_lstat(destination)
            destination_identity = (
                int(destination_status.st_dev),
                int(destination_status.st_ino),
            )
            if (
                staged_identity is None
                or destination_identity != staged_identity
            ):
                raise HemiQAnalysisError(
                    "analysis destination was replaced during failed rename"
                )
            _hide_failed_publication(
                destination,
                expected_dev=destination_identity[0],
                expected_ino=destination_identity[1],
            )
        elif grid._path_exists(stage_path):
            candidate = stage_path
            quarantine = parent / (
                f".{destination.name}.failed-{uuid.uuid4().hex}"
            )
            source_parent = grid._open_directory_absolute(candidate.parent)
            quarantine_parent = grid._open_directory_absolute(quarantine.parent)
            try:
                candidate_status = os.stat(
                    candidate.name,
                    dir_fd=source_parent,
                    follow_symlinks=False,
                )
                if staged_identity is None or (
                    int(candidate_status.st_dev),
                    int(candidate_status.st_ino),
                ) != staged_identity:
                    raise HemiQAnalysisError(
                        "analysis stage was replaced before cleanup"
                    )
                grid._rename_noreplace_at(
                    source_parent,
                    candidate.name,
                    quarantine_parent,
                    quarantine.name,
                )
                os.fsync(source_parent)
                os.fsync(quarantine_parent)
            finally:
                os.close(source_parent)
                os.close(quarantine_parent)
        raise
    finally:
        if publication_snapshot is not None:
            publication_snapshot.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = analyze_and_publish(
        project_root=arguments.project_root,
        cache_root=arguments.cache_root,
        run_root=arguments.run_root,
        destination=arguments.destination,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
