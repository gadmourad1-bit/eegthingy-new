from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from benchmark.development_screen import (
    CANDIDATE_MODELS,
    CURATED_SPLITS,
    EVALUATION_SCOPE,
    PLAN_SCHEMA,
    RECORD_SCHEMA,
    WORKER_COUNT,
    ScreenJob,
    _canonical_cuda_uuid,
    assemble_manifest,
    assert_validation_scope,
    curated_jobs,
    jobs_for_worker,
    run_job_atomically,
    validate_record_payload,
)


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class _CudaUuidLike:
    def __str__(self) -> str:
        return "576ef18e-b48f-c787-e91d-f93658b27b8e"


def _manifest() -> dict[str, object]:
    cache_identity: dict[str, object] = {}
    split_identity: dict[str, object] = {}
    for dataset, subjects in CURATED_SPLITS:
        for subject in subjects:
            subject_key = f"{dataset}:s{subject:03d}"
            split_key = f"{dataset}:s{subject:03d}:f00"
            cache_identity[subject_key] = {
                "relative_path": f"{dataset}/subject_{subject:03d}.npz",
                "file_size_bytes": 123,
                "file_sha256": _digest(f"file-{subject_key}"),
                "array_sha256": _digest(f"array-{subject_key}"),
            }
            split_identity[split_key] = {
                "train": {
                    "count": 20,
                    "rows_sha256": _digest(f"train-{split_key}"),
                },
                "validation": {
                    "count": 10,
                    "rows_sha256": _digest(f"validation-{split_key}"),
                },
            }
    return assemble_manifest(
        cache_root="/synthetic/cache",
        cache_identity=cache_identity,
        split_identity=split_identity,
        source_hashes={"src/benchmark/development_screen.py": _digest("source")},
        environment_identity={"python": "synthetic"},
    )


def _record(job: ScreenJob, plan_sha256: str) -> dict[str, object]:
    direct = job.model == "chsdnet_direct"
    return {
        "schema": RECORD_SCHEMA,
        "created_at": "2026-07-29T00:00:00+00:00",
        "plan_sha256": plan_sha256,
        "job_id": job.job_id,
        "job": job.identity(),
        "evaluation_scope": EVALUATION_SCOPE,
        "confirmation_evidence": False,
        "cache_identity": {
            "array_sha256": _digest("array"),
            "file_sha256": _digest("file"),
        },
        "source_identity": {"runner.py": _digest("runner")},
        "split": {
            "train": {"count": 20, "rows_sha256": _digest("train")},
            "validation": {
                "count": 10,
                "rows_sha256": _digest("validation"),
            },
        },
        "scaler": {
            "fit_scope": "training_only",
            "mean_sha256": _digest("mean"),
            "std_sha256": _digest("std"),
            "contract": {"fit_rows": "training_only"},
        },
        "model": {
            "identity": {"requested_name": job.model},
            "parameter_count": 123,
            "constructed_state_sha256": _digest("constructed"),
            "initial_state_sha256": _digest("initial"),
            "selected_state_sha256": _digest("selected"),
            "source_statistics": {
                "available": direct,
                "ran": direct,
                "fit_scope": "training_rows_only",
                "outcome_aware": False,
            },
        },
        "fit": {
            "best_epoch": 3,
            "epochs_run": 8,
            "best_validation_loss": 0.42,
            "early_stopping": True,
            "config": {"seed": 7},
        },
        "validation_metrics": {
            "accuracy": 0.8,
            "balanced_accuracy": 0.79,
            "cohen_kappa": 0.6,
            "negative_log_likelihood": 0.42,
            "roc_auc": 0.85,
        },
        "timing": {
            "construction_seconds": 0.1,
            "fit_seconds": 1.0,
            "evaluation_seconds": 0.1,
            "total_seconds": 1.2,
        },
        "resource_preflight": {
            "gpu": {"gpu_uuid": "GPU-synthetic"},
            "disk": {"free_gib": 100.0},
        },
    }


def test_manifest_is_exact_fixed_136_job_cartesian_screen() -> None:
    manifest = _manifest()
    jobs = curated_jobs()
    assert manifest["schema"] == PLAN_SCHEMA
    assert manifest["evaluation_scope"] == EVALUATION_SCOPE
    assert manifest["confirmation_evidence"] is False
    assert manifest["n_jobs"] == 136
    assert len(jobs) == 136
    assert {job.model for job in jobs} == set(CANDIDATE_MODELS)
    assert {job.seed for job in jobs} == {7}
    assert {job.fold for job in jobs} == {0}
    assert len({job.job_id for job in jobs}) == 136
    factories = {
        item["name"]: item["factory"]  # type: ignore[index]
        for item in manifest["models"]  # type: ignore[union-attr]
    }
    assert factories["chsdnet_direct"] == (
        "benchmark.chsd_direct:make_chsd_direct_model"
    )
    assert factories["chsdnet_joint"] == (
        "benchmark.chsd_joint:make_chsd_joint_model"
    )
    assert_validation_scope(manifest)


def test_static_four_worker_partitions_are_disjoint_and_complete() -> None:
    jobs = curated_jobs()
    partitions = [
        jobs_for_worker(jobs, worker_index)
        for worker_index in range(WORKER_COUNT)
    ]
    job_id_sets = [{job.job_id for job in partition} for partition in partitions]
    for left_index, left in enumerate(job_id_sets):
        for right in job_id_sets[left_index + 1 :]:
            assert left.isdisjoint(right)
    assert set().union(*job_id_sets) == {job.job_id for job in jobs}
    assert [len(partition) for partition in partitions] == [34, 34, 34, 34]


def test_pytorch_non_string_cuda_uuid_is_canonicalized() -> None:
    assert _canonical_cuda_uuid(_CudaUuidLike()) == (
        "GPU-576ef18e-b48f-c787-e91d-f93658b27b8e"
    )


def test_atomic_record_resume_skips_a_valid_completed_job(tmp_path: Path) -> None:
    manifest = _manifest()
    job = curated_jobs()[0]
    path = tmp_path / "record.json"
    calls = 0

    def execute(current: ScreenJob) -> dict[str, object]:
        nonlocal calls
        calls += 1
        assert current == job
        return _record(job, str(manifest["plan_sha256"]))

    first = run_job_atomically(
        record_path=path,
        job=job,
        plan_sha256=str(manifest["plan_sha256"]),
        executor=execute,
    )
    second = run_job_atomically(
        record_path=path,
        job=job,
        plan_sha256=str(manifest["plan_sha256"]),
        executor=execute,
    )
    assert first is True
    assert second is False
    assert calls == 1
    assert not tuple(tmp_path.glob("*.partial"))


def test_record_schema_rejects_evidence_outside_validation_scope() -> None:
    manifest = _manifest()
    job = curated_jobs()[0]
    record = _record(job, str(manifest["plan_sha256"]))
    validate_record_payload(
        record,
        job=job,
        plan_sha256=str(manifest["plan_sha256"]),
    )
    direct_job = next(
        candidate
        for candidate in curated_jobs()
        if candidate.model == "chsdnet_direct"
    )
    direct_record = _record(direct_job, str(manifest["plan_sha256"]))
    validate_record_payload(
        direct_record,
        job=direct_job,
        plan_sha256=str(manifest["plan_sha256"]),
    )
    assert direct_record["model"]["source_statistics"]["ran"] is True  # type: ignore[index]

    leaked = copy.deepcopy(record)
    leaked["split"]["test"] = {  # type: ignore[index]
        "count": 10,
        "rows_sha256": _digest("forbidden"),
    }
    with pytest.raises(ValueError):
        validate_record_payload(
            leaked,
            job=job,
            plan_sha256=str(manifest["plan_sha256"]),
        )

    leaked = copy.deepcopy(record)
    leaked["fit"]["history"] = []  # type: ignore[index]
    with pytest.raises(ValueError):
        validate_record_payload(
            leaked,
            job=job,
            plan_sha256=str(manifest["plan_sha256"]),
        )
