from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from ieee_mi.conditioned_screen import (
    CONDITIONED_MODELS,
    EXPECTED_JOB_COUNT,
    EXPECTED_JOBS_PER_WORKER,
    EXPECTED_REFERENCE_PLAN_SHA256,
    PLAN_SCHEMA,
    RECORD_SCHEMA,
    REFERENCE_MODELS,
    SOURCE_FILES,
    WORKER_COUNT,
    ConditionedScreenError,
    ScreenJob,
    _canonical_cuda_uuid,
    _validate_record_bindings,
    assemble_manifest,
    conditioned_jobs,
    jobs_for_worker,
    run_job_atomically,
    validate_manifest_payload,
    validate_record_payload,
    write_or_validate_plan,
)
from ieee_mi.development_screen import (
    CURATED_SPLITS,
    EVALUATION_SCOPE,
    PLAN_SCHEMA as REFERENCE_PLAN_SCHEMA,
    RECORD_SCHEMA as REFERENCE_RECORD_SCHEMA,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


GPU_UUIDS = (
    "GPU-00000000-0000-0000-0000-000000000001",
    "GPU-00000000-0000-0000-0000-000000000002",
    "GPU-00000000-0000-0000-0000-000000000003",
)


class _CudaUuidLike:
    def __str__(self) -> str:
        return "576ef18e-b48f-c787-e91d-f93658b27b8e"


def _environment_identity() -> dict[str, object]:
    return {
        "python": "3.synthetic",
        "executable": "/synthetic/.venv/bin/python",
        "platform": "synthetic",
        "packages": {"torch": "synthetic"},
        "torch_cuda_version": "synthetic",
        "torch_cudnn_version": 999,
        "nvidia_driver_versions": ["synthetic"],
        "required_cublas_workspace_config": ":4096:8",
        "cpu_threads_per_worker": 4,
        "minimum_free_gib": 50.0,
        "uv_executable": "/synthetic/bin/uv",
        "uv_version": "uv 0.synthetic",
        "virtualenv_prefix": "/synthetic/.venv",
        "pyvenv_cfg_sha256": _digest("pyvenv"),
    }


def _identity_maps() -> tuple[dict[str, object], dict[str, object]]:
    cache_identity: dict[str, object] = {}
    split_identity: dict[str, object] = {}
    for dataset, subjects in CURATED_SPLITS:
        for subject in subjects:
            subject_key = f"{dataset}:s{subject:03d}"
            split_key = f"{subject_key}:f00"
            cache_identity[subject_key] = {
                "relative_path": f"cache/{dataset}/subject_{subject:03d}.npz",
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
    return cache_identity, split_identity


def _reference_binding() -> dict[str, object]:
    reference_job_ids = [
        ScreenJob(dataset=dataset, model=model, subject=subject).job_id
        for model in REFERENCE_MODELS
        for dataset, subjects in CURATED_SPLITS
        for subject in subjects
    ]
    digest = hashlib.sha256(
        json.dumps(
            reference_job_ids,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "run_root": "/synthetic/reference-v2",
        "plan_schema": REFERENCE_PLAN_SCHEMA,
        "record_schema": REFERENCE_RECORD_SCHEMA,
        "plan_sha256": EXPECTED_REFERENCE_PLAN_SHA256,
        "evaluation_scope": EVALUATION_SCOPE,
        "models": list(REFERENCE_MODELS),
        "selected_record_count": 51,
        "selected_job_ids_sha256": digest,
    }


def _manifest(source_token: str = "source") -> dict[str, object]:
    cache_identity, split_identity = _identity_maps()
    return assemble_manifest(
        cache_root="/synthetic/cache",
        cache_identity=cache_identity,
        split_identity=split_identity,
        source_hashes={
            relative: _digest(f"{source_token}-{relative}")
            for relative in SOURCE_FILES
        },
        environment_identity=_environment_identity(),
        worker_gpu_uuids=GPU_UUIDS,
        reference_binding=_reference_binding(),
    )


def _record(job: ScreenJob, plan: dict[str, object]) -> dict[str, object]:
    subject_key = f"{job.dataset}:s{job.subject:03d}"
    split_key = f"{subject_key}:f{job.fold:02d}"
    return {
        "schema": RECORD_SCHEMA,
        "created_at": "2026-07-29T00:00:00+00:00",
        "plan_sha256": plan["plan_sha256"],
        "job_id": job.job_id,
        "job": job.identity(),
        "evaluation_scope": EVALUATION_SCOPE,
        "confirmation_evidence": False,
        "cache_identity": {
            "array_sha256": plan["cache_identity"][subject_key]["array_sha256"],  # type: ignore[index]
            "file_sha256": plan["cache_identity"][subject_key]["file_sha256"],  # type: ignore[index]
        },
        "source_identity": plan["source_identity"],
        "split": plan["split_identity"][split_key],  # type: ignore[index]
        "scaler": {
            "fit_scope": "training_only",
            "mean_sha256": _digest("mean"),
            "std_sha256": _digest("std"),
            "contract": plan["channel_scaling"],
        },
        "model": {
            "identity": {"requested_name": job.model},
            "parameter_count": 123,
            "constructed_state_sha256": _digest("constructed"),
            "initial_state_sha256": _digest("initial"),
            "selected_state_sha256": _digest("selected"),
            "source_statistics": {
                "available": False,
                "ran": False,
                "fit_scope": "training_rows_only",
                "outcome_aware": False,
            },
        },
        "fit": {
            "best_epoch": 3,
            "epochs_run": 8,
            "best_validation_loss": 0.42,
            "early_stopping": True,
            "config": plan["train_config"],
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
            "gpu": {
                "gpu_uuid": GPU_UUIDS[CONDITIONED_MODELS.index(job.model)],
                "foreign_compute_processes": [],
            },
            "disk": {"free_gib": 100.0, "minimum_free_gib": 50.0},
        },
    }


def test_exact_51_job_cardinality_and_one_variant_static_partitions() -> None:
    plan = _manifest()
    jobs = conditioned_jobs()
    assert plan["schema"] == PLAN_SCHEMA
    assert plan["n_jobs"] == EXPECTED_JOB_COUNT == 51
    assert len(jobs) == EXPECTED_JOB_COUNT
    assert len({job.job_id for job in jobs}) == EXPECTED_JOB_COUNT
    assert {job.seed for job in jobs} == {7}
    assert {job.fold for job in jobs} == {0}

    partitions = [
        jobs_for_worker(jobs, index) for index in range(WORKER_COUNT)
    ]
    assert [len(partition) for partition in partitions] == [
        EXPECTED_JOBS_PER_WORKER
    ] * WORKER_COUNT
    assert [
        {job.model for job in partition} for partition in partitions
    ] == [{model} for model in CONDITIONED_MODELS]
    ids = [{job.job_id for job in partition} for partition in partitions]
    assert set().union(*ids) == {job.job_id for job in jobs}
    assert all(
        left.isdisjoint(right)
        for index, left in enumerate(ids)
        for right in ids[index + 1 :]
    )


def test_plan_hash_is_immutable_and_covers_exact_dependency_closure(
    tmp_path: Path,
) -> None:
    first = _manifest("first")
    same = _manifest("first")
    changed = _manifest("changed")
    assert first == same
    assert first["plan_sha256"] != changed["plan_sha256"]
    assert set(first["source_identity"]) == set(SOURCE_FILES)  # type: ignore[arg-type]
    validate_manifest_payload(first)
    assert write_or_validate_plan(tmp_path, first) is True
    assert write_or_validate_plan(tmp_path, same) is False
    with pytest.raises(ConditionedScreenError):
        write_or_validate_plan(tmp_path, changed)


def test_atomic_resume_is_idempotent_and_corruption_fails_closed(
    tmp_path: Path,
) -> None:
    plan = _manifest()
    job = conditioned_jobs()[0]
    record_path = tmp_path / "record.json"
    calls = 0

    def executor(current: ScreenJob) -> dict[str, object]:
        nonlocal calls
        calls += 1
        assert current == job
        return _record(job, plan)

    assert run_job_atomically(
        record_path=record_path,
        job=job,
        plan_sha256=str(plan["plan_sha256"]),
        executor=executor,
    )
    assert not run_job_atomically(
        record_path=record_path,
        job=job,
        plan_sha256=str(plan["plan_sha256"]),
        executor=executor,
    )
    assert calls == 1
    corrupt = json.loads(record_path.read_text(encoding="utf-8"))
    corrupt["validation_metrics"]["balanced_accuracy"] = float("nan")
    record_path.write_text(json.dumps(corrupt), encoding="utf-8")
    with pytest.raises(ValueError):
        run_job_atomically(
            record_path=record_path,
            job=job,
            plan_sha256=str(plan["plan_sha256"]),
            executor=executor,
        )
    assert calls == 1


def test_forbidden_excluded_evidence_and_optional_hook_semantics() -> None:
    plan = _manifest()
    job = conditioned_jobs()[0]
    record = _record(job, plan)
    validate_record_payload(
        record, job=job, plan_sha256=str(plan["plan_sha256"])
    )
    _validate_record_bindings(record, job=job, plan=plan)

    leaked = copy.deepcopy(record)
    leaked["split"]["test"] = {  # type: ignore[index]
        "count": 10,
        "rows_sha256": _digest("excluded"),
    }
    with pytest.raises(ValueError):
        validate_record_payload(
            leaked, job=job, plan_sha256=str(plan["plan_sha256"])
        )

    leaked = copy.deepcopy(record)
    leaked["labels"] = [0, 1]
    with pytest.raises(ValueError):
        validate_record_payload(
            leaked, job=job, plan_sha256=str(plan["plan_sha256"])
        )

    inconsistent_hook = copy.deepcopy(record)
    inconsistent_hook["model"]["source_statistics"]["available"] = True  # type: ignore[index]
    with pytest.raises(ValueError):
        validate_record_payload(
            inconsistent_hook,
            job=job,
            plan_sha256=str(plan["plan_sha256"]),
        )

    validation_fit_scaler = copy.deepcopy(record)
    validation_fit_scaler["scaler"]["fit_scope"] = "training_and_validation"  # type: ignore[index]
    with pytest.raises(ValueError, match="source-only"):
        _validate_record_bindings(
            validation_fit_scaler, job=job, plan=plan
        )


def test_reference_sha_schema_and_job_digest_are_bound() -> None:
    cache_identity, split_identity = _identity_maps()
    bad = _reference_binding()
    bad["plan_sha256"] = _digest("wrong-reference")
    with pytest.raises(ValueError, match="reference plan SHA"):
        assemble_manifest(
            cache_root="/synthetic/cache",
            cache_identity=cache_identity,
            split_identity=split_identity,
            source_hashes={
                relative: _digest(relative) for relative in SOURCE_FILES
            },
            environment_identity=_environment_identity(),
            worker_gpu_uuids=GPU_UUIDS,
            reference_binding=bad,
        )

    bad = _reference_binding()
    bad["selected_job_ids_sha256"] = _digest("wrong-jobs")
    with pytest.raises(ValueError, match="reference job identity"):
        assemble_manifest(
            cache_root="/synthetic/cache",
            cache_identity=cache_identity,
            split_identity=split_identity,
            source_hashes={
                relative: _digest(relative) for relative in SOURCE_FILES
            },
            environment_identity=_environment_identity(),
            worker_gpu_uuids=GPU_UUIDS,
            reference_binding=bad,
        )


def test_exact_uuid_supports_non_string_pytorch_uuid_and_rejects_aliases() -> None:
    assert _canonical_cuda_uuid(_CudaUuidLike()) == (
        "GPU-576ef18e-b48f-c787-e91d-f93658b27b8e"
    )
    cache_identity, split_identity = _identity_maps()
    with pytest.raises(ValueError, match="full NVIDIA UUID"):
        assemble_manifest(
            cache_root="/synthetic/cache",
            cache_identity=cache_identity,
            split_identity=split_identity,
            source_hashes={
                relative: _digest(relative) for relative in SOURCE_FILES
            },
            environment_identity=_environment_identity(),
            worker_gpu_uuids=("0", GPU_UUIDS[1], GPU_UUIDS[2]),
            reference_binding=_reference_binding(),
        )
