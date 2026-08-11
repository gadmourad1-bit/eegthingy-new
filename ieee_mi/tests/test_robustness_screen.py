from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from ieee_mi.data import split_indices
from ieee_mi.robustness_screen import (
    AMENDMENT_LABEL,
    CANDIDATE_MODEL,
    DECISION_RULE,
    EVALUATION_SCOPE,
    EXPECTED_JOB_COUNT,
    EXPECTED_JOBS_PER_WORKER,
    EXPECTED_SUBJECT_COUNT,
    EXPECTED_V2_PLAN_SHA256,
    EXPECTED_V3_ANALYSIS_ARTIFACT_SHA256,
    EXPECTED_V3_PLAN_SHA256,
    HISTORICAL_V3_DECISION,
    MODELS,
    PLAN_SCHEMA,
    RECORD_SCHEMA,
    REFERENCE_MODELS,
    ROBUSTNESS_COHORTS,
    SOURCE_FILES,
    WORKER_COUNT,
    RobustnessScreenError,
    ScreenJob,
    _atomic_create_json,
    _mapping_sha256,
    _read_selected_npy_rows,
    _record_path,
    _validate_robustness_environment_identity,
    _validate_record_bindings,
    assemble_manifest,
    jobs_for_worker,
    robustness_jobs,
    robustness_rows,
    run_job_atomically,
    validate_manifest_payload,
    validate_record_payload,
    write_or_validate_plan,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


GPU_UUIDS = (
    "GPU-00000000-0000-0000-0000-000000000001",
    "GPU-00000000-0000-0000-0000-000000000002",
    "GPU-00000000-0000-0000-0000-000000000003",
    "GPU-00000000-0000-0000-0000-000000000004",
)


def environment_identity() -> dict[str, object]:
    freeze = ["einops==synthetic", "torch==synthetic"]
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
        "uv_pip_freeze": freeze,
        "uv_pip_freeze_sha256": hashlib.sha256(
            "\n".join(freeze).encode("utf-8")
        ).hexdigest(),
    }


def identity_maps() -> tuple[dict[str, object], dict[str, object]]:
    cache_identity: dict[str, object] = {}
    split_identity: dict[str, object] = {}
    for dataset, subjects in ROBUSTNESS_COHORTS:
        for subject in subjects:
            subject_key = f"{dataset}:s{subject:03d}"
            split_key = f"{subject_key}:f00"
            cache_identity[subject_key] = {
                "relative_path": (
                    f"ieee-mi-cache-v2/harmonized/{dataset}/subject_{subject:03d}.npz"
                ),
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


def lineage() -> dict[str, object]:
    return {
        "amendment_label": AMENDMENT_LABEL,
        "prior_decision_is_rewritten": False,
        "reference_v2": {
            "run_root": "/synthetic/reference-v2",
            "plan_schema": "ieee-mi-opened-development-screen-plan-v1",
            "plan_sha256": EXPECTED_V2_PLAN_SHA256,
        },
        "conditioned_v3": {
            "run_root": "/synthetic/conditioned-v3",
            "plan_schema": ("ieee-mi-chsd-conditioned-development-screen-plan-v3"),
            "plan_sha256": EXPECTED_V3_PLAN_SHA256,
        },
        "conditioned_v3_analysis": {
            "artifact_path": "/synthetic/analysis-v3/analysis.json",
            "analysis_schema": ("ieee-mi-chsd-conditioned-screen-analysis-v3"),
            "artifact_sha256": EXPECTED_V3_ANALYSIS_ARTIFACT_SHA256,
            "decision": HISTORICAL_V3_DECISION,
            "selected_model": None,
        },
    }


def external_identity() -> dict[str, object]:
    return {
        "repository": "/synthetic/TCFormer",
        "commit": "74c89b7ab8c64e4eb51e0f748dd87dd4c94e68c5",
        "tracked_dirty": False,
        "runtime_file_sha256": {
            relative: _digest(relative)
            for relative in (
                "models/tcformer.py",
                "models/modules.py",
                "models/channel_group_attention.py",
                "utils/weight_initialization.py",
            )
        },
    }


def synthetic_plan() -> dict[str, object]:
    cache_identity, split_identity = identity_maps()
    return assemble_manifest(
        cache_root="/synthetic/cache",
        cache_identity=cache_identity,
        split_identity=split_identity,
        source_hashes={relative: _digest(relative) for relative in SOURCE_FILES},
        environment_identity=environment_identity(),
        tcformer_identity=external_identity(),
        worker_gpu_uuids=GPU_UUIDS,
        lineage=lineage(),
    )


def synthetic_record(
    plan: dict[str, object],
    job: ScreenJob,
    *,
    balanced_accuracy: float = 0.70,
) -> dict[str, object]:
    subject_key = f"{job.dataset}:s{job.subject:03d}"
    split_key = f"{subject_key}:f{job.fold:02d}"
    worker_index = MODELS.index(job.model)
    model_identity: dict[str, object] = {
        "requested_name": job.model,
        "class": "synthetic.Model",
        "uses_positions": False,
    }
    if job.model == "tcformer":
        model_identity["third_party_provenance"] = {
            "repository": plan["external_source_identity"]["tcformer"][  # type: ignore[index]
                "repository"
            ],
            "commit": plan["external_source_identity"]["tcformer"][  # type: ignore[index]
                "commit"
            ],
            "tracked_dirty": False,
        }
    return {
        "schema": RECORD_SCHEMA,
        "created_at": "2026-07-29T00:00:00+00:00",
        "plan_sha256": plan["plan_sha256"],
        "job_id": job.job_id,
        "job": job.identity(),
        "evaluation_scope": EVALUATION_SCOPE,
        "confirmation_evidence": False,
        "cache_identity": {
            "array_sha256": plan["cache_identity"][subject_key][  # type: ignore[index]
                "array_sha256"
            ],
            "file_sha256": plan["cache_identity"][subject_key][  # type: ignore[index]
                "file_sha256"
            ],
        },
        "source_identity": copy.deepcopy(plan["source_identity"]),
        "environment_identity_sha256": _mapping_sha256(
            plan["environment_identity"]  # type: ignore[arg-type]
        ),
        "split": copy.deepcopy(
            plan["split_identity"][split_key]  # type: ignore[index]
        ),
        "scaler": {
            "fit_scope": "training_only",
            "mean_sha256": _digest(f"mean-{job.job_id}"),
            "std_sha256": _digest(f"std-{job.job_id}"),
            "contract": copy.deepcopy(plan["channel_scaling"]),
        },
        "model": {
            "identity": model_identity,
            "parameter_count": 100,
            "constructed_state_sha256": _digest("constructed"),
            "initial_state_sha256": _digest("constructed"),
            "selected_state_sha256": _digest("selected"),
            "source_statistics": {
                "available": False,
                "ran": False,
                "fit_scope": "training_rows_only",
                "outcome_aware": False,
            },
        },
        "fit": {
            "best_epoch": 10,
            "epochs_run": 20,
            "best_validation_loss": 0.5,
            "early_stopping": True,
            "config": copy.deepcopy(plan["train_config"]),
        },
        "validation_metrics": {
            "accuracy": balanced_accuracy,
            "balanced_accuracy": balanced_accuracy,
            "cohen_kappa": max(-1.0, 2.0 * balanced_accuracy - 1.0),
            "negative_log_likelihood": 0.5,
            "roc_auc": balanced_accuracy,
        },
        "timing": {
            "construction_seconds": 0.1,
            "fit_seconds": 1.0,
            "evaluation_seconds": 0.1,
            "total_seconds": 1.2,
        },
        "resource_preflight": {
            "gpu": {
                "gpu_uuid": GPU_UUIDS[worker_index],
                "pci_bus_id": "00000000:01:00.0",
                "name": "Synthetic GPU",
                "utilization_percent": 0.0,
                "memory_used_mib": 0.0,
                "own_pid": 123,
                "own_compute_memory_mib": 0.0,
                "foreign_compute_processes": [],
            },
            "disk": {
                "path": "/synthetic/run",
                "free_bytes": 64 * 1024**3,
                "free_gib": 60.0,
                "minimum_free_gib": 50.0,
            },
        },
    }


def write_complete_run(
    run_root: Path,
    *,
    scores: dict[str, float] | None = None,
) -> dict[str, object]:
    plan = synthetic_plan()
    write_or_validate_plan(run_root, plan)
    by_model = scores or {
        CANDIDATE_MODEL: 0.70,
        "cardinal_fbc_micro_extended": 0.72,
        "fbcnet": 0.71,
        "tcformer": 0.69,
    }
    for job in robustness_jobs():
        record = synthetic_record(plan, job, balanced_accuracy=by_model[job.model])
        assert _atomic_create_json(_record_path(run_root, job), record)
    return plan


def test_exact_disjoint_cohort_and_460_model_owned_jobs() -> None:
    expected_prior = {
        "local_exp4": {1, 4, 7},
        "bnci2014_001": {1, 5, 9},
        "bnci2014_004": {1, 5, 9},
        "cho2017": {1, 18, 35, 52},
        "physionet_mi": {1, 18, 36, 54},
    }
    expected_counts = {
        "local_exp4": 5,
        "bnci2014_001": 6,
        "bnci2014_004": 6,
        "cho2017": 48,
        "physionet_mi": 50,
    }
    assert sum(len(subjects) for _, subjects in ROBUSTNESS_COHORTS) == 115
    for dataset, subjects in ROBUSTNESS_COHORTS:
        assert len(subjects) == expected_counts[dataset]
        assert not (set(subjects) & expected_prior[dataset])

    jobs = robustness_jobs()
    assert len(jobs) == EXPECTED_JOB_COUNT == 460
    assert len({job.job_id for job in jobs}) == 460
    assert {job.fold for job in jobs} == {0}
    assert {job.seed for job in jobs} == {7}
    assert MODELS == (
        CANDIDATE_MODEL,
        "cardinal_fbc_micro_extended",
        "fbcnet",
        "tcformer",
    )
    partitions = [jobs_for_worker(jobs, index) for index in range(WORKER_COUNT)]
    assert [len(partition) for partition in partitions] == [115] * 4
    assert all(
        {job.model for job in partition} == {MODELS[index]}
        for index, partition in enumerate(partitions)
    )
    assert set().union(*(set(partition) for partition in partitions)) == set(jobs)


@pytest.mark.parametrize(
    ("subject", "trial_count", "train_count", "validation_count"),
    ((2, 200, 120, 40), (7, 240, 144, 48), (9, 240, 144, 48), (46, 240, 144, 48)),
)
def test_cho_metadata_only_rows_match_formal_fold_zero(
    subject: int,
    trial_count: int,
    train_count: int,
    validation_count: int,
) -> None:
    sessions = np.asarray(["session"] * trial_count)
    runs = np.asarray(["run"] * trial_count)
    train, validation = robustness_rows("cho2017", subject, sessions, runs)
    labels = np.repeat((0, 1), trial_count // 2).astype(np.int64)
    formal_train, formal_validation, _ = split_indices(
        "cho2017",
        labels,
        sessions,
        runs,
        fold=0,
        subject=subject,
    )
    assert len(train) == train_count
    assert len(validation) == validation_count
    np.testing.assert_array_equal(train, formal_train)
    np.testing.assert_array_equal(validation, formal_validation)
    assert not np.intersect1d(train, validation).size


def test_selected_npy_reader_never_allocates_excluded_rows(tmp_path: Path) -> None:
    path = tmp_path / "cache.npz"
    x = np.arange(12 * 3 * 4, dtype=np.float32).reshape(12, 3, 4)
    y = np.arange(12, dtype=np.int64)
    with path.open("wb") as handle:
        np.savez_compressed(handle, x=x, y=y)
    rows = np.asarray([9, 1, 7, 2], dtype=np.int64)
    selected_x, x_shape = _read_selected_npy_rows(
        path,
        "x",
        rows,
        expected_dtype=np.dtype(np.float32),
        expected_ndim=3,
    )
    selected_y, y_shape = _read_selected_npy_rows(
        path,
        "y",
        rows,
        expected_dtype=np.dtype(np.int64),
        expected_ndim=1,
    )
    assert x_shape == x.shape
    assert y_shape == y.shape
    np.testing.assert_array_equal(selected_x, x[rows])
    np.testing.assert_array_equal(selected_y, y[rows])


def test_complete_uv_freeze_is_bound_and_requires_einops() -> None:
    identity = environment_identity()
    _validate_robustness_environment_identity(identity)
    missing = copy.deepcopy(identity)
    missing["uv_pip_freeze"] = ["torch==synthetic"]
    missing["uv_pip_freeze_sha256"] = hashlib.sha256(
        b"torch==synthetic"
    ).hexdigest()
    with pytest.raises(ValueError, match="einops"):
        _validate_robustness_environment_identity(missing)


def test_manifest_freezes_lineage_decision_rule_sources_and_four_gpus() -> None:
    plan = synthetic_plan()
    validate_manifest_payload(plan)
    assert plan["schema"] == PLAN_SCHEMA
    assert plan["n_subjects"] == EXPECTED_SUBJECT_COUNT
    assert plan["n_jobs"] == EXPECTED_JOB_COUNT
    assert plan["jobs_per_worker"] == EXPECTED_JOBS_PER_WORKER
    assert plan["decision_rule"] == dict(DECISION_RULE)
    assert plan["lineage"]["reference_v2"]["plan_sha256"] == (  # type: ignore[index]
        EXPECTED_V2_PLAN_SHA256
    )
    assert plan["lineage"]["conditioned_v3"]["plan_sha256"] == (  # type: ignore[index]
        EXPECTED_V3_PLAN_SHA256
    )
    assert (
        plan["lineage"]["conditioned_v3_analysis"][  # type: ignore[index]
            "artifact_sha256"
        ]
        == EXPECTED_V3_ANALYSIS_ARTIFACT_SHA256
    )
    assert (
        plan["lineage"]["conditioned_v3_analysis"][  # type: ignore[index]
            "decision"
        ]
        == HISTORICAL_V3_DECISION
    )
    assert plan["lineage"]["prior_decision_is_rewritten"] is False  # type: ignore[index]
    assert "ieee_mi/robustness_analysis.py" in plan["source_identity"]
    assert [
        binding["gpu_uuid"]
        for binding in plan["worker_bindings"]  # type: ignore[union-attr]
    ] == list(GPU_UUIDS)


def test_manifest_rejects_history_rewrite_cache_drift_and_gpu_alias() -> None:
    cache_identity, split_identity = identity_maps()
    bad_lineage = lineage()
    bad_lineage["conditioned_v3_analysis"]["decision"] = (  # type: ignore[index]
        "choose_one_for_next_prespecified_stage"
    )
    with pytest.raises(ValueError, match="analysis lineage"):
        assemble_manifest(
            cache_root="/synthetic/cache",
            cache_identity=cache_identity,
            split_identity=split_identity,
            source_hashes={relative: _digest(relative) for relative in SOURCE_FILES},
            environment_identity=environment_identity(),
            tcformer_identity=external_identity(),
            worker_gpu_uuids=GPU_UUIDS,
            lineage=bad_lineage,
        )

    bad_cache = copy.deepcopy(cache_identity)
    first_key = min(bad_cache)
    bad_cache[first_key]["relative_path"] = (  # type: ignore[index]
        "ieee-mi-cache-v2/harmonized/wrong/subject_001.npz"
    )
    with pytest.raises(ValueError, match="exact v2/harmonized"):
        assemble_manifest(
            cache_root="/synthetic/cache",
            cache_identity=bad_cache,
            split_identity=split_identity,
            source_hashes={relative: _digest(relative) for relative in SOURCE_FILES},
            environment_identity=environment_identity(),
            tcformer_identity=external_identity(),
            worker_gpu_uuids=GPU_UUIDS,
            lineage=lineage(),
        )

    with pytest.raises(ValueError, match="full NVIDIA UUID"):
        assemble_manifest(
            cache_root="/synthetic/cache",
            cache_identity=cache_identity,
            split_identity=split_identity,
            source_hashes={relative: _digest(relative) for relative in SOURCE_FILES},
            environment_identity=environment_identity(),
            tcformer_identity=external_identity(),
            worker_gpu_uuids=("0", *GPU_UUIDS[1:]),
            lineage=lineage(),
        )


def test_record_is_aggregate_only_and_fully_plan_bound() -> None:
    plan = synthetic_plan()
    job = robustness_jobs()[0]
    record = synthetic_record(plan, job)
    validate_record_payload(record, job=job, plan_sha256=str(plan["plan_sha256"]))
    _validate_record_bindings(record, job=job, plan=plan)

    leaked = copy.deepcopy(record)
    leaked["split"]["test"] = {  # type: ignore[index]
        "count": 10,
        "rows_sha256": _digest("forbidden"),
    }
    with pytest.raises(ValueError, match="split exceeds"):
        validate_record_payload(leaked, job=job, plan_sha256=str(plan["plan_sha256"]))

    leaked = copy.deepcopy(record)
    leaked["predictions"] = [0, 1]
    with pytest.raises(ValueError):
        validate_record_payload(leaked, job=job, plan_sha256=str(plan["plan_sha256"]))

    changed = copy.deepcopy(record)
    changed["environment_identity_sha256"] = _digest("other")
    with pytest.raises(ValueError, match="environment identity"):
        _validate_record_bindings(changed, job=job, plan=plan)

    busy = copy.deepcopy(record)
    busy["resource_preflight"]["gpu"]["foreign_compute_processes"] = [  # type: ignore[index]
        {"pid": 9}
    ]
    with pytest.raises(ValueError, match="foreign GPU"):
        _validate_record_bindings(busy, job=job, plan=plan)

    low_disk = copy.deepcopy(record)
    low_disk["resource_preflight"]["disk"]["free_gib"] = 49.99  # type: ignore[index]
    with pytest.raises(ValueError, match="50 GiB"):
        _validate_record_bindings(low_disk, job=job, plan=plan)

    tcformer_job = next(item for item in robustness_jobs() if item.model == "tcformer")
    tcformer_record = synthetic_record(plan, tcformer_job)
    _validate_record_bindings(tcformer_record, job=tcformer_job, plan=plan)
    tcformer_record["model"]["identity"]["third_party_provenance"][  # type: ignore[index]
        "commit"
    ] = "0" * 40
    with pytest.raises(ValueError, match="TCFormer record provenance"):
        _validate_record_bindings(tcformer_record, job=tcformer_job, plan=plan)


def test_atomic_resume_skips_valid_and_fails_on_corruption(
    tmp_path: Path,
) -> None:
    plan = synthetic_plan()
    first, second = robustness_jobs()[:2]
    first_path = _record_path(tmp_path, first)
    calls: list[str] = []

    def execute(job: ScreenJob) -> dict[str, object]:
        calls.append(job.job_id)
        return synthetic_record(plan, job)

    assert run_job_atomically(
        record_path=first_path,
        job=first,
        plan=plan,
        executor=execute,
    )
    assert calls == [first.job_id]
    assert not run_job_atomically(
        record_path=first_path,
        job=first,
        plan=plan,
        executor=execute,
    )
    assert calls == [first.job_id]

    second_path = _record_path(tmp_path, second)
    second_path.parent.mkdir(parents=True, exist_ok=True)
    second_path.write_text('{"corrupt": true}\\n', encoding="utf-8")
    with pytest.raises(ValueError):
        run_job_atomically(
            record_path=second_path,
            job=second,
            plan=plan,
            executor=execute,
        )
    assert calls == [first.job_id]


def test_plan_is_immutable_on_resume(tmp_path: Path) -> None:
    plan = synthetic_plan()
    assert write_or_validate_plan(tmp_path, plan)
    assert not write_or_validate_plan(tmp_path, plan)
    changed = copy.deepcopy(plan)
    changed["cache_root"] = "/different/cache"
    changed["plan_sha256"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in changed.items() if key != "plan_sha256"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    with pytest.raises(RobustnessScreenError):
        write_or_validate_plan(tmp_path, changed)


def test_model_and_reference_roster_are_fixed() -> None:
    assert CANDIDATE_MODEL == "chsdnet_conditioned_005"
    assert REFERENCE_MODELS == (
        "cardinal_fbc_micro_extended",
        "fbcnet",
        "tcformer",
    )
