from __future__ import annotations

import copy
import io
import json
import os
import stat
import subprocess
import threading
import time
import zipfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmark import author_recipe_analysis, author_recipe_grid, full_grid
from benchmark.reference_training import (
    FBCNetReferenceConfig,
    TCFormerReferenceConfig,
)


def _split(
    *,
    dataset: str = "tiny",
    subject: int = 1,
    fold: int = 0,
    digest: str = "a" * 64,
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "subject": subject,
        "fold": fold,
        "cache_array_sha256": digest,
        "trial_count": 6,
        "partitions": {
            "train": {
                "count": 2,
                "rows_sha256": author_recipe_grid._rows_sha256([0, 1]),
            },
            "validation": {
                "count": 2,
                "rows_sha256": author_recipe_grid._rows_sha256([2, 3]),
            },
            "source": {
                "count": 4,
                "rows_sha256": author_recipe_grid._rows_sha256([0, 1, 2, 3]),
            },
            "test": {
                "count": 2,
                "rows_sha256": author_recipe_grid._rows_sha256([4, 5]),
            },
        },
    }


def _tiny_recipe_contracts() -> dict[str, object]:
    tcformer = TCFormerReferenceConfig(
        epochs=2,
        warmup_epochs=1,
        batch_size=2,
        device="cuda",
        horizon_provenance="synthetic_test",
    )
    fbcnet = FBCNetReferenceConfig(
        stage1_max_epochs=2,
        stage1_patience=1,
        stage2_max_epochs=2,
        batch_size=2,
        device="cuda",
    )
    return {
        "reference.tcformer": {
            "component_callable": "test.tcformer",
            "factory_callable": "test.factory",
            "per_dataset_config": {"tiny": asdict(tcformer)},
            "selection": "prespecified_fixed_horizon_no_outcome_selection",
            "final_fit": "train_plus_validation_for_prespecified_horizon",
        },
        "reference.fbcnet": {
            "component_callable": "test.fbcnet",
            "factory_callable": "test.factory",
            "config": asdict(fbcnet),
            "selection": "train_fit_validation_inaccuracy_patience",
            "final_fit": "test_stage_two",
        },
    }


def _tiny_plan(*, seeds: tuple[int, ...] = (7,)) -> dict[str, object]:
    return author_recipe_grid.assemble_plan(
        dataset_contracts={
            "tiny": {
                "subjects": [1],
                "folds": [0],
                "n_classes": 2,
                "protocol": "synthetic",
                "preprocessing": {"sfreq_hz": 8.0, "n_times": 8},
            }
        },
        references=author_recipe_grid.REFERENCES,
        seeds=seeds,
        cache_identity={
            "tiny:s001": {
                "array_sha256": "a" * 64,
                "shape": [6, 3, 8],
            }
        },
        split_identity={"tiny:s001:f00": _split()},
        source_identity={"runner.py": "b" * 64},
        environment_identity={
            "python": "test",
            "nvidia_driver_versions": ["test-driver"],
        },
        reference_provenance={
            "reference.tcformer": {"commit": "c" * 40},
            "reference.fbcnet": {"commit": "d" * 40},
        },
        recipe_contracts=_tiny_recipe_contracts(),
        executor="test.injected:executor",
        worker_cpu_threads=2,
    )


def _formal_roster_synthetic_plan(
    *,
    executor: str = author_recipe_grid.DEFAULT_EXECUTOR,
) -> dict[str, object]:
    contracts: dict[str, object] = {}
    caches: dict[str, object] = {}
    splits: dict[str, object] = {}
    for dataset in author_recipe_grid.OPENED_DATASETS:
        contracts[dataset] = {
            "subjects": [1],
            "folds": [0],
            "n_classes": 2,
            "protocol": "synthetic",
            "preprocessing": {"sfreq_hz": 8.0, "n_times": 8},
        }
        subject_key = author_recipe_grid._subject_key(dataset, 1)
        caches[subject_key] = {
            "array_sha256": "a" * 64,
            "shape": [6, 3, 8],
        }
        splits[author_recipe_grid._split_key(dataset, 1, 0)] = _split(
            dataset=dataset,
        )
    recipes = _tiny_recipe_contracts()
    recipes["reference.tcformer"]["per_dataset_config"] = {
        dataset: copy.deepcopy(
            recipes["reference.tcformer"]["per_dataset_config"]["tiny"]
        )
        for dataset in author_recipe_grid.OPENED_DATASETS
    }
    return author_recipe_grid.assemble_plan(
        dataset_contracts=contracts,
        references=author_recipe_grid.REFERENCES,
        seeds=author_recipe_grid.FORMAL_SEEDS,
        cache_identity=caches,
        split_identity=splits,
        source_identity={"runner.py": "b" * 64},
        environment_identity={"python": "test"},
        reference_provenance={
            "reference.tcformer": {"commit": "c" * 40},
            "reference.fbcnet": {"commit": "d" * 40},
        },
        recipe_contracts=recipes,
        executor=executor,
        worker_cpu_threads=2,
    )


def _fit_metadata(
    plan: dict[str, object],
    job: author_recipe_grid.Job,
) -> dict[str, object]:
    common = {
        "seed_installed_before_construction": True,
        "initial_state_sha256": "1" * 64,
        "final_state_sha256": "2" * 64,
        "final_optimizer_state_sha256": "3" * 64,
        "source_count": 4,
        "parameter_count": 10,
    }
    if job.reference == "reference.tcformer":
        fit = {
            **common,
            "epochs_run": 2,
        }
        scalers = {
            "source_mean_sha256": "4" * 64,
            "source_std_sha256": "5" * 64,
        }
    else:
        fit = {
            **common,
            "stage1_best_epoch": 0,
            "stage1_stop": "relative_no_decrease_patience",
            "stage1_epochs_run": 1,
            "stage2_epochs_run": 1,
            "stage2_stop": "max_epochs",
            "stage1_train_loss_threshold": 0.5,
            "stage1_threshold_origin": (
                "last_stage1_epoch_before_best_restore"
            ),
            "stage1_threshold_origin_epoch": 0,
            "stage1_best_model_state_sha256": "6" * 64,
            "stage1_best_optimizer_state_sha256": "7" * 64,
            "stage2_start_model_state_sha256": "6" * 64,
            "stage2_start_optimizer_state_sha256": "7" * 64,
            "original_validation_count": 2,
        }
        scalers = {
            "selection_mean_sha256": "4" * 64,
            "selection_std_sha256": "5" * 64,
        }
    return {
        "track": author_recipe_grid.TRACK,
        "cache_array_sha256": "a" * 64,
        "channels": ["C3", "Cz", "C4"],
        "split": author_recipe_grid._split_metadata(
            plan["split_identity"]["tiny:s001:f00"]
        ),
        "scalers": scalers,
        "recipe": author_recipe_grid._expected_recipe_metadata(plan, job),
        "fit": fit,
        "protocol": author_recipe_grid._expected_protocol(job.reference),
        "timing_seconds": {
            "source_fit": 0.1,
            "test_inference": 0.01,
            "job_total": 0.11,
        },
        "cuda_peak_memory_bytes": 1024,
        "runtime": {
            "worker_cpu_threads": 2,
            "worker_interop_threads": 1,
            "thread_environment": {
                name: "2"
                for name in author_recipe_grid.THREAD_ENVIRONMENT_VARIABLES
            },
            "nvidia_driver_versions": ["test-driver"],
        },
    }


def _safe_gpu() -> full_grid.GPUStatus:
    return full_grid.GPUStatus(
        safe=True,
        gpu="GPU-test",
        utilization_percent=0.0,
        memory_used_mib=0.0,
        own_compute_memory_mib=0.0,
        foreign_processes=(),
        reason="idle",
    )


def _safe_disk() -> full_grid.DiskStatus:
    return full_grid.DiskStatus(
        safe=True,
        path="/synthetic",
        free_bytes=100 * 1024**3,
        total_bytes=200 * 1024**3,
        free_gib=100.0,
        minimum_free_gib=50.0,
        reason="safe",
    )


def _safe_resources() -> full_grid.ResourceStatus:
    return full_grid.combine_resource_status(_safe_gpu(), _safe_disk())


def _low_disk() -> full_grid.DiskStatus:
    return full_grid.DiskStatus(
        safe=False,
        path="/synthetic",
        free_bytes=49 * 1024**3,
        total_bytes=200 * 1024**3,
        free_gib=49.0,
        minimum_free_gib=50.0,
        reason="below low-water mark",
    )


def _publish(
    run_root: Path,
    plan: dict[str, object],
    job: author_recipe_grid.Job,
    probabilities: np.ndarray | None = None,
) -> Path:
    claim = author_recipe_grid.acquire_claim(
        run_root, plan, job, before_claim=_safe_resources
    )
    try:
        return author_recipe_grid.commit_job_output(
            run_root,
            plan,
            claim,
            metadata=_fit_metadata(plan, job),
            test_rows=np.asarray([4, 5], dtype=np.int64),
            probabilities=(
                np.asarray([[0.8, 0.2], [0.1, 0.9]], dtype=np.float64)
                if probabilities is None
                else probabilities
            ),
        )
    finally:
        author_recipe_grid.release_claim(claim)


def _install_fake_cache(
    monkeypatch: pytest.MonkeyPatch,
    plan: dict[str, object],
) -> None:
    fake_cache = {
        "identity": copy.deepcopy(plan["cache_identity"]["tiny:s001"]),
        "x": np.zeros((6, 3, 8), dtype=np.float32),
        "y": np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64),
        "positions": np.eye(3, dtype=np.float32),
        "channel_names": np.asarray(["C3", "Cz", "C4"]),
        "sessions": np.asarray(["a"] * 6),
        "runs": np.asarray(["a"] * 6),
    }
    import benchmark.data as data_module

    monkeypatch.setattr(
        author_recipe_grid,
        "_load_subject_cache_secure",
        lambda *unused, **unused_keywords: copy.deepcopy(fake_cache),
    )
    monkeypatch.setattr(
        data_module,
        "split_indices",
        lambda *unused, **unused_keywords: (
            np.asarray([0, 1], dtype=np.int64),
            np.asarray([2, 3], dtype=np.int64),
            np.asarray([4, 5], dtype=np.int64),
        ),
    )


def _rewrite_record_and_completion(
    output: Path,
    mutate: object,
) -> None:
    record_path = output / "record.json"
    completion_path = output / "completion.json"
    record = json.loads(record_path.read_text())
    mutate(record)
    os.chmod(record_path, 0o600)
    record_path.write_bytes(author_recipe_grid._canonical_bytes(record) + b"\n")
    os.chmod(record_path, 0o444)
    completion = json.loads(completion_path.read_text())
    completion["files"]["record.json"] = author_recipe_grid._sha256_file(
        record_path
    )
    os.chmod(completion_path, 0o600)
    completion_path.write_bytes(
        author_recipe_grid._canonical_bytes(completion) + b"\n"
    )
    os.chmod(completion_path, 0o444)


def _completed_tiny_grid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, object], Path, Path]:
    plan = _tiny_plan()
    root = tmp_path / "run"
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    author_recipe_grid.write_or_validate_plan(root, plan)
    for job in author_recipe_grid.iter_jobs(plan):
        _publish(root, plan, job)
    _install_fake_cache(monkeypatch, plan)
    return plan, root, cache_root


def test_production_grid_has_exact_4480_jobs_and_2240_per_reference() -> None:
    contracts = author_recipe_grid._dataset_contracts()
    caches: dict[str, object] = {}
    splits: dict[str, object] = {}
    for dataset, contract in contracts.items():
        for subject in contract["subjects"]:
            key = author_recipe_grid._subject_key(dataset, subject)
            digest = author_recipe_grid._sha256_bytes(key.encode())
            caches[key] = {"array_sha256": digest}
            for fold in contract["folds"]:
                split = _split(
                    dataset=dataset,
                    subject=subject,
                    fold=fold,
                    digest=digest,
                )
                splits[
                    author_recipe_grid._split_key(dataset, subject, fold)
                ] = split
    plan = author_recipe_grid.assemble_plan(
        dataset_contracts=contracts,
        references=author_recipe_grid.REFERENCES,
        seeds=author_recipe_grid.FORMAL_SEEDS,
        cache_identity=caches,
        split_identity=splits,
        source_identity={"runner.py": "b" * 64},
        environment_identity={"python": "test"},
        reference_provenance={
            reference: {"test": True}
            for reference in author_recipe_grid.REFERENCES
        },
        recipe_contracts=author_recipe_grid._recipe_contracts(),
    )
    jobs = tuple(author_recipe_grid.iter_jobs(plan))
    assert plan["n_jobs"] == len(jobs) == 4_480
    assert Counter(job.reference for job in jobs) == {
        "reference.tcformer": 2_240,
        "reference.fbcnet": 2_240,
    }
    assert len({job.job_id for job in jobs}) == len(jobs)
    assert plan["track"] == author_recipe_grid.TRACK
    assert plan["common_recipe_track"] is False
    assert not set(plan["references"]).intersection(
        full_grid.COMMON_ARCHITECTURES
    )
    assert plan["identity_contract"] == {
        "namespace": "registry_stable_id_author_recipe_adapted",
        "stable_ids": list(author_recipe_grid.REFERENCES),
        "bare_common_architecture_aliases_forbidden": [
            "tcformer",
            "fbcnet",
        ],
    }
    with pytest.raises(ValueError, match="exactly"):
        author_recipe_grid.assemble_plan(
            dataset_contracts=contracts,
            references=("tcformer", "fbcnet"),
            seeds=author_recipe_grid.FORMAL_SEEDS,
            cache_identity=caches,
            split_identity=splits,
            source_identity={"runner.py": "b" * 64},
            environment_identity={"python": "test"},
            reference_provenance={
                "reference.tcformer": {},
                "reference.fbcnet": {},
            },
            recipe_contracts=author_recipe_grid._recipe_contracts(),
        )


def test_plan_source_closure_freezes_analyzer_and_registry_decisions() -> None:
    required = {
        "src/benchmark/__init__.py",
        "src/benchmark/author_recipe_grid.py",
        "src/benchmark/author_recipe_analysis.py",
        "src/benchmark/model_registry.py",
            "src/benchmark/reference_training.py",
            "src/benchmark/baselines.py",
            "src/benchmark/tcformer_source.py",
            "src/benchmark/models.py",
        "src/benchmark/training.py",
        "src/benchmark/data.py",
        "src/benchmark/config.py",
        "src/benchmark/full_grid.py",
        "pyproject.toml",
        "uv.lock",
    }
    assert set(author_recipe_grid.SOURCE_FILES) == required
    contract = author_recipe_grid._analysis_contract(
        {"python": "test", "torch": {"cuda_device_count": 4}}
    )
    analysis_path = Path(author_recipe_analysis.__file__).resolve()
    assert contract["source_identity"] == {
        "src/benchmark/author_recipe_analysis.py": (
            author_recipe_grid._sha256_file(analysis_path)
        )
    }
    assert contract["label_join"] == (
        "after_exact_quiescent_score_blind_grid_audit_only"
    )
    assert contract["common_track_comparability"] == (
        "separately_labelled_author_recipe_results_never_common_recipe_rows"
    )
    assert contract["decision_contract"] == (
        author_recipe_analysis.decision_contract()
    )
    assert contract["decision_contract"]["ece_bins"] == 15


def test_plan_power_cut_repair_is_exact_and_immutable(tmp_path: Path) -> None:
    plan = _tiny_plan()
    root = tmp_path / "run"
    root.mkdir()
    author_recipe_grid._write_json_exclusive(root / "plan.json", plan)
    repaired = author_recipe_grid.write_or_validate_plan(root, plan)
    assert repaired == plan
    assert author_recipe_grid.load_plan(root) == plan
    assert (root / "plan.json").stat().st_mode & 0o222 == 0
    drifted = copy.deepcopy(plan)
    drifted["executor"] = "drifted:executor"
    drifted["plan_sha256"] = author_recipe_grid.plan_sha256(drifted)
    with pytest.raises(author_recipe_grid.AuthorRecipeError, match="differs"):
        author_recipe_grid.write_or_validate_plan(root, drifted)


def test_claim_requires_both_idle_gpu_and_safe_disk(tmp_path: Path) -> None:
    plan = _tiny_plan()
    root = tmp_path / "run"
    author_recipe_grid.write_or_validate_plan(root, plan)
    job = next(author_recipe_grid.iter_jobs(plan))
    with pytest.raises(author_recipe_grid.AuthorRecipeError, match="requires"):
        author_recipe_grid.acquire_claim(root, plan, job)
    with pytest.raises(author_recipe_grid.DiskUnavailable, match="low-water"):
        author_recipe_grid.acquire_claim(
            root,
            plan,
            job,
            before_claim=lambda: full_grid.combine_resource_status(
                _safe_gpu(), _low_disk()
            ),
        )
    assert not author_recipe_grid._claim_path(root, job).exists()


def test_production_runner_has_no_low_disk_escape_hatch(
    tmp_path: Path,
) -> None:
    parser = author_recipe_grid.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "run",
                "--run-root",
                str(tmp_path / "run"),
                "--cache-root",
                str(tmp_path / "cache"),
                "--allow-low-disk",
            ]
        )

    with pytest.raises(full_grid.FullGridError, match="exact 50 GiB floor"):
        author_recipe_grid.worker_loop(
            run_root=tmp_path / "missing",
            cache_root=tmp_path / "cache",
            gpu="GPU-test",
            worker_index=0,
            recover_stale=True,
            retry_failed=False,
            max_attempts_per_job=1,
            poll_seconds=0.0,
            max_utilization_percent=10.0,
            max_foreign_memory_mib=1024.0,
            allow_busy_gpu=False,
            foreign_claim_timeout_seconds=0.0,
            executor=lambda **_: (
                {},
                np.empty(0, dtype=np.int64),
                np.empty((0, 0), dtype=np.float64),
            ),
            gpu_probe=_safe_gpu,
            disk_probe=_safe_disk,
            minimum_free_gib=49.0,
        )


def test_atomic_records_are_score_blind_and_track_separated(
    tmp_path: Path,
) -> None:
    plan = _tiny_plan()
    root = tmp_path / "run"
    author_recipe_grid.write_or_validate_plan(root, plan)
    job = next(author_recipe_grid.iter_jobs(plan))
    output = _publish(root, plan, job)
    payload = author_recipe_grid.validate_completion(root, plan, job)
    assert payload["probabilities"].shape == (2, 2)
    record = json.loads((output / "record.json").read_text())
    assert record["track"] == author_recipe_grid.TRACK
    assert record["common_recipe_track"] is False
    assert record["score_blind"] is True
    assert author_recipe_grid._recursive_forbidden_keys(record) == []
    with np.load(output / "predictions.npz", allow_pickle=False) as archive:
        assert set(archive.files) == {"rows", "probabilities"}
    os.chmod(output / "record.json", 0o600)
    with (output / "record.json").open("a", encoding="utf-8") as handle:
        handle.write("corruption")
    with pytest.raises(
        author_recipe_grid.AuthorRecipeError,
        match="invalid exact file set|checksums|writable file",
    ):
        author_recipe_grid.validate_completion(root, plan, job)


def test_commit_rejects_test_score_or_wrong_rows(tmp_path: Path) -> None:
    plan = _tiny_plan()
    root = tmp_path / "run"
    author_recipe_grid.write_or_validate_plan(root, plan)
    job = next(author_recipe_grid.iter_jobs(plan))
    claim = author_recipe_grid.acquire_claim(
        root, plan, job, before_claim=_safe_resources
    )
    try:
        scored = _fit_metadata(plan, job)
        scored["accuracy"] = 1.0
        with pytest.raises(
            author_recipe_grid.AuthorRecipeError, match="forbidden"
        ):
            author_recipe_grid.commit_job_output(
                root,
                plan,
                claim,
                metadata=scored,
                test_rows=np.asarray([4, 5], dtype=np.int64),
                probabilities=np.asarray(
                    [[0.8, 0.2], [0.1, 0.9]], dtype=np.float64
                ),
            )
        with pytest.raises(
            author_recipe_grid.AuthorRecipeError,
            match="prediction arrays",
        ):
            author_recipe_grid.commit_job_output(
                root,
                plan,
                claim,
                metadata=_fit_metadata(plan, job),
                test_rows=np.asarray([3, 5], dtype=np.int64),
                probabilities=np.asarray(
                    [[0.8, 0.2], [0.1, 0.9]], dtype=np.float64
                ),
            )
    finally:
        author_recipe_grid.release_claim(claim)


def test_stale_claim_and_partial_are_recovered_after_power_cut(
    tmp_path: Path,
) -> None:
    plan = _tiny_plan()
    root = tmp_path / "run"
    author_recipe_grid.write_or_validate_plan(root, plan)
    job = next(author_recipe_grid.iter_jobs(plan))
    claim_path = author_recipe_grid._claim_path(root, job)
    author_recipe_grid._write_json_exclusive(
        claim_path,
        {
            "schema": author_recipe_grid.CLAIM_SCHEMA,
            "created_at": author_recipe_grid._utc_now(),
            "plan_sha256": plan["plan_sha256"],
            "job_id": job.job_id,
            "job": job.identity(),
            "nonce": "dead",
            "owner": {
                "host": author_recipe_grid.socket.gethostname(),
                "pid": os.getpid(),
                "boot_id": "different-boot",
                "start_ticks": "different-process",
            },
            "resource_guard": {},
        },
    )
    partial = root / "partials" / f"{job.job_id}.dead.partial"
    partial.mkdir(parents=True)
    claim = author_recipe_grid.acquire_claim(
        root,
        plan,
        job,
        recover_stale=True,
        before_claim=_safe_resources,
    )
    try:
        assert tuple((root / "quarantine" / "claims").iterdir())
        assert tuple((root / "quarantine" / "partials").iterdir())
    finally:
        author_recipe_grid.release_claim(claim)


def test_valid_completion_recovers_post_rename_stale_claim(
    tmp_path: Path,
) -> None:
    plan = _tiny_plan()
    root = tmp_path / "run"
    author_recipe_grid.write_or_validate_plan(root, plan)
    jobs = tuple(author_recipe_grid.iter_jobs(plan))
    for job in jobs:
        _publish(root, plan, job)
    job = jobs[0]
    claim_path = author_recipe_grid._claim_path(root, job)
    author_recipe_grid._write_json_exclusive(
        claim_path,
        {
            "schema": author_recipe_grid.CLAIM_SCHEMA,
            "created_at": author_recipe_grid._utc_now(),
            "plan_sha256": plan["plan_sha256"],
            "job_id": job.job_id,
            "job": job.identity(),
            "nonce": "interrupted-after-rename",
            "owner": {
                "host": author_recipe_grid.socket.gethostname(),
                "pid": os.getpid(),
                "boot_id": "old-boot",
                "start_ticks": "old-process",
            },
            "resource_guard": {},
        },
    )
    partial = root / "partials" / f"{job.job_id}.interrupted.partial"
    partial.mkdir(parents=True)
    recovered = author_recipe_grid.recover_completed_job_auxiliary_state(
        root, plan, job
    )
    assert len(recovered) == 2
    assert not claim_path.exists()
    assert not partial.exists()
    assert author_recipe_grid.audit_grid(
        root, plan
    )["exact_cartesian_complete"] is True


def test_injected_worker_executes_resumes_and_audits_exactly(
    tmp_path: Path,
) -> None:
    plan = _tiny_plan()
    root = tmp_path / "run"
    author_recipe_grid.write_or_validate_plan(root, plan)
    calls: list[str] = []

    def executor(**kwargs: object) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
        job = kwargs["job"]
        calls.append(job.job_id)
        return (
            _fit_metadata(plan, job),
            np.asarray([4, 5], dtype=np.int64),
            np.asarray([[0.8, 0.2], [0.1, 0.9]], dtype=np.float64),
        )

    code = author_recipe_grid.worker_loop(
        run_root=root,
        cache_root=tmp_path / "cache",
        gpu="GPU-test",
        worker_index=0,
        recover_stale=True,
        retry_failed=False,
        max_attempts_per_job=1,
        poll_seconds=0.0,
        max_utilization_percent=10.0,
        max_foreign_memory_mib=1024.0,
        allow_busy_gpu=False,
        foreign_claim_timeout_seconds=0.0,
        executor=executor,
        gpu_probe=_safe_gpu,
        disk_probe=_safe_disk,
    )
    assert code == 0
    assert len(calls) == 2
    audit = author_recipe_grid.audit_grid(root, plan)
    assert audit["exact_cartesian_complete"] is True
    assert audit["complete_jobs"] == 2
    # A resume validates and reuses both completions without executor calls.
    calls.clear()
    assert (
        author_recipe_grid.worker_loop(
            run_root=root,
            cache_root=tmp_path / "cache",
            gpu="GPU-test",
            worker_index=0,
            recover_stale=True,
            retry_failed=False,
            max_attempts_per_job=1,
            poll_seconds=0.0,
            max_utilization_percent=10.0,
            max_foreign_memory_mib=1024.0,
            allow_busy_gpu=False,
            foreign_claim_timeout_seconds=0.0,
            executor=executor,
            gpu_probe=_safe_gpu,
            disk_probe=_safe_disk,
        )
        == 0
    )
    assert calls == []


def test_label_join_occurs_only_after_exact_audit_and_publishes_scalars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _tiny_plan()
    root = tmp_path / "run"
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    author_recipe_grid.write_or_validate_plan(root, plan)
    jobs = tuple(author_recipe_grid.iter_jobs(plan))
    for job in jobs:
        _publish(root, plan, job)

    fake_cache = {
        "identity": copy.deepcopy(plan["cache_identity"]["tiny:s001"]),
        "x": np.zeros((6, 3, 8), dtype=np.float32),
        "y": np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64),
        "positions": np.eye(3, dtype=np.float32),
        "channel_names": np.asarray(["C3", "Cz", "C4"]),
        "sessions": np.asarray(["a"] * 6),
        "runs": np.asarray(["a"] * 6),
    }
    import benchmark.data as data_module

    monkeypatch.setattr(
        author_recipe_grid,
        "_load_subject_cache_secure",
        lambda *unused, **unused_keywords: copy.deepcopy(fake_cache),
    )
    monkeypatch.setattr(
        data_module,
        "split_indices",
        lambda *unused, **unused_keywords: (
            np.asarray([0, 1], dtype=np.int64),
            np.asarray([2, 3], dtype=np.int64),
            np.asarray([4, 5], dtype=np.int64),
        ),
    )
    monkeypatch.setattr(
        author_recipe_grid,
        "_environment_identity",
        lambda: {"python": "test"},
    )
    result = author_recipe_analysis.compute_analysis(
        run_root=root,
        cache_root=cache_root,
        require_formal_grid=False,
    )
    assert result.summary["track"] == author_recipe_grid.TRACK
    assert result.summary["common_recipe_track"] is False
    assert len(result.tables["overall_summary"]) == 2
    destination = author_recipe_analysis.publish_analysis(
        result,
        run_root=root,
        cache_root=cache_root,
        output_dir=tmp_path / "analysis",
        require_formal_grid=False,
    )
    assert (destination / "manifest.json").is_file()
    serialized = b"".join(
        path.read_bytes()
        for path in destination.iterdir()
        if path.is_file()
    ).lower()
    assert b'"probabilities"' not in serialized
    assert b'"labels"' not in serialized
    assert b"common-recipe" in (destination / "RESULTS.md").read_bytes()


def test_live_claim_blocks_analysis_before_any_cache_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _tiny_plan()
    root = tmp_path / "run"
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    author_recipe_grid.write_or_validate_plan(root, plan)
    jobs = tuple(author_recipe_grid.iter_jobs(plan))
    for job in jobs:
        _publish(root, plan, job)
    job = jobs[0]
    author_recipe_grid._write_json_exclusive(
        author_recipe_grid._claim_path(root, job),
        {
            "schema": author_recipe_grid.CLAIM_SCHEMA,
            "created_at": author_recipe_grid._utc_now(),
            "plan_sha256": plan["plan_sha256"],
            "job_id": job.job_id,
            "job": job.identity(),
            "nonce": "live",
            "owner": author_recipe_grid._process_identity(),
            "resource_guard": _safe_resources().as_dict(),
        },
    )
    import benchmark.data as data_module

    cache_calls = 0

    def forbidden_cache_access(*args: object, **kwargs: object) -> object:
        del args, kwargs
        nonlocal cache_calls
        cache_calls += 1
        raise AssertionError("cache was opened before quiescent audit")

    monkeypatch.setattr(
        author_recipe_grid,
        "_load_subject_cache_secure",
        forbidden_cache_access,
    )
    with pytest.raises(
        author_recipe_analysis.AuthorRecipeAnalysisError,
        match="exact and quiescent",
    ):
        author_recipe_analysis.compute_analysis(
            run_root=root,
            cache_root=cache_root,
            require_formal_grid=False,
        )
    assert cache_calls == 0


@pytest.mark.parametrize(
    "mutation",
    (
        lambda record: record.__setitem__("unexpected", 1),
        lambda record: record["metadata"]["runtime"].__setitem__(
            "unexpected", 1
        ),
        lambda record: record["metadata"]["split"]["partitions"][
            "test"
        ].__setitem__("unexpected", 1),
    ),
)
def test_resealed_record_extras_fail_exact_recursive_schemas(
    tmp_path: Path,
    mutation: object,
) -> None:
    plan = _tiny_plan()
    root = tmp_path / "run"
    author_recipe_grid.write_or_validate_plan(root, plan)
    job = next(author_recipe_grid.iter_jobs(plan))
    output = _publish(root, plan, job)
    _rewrite_record_and_completion(output, mutation)
    with pytest.raises(
        author_recipe_grid.AuthorRecipeError, match="exact schema"
    ):
        author_recipe_grid.validate_completion(root, plan, job)


def test_completion_receipt_rejects_extra_fields_even_when_canonical(
    tmp_path: Path,
) -> None:
    plan = _tiny_plan()
    root = tmp_path / "run"
    author_recipe_grid.write_or_validate_plan(root, plan)
    job = next(author_recipe_grid.iter_jobs(plan))
    output = _publish(root, plan, job)
    path = output / "completion.json"
    value = json.loads(path.read_text())
    value["unexpected"] = "resealed"
    os.chmod(path, 0o600)
    path.write_bytes(author_recipe_grid._canonical_bytes(value) + b"\n")
    os.chmod(path, 0o444)
    with pytest.raises(
        author_recipe_grid.AuthorRecipeError, match="exact schema"
    ):
        author_recipe_grid.validate_completion(root, plan, job)


@pytest.mark.parametrize(
    "alias",
    ("outcomes", "truth", "ground_truth", "y_true", "y_test"),
)
def test_score_blind_aliases_are_forbidden_recursively(
    tmp_path: Path,
    alias: str,
) -> None:
    plan = _tiny_plan()
    root = tmp_path / alias
    author_recipe_grid.write_or_validate_plan(root, plan)
    job = next(author_recipe_grid.iter_jobs(plan))
    claim = author_recipe_grid.acquire_claim(
        root, plan, job, before_claim=_safe_resources
    )
    try:
        metadata = _fit_metadata(plan, job)
        metadata["runtime"][alias] = [0, 1]
        with pytest.raises(
            author_recipe_grid.AuthorRecipeError, match="forbidden"
        ):
            author_recipe_grid.commit_job_output(
                root,
                plan,
                claim,
                metadata=metadata,
                test_rows=np.asarray([4, 5], dtype=np.int64),
                probabilities=np.asarray(
                    [[0.8, 0.2], [0.1, 0.9]], dtype=np.float64
                ),
            )
    finally:
        author_recipe_grid.release_claim(claim)


def test_fbcnet_transition_threshold_and_origin_are_durable(
    tmp_path: Path,
) -> None:
    plan = _tiny_plan()
    root = tmp_path / "run"
    author_recipe_grid.write_or_validate_plan(root, plan)
    job = next(
        job
        for job in author_recipe_grid.iter_jobs(plan)
        if job.reference == "reference.fbcnet"
    )
    _publish(root, plan, job)
    payload = author_recipe_grid.validate_completion(root, plan, job)
    fit = payload["record"]["metadata"]["fit"]
    assert fit["stage1_train_loss_threshold"] == 0.5
    assert (
        fit["stage1_threshold_origin"]
        == "last_stage1_epoch_before_best_restore"
    )
    assert fit["stage1_threshold_origin_epoch"] == 0


def test_plan_symlink_is_rejected_without_following_it(tmp_path: Path) -> None:
    plan = _tiny_plan()
    root = tmp_path / "run"
    author_recipe_grid.write_or_validate_plan(root, plan)
    path = root / "plan.json"
    target = root / "plan-target.json"
    os.rename(path, target)
    path.symlink_to(target.name)
    with pytest.raises(
        author_recipe_grid.AuthorRecipeError, match="symlink|regular"
    ):
        author_recipe_grid.load_plan(root)


def test_fifo_nodes_are_audited_without_opening_or_blocking(
    tmp_path: Path,
) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFO creation is not available")
    plan = _tiny_plan()
    root = tmp_path / "run"
    author_recipe_grid.write_or_validate_plan(root, plan)
    for job in author_recipe_grid.iter_jobs(plan):
        _publish(root, plan, job)
    assert author_recipe_grid.audit_grid(
        root, plan
    )["exact_cartesian_complete"] is True

    root_fifo = root / "unknown-root-fifo"
    os.mkfifo(root_fifo)
    audit = author_recipe_grid.audit_grid(root, plan)
    assert audit["exact_cartesian_complete"] is False
    assert audit["unknown_filesystem_paths"] == 1
    root_fifo.unlink()

    record_fifo = root / "records" / "unknown-fifo"
    os.mkfifo(record_fifo)
    audit = author_recipe_grid.audit_grid(root, plan)
    assert audit["exact_cartesian_complete"] is False
    assert audit["extra_files"] == 1
    record_fifo.unlink()

    claim_fifo = root / "claims" / "ff" / "unknown-fifo"
    claim_fifo.parent.mkdir(parents=True, exist_ok=True)
    os.mkfifo(claim_fifo)
    audit = author_recipe_grid.audit_grid(root, plan)
    assert audit["exact_cartesian_complete"] is False
    assert audit["unknown_claims"] >= 1
    assert "claims/ff/unknown-fifo" in audit["details"][
        "unknown_claim_paths"
    ]
    claim_fifo.unlink()

    partial_fifo = root / "partials" / "unknown-fifo"
    os.mkfifo(partial_fifo)
    audit = author_recipe_grid.audit_grid(root, plan)
    assert audit["exact_cartesian_complete"] is False
    assert audit["partials"] == 1
    partial_fifo.unlink()


@pytest.mark.parametrize("replacement", ("symlink", "regular_inode"))
def test_claim_release_rejects_path_and_inode_swaps(
    tmp_path: Path,
    replacement: str,
) -> None:
    plan = _tiny_plan()
    root = tmp_path / replacement
    author_recipe_grid.write_or_validate_plan(root, plan)
    job = next(author_recipe_grid.iter_jobs(plan))
    claim = author_recipe_grid.acquire_claim(
        root, plan, job, before_claim=_safe_resources
    )
    original = claim.path.parent / f".{claim.path.name}.original"
    os.rename(claim.path, original)
    if replacement == "symlink":
        target = claim.path.parent / "unrelated-target.json"
        target.write_text("{}\n", encoding="utf-8")
        claim.path.symlink_to(target.name)
    else:
        author_recipe_grid._write_bytes_exclusive(
            claim.path,
            original.read_bytes(),
            root=root,
        )
    with pytest.raises(
        author_recipe_grid.AuthorRecipeError,
        match="inode|symlink|regular",
    ):
        author_recipe_grid.release_claim(claim)
    assert original.is_file()
    if replacement == "symlink":
        assert target.read_text(encoding="utf-8") == "{}\n"
    os.unlink(claim.path)
    os.rename(original, claim.path)
    author_recipe_grid.release_claim(claim)


def test_formal_executor_cannot_be_self_consistently_substituted(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit):
        author_recipe_grid.build_parser().parse_args(
            [
                "run",
                "--run-root",
                str(tmp_path / "run"),
                "--cache-root",
                str(tmp_path / "cache"),
                "--executor",
                "attacker.module:executor",
            ]
        )
    with pytest.raises(
        author_recipe_grid.AuthorRecipeError, match="default executor"
    ):
        _formal_roster_synthetic_plan(executor="attacker.module:executor")

    plan = _formal_roster_synthetic_plan()
    plan["executor"] = "attacker.module:executor"
    plan["plan_sha256"] = author_recipe_grid.plan_sha256(plan)
    root = tmp_path / "formal-plan"
    root.mkdir()
    author_recipe_grid._write_json_exclusive(
        root / "plan.json", plan, root=root
    )
    author_recipe_grid._write_bytes_exclusive(
        root / "plan.sha256",
        f"{plan['plan_sha256']}\n".encode("ascii"),
        root=root,
    )
    with pytest.raises(
        author_recipe_grid.AuthorRecipeError, match="default executor"
    ):
        author_recipe_grid.load_plan(root)
    with pytest.raises(
        author_recipe_grid.AuthorRecipeError, match="default executor"
    ):
        author_recipe_grid.verify_runtime_identity(
            _tiny_plan(), cache_root=tmp_path / "unused"
        )
    valid_root = tmp_path / "formal-worker"
    author_recipe_grid.write_or_validate_plan(
        valid_root, _formal_roster_synthetic_plan()
    )
    with pytest.raises(
        author_recipe_grid.AuthorRecipeError, match="inject"
    ):
        author_recipe_grid.worker_loop(
            run_root=valid_root,
            cache_root=tmp_path / "unused",
            gpu="GPU-test",
            worker_index=0,
            recover_stale=True,
            retry_failed=False,
            max_attempts_per_job=1,
            poll_seconds=0.0,
            max_utilization_percent=10.0,
            max_foreign_memory_mib=1024.0,
            allow_busy_gpu=False,
            foreign_claim_timeout_seconds=0.0,
            executor=lambda **_: (
                {},
                np.empty(0, dtype=np.int64),
                np.empty((0, 0), dtype=np.float64),
            ),
            gpu_probe=_safe_gpu,
            disk_probe=_safe_disk,
        )


@pytest.mark.parametrize("replacement", ("lost", "replacement_inode"))
def test_commit_rechecks_exact_live_claim_immediately_before_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement: str,
) -> None:
    plan = _tiny_plan()
    root = tmp_path / replacement
    author_recipe_grid.write_or_validate_plan(root, plan)
    job = next(author_recipe_grid.iter_jobs(plan))
    claim = author_recipe_grid.acquire_claim(
        root, plan, job, before_claim=_safe_resources
    )
    original_validate = author_recipe_grid._validate_claim_payload
    calls = 0
    preserved = claim.path.parent / f".{claim.path.name}.preserved"

    def lose_claim(
        run_root: Path,
        current_plan: object,
        current_claim: author_recipe_grid.Claim,
    ) -> object:
        nonlocal calls
        calls += 1
        if calls == 2:
            if replacement == "lost":
                os.unlink(claim.path)
            else:
                os.rename(claim.path, preserved)
                author_recipe_grid._write_bytes_exclusive(
                    claim.path,
                    preserved.read_bytes(),
                    root=root,
                )
        return original_validate(run_root, current_plan, current_claim)

    monkeypatch.setattr(
        author_recipe_grid, "_validate_claim_payload", lose_claim
    )
    with pytest.raises(
        (FileNotFoundError, author_recipe_grid.AuthorRecipeError)
    ):
        author_recipe_grid.commit_job_output(
            root,
            plan,
            claim,
            metadata=_fit_metadata(plan, job),
            test_rows=np.asarray([4, 5], dtype=np.int64),
            probabilities=np.asarray(
                [[0.8, 0.2], [0.1, 0.9]], dtype=np.float64
            ),
        )
    assert not os.path.lexists(author_recipe_grid._record_directory(root, job))
    assert any((root / "quarantine" / "partials").iterdir())
    monkeypatch.setattr(
        author_recipe_grid, "_validate_claim_payload", original_validate
    )
    if replacement == "replacement_inode":
        os.unlink(claim.path)
        os.rename(preserved, claim.path)
    author_recipe_grid.release_claim(claim)


@pytest.mark.parametrize(
    ("free_gib", "minimum_free_gib"),
    ((1.0, 1.0), (49.0, 50.0)),
)
def test_acquire_claim_enforces_frozen_fifty_gib_probe_floor(
    tmp_path: Path,
    free_gib: float,
    minimum_free_gib: float,
) -> None:
    plan = _tiny_plan()
    root = tmp_path / f"floor-{free_gib}"
    author_recipe_grid.write_or_validate_plan(root, plan)
    job = next(author_recipe_grid.iter_jobs(plan))
    disk = full_grid.DiskStatus(
        safe=True,
        path="/synthetic",
        free_bytes=int(free_gib * 1024**3),
        total_bytes=200 * 1024**3,
        free_gib=free_gib,
        minimum_free_gib=minimum_free_gib,
        reason="forged safe probe",
    )
    with pytest.raises(
        author_recipe_grid.AuthorRecipeError, match="below its floor"
    ):
        author_recipe_grid.acquire_claim(
            root,
            plan,
            job,
            before_claim=lambda: full_grid.combine_resource_status(
                _safe_gpu(), disk
            ),
        )
    assert not os.path.lexists(author_recipe_grid._claim_path(root, job))


def test_power_cut_release_tombstone_is_safely_recovered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _tiny_plan()
    root = tmp_path / "released-tombstone"
    author_recipe_grid.write_or_validate_plan(root, plan)
    job = next(author_recipe_grid.iter_jobs(plan))
    claim = author_recipe_grid.acquire_claim(
        root, plan, job, before_claim=_safe_resources
    )
    author_recipe_grid.commit_job_output(
        root,
        plan,
        claim,
        metadata=_fit_metadata(plan, job),
        test_rows=np.asarray([4, 5], dtype=np.int64),
        probabilities=np.asarray(
            [[0.8, 0.2], [0.1, 0.9]], dtype=np.float64
        ),
    )
    original_quarantine = author_recipe_grid._quarantine

    def power_cut(
        run_root: Path,
        path: Path,
        *,
        category: str,
        reason: str,
        expected_identity: tuple[int, int],
    ) -> Path:
        if ".released-" in path.name:
            raise OSError("simulated power cut")
        return original_quarantine(
            run_root,
            path,
            category=category,
            reason=reason,
            expected_identity=expected_identity,
        )

    monkeypatch.setattr(author_recipe_grid, "_quarantine", power_cut)
    with pytest.raises(OSError, match="power cut"):
        author_recipe_grid.release_claim(claim)
    monkeypatch.setattr(
        author_recipe_grid, "_quarantine", original_quarantine
    )
    tombstones = tuple(
        entry
        for entry in claim.path.parent.iterdir()
        if ".released-" in entry.name
    )
    assert len(tombstones) == 1
    recovered = author_recipe_grid.recover_completed_job_auxiliary_state(
        root, plan, job
    )
    assert recovered
    assert not tombstones[0].exists()
    assert any((root / "quarantine" / "claims").iterdir())
    author_recipe_grid.validate_completion(root, plan, job)


def test_publication_lock_stale_recovery_and_idempotent_exact_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, root, cache_root = _completed_tiny_grid(tmp_path, monkeypatch)
    result = author_recipe_analysis.compute_analysis(
        run_root=root,
        cache_root=cache_root,
        require_formal_grid=False,
    )
    destination = tmp_path / "idempotent-analysis"
    lock = tmp_path / ".idempotent-analysis.author-analysis.lock"
    author_recipe_grid._write_json_exclusive(
        lock,
        {
            "schema": author_recipe_analysis.PUBLICATION_LOCK_SCHEMA,
            "created_at": author_recipe_grid._utc_now(),
            "plan_sha256": plan["plan_sha256"],
            "destination": str(destination),
            "nonce": "a" * 32,
            "owner": {
                "host": author_recipe_grid.socket.gethostname(),
                "pid": 2**31 - 1,
                "boot_id": "dead-boot",
                "start_ticks": "dead-start",
            },
        },
        root=tmp_path,
    )
    published = author_recipe_analysis.publish_analysis(
        result,
        run_root=root,
        cache_root=cache_root,
        output_dir=destination,
        require_formal_grid=False,
    )
    inode = (published.stat().st_dev, published.stat().st_ino)
    quarantine = (
        tmp_path / ".idempotent-analysis.author-analysis-quarantine"
    )
    assert any("stale" in entry.name for entry in quarantine.iterdir())
    assert (
        author_recipe_analysis.publish_analysis(
            result,
            run_root=root,
            cache_root=cache_root,
            output_dir=destination,
            require_formal_grid=False,
        )
        == destination
    )
    assert (destination.stat().st_dev, destination.stat().st_ino) == inode
    author_recipe_analysis._validate_published_directory(
        destination, result
    )
    live_destination = tmp_path / "live-lock-analysis"
    live_lock = tmp_path / ".live-lock-analysis.author-analysis.lock"
    author_recipe_grid._write_json_exclusive(
        live_lock,
        {
            "schema": author_recipe_analysis.PUBLICATION_LOCK_SCHEMA,
            "created_at": author_recipe_grid._utc_now(),
            "plan_sha256": "f" * 64,
            "destination": str(live_destination),
            "nonce": "b" * 32,
            "owner": author_recipe_grid._process_identity(),
        },
        root=tmp_path,
    )
    with pytest.raises(
        author_recipe_analysis.AuthorRecipeAnalysisError,
        match="live publication lock",
    ):
        author_recipe_analysis.publish_analysis(
            result,
            run_root=root,
            cache_root=cache_root,
            output_dir=live_destination,
            require_formal_grid=False,
        )
    assert live_lock.exists()


@pytest.mark.parametrize("insertion", ("partial", "claim"))
def test_run_root_fence_blocks_workers_and_inserted_state_blocks_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    insertion: str,
) -> None:
    plan, root, cache_root = _completed_tiny_grid(tmp_path, monkeypatch)
    result = author_recipe_analysis.compute_analysis(
        run_root=root,
        cache_root=cache_root,
        require_formal_grid=False,
    )
    foreign_fence_path = (
        root / author_recipe_grid.PUBLICATION_FENCE_FILENAME
    )
    author_recipe_grid._write_json_exclusive(
        foreign_fence_path,
        {
            "schema": author_recipe_grid.PUBLICATION_FENCE_SCHEMA,
            "created_at": author_recipe_grid._utc_now(),
            "plan_sha256": "f" * 64,
            "nonce": "c" * 32,
            "owner": author_recipe_grid._process_identity(),
            "destination": str(tmp_path / "foreign-plan-output"),
        },
        root=root,
    )
    with pytest.raises(
        author_recipe_grid.ClaimUnavailable, match="live"
    ):
        author_recipe_grid.acquire_claim(
            root,
            plan,
            next(author_recipe_grid.iter_jobs(plan)),
            before_claim=_safe_resources,
        )
    assert foreign_fence_path.exists()
    fence_stat = os.lstat(foreign_fence_path)
    author_recipe_grid._quarantine(
        root,
        foreign_fence_path,
        category="publication-fences",
        reason="test-cleanup",
        expected_identity=(
            int(fence_stat.st_dev),
            int(fence_stat.st_ino),
        ),
    )
    fence = author_recipe_grid.acquire_publication_fence(
        root, plan, destination=tmp_path / "held"
    )
    try:
        with pytest.raises(
            author_recipe_grid.ClaimUnavailable, match="fenced"
        ):
            author_recipe_grid.acquire_claim(
                root,
                plan,
                next(author_recipe_grid.iter_jobs(plan)),
                before_claim=_safe_resources,
            )
    finally:
        author_recipe_grid.release_publication_fence(fence, plan)

    original_close = (
        author_recipe_analysis._close_publication_input_window
    )
    calls = 0

    def insert_partial(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 2:
            if insertion == "partial":
                partial_root = author_recipe_grid._safe_mkdir(
                    root / "partials", root=root
                )
                author_recipe_grid._safe_mkdir(
                    partial_root / "injected.partial",
                    root=root,
                    exist_ok=False,
                )
            else:
                job = next(author_recipe_grid.iter_jobs(plan))
                author_recipe_grid._write_json_exclusive(
                    author_recipe_grid._claim_path(root, job),
                    {
                        "schema": author_recipe_grid.CLAIM_SCHEMA,
                        "created_at": author_recipe_grid._utc_now(),
                        "plan_sha256": plan["plan_sha256"],
                        "job_id": job.job_id,
                        "job": job.identity(),
                        "nonce": "d" * 32,
                        "owner": author_recipe_grid._process_identity(),
                        "resource_guard": _safe_resources().as_dict(),
                    },
                    root=root,
                )
        return original_close(*args, **kwargs)

    monkeypatch.setattr(
        author_recipe_analysis,
        "_close_publication_input_window",
        insert_partial,
    )
    output = tmp_path / "fenced-analysis"
    with pytest.raises(
        author_recipe_analysis.AuthorRecipeAnalysisError,
        match="exact and quiescent",
    ):
        author_recipe_analysis.publish_analysis(
            result,
            run_root=root,
            cache_root=cache_root,
            output_dir=output,
            require_formal_grid=False,
        )
    assert calls == 2
    assert not output.exists()
    assert not os.path.lexists(
        root / author_recipe_grid.PUBLICATION_FENCE_FILENAME
    )


@pytest.mark.parametrize("leaf_kind", ("fifo", "symlink", "hardlink"))
def test_cache_leaf_is_nofollow_regular_single_link_snapshot(
    tmp_path: Path,
    leaf_kind: str,
) -> None:
    import benchmark.data as data_module

    cache_root = tmp_path / f"cache-{leaf_kind}"
    path = data_module._cache_path(
        cache_root,
        "local_exp4",
        1,
        data_module.DEFAULT_MONTAGE_PROFILE,
    )
    path.parent.mkdir(parents=True)
    target = tmp_path / f"target-{leaf_kind}.npz"
    target.write_bytes(b"not an EEG cache")
    if leaf_kind == "fifo":
        os.mkfifo(path)
    elif leaf_kind == "symlink":
        path.symlink_to(target)
    else:
        os.link(target, path)
    started = time.monotonic()
    with pytest.raises(author_recipe_grid.AuthorRecipeError):
        author_recipe_grid._load_subject_cache_secure(
            "local_exp4", 1, cache_root=cache_root
        )
    assert time.monotonic() - started < 2.0


def test_ece_bins_are_frozen_in_metrics_compute_and_cli(
    tmp_path: Path,
) -> None:
    probabilities = np.asarray(
        [[0.8, 0.2], [0.1, 0.9]], dtype=np.float64
    )
    with pytest.raises(
        author_recipe_analysis.AuthorRecipeAnalysisError,
        match="non-frozen",
    ):
        author_recipe_analysis.classification_metrics(
            np.asarray([0, 1], dtype=np.int64),
            probabilities,
            n_classes=2,
            ece_bins=14,
        )
    with pytest.raises(ValueError, match="frozen"):
        author_recipe_analysis.compute_analysis(
            run_root=tmp_path / "missing",
            cache_root=tmp_path / "missing-cache",
            ece_bins=14,
            require_formal_grid=False,
        )
    with pytest.raises(SystemExit):
        author_recipe_analysis.build_parser().parse_args(
            [
                "--run-root",
                "run",
                "--cache-root",
                "cache",
                "--output-dir",
                "out",
                "--ece-bins",
                "14",
            ]
        )


def test_resealed_result_mutation_fails_fresh_locked_recomputation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, root, cache_root = _completed_tiny_grid(tmp_path, monkeypatch)
    result = author_recipe_analysis.compute_analysis(
        run_root=root,
        cache_root=cache_root,
        require_formal_grid=False,
    )
    result.tables["job_metrics"][0]["accuracy"] = 0.25
    result._seal = author_recipe_analysis._result_seal(result)
    with pytest.raises(
        author_recipe_analysis.AuthorRecipeAnalysisError,
        match="fresh publication-time recomputation",
    ):
        author_recipe_analysis.publish_analysis(
            result,
            run_root=root,
            cache_root=cache_root,
            output_dir=tmp_path / "tampered-analysis",
            require_formal_grid=False,
        )
    assert not (tmp_path / "tampered-analysis").exists()


def test_formal_publication_rechecks_runtime_inside_owned_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, root, cache_root = _completed_tiny_grid(tmp_path, monkeypatch)
    result = author_recipe_analysis.compute_analysis(
        run_root=root,
        cache_root=cache_root,
        require_formal_grid=False,
    )
    original_audit = author_recipe_analysis.audit_exact_grid

    def tiny_audit(
        run_root: object,
        *,
        require_formal_grid: bool = True,
        publication_fence: object = None,
    ) -> object:
        del require_formal_grid
        return original_audit(
            run_root,
            require_formal_grid=False,
            publication_fence=publication_fence,
        )

    monkeypatch.setattr(
        author_recipe_analysis, "audit_exact_grid", tiny_audit
    )
    output = tmp_path / "formal-analysis"
    lock = tmp_path / ".formal-analysis.author-analysis.lock"
    calls: list[Path] = []

    def verify(plan: object, *, cache_root: Path, **unused: object) -> None:
        del plan, unused
        assert os.path.lexists(lock)
        calls.append(cache_root)

    monkeypatch.setattr(
        author_recipe_grid, "verify_runtime_identity", verify
    )
    destination = author_recipe_analysis.publish_analysis(
        result,
        run_root=root,
        cache_root=cache_root,
        output_dir=output,
        require_formal_grid=True,
    )
    assert destination == output
    assert len(calls) >= 3


def test_publication_rejects_symlinked_output_ancestor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, root, cache_root = _completed_tiny_grid(tmp_path, monkeypatch)
    result = author_recipe_analysis.compute_analysis(
        run_root=root,
        cache_root=cache_root,
        require_formal_grid=False,
    )
    real = tmp_path / "real-output-parent"
    real.mkdir()
    alias = tmp_path / "aliased-output-parent"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(
        author_recipe_analysis.AuthorRecipeAnalysisError, match="unsafe ancestor"
    ):
        author_recipe_analysis.publish_analysis(
            result,
            run_root=root,
            cache_root=cache_root,
            output_dir=alias / "analysis",
            require_formal_grid=False,
        )


def test_native_no_replace_allows_one_concurrent_directory_winner(
    tmp_path: Path,
) -> None:
    root = tmp_path / "atomic-layout"
    source_a = root / "partials" / "a.partial"
    source_b = root / "partials" / "b.partial"
    destination = root / "records" / "winner"
    source_a.mkdir(parents=True)
    source_b.mkdir(parents=True)
    destination.parent.mkdir(parents=True)
    (source_a / "identity").write_text("a", encoding="utf-8")
    (source_b / "identity").write_text("b", encoding="utf-8")
    identities = {
        source.name: (
            int(source.stat().st_dev),
            int(source.stat().st_ino),
        )
        for source in (source_a, source_b)
    }
    barrier = threading.Barrier(2)
    outcomes: list[tuple[str, str]] = []

    def publish(source: Path) -> None:
        barrier.wait()
        try:
            author_recipe_grid._rename_leaf_nofollow(
                source,
                destination,
                root=root,
                expected_source_identity=identities[source.name],
            )
        except FileExistsError:
            outcomes.append((source.name, "lost"))
        else:
            outcomes.append((source.name, "won"))

    threads = [
        threading.Thread(target=publish, args=(source,))
        for source in (source_a, source_b)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5.0)
    assert all(not thread.is_alive() for thread in threads)
    assert sorted(status for _, status in outcomes) == ["lost", "won"]
    winner = next(name for name, status in outcomes if status == "won")
    loser = next(name for name, status in outcomes if status == "lost")
    assert (destination / "identity").read_text(encoding="utf-8") == winner[0]
    loser_path = source_a if loser == source_a.name else source_b
    assert (
        int(loser_path.stat().st_dev),
        int(loser_path.stat().st_ino),
    ) == identities[loser]


def test_completion_snapshot_allows_concurrent_sibling_result_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real sibling publication must not invalidate a held result package."""

    plan = _tiny_plan(seeds=(7, 8))
    root = tmp_path / "run"
    author_recipe_grid.write_or_validate_plan(root, plan)
    jobs = tuple(author_recipe_grid.iter_jobs(plan))
    first, sibling = jobs[0], jobs[1]
    assert first.reference == sibling.reference
    _publish(root, plan, first)
    start = threading.Event()
    finished = threading.Event()
    failure: list[BaseException] = []

    def publish_sibling() -> None:
        start.wait(timeout=5.0)
        try:
            _publish(root, plan, sibling)
        except BaseException as error:
            failure.append(error)
        finally:
            finished.set()

    worker = threading.Thread(target=publish_sibling)
    worker.start()
    main_thread = threading.get_ident()
    original_pread = author_recipe_grid.os.pread
    triggered = False

    def create_sibling_during_read(
        descriptor: int, size: int, offset: int
    ) -> bytes:
        nonlocal triggered
        if threading.get_ident() == main_thread and not triggered:
            triggered = True
            start.set()
            assert finished.wait(timeout=5.0)
        return original_pread(descriptor, size, offset)

    monkeypatch.setattr(
        author_recipe_grid.os,
        "pread",
        create_sibling_during_read,
    )
    payload = author_recipe_grid.validate_completion(root, plan, first)
    worker.join(timeout=5.0)
    assert not worker.is_alive()
    assert failure == []
    assert triggered is True
    assert payload["record"]["job_id"] == first.job_id
    author_recipe_grid.validate_completion(root, plan, sibling)


@pytest.mark.parametrize("restore_original", (False, True))
def test_completion_snapshot_rejects_leaf_a_b_and_a_b_a(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    restore_original: bool,
) -> None:
    plan = _tiny_plan()
    root = tmp_path / "run"
    author_recipe_grid.write_or_validate_plan(root, plan)
    job = next(author_recipe_grid.iter_jobs(plan))
    output = _publish(root, plan, job)
    record = output / "record.json"
    saved = output / "record.saved"
    original_pread = author_recipe_grid.os.pread
    swapped = False

    def swap_during_snapshot(
        descriptor: int, size: int, offset: int
    ) -> bytes:
        nonlocal swapped
        if not swapped:
            swapped = True
            os.chmod(output, 0o700)
            os.rename(record, saved)
            record.write_bytes(b'{"replacement":true}\n')
            os.chmod(record, 0o444)
            if restore_original:
                os.unlink(record)
                os.rename(saved, record)
            os.chmod(output, 0o555)
        return original_pread(descriptor, size, offset)

    monkeypatch.setattr(
        author_recipe_grid.os, "pread", swap_during_snapshot
    )
    with pytest.raises(author_recipe_grid.AuthorRecipeError, match="changed"):
        author_recipe_grid.validate_completion(root, plan, job)


def test_completion_snapshot_holds_original_across_ancestor_a_b_a(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ancestor substitution cannot redirect any held directory/child FD."""

    plan = _tiny_plan()
    root = tmp_path / "run"
    author_recipe_grid.write_or_validate_plan(root, plan)
    job = next(author_recipe_grid.iter_jobs(plan))
    output = _publish(root, plan, job)
    ancestor = output.parent
    saved = ancestor.with_name(f"{ancestor.name}.saved")
    original_pread = author_recipe_grid.os.pread
    swapped = False

    def swap_ancestor(descriptor: int, size: int, offset: int) -> bytes:
        nonlocal swapped
        if not swapped:
            swapped = True
            os.rename(ancestor, saved)
            ancestor.mkdir()
            os.rmdir(ancestor)
            os.rename(saved, ancestor)
        return original_pread(descriptor, size, offset)

    monkeypatch.setattr(author_recipe_grid.os, "pread", swap_ancestor)
    payload = author_recipe_grid.validate_completion(root, plan, job)
    assert payload["record"]["job_id"] == job.job_id
    assert np.array_equal(
        payload["rows"], np.asarray([4, 5], dtype=np.int64)
    )


def test_completion_rejects_duplicate_raw_npz_members(
    tmp_path: Path,
) -> None:
    plan = _tiny_plan()
    root = tmp_path / "run"
    author_recipe_grid.write_or_validate_plan(root, plan)
    job = next(author_recipe_grid.iter_jobs(plan))
    output = _publish(root, plan, job)
    prediction = output / "predictions.npz"
    with zipfile.ZipFile(prediction, mode="r") as archive:
        rows_payload = archive.read("rows.npy")
        probabilities_payload = archive.read("probabilities.npy")
    duplicate = io.BytesIO()
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(
            duplicate, mode="w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            archive.writestr("rows.npy", rows_payload)
            archive.writestr("rows.npy", rows_payload)
            archive.writestr("probabilities.npy", probabilities_payload)
    os.chmod(prediction, 0o600)
    prediction.write_bytes(duplicate.getvalue())
    os.chmod(prediction, 0o444)
    completion_path = output / "completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["files"]["predictions.npz"] = (
        author_recipe_grid._sha256_file(prediction)
    )
    os.chmod(completion_path, 0o600)
    completion_path.write_bytes(
        author_recipe_grid._canonical_bytes(completion) + b"\n"
    )
    os.chmod(completion_path, 0o444)
    with pytest.raises(
        author_recipe_grid.AuthorRecipeError,
        match="duplicate raw members",
    ):
        author_recipe_grid.validate_completion(root, plan, job)


@pytest.mark.parametrize("alias_kind", ("symlink", "hardlink"))
def test_completion_and_audit_reject_aliased_artifact(
    tmp_path: Path,
    alias_kind: str,
) -> None:
    plan = _tiny_plan()
    root = tmp_path / "run"
    author_recipe_grid.write_or_validate_plan(root, plan)
    job = next(author_recipe_grid.iter_jobs(plan))
    output = _publish(root, plan, job)
    record = output / "record.json"
    saved = output / "record.saved"
    os.chmod(output, 0o700)
    os.rename(record, saved)
    if alias_kind == "symlink":
        record.symlink_to(saved.name)
    else:
        os.link(saved, record)
    os.chmod(output, 0o555)
    with pytest.raises(author_recipe_grid.AuthorRecipeError):
        author_recipe_grid.validate_completion(root, plan, job)
    audit = author_recipe_grid.audit_grid(root, plan)
    assert job.job_id in audit["details"]["corrupt"]
    assert audit["exact_cartesian_complete"] is False


def test_commit_move_then_raise_rolls_back_only_published_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _tiny_plan()
    root = tmp_path / "run"
    author_recipe_grid.write_or_validate_plan(root, plan)
    job = next(author_recipe_grid.iter_jobs(plan))
    claim = author_recipe_grid.acquire_claim(
        root, plan, job, before_claim=_safe_resources
    )
    destination = author_recipe_grid._record_directory(root, job)
    original_rename = author_recipe_grid._rename_leaf_nofollow

    def move_then_raise(
        source: Path,
        target: Path,
        **keywords: object,
    ) -> tuple[int, int]:
        identity = original_rename(source, target, **keywords)
        if target == destination:
            raise OSError("simulated post-rename fsync failure")
        return identity

    monkeypatch.setattr(
        author_recipe_grid, "_rename_leaf_nofollow", move_then_raise
    )
    try:
        with pytest.raises(OSError, match="post-rename fsync"):
            author_recipe_grid.commit_job_output(
                root,
                plan,
                claim,
                metadata=_fit_metadata(plan, job),
                test_rows=np.asarray([4, 5], dtype=np.int64),
                probabilities=np.asarray(
                    [[0.8, 0.2], [0.1, 0.9]], dtype=np.float64
                ),
            )
        assert not os.path.lexists(destination)
        residues = tuple((root / "quarantine" / "records").iterdir())
        assert len(residues) == 1
        assert stat.S_IMODE(residues[0].stat().st_mode) == 0o555
    finally:
        author_recipe_grid.release_claim(claim)


def test_commit_seal_then_raise_leaves_no_canonical_residue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _tiny_plan()
    root = tmp_path / "run"
    author_recipe_grid.write_or_validate_plan(root, plan)
    job = next(author_recipe_grid.iter_jobs(plan))
    claim = author_recipe_grid.acquire_claim(
        root, plan, job, before_claim=_safe_resources
    )
    destination = author_recipe_grid._record_directory(root, job)
    original_seal = author_recipe_grid._seal_directory_read_only

    def seal_then_raise(path: Path, **keywords: object) -> tuple[int, ...]:
        fingerprint = original_seal(path, **keywords)
        if path == destination:
            raise OSError("simulated completion seal fsync failure")
        return fingerprint

    monkeypatch.setattr(
        author_recipe_grid, "_seal_directory_read_only", seal_then_raise
    )
    try:
        with pytest.raises(OSError, match="seal fsync"):
            author_recipe_grid.commit_job_output(
                root,
                plan,
                claim,
                metadata=_fit_metadata(plan, job),
                test_rows=np.asarray([4, 5], dtype=np.int64),
                probabilities=np.asarray(
                    [[0.8, 0.2], [0.1, 0.9]], dtype=np.float64
                ),
            )
        assert not os.path.lexists(destination)
        residue = next((root / "quarantine" / "records").iterdir())
        assert stat.S_IMODE(residue.stat().st_mode) == 0o555
    finally:
        monkeypatch.setattr(
            author_recipe_grid,
            "_seal_directory_read_only",
            original_seal,
        )
        author_recipe_grid.release_claim(claim)


def test_post_publish_rebind_failure_leaves_no_writable_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _tiny_plan()
    root = tmp_path / "run"
    author_recipe_grid.write_or_validate_plan(root, plan)
    job = next(author_recipe_grid.iter_jobs(plan))
    claim = author_recipe_grid.acquire_claim(
        root, plan, job, before_claim=_safe_resources
    )
    destination = author_recipe_grid._record_directory(root, job)
    original_rebind = author_recipe_grid._rebind_authoritative_inputs
    calls = 0

    def fail_second_rebind(**keywords: object) -> object:
        nonlocal calls
        calls += 1
        result = original_rebind(**keywords)
        if calls == 2:
            raise author_recipe_grid.AuthorRecipeError(
                "simulated authority drift"
            )
        return result

    monkeypatch.setattr(
        author_recipe_grid,
        "_rebind_authoritative_inputs",
        fail_second_rebind,
    )
    try:
        with pytest.raises(
            author_recipe_grid.AuthorRecipeError,
            match="authority drift",
        ):
            author_recipe_grid.commit_job_output(
                root,
                plan,
                claim,
                metadata=_fit_metadata(plan, job),
                test_rows=np.asarray([4, 5], dtype=np.int64),
                probabilities=np.asarray(
                    [[0.8, 0.2], [0.1, 0.9]], dtype=np.float64
                ),
            )
        assert calls == 2
        assert not os.path.lexists(destination)
        for residue in (root / "quarantine" / "records").iterdir():
            assert stat.S_IMODE(residue.stat().st_mode) == 0o555
    finally:
        monkeypatch.setattr(
            author_recipe_grid,
            "_rebind_authoritative_inputs",
            original_rebind,
        )
        author_recipe_grid.release_claim(claim)


def test_claim_release_preserves_replacement_and_rolls_back_owned_record(
    tmp_path: Path,
) -> None:
    plan = _tiny_plan()
    root = tmp_path / "run"
    author_recipe_grid.write_or_validate_plan(root, plan)
    job = next(author_recipe_grid.iter_jobs(plan))
    claim = author_recipe_grid.acquire_claim(
        root, plan, job, before_claim=_safe_resources
    )
    destination = author_recipe_grid.commit_job_output(
        root,
        plan,
        claim,
        metadata=_fit_metadata(plan, job),
        test_rows=np.asarray([4, 5], dtype=np.int64),
        probabilities=np.asarray(
            [[0.8, 0.2], [0.1, 0.9]], dtype=np.float64
        ),
    )
    original_claim = claim.path.with_name(f".{claim.path.name}.original")
    claim_bytes = claim.path.read_bytes()
    os.rename(claim.path, original_claim)
    author_recipe_grid._write_bytes_exclusive(
        claim.path, claim_bytes, root=root
    )
    replacement = claim.path.stat()
    with pytest.raises(
        author_recipe_grid.AuthorRecipeError,
        match="inode differs|authority|replaced",
    ):
        author_recipe_grid.release_claim(claim)
    assert (
        int(claim.path.stat().st_dev),
        int(claim.path.stat().st_ino),
    ) == (int(replacement.st_dev), int(replacement.st_ino))
    assert not os.path.lexists(destination)


def test_analysis_move_then_raise_removes_exact_published_residue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, root, cache_root = _completed_tiny_grid(tmp_path, monkeypatch)
    result = author_recipe_analysis.compute_analysis(
        run_root=root,
        cache_root=cache_root,
        require_formal_grid=False,
    )
    destination = tmp_path / "move-then-raise-analysis"
    original_rename = author_recipe_grid._rename_leaf_nofollow

    def move_then_raise(
        source: Path,
        target: Path,
        **keywords: object,
    ) -> tuple[int, int]:
        identity = original_rename(source, target, **keywords)
        if target == destination:
            raise OSError("simulated analysis post-rename failure")
        return identity

    monkeypatch.setattr(
        author_recipe_grid, "_rename_leaf_nofollow", move_then_raise
    )
    with pytest.raises(OSError, match="analysis post-rename"):
        author_recipe_analysis.publish_analysis(
            result,
            run_root=root,
            cache_root=cache_root,
            output_dir=destination,
            require_formal_grid=False,
        )
    assert not os.path.lexists(destination)
    quarantine = (
        tmp_path / ".move-then-raise-analysis.author-analysis-quarantine"
    )
    assert any(entry.is_dir() for entry in quarantine.iterdir())


def test_analysis_seal_then_raise_leaves_no_canonical_residue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, root, cache_root = _completed_tiny_grid(tmp_path, monkeypatch)
    result = author_recipe_analysis.compute_analysis(
        run_root=root,
        cache_root=cache_root,
        require_formal_grid=False,
    )
    destination = tmp_path / "analysis-seal-failure"
    original_seal = author_recipe_grid._seal_directory_read_only

    def seal_then_raise(path: Path, **keywords: object) -> tuple[int, ...]:
        fingerprint = original_seal(path, **keywords)
        if path == destination:
            raise OSError("simulated analysis seal fsync failure")
        return fingerprint

    monkeypatch.setattr(
        author_recipe_grid, "_seal_directory_read_only", seal_then_raise
    )
    with pytest.raises(OSError, match="analysis seal fsync"):
        author_recipe_analysis.publish_analysis(
            result,
            run_root=root,
            cache_root=cache_root,
            output_dir=destination,
            require_formal_grid=False,
        )
    assert not os.path.lexists(destination)
    quarantine = (
        tmp_path / ".analysis-seal-failure.author-analysis-quarantine"
    )
    residue = next(entry for entry in quarantine.iterdir() if entry.is_dir())
    assert stat.S_IMODE(residue.stat().st_mode) == 0o555


def test_analysis_holds_published_child_fds_through_final_rebind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, root, cache_root = _completed_tiny_grid(tmp_path, monkeypatch)
    result = author_recipe_analysis.compute_analysis(
        run_root=root,
        cache_root=cache_root,
        require_formal_grid=False,
    )
    destination = tmp_path / "held-analysis"
    original_rebind = (
        author_recipe_analysis._rebind_analysis_publication_authority
    )
    calls = 0

    def mutate_after_capture(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        value = original_rebind(*args, **kwargs)
        if calls == 2:
            artifact = destination / "analysis.json"
            saved = destination / "analysis.saved"
            os.chmod(destination, 0o700)
            os.rename(artifact, saved)
            artifact.write_bytes(b'{"replacement":true}\n')
            os.chmod(artifact, 0o444)
            os.chmod(destination, 0o555)
        return value

    monkeypatch.setattr(
        author_recipe_analysis,
        "_rebind_analysis_publication_authority",
        mutate_after_capture,
    )
    with pytest.raises(
        author_recipe_analysis.AuthorRecipeAnalysisError,
        match="artifact changed|directory binding changed|file set changed",
    ):
        author_recipe_analysis.publish_analysis(
            result,
            run_root=root,
            cache_root=cache_root,
            output_dir=destination,
            require_formal_grid=False,
        )
    assert calls == 2
    assert not os.path.lexists(destination)


def test_analysis_lock_release_preserves_replacement_and_rolls_back_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, root, cache_root = _completed_tiny_grid(tmp_path, monkeypatch)
    result = author_recipe_analysis.compute_analysis(
        run_root=root,
        cache_root=cache_root,
        require_formal_grid=False,
    )
    destination = tmp_path / "lock-replacement-analysis"
    lock = tmp_path / ".lock-replacement-analysis.author-analysis.lock"
    original_release = author_recipe_analysis._release_output_lock
    replacement_identity: list[tuple[int, int]] = []

    def replace_before_release(
        path: Path, **keywords: object
    ) -> None:
        original = path.with_name(f"{path.name}.original")
        payload = path.read_bytes()
        os.rename(path, original)
        author_recipe_grid._write_bytes_exclusive(
            path, payload, root=tmp_path
        )
        observed = path.stat()
        replacement_identity.append(
            (int(observed.st_dev), int(observed.st_ino))
        )
        original_release(path, **keywords)

    monkeypatch.setattr(
        author_recipe_analysis,
        "_release_output_lock",
        replace_before_release,
    )
    with pytest.raises(
        (author_recipe_grid.AuthorRecipeError,
         author_recipe_analysis.AuthorRecipeAnalysisError),
    ):
        author_recipe_analysis.publish_analysis(
            result,
            run_root=root,
            cache_root=cache_root,
            output_dir=destination,
            require_formal_grid=False,
        )
    assert replacement_identity
    assert (
        int(lock.stat().st_dev),
        int(lock.stat().st_ino),
    ) == replacement_identity[0]
    assert not os.path.lexists(destination)


def _write_uv_fixture(root: Path, dependencies: dict[str, str]) -> None:
    root.mkdir()
    dependency_lines = ",\n".join(
        f'    "{name}=={version}"'
        for name, version in dependencies.items()
    )
    (root / "pyproject.toml").write_text(
        "[project]\n"
        'name = "fixture"\n'
        'version = "0.1.0"\n'
        'requires-python = "==3.12.*"\n'
        "dependencies = [\n"
        f"{dependency_lines}\n"
        "]\n",
        encoding="utf-8",
    )
    packages = "\n".join(
        "[[package]]\n"
        f'name = "{name}"\n'
        f'version = "{version}"\n'
        'source = { registry = "https://pypi.org/simple" }\n'
        for name, version in dependencies.items()
    )
    (root / "uv.lock").write_text(
        'version = 1\nrevision = 3\nrequires-python = "==3.12.*"\n\n'
        + packages,
        encoding="utf-8",
    )


def test_uv_contract_requires_exact_pins_and_checks_lock_venv_and_extras(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dependencies = {
        name: f"1.0.{index}"
        for index, name in enumerate(
            sorted(author_recipe_grid.REQUIRED_DIRECT_DEPENDENCIES)
        )
    }
    root = tmp_path / "uv-project"
    _write_uv_fixture(root, dependencies)
    parsed = author_recipe_grid._validate_uv_project_contract(
        root, check_runtime=False
    )
    assert parsed["required_exact_pins"] == dict(sorted(dependencies.items()))

    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    (root / "pyproject.toml").write_text(
        text.replace(
            f'torch=={dependencies["torch"]}',
            f'torch>={dependencies["torch"]}',
        ),
        encoding="utf-8",
    )
    with pytest.raises(
        author_recipe_grid.AuthorRecipeError, match="exact == pin"
    ):
        author_recipe_grid._validate_uv_project_contract(
            root, check_runtime=False
        )
    (root / "pyproject.toml").write_text(text, encoding="utf-8")

    venv = root / ".venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").symlink_to("/usr/bin/python3")
    monkeypatch.setattr(author_recipe_grid.sys, "prefix", str(venv))
    monkeypatch.setattr(
        author_recipe_grid.sys, "executable", str(venv / "bin" / "python")
    )
    commands: list[tuple[str, ...]] = []

    def successful_run(command: object, **unused: object) -> object:
        del unused
        commands.append(tuple(command))
        return SimpleNamespace(stdout="audited", stderr="")

    monkeypatch.setattr(
        author_recipe_grid.subprocess, "run", successful_run
    )
    identity = author_recipe_grid._validate_uv_project_contract(root)
    assert identity["lock_current"] is True
    assert identity["environment_exact_no_extras"] is True
    assert identity["pip_check"] is True
    assert any(command[:3] == ("uv", "lock", "--check") for command in commands)
    assert any(
        command[:3] == ("uv", "sync", "--check")
        and "--frozen" in command
        and "--no-dev" in command
        for command in commands
    )
    assert any(command[:3] == ("uv", "pip", "check") for command in commands)

    def extra_environment(command: object, **unused: object) -> object:
        del unused
        if tuple(command)[:3] == ("uv", "sync", "--check"):
            raise subprocess.CalledProcessError(1, command)
        return SimpleNamespace(stdout="ok", stderr="")

    monkeypatch.setattr(
        author_recipe_grid.subprocess, "run", extra_environment
    )
    with pytest.raises(
        author_recipe_grid.AuthorRecipeError, match="failed closed"
    ):
        author_recipe_grid._validate_uv_project_contract(root)
