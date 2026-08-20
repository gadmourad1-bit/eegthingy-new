from __future__ import annotations

import copy
import io
import json
import os
import shutil
import time
import zipfile
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmark import procedure_grid
from benchmark.config import (
    LOCAL_EXP4_SUBJECT_RUNS,
    coordinate_contract_for_dataset,
    dataset_spec,
    preprocessing_for_dataset,
)


def _forensic_root(run_root: Path) -> Path:
    return run_root.parent / f".{run_root.name}{procedure_grid.FORENSIC_ROOT_SUFFIX}"


def _split_identity(
    *,
    dataset: str = "tiny",
    subject: int = 1,
    fold: int = 0,
    digest: str = "a" * 64,
) -> dict[str, object]:
    return procedure_grid._one_split_identity(
        dataset=dataset,
        subject=subject,
        fold=fold,
        cache_array_sha256=digest,
        trial_count=8,
        train_rows=[0, 1],
        validation_rows=[2, 3],
        test_rows=[4, 5, 6, 7],
    )


def test_row_identity_preserves_established_digest_contract() -> None:
    rows = np.asarray([0, 1, 2, 3], dtype=np.int64)
    expected = "a0d130a0485bed50fd3d930c445f954be83116ab904e712ef29612ce2e13d20a"
    assert procedure_grid._rows_sha256(rows) == expected
    assert _split_identity()["partitions"]["source"] == {
        "count": 4,
        "rows_sha256": expected,
    }


def _config_inventory(
    stable_id: str = "architecture.cameo",
) -> dict[str, object]:
    inventory = procedure_grid._config_inventory()
    inventory["available"] = [
        value for value in inventory["available"] if value["stable_id"] == stable_id
    ]
    return inventory


def _tiny_plan(
    *,
    stable_id: str = "architecture.cameo",
    seeds: tuple[int, ...] = (7,),
) -> dict[str, object]:
    digest = "a" * 64
    return procedure_grid.assemble_plan(
        dataset_contracts={
            "tiny": {
                "subjects": [1],
                "folds": [0],
                "n_classes": 2,
                "events": ["left", "right"],
                "protocol": "synthetic",
                "preprocessing": {"sfreq_hz": 128.0, "n_times": 256},
            }
        },
        cache_identity={
            "tiny:s001": {
                "array_sha256": digest,
                "trial_count": 8,
                "n_channels": 3,
                "shape": [8, 3, 256],
                "channels": ["C3", "Cz", "C4"],
                "positions_manifest": procedure_grid._array_manifest(
                    np.zeros((3, 3), dtype=np.float32)
                ),
            }
        },
        split_identity={"tiny:s001:f00": _split_identity(digest=digest)},
        source_identity={"runner.py": "c" * 64},
        environment_identity={"uv_version": "uv test"},
        config_inventory=_config_inventory(stable_id),
        procedure_ids=(stable_id,),
        seeds=seeds,
        cpu_threads=1,
        publishable=False,
    )


def _formal_semantic_plan() -> dict[str, object]:
    contracts = procedure_grid._dataset_contracts()
    digest = "a" * 64
    cache_identity: dict[str, object] = {}
    split_identity: dict[str, object] = {}
    for dataset, contract in contracts.items():
        for subject in contract["subjects"]:
            coordinates = coordinate_contract_for_dataset(dataset, "harmonized")
            channels = list(coordinates["ordered_channels"])
            cache = {
                "dataset": procedure_grid._json_ready(asdict(dataset_spec(dataset))),
                "subject": subject,
                "montage_profile": "harmonized",
                "preprocessing": preprocessing_for_dataset(dataset),
                "coordinates": coordinates,
                "shape": [
                    8,
                    len(channels),
                    contract["preprocessing"]["n_times"],
                ],
                "channels": channels,
                "array_sha256": digest,
                "trial_count": 8,
                "n_channels": len(channels),
                "positions_manifest": {
                    "shape": [len(channels), 3],
                    "dtype": np.dtype(np.float32).str,
                    "sha256": "b" * 64,
                },
            }
            if dataset == "local_exp4":
                cache["source_manifest"] = [
                    {
                        "subject": subject,
                        "run": run,
                        "filename": (
                            f"exp4_subject{subject}_training_{run}_mi_raw.fif"
                        ),
                        "size_bytes": 1,
                        "sha256": "d" * 64,
                    }
                    for run in LOCAL_EXP4_SUBJECT_RUNS[subject]
                ]
            cache_identity[procedure_grid._subject_key(dataset, subject)] = cache
            for fold in contract["folds"]:
                split_identity[procedure_grid._split_key(dataset, subject, fold)] = (
                    procedure_grid._one_split_identity(
                        dataset=dataset,
                        subject=subject,
                        fold=fold,
                        cache_array_sha256=digest,
                        trial_count=8,
                        train_rows=[0, 1],
                        validation_rows=[2, 3],
                        test_rows=[4, 5, 6, 7],
                    )
                )
    gpu_uuid = "GPU-00000000-0000-0000-0000-000000000001"
    environment = {
        "schema": "eeg-mi-procedure-environment-v1",
        "python": "test",
        "executable": "/test/python",
        "prefix": "/test",
        "base_prefix": "/base",
        "virtual_environment": "/test",
        "platform": "test",
        "uv_version": "uv test",
        "packages": [],
        "packages_sha256": procedure_grid._sha256_bytes(
            procedure_grid._canonical_bytes([])
        ),
        "torch": {
            "version": "test",
            "cuda_runtime": "test",
            "cuda_available": True,
        },
        "nvidia": {
            "available": True,
            "gpus": [
                {
                    "index": "0",
                    "uuid": gpu_uuid,
                    "pci_bus_id": "0000:00:00.0",
                    "name": "test",
                    "driver_version": "test",
                }
            ],
        },
        "cublas_workspace_config": procedure_grid.REQUIRED_CUBLAS_WORKSPACE_CONFIG,
        "thread_environment": {
            name: "4" for name in procedure_grid.THREAD_ENVIRONMENT_VARIABLES
        },
    }
    return procedure_grid.assemble_plan(
        dataset_contracts=contracts,
        cache_identity=cache_identity,
        split_identity=split_identity,
        source_identity=procedure_grid._source_identity(),
        environment_identity=environment,
        config_inventory=procedure_grid._config_inventory(),
    )


def _metadata(
    plan: dict[str, object],
    job: procedure_grid.Job,
) -> dict[str, object]:
    expected_config = plan["config_inventory"]["available"][0]
    digests = iter("3456789abcdef" * 8)

    def manifest(shape):
        return {
            "shape": list(shape),
            "dtype": "<f4",
            "sha256": next(digests) * 64,
        }

    preprocessing = {
        name: manifest([1])
        for name in (
            "raw_mean",
            "raw_std",
            "anchor_log_reference",
            "anchor_scaler_mean",
            "anchor_scaler_scale",
            "anchor_logistic_coef",
            "anchor_logistic_intercept",
        )
    }
    tangent_features = 24

    def pair(count):
        return {
            "original": manifest([count, tangent_features]),
            "reflected": manifest([count, tangent_features]),
        }

    if job.stable_id == "architecture.cameo":
        decision = {"selected_mixture": "geo", "selected_rho": 0.0}
    elif job.stable_id == "architecture.orbit_v3":
        decision = {"selected_candidate": "fused"}
    else:
        decision = {}

    return {
        "cache_array_sha256": "a" * 64,
        "channels": ["C3", "Cz", "C4"],
        "split": copy.deepcopy(plan["split_identity"]["tiny:s001:f00"]),
        "frozen_config": {
            "schema": "eeg-mi-local-procedure-config-v2",
            "version": 1,
            "model": job.stable_id,
            "config_file": expected_config["config_path"],
            "config_file_bytes": expected_config["config_bytes"],
            "config_file_sha256": expected_config["config_sha256"],
            "settings_sha256": expected_config["settings_sha256"],
            "config": copy.deepcopy(expected_config["settings"]),
            "runtime_overrides": {"seed": job.seed, "device": "cpu"},
        },
        "view_pipeline": {
            "schema": procedure_grid.VIEW_SCHEMA,
            "mirror_index": [2, 1, 0],
            "raw": {
                "original": manifest([8, 3, 256]),
                "reflected": manifest([8, 3, 256]),
            },
            "four_band_spd": {
                "original": manifest([8, 4, 3, 3]),
                "reflected": manifest([8, 4, 3, 3]),
            },
            "selection_tangent": {
                "train": pair(2),
                "validation": pair(2),
            },
            "refit_tangent": {
                "source": pair(4),
                "test": pair(4),
            },
        },
        "fit": {
            "seed_installed_before_construction": True,
            "selected_epoch_index": 2,
            "selection_epochs_run": 3,
            "selection_decision": decision,
            "selection_start_reset_sha256": "1" * 64,
            "refit_start_reset_sha256": "1" * 64,
            "refit_state_sha256": "2" * 64,
            "reset_hash_exclusions": list(
                procedure_grid.RESET_EXCLUSIONS_BY_PROCEDURE[
                    procedure_grid.PROCEDURE_BY_ID[job.stable_id].key
                ]
            ),
            "reset_verified": True,
            "selected_epoch_count": 3,
            "refit_epochs_run": 3,
            "parameter_count": 10,
            "trainable_parameter_count": 10,
            "selection_preprocessing": copy.deepcopy(preprocessing),
            "refit_preprocessing": copy.deepcopy(preprocessing),
        },
        "protocol": {
            "selection_preprocessing_rows": "train_only",
            "selection_optimization_rows": "train_only",
            "selection_decision_rows": "validation_only",
            "refit_preprocessing_rows": "train_plus_validation",
            "refit_optimization_rows": "train_plus_validation",
            "test_use": "single_predict_proba_call_only",
            "test_performance_computed": False,
            "transductive_calibration": False,
        },
        "timing_seconds": {
            "selection_fit": 0.1,
            "refit_fit": 0.2,
            "test_inference": 0.01,
            "job_total": 0.31,
        },
        "runtime": {
            "cpu_threads": 1,
            "torch_interop_threads": 1,
            "device": "cpu",
            "physical_gpu_uuid": None,
            "cuda_visible_devices": None,
            "thread_environment": {
                name: "1" for name in procedure_grid.THREAD_ENVIRONMENT_VARIABLES
            },
            "cuda_peak_memory_bytes": 0,
        },
    }


def _safe_resource(tmp_path: Path) -> dict[str, object]:
    return {
        "safe": True,
        "gpu": {
            "safe": True,
            "gpu": "test",
            "utilization_percent": 0.0,
            "memory_used_mib": 0.0,
            "own_compute_memory_mib": 0.0,
            "foreign_processes": [],
            "reason": "injected test GPU",
            "gpu_uuid": None,
            "pci_bus_id": None,
            "name": None,
        },
        "disk": {
            "safe": True,
            "path": str(tmp_path),
            "free_bytes": 100 * 1024**3,
            "total_bytes": 200 * 1024**3,
            "free_gib": 100.0,
            "minimum_free_gib": procedure_grid.DEFAULT_MIN_FREE_GIB,
            "reason": "injected test disk",
        },
        "reason": "injected safe resource",
    }


def _formal_tiny_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    Path,
    dict[str, object],
    procedure_grid.Job,
    procedure_grid.GPULease,
    dict[str, object],
    dict[str, object],
]:
    """Create a minimal formal-path fixture without weakening production code."""

    tmp_path.mkdir(parents=True, exist_ok=True)
    project = tmp_path / "project"
    project.mkdir()
    run = tmp_path / "run"
    plan = procedure_grid.write_or_validate_plan(run, _tiny_plan())
    gpu_uuid = "GPU-00000000-0000-0000-0000-000000000001"
    plan["publication_mode"] = procedure_grid.FORMAL_PUBLICATION_MODE
    plan["execution_config"]["device"] = "cuda:0"
    plan["execution_config"]["physical_gpu_uuid_roster"] = [gpu_uuid]
    plan["execution_config"]["minimum_free_gib"] = procedure_grid.DEFAULT_MIN_FREE_GIB
    monkeypatch.setattr(procedure_grid, "PROJECT_ROOT", project)
    monkeypatch.setattr(
        procedure_grid, "validate_plan_semantics", lambda *args, **kwargs: None
    )
    lease = procedure_grid.acquire_gpu_lease(run, plan, gpu_uuid)
    resource = _safe_resource(run)
    resource["gpu"].update(
        {
            "gpu": gpu_uuid,
            "gpu_uuid": gpu_uuid,
            "pci_bus_id": "00000000:01:00.0",
            "name": "Synthetic GPU",
        }
    )
    job = next(procedure_grid.iter_jobs(plan))
    metadata = _metadata(plan, job)
    metadata["frozen_config"]["runtime_overrides"]["device"] = "cuda:0"
    metadata["runtime"].update(
        {
            "device": "cuda:0",
            "physical_gpu_uuid": gpu_uuid,
            "cuda_visible_devices": gpu_uuid,
        }
    )
    return run, plan, job, lease, resource, metadata


def test_formal_grid_is_exact_8780_jobs_and_excludes_blocked_orbits() -> None:
    contracts: dict[str, object] = {}
    for dataset in procedure_grid.BINARY_DATASETS:
        spec = dataset_spec(dataset)
        contracts[dataset] = {
            "subjects": list(spec.development_subjects),
            "folds": list(procedure_grid.DATASET_FOLDS[dataset]),
            "n_classes": 2,
            "events": list(spec.events),
            "protocol": spec.protocol,
            "preprocessing": {
                "sfreq_hz": 128.0,
                "n_times": 256 if dataset == "local_exp4" else 320,
            },
        }
    plan = procedure_grid.assemble_plan(
        dataset_contracts=contracts,
        cache_identity={},
        split_identity={},
        source_identity={},
        environment_identity={},
        config_inventory={"available": [], "blocked": []},
    )
    jobs = tuple(procedure_grid.iter_jobs(plan))
    assert plan["n_jobs"] == len(jobs) == 8_780
    assert len({job.job_id for job in jobs}) == len(jobs)
    blocked = {value["stable_id"] for value in procedure_grid.BLOCKED_ORBIT_PROVENANCE}
    assert not blocked.intersection(job.stable_id for job in jobs)
    assert {job.stable_id for job in jobs} == {
        value.stable_id for value in procedure_grid.PROCEDURES
    }


def test_exact_formal_semantic_validator_rejects_rehashed_contract_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _formal_semantic_plan()
    procedure_grid.validate_plan_semantics(plan, allow_unsealed=True)
    monkeypatch.setattr(procedure_grid, "_require_formal_release_runtime", lambda: None)
    sealed = procedure_grid.write_or_validate_plan(tmp_path, plan)
    assert procedure_grid.load_plan(tmp_path) == sealed
    attacks = (
        ("blocked_procedure_ids", []),
        ("executor", "attacker:execute"),
        ("view_contract", {"schema": "false"}),
    )
    for key, value in attacks:
        attacked = copy.deepcopy(sealed)
        attacked[key] = value
        attacked["plan_sha256"] = procedure_grid.plan_sha256(attacked)
        with pytest.raises(procedure_grid.ProcedureGridError):
            procedure_grid.validate_plan_semantics(attacked)
    attacked = copy.deepcopy(sealed)
    attacked["execution_config"]["minimum_free_gib"] = 0.0
    attacked["plan_sha256"] = procedure_grid.plan_sha256(attacked)
    with pytest.raises(procedure_grid.ProcedureGridError):
        procedure_grid.validate_plan_semantics(attacked)


def test_orbit_inventory_is_read_only_and_fails_closed(monkeypatch) -> None:
    opened: list[Path] = []
    real_loader = procedure_grid._strict_json_load

    def recording_loader(path: Path) -> dict[str, object]:
        opened.append(path)
        return real_loader(path)

    monkeypatch.setattr(procedure_grid, "_strict_json_load", recording_loader)
    inventory = procedure_grid._config_inventory()
    assert len(inventory["available"]) == 4
    assert {value["stable_id"] for value in inventory["blocked"]} == {
        "architecture.orbit_v1",
        "architecture.orbit_v2",
        "architecture.orbit_v4",
        "architecture.orbit_v5",
    }
    assert all("results" not in path.parts for path in opened)
    by_id = {value["stable_id"]: value for value in inventory["blocked"]}
    assert (
        by_id["architecture.orbit_v1"][
            "deterministic_config_only_extraction_without_inference"
        ]
        is False
    )
    assert by_id["architecture.orbit_v1"]["missing_current_config_fields"] == [
        "normalization"
    ]
    for stable_id in (
        "architecture.orbit_v2",
        "architecture.orbit_v4",
        "architecture.orbit_v5",
    ):
        assert by_id[stable_id]["complete_config_object_in_historical_artifact"] is True
        assert by_id[stable_id]["status"] == ("blocked_historical_post_hoc")
    with pytest.raises(ValueError):
        procedure_grid.assemble_plan(
            dataset_contracts={
                "tiny": {
                    "subjects": [1],
                    "folds": [0],
                    "n_classes": 2,
                }
            },
            cache_identity={},
            split_identity={},
            source_identity={},
            environment_identity={},
            config_inventory=inventory,
            procedure_ids=("architecture.orbit_v2",),
            seeds=(7,),
        )


def test_available_inventory_matches_typed_runtime_config_identity() -> None:
    import torch  # noqa: F401  # fail, rather than skip, without runtime
    from benchmark import local_outer_refit_benchmark as legacy

    inventory = {
        value["stable_id"]: value
        for value in procedure_grid._config_inventory()["available"]
    }
    assert {
        key: tuple(value) for key, value in legacy._RESET_EXCLUSIONS.items()
    } == dict(procedure_grid.RESET_EXCLUSIONS_BY_PROCEDURE)
    for key, spec in legacy._PROCEDURE_CONFIGS.items():
        assert (
            spec.architecture_fields
            == (procedure_grid.PROCEDURE_CONFIG_FIELDS[key]["architecture"])
        )
        assert (
            spec.training_fields
            == (procedure_grid.PROCEDURE_CONFIG_FIELDS[key]["training"])
        )
    for spec in procedure_grid.PROCEDURES:
        _config, identity = legacy._load_frozen_config(spec.key, seed=7, device="cpu")
        expected = inventory[spec.stable_id]
        assert identity["model"] == spec.stable_id
        assert identity["config_file"] == expected["config_path"]
        assert identity["config_file_bytes"] == expected["config_bytes"]
        assert identity["config_file_sha256"] == expected["config_sha256"]
        assert identity["settings_sha256"] == expected["settings_sha256"]
        assert identity["runtime_overrides"] == {
            "seed": 7,
            "device": "cpu",
        }


@pytest.mark.parametrize(
    "stable_id", [value.stable_id for value in procedure_grid.PROCEDURES]
)
def test_metadata_validator_preserves_each_method_selection_contract(
    stable_id: str,
) -> None:
    plan = _tiny_plan(stable_id=stable_id)
    job = next(procedure_grid.iter_jobs(plan))
    procedure_grid._validate_metadata(plan, job, _metadata(plan, job))


@pytest.mark.parametrize("n_times", [256, 320])
def test_paired_view_transform_is_deterministic_spd_and_involutive(
    n_times: int,
) -> None:
    generator = np.random.default_rng(23)
    raw = generator.normal(size=(5, 5, n_times)).astype(np.float32)
    channels = ("C3", "Cz", "C4", "P1", "P2")
    first = procedure_grid.derive_paired_views(raw, channel_names=channels)
    second = procedure_grid.derive_paired_views(raw.copy(), channel_names=channels)
    for name in (
        "raw",
        "raw_reflected",
        "covariance",
        "covariance_reflected",
        "mirror_index",
    ):
        assert np.array_equal(first[name], second[name])
    mirror = first["mirror_index"]
    assert np.array_equal(mirror[mirror], np.arange(len(channels)))
    assert np.array_equal(first["raw_reflected"][:, mirror, :], first["raw"])
    assert np.array_equal(
        first["covariance_reflected"][..., mirror, :][..., :, mirror],
        first["covariance"],
    )
    eigenvalues = np.linalg.eigvalsh(first["covariance"].astype(np.float64))
    assert np.all(eigenvalues > 0.0)
    assert first["covariance"].shape == (5, 4, 5, 5)


def test_plan_is_canonical_and_repairs_only_orphaned_digest(
    tmp_path: Path,
) -> None:
    plan = _tiny_plan()
    observed = procedure_grid.write_or_validate_plan(tmp_path, plan)
    assert observed["created_at"] != procedure_grid.PLAN_CREATED_AT_SENTINEL
    assert procedure_grid._plan_semantic_payload(observed) == (
        procedure_grid._plan_semantic_payload(plan)
    )
    plan = observed
    assert procedure_grid.load_plan(tmp_path) == plan
    (tmp_path / "plan.sha256").unlink()
    repaired = procedure_grid.load_or_repair_plan(tmp_path)
    assert repaired == plan
    os.chmod(tmp_path / "plan.sha256", 0o644)
    (tmp_path / "plan.sha256").write_text("0" * 64 + "\n")
    with pytest.raises(procedure_grid.ProcedureGridError):
        procedure_grid.load_plan(tmp_path)


def test_atomic_completion_is_score_blind_and_auditable(
    tmp_path: Path,
) -> None:
    plan = _tiny_plan()
    plan = procedure_grid.write_or_validate_plan(tmp_path, plan)
    job = next(procedure_grid.iter_jobs(plan))
    claim = procedure_grid.acquire_claim(
        tmp_path,
        plan,
        job,
        resource_guard=_safe_resource(tmp_path),
        recover_stale=False,
    )
    probabilities = np.asarray(
        [[0.8, 0.2], [0.2, 0.8], [0.7, 0.3], [0.3, 0.7]],
        dtype=np.float64,
    )
    procedure_grid.commit_job_output(
        tmp_path,
        plan,
        claim,
        metadata=_metadata(plan, job),
        test_rows=np.arange(4, 8, dtype=np.int64),
        probabilities=probabilities,
    )
    assert procedure_grid.audit_grid(tmp_path, plan)["complete"] is False
    procedure_grid.release_claim(claim)
    audit = procedure_grid.audit_grid(tmp_path, plan)
    assert audit["complete"] is True
    payload = procedure_grid.validate_completion(tmp_path, plan, job)
    directory = procedure_grid._record_directory(tmp_path, job)
    assert directory.stat().st_mode & 0o777 == 0o555
    assert set(path.name for path in directory.iterdir()) == (
        procedure_grid.FINAL_FILENAMES
    )
    serialized = json.dumps(payload["record"]).lower()
    assert '"labels"' not in serialized
    assert '"accuracy"' not in serialized
    assert np.array_equal(payload["probabilities"], probabilities)
    record = directory / "record.json"
    hardlink = tmp_path.parent / f"{tmp_path.name}-record-hardlink.json"
    os.link(record, hardlink)
    try:
        assert procedure_grid.audit_grid(tmp_path, plan)["corrupt_jobs"] == 1
    finally:
        hardlink.unlink()
    assert procedure_grid.audit_grid(tmp_path, plan)["complete"] is True
    os.chmod(record, 0o644)
    record.write_text(record.read_text() + " ")
    assert procedure_grid.audit_grid(tmp_path, plan)["corrupt_jobs"] == 1


def test_directory_seal_fsyncs_before_and_after_chmod(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = tmp_path / "record-stage"
    directory.mkdir()
    descriptor = procedure_grid._open_directory_absolute(directory)
    real_fsync = procedure_grid.os.fsync
    real_fchmod = procedure_grid.os.fchmod
    events: list[str] = []

    def tracked_fsync(selected: int) -> None:
        if selected == descriptor:
            events.append("fsync")
        real_fsync(selected)

    def tracked_fchmod(selected: int, mode: int) -> None:
        if selected == descriptor:
            events.append(f"chmod:{mode:o}")
        real_fchmod(selected, mode)

    monkeypatch.setattr(procedure_grid.os, "fsync", tracked_fsync)
    monkeypatch.setattr(procedure_grid.os, "fchmod", tracked_fchmod)
    try:
        procedure_grid._seal_directory_read_only(descriptor, directory)
    finally:
        os.close(descriptor)
    assert events == ["fsync", "chmod:555", "fsync"]


def test_completion_validation_rejects_directory_inode_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = procedure_grid.write_or_validate_plan(tmp_path, _tiny_plan())
    job = next(procedure_grid.iter_jobs(plan))
    claim = procedure_grid.acquire_claim(
        tmp_path,
        plan,
        job,
        resource_guard=_safe_resource(tmp_path),
        recover_stale=False,
    )
    procedure_grid.commit_job_output(
        tmp_path,
        plan,
        claim,
        metadata=_metadata(plan, job),
        test_rows=np.arange(4, 8, dtype=np.int64),
        probabilities=np.full((4, 2), 0.5, dtype=np.float64),
    )
    procedure_grid.release_claim(claim)
    directory = procedure_grid._record_directory(tmp_path, job)
    original = procedure_grid._read_unique_regular_at
    replaced = {"done": False}

    def read_then_replace(*args, **kwargs):
        payload = original(*args, **kwargs)
        if (
            kwargs.get("source", "").endswith("completion.json")
            and not replaced["done"]
        ):
            replaced["done"] = True
            hidden = directory.parent / f".{directory.name}.replaced"
            procedure_grid._atomic_rename_noreplace(directory, hidden)
            shutil.copytree(hidden, directory)
        return payload

    monkeypatch.setattr(procedure_grid, "_read_unique_regular_at", read_then_replace)
    with pytest.raises(procedure_grid.ProcedureGridError, match="inode"):
        procedure_grid.validate_completion(tmp_path, plan, job)
    assert replaced["done"] is True


def test_completion_validation_rejects_writable_record_directory(
    tmp_path: Path,
) -> None:
    plan = procedure_grid.write_or_validate_plan(tmp_path, _tiny_plan())
    job = next(procedure_grid.iter_jobs(plan))
    claim = procedure_grid.acquire_claim(
        tmp_path,
        plan,
        job,
        resource_guard=_safe_resource(tmp_path),
        recover_stale=False,
    )
    procedure_grid.commit_job_output(
        tmp_path,
        plan,
        claim,
        metadata=_metadata(plan, job),
        test_rows=np.arange(4, 8, dtype=np.int64),
        probabilities=np.full((4, 2), 0.5, dtype=np.float64),
    )
    procedure_grid.release_claim(claim)
    directory = procedure_grid._record_directory(tmp_path, job)
    os.chmod(directory, 0o755)
    with pytest.raises(procedure_grid.ProcedureGridError, match="0555"):
        procedure_grid.validate_completion(tmp_path, plan, job)


def test_completion_validation_rejects_duplicate_raw_npz_member(
    tmp_path: Path,
) -> None:
    plan = procedure_grid.write_or_validate_plan(tmp_path, _tiny_plan())
    job = next(procedure_grid.iter_jobs(plan))
    claim = procedure_grid.acquire_claim(
        tmp_path,
        plan,
        job,
        resource_guard=_safe_resource(tmp_path),
        recover_stale=False,
    )
    procedure_grid.commit_job_output(
        tmp_path,
        plan,
        claim,
        metadata=_metadata(plan, job),
        test_rows=np.arange(4, 8, dtype=np.int64),
        probabilities=np.full((4, 2), 0.5, dtype=np.float64),
    )
    procedure_grid.release_claim(claim)

    directory = procedure_grid._record_directory(tmp_path, job)
    prediction_path = directory / "predictions.npz"
    completion_path = directory / "completion.json"
    members: list[tuple[str, bytes, int]] = []
    with zipfile.ZipFile(io.BytesIO(prediction_path.read_bytes())) as source:
        for entry in source.infolist():
            members.append((entry.filename, source.read(entry), entry.compress_type))
    forged = io.BytesIO()
    with zipfile.ZipFile(forged, mode="w") as destination:
        for name, payload, compression in members:
            destination.writestr(name, payload, compress_type=compression)
        with pytest.warns(UserWarning, match="Duplicate name"):
            destination.writestr(
                members[0][0],
                members[0][1],
                compress_type=members[0][2],
            )

    os.chmod(prediction_path, 0o644)
    prediction_path.write_bytes(forged.getvalue())
    os.chmod(prediction_path, 0o444)
    completion = procedure_grid._strict_json_load(completion_path)
    completion["files"]["predictions.npz"] = procedure_grid._sha256_file(
        prediction_path
    )
    os.chmod(completion_path, 0o644)
    completion_path.write_bytes(procedure_grid._canonical_bytes(completion) + b"\n")
    os.chmod(completion_path, 0o444)

    with pytest.raises(
        procedure_grid.ProcedureGridError,
        match="exact ZIP member order",
    ):
        procedure_grid.validate_completion(tmp_path, plan, job)


def test_stale_claim_and_partial_are_recoverable_without_deletion(
    tmp_path: Path,
) -> None:
    plan = _tiny_plan()
    plan = procedure_grid.write_or_validate_plan(tmp_path, plan)
    job = next(procedure_grid.iter_jobs(plan))
    path = procedure_grid._claim_path(tmp_path, job)
    procedure_grid._write_json_exclusive(
        path,
        {
            "schema": procedure_grid.CLAIM_SCHEMA,
            "created_at": procedure_grid._utc_now(),
            "plan_sha256": plan["plan_sha256"],
            "job_id": job.job_id,
            "job": job.identity(),
            "nonce": "0" * 32,
            "owner": {
                "hostname": procedure_grid.socket.gethostname(),
                "pid": 999_999_999,
                "boot_id": procedure_grid._boot_id(),
                "process_start_token": "1",
            },
            "resource_guard": _safe_resource(tmp_path),
            "gpu_lease_receipt": None,
        },
    )
    partial = tmp_path / "partials" / f"{job.job_id}.old.partial"
    partial.mkdir(parents=True)
    (partial / "evidence").write_text("power cut")
    destination = procedure_grid._record_directory(tmp_path, job)
    destination.parent.mkdir(parents=True)
    adjacent = destination.parent / f".{destination.name}.old.partial"
    adjacent.mkdir()
    (adjacent / "evidence").write_text("power cut after staging")
    os.chmod(adjacent / "evidence", 0o444)
    os.chmod(adjacent, 0o555)
    claim = procedure_grid.acquire_claim(
        tmp_path,
        plan,
        job,
        resource_guard=_safe_resource(tmp_path),
        recover_stale=True,
    )
    recovered = procedure_grid._recover_job_partials(tmp_path, job)
    assert len(recovered) == 2
    assert list((_forensic_root(tmp_path) / "quarantine" / "stale_claims").iterdir())
    assert list(
        (_forensic_root(tmp_path) / "quarantine" / "power_cut_partials").iterdir()
    )
    procedure_grid.release_claim(claim)


def test_worker_recovers_claim_left_after_record_rename(
    tmp_path: Path,
) -> None:
    plan = _tiny_plan()
    plan = procedure_grid.write_or_validate_plan(tmp_path, plan)
    job = next(procedure_grid.iter_jobs(plan))
    resource = _safe_resource(tmp_path)
    live_claim = procedure_grid.acquire_claim(
        tmp_path,
        plan,
        job,
        resource_guard=resource,
        recover_stale=False,
    )
    procedure_grid.commit_job_output(
        tmp_path,
        plan,
        live_claim,
        metadata=_metadata(plan, job),
        test_rows=np.arange(4, 8, dtype=np.int64),
        probabilities=np.full((4, 2), 0.5, dtype=np.float64),
    )
    # Simulate a reboot after the destination rename but before claim release.
    procedure_grid.release_claim(live_claim)
    claim_path = procedure_grid._claim_path(tmp_path, job)
    procedure_grid._write_json_exclusive(
        claim_path,
        {
            "schema": procedure_grid.CLAIM_SCHEMA,
            "created_at": procedure_grid._utc_now(),
            "plan_sha256": plan["plan_sha256"],
            "job_id": job.job_id,
            "job": job.identity(),
            "nonce": "f" * 32,
            "owner": {
                "hostname": procedure_grid.socket.gethostname(),
                "pid": 999_999_999,
                "boot_id": procedure_grid._boot_id(),
                "process_start_token": "1",
            },
            "resource_guard": resource,
            "gpu_lease_receipt": None,
        },
    )
    code = procedure_grid.worker_loop(
        run_root=tmp_path,
        cache_root=tmp_path / "cache",
        gpu="test",
        device="cpu",
        worker_index=0,
        cpu_threads=1,
        recover_stale=True,
        executor=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("completed job must not execute again")
        ),
        resource_probe=lambda: resource,
        verify_identity=False,
    )
    assert code == 0
    assert procedure_grid.audit_grid(tmp_path, plan)["complete"] is True
    assert list(
        (_forensic_root(tmp_path) / "quarantine" / "post_rename_stale_claims").iterdir()
    )


def test_tiny_cache_fixture_binds_exact_source_geometry() -> None:
    plan = _tiny_plan()
    expected = plan["cache_identity"]["tiny:s001"]
    observed = procedure_grid._cache_identity_from_loaded(
        {
            "x": np.zeros((8, 3, 256), dtype=np.float32),
            "y": np.arange(8, dtype=np.int64) % 2,
            "positions": np.zeros((3, 3), dtype=np.float32),
            "channel_names": np.asarray(["C3", "Cz", "C4"]),
            "sessions": np.asarray(["s"] * 8),
            "runs": np.asarray(["r"] * 8),
            "identity": {
                key: copy.deepcopy(value)
                for key, value in expected.items()
                if key not in {"trial_count", "n_channels"}
            },
        }
    )
    assert observed == expected


def test_execute_job_uses_source_only_reset_then_one_test_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    plan = _tiny_plan()
    job = next(procedure_grid.iter_jobs(plan))
    labels = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    raw = np.zeros((8, 3, 256), dtype=np.float32)
    raw[:, 0, 0] = np.arange(8)
    data = {
        "x": raw,
        "y": labels,
        "sessions": np.asarray(["s"] * 8),
        "runs": np.asarray(["r"] * 8),
        "channel_names": np.asarray(["C3", "Cz", "C4"]),
        "positions": np.zeros((3, 3), dtype=np.float32),
        "identity": {"array_sha256": "a" * 64},
    }
    data["identity"].update(
        {
            key: value
            for key, value in plan["cache_identity"]["tiny:s001"].items()
            if key not in {"trial_count", "n_channels"}
        }
    )
    monkeypatch.setattr(
        "benchmark.data.load_subject_cache",
        lambda *args, **kwargs: copy.deepcopy(data),
    )
    monkeypatch.setattr(
        "benchmark.data.split_indices",
        lambda *args, **kwargs: (
            np.asarray([0, 1], dtype=np.int64),
            np.asarray([2, 3], dtype=np.int64),
            np.asarray([4, 5, 6, 7], dtype=np.int64),
        ),
    )
    covariance = np.repeat(
        np.eye(3, dtype=np.float32)[None, None, :, :],
        8 * 4,
        axis=0,
    ).reshape(8, 4, 3, 3)
    mirror = np.asarray([2, 1, 0], dtype=np.int64)
    monkeypatch.setattr(
        procedure_grid,
        "derive_paired_views",
        lambda *args, **kwargs: {
            "raw": raw.copy(),
            "raw_reflected": raw[:, mirror, :].copy(),
            "covariance": covariance.copy(),
            "covariance_reflected": covariance.copy(),
            "mirror_index": mirror.copy(),
        },
    )
    monkeypatch.setattr(
        procedure_grid, "_configure_determinism", lambda *args, **kwargs: None
    )

    from benchmark import local_outer_refit_benchmark as legacy

    config_identity = {
        "schema": "eeg-mi-local-procedure-config-v2",
        "version": 1,
        "model": "architecture.cameo",
        "config_file": "configs/local_procedures/cameo_v1.json",
        "config_file_bytes": 829,
        "config_file_sha256": (
            procedure_grid.PROCEDURE_BY_ID["architecture.cameo"].config_sha256
        ),
        "settings_sha256": "b" * 64,
        "config": {"epochs": 3},
        "runtime_overrides": {"seed": 7, "device": "cpu"},
    }
    monkeypatch.setattr(
        legacy,
        "_load_frozen_config",
        lambda *args, **kwargs: (SimpleNamespace(), config_identity),
    )

    class Anchor:
        def transform(self, values):
            return np.asarray(values).reshape(len(values), -1)

    class Selection:
        def __init__(self) -> None:
            self.anchor_ = Anchor()
            self.initial_model_state_ = {"w": torch.tensor([1.0])}

    calls = {"predict": 0, "refit": 0}

    class Refit(Selection):
        def predict_proba(self, raw_test, covariance_test):
            calls["predict"] += 1
            assert calls["refit"] == 1
            assert raw_test[:, 0, 0].tolist() == [4.0, 5.0, 6.0, 7.0]
            return np.asarray(
                [[0.6, 0.4], [0.4, 0.6], [0.7, 0.3], [0.3, 0.7]],
                dtype=np.float64,
            )

    selection = Selection()
    refit = Refit()
    state_hash = legacy._state_sha256(selection.initial_model_state_)

    def fake_selection(*args, **kwargs):
        assert kwargs["raw_train"][:, 0, 0].tolist() == [0.0, 1.0]
        assert kwargs["raw_validation"][:, 0, 0].tolist() == [2.0, 3.0]
        return selection, {
            "best_epoch_zero_based": 2,
            "selected_epoch_count": 3,
            "epochs_run": 4,
            "selection_decision": {
                "selected_mixture": "geo",
                "selected_rho": 0.0,
            },
            "fitted_preprocessing": _metadata(plan, job)["fit"][
                "selection_preprocessing"
            ],
        }

    def fake_refit(*args, **kwargs):
        calls["refit"] += 1
        assert kwargs["raw_source"][:, 0, 0].tolist() == [
            0.0,
            1.0,
            2.0,
            3.0,
        ]
        return refit, {
            "refit_start_reset_sha256": state_hash,
            "reset_verified": True,
            "epoch_count": 3,
            "model_state_sha256": "5" * 64,
            "parameter_count": 10,
            "trainable_parameter_count": 10,
            "fitted_preprocessing": _metadata(plan, job)["fit"]["refit_preprocessing"],
        }

    monkeypatch.setattr(legacy, "_selection_fit", fake_selection)
    monkeypatch.setattr(legacy, "_refit", fake_refit)
    metadata, rows, probabilities = procedure_grid.execute_procedure_job(
        job=job,
        plan=plan,
        cache_root=Path("/not/opened"),
        device="cpu",
        cpu_threads=1,
    )
    assert calls == {"predict": 1, "refit": 1}
    assert rows.tolist() == [4, 5, 6, 7]
    assert probabilities.shape == (4, 2)
    assert metadata["protocol"]["test_use"] == ("single_predict_proba_call_only")
    assert not procedure_grid._recursive_forbidden_keys(metadata)


def test_worker_uses_injected_executor_and_leaves_no_claim(
    tmp_path: Path,
) -> None:
    plan = _tiny_plan()
    plan = procedure_grid.write_or_validate_plan(tmp_path, plan)
    calls: list[str] = []

    def executor(**kwargs):
        job = kwargs["job"]
        calls.append(job.job_id)
        return (
            _metadata(plan, job),
            np.arange(4, 8, dtype=np.int64),
            np.full((4, 2), 0.5, dtype=np.float64),
        )

    code = procedure_grid.worker_loop(
        run_root=tmp_path,
        cache_root=tmp_path / "cache",
        gpu="test",
        device="cpu",
        worker_index=0,
        cpu_threads=1,
        recover_stale=True,
        executor=executor,
        resource_probe=lambda: _safe_resource(tmp_path),
        verify_identity=False,
    )
    assert code == 0
    assert len(calls) == 1
    assert procedure_grid.audit_grid(tmp_path, plan)["complete"] is True


def test_plan_reassembly_is_deterministic_and_repairs_real_cli_power_cut(
    tmp_path: Path,
) -> None:
    first = _tiny_plan()
    second = _tiny_plan()
    assert first == second
    sealed = procedure_grid.write_or_validate_plan(tmp_path, first)
    original_timestamp = sealed["created_at"]
    (tmp_path / "plan.sha256").unlink()
    repaired = procedure_grid.write_or_validate_plan(tmp_path, second)
    assert repaired["created_at"] == original_timestamp
    assert repaired == sealed


def test_rehashed_semantically_false_plan_is_rejected_on_load(
    tmp_path: Path,
) -> None:
    plan = procedure_grid.write_or_validate_plan(tmp_path, _tiny_plan())
    attacked = copy.deepcopy(plan)
    attacked["protocol"]["test"] = "producer may score held-out labels"
    attacked["plan_sha256"] = procedure_grid.plan_sha256(attacked)
    plan_path = tmp_path / "plan.json"
    digest_path = tmp_path / "plan.sha256"
    os.chmod(plan_path, 0o644)
    os.chmod(digest_path, 0o644)
    plan_path.write_bytes(procedure_grid._canonical_bytes(attacked) + b"\n")
    digest_path.write_text(attacked["plan_sha256"] + "\n", encoding="ascii")
    os.chmod(plan_path, 0o444)
    os.chmod(digest_path, 0o444)
    with pytest.raises(procedure_grid.ProcedureGridError, match="contract"):
        procedure_grid.load_plan(tmp_path)


def test_metadata_rejects_config_hash_candidate_and_reset_mutations() -> None:
    plan = _tiny_plan()
    job = next(procedure_grid.iter_jobs(plan))
    base = _metadata(plan, job)

    changed_config = copy.deepcopy(base)
    changed_config["frozen_config"]["config"]["epochs"] = 3
    with pytest.raises(procedure_grid.ProcedureGridError):
        procedure_grid._validate_metadata(plan, job, changed_config)

    changed_route = copy.deepcopy(base)
    changed_route["fit"]["selection_decision"] = {
        "selected_mixture": "posthoc_not_in_config",
        "selected_rho": 0.123456,
    }
    with pytest.raises(procedure_grid.ProcedureGridError):
        procedure_grid._validate_metadata(plan, job, changed_route)

    changed_reset = copy.deepcopy(base)
    changed_reset["fit"]["reset_hash_exclusions"] = ["all.model.parameters"]
    with pytest.raises(procedure_grid.ProcedureGridError):
        procedure_grid._validate_metadata(plan, job, changed_reset)


def test_resealed_record_and_completion_outcome_aliases_are_rejected(
    tmp_path: Path,
) -> None:
    plan = procedure_grid.write_or_validate_plan(tmp_path, _tiny_plan())
    job = next(procedure_grid.iter_jobs(plan))
    claim = procedure_grid.acquire_claim(
        tmp_path,
        plan,
        job,
        resource_guard=_safe_resource(tmp_path),
        recover_stale=False,
    )
    procedure_grid.commit_job_output(
        tmp_path,
        plan,
        claim,
        metadata=_metadata(plan, job),
        test_rows=np.arange(4, 8, dtype=np.int64),
        probabilities=np.full((4, 2), 0.5, dtype=np.float64),
    )
    procedure_grid.release_claim(claim)
    directory = procedure_grid._record_directory(tmp_path, job)
    record_path = directory / "record.json"
    completion_path = directory / "completion.json"
    record = procedure_grid._strict_json_load(record_path)
    completion = procedure_grid._strict_json_load(completion_path)
    record["metadata"]["hidden_outcome_vector"] = [0, 1, 0, 1]
    os.chmod(record_path, 0o644)
    record_path.write_bytes(procedure_grid._canonical_bytes(record) + b"\n")
    os.chmod(record_path, 0o444)
    completion["files"]["record.json"] = procedure_grid._sha256_file(record_path)
    completion["ground_truth_vector"] = [0, 1, 0, 1]
    os.chmod(completion_path, 0o644)
    completion_path.write_bytes(procedure_grid._canonical_bytes(completion) + b"\n")
    os.chmod(completion_path, 0o444)
    with pytest.raises(procedure_grid.ProcedureGridError):
        procedure_grid.validate_completion(tmp_path, plan, job)


def test_commit_requires_the_same_live_claim_inode(tmp_path: Path) -> None:
    plan = procedure_grid.write_or_validate_plan(tmp_path, _tiny_plan())
    job = next(procedure_grid.iter_jobs(plan))
    claim = procedure_grid.acquire_claim(
        tmp_path,
        plan,
        job,
        resource_guard=_safe_resource(tmp_path),
        recover_stale=False,
    )
    claim.path.unlink()
    with pytest.raises(procedure_grid.ProcedureGridError, match="claim"):
        procedure_grid.commit_job_output(
            tmp_path,
            plan,
            claim,
            metadata=_metadata(plan, job),
            test_rows=np.arange(4, 8, dtype=np.int64),
            probabilities=np.full((4, 2), 0.5, dtype=np.float64),
        )


def test_symlinked_records_root_cannot_escape_and_special_claim_is_audited(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    outside = tmp_path / "outside"
    outside.mkdir()
    plan = procedure_grid.write_or_validate_plan(run, _tiny_plan())
    (run / "records").symlink_to(outside, target_is_directory=True)
    job = next(procedure_grid.iter_jobs(plan))
    claim = procedure_grid.acquire_claim(
        run,
        plan,
        job,
        resource_guard=_safe_resource(run),
        recover_stale=False,
    )
    with pytest.raises(procedure_grid.ProcedureGridError):
        procedure_grid.commit_job_output(
            run,
            plan,
            claim,
            metadata=_metadata(plan, job),
            test_rows=np.arange(4, 8, dtype=np.int64),
            probabilities=np.full((4, 2), 0.5, dtype=np.float64),
        )
    procedure_grid.release_claim(claim)
    assert list(outside.iterdir()) == []
    (run / "records").unlink()
    claims = run / "claims"
    os.mkfifo(claims / "live-worker.fifo")
    audit = procedure_grid.audit_grid(run, plan)
    assert audit["complete"] is False
    assert "claims/live-worker.fifo" in audit["residual_claim_paths"]
    assert any("special node" in value for value in audit["unsafe_paths"])


def test_foreign_claim_is_never_stolen_by_wall_clock_age() -> None:
    value = {
        "created_at": "2000-01-01T00:00:00+00:00",
        "owner": {
            "hostname": "different-live-host",
            "pid": 999999,
            "boot_id": "00000000-0000-4000-8000-000000000001",
            "process_start_token": "1",
        },
    }
    assert procedure_grid._claim_is_live(value, foreign_timeout_seconds=0.0) is True
    malformed = copy.deepcopy(value)
    malformed["owner"]["pid"] = 0
    assert procedure_grid._claim_is_live(malformed) is False


def _uv_closure_fixture() -> tuple[str, str, dict[str, str]]:
    runtime = {
        name: f"1.0.{index}"
        for index, name in enumerate(
            sorted(procedure_grid.REQUIRED_UV_RUNTIME_PACKAGES), start=1
        )
    }
    groups = {
        group: {
            name: f"2.0.{index}" for index, name in enumerate(sorted(required), start=1)
        }
        for group, required in procedure_grid.REQUIRED_UV_DEPENDENCY_GROUPS.items()
    }
    runtime_requirements = ",\n".join(
        f'  "{name}=={version}"' for name, version in runtime.items()
    )
    group_tables = "\n".join(
        f"{group} = [\n"
        + ",\n".join(f'  "{name}=={version}"' for name, version in pins.items())
        + "\n]"
        for group, pins in groups.items()
    )
    pyproject = f"""
[project]
name = "synthetic-procedure"
version = "1.0.0"
requires-python = "==3.12.*"
dependencies = [
{runtime_requirements}
]
[dependency-groups]
{group_tables}
[tool.uv]
default-groups = []
"""
    root_dependencies = ",\n".join(f'  {{ name = "{name}" }}' for name in runtime)
    root_groups = "\n".join(
        f"{group} = [" + ", ".join(f'{{ name = "{name}" }}' for name in pins) + "]"
        for group, pins in groups.items()
    )
    requires_dist = ",\n".join(
        f'  {{ name = "{name}", specifier = "=={version}" }}'
        for name, version in runtime.items()
    )
    requires_dev = "\n".join(
        f"{group} = ["
        + ", ".join(
            f'{{ name = "{name}", specifier = "=={version}" }}'
            for name, version in pins.items()
        )
        + "]"
        for group, pins in groups.items()
    )
    package_entries = "\n".join(
        f"""
[[package]]
name = "{name}"
version = "{version}"
source = {{ registry = "https://example.invalid/simple" }}
"""
        for name, version in {
            **runtime,
            **{
                name: version
                for pins in groups.values()
                for name, version in pins.items()
            },
        }.items()
    )
    lock = f"""
version = 1
revision = 3
requires-python = "==3.12.*"

[[package]]
name = "synthetic-procedure"
version = "1.0.0"
source = {{ virtual = "." }}
dependencies = [
{root_dependencies}
]
[package.dev-dependencies]
{root_groups}
[package.metadata]
requires-dist = [
{requires_dist}
]
[package.metadata.requires-dev]
{requires_dev}
{package_entries}
"""
    return pyproject, lock, runtime


def test_uv_dependency_closure_is_exact_runtime_test_docs_contract() -> None:
    pyproject, lock, runtime = _uv_closure_fixture()
    identity = procedure_grid.validate_uv_dependency_closure(pyproject, lock)
    assert identity["runtime_locked_package_versions"] == runtime
    attacks = (
        pyproject.replace("torch==", "torch>=", 1),
        pyproject.replace("default-groups = []", 'default-groups = ["docs"]'),
    )
    for attacked in attacks:
        with pytest.raises(procedure_grid.ProcedureGridError):
            procedure_grid.validate_uv_dependency_closure(attacked, lock)
    with pytest.raises(procedure_grid.ProcedureGridError, match="groups"):
        procedure_grid.validate_uv_dependency_closure(
            pyproject,
            lock.replace(
                "[package.dev-dependencies]\ndocs = [",
                "[package.dev-dependencies]\ndocumentation = [",
                1,
            ),
        )
    with pytest.raises(procedure_grid.ProcedureGridError):
        procedure_grid.validate_uv_dependency_closure(
            pyproject,
            lock
            + """
[[package]]
name = "unreachable-extra"
version = "9.9.9"
source = { registry = "https://example.invalid/simple" }
""",
        )


def test_formal_runtime_is_exact_runtime_only_installed_closure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pyproject, lock, runtime = _uv_closure_fixture()
    monkeypatch.setattr(procedure_grid, "_require_uv_virtual_environment", lambda: None)
    monkeypatch.setattr(
        procedure_grid,
        "_release_uv_manifest_texts",
        lambda: (pyproject, lock),
    )

    def distributions(values):
        return [
            SimpleNamespace(metadata={"Name": name}, version=version)
            for name, version in values.items()
        ]

    monkeypatch.setattr(
        procedure_grid.importlib.metadata,
        "distributions",
        lambda: distributions(runtime),
    )
    procedure_grid._require_formal_release_runtime()
    with_extra = {**runtime, "pytest": "2.0.1"}
    monkeypatch.setattr(
        procedure_grid.importlib.metadata,
        "distributions",
        lambda: distributions(with_extra),
    )
    with pytest.raises(procedure_grid.ProcedureGridError, match="runtime-only"):
        procedure_grid._require_formal_release_runtime()


def test_source_closure_includes_initializers_registry_and_uv_manifests() -> None:
    assert {
        "src/benchmark/__init__.py",
        "src/benchmark/research/__init__.py",
        "src/benchmark/model_registry.py",
        "src/benchmark/project_gpu_leases.py",
        "pyproject.toml",
        "uv.lock",
    }.issubset(procedure_grid.SOURCE_FILES)


def test_source_identity_rejects_symlink_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = tmp_path / "real.py"
    real.write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "alias.py").symlink_to(real)
    monkeypatch.setattr(procedure_grid, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(procedure_grid, "SOURCE_FILES", ("alias.py",))
    with pytest.raises(procedure_grid.ProcedureGridError):
        procedure_grid._source_identity()


def test_secure_reader_rejects_fifo_and_symlinked_ancestor_immediately(
    tmp_path: Path,
) -> None:
    fifo = tmp_path / "cache.fifo"
    os.mkfifo(fifo)
    started = time.monotonic()
    with pytest.raises(procedure_grid.ProcedureGridError):
        procedure_grid._read_regular_file(fifo)
    assert time.monotonic() - started < 1.0

    real = tmp_path / "real"
    real.mkdir()
    (real / "cache.json").write_text("{}\n", encoding="utf-8")
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(procedure_grid.ProcedureGridError, match="symlink"):
        procedure_grid._read_regular_file(alias / "cache.json")


def test_recursive_plan_schemas_reject_nested_outcome_aliases() -> None:
    base = _tiny_plan()
    cache_attack = copy.deepcopy(base)
    cache_attack["cache_identity"]["tiny:s001"]["dataset"] = {"hidden_labels": [0, 1]}
    cache_attack["plan_sha256"] = procedure_grid.plan_sha256(cache_attack)
    with pytest.raises(procedure_grid.ProcedureGridError, match="outcome"):
        procedure_grid.validate_plan_semantics(cache_attack, allow_unsealed=True)

    dataset_attack = copy.deepcopy(base)
    dataset_attack["datasets"]["tiny"]["preprocessing"]["nested"] = {
        "ground_truth": [0, 1]
    }
    dataset_attack["plan_sha256"] = procedure_grid.plan_sha256(dataset_attack)
    with pytest.raises(procedure_grid.ProcedureGridError, match="outcome"):
        procedure_grid.validate_plan_semantics(dataset_attack, allow_unsealed=True)


def test_split_identity_rejects_boolean_subject_fold_and_trial_count() -> None:
    for field, value in (
        ("subject", True),
        ("fold", False),
        ("trial_count", True),
    ):
        attacked = _tiny_plan()
        attacked["split_identity"]["tiny:s001:f00"][field] = value
        attacked["plan_sha256"] = procedure_grid.plan_sha256(attacked)
        with pytest.raises(procedure_grid.ProcedureGridError, match="split"):
            procedure_grid.validate_plan_semantics(attacked, allow_unsealed=True)


def test_resource_status_uses_documented_probe_disk_signature(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from benchmark import full_grid

    calls: list[tuple[Path, float]] = []

    def probe_disk(
        path: Path,
        *,
        minimum_free_gib: float,
    ) -> SimpleNamespace:
        calls.append((path, minimum_free_gib))
        return SimpleNamespace(
            safe=True,
            reason="enough disk",
            as_dict=lambda: {
                "safe": True,
                "path": str(path),
                "free_bytes": 100 * 1024**3,
                "total_bytes": 200 * 1024**3,
                "free_gib": 100.0,
                "minimum_free_gib": minimum_free_gib,
                "reason": "enough disk",
            },
        )

    monkeypatch.setattr(full_grid, "probe_disk", probe_disk)
    monkeypatch.setattr(
        procedure_grid,
        "_strict_gpu_status",
        lambda **kwargs: {
            "safe": True,
            "reason": "idle",
        },
    )
    status = procedure_grid._resource_status(
        gpu="GPU-00000000-0000-0000-0000-000000000001",
        run_root=tmp_path,
        minimum_free_gib=50.0,
        allow_active_owner=True,
    )
    assert status["safe"] is True
    assert calls == [(tmp_path, 50.0)]


def test_strict_nvidia_probe_fails_closed_on_every_malformed_process_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested = "GPU-00000000-0000-0000-0000-000000000001"
    other = "GPU-00000000-0000-0000-0000-000000000002"
    identity = f"{requested}, 00000000:01:00.0, Test GPU, 0, 0\n"

    def run_probe(process_rows: str) -> dict[str, object]:
        outputs = iter((identity, process_rows))
        monkeypatch.setattr(
            procedure_grid.subprocess,
            "run",
            lambda *args, **kwargs: SimpleNamespace(stdout=next(outputs)),
        )
        return procedure_grid._strict_gpu_status(
            gpu=requested,
            allow_active_owner=False,
        )

    for malformed in (
        f"1, {other}, process, NaN\n",
        f"0, {other}, process, 1\n",
        f"1, {other}, process\n",
        "1, not-a-gpu, process, 1\n",
    ):
        status = run_probe(malformed)
        assert status["safe"] is False
        assert "failed closed" in status["reason"]
    foreign = run_probe(f"999999, {requested}, process, 1\n")
    assert foreign["safe"] is False
    assert len(foreign["foreign_processes"]) == 1


def test_hard_disk_floor_uses_exact_bound_filesystem_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    block_size = 4096
    required = int(procedure_grid.DEFAULT_MIN_FREE_GIB * 1024**3)
    monkeypatch.setattr(
        procedure_grid.os,
        "fstatvfs",
        lambda descriptor: SimpleNamespace(
            f_bavail=required // block_size - 1,
            f_frsize=block_size,
        ),
    )
    with pytest.raises(procedure_grid.ProcedureGridError, match="bytes free"):
        procedure_grid._require_hard_disk_floor(
            tmp_path, procedure_grid.DEFAULT_MIN_FREE_GIB
        )
    monkeypatch.setattr(
        procedure_grid.os,
        "fstatvfs",
        lambda descriptor: SimpleNamespace(
            f_bavail=required // block_size,
            f_frsize=block_size,
        ),
    )
    procedure_grid._require_hard_disk_floor(
        tmp_path, procedure_grid.DEFAULT_MIN_FREE_GIB
    )


def test_claim_loss_after_destination_rename_quarantines_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = procedure_grid.write_or_validate_plan(tmp_path, _tiny_plan())
    job = next(procedure_grid.iter_jobs(plan))
    claim = procedure_grid.acquire_claim(
        tmp_path,
        plan,
        job,
        resource_guard=_safe_resource(tmp_path),
        recover_stale=False,
    )
    original = procedure_grid._recheck_owned_claim_path
    calls = {"count": 0}

    def disappear_after_precheck(
        selected_claim: procedure_grid.Claim, descriptor: int
    ) -> None:
        original(selected_claim, descriptor)
        calls["count"] += 1
        if calls["count"] == 1:
            selected_claim.path.unlink()

    monkeypatch.setattr(
        procedure_grid,
        "_recheck_owned_claim_path",
        disappear_after_precheck,
    )
    with pytest.raises(procedure_grid.ProcedureGridError, match="claim"):
        procedure_grid.commit_job_output(
            tmp_path,
            plan,
            claim,
            metadata=_metadata(plan, job),
            test_rows=np.arange(4, 8, dtype=np.int64),
            probabilities=np.full((4, 2), 0.5, dtype=np.float64),
        )
    assert not procedure_grid._record_directory(tmp_path, job).exists()
    assert list(
        (_forensic_root(tmp_path) / "quarantine" / "claim_lost_after_commit").iterdir()
    )
    with pytest.raises(procedure_grid.ProcedureGridError, match="claim"):
        procedure_grid.release_claim(claim)


def test_formal_claim_and_result_rename_are_inside_held_gpu_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, plan, job, lease, resource, metadata = _formal_tiny_fixture(
        tmp_path, monkeypatch
    )
    monkeypatch.setattr(
        procedure_grid, "_require_hard_disk_floor", lambda *args, **kwargs: None
    )
    real_guard = procedure_grid.guard_gpu_lease
    real_rename = procedure_grid._atomic_rename_noreplace
    state = {"depth": 0}
    guarded_publications: list[str] = []
    claim_destination = procedure_grid._claim_path(run, job)
    result_destination = procedure_grid._record_directory(run, job)

    @contextmanager
    def tracked_guard(selected_lease):
        with real_guard(selected_lease) as receipt:
            state["depth"] += 1
            try:
                yield receipt
            finally:
                state["depth"] -= 1

    def tracked_rename(source: Path, destination: Path) -> None:
        if destination == claim_destination:
            assert state["depth"] > 0
            guarded_publications.append("claim")
        if destination == result_destination:
            assert state["depth"] > 0
            assert source.parent == destination.parent
            assert source.stat().st_mode & 0o777 == 0o555
            guarded_publications.append("result")
        real_rename(source, destination)

    monkeypatch.setattr(procedure_grid, "guard_gpu_lease", tracked_guard)
    monkeypatch.setattr(procedure_grid, "_atomic_rename_noreplace", tracked_rename)
    claim: procedure_grid.Claim | None = None
    try:
        claim = procedure_grid.acquire_claim(
            run,
            plan,
            job,
            resource_guard=resource,
            gpu_lease=lease,
            recover_stale=False,
        )
        destination = procedure_grid.commit_job_output(
            run,
            plan,
            claim,
            metadata=metadata,
            test_rows=np.asarray([4, 5, 6, 7], dtype=np.int64),
            probabilities=np.asarray(
                [[0.6, 0.4], [0.4, 0.6], [0.7, 0.3], [0.3, 0.7]],
                dtype=np.float64,
            ),
            resource_recheck=resource,
            gpu_lease=lease,
        )
        assert destination == result_destination
        assert guarded_publications == ["claim", "result"]
        assert (
            claim.value["gpu_lease_receipt"]
            == (
                procedure_grid.validate_completion(run, plan, job)["record"][
                    "commit_gpu_lease_receipt"
                ]
            )
        )
    finally:
        if claim is not None and claim.path.exists():
            procedure_grid.release_claim(claim)
        procedure_grid.release_gpu_lease(lease)


def test_formal_claim_disk_floor_blocks_prewrite_and_prerename_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for fail_at in (1, 2):
        with monkeypatch.context() as scoped:
            run, plan, job, lease, resource, _metadata_value = _formal_tiny_fixture(
                tmp_path / f"claim-{fail_at}", scoped
            )
            calls = {"count": 0}

            def require_floor(*args, **kwargs) -> None:
                calls["count"] += 1
                if calls["count"] == fail_at:
                    raise procedure_grid.ProcedureGridError("synthetic hard disk floor")

            scoped.setattr(procedure_grid, "_require_hard_disk_floor", require_floor)
            try:
                with pytest.raises(
                    procedure_grid.ProcedureGridError,
                    match="hard disk floor",
                ):
                    procedure_grid.acquire_claim(
                        run,
                        plan,
                        job,
                        resource_guard=resource,
                        gpu_lease=lease,
                        recover_stale=False,
                    )
                assert not procedure_grid._claim_path(run, job).exists()
            finally:
                procedure_grid.release_gpu_lease(lease)


def test_formal_commit_disk_floor_blocks_prewrite_and_prerename_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for fail_at in (1, 5):
        with monkeypatch.context() as scoped:
            run, plan, job, lease, resource, metadata = _formal_tiny_fixture(
                tmp_path / f"commit-{fail_at}", scoped
            )
            scoped.setattr(
                procedure_grid,
                "_require_hard_disk_floor",
                lambda *args, **kwargs: None,
            )
            claim = procedure_grid.acquire_claim(
                run,
                plan,
                job,
                resource_guard=resource,
                gpu_lease=lease,
                recover_stale=False,
            )
            calls = {"count": 0}

            def require_floor(*args, **kwargs) -> None:
                calls["count"] += 1
                if calls["count"] == fail_at:
                    raise procedure_grid.ProcedureGridError("synthetic hard disk floor")

            scoped.setattr(procedure_grid, "_require_hard_disk_floor", require_floor)
            try:
                with pytest.raises(
                    procedure_grid.ProcedureGridError,
                    match="hard disk floor",
                ):
                    procedure_grid.commit_job_output(
                        run,
                        plan,
                        claim,
                        metadata=metadata,
                        test_rows=np.asarray([4, 5, 6, 7], dtype=np.int64),
                        probabilities=np.asarray(
                            [
                                [0.6, 0.4],
                                [0.4, 0.6],
                                [0.7, 0.3],
                                [0.3, 0.7],
                            ],
                            dtype=np.float64,
                        ),
                        resource_recheck=resource,
                        gpu_lease=lease,
                    )
                assert not procedure_grid._record_directory(run, job).exists()
            finally:
                procedure_grid.release_claim(claim)
                procedure_grid.release_gpu_lease(lease)


def test_publication_fence_replacement_is_detected_while_locked(
    tmp_path: Path,
) -> None:
    plan = procedure_grid.write_or_validate_plan(tmp_path, _tiny_plan())
    with pytest.raises(procedure_grid.PublicationFenceLost):
        with procedure_grid.publication_fence(
            tmp_path,
            plan=plan,
            exclusive=True,
        ):
            path = procedure_grid._publication_fence_path(tmp_path)
            path.unlink()
            procedure_grid._write_json_exclusive(
                path, procedure_grid._publication_fence_value(plan)
            )


def test_formal_uv_gate_precedes_run_directory_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _formal_semantic_plan()
    target = tmp_path / "must-not-exist"

    def blocked() -> None:
        raise procedure_grid.ProcedureGridError("synthetic UV gate")

    monkeypatch.setattr(procedure_grid, "_require_formal_release_runtime", blocked)
    with pytest.raises(procedure_grid.ProcedureGridError, match="UV gate"):
        procedure_grid.write_or_validate_plan(target, plan)
    assert not target.exists()


def test_formal_uv_gate_requires_virtual_environment_to_equal_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = tmp_path / "uv-venv"
    monkeypatch.setattr(procedure_grid.sys, "prefix", str(prefix))
    monkeypatch.setattr(procedure_grid.sys, "base_prefix", str(tmp_path / "system"))
    monkeypatch.setattr(procedure_grid, "_uv_version", lambda: "uv test")
    monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path / "other"))
    with pytest.raises(procedure_grid.ProcedureGridError, match="VIRTUAL_ENV"):
        procedure_grid._require_uv_virtual_environment()
    monkeypatch.setenv("VIRTUAL_ENV", str(prefix))
    procedure_grid._require_uv_virtual_environment()


def test_config_inventory_hashes_and_parses_each_config_snapshot_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = procedure_grid._read_regular_file
    counts: dict[Path, int] = {}

    def recording(path: Path) -> bytes:
        selected = Path(path)
        counts[selected] = counts.get(selected, 0) + 1
        return original(selected)

    monkeypatch.setattr(procedure_grid, "_read_regular_file", recording)
    procedure_grid._config_inventory()
    for spec in procedure_grid.PROCEDURES:
        assert counts[procedure_grid.PROJECT_ROOT / spec.config_path] == 1
