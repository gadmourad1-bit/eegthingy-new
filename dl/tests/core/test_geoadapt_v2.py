from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import os
import shutil
import stat
import sys
import warnings
import zipfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmark import geoadapt_v2
from benchmark import geoadapt_v2_analysis


TEST_NONCE = "ffffffffffff4fff8fffffffffffffff"


class SyntheticPublicationInterrupt(BaseException):
    pass


def _split_identity(
    *,
    dataset: str,
    subject: int,
    fold: int,
    digest: str,
    trial_count: int = 12,
) -> dict[str, object]:
    train = np.asarray([0, 1, 2, 3], dtype=np.int64)
    validation = np.asarray([4, 5], dtype=np.int64)
    source = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64)
    test = np.arange(6, trial_count, dtype=np.int64)
    return {
        "dataset": dataset,
        "subject": subject,
        "fold": fold,
        "cache_array_sha256": digest,
        "trial_count": trial_count,
        "partitions": {
            "train": {
                "count": len(train),
                "rows_sha256": geoadapt_v2._rows_sha256(train),
            },
            "validation": {
                "count": len(validation),
                "rows_sha256": geoadapt_v2._rows_sha256(validation),
            },
            "source": {
                "count": len(source),
                "rows_sha256": geoadapt_v2._rows_sha256(source),
            },
            "test": {
                "count": len(test),
                "rows_sha256": geoadapt_v2._rows_sha256(test),
            },
        },
    }


def _tiny_plan(
    *,
    seeds: tuple[int, ...] = (7,),
    minimum_free_gib: float = 0.0,
    stable_id: str = "architecture.geoadaptnet",
) -> dict[str, object]:
    digest = "a" * 64
    return geoadapt_v2.assemble_plan(
        dataset_contracts={
            "tiny": {
                "subjects": [1],
                "folds": [0],
                "n_classes": 2,
                "protocol": "synthetic",
                "preprocessing": {"sfreq_hz": 128.0, "n_times": 96},
                "montage_profile": "harmonized",
            }
        },
        stable_configs={
            stable_id: copy.deepcopy(geoadapt_v2.STABLE_CONFIGS[stable_id])
        },
        eligibility={
            "tiny": {
                stable_id: {"status": "eligible", "reason": None},
            },
            "bnci2014_001": {
                stable_id: {
                    "status": "not_applicable",
                    "reason": geoadapt_v2.BNCI001_NA_REASON,
                }
            },
        },
        seeds=seeds,
        cache_identity={
            "tiny:s001": {
                "array_sha256": digest,
                "shape": [12, 4, 96],
            }
        },
        split_identity={
            "tiny:s001:f00": _split_identity(
                dataset="tiny",
                subject=1,
                fold=0,
                digest=digest,
            )
        },
        source_identity={"runner.py": "b" * 64},
        environment_identity={"python": "synthetic", "uv_version": "uv test"},
        project_identity={
            "pyproject_sha256": "c" * 64,
            "uv_lock_sha256": "d" * 64,
            "required_runtime_direct_dependencies": [
                "numpy",
                "scikit-learn",
                "scipy",
                "torch",
            ],
            "runtime_direct_dependency_pins": {
                "numpy": "1.0.0",
                "scikit-learn": "1.0.0",
                "scipy": "1.0.0",
                "torch": "1.0.0",
            },
            "project_direct_dependency_pins": {
                "numpy": "1.0.0",
                "scikit-learn": "1.0.0",
                "scipy": "1.0.0",
                "torch": "1.0.0",
            },
            "test_dependency_group": "test",
            "required_test_group_dependencies": ["pytest"],
            "test_group_dependency_pins": {"pytest": "1.0.0"},
            "locked_package_closure_sha256": "e" * 64,
        },
        feature_contract={"schema": "synthetic"},
        minimum_free_gib=minimum_free_gib,
        worker_cpu_threads=2,
    )


def _fake_data() -> dict[str, object]:
    rng = np.random.default_rng(20260729)
    x = rng.normal(size=(12, 4, 96)).astype(np.float32)
    y = np.asarray([0, 1] * 6, dtype=np.int64)
    return {
        "x": x,
        "y": y,
        "positions": np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        "channel_names": np.asarray(["C3", "C4", "Cz", "Fz"]),
        "sessions": np.asarray(["s"] * 12),
        "runs": np.asarray(["r"] * 12),
        "identity": {
            "array_sha256": "a" * 64,
            "shape": [12, 4, 96],
        },
    }


def _fake_loader(*args, **kwargs) -> dict[str, object]:
    del args, kwargs
    return _fake_data()


def _fake_splitter(*args, **kwargs):
    del args, kwargs
    return (
        np.asarray([0, 1, 2, 3], dtype=np.int64),
        np.asarray([4, 5], dtype=np.int64),
        np.arange(6, 12, dtype=np.int64),
    )


def _runtime_stub(
    device: str,
    *,
    worker_cpu_threads: int,
    portable_compute_contract: dict[str, object] | None = None,
) -> dict[str, object]:
    environment = (
        {"python": "synthetic", "uv_version": "uv test"}
        if portable_compute_contract is None
        else portable_compute_contract
    )
    return {
        "device": device,
        "portable_compute_contract_sha256": geoadapt_v2._sha256_bytes(
            geoadapt_v2._canonical_bytes(environment)
        ),
        "torch_determinism": {
            **{
                key: value
                for key, value in geoadapt_v2.DETERMINISTIC_TORCH_CONTRACT.items()
                if key != "cublas_workspace_config_on_cuda"
            },
            "cublas_workspace_config": None,
            "cuda_available": False,
        },
        "worker_cpu_threads": worker_cpu_threads,
        "worker_interop_threads": 1,
        "thread_environment": {
            name: str(worker_cpu_threads)
            for name in geoadapt_v2.THREAD_ENVIRONMENT_VARIABLES
        },
        "node_inventory": {"host": "tiny-host", "cuda": None},
    }


def _device_resource_stub(device: str) -> dict[str, object]:
    return {
        "requested_device": device,
        "resolved_device": device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "node_inventory": {"host": "tiny-host", "cuda": None},
    }


def _cuda_device_resource_stub(
    gpu_uuid: str = "GPU-00000000-0000-0000-0000-000000000001",
) -> dict[str, object]:
    return {
        "requested_device": "cuda:0",
        "resolved_device": "cuda:0",
        "cuda_visible_devices": gpu_uuid,
        "node_inventory": {
            "host": "tiny-host",
            "cuda": {
                "logical_index": 0,
                "name": "Synthetic GPU",
                "capability": [12, 0],
                "total_memory_bytes": 1,
                "cuda_visible_devices": gpu_uuid,
                "inventory": [
                    {
                        "index": 3,
                        "uuid": gpu_uuid,
                        "name": "Synthetic GPU",
                        "driver_version": "999.0",
                        "memory_total_mib": 1,
                    }
                ],
            },
        },
    }


def _gpu_status_stub(
    gpu_uuid: str,
    *,
    allow_active_owner: bool,
) -> dict[str, object]:
    return {
        "schema": "eeg-mi-geoadapt-v2-cooperative-gpu-status-v1",
        "gpu_uuid": gpu_uuid,
        "utilization_percent": 0.0,
        "memory_used_mib": 0.0,
        "foreign_processes": [],
        "allow_active_owner": allow_active_owner,
        "utilization_limit_percent": (100.0 if allow_active_owner else 10.0),
        "safe": True,
        "reason": "resources available",
    }


def _cuda_resource_guard_stub(
    *,
    root: Path,
    gpu_uuid: str,
    receipt: dict[str, object],
    allow_active_owner: bool,
    host: str = "tiny-host",
) -> dict[str, object]:
    resource = _cuda_device_resource_stub(gpu_uuid)
    resource["node_inventory"]["host"] = host
    return {
        "disk": {
            "safe": True,
            "path": str(root),
            "free_bytes": 100 * 1024**3,
            "total_bytes": 200 * 1024**3,
            "free_gib": 100.0,
            "minimum_free_gib": 0.0,
        },
        "requested_device": "cuda:0",
        "cuda_visible_devices": gpu_uuid,
        "worker_cpu_threads": 2,
        "device_resource": resource,
        "gpu_status": _gpu_status_stub(
            gpu_uuid,
            allow_active_owner=allow_active_owner,
        ),
        "gpu_lease_receipt": copy.deepcopy(receipt),
    }


def _allow_tiny_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        geoadapt_v2,
        "validate_production_semantics",
        geoadapt_v2._validate_plan_shape,
    )
    monkeypatch.setattr(
        geoadapt_v2,
        "_device_resource_identity",
        _device_resource_stub,
    )
    monkeypatch.setattr(
        geoadapt_v2,
        "_rebind_authoritative_inputs",
        lambda **kwargs: kwargs["plan"],
    )
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    for name in geoadapt_v2.THREAD_ENVIRONMENT_VARIABLES:
        monkeypatch.setenv(name, "2")


def _backend_probabilities(prepared: geoadapt_v2.PreparedJob) -> np.ndarray:
    # Fixed row-order probabilities; the backend has no held-out outcome field.
    return np.asarray(
        [
            [0.8, 0.2],
            [0.2, 0.8],
            [0.7, 0.3],
            [0.3, 0.7],
            [0.6, 0.4],
            [0.4, 0.6],
        ],
        dtype=np.float64,
    )[: len(prepared.test_rows)]


def _tiny_backend(*, job, plan, prepared, device):
    del job, plan, device
    assert not hasattr(prepared, "y_test")
    return (
        {
            "initial_state_sha256": "1" * 64,
            "selection_state_sha256": "2" * 64,
            "reset_state_sha256": "1" * 64,
            "reset_verified": True,
            "source_selected_epoch_index": 1,
            "selection_epochs_run": 3,
            "refit_epochs_run": 2,
            "refit_state_sha256": "3" * 64,
            "parameter_count": 42,
            "selection_fit_seconds": 0.1,
            "refit_fit_seconds": 0.2,
            "test_inference_seconds": 0.01,
            "cuda_peak_memory_bytes": 0,
        },
        _backend_probabilities(prepared),
    )


def _claim(
    job: geoadapt_v2.Job,
    tmp_path: Path,
    plan: dict[str, object],
) -> geoadapt_v2.Claim:
    path = geoadapt_v2._claim_path(tmp_path, job)
    owner = {
        "host": "tiny-host",
        "pid": 1,
        "start_marker": "tiny-start",
        "boot_marker": "tiny-boot",
    }
    resource_guard = {
        "disk": {
            "safe": True,
            "path": str(tmp_path),
            "free_bytes": 100 * 1024**3,
            "total_bytes": 200 * 1024**3,
            "free_gib": 100.0,
            "minimum_free_gib": 0.0,
        },
        "requested_device": "cpu",
        "cuda_visible_devices": None,
        "worker_cpu_threads": 2,
        "device_resource": _device_resource_stub("cpu"),
        "gpu_status": None,
        "gpu_lease_receipt": None,
    }
    geoadapt_v2._write_json_exclusive(
        path,
        {
            "schema": geoadapt_v2.CLAIM_SCHEMA,
            "created_at": geoadapt_v2._utc_now(),
            "plan_sha256": plan["plan_sha256"],
            "job_id": job.job_id,
            "job": job.identity(),
            "nonce": TEST_NONCE,
            "owner": owner,
            "resource_guard": resource_guard,
            "gpu_lease_receipt": None,
        },
    )
    identity = geoadapt_v2._anchored_lstat(path)
    return geoadapt_v2.Claim(
        job=job,
        path=path,
        nonce=TEST_NONCE,
        owner=owner,
        resource_guard=resource_guard,
        gpu_lease_receipt=None,
        st_dev=identity.st_dev,
        st_ino=identity.st_ino,
    )


def _commit_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    seeds: tuple[int, ...] = (7,),
):
    _allow_tiny_plan(monkeypatch)
    plan = _tiny_plan(seeds=seeds)
    geoadapt_v2.write_or_validate_plan(tmp_path, plan)
    monkeypatch.setattr(geoadapt_v2, "_runtime_identity", _runtime_stub)
    job = next(geoadapt_v2.iter_jobs(plan))
    metadata, rows, probabilities = geoadapt_v2.execute_job(
        job=job,
        plan=plan,
        cache_root=tmp_path,
        device="cpu",
        backend=_tiny_backend,
        cache_loader=_fake_loader,
        splitter=_fake_splitter,
    )
    claim = _claim(job, tmp_path, plan)
    return plan, job, claim, metadata, rows, probabilities


def test_production_grid_has_6495_jobs_and_preserves_na_cells() -> None:
    contracts = geoadapt_v2._dataset_contracts()
    caches: dict[str, object] = {}
    splits: dict[str, object] = {}
    for dataset, contract in contracts.items():
        for subject in contract["subjects"]:
            key = geoadapt_v2._subject_key(dataset, subject)
            digest = geoadapt_v2._sha256_bytes(key.encode())
            channels = list(
                geoadapt_v2.channels_for_dataset(
                    dataset,
                    geoadapt_v2.MONTAGE_PROFILE,
                )
                or ()
            )
            cache_identity = {
                "array_sha256": digest,
                "channels": channels,
                "coordinates": geoadapt_v2.coordinate_contract_for_dataset(
                    dataset,
                    geoadapt_v2.MONTAGE_PROFILE,
                    tuple(channels),
                ),
                "dataset": copy.deepcopy(contract["dataset_spec"]),
                "montage_profile": geoadapt_v2.MONTAGE_PROFILE,
                "preprocessing": copy.deepcopy(contract["preprocessing"]),
                "shape": [
                    12,
                    len(channels),
                    int(contract["preprocessing"]["n_times"]),
                ],
                "subject": subject,
            }
            if dataset == "local_exp4":
                cache_identity["source_manifest"] = [
                    {
                        "filename": f"exp4_subject{subject}_training_1_mi_raw.fif",
                        "run": 1,
                        "sha256": "f" * 64,
                        "size_bytes": 1,
                        "subject": subject,
                    }
                ]
            caches[key] = cache_identity
            for fold in contract["folds"]:
                splits[geoadapt_v2._split_key(dataset, subject, fold)] = (
                    _split_identity(
                        dataset=dataset,
                        subject=subject,
                        fold=fold,
                        digest=digest,
                    )
                )
    plan = geoadapt_v2.assemble_plan(
        dataset_contracts=contracts,
        stable_configs=geoadapt_v2.STABLE_CONFIGS,
        eligibility=geoadapt_v2._eligibility_matrix(),
        seeds=geoadapt_v2.SEEDS,
        cache_identity=caches,
        split_identity=splits,
        source_identity={
            name: geoadapt_v2._sha256_bytes(name.encode())
            for name in geoadapt_v2.SOURCE_FILES
        },
        environment_identity={
            "schema": "eeg-mi-portable-compute-contract-v1",
            "python_version": "3.12.0",
            "python_implementation": "CPython",
            "platform_system": "Linux",
            "platform_machine": "x86_64",
            "uv_version": "uv 1.0.0",
            "packages": [
                [name, "1.0.0"]
                for name in sorted(geoadapt_v2.REQUIRED_RUNTIME_DIRECT_DEPENDENCIES)
            ],
            "packages_sha256": geoadapt_v2._sha256_bytes(
                geoadapt_v2._canonical_bytes(
                    [
                        [name, "1.0.0"]
                        for name in sorted(
                            geoadapt_v2.REQUIRED_RUNTIME_DIRECT_DEPENDENCIES
                        )
                    ]
                )
            ),
            "torch": {
                "version": "1.0.0",
                "cuda_build": None,
                "cudnn": None,
            },
            "deterministic_torch_contract": copy.deepcopy(
                geoadapt_v2.DETERMINISTIC_TORCH_CONTRACT
            ),
        },
        project_identity={
            "pyproject_sha256": "b" * 64,
            "uv_lock_sha256": "c" * 64,
            "required_runtime_direct_dependencies": sorted(
                geoadapt_v2.REQUIRED_RUNTIME_DIRECT_DEPENDENCIES
            ),
            "runtime_direct_dependency_pins": {
                name: "1.0.0"
                for name in geoadapt_v2.REQUIRED_RUNTIME_DIRECT_DEPENDENCIES
            },
            "project_direct_dependency_pins": {
                name: "1.0.0"
                for name in geoadapt_v2.REQUIRED_RUNTIME_DIRECT_DEPENDENCIES
            },
            "test_dependency_group": "test",
            "required_test_group_dependencies": sorted(
                geoadapt_v2.REQUIRED_TEST_GROUP_DEPENDENCIES
            ),
            "test_group_dependency_pins": {"pytest": "9.0.0"},
            "required_docs_group_dependencies": sorted(
                geoadapt_v2.REQUIRED_DOCS_GROUP_DEPENDENCIES
            ),
            "docs_group_dependency_pins": {
                name: "1.0.0" for name in geoadapt_v2.REQUIRED_DOCS_GROUP_DEPENDENCIES
            },
            "dependency_group_pins": {
                "docs": {
                    name: "1.0.0"
                    for name in geoadapt_v2.REQUIRED_DOCS_GROUP_DEPENDENCIES
                },
                "test": {"pytest": "9.0.0"},
            },
            "runtime_locked_package_versions": {
                name: "1.0.0"
                for name in geoadapt_v2.REQUIRED_RUNTIME_DIRECT_DEPENDENCIES
            },
            "runtime_locked_package_closure_sha256": (
                geoadapt_v2._sha256_bytes(
                    geoadapt_v2._canonical_bytes(
                        {
                            name: "1.0.0"
                            for name in (
                                geoadapt_v2.REQUIRED_RUNTIME_DIRECT_DEPENDENCIES
                            )
                        }
                    )
                )
            ),
            "locked_package_closure_sha256": "d" * 64,
            "project_name": "eegthingy",
            "project_version": "0.1.0",
            "requires_python": "==3.12.*",
            "uv_lock_version": 1,
            "uv_lock_revision": 1,
        },
        feature_contract=geoadapt_v2._feature_contract(),
        minimum_free_gib=50.0,
    )
    geoadapt_v2.validate_production_semantics(plan)
    jobs = tuple(geoadapt_v2.iter_jobs(plan))
    assert len(jobs) == plan["n_jobs"] == 6_495
    assert len({job.job_id for job in jobs}) == len(jobs)
    assert Counter(job.stable_id for job in jobs) == {
        "architecture.geoadaptnet": 2_150,
        "architecture.geoadaptnet_fb": 2_195,
        "architecture.geoadaptnet_fbsp": 2_150,
    }
    assert not any(
        job.dataset == "bnci2014_004"
        and job.stable_id
        in {
            "architecture.geoadaptnet",
            "architecture.geoadaptnet_fbsp",
        }
        for job in jobs
    )
    assert plan["eligibility"]["bnci2014_001"]["architecture.geoadaptnet"] == {
        "status": "not_applicable",
        "reason": geoadapt_v2.BNCI001_NA_REASON,
    }
    not_applicable = {
        (dataset, stable_id, cell["reason"])
        for dataset, cells in plan["eligibility"].items()
        for stable_id, cell in cells.items()
        if cell["status"] == "not_applicable"
    }
    assert not_applicable == {
        (
            "bnci2014_001",
            stable_id,
            geoadapt_v2.BNCI001_NA_REASON,
        )
        for stable_id in geoadapt_v2.STABLE_IDS
    } | {
        (
            "bnci2014_004",
            "architecture.geoadaptnet",
            geoadapt_v2.GEOADAPT_BNCI004_NA_REASON,
        ),
        (
            "bnci2014_004",
            "architecture.geoadaptnet_fbsp",
            geoadapt_v2.FBSP_BNCI004_NA_REASON,
        ),
    }
    mutations = (
        lambda value: value["seeds"].__setitem__(0, 8),
        lambda value: value["stable_configs"]["architecture.geoadaptnet"][
            "model"
        ].__setitem__("dropout", 0.5),
        lambda value: value["eligibility"]["bnci2014_001"][
            "architecture.geoadaptnet"
        ].__setitem__("reason", "changed"),
        lambda value: value["datasets"]["cho2017"]["preprocessing"].__setitem__(
            "sfreq_hz", 64.0
        ),
        lambda value: value.__setitem__("n_jobs", 6_495.0),
        lambda value: value["execution_config"].__setitem__(
            "worker_cpu_threads",
            True,
        ),
        lambda value: value["execution_config"].__setitem__(
            "minimum_free_gib",
            50,
        ),
        lambda value: value["project_identity"].__setitem__(
            "uv_lock_version",
            True,
        ),
    )
    for mutate in mutations:
        drifted = copy.deepcopy(plan)
        mutate(drifted)
        drifted["plan_sha256"] = geoadapt_v2.plan_sha256(drifted)
        with pytest.raises(geoadapt_v2.GeoAdaptV2Error):
            geoadapt_v2.validate_production_semantics(drifted)


def test_fb_and_fbsp_stable_configs_are_not_runtime_aliases() -> None:
    fb = geoadapt_v2.STABLE_CONFIGS["architecture.geoadaptnet_fb"]
    fbsp = geoadapt_v2.STABLE_CONFIGS["architecture.geoadaptnet_fbsp"]
    assert fb["model"]["reduced_dim"] is None
    assert fbsp["model"]["reduced_dim"] == 8
    assert fb["runtime_key"] != fbsp["runtime_key"]


def test_registry_and_plan_share_literal_bnci004_eligibility() -> None:
    from benchmark.model_registry import record_by_id

    expected = geoadapt_v2._eligibility_matrix()
    for stable_id in geoadapt_v2.STABLE_IDS:
        record = record_by_id(stable_id)
        observed = {
            decision.dataset: {
                "status": "eligible" if decision.eligible else "not_applicable",
                "reason": decision.reason,
            }
            for decision in record.dataset_eligibility
        }
        for dataset in (*geoadapt_v2.DATASET_ORDER, geoadapt_v2.PUBLIC_NA_DATASET):
            assert observed[dataset] == expected[dataset][stable_id]
    assert (
        expected["bnci2014_004"]["architecture.geoadaptnet_fb"]["status"] == "eligible"
    )


def test_sinc_warm_start_has_exact_effective_cutoffs_and_safe_constraints() -> None:
    import torch

    from benchmark.research.filterbank_net import SincFilterBank

    bands = list(geoadapt_v2.FILTER_BANDS_HZ)
    bank = SincFilterBank(4, 128.0, init_bands=bands, min_hz=2.0, min_bw=2.0)
    low, high = bank.effective_cutoffs()
    assert np.allclose(low.detach().numpy(), np.asarray(bands)[:, 0], atol=1e-12)
    assert np.allclose(high.detach().numpy(), np.asarray(bands)[:, 1], atol=1e-12)
    optimizer = torch.optim.AdamW(bank.parameters(), lr=5.0)
    for _ in range(4):
        optimizer.zero_grad()
        perturbed_low, perturbed_high = bank.effective_cutoffs()
        loss = (
            torch.randn_like(perturbed_low).mul(perturbed_low).sum()
            + torch.randn_like(perturbed_high).mul(perturbed_high).sum()
        )
        loss.backward()
        optimizer.step()
    constrained_low, constrained_high = bank.effective_cutoffs()
    assert torch.all(torch.diff(constrained_low) > 0.0)
    assert torch.all(torch.diff(constrained_high) > 0.0)
    assert torch.all(constrained_high - constrained_low > bank.min_bw)
    assert torch.all(constrained_high <= 64.0)
    with pytest.raises(ValueError, match="strictly ordered"):
        SincFilterBank(
            2,
            128.0,
            init_bands=[(11.0, 15.0), (8.0, 12.0)],
        )
    with pytest.raises(ValueError, match="Nyquist"):
        SincFilterBank(1, 128.0, init_bands=[(60.0, 65.0)])


def test_torch_determinism_is_fail_closed_and_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "CUBLAS_WORKSPACE_CONFIG",
        geoadapt_v2.DETERMINISTIC_TORCH_CONTRACT["cublas_workspace_config_on_cuda"],
    )
    geoadapt_v2._configure_torch(7, deterministic=True)
    observed = geoadapt_v2._torch_determinism_identity()
    for key in (
        "deterministic_algorithms",
        "warn_only",
        "cudnn_deterministic",
        "cudnn_benchmark",
        "cuda_matmul_tf32",
        "cudnn_tf32",
        "float32_matmul_precision",
    ):
        assert observed[key] == geoadapt_v2.DETERMINISTIC_TORCH_CONTRACT[key]
    with pytest.raises(geoadapt_v2.GeoAdaptV2Error, match="nondeterministic"):
        geoadapt_v2._configure_torch(7, deterministic=False)


def test_source_uv_and_spd_transform_identities_are_frozen(
    tmp_path: Path,
) -> None:
    sources = geoadapt_v2._source_identity()
    required = {
        "src/benchmark/geoadapt_v2.py",
        "src/benchmark/geoadapt_v2_analysis.py",
        "src/benchmark/data.py",
        "src/benchmark/config.py",
        "src/benchmark/model_registry.py",
        "src/benchmark/research/model.py",
        "src/benchmark/research/filterbank_net.py",
        "src/benchmark/research/engine.py",
        "src/benchmark/shared/spd.py",
        "pyproject.toml",
        "uv.lock",
    }
    assert required.issubset(sources)
    assert all(
        len(value) == 64 and set(value) <= set("0123456789abcdef")
        for value in sources.values()
    )
    feature = geoadapt_v2._feature_contract()
    assert feature["schema"] == "eeg-mi-geoadapt-four-band-spd-view-v1"
    assert feature["bands_hz"] == [
        [8.0, 12.0],
        [11.0, 15.0],
        [14.0, 20.0],
        [20.0, 30.0],
    ]
    assert len(feature["coefficient_sha256"]) == 64
    assert feature["fit_scope"]["held_out_outcomes"] == "never used"

    project = tmp_path / "uv-project"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        """
[project]
name = "tiny"
version = "0.0.0"
requires-python = "==3.12.*"
dependencies = [
  "numpy==2.4.4",
  "scipy==1.18.0",
  "scikit-learn==1.8.0",
  "torch==2.6.0",
]

[dependency-groups]
test = ["pytest==9.0.0"]
docs = [
  "pdfplumber==0.11.9",
  "pypdf==6.9.0",
  "reportlab==4.4.10",
]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (project / "uv.lock").write_text(
        """
version = 1
revision = 1
requires-python = "==3.12.*"

[[package]]
name = "tiny"
version = "0.0.0"
source = { virtual = "." }
dependencies = [
  { name = "numpy" },
  { name = "scikit-learn" },
  { name = "scipy" },
  { name = "torch" },
]
dev-dependencies = { docs = [{ name = "pdfplumber" }, { name = "pypdf" }, { name = "reportlab" }], test = [{ name = "pytest" }] }
[package.metadata]
requires-dist = [
  { name = "numpy", specifier = "==2.4.4" },
  { name = "scikit-learn", specifier = "==1.8.0" },
  { name = "scipy", specifier = "==1.18.0" },
  { name = "torch", specifier = "==2.6.0" },
]
requires-dev = { docs = [{ name = "pdfplumber", specifier = "==0.11.9" }, { name = "pypdf", specifier = "==6.9.0" }, { name = "reportlab", specifier = "==4.4.10" }], test = [{ name = "pytest", specifier = "==9.0.0" }] }

[[package]]
name = "numpy"
version = "2.4.4"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "pdfplumber"
version = "0.11.9"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "pypdf"
version = "6.9.0"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "pytest"
version = "9.0.0"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "reportlab"
version = "4.4.10"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "scikit-learn"
version = "1.8.0"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "scipy"
version = "1.18.0"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "torch"
version = "2.6.0"
source = { registry = "https://pypi.org/simple" }
""".lstrip(),
        encoding="utf-8",
    )
    identity = geoadapt_v2._validate_uv_project_contract(project)
    assert identity["required_runtime_direct_dependencies"] == [
        "numpy",
        "scikit-learn",
        "scipy",
        "torch",
    ]
    assert identity["test_dependency_group"] == "test"
    assert identity["test_group_dependency_pins"] == {"pytest": "9.0.0"}
    assert identity["docs_group_dependency_pins"] == {
        "pdfplumber": "0.11.9",
        "pypdf": "6.9.0",
        "reportlab": "4.4.10",
    }
    assert identity["dependency_group_pins"] == {
        "docs": {
            "pdfplumber": "0.11.9",
            "pypdf": "6.9.0",
            "reportlab": "4.4.10",
        },
        "test": {"pytest": "9.0.0"},
    }
    assert identity["runtime_locked_package_versions"] == {
        "numpy": "2.4.4",
        "scikit-learn": "1.8.0",
        "scipy": "1.18.0",
        "torch": "2.6.0",
    }
    assert len(identity["pyproject_sha256"]) == 64
    assert len(identity["uv_lock_sha256"]) == 64
    with (project / "uv.lock").open("a", encoding="utf-8") as handle:
        handle.write(
            """

[[package]]
name = "orphan"
version = "1.0.0"
source = { registry = "https://pypi.org/simple" }
"""
        )
    with pytest.raises(geoadapt_v2.GeoAdaptV2Error, match="outside.*closure"):
        geoadapt_v2._validate_uv_project_contract(project)


def test_active_uv_environment_must_be_exact_runtime_only_closure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = tmp_path / "runtime-venv"
    environment.mkdir()
    monkeypatch.setattr(geoadapt_v2.sys, "prefix", str(environment))
    monkeypatch.setattr(
        geoadapt_v2.sys,
        "base_prefix",
        str(tmp_path / "system-python"),
    )
    monkeypatch.setenv("VIRTUAL_ENV", str(environment))
    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        del kwargs
        commands.append(list(command))
        return SimpleNamespace(stdout="uv 0.11.8\n")

    distributions = [
        SimpleNamespace(metadata={"Name": "numpy"}, version="2.4.4"),
        SimpleNamespace(metadata={"Name": "torch"}, version="2.6.0"),
    ]
    monkeypatch.setattr(geoadapt_v2.subprocess, "run", fake_run)
    monkeypatch.setattr(
        geoadapt_v2.importlib.metadata,
        "distributions",
        lambda: tuple(distributions),
    )
    identity = {
        "runtime_locked_package_versions": {
            "numpy": "2.4.4",
            "torch": "2.6.0",
        }
    }
    geoadapt_v2._require_uv_virtual_environment(identity)
    assert commands[1][0:3] == ["uv", "sync", "--check"]
    assert "--active" in commands[1]
    assert "--frozen" in commands[1]
    assert "--no-default-groups" in commands[1]
    assert "--no-install-project" in commands[1]

    distributions.append(SimpleNamespace(metadata={"Name": "pytest"}, version="9.0.0"))
    with pytest.raises(
        geoadapt_v2.GeoAdaptV2Error,
        match="runtime-only.*test/docs/extras",
    ):
        geoadapt_v2._require_uv_virtual_environment(identity)


def test_ece_is_frozen_at_fifteen_bins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import inspect

    assert (
        "ece_bins"
        not in inspect.signature(geoadapt_v2_analysis.classification_metrics).parameters
    )
    assert (
        "ece_bins"
        not in inspect.signature(geoadapt_v2_analysis.analyze_grid).parameters
    )
    truth = np.asarray([0, 1, 0, 1], dtype=np.int64)
    probabilities = np.asarray(
        [[0.95, 0.05], [0.4, 0.6], [0.55, 0.45], [0.2, 0.8]],
        dtype=np.float64,
    )
    expected = geoadapt_v2_analysis.classification_metrics(
        truth,
        probabilities,
    )["ece"]
    monkeypatch.setattr(geoadapt_v2_analysis, "DEFAULT_ECE_BINS", 2)
    assert (
        geoadapt_v2_analysis.classification_metrics(truth, probabilities)["ece"]
        == expected
    )
    with pytest.raises(SystemExit):
        geoadapt_v2_analysis._build_parser().parse_args(
            [
                "--run-root",
                "run",
                "--cache-root",
                "cache",
                "--output-root",
                "output",
                "--ece-bins",
                "10",
            ]
        )


@pytest.mark.parametrize(
    ("stable_id", "reason", "match"),
    (
        (
            "architecture.geoadaptnet",
            geoadapt_v2.GEOADAPT_BNCI004_NA_REASON,
            "new variant",
        ),
        (
            "architecture.geoadaptnet_fbsp",
            geoadapt_v2.FBSP_BNCI004_NA_REASON,
            "collapse/alias",
        ),
    ),
)
def test_dimensionally_invalid_bnci004_cells_rejected_before_cache_access(
    tmp_path: Path,
    stable_id: str,
    reason: str,
    match: str,
) -> None:
    plan = _tiny_plan(stable_id=stable_id)
    plan["eligibility"]["bnci2014_004"] = {
        stable_id: {
            "status": "not_applicable",
            "reason": reason,
        }
    }
    job = geoadapt_v2.Job(
        dataset="bnci2014_004",
        stable_id=stable_id,
        subject=1,
        fold=0,
        seed=7,
    )
    calls = 0

    def forbidden_loader(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("N/A rejection must happen before cache access")

    with pytest.raises(
        geoadapt_v2.NotApplicableError,
        match=match,
    ):
        geoadapt_v2.prepare_job(
            job=job,
            plan=plan,
            cache_root=tmp_path,
            cache_loader=forbidden_loader,
            splitter=_fake_splitter,
        )
    assert calls == 0


def test_four_band_spd_view_is_deterministic_and_positive() -> None:
    rng = np.random.default_rng(8)
    x = rng.normal(size=(3, 5, 96)).astype(np.float32)
    before = x.copy()
    first = geoadapt_v2.derive_four_band_spd(x)
    second = geoadapt_v2.derive_four_band_spd(x)
    assert np.array_equal(x, before)
    assert np.array_equal(first, second)
    assert first.shape == (3, 4, 5, 5)
    assert first.dtype == np.float32
    assert np.min(np.linalg.eigvalsh(first.astype(np.float64))) > 0.0


@pytest.mark.parametrize(
    "epochs",
    (
        np.zeros((2, 4, 3, 96), dtype=np.float32),
        np.ones((2, 4, 3, 96), dtype=np.float32),
        np.broadcast_to(
            np.linspace(-1.0, 1.0, 96, dtype=np.float32),
            (2, 4, 3, 96),
        ).copy(),
        (
            np.broadcast_to(
                np.linspace(-1.0, 1.0, 96, dtype=np.float32),
                (2, 4, 3, 96),
            ).copy()
            * np.float32(1e-30)
        ),
    ),
    ids=("zero", "constant", "rank_deficient", "tiny"),
)
def test_spd_covariances_are_positive_in_persisted_float32(
    epochs: np.ndarray,
) -> None:
    result = geoadapt_v2.spd_covariances(epochs)
    assert result.dtype == np.float32
    assert np.all(np.isfinite(result))
    assert np.min(np.linalg.eigvalsh(result.astype(np.float64))) > 0.0


def test_prepare_job_scalers_are_source_only(tmp_path: Path) -> None:
    plan = _tiny_plan()
    job = next(geoadapt_v2.iter_jobs(plan))
    baseline = geoadapt_v2.prepare_job(
        job=job,
        plan=plan,
        cache_root=tmp_path,
        cache_loader=_fake_loader,
        splitter=_fake_splitter,
    )
    changed = _fake_data()
    changed["x"][6:] = np.float32(1e6)

    def changed_loader(*args, **kwargs):
        del args, kwargs
        return changed

    modified = geoadapt_v2.prepare_job(
        job=job,
        plan=plan,
        cache_root=tmp_path,
        cache_loader=changed_loader,
        splitter=_fake_splitter,
    )
    assert np.array_equal(baseline.selection_mean, modified.selection_mean)
    assert np.array_equal(baseline.selection_std, modified.selection_std)
    assert np.array_equal(baseline.refit_mean, modified.refit_mean)
    assert np.array_equal(baseline.refit_std, modified.refit_std)
    assert not hasattr(baseline, "y_test")
    assert not np.array_equal(baseline.x_test, modified.x_test)


def test_injected_executor_withholds_test_outcomes_and_commits_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _allow_tiny_plan(monkeypatch)
    plan = _tiny_plan()
    geoadapt_v2.write_or_validate_plan(tmp_path, plan)
    job = next(geoadapt_v2.iter_jobs(plan))
    calls = 0

    def backend(**kwargs):
        nonlocal calls
        calls += 1
        return _tiny_backend(**kwargs)

    monkeypatch.setattr(geoadapt_v2, "_runtime_identity", _runtime_stub)
    metadata, rows, probabilities = geoadapt_v2.execute_job(
        job=job,
        plan=plan,
        cache_root=tmp_path,
        device="cpu",
        backend=backend,
        cache_loader=_fake_loader,
        splitter=_fake_splitter,
    )
    assert calls == 1
    assert not geoadapt_v2._recursive_forbidden_keys(metadata)
    claim = _claim(job, tmp_path, plan)
    output = geoadapt_v2.commit_job_output(
        tmp_path,
        plan,
        claim,
        metadata=metadata,
        test_rows=rows,
        probabilities=probabilities,
    )
    assert output.is_dir()
    artifact = geoadapt_v2.validate_completion(tmp_path, plan, job)
    assert np.array_equal(artifact["rows"], rows)
    assert np.array_equal(artifact["probabilities"], probabilities)
    geoadapt_v2.release_claim(claim, plan)
    assert geoadapt_v2.audit_grid(tmp_path, plan)["exact_cartesian_complete"]


def test_plan_two_file_transaction_recovers_after_first_final_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_tiny_plan(monkeypatch)
    plan = _tiny_plan()
    real_rename = geoadapt_v2._rename_noreplace
    digest_path = tmp_path / "plan.sha256"

    def power_cut(source, destination, **kwargs):
        if geoadapt_v2._absolute_path(destination) == digest_path:
            raise OSError("simulated power cut")
        return real_rename(source, destination, **kwargs)

    monkeypatch.setattr(geoadapt_v2, "_rename_noreplace", power_cut)
    with pytest.raises(OSError, match="power cut"):
        geoadapt_v2.write_or_validate_plan(tmp_path, plan)
    assert (tmp_path / "plan.json").is_file()
    assert not digest_path.exists()
    assert (tmp_path / ".plan.publication.transaction").is_dir()

    monkeypatch.setattr(geoadapt_v2, "_rename_noreplace", real_rename)
    assert geoadapt_v2.write_or_validate_plan(tmp_path, plan) == plan
    assert geoadapt_v2.load_plan(tmp_path) == plan
    assert not (tmp_path / ".plan.publication.transaction").exists()


def test_record_directory_is_sealed_before_publish_and_writable_mode_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, job, claim, metadata, rows, probabilities = _commit_inputs(
        tmp_path,
        monkeypatch,
    )
    output = geoadapt_v2.commit_job_output(
        tmp_path,
        plan,
        claim,
        metadata=metadata,
        test_rows=rows,
        probabilities=probabilities,
    )
    assert stat.S_IMODE(output.stat().st_mode) == 0o555
    geoadapt_v2.validate_completion(tmp_path, plan, job)

    os.chmod(output, 0o755)
    with pytest.raises(
        geoadapt_v2.GeoAdaptV2Error,
        match="immutable 0555",
    ):
        geoadapt_v2.validate_completion(tmp_path, plan, job)
    os.chmod(output, 0o555)
    geoadapt_v2.release_claim(claim, plan)


def test_prediction_npz_duplicate_raw_member_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, job, claim, metadata, rows, probabilities = _commit_inputs(
        tmp_path,
        monkeypatch,
    )
    output = geoadapt_v2.commit_job_output(
        tmp_path,
        plan,
        claim,
        metadata=metadata,
        test_rows=rows,
        probabilities=probabilities,
    )
    prediction_path = output / "predictions.npz"
    original = prediction_path.read_bytes()
    with zipfile.ZipFile(io.BytesIO(original), mode="r") as archive:
        duplicate_payload = archive.read("rows.npy")
    attacked = io.BytesIO(original)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(attacked, mode="a") as archive:
            archive.writestr("rows.npy", duplicate_payload)

    os.chmod(output, 0o755)
    os.chmod(prediction_path, 0o600)
    prediction_path.write_bytes(attacked.getvalue())
    os.chmod(prediction_path, 0o444)
    completion_path = output / "completion.json"
    completion = geoadapt_v2.strict_load(completion_path)
    completion["files"]["predictions.npz"] = geoadapt_v2._sha256_file(prediction_path)
    geoadapt_v2._atomic_json(completion_path, completion)
    os.chmod(output, 0o555)

    with pytest.raises(
        geoadapt_v2.GeoAdaptV2Error,
        match="exact ZIP member order",
    ):
        geoadapt_v2.validate_completion(tmp_path, plan, job)
    geoadapt_v2.release_claim(claim, plan)


def test_copied_record_cannot_select_its_original_disk_as_new_run_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source-run"
    copied_root = tmp_path / "copied-run"
    plan, job, claim, metadata, rows, probabilities = _commit_inputs(
        source_root,
        monkeypatch,
    )
    source = geoadapt_v2.commit_job_output(
        source_root,
        plan,
        claim,
        metadata=metadata,
        test_rows=rows,
        probabilities=probabilities,
    )
    geoadapt_v2.write_or_validate_plan(copied_root, plan)
    copied = geoadapt_v2._record_directory(copied_root, job)
    copied.parent.mkdir(parents=True)
    shutil.copytree(source, copied)
    with pytest.raises(
        geoadapt_v2.GeoAdaptV2Error,
        match="claim disk guard",
    ):
        geoadapt_v2.validate_completion(copied_root, plan, job)
    geoadapt_v2.release_claim(claim, plan)


def test_record_move_then_helper_failure_quarantines_exact_published_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, job, claim, metadata, rows, probabilities = _commit_inputs(
        tmp_path,
        monkeypatch,
    )
    destination = geoadapt_v2._record_directory(tmp_path, job)
    real_rename = geoadapt_v2._rename_noreplace
    moved_identity: tuple[int, int] | None = None
    rebind_states: list[bool] = []

    def observe_rebind(**kwargs):
        rebind_states.append(destination.exists())
        return kwargs["plan"]

    def move_then_fail(source, target, **kwargs):
        nonlocal moved_identity
        real_rename(source, target, **kwargs)
        if geoadapt_v2._absolute_path(target) == destination:
            observed = geoadapt_v2._anchored_lstat(destination)
            moved_identity = (int(observed.st_dev), int(observed.st_ino))
            raise OSError("simulated post-move record failure")

    monkeypatch.setattr(
        geoadapt_v2,
        "_rebind_authoritative_inputs",
        observe_rebind,
    )
    monkeypatch.setattr(geoadapt_v2, "_rename_noreplace", move_then_fail)
    with pytest.raises(OSError, match="post-move record failure"):
        geoadapt_v2.commit_job_output(
            tmp_path,
            plan,
            claim,
            metadata=metadata,
            test_rows=rows,
            probabilities=probabilities,
        )
    assert moved_identity is not None
    assert rebind_states == [False, True]
    assert not destination.exists()
    quarantined = tuple((tmp_path / "quarantine" / "records").iterdir())
    assert len(quarantined) == 1
    assert (
        int(quarantined[0].stat().st_dev),
        int(quarantined[0].stat().st_ino),
    ) == moved_identity


def test_record_commit_rebinds_all_authorities_immediately_around_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, job, claim, metadata, rows, probabilities = _commit_inputs(
        tmp_path,
        monkeypatch,
    )
    destination = geoadapt_v2._record_directory(tmp_path, job)
    events: list[tuple[str, bool]] = []
    real_rename = geoadapt_v2._rename_noreplace

    def rebind(**kwargs):
        assert kwargs["cache_root"] == tmp_path
        assert kwargs["job"] == job
        assert kwargs["claim"] == claim
        events.append(("rebind", destination.exists()))
        return plan

    def observe_rename(source, target, **kwargs):
        if geoadapt_v2._absolute_path(target) == destination:
            events.append(("rename-before", destination.exists()))
            real_rename(source, target, **kwargs)
            events.append(("rename-after", destination.exists()))
            return
        real_rename(source, target, **kwargs)

    monkeypatch.setattr(geoadapt_v2, "_rebind_authoritative_inputs", rebind)
    monkeypatch.setattr(geoadapt_v2, "_rename_noreplace", observe_rename)
    geoadapt_v2.commit_job_output(
        tmp_path,
        plan,
        claim,
        cache_root=tmp_path,
        metadata=metadata,
        test_rows=rows,
        probabilities=probabilities,
    )
    assert events == [
        ("rebind", False),
        ("rename-before", False),
        ("rename-after", True),
        ("rebind", True),
    ]


def test_record_postcommit_baseexception_hides_exact_published_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, job, claim, metadata, rows, probabilities = _commit_inputs(
        tmp_path,
        monkeypatch,
    )
    destination = geoadapt_v2._record_directory(tmp_path, job)
    calls = 0

    def interrupt_second_rebind(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            assert destination.exists()
            raise SyntheticPublicationInterrupt("simulated post-identity interrupt")
        return kwargs["plan"]

    monkeypatch.setattr(
        geoadapt_v2,
        "_rebind_authoritative_inputs",
        interrupt_second_rebind,
    )
    with pytest.raises(
        SyntheticPublicationInterrupt,
        match="post-identity interrupt",
    ):
        geoadapt_v2.commit_job_output(
            tmp_path,
            plan,
            claim,
            cache_root=tmp_path,
            metadata=metadata,
            test_rows=rows,
            probabilities=probabilities,
        )
    assert calls == 2
    assert not destination.exists()
    quarantined = tuple((tmp_path / "quarantine" / "records").iterdir())
    assert len(quarantined) == 1
    assert claim.publication_state.record_identity is None


def test_claim_staged_publication_is_quarantined_and_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_tiny_plan(monkeypatch)
    plan = _tiny_plan()
    geoadapt_v2.write_or_validate_plan(tmp_path, plan)
    job = next(geoadapt_v2.iter_jobs(plan))
    claim_path = geoadapt_v2._claim_path(tmp_path, job)
    real_rename = geoadapt_v2._rename_noreplace

    def power_cut(source, destination, **kwargs):
        if geoadapt_v2._absolute_path(destination) == claim_path:
            raise OSError("simulated claim power cut")
        return real_rename(source, destination, **kwargs)

    monkeypatch.setattr(geoadapt_v2, "_rename_noreplace", power_cut)
    with pytest.raises(OSError, match="claim power cut"):
        geoadapt_v2._write_json_exclusive(claim_path, {"torn": True})
    assert not claim_path.exists()
    assert len(geoadapt_v2._staged_publication_paths(claim_path)) == 1

    monkeypatch.setattr(geoadapt_v2, "_rename_noreplace", real_rename)
    monkeypatch.setattr(
        geoadapt_v2,
        "_process_identity",
        lambda: {
            "host": "tiny-host",
            "pid": 1,
            "start_marker": "tiny-start",
            "boot_marker": "tiny-boot",
        },
    )
    claim = geoadapt_v2.acquire_claim(tmp_path, plan, job, device="cpu")
    assert geoadapt_v2.strict_load(claim.path)["nonce"] == claim.nonce
    recovered = tuple((tmp_path / "quarantine" / "claim-publications").iterdir())
    assert len(recovered) == 1
    assert ".powercut." in recovered[0].name
    geoadapt_v2.release_claim(claim, plan)


def test_analysis_gate_staged_publication_is_quarantined_and_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_tiny_plan(monkeypatch)
    plan = _tiny_plan()
    geoadapt_v2.write_or_validate_plan(tmp_path, plan)
    marker = geoadapt_v2._analysis_gate_path(tmp_path)
    real_rename = geoadapt_v2._rename_noreplace

    def power_cut(source, destination, **kwargs):
        if geoadapt_v2._absolute_path(destination) == marker:
            raise OSError("simulated gate power cut")
        return real_rename(source, destination, **kwargs)

    monkeypatch.setattr(geoadapt_v2, "_rename_noreplace", power_cut)
    with pytest.raises(OSError, match="gate power cut"):
        geoadapt_v2.acquire_analysis_gate(tmp_path, plan)
    assert not marker.exists()
    assert len(geoadapt_v2._staged_publication_paths(marker)) == 1

    monkeypatch.setattr(geoadapt_v2, "_rename_noreplace", real_rename)
    gate = geoadapt_v2.acquire_analysis_gate(tmp_path, plan)
    recovered = tuple(
        (tmp_path / "quarantine" / "analysis-gate-publications").iterdir()
    )
    assert len(recovered) == 1
    geoadapt_v2.release_analysis_gate(gate)


def test_publication_lock_staged_write_is_quarantined_and_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _tiny_plan()
    output_root = tmp_path / "analysis"
    lock_path = geoadapt_v2_analysis._publication_lock_path(output_root)
    real_rename = geoadapt_v2._rename_noreplace

    def power_cut(source, destination, **kwargs):
        if geoadapt_v2._absolute_path(destination) == lock_path:
            raise OSError("simulated publication-lock power cut")
        return real_rename(source, destination, **kwargs)

    monkeypatch.setattr(geoadapt_v2, "_rename_noreplace", power_cut)
    with pytest.raises(OSError, match="publication-lock power cut"):
        geoadapt_v2_analysis._acquire_publication_lock(output_root, plan)
    assert not lock_path.exists()
    assert len(geoadapt_v2._staged_publication_paths(lock_path)) == 1

    monkeypatch.setattr(geoadapt_v2, "_rename_noreplace", real_rename)
    lock = geoadapt_v2_analysis._acquire_publication_lock(output_root, plan)
    quarantine = tmp_path / ".geoadapt-v2-publication-quarantine"
    assert len(tuple(quarantine.iterdir())) == 1
    geoadapt_v2_analysis._release_publication_lock(lock)


def test_failure_staged_publication_is_quarantined_and_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_tiny_plan(monkeypatch)
    plan = _tiny_plan()
    geoadapt_v2.write_or_validate_plan(tmp_path, plan)
    job = next(geoadapt_v2.iter_jobs(plan))
    claim = _claim(job, tmp_path, plan)
    expected = (
        geoadapt_v2._failure_root(tmp_path, job) / f"{job.job_id}.attempt-0001.json"
    )
    real_rename = geoadapt_v2._rename_noreplace

    def power_cut(source, destination, **kwargs):
        if geoadapt_v2._absolute_path(destination) == expected:
            raise OSError("simulated failure-record power cut")
        return real_rename(source, destination, **kwargs)

    monkeypatch.setattr(geoadapt_v2, "_rename_noreplace", power_cut)
    with pytest.raises(OSError, match="failure-record power cut"):
        geoadapt_v2._record_failure(
            tmp_path,
            plan,
            claim,
            RuntimeError("first"),
        )
    assert not expected.exists()
    assert len(geoadapt_v2._staged_publication_paths(expected)) == 1

    monkeypatch.setattr(geoadapt_v2, "_rename_noreplace", real_rename)
    path = geoadapt_v2._record_failure(
        tmp_path,
        plan,
        claim,
        RuntimeError("second"),
    )
    assert path == expected
    assert geoadapt_v2._validate_failure_record(path, plan=plan, job=job)
    recovered = tuple((tmp_path / "quarantine" / "failure-publications").iterdir())
    assert len(recovered) == 1
    geoadapt_v2.release_claim(claim, plan)


def test_power_cut_claim_and_partial_recovery_is_quarantined(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _allow_tiny_plan(monkeypatch)
    plan = _tiny_plan()
    geoadapt_v2.write_or_validate_plan(tmp_path, plan)
    job = next(geoadapt_v2.iter_jobs(plan))
    monkeypatch.setattr(geoadapt_v2, "_runtime_identity", _runtime_stub)
    metadata, rows, probabilities = geoadapt_v2.execute_job(
        job=job,
        plan=plan,
        cache_root=tmp_path,
        device="cpu",
        backend=_tiny_backend,
        cache_loader=_fake_loader,
        splitter=_fake_splitter,
    )
    claim = _claim(job, tmp_path, plan)
    geoadapt_v2.commit_job_output(
        tmp_path,
        plan,
        claim,
        metadata=metadata,
        test_rows=rows,
        probabilities=probabilities,
    )
    os.chmod(claim.path, 0o600)
    claim.path.unlink()
    geoadapt_v2._write_json_exclusive(
        claim.path,
        {
            "schema": geoadapt_v2.CLAIM_SCHEMA,
            "created_at": geoadapt_v2._utc_now(),
            "plan_sha256": plan["plan_sha256"],
            "job_id": job.job_id,
            "job": job.identity(),
            "nonce": claim.nonce,
            "owner": {
                "host": geoadapt_v2.socket.gethostname(),
                "pid": 999_999_999,
                "start_marker": "dead",
                "boot_marker": geoadapt_v2._boot_marker(),
            },
            "resource_guard": {
                **copy.deepcopy(dict(claim.resource_guard)),
                "device_resource": {
                    **copy.deepcopy(dict(claim.resource_guard["device_resource"])),
                    "node_inventory": {
                        "host": geoadapt_v2.socket.gethostname(),
                        "cuda": None,
                    },
                },
            },
            "gpu_lease_receipt": None,
        },
    )
    partial = tmp_path / "partials" / f"{job.job_id}.dead.partial"
    partial.mkdir(parents=True)
    with pytest.raises(geoadapt_v2.ClaimUnavailable, match="already complete"):
        geoadapt_v2.acquire_claim(
            tmp_path,
            plan,
            job,
            device="cpu",
            recover_stale=True,
        )
    recovered = tuple(
        path
        for path in (tmp_path / "quarantine").rglob("*")
        if path.is_file() or path.is_dir()
    )
    recovered_leaves = tuple(
        path
        for path in recovered
        if ".postcommit." in path.name or ".recovered." in path.name
    )
    assert len(recovered_leaves) == 2
    audit = geoadapt_v2.audit_grid(tmp_path, plan)
    assert audit["quiescent"] is True
    assert audit["exact_cartesian_complete"] is True


def test_partial_recovery_never_quarantines_a_replacement_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_tiny_plan(monkeypatch)
    plan = _tiny_plan()
    geoadapt_v2.write_or_validate_plan(tmp_path, plan)
    job = next(geoadapt_v2.iter_jobs(plan))
    partial_root = tmp_path / "partials"
    partial_root.mkdir()
    partial = partial_root / f"{job.job_id}.abandoned.partial"
    partial.write_bytes(b"validated-loser")
    displaced = tmp_path / "displaced-partial"
    real_quarantine = geoadapt_v2._quarantine
    attacked = False

    def replace_then_quarantine(run_root, path, **kwargs):
        nonlocal attacked
        if geoadapt_v2._absolute_path(path) == partial and not attacked:
            os.replace(partial, displaced)
            partial.write_bytes(b"replacement-must-survive")
            attacked = True
        return real_quarantine(run_root, path, **kwargs)

    monkeypatch.setattr(geoadapt_v2, "_quarantine", replace_then_quarantine)
    with pytest.raises(
        geoadapt_v2.ClaimUnavailable,
        match="identity changed",
    ):
        geoadapt_v2._quarantine_partials(tmp_path, job)
    assert attacked
    assert partial.read_bytes() == b"replacement-must-survive"
    assert displaced.read_bytes() == b"validated-loser"


def test_stale_claim_recovery_never_quarantines_a_replacement_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_tiny_plan(monkeypatch)
    plan = _tiny_plan()
    geoadapt_v2.write_or_validate_plan(tmp_path, plan)
    job = next(geoadapt_v2.iter_jobs(plan))
    claim = _claim(job, tmp_path, plan)
    displaced = tmp_path / "displaced-claim"
    real_quarantine = geoadapt_v2._quarantine
    attacked = False
    monkeypatch.setattr(
        geoadapt_v2,
        "_owner_state",
        lambda *args, **kwargs: "dead_local",
    )

    def replace_then_quarantine(run_root, path, **kwargs):
        nonlocal attacked
        if geoadapt_v2._absolute_path(path) == claim.path and not attacked:
            os.replace(claim.path, displaced)
            claim.path.write_bytes(b"replacement-must-survive")
            attacked = True
        return real_quarantine(run_root, path, **kwargs)

    monkeypatch.setattr(geoadapt_v2, "_quarantine", replace_then_quarantine)
    with pytest.raises(
        geoadapt_v2.ClaimUnavailable,
        match="identity changed",
    ):
        geoadapt_v2._recover_existing_claim_authority(
            tmp_path,
            plan,
            job,
            claim.path,
            recover_stale=True,
            foreign_claim_timeout_seconds=0.0,
        )
    assert attacked
    assert claim.path.read_bytes() == b"replacement-must-survive"
    assert displaced.exists()


def test_analysis_lock_recovery_never_quarantines_a_replacement_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _tiny_plan()
    output_root = tmp_path / "analysis"
    lock = geoadapt_v2_analysis._publication_lock_path(output_root)
    geoadapt_v2._write_json_exclusive(lock, {"malformed": True})
    displaced = tmp_path / "displaced-analysis-lock"
    real_quarantine = geoadapt_v2_analysis._quarantine_publication_residue
    attacked = False

    def replace_then_quarantine(path, **kwargs):
        nonlocal attacked
        if geoadapt_v2._absolute_path(path) == lock and not attacked:
            os.replace(lock, displaced)
            lock.write_bytes(b"replacement-must-survive")
            attacked = True
        return real_quarantine(path, **kwargs)

    monkeypatch.setattr(
        geoadapt_v2_analysis,
        "_quarantine_publication_residue",
        replace_then_quarantine,
    )
    with pytest.raises(
        geoadapt_v2_analysis.PublicationContention,
        match="inode changed",
    ):
        geoadapt_v2_analysis._acquire_publication_lock(output_root, plan)
    assert attacked
    assert lock.read_bytes() == b"replacement-must-survive"
    assert displaced.exists()


def test_analysis_lock_a_to_b_swap_preserves_b_lock_and_referenced_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _tiny_plan()
    output_root = tmp_path / "analysis"
    lock = geoadapt_v2_analysis._publication_lock_path(output_root)
    owner = {
        "host": "dead-host",
        "pid": 999_999_999,
        "start_marker": "dead-start",
        "boot_marker": "dead-boot",
    }
    nonce_a = "aaaaaaaaaaaa4aaa8aaaaaaaaaaaaaaa"
    nonce_b = "bbbbbbbbbbbb4bbb8bbbbbbbbbbbbbbb"

    def lock_payload(nonce: str) -> dict[str, object]:
        return {
            "schema": "eeg-mi-geoadapt-v2-analysis-lock-v2",
            "created_at": geoadapt_v2._utc_now(),
            "plan_sha256": plan["plan_sha256"],
            "output_root": str(output_root),
            "nonce": nonce,
            "owner": owner,
        }

    geoadapt_v2._write_json_exclusive(lock, lock_payload(nonce_a))
    partial_a = tmp_path / f".analysis.{nonce_a}.partial"
    partial_b = tmp_path / f".analysis.{nonce_b}.partial"
    partial_a.mkdir()
    partial_b.mkdir()
    (partial_a / "owner").write_bytes(b"A")
    (partial_b / "owner").write_bytes(b"B")
    displaced_lock = tmp_path / "lock-a-displaced"
    real_bound_read = geoadapt_v2_analysis._read_expected_publication_lock
    swapped = False

    def swap_to_b_before_bound_read(path, *, expected_identity):
        nonlocal swapped
        if not swapped:
            os.replace(lock, displaced_lock)
            geoadapt_v2._write_json_exclusive(lock, lock_payload(nonce_b))
            swapped = True
        return real_bound_read(path, expected_identity=expected_identity)

    monkeypatch.setattr(
        geoadapt_v2_analysis,
        "_read_expected_publication_lock",
        swap_to_b_before_bound_read,
    )
    with pytest.raises(
        geoadapt_v2_analysis.PublicationContention,
        match="inode changed",
    ):
        geoadapt_v2_analysis._acquire_publication_lock(output_root, plan)
    assert swapped
    assert geoadapt_v2.strict_load(lock)["nonce"] == nonce_b
    assert (partial_a / "owner").read_bytes() == b"A"
    assert (partial_b / "owner").read_bytes() == b"B"
    quarantine = tmp_path / ".geoadapt-v2-publication-quarantine"
    assert not quarantine.exists() or not tuple(quarantine.iterdir())


def test_claim_symlink_is_rejected_without_following_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_tiny_plan(monkeypatch)
    plan = _tiny_plan()
    geoadapt_v2.write_or_validate_plan(tmp_path, plan)
    job = next(geoadapt_v2.iter_jobs(plan))
    outside = tmp_path / "outside-sentinel.json"
    sentinel = b'{"do_not_touch":true}\n'
    outside.write_bytes(sentinel)
    claim_path = geoadapt_v2._claim_path(tmp_path, job)
    claim_path.parent.mkdir(parents=True)
    claim_path.symlink_to(outside)
    with pytest.raises(geoadapt_v2.GeoAdaptV2Error, match="symbolic links"):
        geoadapt_v2.acquire_claim(
            tmp_path,
            plan,
            job,
            device="cpu",
            recover_stale=True,
        )
    assert outside.read_bytes() == sentinel
    assert claim_path.is_symlink()


def test_malformed_claim_is_quarantined_but_foreign_claim_is_never_stolen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_tiny_plan(monkeypatch)
    plan = _tiny_plan()
    geoadapt_v2.write_or_validate_plan(tmp_path, plan)
    job = next(geoadapt_v2.iter_jobs(plan))
    claim_path = geoadapt_v2._claim_path(tmp_path, job)
    geoadapt_v2._write_json_exclusive(claim_path, {"malformed": True})
    monkeypatch.setattr(
        geoadapt_v2,
        "_process_identity",
        lambda: {
            "host": "tiny-host",
            "pid": 1,
            "start_marker": "tiny-start",
            "boot_marker": "tiny-boot",
        },
    )
    recovered_claim = geoadapt_v2.acquire_claim(
        tmp_path,
        plan,
        job,
        device="cpu",
    )
    assert claim_path.is_file()
    malformed = tuple((tmp_path / "quarantine" / "claims").iterdir())
    assert len(malformed) == 1
    assert ".malformed." in malformed[0].name
    geoadapt_v2.release_claim(recovered_claim, plan)
    valid_claim = _claim(job, tmp_path, plan)
    os.chmod(valid_claim.path, 0o600)
    valid_claim.path.unlink()
    geoadapt_v2._write_json_exclusive(
        claim_path,
        {
            "schema": geoadapt_v2.CLAIM_SCHEMA,
            "created_at": geoadapt_v2._utc_now(),
            "plan_sha256": plan["plan_sha256"],
            "job_id": job.job_id,
            "job": job.identity(),
            "nonce": "aaaaaaaaaaaa4aaa8aaaaaaaaaaaaaaa",
            "owner": {
                "host": "other-node",
                "pid": 1234,
                "start_marker": "unknown",
                "boot_marker": "other-boot",
            },
            "resource_guard": {
                **copy.deepcopy(dict(valid_claim.resource_guard)),
                "device_resource": {
                    **copy.deepcopy(
                        dict(valid_claim.resource_guard["device_resource"])
                    ),
                    "node_inventory": {"host": "other-node", "cuda": None},
                },
            },
            "gpu_lease_receipt": None,
        },
    )
    with pytest.raises(geoadapt_v2.ClaimUnavailable, match="foreign_frozen"):
        geoadapt_v2.acquire_claim(tmp_path, plan, job, device="cpu")
    with pytest.raises(geoadapt_v2.GeoAdaptV2Error, match="timeouts"):
        geoadapt_v2.acquire_claim(
            tmp_path,
            plan,
            job,
            device="cpu",
            foreign_claim_timeout_seconds=1.0,
        )


def test_unsafe_dataset_path_component_is_rejected() -> None:
    plan = _tiny_plan()
    with pytest.raises(ValueError, match="unsafe dataset identity"):
        geoadapt_v2.assemble_plan(
            dataset_contracts={"../escape": plan["datasets"]["tiny"]},
            stable_configs=plan["stable_configs"],
            eligibility={
                "../escape": {
                    "architecture.geoadaptnet": {
                        "status": "eligible",
                        "reason": None,
                    }
                }
            },
            seeds=(7,),
            cache_identity={
                "../escape:s001": {
                    "array_sha256": "a" * 64,
                }
            },
            split_identity={
                "../escape:s001:f00": plan["split_identity"]["tiny:s001:f00"]
            },
            source_identity={"runner.py": "b" * 64},
            environment_identity={"python": "test"},
            project_identity={"lock": "test"},
            feature_contract={"schema": "test"},
        )


def test_filesystem_helpers_reject_ancestor_symlinks_and_special_leaves(
    tmp_path: Path,
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(geoadapt_v2.GeoAdaptV2Error, match="alias|symbolic"):
        geoadapt_v2._write_json_exclusive(alias / "artifact.json", {"ok": True})
    assert not (real / "artifact.json").exists()

    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(geoadapt_v2.GeoAdaptV2Error, match="regular file|special"):
        geoadapt_v2._safe_read_bytes(fifo)


def test_safe_read_rejects_leaf_inode_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "target.bin"
    replacement = tmp_path / "replacement.bin"
    target.write_bytes(b"same bytes")
    replacement.write_bytes(b"same bytes")
    real_read = geoadapt_v2.os.read
    replaced = False

    def attacked_read(descriptor: int, count: int) -> bytes:
        nonlocal replaced
        chunk = real_read(descriptor, count)
        if chunk and not replaced:
            replaced = True
            os.replace(replacement, target)
        return chunk

    monkeypatch.setattr(geoadapt_v2.os, "read", attacked_read)
    with pytest.raises(
        geoadapt_v2.GeoAdaptV2Error, match="changed while being read|inode"
    ):
        geoadapt_v2._safe_read_bytes(target)


def test_anchored_noreplace_rename_rejects_replaced_source_inode(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.partial"
    source.mkdir()
    identity = geoadapt_v2._anchored_lstat(source)
    original = tmp_path / "original.partial"
    source.rename(original)
    source.mkdir()
    destination = tmp_path / "published"
    with pytest.raises(geoadapt_v2.GeoAdaptV2Error, match="inode changed"):
        geoadapt_v2._rename_noreplace(
            source,
            destination,
            expected_source_dev=int(identity.st_dev),
            expected_source_ino=int(identity.st_ino),
        )
    assert original.is_dir()
    assert source.is_dir()
    assert not destination.exists()


@pytest.mark.parametrize(
    "alias",
    (
        "Y_TEST",
        "ground-Truth",
        "actual outcomes",
        "Predicted.Labels",
        "metric_payload",
        "Accuracy (%)",
        "class__labels",
    ),
)
def test_score_blind_forbidden_aliases_are_normalized(alias: str) -> None:
    violations = geoadapt_v2._recursive_forbidden_keys(
        {"safe": {"nested": {alias: [0, 1]}}}
    )
    assert violations == [f"record.safe.nested.{alias}"]


@pytest.mark.parametrize("bad_pid", (True, False, None, "12", 0, -1))
def test_owner_schema_rejects_bool_or_nonpositive_pid(bad_pid: object) -> None:
    with pytest.raises(geoadapt_v2.GeoAdaptV2Error, match="owner object"):
        geoadapt_v2._validate_owner_mapping(
            {
                "host": "host",
                "pid": bad_pid,
                "start_marker": "start",
                "boot_marker": "boot",
            }
        )


@pytest.mark.parametrize("missing_marker", ("start_marker", "boot_marker"))
def test_owner_schema_rejects_null_markers(missing_marker: str) -> None:
    owner = {
        "host": "host",
        "pid": 12,
        "start_marker": "start",
        "boot_marker": "boot",
    }
    owner[missing_marker] = None
    with pytest.raises(geoadapt_v2.GeoAdaptV2Error, match="owner object"):
        geoadapt_v2._validate_owner_mapping(owner)


def test_claim_hardlink_is_rejected_without_chmod_external(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_tiny_plan(monkeypatch)
    plan = _tiny_plan()
    geoadapt_v2.write_or_validate_plan(tmp_path, plan)
    job = next(geoadapt_v2.iter_jobs(plan))
    external = tmp_path / "external-claim"
    external.write_bytes(b"external\n")
    os.chmod(external, 0o640)
    mode_before = external.stat().st_mode
    claim_path = geoadapt_v2._claim_path(tmp_path, job)
    claim_path.parent.mkdir(parents=True)
    os.link(external, claim_path)
    with pytest.raises(geoadapt_v2.GeoAdaptV2Error, match="hard link"):
        geoadapt_v2.acquire_claim(tmp_path, plan, job, device="cpu")
    assert external.read_bytes() == b"external\n"
    assert external.stat().st_mode == mode_before


def test_record_hardlink_is_rejected_without_chmod_external(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, job, claim, metadata, rows, probabilities = _commit_inputs(
        tmp_path, monkeypatch
    )
    output = geoadapt_v2.commit_job_output(
        tmp_path,
        plan,
        claim,
        metadata=metadata,
        test_rows=rows,
        probabilities=probabilities,
    )
    record = output / "record.json"
    external = tmp_path / "record-copy"
    os.link(record, external)
    mode_before = external.stat().st_mode
    with pytest.raises(geoadapt_v2.GeoAdaptV2Error, match="hard link"):
        geoadapt_v2.validate_completion(tmp_path, plan, job)
    assert external.stat().st_mode == mode_before


def test_completion_hashes_and_decodes_one_prediction_byte_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, job, claim, metadata, rows, probabilities = _commit_inputs(
        tmp_path, monkeypatch
    )
    output = geoadapt_v2.commit_job_output(
        tmp_path,
        plan,
        claim,
        metadata=metadata,
        test_rows=rows,
        probabilities=probabilities,
    )
    prediction_path = output / "predictions.npz"
    real_load = geoadapt_v2.np.load
    prediction_decodes = 0

    def counted_load(source, *args, **kwargs):
        nonlocal prediction_decodes
        if isinstance(source, io.BytesIO):
            prediction_decodes += 1
        return real_load(source, *args, **kwargs)

    monkeypatch.setattr(geoadapt_v2.np, "load", counted_load)
    artifact = geoadapt_v2.validate_completion(tmp_path, plan, job)
    assert prediction_decodes == 1
    assert artifact["artifact_bytes"]["predictions.npz"] == prediction_path.read_bytes()
    assert artifact["artifact_sha256"]["predictions.npz"] == hashlib.sha256(
        artifact["artifact_bytes"]["predictions.npz"]
    ).hexdigest()


def test_completion_snapshot_rejects_coherent_a_b_a_package_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, job, claim, metadata, rows, probabilities = _commit_inputs(
        tmp_path,
        monkeypatch,
    )
    canonical = geoadapt_v2.commit_job_output(
        tmp_path,
        plan,
        claim,
        metadata=metadata,
        test_rows=rows,
        probabilities=probabilities,
    )
    held_a = tmp_path / "held-package-a"
    package_b = tmp_path / "package-b"
    held_a.mkdir()
    shutil.copytree(canonical, package_b)
    os.chmod(package_b, 0o755)
    reversed_probabilities = np.ascontiguousarray(
        probabilities[:, ::-1],
        dtype=np.float64,
    )
    os.chmod(package_b / "predictions.npz", 0o600)
    with (package_b / "predictions.npz").open("wb") as handle:
        np.savez_compressed(
            handle,
            rows=np.ascontiguousarray(rows, dtype=np.int64),
            probabilities=reversed_probabilities,
        )
    os.chmod(package_b / "predictions.npz", 0o444)
    completion_b = geoadapt_v2.strict_load(package_b / "completion.json")
    completion_b["files"]["predictions.npz"] = geoadapt_v2._sha256_file(
        package_b / "predictions.npz"
    )
    geoadapt_v2._atomic_json(package_b / "completion.json", completion_b)
    os.chmod(package_b, 0o555)

    real_read = geoadapt_v2.os.read
    read_calls = 0
    package_is_b = False

    def exchange(source: Path, destination: Path) -> None:
        os.chmod(canonical, 0o755)
        os.chmod(source, 0o755)
        for name in sorted(geoadapt_v2.FINAL_FILENAMES):
            os.replace(canonical / name, destination / name)
            os.replace(source / name, canonical / name)
        os.chmod(canonical, 0o555)

    def toggle_a_b_a(descriptor: int, maximum: int) -> bytes:
        nonlocal read_calls, package_is_b
        if read_calls == 0:
            exchange(package_b, held_a)
            package_is_b = True
        payload = real_read(descriptor, maximum)
        read_calls += 1
        if read_calls == 6:
            exchange(held_a, package_b)
            package_is_b = False
        return payload

    monkeypatch.setattr(geoadapt_v2.os, "read", toggle_a_b_a)
    with pytest.raises(
        geoadapt_v2.GeoAdaptV2Error,
        match=r"changed (during|while) completion package",
    ):
        geoadapt_v2.validate_completion(tmp_path, plan, job)
    assert read_calls == 6
    assert package_is_b is False
    live = geoadapt_v2.validate_completion(tmp_path, plan, job)
    assert np.array_equal(live["probabilities"], probabilities)


@pytest.mark.parametrize(
    "mutation",
    ("record_extra", "record_bool_count", "completion_extra", "completion_bad_files"),
)
def test_record_and_completion_exact_schemas_reject_mutations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    plan, job, claim, metadata, rows, probabilities = _commit_inputs(
        tmp_path, monkeypatch
    )
    output = geoadapt_v2.commit_job_output(
        tmp_path,
        plan,
        claim,
        metadata=metadata,
        test_rows=rows,
        probabilities=probabilities,
    )
    record_path = output / "record.json"
    completion_path = output / "completion.json"
    record = geoadapt_v2.strict_load(record_path)
    completion = geoadapt_v2.strict_load(completion_path)
    os.chmod(output, 0o755)
    if mutation == "record_extra":
        record["unexpected"] = "field"
    elif mutation == "record_bool_count":
        record["test_count"] = True
    elif mutation == "completion_extra":
        completion["unexpected"] = "field"
    else:
        completion["files"] = ["record.json", "predictions.npz"]
    if mutation.startswith("record_"):
        geoadapt_v2._atomic_json(record_path, record)
        completion["files"]["record.json"] = geoadapt_v2._sha256_file(record_path)
    geoadapt_v2._atomic_json(completion_path, completion)
    os.chmod(output, 0o555)
    with pytest.raises(
        (geoadapt_v2.GeoAdaptV2Error, TypeError),
        match="exact schema|schema or type",
    ):
        geoadapt_v2.validate_completion(tmp_path, plan, job)


@pytest.mark.parametrize("attack", ("delete", "replace"))
def test_claim_loss_after_final_check_quarantines_renamed_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attack: str,
) -> None:
    plan, job, claim, metadata, rows, probabilities = _commit_inputs(
        tmp_path, monkeypatch
    )
    real_rename = geoadapt_v2._rename_noreplace

    def attacked_rename(source, destination, **kwargs):
        real_rename(source, destination, **kwargs)
        if geoadapt_v2._absolute_path(destination) == geoadapt_v2._record_directory(
            tmp_path, job
        ):
            os.chmod(claim.path, 0o600)
            claim.path.unlink()
            if attack == "replace":
                geoadapt_v2._write_json_exclusive(
                    claim.path,
                    {"malformed": True},
                )

    monkeypatch.setattr(geoadapt_v2, "_rename_noreplace", attacked_rename)
    with pytest.raises((FileNotFoundError, geoadapt_v2.GeoAdaptV2Error)):
        geoadapt_v2.commit_job_output(
            tmp_path,
            plan,
            claim,
            metadata=metadata,
            test_rows=rows,
            probabilities=probabilities,
        )
    assert not geoadapt_v2._path_exists(geoadapt_v2._record_directory(tmp_path, job))
    quarantined = tuple((tmp_path / "quarantine" / "records").iterdir())
    assert len(quarantined) == 1
    assert "claimlost" in quarantined[0].name
    assert geoadapt_v2._audit_quarantine(tmp_path, plan)[1] == ()


def test_record_is_quarantined_if_held_gpu_guard_is_lost_after_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, job, cpu_claim, metadata, rows, probabilities = _commit_inputs(
        tmp_path, monkeypatch
    )
    gpu_uuid = "GPU-00000000-0000-0000-0000-000000000001"
    receipt = {"lease": "frozen-receipt"}
    claim_guard = _cuda_resource_guard_stub(
        root=tmp_path,
        gpu_uuid=gpu_uuid,
        receipt=receipt,
        allow_active_owner=False,
    )
    commit_guard = _cuda_resource_guard_stub(
        root=tmp_path,
        gpu_uuid=gpu_uuid,
        receipt=receipt,
        allow_active_owner=True,
    )
    claim_payload = geoadapt_v2.strict_load(cpu_claim.path)
    claim_payload["resource_guard"] = copy.deepcopy(claim_guard)
    claim_payload["gpu_lease_receipt"] = copy.deepcopy(receipt)
    geoadapt_v2._atomic_json(cpu_claim.path, claim_payload)
    claim_identity = geoadapt_v2._anchored_lstat(cpu_claim.path)
    claim = geoadapt_v2.Claim(
        job=job,
        path=cpu_claim.path,
        nonce=cpu_claim.nonce,
        owner=cpu_claim.owner,
        resource_guard=claim_guard,
        gpu_lease_receipt=receipt,
        st_dev=int(claim_identity.st_dev),
        st_ino=int(claim_identity.st_ino),
    )
    cuda_resource = copy.deepcopy(claim_guard["device_resource"])
    metadata["runtime"]["device"] = "cuda:0"
    metadata["runtime"]["node_inventory"] = copy.deepcopy(
        cuda_resource["node_inventory"]
    )
    metadata["runtime"]["torch_determinism"]["cuda_available"] = True
    metadata["runtime"]["torch_determinism"]["cublas_workspace_config"] = (
        geoadapt_v2.DETERMINISTIC_TORCH_CONTRACT["cublas_workspace_config_on_cuda"]
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", gpu_uuid)
    monkeypatch.setattr(
        geoadapt_v2.project_gpu_leases,
        "validate_gpu_lease_receipt",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        geoadapt_v2,
        "_device_resource_identity",
        lambda device: copy.deepcopy(cuda_resource),
    )
    monkeypatch.setattr(
        geoadapt_v2,
        "_cooperative_gpu_status",
        lambda gpu_uuid, *, allow_active_owner: _gpu_status_stub(
            gpu_uuid,
            allow_active_owner=allow_active_owner,
        ),
    )
    monkeypatch.setattr(
        geoadapt_v2,
        "_assert_bound_gpu_lease",
        lambda lease, expected_receipt: expected_receipt,
    )
    destination = geoadapt_v2._record_directory(tmp_path, job)
    events: list[tuple[str, bool]] = []
    fake_lease = object()

    @contextlib.contextmanager
    def lost_after_rename(lease):
        assert lease is fake_lease
        events.append(("enter", geoadapt_v2._path_exists(destination)))
        yield copy.deepcopy(receipt)
        events.append(("post", geoadapt_v2._path_exists(destination)))
        raise geoadapt_v2.project_gpu_leases.ProjectGPULeaseError(
            "simulated post-rename lease loss"
        )

    monkeypatch.setattr(geoadapt_v2, "guard_gpu_lease", lost_after_rename)
    with pytest.raises(
        geoadapt_v2.GeoAdaptV2Error,
        match="ownership was lost at publication",
    ):
        geoadapt_v2.commit_job_output(
            tmp_path,
            plan,
            claim,
            metadata=metadata,
            test_rows=rows,
            probabilities=probabilities,
            resource_recheck=commit_guard,
            commit_gpu_lease_receipt=receipt,
            gpu_lease=fake_lease,
        )
    assert events == [("enter", False), ("post", True)]
    assert not geoadapt_v2._path_exists(destination)
    quarantine = tuple((tmp_path / "quarantine" / "records").iterdir())
    assert len(quarantine) == 1
    assert "failed" in quarantine[0].name
    assert geoadapt_v2._audit_quarantine(tmp_path, plan)[1] == ()


def test_commit_binds_live_device_resource_to_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, _job, claim, metadata, rows, probabilities = _commit_inputs(
        tmp_path, monkeypatch
    )
    changed = copy.deepcopy(_device_resource_stub("cpu"))
    changed["node_inventory"]["host"] = "changed-host"
    monkeypatch.setattr(
        geoadapt_v2,
        "_device_resource_identity",
        lambda device: changed,
    )
    with pytest.raises(geoadapt_v2.GeoAdaptV2Error, match="live device resource"):
        geoadapt_v2.commit_job_output(
            tmp_path,
            plan,
            claim,
            metadata=metadata,
            test_rows=rows,
            probabilities=probabilities,
        )
    assert not (tmp_path / "partials").exists()


def test_commit_rechecks_disk_floor_immediately_before_each_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, job, claim, metadata, rows, probabilities = _commit_inputs(
        tmp_path, monkeypatch
    )
    phases: list[str] = []

    def disk_floor(path, *, plan, phase):
        del path, plan
        phases.append(phase)
        if phase == "prediction write":
            raise geoadapt_v2.DiskUnavailable("simulated low disk")
        return copy.deepcopy(dict(claim.resource_guard["disk"]))

    monkeypatch.setattr(geoadapt_v2, "_require_disk_floor", disk_floor)
    with pytest.raises(geoadapt_v2.DiskUnavailable, match="simulated low disk"):
        geoadapt_v2.commit_job_output(
            tmp_path,
            plan,
            claim,
            metadata=metadata,
            test_rows=rows,
            probabilities=probabilities,
        )
    assert phases == ["record write", "prediction write"]
    assert not geoadapt_v2._path_exists(geoadapt_v2._record_directory(tmp_path, job))
    quarantined = tuple((tmp_path / "quarantine" / "partials").iterdir())
    assert len(quarantined) == 1


def test_physical_gpu_uuid_requires_one_full_visible_uuid() -> None:
    upper = "GPU-AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE"
    resource = _cuda_device_resource_stub(upper)
    assert (
        geoadapt_v2._physical_gpu_uuid_from_device_resource(resource)
        == "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    )

    for invalid_visibility in (
        None,
        "0",
        "GPU-aaaaaaaa-bbbb-cccc-dddd",
        f"{upper},GPU-00000000-0000-0000-0000-000000000001",
    ):
        attacked = copy.deepcopy(resource)
        attacked["cuda_visible_devices"] = invalid_visibility
        attacked["node_inventory"]["cuda"]["cuda_visible_devices"] = invalid_visibility
        with pytest.raises(
            geoadapt_v2.GeoAdaptV2Error,
            match="one full|exactly one",
        ):
            geoadapt_v2._physical_gpu_uuid_from_device_resource(attacked)


def test_physical_gpu_uuid_must_match_unique_inventory_row() -> None:
    resource = _cuda_device_resource_stub()
    resource["node_inventory"]["cuda"]["inventory"] = []
    with pytest.raises(geoadapt_v2.GeoAdaptV2Error, match="absent or duplicated"):
        geoadapt_v2._physical_gpu_uuid_from_device_resource(resource)

    resource = _cuda_device_resource_stub()
    resource["node_inventory"]["cuda"]["inventory"].append(
        copy.deepcopy(resource["node_inventory"]["cuda"]["inventory"][0])
    )
    with pytest.raises(geoadapt_v2.GeoAdaptV2Error, match="absent or duplicated"):
        geoadapt_v2._physical_gpu_uuid_from_device_resource(resource)


def test_project_gpu_worker_lease_is_cuda_only_and_uses_central_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _tiny_plan()
    calls: list[tuple[str, object]] = []

    def forbidden_acquire(**kwargs):
        del kwargs
        raise AssertionError("CPU workers must not acquire a GPU lease")

    monkeypatch.setattr(geoadapt_v2, "acquire_gpu_lease", forbidden_acquire)
    monkeypatch.setattr(
        geoadapt_v2,
        "_device_resource_identity",
        _device_resource_stub,
    )
    with geoadapt_v2._project_gpu_worker_lease(
        run_root=tmp_path,
        plan=plan,
        device="cpu",
    ) as lease:
        assert lease is None

    gpu_uuid = "GPU-00000000-0000-0000-0000-000000000001"
    resource = _cuda_device_resource_stub(gpu_uuid)

    class FakeLease:
        value = {"gpu_uuid": gpu_uuid}

    fake_lease = FakeLease()

    def acquire(**kwargs):
        calls.append(("acquire", kwargs))
        return fake_lease

    def release(lease):
        calls.append(("release", lease))

    monkeypatch.setattr(geoadapt_v2, "acquire_gpu_lease", acquire)
    monkeypatch.setattr(geoadapt_v2, "release_gpu_lease", release)
    monkeypatch.setattr(geoadapt_v2, "assert_gpu_lease", lambda lease: None)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", gpu_uuid)
    monkeypatch.setattr(
        geoadapt_v2,
        "_device_resource_identity",
        lambda device: copy.deepcopy(resource),
    )
    with geoadapt_v2._project_gpu_worker_lease(
        run_root=tmp_path,
        plan=plan,
        device="cuda:0",
    ) as lease:
        assert lease is fake_lease
        assert [name for name, _ in calls] == ["acquire"]
    assert [name for name, _ in calls] == ["acquire", "release"]
    acquire_call = calls[0][1]
    assert acquire_call["run_root"] == tmp_path
    assert acquire_call["plan_sha256"] == plan["plan_sha256"]
    assert acquire_call["gpu_uuid"] == gpu_uuid
    assert acquire_call["track_scope"] == "geoadapt-v2"
    assert (
        acquire_call["project_root"] == Path(geoadapt_v2.__file__).resolve().parents[2]
    )
    assert calls[1][1] is fake_lease


def test_project_gpu_worker_lease_maps_retryable_contention(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gpu_uuid = "GPU-00000000-0000-0000-0000-000000000001"
    resource = _cuda_device_resource_stub(gpu_uuid)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", gpu_uuid)
    monkeypatch.setattr(
        geoadapt_v2,
        "_device_resource_identity",
        lambda device: copy.deepcopy(resource),
    )

    def busy(**kwargs):
        del kwargs
        raise geoadapt_v2.project_gpu_leases.GPUWorkerUnavailable("busy")

    monkeypatch.setattr(geoadapt_v2, "acquire_gpu_lease", busy)
    with pytest.raises(geoadapt_v2.GPUWorkerUnavailable, match="busy"):
        with geoadapt_v2._project_gpu_worker_lease(
            run_root=tmp_path,
            plan=_tiny_plan(),
            device="cuda:0",
        ):
            raise AssertionError("busy lease must not enter worker body")


def test_claim_publish_is_inside_held_gpu_registry_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_tiny_plan(monkeypatch)
    plan = _tiny_plan()
    geoadapt_v2.write_or_validate_plan(tmp_path, plan)
    job = next(geoadapt_v2.iter_jobs(plan))
    path = geoadapt_v2._claim_path(tmp_path, job)
    gpu_uuid = "GPU-00000000-0000-0000-0000-000000000001"
    receipt = {"lease": "frozen-receipt"}
    resource_guard = _cuda_resource_guard_stub(
        root=tmp_path,
        gpu_uuid=gpu_uuid,
        receipt=receipt,
        allow_active_owner=False,
        host=geoadapt_v2.socket.gethostname(),
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", gpu_uuid)
    monkeypatch.setattr(
        geoadapt_v2.project_gpu_leases,
        "validate_gpu_lease_receipt",
        lambda *args, **kwargs: None,
    )
    events: list[tuple[str, bool]] = []
    fake_lease = object()

    @contextlib.contextmanager
    def held_guard(lease):
        assert lease is fake_lease
        events.append(("enter", geoadapt_v2._path_exists(path)))
        yield copy.deepcopy(receipt)
        events.append(("exit", geoadapt_v2._path_exists(path)))

    monkeypatch.setattr(geoadapt_v2, "guard_gpu_lease", held_guard)
    claim = geoadapt_v2.acquire_claim(
        tmp_path,
        plan,
        job,
        device="cuda:0",
        resource_guard=resource_guard,
        gpu_lease=fake_lease,
    )
    assert events == [("enter", False), ("exit", True)]
    assert claim.gpu_lease_receipt == receipt
    assert geoadapt_v2.strict_load(path)["gpu_lease_receipt"] == receipt


def test_claim_is_quarantined_if_held_gpu_guard_is_lost_after_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_tiny_plan(monkeypatch)
    plan = _tiny_plan()
    geoadapt_v2.write_or_validate_plan(tmp_path, plan)
    job = next(geoadapt_v2.iter_jobs(plan))
    path = geoadapt_v2._claim_path(tmp_path, job)
    gpu_uuid = "GPU-00000000-0000-0000-0000-000000000001"
    receipt = {"lease": "frozen-receipt"}
    resource_guard = _cuda_resource_guard_stub(
        root=tmp_path,
        gpu_uuid=gpu_uuid,
        receipt=receipt,
        allow_active_owner=False,
        host=geoadapt_v2.socket.gethostname(),
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", gpu_uuid)
    monkeypatch.setattr(
        geoadapt_v2.project_gpu_leases,
        "validate_gpu_lease_receipt",
        lambda *args, **kwargs: None,
    )

    @contextlib.contextmanager
    def lost_after_publish(lease):
        del lease
        yield copy.deepcopy(receipt)
        raise geoadapt_v2.project_gpu_leases.ProjectGPULeaseError(
            "simulated lease loss"
        )

    monkeypatch.setattr(
        geoadapt_v2,
        "guard_gpu_lease",
        lost_after_publish,
    )
    with pytest.raises(
        geoadapt_v2.GeoAdaptV2Error,
        match="ownership was lost at publication",
    ):
        geoadapt_v2.acquire_claim(
            tmp_path,
            plan,
            job,
            device="cuda:0",
            resource_guard=resource_guard,
            gpu_lease=object(),
        )
    assert not geoadapt_v2._path_exists(path)
    quarantine = tmp_path / "quarantine" / "claims"
    entries = tuple(quarantine.iterdir())
    assert len(entries) == 1
    assert "leaselost" in entries[0].name
    assert geoadapt_v2._audit_quarantine(tmp_path, plan)[1] == ()


def test_cooperative_gpu_probe_rejects_foreign_processes_and_malformed_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gpu_uuid = "GPU-00000000-0000-0000-0000-000000000001"

    def foreign_rows(arguments):
        if str(arguments[0]).startswith("--query-gpu="):
            return [[gpu_uuid, "3", "128"]]
        return [[gpu_uuid, "999999", "other-user", "128"]]

    monkeypatch.setattr(geoadapt_v2, "_nvidia_csv_rows", foreign_rows)
    status = geoadapt_v2._cooperative_gpu_status(
        gpu_uuid,
        allow_active_owner=False,
    )
    assert status["safe"] is False
    assert status["foreign_processes"][0]["pid"] == 999999

    monkeypatch.setattr(
        geoadapt_v2,
        "_nvidia_csv_rows",
        lambda arguments: (
            [[gpu_uuid, "3", "128"]]
            if str(arguments[0]).startswith("--query-gpu=")
            else [["malformed"]]
        ),
    )
    with pytest.raises(geoadapt_v2.GeoAdaptV2Error, match="malformed process"):
        geoadapt_v2._cooperative_gpu_status(
            gpu_uuid,
            allow_active_owner=False,
        )


def test_process_start_marker_parses_parenthesized_proc_comm() -> None:
    suffix = ["R", *(str(value) for value in range(4, 22)), "987654321"]
    text = f"123 (worker (nested) name)) {' '.join(suffix)}\n"
    assert geoadapt_v2._parse_linux_proc_stat_start_marker(text, pid=123) == "987654321"
    assert geoadapt_v2._parse_linux_proc_stat_start_marker(text, pid=124) is None
    assert (
        geoadapt_v2._parse_linux_proc_stat_start_marker(
            "123 malformed proc stat\n",
            pid=123,
        )
        is None
    )


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="requires live Linux procfs",
)
def test_live_linux_process_identity_matches_shared_lease_helper() -> None:
    pid = os.getpid()
    shared_start = geoadapt_v2.project_gpu_leases._process_start_ticks(pid)
    shared_boot = geoadapt_v2.project_gpu_leases._boot_id()
    assert shared_start is not None
    assert shared_boot is not None

    expected_start = str(shared_start)
    expected_boot = hashlib.sha256(shared_boot.encode("ascii")).hexdigest()
    assert geoadapt_v2._process_start_marker(pid) == expected_start
    assert geoadapt_v2._boot_marker() == expected_boot

    identity = geoadapt_v2._process_identity()
    assert identity["pid"] == pid
    assert identity["start_marker"] == expected_start
    assert identity["boot_marker"] == expected_boot


def test_release_quarantines_record_if_owned_claim_disappeared(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, job, claim, metadata, rows, probabilities = _commit_inputs(
        tmp_path, monkeypatch
    )
    geoadapt_v2.commit_job_output(
        tmp_path,
        plan,
        claim,
        metadata=metadata,
        test_rows=rows,
        probabilities=probabilities,
    )
    os.chmod(claim.path, 0o600)
    claim.path.unlink()
    with pytest.raises(geoadapt_v2.GeoAdaptV2Error, match="record quarantined"):
        geoadapt_v2.release_claim(claim, plan)
    assert not geoadapt_v2._path_exists(geoadapt_v2._record_directory(tmp_path, job))
    assert geoadapt_v2._audit_quarantine(tmp_path, plan)[1] == ()


def test_release_claim_never_quarantines_replacement_record_b(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, job, claim, metadata, rows, probabilities = _commit_inputs(
        tmp_path,
        monkeypatch,
    )
    canonical = geoadapt_v2.commit_job_output(
        tmp_path,
        plan,
        claim,
        metadata=metadata,
        test_rows=rows,
        probabilities=probabilities,
    )
    identity_a = claim.publication_state.record_identity
    assert identity_a == (int(canonical.stat().st_dev), int(canonical.stat().st_ino))
    replacement_b = tmp_path / "replacement-record-b"
    shutil.copytree(canonical, replacement_b)
    identity_b = (
        int(replacement_b.stat().st_dev),
        int(replacement_b.stat().st_ino),
    )
    assert identity_b != identity_a
    displaced_a = tmp_path / "record-a-displaced"
    os.chmod(canonical, 0o755)
    os.chmod(replacement_b, 0o755)
    os.replace(canonical, displaced_a)
    os.replace(replacement_b, canonical)
    os.chmod(canonical, 0o555)
    os.chmod(claim.path, 0o600)
    claim.path.unlink()

    with pytest.raises(
        geoadapt_v2.GeoAdaptV2Error,
        match="replacement left untouched",
    ):
        geoadapt_v2.release_claim(claim, plan)
    assert (int(canonical.stat().st_dev), int(canonical.stat().st_ino)) == identity_b
    assert geoadapt_v2.validate_completion(tmp_path, plan, job)["record"]["job_id"] == (
        job.job_id
    )
    assert (int(displaced_a.stat().st_dev), int(displaced_a.stat().st_ino)) == (
        identity_a
    )
    assert claim.publication_state.record_identity is None


def test_exclusive_gate_detects_coordination_lock_inode_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_tiny_plan(monkeypatch)
    plan = _tiny_plan()
    geoadapt_v2.write_or_validate_plan(tmp_path, plan)
    gate = geoadapt_v2.acquire_analysis_gate(tmp_path, plan)
    lock = geoadapt_v2._coordination_lock_path(tmp_path)
    os.unlink(lock)
    geoadapt_v2._write_bytes_exclusive(lock, b"replacement\n")
    try:
        with pytest.raises(
            geoadapt_v2.GeoAdaptV2Error,
            match="inode changed|hard link",
        ):
            geoadapt_v2.assert_analysis_gate(gate, tmp_path, plan)
    finally:
        geoadapt_v2.fcntl.flock(
            gate.lock_descriptor,
            geoadapt_v2.fcntl.LOCK_UN,
        )
        os.close(gate.lock_descriptor)
        for path in (gate.marker_path, lock):
            if path.exists():
                os.chmod(path, 0o600)
                path.unlink()


def test_power_cut_publication_lock_and_partial_are_recoverable(
    tmp_path: Path,
) -> None:
    plan = _tiny_plan()
    output = tmp_path / "analysis"
    lock = geoadapt_v2_analysis._publication_lock_path(output)
    nonce = "aaaaaaaaaaaa4aaa8aaaaaaaaaaaaaaa"
    geoadapt_v2._write_json_exclusive(
        lock,
        {
            "schema": "eeg-mi-geoadapt-v2-analysis-lock-v2",
            "created_at": geoadapt_v2._utc_now(),
            "plan_sha256": plan["plan_sha256"],
            "output_root": str(output),
            "nonce": nonce,
            "owner": {
                "host": geoadapt_v2.socket.gethostname(),
                "pid": 999_999_999,
                "start_marker": "dead",
                "boot_marker": geoadapt_v2._boot_marker(),
            },
        },
    )
    partial = tmp_path / f".{output.name}.{nonce}.partial"
    partial.mkdir()
    new_lock = geoadapt_v2_analysis._acquire_publication_lock(
        output,
        plan,
    )
    assert new_lock.path == lock
    assert new_lock.nonce != nonce
    quarantine = tmp_path / ".geoadapt-v2-publication-quarantine"
    names = {path.name for path in quarantine.iterdir()}
    assert any(".powercut." in name and "publish.lock" in name for name in names)
    assert any(".powercut." in name and ".partial" in name for name in names)
    geoadapt_v2_analysis._release_publication_lock(new_lock)


def test_analysis_cli_enters_gate_before_analysis_and_never_prefetches_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _tiny_plan()
    events: list[str] = []
    gate_token = object()

    monkeypatch.setattr(
        geoadapt_v2,
        "load_plan",
        lambda run_root: events.append("load-plan") or plan,
    )
    monkeypatch.setattr(
        geoadapt_v2,
        "verify_static_identity",
        lambda active_plan: events.append("verify-static"),
    )

    def forbidden_preflight(*args, **kwargs):
        raise AssertionError("CLI must not open caches before the analysis audit")

    monkeypatch.setattr(
        geoadapt_v2,
        "verify_cache_and_splits",
        forbidden_preflight,
    )

    @contextlib.contextmanager
    def fake_gate(run_root, active_plan):
        events.append("gate-enter")
        yield gate_token
        events.append("gate-exit")

    monkeypatch.setattr(geoadapt_v2, "analysis_gate", fake_gate)

    def fake_publish(**kwargs):
        assert kwargs["gate"] is gate_token
        assert kwargs["cache_root"] == tmp_path / "cache"
        assert "result" not in kwargs
        events.append("publish")
        return Path(kwargs["output_root"])

    monkeypatch.setattr(geoadapt_v2_analysis, "publish_analysis", fake_publish)
    monkeypatch.setattr(
        geoadapt_v2,
        "strict_load",
        lambda path: {"table_row_counts": {}},
    )
    assert (
        geoadapt_v2_analysis.main(
            [
                "--run-root",
                str(tmp_path / "run"),
                "--cache-root",
                str(tmp_path / "cache"),
                "--output-root",
                str(tmp_path / "analysis"),
            ]
        )
        == 0
    )
    assert events == [
        "load-plan",
        "verify-static",
        "gate-enter",
        "publish",
        "gate-exit",
    ]


def test_common_support_statistics_are_paired_and_holm_adjusted() -> None:
    offsets = {
        "architecture.geoadaptnet": 0.03,
        "architecture.geoadaptnet_fb": 0.02,
        "architecture.geoadaptnet_fbsp": 0.01,
    }
    subject_rows = []
    for stable_id, offset in offsets.items():
        for dataset in geoadapt_v2_analysis.COMMON_SUPPORT_DATASETS:
            for subject in geoadapt_v2.DATASET_SUBJECTS[dataset]:
                subject_rows.append(
                    {
                        "stable_id": stable_id,
                        "dataset": dataset,
                        "subject": subject,
                        "balanced_accuracy": 0.6 + offset + subject * 1e-5,
                    }
                )
    summary, paired = geoadapt_v2_analysis._common_support_tables(
        subject_rows,
        geoadapt_v2.STABLE_IDS,
    )
    assert len(summary) == 3
    assert len(paired) == 3
    assert {row["dataset_coverage"] for row in (*summary, *paired)} == {
        "local_exp4,cho2017,physionet_mi"
    }
    assert {row["n_paired_subjects"] for row in paired} == {114}
    assert all(
        0.0 <= row["randomization_p_two_sided"] <= row["holm_adjusted_p"] <= 1.0
        for row in paired
    )
    assert all(
        row["bootstrap_ci95_low"]
        <= row["mean_difference_a_minus_b"]
        <= row["bootstrap_ci95_high"]
        for row in paired
    )


def test_analysis_requires_quiescence_and_publishes_aggregate_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _allow_tiny_plan(monkeypatch)
    monkeypatch.setattr(
        geoadapt_v2_analysis,
        "_common_support_tables",
        lambda subject_rows, stable_ids: ([], []),
    )
    from benchmark import data as data_module

    monkeypatch.setattr(data_module, "load_subject_cache", _fake_loader)
    monkeypatch.setattr(geoadapt_v2, "verify_static_identity", lambda plan: None)
    monkeypatch.setattr(
        geoadapt_v2,
        "verify_cache_and_splits",
        lambda plan, *, cache_root: None,
    )
    run_root = tmp_path / "run"
    output_root = tmp_path / "analysis"
    plan = _tiny_plan(seeds=(7, 17))
    geoadapt_v2.write_or_validate_plan(run_root, plan)
    monkeypatch.setattr(geoadapt_v2, "_runtime_identity", _runtime_stub)
    for job in geoadapt_v2.iter_jobs(plan):
        metadata, rows, probabilities = geoadapt_v2.execute_job(
            job=job,
            plan=plan,
            cache_root=tmp_path,
            device="cpu",
            backend=_tiny_backend,
            cache_loader=_fake_loader,
            splitter=_fake_splitter,
        )
        claim = _claim(job, run_root, plan)
        geoadapt_v2.commit_job_output(
            run_root,
            plan,
            claim,
            metadata=metadata,
            test_rows=rows,
            probabilities=probabilities,
        )
        geoadapt_v2.release_claim(claim, plan)
    stray = run_root / "partials" / "stray.partial"
    stray.mkdir(parents=True)
    outcome_accesses = 0
    publication_floor_phases: list[str] = []
    real_disk_floor = geoadapt_v2._require_disk_floor

    def capture_publication_floor(path, *, plan, phase):
        publication_floor_phases.append(phase)
        return real_disk_floor(path, plan=plan, phase=phase)

    monkeypatch.setattr(
        geoadapt_v2,
        "_require_disk_floor",
        capture_publication_floor,
    )

    def forbidden_outcome_loader(*args, **kwargs):
        nonlocal outcome_accesses
        outcome_accesses += 1
        raise AssertionError("outcomes must remain inaccessible before audit")

    with geoadapt_v2.analysis_gate(run_root, plan) as gate:
        with pytest.raises(
            geoadapt_v2_analysis.GeoAdaptV2AnalysisError,
            match="complete and quiescent",
        ):
            geoadapt_v2_analysis.analyze_grid(
                run_root=run_root,
                cache_root=tmp_path,
                gate=gate,
                plan=plan,
                cache_loader=forbidden_outcome_loader,
            )
    assert outcome_accesses == 0
    stray.rmdir()
    with geoadapt_v2.analysis_gate(run_root, plan) as gate:
        result = geoadapt_v2_analysis.analyze_grid(
            run_root=run_root,
            cache_root=tmp_path,
            gate=gate,
            plan=plan,
            cache_loader=_fake_loader,
        )
        assert result.summary["common_recipe_result"] is False
        assert result.summary["table_row_counts"]["job_metrics"] == 2
        serialized = json.dumps(
            {"summary": result.summary, "tables": result.tables},
            sort_keys=True,
        ).lower()
        assert '"labels"' not in serialized
        assert '"probabilities"' not in serialized
        geoadapt_v2_analysis._validate_analysis_result_exact(result, plan=plan)
        bad_cardinality = copy.deepcopy(result)
        bad_cardinality.tables["job_metrics"].pop()
        with pytest.raises(
            geoadapt_v2_analysis.GeoAdaptV2AnalysisError,
            match="cardinality",
        ):
            geoadapt_v2_analysis._validate_analysis_result_exact(
                bad_cardinality,
                plan=plan,
            )
        bad_type = copy.deepcopy(result)
        bad_type.tables["overall_summary"][0]["accuracy"] = 1
        with pytest.raises(
            geoadapt_v2_analysis.GeoAdaptV2AnalysisError,
            match="finite or null",
        ):
            geoadapt_v2_analysis._validate_analysis_result_exact(
                bad_type,
                plan=plan,
            )
        with pytest.raises(
            geoadapt_v2_analysis.GeoAdaptV2AnalysisError,
            match="outside the immutable run root",
        ):
            geoadapt_v2_analysis.publish_analysis(
                run_root=run_root,
                cache_root=tmp_path,
                output_root=run_root / "nested-analysis",
                gate=gate,
            )
        published = geoadapt_v2_analysis.publish_analysis(
            run_root=run_root,
            cache_root=tmp_path,
            output_root=output_root,
            gate=gate,
        )
    assert published == output_root
    assert publication_floor_phases[0:2] == [
        "publication summary write",
        "publication input-ledger write",
    ]
    assert publication_floor_phases[-2:] == [
        "publication manifest write",
        "publication commit",
    ]
    assert {
        phase.removeprefix("publication ").removesuffix(" write")
        for phase in publication_floor_phases[2:-2]
    } == {f"{name}.csv" for name in result.tables}
    assert (published / "manifest.json").is_file()
    with (published / "overall_summary.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(__import__("csv").DictReader(handle))
    assert rows[0]["track"] == geoadapt_v2.TRACK
    assert rows[0]["n_datasets"] == "1"
    external = tmp_path / "published-summary-hardlink"
    os.link(published / "summary.json", external)
    mode_before = external.stat().st_mode
    with geoadapt_v2.analysis_gate(run_root, plan) as gate:
        with pytest.raises(geoadapt_v2.GeoAdaptV2Error, match="hard link"):
            geoadapt_v2_analysis.publish_analysis(
                run_root=run_root,
                cache_root=tmp_path,
                output_root=output_root,
                gate=gate,
            )
    assert external.stat().st_mode == mode_before


def test_analysis_aborts_if_predictions_change_after_metric_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_tiny_plan(monkeypatch)
    monkeypatch.setattr(
        geoadapt_v2_analysis,
        "_common_support_tables",
        lambda subject_rows, stable_ids: ([], []),
    )
    from benchmark import data as data_module

    monkeypatch.setattr(data_module, "load_subject_cache", _fake_loader)
    monkeypatch.setattr(geoadapt_v2, "verify_static_identity", lambda plan: None)
    monkeypatch.setattr(
        geoadapt_v2,
        "verify_cache_and_splits",
        lambda plan, *, cache_root: None,
    )
    run_root = tmp_path / "run"
    output_root = tmp_path / "analysis"
    plan = _tiny_plan()
    geoadapt_v2.write_or_validate_plan(run_root, plan)
    monkeypatch.setattr(geoadapt_v2, "_runtime_identity", _runtime_stub)
    job = next(geoadapt_v2.iter_jobs(plan))
    metadata, rows, probabilities = geoadapt_v2.execute_job(
        job=job,
        plan=plan,
        cache_root=tmp_path,
        device="cpu",
        backend=_tiny_backend,
        cache_loader=_fake_loader,
        splitter=_fake_splitter,
    )
    claim = _claim(job, run_root, plan)
    geoadapt_v2.commit_job_output(
        run_root,
        plan,
        claim,
        metadata=metadata,
        test_rows=rows,
        probabilities=probabilities,
    )
    geoadapt_v2.release_claim(claim, plan)

    captured: dict[str, geoadapt_v2_analysis.CompletionSnapshot] = {}
    real_capture = geoadapt_v2_analysis._capture_completion_snapshot

    def capture_with_identity(*args, **kwargs):
        snapshot = real_capture(*args, **kwargs)
        assert snapshot.artifact_sha256 == {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in snapshot.artifact_bytes.items()
        }
        captured["snapshot"] = snapshot
        return snapshot

    monkeypatch.setattr(
        geoadapt_v2_analysis,
        "_capture_completion_snapshot",
        capture_with_identity,
    )
    real_metrics = geoadapt_v2_analysis.classification_metrics
    replaced = {"done": False}

    def score_then_replace(y_true, values, *, n_classes=2):
        metrics = real_metrics(y_true, values, n_classes=n_classes)
        if replaced["done"]:
            return metrics
        snapshot = captured["snapshot"]
        assert metrics["accuracy"] == 1.0
        replacement_probabilities = np.ascontiguousarray(
            snapshot.probabilities[:, ::-1],
            dtype=np.float64,
        )
        assert (
            real_metrics(
                y_true,
                replacement_probabilities,
                n_classes=n_classes,
            )["accuracy"]
            == 0.0
        )
        prediction_buffer = io.BytesIO()
        np.savez_compressed(
            prediction_buffer,
            rows=np.ascontiguousarray(snapshot.rows, dtype=np.int64),
            probabilities=replacement_probabilities,
        )
        prediction_payload = prediction_buffer.getvalue()
        replacement_completion = copy.deepcopy(dict(snapshot.completion))
        replacement_completion["files"]["predictions.npz"] = hashlib.sha256(
            prediction_payload
        ).hexdigest()
        completion_payload = (
            geoadapt_v2._canonical_bytes(replacement_completion) + b"\n"
        )
        directory = geoadapt_v2._record_directory(run_root, job)
        staged_prediction = directory / ".predictions.attack"
        staged_completion = directory / ".completion.attack"
        os.chmod(directory, 0o755)
        try:
            geoadapt_v2._write_bytes_exclusive(
                staged_prediction,
                prediction_payload,
            )
            geoadapt_v2._write_bytes_exclusive(
                staged_completion,
                completion_payload,
            )
            os.replace(staged_prediction, directory / "predictions.npz")
            os.replace(staged_completion, directory / "completion.json")
            geoadapt_v2._fsync_directory(directory)
        finally:
            os.chmod(directory, 0o555)
            geoadapt_v2._fsync_directory(directory)
        replaced["done"] = True
        return metrics

    monkeypatch.setattr(
        geoadapt_v2_analysis,
        "classification_metrics",
        score_then_replace,
    )
    with geoadapt_v2.analysis_gate(run_root, plan) as gate:
        with pytest.raises(
            geoadapt_v2_analysis.GeoAdaptV2AnalysisError,
            match="captured input ledger changed",
        ):
            geoadapt_v2_analysis.publish_analysis(
                run_root=run_root,
                cache_root=tmp_path,
                output_root=output_root,
                gate=gate,
            )
    assert replaced["done"] is True
    assert not output_root.exists()
    assert not geoadapt_v2_analysis._publication_lock_path(output_root).exists()
    assert not tuple(tmp_path.glob(f".{output_root.name}.*.partial"))
    live = geoadapt_v2.validate_completion(run_root, plan, job)
    assert (
        real_metrics(
            _fake_data()["y"][live["rows"]],
            live["probabilities"],
            n_classes=2,
        )["accuracy"]
        == 0.0
    )


@pytest.mark.parametrize(
    "fault",
    (
        "gate-ownership",
        "lock-ownership",
        "lock-release",
        "move-then-raise",
        "post-rebind-baseexception",
    ),
)
def test_analysis_postcommit_ownership_or_release_failure_quarantines_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    _allow_tiny_plan(monkeypatch)
    monkeypatch.setattr(
        geoadapt_v2_analysis,
        "_common_support_tables",
        lambda subject_rows, stable_ids: ([], []),
    )
    from benchmark import data as data_module

    monkeypatch.setattr(data_module, "load_subject_cache", _fake_loader)
    monkeypatch.setattr(geoadapt_v2, "verify_static_identity", lambda plan: None)
    monkeypatch.setattr(
        geoadapt_v2,
        "verify_cache_and_splits",
        lambda plan, *, cache_root: None,
    )
    run_root = tmp_path / "run"
    output_root = tmp_path / "analysis"
    plan = _tiny_plan()
    geoadapt_v2.write_or_validate_plan(run_root, plan)
    monkeypatch.setattr(geoadapt_v2, "_runtime_identity", _runtime_stub)
    job = next(geoadapt_v2.iter_jobs(plan))
    metadata, rows, probabilities = geoadapt_v2.execute_job(
        job=job,
        plan=plan,
        cache_root=tmp_path,
        device="cpu",
        backend=_tiny_backend,
        cache_loader=_fake_loader,
        splitter=_fake_splitter,
    )
    claim = _claim(job, run_root, plan)
    geoadapt_v2.commit_job_output(
        run_root,
        plan,
        claim,
        metadata=metadata,
        test_rows=rows,
        probabilities=probabilities,
    )
    geoadapt_v2.release_claim(claim, plan)
    analysis_rebind_states: list[bool] = []

    if fault == "gate-ownership":
        real_assert_gate = geoadapt_v2.assert_analysis_gate

        def fail_gate_after_rename(gate, root, active_plan):
            if geoadapt_v2._path_exists(output_root):
                raise geoadapt_v2.GeoAdaptV2Error(
                    "simulated postcommit gate ownership loss"
                )
            return real_assert_gate(gate, root, active_plan)

        monkeypatch.setattr(
            geoadapt_v2,
            "assert_analysis_gate",
            fail_gate_after_rename,
        )
    elif fault == "lock-ownership":
        real_assert_lock = geoadapt_v2_analysis._assert_publication_lock

        def fail_lock_after_rename(lock, *, output_root, plan):
            if geoadapt_v2._path_exists(output_root):
                raise geoadapt_v2_analysis.GeoAdaptV2AnalysisError(
                    "simulated postcommit publication-lock ownership loss"
                )
            return real_assert_lock(
                lock,
                output_root=output_root,
                plan=plan,
            )

        monkeypatch.setattr(
            geoadapt_v2_analysis,
            "_assert_publication_lock",
            fail_lock_after_rename,
        )
    elif fault == "lock-release":
        real_release = geoadapt_v2_analysis._release_publication_lock

        def fail_after_release(lock):
            real_release(lock)
            raise geoadapt_v2_analysis.GeoAdaptV2AnalysisError(
                "simulated publication-lock release failure"
            )

        monkeypatch.setattr(
            geoadapt_v2_analysis,
            "_release_publication_lock",
            fail_after_release,
        )
    elif fault == "move-then-raise":
        real_rename = geoadapt_v2._rename_noreplace
        real_rebind = geoadapt_v2_analysis._rebind_analysis_commit_authority

        def move_then_raise(source, destination, **kwargs):
            real_rename(source, destination, **kwargs)
            if geoadapt_v2._absolute_path(destination) == output_root:
                raise OSError("simulated post-move analysis failure")

        def observe_rebind(**kwargs):
            analysis_rebind_states.append(output_root.exists())
            return real_rebind(**kwargs)

        monkeypatch.setattr(
            geoadapt_v2_analysis,
            "_rebind_analysis_commit_authority",
            observe_rebind,
        )
        monkeypatch.setattr(geoadapt_v2, "_rename_noreplace", move_then_raise)
    else:
        real_rebind = geoadapt_v2_analysis._rebind_analysis_commit_authority
        rebind_calls = 0

        def interrupt_after_identity(**kwargs):
            nonlocal rebind_calls
            rebind_calls += 1
            if rebind_calls == 2:
                assert output_root.exists()
                raise SyntheticPublicationInterrupt(
                    "simulated analysis post-identity interrupt"
                )
            return real_rebind(**kwargs)

        monkeypatch.setattr(
            geoadapt_v2_analysis,
            "_rebind_analysis_commit_authority",
            interrupt_after_identity,
        )

    gate = geoadapt_v2.acquire_analysis_gate(run_root, plan)
    try:
        with pytest.raises(
            (
                SyntheticPublicationInterrupt,
                OSError,
                geoadapt_v2.GeoAdaptV2Error,
                geoadapt_v2_analysis.GeoAdaptV2AnalysisError,
            ),
            match="simulated",
        ):
            geoadapt_v2_analysis.publish_analysis(
                run_root=run_root,
                cache_root=tmp_path,
                output_root=output_root,
                gate=gate,
            )
    finally:
        geoadapt_v2.release_analysis_gate(gate)
    assert not output_root.exists()
    if fault == "move-then-raise":
        assert analysis_rebind_states == [False, True]
    quarantine = tmp_path / ".geoadapt-v2-publication-quarantine"
    quarantined = tuple(quarantine.iterdir())
    assert len(quarantined) == 1
    expected_reason = "releasefailed" if fault == "lock-release" else "failed"
    assert f".{expected_reason}." in quarantined[0].name


def test_publisher_cannot_accept_caller_result_or_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _allow_tiny_plan(monkeypatch)
    monkeypatch.setattr(
        geoadapt_v2_analysis,
        "_common_support_tables",
        lambda subject_rows, stable_ids: ([], []),
    )
    run_root = tmp_path / "run"
    plan = _tiny_plan()
    geoadapt_v2.write_or_validate_plan(run_root, plan)
    monkeypatch.setattr(geoadapt_v2, "_runtime_identity", _runtime_stub)
    job = next(geoadapt_v2.iter_jobs(plan))
    metadata, rows, probabilities = geoadapt_v2.execute_job(
        job=job,
        plan=plan,
        cache_root=tmp_path,
        device="cpu",
        backend=_tiny_backend,
        cache_loader=_fake_loader,
        splitter=_fake_splitter,
    )
    claim = _claim(job, run_root, plan)
    geoadapt_v2.commit_job_output(
        run_root,
        plan,
        claim,
        metadata=metadata,
        test_rows=rows,
        probabilities=probabilities,
    )
    geoadapt_v2.release_claim(claim, plan)
    with geoadapt_v2.analysis_gate(run_root, plan) as gate:
        result = geoadapt_v2_analysis.analyze_grid(
            run_root=run_root,
            cache_root=tmp_path,
            gate=gate,
            plan=plan,
            cache_loader=_fake_loader,
        )
        result.tables["overall_summary"][0]["accuracy"] = 0.0
        with pytest.raises(TypeError, match="unexpected keyword argument 'result'"):
            geoadapt_v2_analysis.publish_analysis(
                result=result,
                run_root=run_root,
                cache_root=tmp_path,
                output_root=tmp_path / "analysis",
                gate=gate,
            )
        with pytest.raises(
            TypeError,
            match="unexpected keyword argument 'cache_loader'",
        ):
            geoadapt_v2_analysis.publish_analysis(
                run_root=run_root,
                cache_root=tmp_path,
                output_root=tmp_path / "analysis",
                gate=gate,
                cache_loader=_fake_loader,
            )
