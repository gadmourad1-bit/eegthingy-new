from __future__ import annotations

import copy
import errno
import hashlib
import importlib
import inspect
import json
import math
import os
import stat
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pytest

from benchmark import full_grid, project_gpu_leases


def test_formal_source_files_exist_in_src_layout() -> None:
    package_root = Path(full_grid.__file__).resolve().parent
    assert "runner.py" in full_grid.SOURCE_FILES
    assert "benchmark.py" not in full_grid.SOURCE_FILES
    assert all((package_root / name).is_file() for name in full_grid.SOURCE_FILES)


def test_worker_lazy_imports_use_src_layout_module_names() -> None:
    assert full_grid.WORKER_LAZY_IMPORT_MODULES == (
        "runner",
        "baselines",
        "models",
        "training",
        "data",
    )
    assert "benchmark" not in full_grid.WORKER_LAZY_IMPORT_MODULES
    for module_name in full_grid.WORKER_LAZY_IMPORT_MODULES:
        importlib.import_module(f"benchmark.{module_name}")


def _partition(rows: list[int]) -> dict[str, Any]:
    values = np.asarray(rows, dtype=np.int64)
    return {
        "count": len(values),
        "rows_sha256": full_grid._rows_sha256(values),
    }


def test_row_identity_preserves_established_digest_contract() -> None:
    rows = np.asarray([0, 1, 2, 3], dtype=np.int64)
    assert full_grid._rows_sha256(rows) == (
        "f079e084b609273c7252204c14491908916838b2276e687092c155feaf67c309"
    )
    assert full_grid._partition_identity(rows) == {
        "count": 4,
        "rows_sha256": (
            "f079e084b609273c7252204c14491908916838b2276e687092c155feaf67c309"
        ),
    }


def tiny_plan(
    *,
    architectures: tuple[str, ...] = ("model_a", "tcformer"),
    executor: str = "tests.fake:executor",
    datasets: tuple[str, ...] = ("toy",),
) -> dict[str, Any]:
    dataset_contracts: dict[str, dict[str, Any]] = {}
    cache: dict[str, dict[str, Any]] = {}
    split: dict[str, dict[str, Any]] = {}
    for dataset in datasets:
        dataset_contracts[dataset] = {
            "subjects": [1],
            "folds": [0],
            "n_classes": 2,
            "protocol": "synthetic-test-only",
            "preprocessing": {"sfreq_hz": 128.0, "n_times": 4},
        }
        cache[f"{dataset}:s001"] = {"array_sha256": "a" * 64}
        split[f"{dataset}:s001:f00"] = {
            "dataset": dataset,
            "subject": 1,
            "fold": 0,
            "cache_array_sha256": "a" * 64,
            "trial_count": 4,
            "partitions": {
                "train": _partition([0]),
                "validation": _partition([1]),
                "source": _partition([0, 1]),
                "test": _partition([2, 3]),
            },
        }
    return full_grid.assemble_plan(
        dataset_contracts=dataset_contracts,
        architectures=architectures,
        seeds=(7,),
        train_config={"epochs": 2},
        cache_identity=cache,
        split_identity=split,
        source_identity={"tests/fake.py": "b" * 64},
        environment_identity={},
        executor=executor,
        worker_cpu_threads=2,
        analysis_contract=None,
        publishable=False,
    )


def _valid_formal_cache_identity(
    dataset: str = "bnci2014_004",
    subject: int = 1,
) -> dict[str, Any]:
    from benchmark.config import (
        DEFAULT_MONTAGE_PROFILE,
        channels_for_dataset,
        coordinate_contract_for_dataset,
        dataset_spec,
        preprocessing_for_dataset,
    )

    channels = tuple(
        channels_for_dataset(dataset, DEFAULT_MONTAGE_PROFILE) or ()
    )
    preprocessing = copy.deepcopy(preprocessing_for_dataset(dataset))
    return {
        "array_sha256": "a" * 64,
        "channels": list(channels),
        "coordinates": copy.deepcopy(
            coordinate_contract_for_dataset(
                dataset,
                DEFAULT_MONTAGE_PROFILE,
                channels,
            )
        ),
        "dataset": json.loads(
            full_grid._canonical_bytes(asdict(dataset_spec(dataset)))
        ),
        "montage_profile": DEFAULT_MONTAGE_PROFILE,
        "preprocessing": preprocessing,
        "shape": [
            10,
            len(channels),
            int(preprocessing["n_times"]),
        ],
        "subject": subject,
    }


def resource_status(
    *,
    free_gib: float = 60.0,
    minimum_free_gib: float = 50.0,
) -> full_grid.ResourceStatus:
    return full_grid.combine_resource_status(
        full_grid.GPUStatus(
            safe=True,
            gpu="test-cpu",
            utilization_percent=0.0,
            memory_used_mib=0.0,
            own_compute_memory_mib=0.0,
            foreign_processes=(),
            reason="synthetic idle guard",
            gpu_uuid=None,
            pci_bus_id=None,
            name=None,
        ),
        full_grid.DiskStatus(
            safe=True,
            path="/synthetic",
            free_bytes=int(free_gib * 1024**3),
            total_bytes=int(100 * 1024**3),
            free_gib=free_gib,
            minimum_free_gib=minimum_free_gib,
            reason="synthetic disk guard",
        ),
    )


def _gpu_status(
    *,
    safe: bool,
    utilization_percent: float,
    memory_used_mib: float = 512.0,
    own_compute_memory_mib: float = 500.0,
    foreign_processes: tuple[dict[str, Any], ...] = (),
    reason: str | None = None,
) -> full_grid.GPUStatus:
    return full_grid.GPUStatus(
        safe=safe,
        gpu="GPU-00000001-0000-0000-0000-000000000001",
        utilization_percent=utilization_percent,
        memory_used_mib=memory_used_mib,
        own_compute_memory_mib=own_compute_memory_mib,
        foreign_processes=foreign_processes,
        reason=reason or ("idle" if safe else "utilization above threshold"),
        gpu_uuid="GPU-00000001-0000-0000-0000-000000000001",
        pci_bus_id="0000:01:00.0",
        name="Test GPU",
    )


def _disk_status(*, safe: bool = True) -> full_grid.DiskStatus:
    return full_grid.DiskStatus(
        safe=safe,
        path="/synthetic",
        free_bytes=60 * 1024**3 if safe else 1,
        total_bytes=100 * 1024**3,
        free_gib=60.0 if safe else 0.0,
        minimum_free_gib=50.0,
        reason="above floor" if safe else "below floor",
    )


def test_publication_probe_waits_for_own_utilization_to_cool() -> None:
    observed = iter(
        (
            _gpu_status(safe=False, utilization_percent=42.0),
            _gpu_status(safe=True, utilization_percent=0.0),
        )
    )
    clock = [0.0]
    sleeps: list[float] = []

    def sleep(value: float) -> None:
        sleeps.append(value)
        clock[0] += value

    result = full_grid._wait_for_publication_resources(
        gpu_probe=lambda: next(observed),
        disk_probe=_disk_status,
        max_wait_seconds=1.0,
        poll_seconds=0.25,
        monotonic=lambda: clock[0],
        sleeper=sleep,
    )

    assert result.safe
    assert sleeps == [0.25]


def test_publication_probe_rejects_foreign_process_without_waiting() -> None:
    calls = 0
    sleeps: list[float] = []

    def probe() -> full_grid.GPUStatus:
        nonlocal calls
        calls += 1
        return _gpu_status(
            safe=False,
            utilization_percent=42.0,
            foreign_processes=(
                {
                    "pid": 1234,
                    "process_name": "foreign",
                    "used_gpu_memory_mib": 128.0,
                },
            ),
            reason="1 foreign compute process; utilization above threshold",
        )

    result = full_grid._wait_for_publication_resources(
        gpu_probe=probe,
        disk_probe=_disk_status,
        max_wait_seconds=1.0,
        poll_seconds=0.25,
        monotonic=lambda: 0.0,
        sleeper=sleeps.append,
    )

    assert not result.safe
    assert calls == 1
    assert sleeps == []


@pytest.mark.parametrize(
    "gpu",
    (
        _gpu_status(
            safe=False,
            utilization_percent=42.0,
            own_compute_memory_mib=0.0,
        ),
        _gpu_status(
            safe=False,
            utilization_percent=42.0,
            memory_used_mib=2_000.0,
            own_compute_memory_mib=500.0,
            reason="utilization and foreign memory above thresholds",
        ),
    ),
)
def test_publication_probe_does_not_wait_without_own_only_explanation(
    gpu: full_grid.GPUStatus,
) -> None:
    calls = 0
    sleeps: list[float] = []

    def probe() -> full_grid.GPUStatus:
        nonlocal calls
        calls += 1
        return gpu

    result = full_grid._wait_for_publication_resources(
        gpu_probe=probe,
        disk_probe=_disk_status,
        max_wait_seconds=1.0,
        poll_seconds=0.25,
        monotonic=lambda: 0.0,
        sleeper=sleeps.append,
    )

    assert not result.safe
    assert calls == 1
    assert sleeps == []


def test_publication_probe_times_out_without_relaxing_utilization_limit() -> None:
    calls = 0
    clock = [0.0]

    def probe() -> full_grid.GPUStatus:
        nonlocal calls
        calls += 1
        return _gpu_status(safe=False, utilization_percent=42.0)

    def sleep(value: float) -> None:
        clock[0] += value

    result = full_grid._wait_for_publication_resources(
        gpu_probe=probe,
        disk_probe=_disk_status,
        max_wait_seconds=0.5,
        poll_seconds=0.25,
        monotonic=lambda: clock[0],
        sleeper=sleep,
    )

    assert not result.safe
    assert result.gpu["utilization_percent"] == 42.0
    assert calls == 3
    assert clock[0] == 0.5


def test_publication_probe_rejects_disk_floor_without_waiting() -> None:
    sleeps: list[float] = []
    result = full_grid._wait_for_publication_resources(
        gpu_probe=lambda: _gpu_status(
            safe=False,
            utilization_percent=42.0,
        ),
        disk_probe=lambda: _disk_status(safe=False),
        max_wait_seconds=1.0,
        poll_seconds=0.25,
        monotonic=lambda: 0.0,
        sleeper=sleeps.append,
    )

    assert not result.safe
    assert not result.disk["safe"]
    assert sleeps == []


def metadata(plan: dict[str, Any], job: full_grid.Job) -> dict[str, Any]:
    return {
        "cache_array_sha256": "a" * 64,
        "channels": ["C3", "C4"],
        "split": full_grid._split_metadata(
            full_grid._planned_split_identity(plan, job)
        ),
        "scalers": {
            "selection_mean_sha256": "1" * 64,
            "selection_std_sha256": "2" * 64,
            "refit_mean_sha256": "3" * 64,
            "refit_std_sha256": "4" * 64,
        },
        "fit": {
            "seed_installed_before_construction": True,
            "initial_state_sha256": "5" * 64,
            "selection_state_sha256": "6" * 64,
            "reset_state_sha256": "5" * 64,
            "reset_verified": True,
            "source_selected_epoch": 0,
            "selection_epochs_run": 1,
            "refit_epochs_run": 1,
            "refit_state_sha256": "7" * 64,
            "parameter_count": 10,
            "architecture": {
                "requested_name": job.model,
                "class": "tests.Model",
                "wrapped_class": "tests.Model",
                "scalar_attributes": {},
            },
        },
        "protocol": {
            "selection_scaler_rows": "train_only",
            "selection_optimization_rows": "train_only",
            "epoch_selection_rows": "validation_only",
            "refit_scaler_rows": "train_plus_validation",
            "refit_optimization_rows": "train_plus_validation",
            "test_use": "prediction_only",
            "test_performance_computed": False,
        },
        "timing_seconds": {
            "selection_fit": 1.0,
            "refit_fit": 2.0,
            "test_inference": 0.1,
            "job_total": 3.1,
        },
        "cuda_peak_memory_bytes": 0,
        "runtime": {
            "worker_cpu_threads": 2,
            "worker_interop_threads": 1,
            "thread_environment": {
                name: "2" for name in full_grid.THREAD_ENVIRONMENT_VARIABLES
            },
            "nvidia_driver_versions": None,
            "physical_gpu_uuid": None,
            "cuda_visible_devices": None,
        },
    }


def initialize(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    root = tmp_path / "run"
    plan = tiny_plan()
    full_grid.write_or_validate_plan(root, plan)
    return root, plan


def one_job(plan: dict[str, Any], model: str = "model_a") -> full_grid.Job:
    return full_grid.Job(dataset="toy", model=model, subject=1, fold=0, seed=7)


def claim_job(
    root: Path,
    plan: dict[str, Any],
    job: full_grid.Job,
) -> full_grid.Claim:
    return full_grid.acquire_claim(
        root,
        plan,
        job,
        before_claim=resource_status,
    )


def complete_job(
    root: Path,
    plan: dict[str, Any],
    job: full_grid.Job,
) -> Path:
    claim = claim_job(root, plan, job)
    try:
        return full_grid.commit_job_output(
            root,
            plan,
            claim,
            metadata=metadata(plan, job),
            test_rows=np.asarray([2, 3], dtype=np.int64),
            probabilities=np.asarray([[0.8, 0.2], [0.3, 0.7]], dtype=np.float64),
        )
    finally:
        full_grid.release_claim(claim)


def test_exact_common_roster_and_formal_cardinality() -> None:
    assert len(full_grid.COMMON_ARCHITECTURES) == 43
    assert len(set(full_grid.COMMON_ARCHITECTURES)) == 43
    assert full_grid.COMMON_ARCHITECTURES[-1] == "tcformer"
    contracts = full_grid._dataset_contracts()
    subject_folds = sum(
        len(value["subjects"]) * len(value["folds"])
        for value in contracts.values()
    )
    assert subject_folds == 448
    assert (
        subject_folds
        * len(full_grid.COMMON_ARCHITECTURES)
        * len(full_grid.FORMAL_SEEDS)
        == full_grid.FORMAL_EXPECTED_JOBS
        == 96_320
    )
    assert full_grid.TRACK_SCOPE == "common_recipe_43_only"


def test_formal_constructor_cannot_inject_roster_or_executor() -> None:
    assert tuple(inspect.signature(full_grid.build_plan).parameters) == (
        "cache_root",
        "worker_cpu_threads",
    )
    kwargs = {
        "dataset_contracts": {"toy": {"subjects": [1], "folds": [0], "n_classes": 2}},
        "architectures": ("tcformer",),
        "seeds": (7,),
        "train_config": {},
        "cache_identity": {"toy:s001": {"array_sha256": "a" * 64}},
        "split_identity": {
            "toy:s001:f00": {
                "dataset": "toy",
                "subject": 1,
                "fold": 0,
                "cache_array_sha256": "a" * 64,
                "trial_count": 4,
                "partitions": {
                    "train": _partition([0]),
                    "validation": _partition([1]),
                    "source": _partition([0, 1]),
                    "test": _partition([2, 3]),
                },
            }
        },
        "source_identity": {"x": "a" * 64},
        "environment_identity": {},
        "analysis_contract": {},
        "publishable": True,
    }
    with pytest.raises(ValueError, match="exact 43-model"):
        full_grid.assemble_plan(**kwargs)
    kwargs["architectures"] = full_grid.COMMON_ARCHITECTURES
    kwargs["executor"] = "evil:executor"
    with pytest.raises(ValueError, match="DEFAULT_EXECUTOR"):
        full_grid.assemble_plan(**kwargs)


def test_test_injection_is_permanently_nonpublishable() -> None:
    plan = tiny_plan(executor="tests.fake:executor")
    assert plan["publication_mode"] == full_grid.TEST_PUBLICATION_MODE
    assert plan["analysis_contract"] is None
    forged = copy.deepcopy(plan)
    forged["publication_mode"] = full_grid.FORMAL_PUBLICATION_MODE
    forged["plan_sha256"] = full_grid.plan_sha256(forged)
    with pytest.raises(full_grid.FullGridError, match="formal plan"):
        full_grid.validate_plan_semantics(forged)


def test_plan_is_deterministic_canonical_and_power_cut_repairable(tmp_path: Path) -> None:
    plan = tiny_plan()
    assert plan == tiny_plan()
    root = tmp_path / "run"
    observed = full_grid.write_or_validate_plan(root, plan)
    assert observed == plan
    assert full_grid.load_plan(root) == plan
    assert (root / "plan.json").read_bytes() == full_grid._canonical_bytes(plan) + b"\n"
    (root / "plan.sha256").chmod(0o600)
    (root / "plan.sha256").unlink()
    repaired = full_grid.load_or_repair_plan(root)
    assert repaired == plan
    assert (root / "plan.sha256").stat().st_mode & 0o222 == 0


def test_environment_package_inventory_ignores_dynamic_sys_path_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    before = full_grid._environment_package_inventory()
    injected = tmp_path / "injected_package-9.9.dist-info"
    injected.mkdir()
    (injected / "METADATA").write_text(
        "Metadata-Version: 2.1\n"
        "Name: injected-package\n"
        "Version: 9.9\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "path", [str(tmp_path), *sys.path])

    assert any(
        distribution.metadata.get("Name") == "injected-package"
        for distribution in full_grid.importlib.metadata.distributions()
    )
    assert full_grid._environment_package_inventory() == before
    assert ("injected-package", "9.9") not in before


def test_plan_roundtrip_uses_explicit_dataset_order_not_mapping_order(
    tmp_path: Path,
) -> None:
    plan = tiny_plan(datasets=("zeta", "alpha"))
    assert plan["dataset_order"] == ["zeta", "alpha"]
    assert list(plan["datasets"]) == ["zeta", "alpha"]

    serialized = json.loads(full_grid._canonical_bytes(plan))
    assert list(serialized["datasets"]) == ["alpha", "zeta"]

    root = tmp_path / "run"
    full_grid.write_or_validate_plan(root, plan)
    loaded = full_grid.load_plan(root)
    assert loaded["dataset_order"] == ["zeta", "alpha"]
    assert set(loaded["datasets"]) == {"alpha", "zeta"}
    jobs = list(full_grid.iter_jobs(loaded))
    assert jobs[0].dataset == "zeta"
    assert jobs[-1].dataset == "alpha"


def test_dataset_mapping_insertion_order_is_nonsemantic() -> None:
    plan = tiny_plan(datasets=("zeta", "alpha"))
    original_hash = plan["plan_sha256"]
    original_jobs = list(full_grid.iter_jobs(plan))
    plan["datasets"] = dict(reversed(tuple(plan["datasets"].items())))

    assert plan["plan_sha256"] == full_grid.plan_sha256(plan) == original_hash
    full_grid.validate_plan_semantics(plan)
    assert list(full_grid.iter_jobs(plan)) == original_jobs


@pytest.mark.parametrize("mutation", ("missing", "extra"))
def test_plan_rejects_dataset_mapping_membership_mismatch(
    mutation: str,
) -> None:
    plan = tiny_plan(datasets=("zeta", "alpha"))
    if mutation == "missing":
        plan["datasets"].pop("alpha")
    else:
        plan["datasets"]["intruder"] = copy.deepcopy(plan["datasets"]["alpha"])
    plan["plan_sha256"] = full_grid.plan_sha256(plan)
    with pytest.raises(full_grid.FullGridError, match="roster/grid structure"):
        full_grid.validate_plan_semantics(plan)


@pytest.mark.parametrize(
    "dataset_order",
    (
        ["zeta", "zeta"],
        ["zeta", "unknown"],
        ["zeta", 7],
        ["zeta", []],
        ["zeta", ""],
    ),
)
def test_plan_rejects_malformed_dataset_order(
    dataset_order: list[Any],
) -> None:
    plan = tiny_plan(datasets=("zeta", "alpha"))
    plan["dataset_order"] = dataset_order
    plan["plan_sha256"] = full_grid.plan_sha256(plan)
    with pytest.raises(full_grid.FullGridError, match="roster/grid structure"):
        full_grid.validate_plan_semantics(plan)


def test_plan_rejects_unknown_keys_and_self_checksummed_semantic_forgery() -> None:
    plan = tiny_plan()
    plan["hidden"] = "forgery"
    plan["plan_sha256"] = full_grid.plan_sha256(plan)
    with pytest.raises(full_grid.FullGridError, match="schema"):
        full_grid.validate_plan_semantics(plan)
    plan = tiny_plan()
    plan["n_jobs"] = 999
    plan["plan_sha256"] = full_grid.plan_sha256(plan)
    with pytest.raises(full_grid.FullGridError, match="Cartesian"):
        full_grid.validate_plan_semantics(plan)


def test_plan_rejects_extra_cache_identity_and_bool_split_identity() -> None:
    plan = tiny_plan()
    plan["cache_identity"]["toy:s001"]["hidden_labels"] = [0, 1]
    plan["plan_sha256"] = full_grid.plan_sha256(plan)
    with pytest.raises(full_grid.FullGridError, match="cache identity"):
        full_grid.validate_plan_semantics(plan)

    plan = tiny_plan()
    plan["split_identity"]["toy:s001:f00"]["subject"] = True
    plan["plan_sha256"] = full_grid.plan_sha256(plan)
    with pytest.raises(full_grid.FullGridError, match="split identity"):
        full_grid.validate_plan_semantics(plan)


def test_cache_identity_accepts_exact_label_provenance_contract_only() -> None:
    identity = _valid_formal_cache_identity()

    assert identity["preprocessing"]["labels"] == (
        "ordered exactly as DatasetSpec.events"
    )
    assert full_grid._validate_cache_identity_exact(
        identity,
        dataset="bnci2014_004",
        subject=1,
    ) == ("C3", "Cz", "C4")
    assert full_grid._recursive_outcome_aliases(
        {"labels": "ordered exactly as DatasetSpec.events"}
    ) == ["record.labels"]


@pytest.mark.parametrize(
    "value",
    [
        "ordered by an unapproved class contract",
        [0, 1, 0, 1],
    ],
)
def test_cache_identity_rejects_wrong_or_per_trial_labels(value: Any) -> None:
    identity = _valid_formal_cache_identity()
    identity["preprocessing"]["labels"] = value

    with pytest.raises(full_grid.FullGridError, match="outcome aliases"):
        full_grid._validate_cache_identity_exact(
            identity,
            dataset="bnci2014_004",
            subject=1,
        )


def test_cache_identity_label_exemption_is_preprocessing_schema_bound() -> None:
    identity = _valid_formal_cache_identity()
    identity["preprocessing"]["unapproved_provenance"] = "extra"

    with pytest.raises(full_grid.FullGridError, match="outcome aliases"):
        full_grid._validate_cache_identity_exact(
            identity,
            dataset="bnci2014_004",
            subject=1,
        )


def test_cache_identity_rejects_test_labels_and_nested_labels() -> None:
    test_labels = _valid_formal_cache_identity()
    test_labels["preprocessing"]["test_labels"] = [0, 1]
    with pytest.raises(full_grid.FullGridError, match="outcome aliases"):
        full_grid._validate_cache_identity_exact(
            test_labels,
            dataset="bnci2014_004",
            subject=1,
        )

    nested_labels = _valid_formal_cache_identity()
    nested_labels["coordinates"]["nested"] = {
        "labels": "ordered exactly as DatasetSpec.events"
    }
    with pytest.raises(full_grid.FullGridError, match="outcome aliases"):
        full_grid._validate_cache_identity_exact(
            nested_labels,
            dataset="bnci2014_004",
            subject=1,
        )


@pytest.mark.parametrize("alias", sorted(full_grid.OUTCOME_KEY_TOKENS))
def test_cache_identity_rejects_every_existing_outcome_alias(
    alias: str,
) -> None:
    identity = _valid_formal_cache_identity()
    identity["coordinates"]["nested_alias"] = {alias: "forbidden"}

    with pytest.raises(full_grid.FullGridError, match="outcome aliases"):
        full_grid._validate_cache_identity_exact(
            identity,
            dataset="bnci2014_004",
            subject=1,
        )


def test_parser_has_no_candidate_or_safety_bypasses_and_max_three() -> None:
    parser = full_grid.build_parser()
    run = parser.parse_args(
        ["run", "--run-root", "r", "--cache-root", "c"]
    )
    assert run.gpus == "0,1,2"
    help_text = parser.format_help()
    assert "extra-model" not in help_text
    assert "allow-busy" not in help_text
    assert "allow-low" not in help_text
    assert "foreign-claim-timeout" not in help_text
    assert "allow-non-venv" not in help_text
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "run",
                "--run-root",
                "r",
                "--cache-root",
                "c",
                "--max-idle-utilization",
                "100",
            ]
        )
    run.max_idle_utilization = 100.0
    with pytest.raises(ValueError, match="no override"):
        full_grid._validate_worker_safety_values(run)
    assert full_grid._parse_gpus("0,1,2") == ("0", "1", "2")
    with pytest.raises(ValueError, match="1-3"):
        full_grid._parse_gpus("0,1,2,3")


def test_orchestrator_emits_a_worker_command_accepted_by_fixed_parser(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "run"
    cache = tmp_path / "cache"
    root.mkdir()
    cache.mkdir()
    args = full_grid.build_parser().parse_args(
        [
            "run",
            "--run-root",
            str(root),
            "--cache-root",
            str(cache),
            "--gpus",
            "0",
        ]
    )
    plan = tiny_plan()
    identity = full_grid.GPUIdentity(
        index="0",
        uuid="GPU-00000001-0000-0000-0000-000000000001",
        pci_bus_id="0000:01:00.0",
        name="GPU",
    )
    captured: list[list[str]] = []

    class Process:
        def __init__(self, command: list[str], **unused: Any) -> None:
            captured.append(command)

        def wait(self) -> int:
            return 0

        def poll(self) -> int:
            return 0

        def terminate(self) -> None:
            raise AssertionError("completed synthetic process was terminated")

    monkeypatch.setattr(full_grid, "_resolve_gpus", lambda unused: (identity,))
    monkeypatch.setattr(full_grid.subprocess, "Popen", Process)
    monkeypatch.setattr(
        full_grid,
        "audit_grid",
        lambda *unused: {"exact_cartesian_complete": True},
    )
    assert full_grid._spawn_workers(args, plan) == 0
    assert len(captured) == 1
    parsed = full_grid.build_parser().parse_args(captured[0][3:])
    assert parsed.command == "worker"
    assert parsed.gpu == identity.uuid
    assert parsed.min_free_gib == full_grid.DEFAULT_MIN_FREE_GIB


def test_worker_log_open_is_anchored_across_run_root_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "run"
    displaced = tmp_path / "run-displaced"
    cache = tmp_path / "cache"
    root.mkdir()
    cache.mkdir()
    args = full_grid.build_parser().parse_args(
        [
            "run",
            "--run-root",
            str(root),
            "--cache-root",
            str(cache),
            "--gpus",
            "0",
        ]
    )
    plan = tiny_plan()
    identity = full_grid.GPUIdentity(
        index="0",
        uuid="GPU-00000001-0000-0000-0000-000000000001",
        pci_bus_id="0000:01:00.0",
        name="GPU",
    )
    monkeypatch.setattr(full_grid, "_resolve_gpus", lambda unused: (identity,))
    monkeypatch.setattr(
        full_grid.subprocess,
        "Popen",
        lambda *unused, **unused_keywords: pytest.fail(
            "worker must not spawn after ancestor replacement"
        ),
    )
    original_assert = full_grid._assert_directory_descriptor_path
    swapped = False

    def swap_after_log_open(path: Path, descriptor: int) -> None:
        nonlocal swapped
        log_root = root / "worker_logs"
        if (
            path == root
            and not swapped
            and log_root.is_dir()
            and any(log_root.iterdir())
        ):
            root.rename(displaced)
            root.mkdir()
            (root / "worker_logs").mkdir()
            swapped = True
        original_assert(path, descriptor)

    monkeypatch.setattr(
        full_grid,
        "_assert_directory_descriptor_path",
        swap_after_log_open,
    )
    try:
        with pytest.raises(full_grid.FullGridError, match="directory path changed"):
            full_grid._spawn_workers(args, plan)
        assert not any((root / "worker_logs").iterdir())
        displaced_logs = list((displaced / "worker_logs").iterdir())
        assert len(displaced_logs) == 1
        assert displaced_logs[0].name.startswith("worker-00-gpu-0-")
    finally:
        if swapped:
            (root / "worker_logs").rmdir()
            root.rmdir()
            displaced.rename(root)


def test_preflight_enumerates_exact_43_times_four(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_metadata(
        *, cache_root: Path, dataset: str, subject: int
    ) -> dict[str, Any]:
        contract = next(
            value
            for value in full_grid.MODEL_PREFLIGHT_CONTRACTS
            if value["dataset"] == dataset
        )
        channels = tuple(f"C{index}" for index in range(contract["n_channels"]))
        return {
            "path": str(cache_root / f"{dataset}.npz"),
            "identity_shape": [8, contract["n_channels"], contract["n_times"]],
            "declared_cache_array_sha256": "a" * 64,
            "channel_names": channels,
            "positions": np.ones((contract["n_channels"], 3), dtype=np.float32),
            "members_opened": ["identity", "positions", "channel_names"],
        }

    monkeypatch.setattr(full_grid, "_load_preflight_cache_metadata", fake_metadata)
    monkeypatch.setattr(
        full_grid,
        "_source_identity",
        lambda **unused: {"src/benchmark/full_grid.py": "a" * 64},
    )
    monkeypatch.setattr(full_grid, "_environment_identity", lambda: {})
    report = full_grid.run_model_cuda_preflight(
        cache_root=Path("/synthetic"),
        _model_check=lambda **unused: {"parameter_count": 1},
    )
    assert report["architectures"] == list(full_grid.COMMON_ARCHITECTURES)
    assert report["track_scope"] == full_grid.TRACK_SCOPE
    assert report["expected_checks"] == report["completed_checks"] == 172
    assert len(report["results"]) == 172


def test_preflight_rejects_bool_parameter_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_metadata(
        *, cache_root: Path, dataset: str, subject: int
    ) -> dict[str, Any]:
        contract = next(
            value
            for value in full_grid.MODEL_PREFLIGHT_CONTRACTS
            if value["dataset"] == dataset
        )
        channels = tuple(f"C{index}" for index in range(contract["n_channels"]))
        return {
            "path": str(cache_root / f"{dataset}.npz"),
            "identity_shape": [8, contract["n_channels"], contract["n_times"]],
            "declared_cache_array_sha256": "a" * 64,
            "channel_names": channels,
            "positions": np.ones((contract["n_channels"], 3), dtype=np.float32),
            "members_opened": ["identity", "positions", "channel_names"],
        }

    monkeypatch.setattr(full_grid, "_load_preflight_cache_metadata", fake_metadata)
    with pytest.raises(full_grid.FullGridError, match="parameter count"):
        full_grid.run_model_cuda_preflight(
            cache_root=Path("/synthetic"),
            _model_check=lambda **unused: {"parameter_count": True},
        )


def _synthetic_preflight_report(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    def fake_metadata(
        *, cache_root: Path, dataset: str, subject: int
    ) -> dict[str, Any]:
        contract = next(
            value
            for value in full_grid.MODEL_PREFLIGHT_CONTRACTS
            if value["dataset"] == dataset
        )
        channels = tuple(f"C{index}" for index in range(contract["n_channels"]))
        return {
            "path": str(cache_root / f"{dataset}.npz"),
            "identity_shape": [8, contract["n_channels"], contract["n_times"]],
            "declared_cache_array_sha256": "a" * 64,
            "channel_names": channels,
            "positions": np.ones(
                (contract["n_channels"], 3),
                dtype=np.float32,
            ),
            "members_opened": ["identity", "positions", "channel_names"],
        }

    monkeypatch.setattr(full_grid, "_load_preflight_cache_metadata", fake_metadata)
    monkeypatch.setattr(
        full_grid,
        "_source_identity",
        lambda **unused: {"src/benchmark/full_grid.py": "a" * 64},
    )
    monkeypatch.setattr(full_grid, "_environment_identity", lambda: {})
    return full_grid.run_model_cuda_preflight(
        cache_root=Path("/synthetic"),
        _model_check=lambda **unused: {"parameter_count": 1},
    )


def _formalize_preflight_plan(
    report: Mapping[str, Any],
    *,
    gpu_uuid: str,
) -> dict[str, Any]:
    plan = tiny_plan()
    plan["publication_mode"] = full_grid.FORMAL_PUBLICATION_MODE
    plan["source_identity"] = copy.deepcopy(report["source_identity"])
    plan["analysis_contract"] = {
        "environment_identity_sha256": report[
            "environment_identity_sha256"
        ]
    }
    plan["environment_identity"] = {
        "nvidia_gpu_inventory": [
            {
                "index": "0",
                "uuid": gpu_uuid,
                "pci_bus_id": "0000:01:00.0",
                "name": "Synthetic CUDA GPU",
                "driver_version": "1",
            }
        ]
    }
    plan["execution_config"]["physical_gpu_uuid_roster"] = [gpu_uuid]
    plan["cache_identity"] = {
        full_grid._subject_identity_key(
            str(contract["dataset"]),
            int(contract["subject"]),
        ): {"array_sha256": "a" * 64}
        for contract in full_grid.MODEL_PREFLIGHT_CONTRACTS
    }
    return plan


def _formalize_preflight_report(
    report: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    gpu_uuid: str,
    gpu_lease_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    formal = copy.deepcopy(dict(report))
    formal["plan_sha256"] = plan["plan_sha256"]
    formal["physical_gpu_uuid"] = gpu_uuid
    formal["gpu_lease_receipt"] = copy.deepcopy(dict(gpu_lease_receipt))
    formal["execution_backend"] = "cuda_forward_backward_adamw"
    formal["cuda_device_name"] = "Synthetic CUDA GPU"
    for row in formal["results"]:
        row["forward_backward_optimizer_step"] = "passed"
    return formal


def test_formal_preflight_attestation_is_exact_immutable_and_one_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _synthetic_preflight_report(monkeypatch)
    gpu_uuid = "GPU-00000001-0000-0000-0000-000000000001"
    plan = _formalize_preflight_plan(report, gpu_uuid=gpu_uuid)
    root = tmp_path / "run"
    root.mkdir()
    monkeypatch.setattr(full_grid, "_verified_project_root", lambda: tmp_path)
    lease = project_gpu_leases.acquire_gpu_lease(
        project_root=tmp_path,
        run_root=root,
        plan_sha256=plan["plan_sha256"],
        gpu_uuid=gpu_uuid,
        track_scope=full_grid.TRACK_SCOPE,
    )
    try:
        with project_gpu_leases.guard_gpu_lease(lease) as lease_receipt:
            formal_report = _formalize_preflight_report(
                report,
                plan=plan,
                gpu_uuid=gpu_uuid,
                gpu_lease_receipt=lease_receipt,
            )
            attestation = full_grid._publish_preflight_attestation(
                run_root=root,
                plan=plan,
                report=formal_report,
                publication_gpu_lease_receipt=lease_receipt,
            )
        assert attestation["report"]["completed_checks"] == 172
        assert len(attestation["report"]["results"]) == 172
        assert (
            attestation["receipt"]["report_sha256"]
            == attestation["report_sha256"]
        )
        for name in (
            full_grid.PREFLIGHT_REPORT_FILENAME,
            full_grid.PREFLIGHT_RECEIPT_FILENAME,
        ):
            assert (
                root / full_grid.PREFLIGHT_DIRECTORY / name
            ).stat().st_mode & 0o222 == 0
        with project_gpu_leases.guard_gpu_lease(lease) as lease_receipt:
            with pytest.raises(full_grid.FullGridError, match="already exists"):
                full_grid._publish_preflight_attestation(
                    run_root=root,
                    plan=plan,
                    report=formal_report,
                    publication_gpu_lease_receipt=lease_receipt,
                )

        unexpected = root / full_grid.PREFLIGHT_DIRECTORY / "unexpected"
        unexpected.write_bytes(b"x")
        with pytest.raises(full_grid.FullGridError, match="exact completed"):
            full_grid.load_preflight_attestation(root, plan)
        unexpected.unlink()

        receipt_path = (
            root
            / full_grid.PREFLIGHT_DIRECTORY
            / full_grid.PREFLIGHT_RECEIPT_FILENAME
        )
        forged_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        forged_receipt["report_sha256"] = "0" * 64
        receipt_path.chmod(0o600)
        receipt_path.write_bytes(
            full_grid._canonical_bytes(forged_receipt) + b"\n"
        )
        receipt_path.chmod(0o444)
        with pytest.raises(full_grid.FullGridError, match="differs"):
            full_grid.load_preflight_attestation(root, plan)
    finally:
        project_gpu_leases.release_gpu_lease(lease)


def test_preflight_report_first_power_cut_repairs_only_missing_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _synthetic_preflight_report(monkeypatch)
    gpu_uuid = "GPU-00000001-0000-0000-0000-000000000001"
    plan = _formalize_preflight_plan(report, gpu_uuid=gpu_uuid)
    root = tmp_path / "run"
    root.mkdir()
    monkeypatch.setattr(full_grid, "_verified_project_root", lambda: tmp_path)
    first_lease = project_gpu_leases.acquire_gpu_lease(
        project_root=tmp_path,
        run_root=root,
        plan_sha256=plan["plan_sha256"],
        gpu_uuid=gpu_uuid,
        track_scope=full_grid.TRACK_SCOPE,
    )
    original_write = full_grid._write_bytes_exclusive_at

    def power_cut(
        parent_descriptor: int,
        leaf_name: str,
        payload: bytes,
        *,
        mode: int = 0o444,
    ) -> None:
        if leaf_name == full_grid.PREFLIGHT_RECEIPT_FILENAME:
            raise OSError("synthetic power cut after report")
        original_write(
            parent_descriptor,
            leaf_name,
            payload,
            mode=mode,
        )

    try:
        with project_gpu_leases.guard_gpu_lease(first_lease) as first_receipt:
            formal_report = _formalize_preflight_report(
                report,
                plan=plan,
                gpu_uuid=gpu_uuid,
                gpu_lease_receipt=first_receipt,
            )
            monkeypatch.setattr(
                full_grid,
                "_write_bytes_exclusive_at",
                power_cut,
            )
            with pytest.raises(OSError, match="power cut"):
                full_grid._publish_preflight_attestation(
                    run_root=root,
                    plan=plan,
                    report=formal_report,
                    publication_gpu_lease_receipt=first_receipt,
                )
        report_path = (
            root
            / full_grid.PREFLIGHT_DIRECTORY
            / full_grid.PREFLIGHT_REPORT_FILENAME
        )
        original_report_payload = report_path.read_bytes()
        assert not (
            root
            / full_grid.PREFLIGHT_DIRECTORY
            / full_grid.PREFLIGHT_RECEIPT_FILENAME
        ).exists()
    finally:
        project_gpu_leases.release_gpu_lease(first_lease)

    monkeypatch.setattr(full_grid, "_write_bytes_exclusive_at", original_write)
    monkeypatch.setattr(
        full_grid,
        "_preflight_runtime_identity",
        lambda unused: (copy.deepcopy(plan["source_identity"]), {}),
    )
    monkeypatch.setattr(
        full_grid,
        "_formal_preflight_cuda_binding",
        lambda **unused: "Synthetic CUDA GPU",
    )
    second_lease = project_gpu_leases.acquire_gpu_lease(
        project_root=tmp_path,
        run_root=root,
        plan_sha256=plan["plan_sha256"],
        gpu_uuid=gpu_uuid,
        track_scope=full_grid.TRACK_SCOPE,
    )
    try:
        recovered_report = full_grid.run_model_cuda_preflight(
            cache_root=tmp_path / "cache-not-opened",
            plan=plan,
            run_root=root,
            gpu_lease=second_lease,
        )
        attestation = full_grid.load_preflight_attestation(root, plan)
        assert report_path.read_bytes() == original_report_payload
        assert recovered_report == attestation["report"]
        assert (
            attestation["report"]["gpu_lease_receipt"]["lease"]["nonce"]
            != attestation["receipt"]["publication_gpu_lease_receipt"][
                "lease"
            ]["nonce"]
        )
    finally:
        project_gpu_leases.release_gpu_lease(second_lease)


def test_direct_formal_preflight_rejects_cuda_environment_mismatch_before_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _synthetic_preflight_report(monkeypatch)
    gpu_uuid = "GPU-00000001-0000-0000-0000-000000000001"
    plan = _formalize_preflight_plan(report, gpu_uuid=gpu_uuid)
    root = tmp_path / "run"
    root.mkdir()
    monkeypatch.setattr(full_grid, "_verified_project_root", lambda: tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-wrong")
    monkeypatch.setenv("FULL_GRID_GPU_UUID", gpu_uuid)
    lease = project_gpu_leases.acquire_gpu_lease(
        project_root=tmp_path,
        run_root=root,
        plan_sha256=plan["plan_sha256"],
        gpu_uuid=gpu_uuid,
        track_scope=full_grid.TRACK_SCOPE,
    )
    try:
        with pytest.raises(
            full_grid.GPUUnavailable,
            match="CUDA environment differs",
        ):
            full_grid.run_model_cuda_preflight(
                cache_root=tmp_path / "cache-must-not-open",
                plan=plan,
                run_root=root,
                gpu_lease=lease,
            )
        assert not (root / full_grid.PREFLIGHT_DIRECTORY).exists()
    finally:
        project_gpu_leases.release_gpu_lease(lease)


def test_preflight_stage_recovery_is_repeatable_and_preserves_unknown_entries(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    preflight = root / full_grid.PREFLIGHT_DIRECTORY
    preflight.mkdir(parents=True)
    stages = (
        f".report.json.stage-{'a' * 32}",
        f".receipt.json.stage-{'b' * 32}",
    )
    for name in stages:
        (preflight / name).write_bytes(b"crash-stage")
    expected_stages = tuple(sorted(stages))
    assert full_grid._preflight_stage_names(root) == expected_stages
    assert full_grid._cleanup_preflight_stages(root) == expected_stages
    assert full_grid._cleanup_preflight_stages(root) == ()

    exact_stage = f".report.json.stage-{'c' * 32}"
    (preflight / exact_stage).write_bytes(b"known")
    unknown = preflight / ".report.json.stage-not-a-schema-token"
    unknown.write_bytes(b"unknown-must-survive")
    with pytest.raises(full_grid.FullGridError, match="unknown"):
        full_grid._cleanup_preflight_stages(root)
    assert (preflight / exact_stage).read_bytes() == b"known"
    assert unknown.read_bytes() == b"unknown-must-survive"


def test_preflight_stage_cleanup_detects_preflight_directory_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "run"
    preflight = root / full_grid.PREFLIGHT_DIRECTORY
    displaced = root / "preflight-displaced"
    preflight.mkdir(parents=True)
    stage = f".report.json.stage-{'a' * 32}"
    (preflight / stage).write_bytes(b"old-stage")
    original_unlink = full_grid._unlink_exact_preflight_stage_at
    swapped = False

    def unlink_then_swap(descriptor: int, name: str) -> None:
        nonlocal swapped
        original_unlink(descriptor, name)
        preflight.rename(displaced)
        preflight.mkdir()
        (preflight / "replacement-must-survive").write_bytes(b"replacement")
        swapped = True

    monkeypatch.setattr(
        full_grid,
        "_unlink_exact_preflight_stage_at",
        unlink_then_swap,
    )
    with pytest.raises(full_grid.FullGridError, match="directory path changed"):
        full_grid._cleanup_preflight_stages(root)
    assert swapped
    assert (
        preflight / "replacement-must-survive"
    ).read_bytes() == b"replacement"
    assert not (displaced / stage).exists()


def test_full_preflight_guard_exit_failure_quarantines_complete_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _synthetic_preflight_report(monkeypatch)
    gpu_uuid = "GPU-00000001-0000-0000-0000-000000000001"
    plan = _formalize_preflight_plan(report, gpu_uuid=gpu_uuid)
    root = tmp_path / "run"
    root.mkdir()
    monkeypatch.setattr(full_grid, "_verified_project_root", lambda: tmp_path)
    lease = project_gpu_leases.acquire_gpu_lease(
        project_root=tmp_path,
        run_root=root,
        plan_sha256=plan["plan_sha256"],
        gpu_uuid=gpu_uuid,
        track_scope=full_grid.TRACK_SCOPE,
    )
    original_guard = project_gpu_leases.guard_gpu_lease

    @contextmanager
    def fail_after_yield(selected: project_gpu_leases.GPULease):
        with original_guard(selected) as receipt:
            yield receipt
        raise project_gpu_leases.ProjectGPULeaseError(
            "synthetic post-yield lease loss"
        )

    monkeypatch.setattr(
        project_gpu_leases,
        "guard_gpu_lease",
        fail_after_yield,
    )
    try:
        with pytest.raises(full_grid.FullGridError, match="authority was lost"):
            full_grid._publish_preflight_under_lease(
                run_root=root,
                plan=plan,
                gpu_lease=lease,
                report_builder=lambda receipt: _formalize_preflight_report(
                    report,
                    plan=plan,
                    gpu_uuid=gpu_uuid,
                    gpu_lease_receipt=receipt,
                ),
                keep_report_on_postcondition_failure=False,
            )
        assert not (root / full_grid.PREFLIGHT_DIRECTORY).exists()
        quarantine = root / "quarantine" / "preflight"
        reports = list(
            quarantine.glob("report.json.lease-postcondition-report.*")
        )
        receipts = list(
            quarantine.glob("receipt.json.lease-postcondition-receipt.*")
        )
        assert len(reports) == len(receipts) == 1
        assert reports[0].is_file() and receipts[0].is_file()
        with pytest.raises(full_grid.FullGridError, match="requires"):
            full_grid.load_preflight_attestation(root, plan)
    finally:
        project_gpu_leases.release_gpu_lease(lease)


def test_full_preflight_postpublish_load_failure_quarantines_complete_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _synthetic_preflight_report(monkeypatch)
    gpu_uuid = "GPU-00000001-0000-0000-0000-000000000001"
    plan = _formalize_preflight_plan(report, gpu_uuid=gpu_uuid)
    root = tmp_path / "run"
    root.mkdir()
    monkeypatch.setattr(full_grid, "_verified_project_root", lambda: tmp_path)
    lease = project_gpu_leases.acquire_gpu_lease(
        project_root=tmp_path,
        run_root=root,
        plan_sha256=plan["plan_sha256"],
        gpu_uuid=gpu_uuid,
        track_scope=full_grid.TRACK_SCOPE,
    )

    def fail_after_pair(
        unused_root: Path,
        unused_plan: Mapping[str, Any],
    ) -> dict[str, Any]:
        raise full_grid.FullGridError("synthetic postpublish load failure")

    monkeypatch.setattr(
        full_grid,
        "load_preflight_attestation",
        fail_after_pair,
    )
    try:
        with pytest.raises(full_grid.FullGridError, match="postpublish"):
            full_grid._publish_preflight_under_lease(
                run_root=root,
                plan=plan,
                gpu_lease=lease,
                report_builder=lambda receipt: _formalize_preflight_report(
                    report,
                    plan=plan,
                    gpu_uuid=gpu_uuid,
                    gpu_lease_receipt=receipt,
                ),
                keep_report_on_postcondition_failure=False,
            )
        assert not (root / full_grid.PREFLIGHT_DIRECTORY).exists()
        quarantine = root / "quarantine" / "preflight"
        reports = list(
            quarantine.glob("report.json.lease-postcondition-report.*")
        )
        receipts = list(
            quarantine.glob("receipt.json.lease-postcondition-receipt.*")
        )
        assert len(reports) == len(receipts) == 1
        assert reports[0].is_file() and receipts[0].is_file()
    finally:
        project_gpu_leases.release_gpu_lease(lease)


def test_concurrent_preflight_loser_accepts_winner_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _synthetic_preflight_report(monkeypatch)
    gpu_uuid = "GPU-00000001-0000-0000-0000-000000000001"
    plan = _formalize_preflight_plan(report, gpu_uuid=gpu_uuid)
    root = tmp_path / "run"
    root.mkdir()
    monkeypatch.setattr(full_grid, "_verified_project_root", lambda: tmp_path)
    real_lease = project_gpu_leases.acquire_gpu_lease(
        project_root=tmp_path,
        run_root=root,
        plan_sha256=plan["plan_sha256"],
        gpu_uuid=gpu_uuid,
        track_scope=full_grid.TRACK_SCOPE,
    )
    loser_token = object()
    winner_token = object()
    loser_waiting = threading.Event()
    winner_done = threading.Event()
    builders: list[object] = []
    try:
        with project_gpu_leases.guard_gpu_lease(real_lease) as receipt:
            saved_receipt = copy.deepcopy(receipt)

        @contextmanager
        def ordered_authority(
            *,
            gpu_lease: object,
            **unused: Any,
        ):
            if gpu_lease is loser_token:
                loser_waiting.set()
                if not winner_done.wait(timeout=10.0):
                    raise AssertionError("winner did not publish")
            yield copy.deepcopy(saved_receipt)

        monkeypatch.setattr(
            full_grid,
            "_gpu_lease_publication_authority",
            ordered_authority,
        )

        def publish(token: object) -> dict[str, Any]:
            def build(selected_receipt: Mapping[str, Any]) -> dict[str, Any]:
                builders.append(token)
                return _formalize_preflight_report(
                    report,
                    plan=plan,
                    gpu_uuid=gpu_uuid,
                    gpu_lease_receipt=selected_receipt,
                )

            return full_grid._publish_preflight_under_lease(
                run_root=root,
                plan=plan,
                gpu_lease=token,  # type: ignore[arg-type]
                report_builder=build,
                keep_report_on_postcondition_failure=False,
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            loser_future = executor.submit(publish, loser_token)
            assert loser_waiting.wait(timeout=10.0)
            winner = publish(winner_token)
            winner_snapshot = full_grid._preflight_canonical_snapshot(root)
            winner_done.set()
            loser = loser_future.result(timeout=10.0)

        assert loser == winner
        assert builders == [winner_token]
        assert full_grid._preflight_canonical_snapshot(root) == winner_snapshot
        assert not (root / "quarantine" / "preflight").exists()
    finally:
        winner_done.set()
        project_gpu_leases.release_gpu_lease(real_lease)


def test_preflight_cleanup_preserves_winner_that_reuses_tracked_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _synthetic_preflight_report(monkeypatch)
    gpu_uuid = "GPU-00000001-0000-0000-0000-000000000001"
    plan = _formalize_preflight_plan(report, gpu_uuid=gpu_uuid)
    root = tmp_path / "run"
    root.mkdir()
    monkeypatch.setattr(full_grid, "_verified_project_root", lambda: tmp_path)
    lease = project_gpu_leases.acquire_gpu_lease(
        project_root=tmp_path,
        run_root=root,
        plan_sha256=plan["plan_sha256"],
        gpu_uuid=gpu_uuid,
        track_scope=full_grid.TRACK_SCOPE,
    )
    try:
        with project_gpu_leases.guard_gpu_lease(lease) as receipt:
            formal_report = _formalize_preflight_report(
                report,
                plan=plan,
                gpu_uuid=gpu_uuid,
                gpu_lease_receipt=receipt,
            )
            full_grid._publish_preflight_attestation(
                run_root=root,
                plan=plan,
                report=formal_report,
                publication_gpu_lease_receipt=receipt,
            )
        winner = full_grid._preflight_canonical_snapshot(root)
        tracker = {
            "initial_directory_identity": None,
            "initial_artifacts": {},
            "directory_identity": winner["directory_identity"],
            # Model the loser having published the report, followed by a
            # serialized winner that validated those bytes and added receipt.
            "artifacts": {
                full_grid.PREFLIGHT_REPORT_FILENAME: copy.deepcopy(
                    winner["artifacts"][full_grid.PREFLIGHT_REPORT_FILENAME]
                )
            },
        }
        with project_gpu_leases._registry_lock(tmp_path):
            assert full_grid._invalidate_preflight_publication(
                root,
                keep_report=False,
                plan=plan,
                publication_tracker=tracker,
            ) == ()
        assert full_grid._preflight_canonical_snapshot(root) == winner
        assert full_grid.load_preflight_attestation(root, plan)
        assert not (root / "quarantine" / "preflight").exists()
    finally:
        project_gpu_leases.release_gpu_lease(lease)


def test_report_only_recovery_guard_exit_failure_preserves_report_and_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _synthetic_preflight_report(monkeypatch)
    gpu_uuid = "GPU-00000001-0000-0000-0000-000000000001"
    plan = _formalize_preflight_plan(report, gpu_uuid=gpu_uuid)
    root = tmp_path / "run"
    root.mkdir()
    monkeypatch.setattr(full_grid, "_verified_project_root", lambda: tmp_path)
    lease = project_gpu_leases.acquire_gpu_lease(
        project_root=tmp_path,
        run_root=root,
        plan_sha256=plan["plan_sha256"],
        gpu_uuid=gpu_uuid,
        track_scope=full_grid.TRACK_SCOPE,
    )
    original_write = full_grid._write_bytes_exclusive_at
    original_guard = project_gpu_leases.guard_gpu_lease

    def cut_receipt(
        descriptor: int,
        name: str,
        payload: bytes,
        *,
        mode: int = 0o444,
    ) -> None:
        if name == full_grid.PREFLIGHT_RECEIPT_FILENAME:
            raise OSError("synthetic report-only power cut")
        original_write(descriptor, name, payload, mode=mode)

    try:
        with original_guard(lease) as receipt:
            formal_report = _formalize_preflight_report(
                report,
                plan=plan,
                gpu_uuid=gpu_uuid,
                gpu_lease_receipt=receipt,
            )
            monkeypatch.setattr(
                full_grid,
                "_write_bytes_exclusive_at",
                cut_receipt,
            )
            with pytest.raises(OSError, match="report-only"):
                full_grid._publish_preflight_attestation(
                    run_root=root,
                    plan=plan,
                    report=formal_report,
                    publication_gpu_lease_receipt=receipt,
                )
        monkeypatch.setattr(
            full_grid,
            "_write_bytes_exclusive_at",
            original_write,
        )
        report_path = (
            root
            / full_grid.PREFLIGHT_DIRECTORY
            / full_grid.PREFLIGHT_REPORT_FILENAME
        )
        report_payload = report_path.read_bytes()

        @contextmanager
        def fail_after_yield(selected: project_gpu_leases.GPULease):
            with original_guard(selected) as receipt:
                yield receipt
            raise project_gpu_leases.ProjectGPULeaseError(
                "synthetic post-yield lease loss"
            )

        monkeypatch.setattr(
            project_gpu_leases,
            "guard_gpu_lease",
            fail_after_yield,
        )
        monkeypatch.setattr(
            full_grid,
            "_formal_preflight_cuda_binding",
            lambda **unused: "Synthetic CUDA GPU",
        )
        monkeypatch.setattr(
            full_grid,
            "_preflight_runtime_identity",
            lambda unused: (copy.deepcopy(plan["source_identity"]), {}),
        )
        with pytest.raises(full_grid.FullGridError, match="authority was lost"):
            full_grid.run_model_cuda_preflight(
                cache_root=tmp_path / "cache-not-opened",
                plan=plan,
                run_root=root,
                gpu_lease=lease,
            )
        assert report_path.read_bytes() == report_payload
        assert not (
            root
            / full_grid.PREFLIGHT_DIRECTORY
            / full_grid.PREFLIGHT_RECEIPT_FILENAME
        ).exists()
        quarantined = list(
            (root / "quarantine" / "preflight").glob(
                "receipt.json.lease-postcondition-receipt.*"
            )
        )
        assert len(quarantined) == 1 and quarantined[0].is_file()

        monkeypatch.setattr(
            project_gpu_leases,
            "guard_gpu_lease",
            original_guard,
        )
        recovered = full_grid.run_model_cuda_preflight(
            cache_root=tmp_path / "cache-not-opened",
            plan=plan,
            run_root=root,
            gpu_lease=lease,
        )
        assert recovered == full_grid.load_preflight_attestation(
            root,
            plan,
        )["report"]
        assert report_path.read_bytes() == report_payload
    finally:
        project_gpu_leases.release_gpu_lease(lease)


def test_preflight_pair_loader_rejects_run_root_ancestor_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _synthetic_preflight_report(monkeypatch)
    gpu_uuid = "GPU-00000001-0000-0000-0000-000000000001"
    plan = _formalize_preflight_plan(report, gpu_uuid=gpu_uuid)
    root = tmp_path / "run"
    displaced = tmp_path / "run-displaced"
    root.mkdir()
    monkeypatch.setattr(full_grid, "_verified_project_root", lambda: tmp_path)
    lease = project_gpu_leases.acquire_gpu_lease(
        project_root=tmp_path,
        run_root=root,
        plan_sha256=plan["plan_sha256"],
        gpu_uuid=gpu_uuid,
        track_scope=full_grid.TRACK_SCOPE,
    )
    swapped = False
    try:
        with project_gpu_leases.guard_gpu_lease(lease) as lease_receipt:
            formal_report = _formalize_preflight_report(
                report,
                plan=plan,
                gpu_uuid=gpu_uuid,
                gpu_lease_receipt=lease_receipt,
            )
            full_grid._publish_preflight_attestation(
                run_root=root,
                plan=plan,
                report=formal_report,
                publication_gpu_lease_receipt=lease_receipt,
            )
        original_read = full_grid._read_unique_regular_at

        def swap_after_report(
            parent_descriptor: int,
            leaf_name: str,
        ) -> bytes:
            nonlocal swapped
            payload = original_read(parent_descriptor, leaf_name)
            if (
                leaf_name == full_grid.PREFLIGHT_REPORT_FILENAME
                and not swapped
            ):
                root.rename(displaced)
                root.mkdir()
                swapped = True
            return payload

        monkeypatch.setattr(
            full_grid,
            "_read_unique_regular_at",
            swap_after_report,
        )
        with pytest.raises(
            full_grid.FullGridError,
            match="directory path changed",
        ):
            full_grid.load_preflight_attestation(root, plan)
        assert not (root / full_grid.PREFLIGHT_DIRECTORY).exists()
        assert (
            displaced
            / full_grid.PREFLIGHT_DIRECTORY
            / full_grid.PREFLIGHT_RECEIPT_FILENAME
        ).is_file()
    finally:
        if swapped:
            root.rmdir()
            displaced.rename(root)
        project_gpu_leases.release_gpu_lease(lease)


def test_formal_claim_refuses_missing_preflight_before_creating_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _synthetic_preflight_report(monkeypatch)
    gpu_uuid = "GPU-00000001-0000-0000-0000-000000000001"
    plan = _formalize_preflight_plan(report, gpu_uuid=gpu_uuid)
    root = tmp_path / "run"
    root.mkdir()
    monkeypatch.setattr(
        full_grid,
        "validate_plan_semantics",
        lambda *unused, **unused_keywords: None,
    )
    lease = project_gpu_leases.acquire_gpu_lease(
        project_root=tmp_path,
        run_root=root,
        plan_sha256=plan["plan_sha256"],
        gpu_uuid=gpu_uuid,
        track_scope=full_grid.TRACK_SCOPE,
    )
    try:
        with pytest.raises(full_grid.FullGridError, match="exact completed"):
            full_grid.acquire_claim(
                root,
                plan,
                one_job(plan),
                before_claim=resource_status,
                gpu_lease=lease,
            )
        assert list(root.iterdir()) == []
    finally:
        project_gpu_leases.release_gpu_lease(lease)


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo"])
def test_descriptor_first_regular_read_rejects_aliases_and_special_nodes(
    tmp_path: Path,
    kind: str,
) -> None:
    original = tmp_path / "original"
    original.write_bytes(b"payload")
    path = tmp_path / "target"
    if kind == "symlink":
        path.symlink_to(original)
    elif kind == "hardlink":
        os.link(original, path)
    else:
        os.mkfifo(path)
    with pytest.raises((OSError, full_grid.FullGridError)):
        full_grid._read_unique_regular(path)


def test_symlinked_ancestor_rejected_before_run_creation(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(full_grid.FullGridError, match="symlinked"):
        full_grid._safe_run_root(alias / "run", create=True)
    assert not (real / "run").exists()


def test_exclusive_write_survives_power_cut_stage_and_cleans_it(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "authority"
    parent.mkdir()
    stage = parent / f".claim.json.stage-{'a' * 32}"
    stage.write_bytes(b'{"torn":')
    destination = parent / "claim.json"
    full_grid._write_bytes_exclusive(destination, b'{"complete":true}\n')
    assert destination.read_bytes() == b'{"complete":true}\n'
    assert not stage.exists()
    assert destination.stat().st_mode & 0o222 == 0


def test_exclusive_writes_serialize_concurrent_stage_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "authority"
    parent.mkdir()
    original_fsync = full_grid.os.fsync

    def slowed_fsync(descriptor: int) -> None:
        time.sleep(0.002)
        original_fsync(descriptor)

    monkeypatch.setattr(full_grid.os, "fsync", slowed_fsync)

    def write(index: int) -> None:
        full_grid._write_bytes_exclusive(
            parent / f"claim-{index}.json",
            b'{"complete":true}\n',
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(write, range(32)))
    assert {path.name for path in parent.iterdir()} == {
        f"claim-{index}.json" for index in range(32)
    }


def test_exclusive_writes_same_leaf_have_one_publish_and_no_stage_loss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "authority"
    parent.mkdir()
    original_fsync = full_grid.os.fsync

    def slowed_fsync(descriptor: int) -> None:
        time.sleep(0.002)
        original_fsync(descriptor)

    monkeypatch.setattr(full_grid.os, "fsync", slowed_fsync)

    def write() -> str:
        try:
            full_grid._write_bytes_exclusive(
                parent / "publication.fence",
                b"fence",
            )
        except FileExistsError:
            return "exists"
        return "published"

    with ThreadPoolExecutor(max_workers=32) as pool:
        outcomes = list(pool.map(lambda _: write(), range(32)))
    assert outcomes.count("published") == 1
    assert outcomes.count("exists") == 31
    assert list(parent.iterdir()) == [parent / "publication.fence"]


def test_exclusive_write_ancestor_swap_never_writes_replacement_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = tmp_path / "authority"
    displaced = tmp_path / "displaced"
    authority.mkdir()
    original = full_grid._write_bytes_exclusive_at

    def swap_then_write(
        descriptor: int,
        name: str,
        payload: bytes,
        *,
        mode: int = 0o444,
        on_publish: Any = None,
    ) -> os.stat_result:
        authority.rename(displaced)
        authority.mkdir()
        return original(
            descriptor,
            name,
            payload,
            mode=mode,
            on_publish=on_publish,
        )

    monkeypatch.setattr(full_grid, "_write_bytes_exclusive_at", swap_then_write)
    with pytest.raises(full_grid.FullGridError, match="directory path changed"):
        full_grid._write_bytes_exclusive(
            authority / "claim.json",
            b'{"complete":true}\n',
        )
    assert not (authority / "claim.json").exists()
    assert (displaced / "claim.json").read_bytes() == b'{"complete":true}\n'


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo"])
def test_cache_open_rejects_aliases_before_numpy(
    tmp_path: Path,
    kind: str,
) -> None:
    from benchmark.config import DEFAULT_MONTAGE_PROFILE
    from benchmark.data import _cache_path

    cache_root = tmp_path / "cache"
    target = _cache_path(
        cache_root,
        "local_exp4",
        1,
        DEFAULT_MONTAGE_PROFILE,
    )
    target.parent.mkdir(parents=True)
    source = tmp_path / "source"
    source.write_bytes(b"not-an-npz")
    if kind == "symlink":
        target.symlink_to(source)
    elif kind == "hardlink":
        os.link(source, target)
    else:
        os.mkfifo(target)
    with pytest.raises((OSError, full_grid.FullGridError)):
        full_grid._load_subject_cache_safely(
            "local_exp4",
            1,
            cache_root=cache_root,
        )


def test_uv_closure_requires_runtime_test_lock_and_no_default_groups() -> None:
    runtime = sorted(full_grid.REQUIRED_UV_RUNTIME_PACKAGES)
    dependencies = ", ".join(json.dumps(value) for value in runtime)
    packages = "\n".join(
        f'[[package]]\nname = "{value}"\nversion = "1"' for value in (*runtime, "pytest")
    )
    pyproject = (
        '[project]\nname="x"\nversion="0"\n'
        f"dependencies=[{dependencies}]\n"
        '[dependency-groups]\ntest=["pytest"]\n'
        '[tool.uv]\ndefault-groups=[]\n'
    )
    full_grid.validate_uv_dependency_closure(pyproject, f"version=1\n{packages}\n")
    with pytest.raises(full_grid.FullGridError, match="UV closure"):
        full_grid.validate_uv_dependency_closure(
            pyproject.replace("default-groups=[]", 'default-groups=["test"]'),
            f"version=1\n{packages}\n",
        )


def test_disk_floor_has_no_bypass() -> None:
    full_grid._validate_disk_threshold(50.0, allow_low_disk=False)
    with pytest.raises(full_grid.FullGridError, match="no bypass"):
        full_grid._validate_disk_threshold(50.0, allow_low_disk=True)
    with pytest.raises(full_grid.FullGridError, match="exact 50"):
        full_grid._validate_disk_threshold(49.999)
    with pytest.raises(full_grid.FullGridError, match="cannot be below"):
        full_grid.probe_disk(Path("/"), minimum_free_gib=49.0)


def test_claim_resource_validation_enforces_floor_inside_acquire(tmp_path: Path) -> None:
    root, plan = initialize(tmp_path)
    with pytest.raises(full_grid.FullGridError, match="hard safety floor"):
        full_grid.acquire_claim(
            root,
            plan,
            one_job(plan),
            before_claim=lambda: resource_status(
                free_gib=60.0,
                minimum_free_gib=49.0,
            ),
        )
    assert not full_grid._claim_path(root, one_job(plan)).exists()


def test_resource_guard_rejects_forged_safe_busy_gpu_and_foreign_memory() -> None:
    plan = tiny_plan()
    value = resource_status().as_dict()
    value["gpu"]["utilization_percent"] = 99.0
    with pytest.raises(full_grid.FullGridError, match="contradict"):
        full_grid._validate_resource_guard(value, plan=plan)

    value = resource_status().as_dict()
    value["gpu"]["memory_used_mib"] = 9000.0
    with pytest.raises(full_grid.FullGridError, match="contradict"):
        full_grid._validate_resource_guard(value, plan=plan)


def test_gpu_probe_rejects_malformed_rows_even_for_other_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gpu_uuid = "GPU-00000001-0000-0000-0000-000000000001"
    outputs = iter(
        (
            f"{gpu_uuid}, 0000:01:00.0, Test GPU, 0, 0\n",
            (
                "not-a-pid, "
                "GPU-00000002-0000-0000-0000-000000000002, python, 100\n"
            ),
        )
    )

    class Result:
        def __init__(self, stdout: str) -> None:
            self.stdout = stdout

    monkeypatch.setattr(
        full_grid.subprocess,
        "run",
        lambda *unused_args, **unused_kwargs: Result(next(outputs)),
    )
    observed = full_grid.probe_gpu(gpu_uuid)
    assert not observed.safe
    assert "failed closed" in observed.reason


def test_claim_owner_rejects_bool_pid_and_null_boot_start(tmp_path: Path) -> None:
    root, plan = initialize(tmp_path)
    job = one_job(plan)
    claim = claim_job(root, plan, job)
    try:
        value = copy.deepcopy(dict(claim.value))
        value["owner"]["pid"] = True
        with pytest.raises(full_grid.FullGridError, match="claim identity"):
            full_grid._validate_claim_value(value, plan=plan, job=job)
        value = copy.deepcopy(dict(claim.value))
        value["owner"]["boot_id"] = None
        with pytest.raises(full_grid.FullGridError, match="claim identity"):
            full_grid._validate_claim_value(value, plan=plan, job=job)
        value = copy.deepcopy(dict(claim.value))
        value["owner"]["start_ticks"] = None
        with pytest.raises(full_grid.FullGridError, match="claim identity"):
            full_grid._validate_claim_value(value, plan=plan, job=job)
    finally:
        full_grid.release_claim(claim)


def test_foreign_claim_never_age_steals_and_positive_timeout_rejected() -> None:
    value = {
        "owner": {
            "host": "other-host",
            "pid": 999999,
            "boot_id": "boot",
            "start_ticks": "start",
        }
    }
    assert full_grid._claim_is_live(value, foreign_claim_timeout_seconds=0.0)
    with pytest.raises(full_grid.FullGridError, match="disabled"):
        full_grid._claim_is_live(value, foreign_claim_timeout_seconds=1.0)
    with pytest.raises(full_grid.FullGridError, match="disabled"):
        full_grid._claim_is_live(
            value,
            foreign_claim_timeout_seconds=math.inf,
        )


def test_malformed_existing_claim_is_not_reclaimed(tmp_path: Path) -> None:
    root, plan = initialize(tmp_path)
    job = one_job(plan)
    path = full_grid._claim_path(root, job)
    full_grid._ensure_real_directory(root, Path("claims") / job.job_id[-2:])
    full_grid._write_json_exclusive(path, {"malformed": True})
    with pytest.raises(full_grid.FullGridError, match="schema"):
        claim_job(root, plan, job)
    assert path.exists()


@pytest.mark.parametrize("reason", ("stale", "completed-stale"))
def test_stale_claim_recovery_never_quarantines_live_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reason: str,
) -> None:
    root, plan = initialize(tmp_path)
    job = one_job(plan)
    path = full_grid._claim_path(root, job)
    full_grid._ensure_real_directory(
        root,
        Path("claims") / job.job_id[-2:],
    )
    stale = {
        "schema": full_grid.CLAIM_SCHEMA,
        "created_at": full_grid._utc_now(),
        "plan_sha256": plan["plan_sha256"],
        "job": job.identity(),
        "job_id": job.job_id,
        "nonce": "a" * 32,
        "owner": {
            "host": os.uname().nodename,
            "pid": 2_000_000_000,
            "boot_id": "dead-boot",
            "start_ticks": "dead-start",
        },
        "resource_guard": resource_status().as_dict(),
        "gpu_lease_receipt": None,
        "preflight_report_sha256": None,
    }
    replacement = copy.deepcopy(stale)
    replacement["nonce"] = "b" * 32
    replacement["owner"] = full_grid._process_identity()
    full_grid._write_json_exclusive(path, stale)
    stale_identity = (path.stat().st_dev, path.stat().st_ino)
    displaced = tmp_path / "validated-stale-claim"
    original_quarantine = full_grid._quarantine
    swapped = False

    def swap_before_quarantine(
        run_root: Path,
        selected: Path,
        *,
        category: str,
        reason: str,
        expected_identity: tuple[int, int] | None = None,
    ) -> Path:
        nonlocal swapped
        if selected == path and not swapped:
            os.replace(path, displaced)
            full_grid._write_json_exclusive(path, replacement)
            swapped = True
        assert expected_identity == stale_identity
        return original_quarantine(
            run_root,
            selected,
            category=category,
            reason=reason,
            expected_identity=expected_identity,
        )

    monkeypatch.setattr(full_grid, "_quarantine", swap_before_quarantine)
    with pytest.raises(full_grid.ClaimUnavailable, match="identity changed"):
        full_grid._quarantine_stale_claim_atomically(
            root,
            path,
            plan=plan,
            job=job,
            reason=reason,
        )
    assert swapped
    assert path.read_bytes() == full_grid._canonical_bytes(replacement) + b"\n"
    assert displaced.read_bytes() == full_grid._canonical_bytes(stale) + b"\n"
    quarantine = root / "quarantine" / "claims"
    assert not quarantine.exists() or not list(quarantine.iterdir())


def test_partial_recovery_never_quarantines_replacement_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, plan = initialize(tmp_path)
    job = one_job(plan)
    partial_root = full_grid._ensure_real_directory(root, "partials")
    path = partial_root / f"{job.job_id}.{'a' * 32}.partial"
    path.mkdir()
    stale_identity = (path.stat().st_dev, path.stat().st_ino)
    displaced = tmp_path / "validated-stale-partial"
    original_quarantine = full_grid._quarantine
    swapped = False

    def swap_before_quarantine(
        run_root: Path,
        selected: Path,
        *,
        category: str,
        reason: str,
        expected_identity: tuple[int, int] | None = None,
    ) -> Path:
        nonlocal swapped
        if selected == path and not swapped:
            os.replace(path, displaced)
            path.mkdir()
            swapped = True
        assert expected_identity == stale_identity
        return original_quarantine(
            run_root,
            selected,
            category=category,
            reason=reason,
            expected_identity=expected_identity,
        )

    monkeypatch.setattr(full_grid, "_quarantine", swap_before_quarantine)
    assert full_grid._quarantine_partials(root, job) == ()
    assert swapped and path.is_dir() and displaced.is_dir()
    quarantine = root / "quarantine" / "partials"
    assert not quarantine.exists() or not list(quarantine.iterdir())


def test_claim_holds_shared_gate_for_lifetime(tmp_path: Path) -> None:
    root, plan = initialize(tmp_path)
    claim = claim_job(root, plan, one_job(plan))
    try:
        with pytest.raises(full_grid.ClaimUnavailable, match="publication"):
            with full_grid.publication_fence(
                root,
                exclusive=True,
                blocking=False,
            ):
                raise AssertionError("unreachable")
    finally:
        full_grid.release_claim(claim)
    with full_grid.publication_fence(root, exclusive=True, blocking=False):
        pass


def test_gate_inode_replacement_is_detected(tmp_path: Path) -> None:
    root, plan = initialize(tmp_path)
    claim = claim_job(root, plan, one_job(plan))
    fence = root / full_grid.PUBLICATION_FENCE_FILENAME
    displaced = root / "fence.displaced"
    os.replace(fence, displaced)
    fence.write_bytes(b"replacement")
    fence.chmod(0o444)
    try:
        with pytest.raises(full_grid.FullGridError, match="ownership was lost"):
            full_grid._assert_publication_fence_descriptor(root, claim.fence_fd)
    finally:
        os.replace(displaced, fence)
        full_grid.release_claim(claim)


def test_commit_validates_exact_record_and_completion_schemas(tmp_path: Path) -> None:
    root, plan = initialize(tmp_path)
    job = one_job(plan)
    destination = complete_job(root, plan, job)
    payload = full_grid.validate_completion(root, plan, job)
    assert set(payload["record"]) == {
        "schema",
        "created_at",
        "plan_sha256",
        "job_id",
        "job",
        "score_blind",
        "claim_resource_guard",
        "commit_resource_guard",
        "gpu_lease_receipt",
        "preflight_report_sha256",
        "test_count",
        "test_rows_sha256",
        "metadata",
    }
    assert set(payload["completion"]) == {
        "schema",
        "created_at",
        "plan_sha256",
        "job_id",
        "job",
        "files",
    }
    assert payload["artifact_sha256"] == {
        name: hashlib.sha256((destination / name).read_bytes()).hexdigest()
        for name in full_grid.FINAL_FILENAMES
    }
    assert all(path.stat().st_mode & 0o222 == 0 for path in destination.iterdir())


def test_record_stage_is_presealed_and_final_directory_keeps_same_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, plan = initialize(tmp_path)
    job = one_job(plan)
    original_publish = full_grid._publish_sealed_directory_noreplace_at
    observed: dict[str, Any] = {}

    def inspect_publish(
        source_descriptor: int,
        source_name: str,
        sealed_descriptor: int,
        destination_descriptor: int,
        destination_name: str,
        **kwargs: Any,
    ) -> None:
        before = os.fstat(sealed_descriptor)
        observed["presealed"] = (
            stat.S_ISDIR(before.st_mode) and not before.st_mode & 0o222
        )
        observed["identity"] = (before.st_dev, before.st_ino)
        original_publish(
            source_descriptor,
            source_name,
            sealed_descriptor,
            destination_descriptor,
            destination_name,
            **kwargs,
        )
        after = os.fstat(sealed_descriptor)
        final_path = os.stat(
            destination_name,
            dir_fd=destination_descriptor,
            follow_symlinks=False,
        )
        observed["final_readonly"] = not after.st_mode & 0o222
        observed["same_inode"] = (
            (after.st_dev, after.st_ino)
            == (final_path.st_dev, final_path.st_ino)
            == observed["identity"]
        )

    monkeypatch.setattr(
        full_grid,
        "_publish_sealed_directory_noreplace_at",
        inspect_publish,
    )
    destination = complete_job(root, plan, job)
    assert observed == {
        "presealed": True,
        "identity": (destination.stat().st_dev, destination.stat().st_ino),
        "final_readonly": True,
        "same_inode": True,
    }


def _sealed_record_stage(path: Path) -> tuple[int, int]:
    path.mkdir()
    for name in full_grid.FINAL_FILENAMES:
        leaf = path / name
        leaf.write_bytes(name.encode("ascii"))
        leaf.chmod(0o444)
    path.chmod(0o555)
    observed = path.stat()
    return observed.st_dev, observed.st_ino


@pytest.mark.parametrize("permission_errno", (errno.EACCES, errno.EPERM))
def test_sealed_directory_publish_retries_only_permission_failure_and_reseals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    permission_errno: int,
) -> None:
    source = tmp_path / "stage"
    identity = _sealed_record_stage(source)
    destination = tmp_path / "final"
    parent_descriptor = full_grid._open_absolute_directory(tmp_path)
    stage_descriptor = full_grid._open_absolute_directory(source)
    original_rename = full_grid._atomic_rename_noreplace_at
    attempts = 0
    monkeypatch.setattr(full_grid.sys, "platform", "linux")

    def deny_once_then_rename(*args: Any) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError(permission_errno, "synthetic NFS sealed-dir denial")
        original_rename(*args)

    monkeypatch.setattr(
        full_grid,
        "_atomic_rename_noreplace_at",
        deny_once_then_rename,
    )
    try:
        full_grid._publish_sealed_directory_noreplace_at(
            parent_descriptor,
            source.name,
            stage_descriptor,
            parent_descriptor,
            destination.name,
            expected_names=full_grid.FINAL_FILENAMES,
        )
        held = os.fstat(stage_descriptor)
    finally:
        os.close(stage_descriptor)
        os.close(parent_descriptor)
    assert attempts == 2
    assert not source.exists()
    assert (destination.stat().st_dev, destination.stat().st_ino) == identity
    assert stat.S_IMODE(destination.stat().st_mode) == 0o555
    assert stat.S_IMODE(held.st_mode) == 0o555
    assert {path.name for path in destination.iterdir()} == set(
        full_grid.FINAL_FILENAMES
    )


def test_sealed_directory_publish_nonpermission_failure_does_not_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "stage"
    identity = _sealed_record_stage(source)
    parent_descriptor = full_grid._open_absolute_directory(tmp_path)
    stage_descriptor = full_grid._open_absolute_directory(source)
    attempts = 0
    monkeypatch.setattr(full_grid.sys, "platform", "linux")

    def fail_once(*unused: Any) -> None:
        nonlocal attempts
        attempts += 1
        raise OSError(errno.EIO, "synthetic I/O failure")

    monkeypatch.setattr(
        full_grid,
        "_atomic_rename_noreplace_at",
        fail_once,
    )
    try:
        with pytest.raises(OSError, match="synthetic I/O"):
            full_grid._publish_sealed_directory_noreplace_at(
                parent_descriptor,
                source.name,
                stage_descriptor,
                parent_descriptor,
                "final",
                expected_names=full_grid.FINAL_FILENAMES,
            )
        held = os.fstat(stage_descriptor)
    finally:
        os.close(stage_descriptor)
        os.close(parent_descriptor)
    assert attempts == 1
    assert (source.stat().st_dev, source.stat().st_ino) == identity
    assert stat.S_IMODE(source.stat().st_mode) == 0o555
    assert stat.S_IMODE(held.st_mode) == 0o555


def test_sealed_quarantine_uses_permission_fallback_and_keeps_exact_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, plan = initialize(tmp_path)
    job = one_job(plan)
    source = complete_job(root, plan, job)
    identity = (source.stat().st_dev, source.stat().st_ino)
    original_rename = full_grid._atomic_rename_noreplace_at
    attempts = 0
    monkeypatch.setattr(full_grid.sys, "platform", "linux")

    def deny_record_once(*args: Any) -> None:
        nonlocal attempts
        source_name = str(args[1])
        if source_name == source.name and attempts == 0:
            attempts += 1
            raise OSError(errno.EACCES, "synthetic NFS quarantine denial")
        attempts += 1
        original_rename(*args)

    monkeypatch.setattr(
        full_grid,
        "_atomic_rename_noreplace_at",
        deny_record_once,
    )
    quarantined = full_grid._quarantine(
        root,
        source,
        category="records",
        reason="corrupt",
        expected_identity=identity,
    )
    assert not source.exists()
    assert (quarantined.stat().st_dev, quarantined.stat().st_ino) == identity
    assert stat.S_IMODE(quarantined.stat().st_mode) == 0o555
    assert all(path.stat().st_mode & 0o222 == 0 for path in quarantined.iterdir())
    assert attempts >= 2


def test_quarantine_recovers_move_then_reseal_failure_for_exact_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, plan = initialize(tmp_path)
    job = one_job(plan)
    source = complete_job(root, plan, job)
    identity = (source.stat().st_dev, source.stat().st_ino)
    original_publish = full_grid._publish_sealed_directory_noreplace_at
    failed = False

    def move_then_fail(*args: Any, **kwargs: Any) -> None:
        nonlocal failed
        original_publish(*args, **kwargs)
        if str(args[1]) == source.name and not failed:
            failed = True
            raise OSError(errno.EIO, "synthetic quarantine reseal failure")

    monkeypatch.setattr(
        full_grid,
        "_publish_sealed_directory_noreplace_at",
        move_then_fail,
    )
    quarantined = full_grid._quarantine(
        root,
        source,
        category="records",
        reason="corrupt",
        expected_identity=identity,
    )
    assert failed
    assert not source.exists()
    assert (quarantined.stat().st_dev, quarantined.stat().st_ino) == identity
    assert stat.S_IMODE(quarantined.stat().st_mode) == 0o555


def test_writable_record_directory_is_never_accepted_as_completion(
    tmp_path: Path,
) -> None:
    root, plan = initialize(tmp_path)
    job = one_job(plan)
    destination = complete_job(root, plan, job)
    destination.chmod(0o755)
    with pytest.raises(
        full_grid.FullGridError,
        match="immutable record directory",
    ):
        full_grid.validate_completion(root, plan, job)


def test_completion_snapshot_rejects_coherent_leaf_package_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, plan = initialize(tmp_path)
    job = one_job(plan)
    canonical = complete_job(root, plan, job)
    replacement_parent = tmp_path / "replacement"
    replacement_parent.mkdir()
    replacement_root, replacement_plan = initialize(replacement_parent)
    assert replacement_plan == plan
    replacement = complete_job(replacement_root, replacement_plan, job)
    replacement_digests = {
        name: hashlib.sha256((replacement / name).read_bytes()).hexdigest()
        for name in full_grid.FINAL_FILENAMES
    }
    displaced = tmp_path / "held-package-a"
    displaced.mkdir()
    canonical_identity = (canonical.stat().st_dev, canonical.stat().st_ino)
    original_fstat = full_grid.os.fstat
    directory_fstats = 0
    swapped = False

    def swap_package_at_close_of_snapshot(
        descriptor: int,
    ) -> os.stat_result:
        nonlocal directory_fstats, swapped
        observed = original_fstat(descriptor)
        if (
            stat.S_ISDIR(observed.st_mode)
            and (observed.st_dev, observed.st_ino) == canonical_identity
        ):
            directory_fstats += 1
        # For the rejected sequential implementation this is its only
        # post-read directory check. The returned stat predates the swap, so
        # checking only dev/inode/mode would accept detached package A while
        # package B is live. The simultaneous-FD/package fingerprint contract
        # must still reject it.
        if directory_fstats == 3 and not swapped:
            canonical.chmod(0o755)
            replacement.chmod(0o755)
            for name in sorted(full_grid.FINAL_FILENAMES):
                os.replace(canonical / name, displaced / name)
                os.replace(replacement / name, canonical / name)
            canonical.chmod(0o555)
            swapped = True
        return observed

    with monkeypatch.context() as context:
        context.setattr(full_grid.os, "fstat", swap_package_at_close_of_snapshot)
        with pytest.raises(
            full_grid.FullGridError,
            match="package read|changed while it was validated",
        ):
            full_grid._validate_completion_bound(root, plan, job)

    assert swapped
    live = full_grid._validate_completion_bound(root, plan, job)
    assert live["artifact_sha256"] == replacement_digests
    assert {path.name for path in displaced.iterdir()} == set(
        full_grid.FINAL_FILENAMES
    )


def test_completion_snapshot_binds_full_directory_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, plan = initialize(tmp_path)
    job = one_job(plan)
    canonical = complete_job(root, plan, job)
    canonical_identity = (canonical.stat().st_dev, canonical.stat().st_ino)
    original_fstat = full_grid.os.fstat
    directory_fstats = 0
    mutated = False

    def mutate_directory_metadata(descriptor: int) -> os.stat_result:
        nonlocal directory_fstats, mutated
        observed = original_fstat(descriptor)
        if (
            stat.S_ISDIR(observed.st_mode)
            and (observed.st_dev, observed.st_ino) == canonical_identity
        ):
            directory_fstats += 1
        if directory_fstats == 3 and not mutated:
            canonical.chmod(0o755)
            canonical.chmod(0o555)
            mutated = True
        return observed

    with monkeypatch.context() as context:
        context.setattr(full_grid.os, "fstat", mutate_directory_metadata)
        with pytest.raises(
            full_grid.FullGridError,
            match="changed while it was validated",
        ):
            full_grid._validate_completion_bound(root, plan, job)
    assert mutated
    full_grid._validate_completion_bound(root, plan, job)


def _rewrite_prediction_archive_with_raw_zip_mutation(
    path: Path,
    mutation: str,
) -> None:
    with zipfile.ZipFile(path, "r") as archive:
        entries = [
            (info.filename, archive.read(info))
            for info in archive.infolist()
        ]
    if mutation == "duplicate":
        entries.append(entries[0])
    elif mutation == "traversal":
        entries[0] = ("../rows.npy", entries[0][1])
    elif mutation == "extra":
        entries.append(("extra.npy", entries[0][1]))
    elif mutation == "order":
        entries = list(reversed(entries))
    else:
        raise AssertionError(mutation)
    replacement = path.with_name(f".{path.name}.{mutation}.replacement")
    with zipfile.ZipFile(
        replacement,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for name, payload in entries:
            archive.writestr(name, payload)
    os.replace(replacement, path)
    path.chmod(0o444)


@pytest.mark.parametrize(
    "mutation",
    ("duplicate", "traversal", "extra", "order"),
)
def test_prediction_reader_rejects_nonexact_raw_zip_member_contract(
    tmp_path: Path,
    mutation: str,
) -> None:
    root, plan = initialize(tmp_path)
    job = one_job(plan)
    destination = complete_job(root, plan, job)
    destination.chmod(0o755)
    predictions = destination / "predictions.npz"
    _rewrite_prediction_archive_with_raw_zip_mutation(
        predictions,
        mutation,
    )
    completion_path = destination / "completion.json"
    completion = full_grid.strict_load(completion_path)
    completion["files"]["predictions.npz"] = hashlib.sha256(
        predictions.read_bytes()
    ).hexdigest()
    completion_path.chmod(0o600)
    completion_path.write_bytes(
        full_grid._canonical_bytes(completion) + b"\n"
    )
    completion_path.chmod(0o444)
    destination.chmod(0o555)
    with pytest.raises(
        (full_grid.FullGridError, RuntimeError),
        match="NPZ|member|archive",
    ):
        full_grid.validate_completion(root, plan, job)


def test_postrename_record_inode_replacement_is_rejected_without_quarantine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, plan = initialize(tmp_path)
    job = one_job(plan)
    claim = claim_job(root, plan, job)
    original_publish = full_grid._publish_sealed_directory_noreplace_at
    displaced_name = f"{full_grid._record_directory(root, job).name}.displaced"

    def replace_after_publish(
        source_descriptor: int,
        source_name: str,
        sealed_descriptor: int,
        destination_descriptor: int,
        destination_name: str,
        **kwargs: Any,
    ) -> None:
        original_publish(
            source_descriptor,
            source_name,
            sealed_descriptor,
            destination_descriptor,
            destination_name,
            **kwargs,
        )
        original_publish(
            destination_descriptor,
            destination_name,
            sealed_descriptor,
            destination_descriptor,
            displaced_name,
            **kwargs,
        )
        os.mkdir(destination_name, 0o700, dir_fd=destination_descriptor)

    monkeypatch.setattr(
        full_grid,
        "_publish_sealed_directory_noreplace_at",
        replace_after_publish,
    )
    try:
        with pytest.raises(
            full_grid.FullGridError,
            match="differs from its immutable staged inode",
        ):
            full_grid.commit_job_output(
                root,
                plan,
                claim,
                metadata=metadata(plan, job),
                test_rows=np.asarray([2, 3], dtype=np.int64),
                probabilities=np.asarray(
                    [[0.8, 0.2], [0.3, 0.7]],
                    dtype=np.float64,
                ),
            )
    finally:
        full_grid.release_claim(claim)
    replacement = full_grid._record_directory(root, job)
    assert replacement.is_dir()
    assert not list(replacement.iterdir())
    displaced = (
        full_grid._record_directory(root, job).parent / displaced_name
    )
    assert displaced.is_dir()
    audit = full_grid.audit_grid(root, plan)
    assert not audit["exact_cartesian_complete"]
    assert audit["extra"] >= 1


def test_corrupt_record_recovery_never_quarantines_replacement_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, plan = initialize(tmp_path)
    job = one_job(plan)
    output = complete_job(root, plan, job)
    corrupt_identity = (output.stat().st_dev, output.stat().st_ino)
    replacement_parent = tmp_path / "replacement-record"
    replacement_parent.mkdir()
    replacement_root, replacement_plan = initialize(replacement_parent)
    replacement = complete_job(replacement_root, replacement_plan, job)
    replacement_identity = (
        replacement.stat().st_dev,
        replacement.stat().st_ino,
    )
    displaced = tmp_path / "validated-corrupt-record"
    original_validate = full_grid._validate_completion_bound
    swapped = False

    def swap_then_reject(
        run_root: Path,
        selected_plan: Mapping[str, Any],
        selected_job: full_grid.Job,
        **kwargs: Any,
    ) -> dict[str, Any]:
        nonlocal swapped
        if run_root == root and selected_job == job and not swapped:
            output.chmod(0o755)
            replacement.chmod(0o755)
            os.replace(output, displaced)
            os.replace(replacement, output)
            displaced.chmod(0o555)
            output.chmod(0o555)
            swapped = True
            raise full_grid.FullGridError("synthetic validated corrupt record")
        return original_validate(
            run_root,
            selected_plan,
            selected_job,
            **kwargs,
        )

    with monkeypatch.context() as context:
        context.setattr(
            full_grid,
            "_validate_completion_bound",
            swap_then_reject,
        )
        with pytest.raises(
            full_grid.ClaimUnavailable,
            match="recovery race",
        ):
            full_grid.acquire_claim(
                root,
                plan,
                job,
                before_claim=resource_status,
            )
    assert swapped
    assert (displaced.stat().st_dev, displaced.stat().st_ino) == corrupt_identity
    assert (output.stat().st_dev, output.stat().st_ino) == replacement_identity
    full_grid._validate_completion_bound(root, plan, job)
    quarantine = root / "quarantine" / "records"
    assert not quarantine.exists() or not list(quarantine.iterdir())


def test_record_publisher_failure_after_move_invalidates_canonical_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, plan = initialize(tmp_path)
    job = one_job(plan)
    claim = claim_job(root, plan, job)
    original_publish = full_grid._publish_sealed_directory_noreplace_at

    def publish_then_fail(
        source_descriptor: int,
        source_name: str,
        sealed_descriptor: int,
        destination_descriptor: int,
        destination_name: str,
        **kwargs: Any,
    ) -> None:
        original_publish(
            source_descriptor,
            source_name,
            sealed_descriptor,
            destination_descriptor,
            destination_name,
            **kwargs,
        )
        raise full_grid.FullGridError(
            "synthetic failure after record directory move"
        )

    monkeypatch.setattr(
        full_grid,
        "_publish_sealed_directory_noreplace_at",
        publish_then_fail,
    )
    try:
        with pytest.raises(full_grid.FullGridError, match="after record"):
            full_grid.commit_job_output(
                root,
                plan,
                claim,
                metadata=metadata(plan, job),
                test_rows=np.asarray([2, 3], dtype=np.int64),
                probabilities=np.asarray(
                    [[0.8, 0.2], [0.3, 0.7]],
                    dtype=np.float64,
                ),
            )
    finally:
        full_grid.release_claim(claim)
    assert not full_grid._record_directory(root, job).exists()
    quarantined = list(
        (root / "quarantine" / "records").glob(
            "*.lease-postcondition.*"
        )
    )
    assert len(quarantined) == 1 and quarantined[0].is_dir()


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("parameter_count", True),
        ("selection_epochs_run", False),
    ),
)
def test_fit_counters_reject_bools(
    field: str,
    value: bool,
) -> None:
    plan = tiny_plan()
    job = one_job(plan)
    observed = metadata(plan, job)
    observed["fit"][field] = value
    with pytest.raises(full_grid.FullGridError, match="fit counters"):
        full_grid._validate_job_metadata(plan, job, observed)


def test_timing_values_reject_bools() -> None:
    plan = tiny_plan()
    job = one_job(plan)
    observed = metadata(plan, job)
    observed["timing_seconds"]["selection_fit"] = True
    with pytest.raises(full_grid.FullGridError, match="timing"):
        full_grid._validate_job_metadata(plan, job, observed)


def test_formal_tcformer_metadata_requires_exact_pinned_provenance() -> None:
    plan = tiny_plan()
    plan["publication_mode"] = full_grid.FORMAL_PUBLICATION_MODE
    plan["tcformer_source_identity"] = {"synthetic": "pinned"}
    gpu_uuid = "GPU-00000001-0000-0000-0000-000000000001"
    plan["execution_config"]["physical_gpu_uuid_roster"] = [gpu_uuid]
    job = one_job(plan, "tcformer")
    observed = metadata(plan, job)
    observed["runtime"]["physical_gpu_uuid"] = gpu_uuid
    observed["runtime"]["cuda_visible_devices"] = gpu_uuid
    observed["fit"]["architecture"] = {
        "requested_name": "tcformer",
        "class": "_eeg_tcformer.models.tcformer.TCFormerModule",
        "wrapped_class": "_eeg_tcformer.models.tcformer.TCFormerModule",
        "scalar_attributes": {},
    }
    with pytest.raises(full_grid.FullGridError, match="provenance"):
        full_grid._validate_job_metadata(plan, job, observed)
    observed["fit"]["architecture"]["third_party_provenance"] = {
        "synthetic": "pinned"
    }
    full_grid._validate_job_metadata(plan, job, observed)


def test_formal_commit_checks_disk_before_creating_prediction_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, plan = initialize(tmp_path)
    job = one_job(plan)
    claim = claim_job(root, plan, job)
    formal = copy.deepcopy(plan)
    formal["publication_mode"] = full_grid.FORMAL_PUBLICATION_MODE
    gpu_uuid = "GPU-00000001-0000-0000-0000-000000000001"
    formal["execution_config"]["physical_gpu_uuid_roster"] = [gpu_uuid]
    observed_metadata = metadata(formal, job)
    observed_metadata["runtime"]["physical_gpu_uuid"] = gpu_uuid
    observed_metadata["runtime"]["cuda_visible_devices"] = gpu_uuid
    unsafe_disk = full_grid.combine_resource_status(
        full_grid.GPUStatus(
            safe=True,
            gpu=gpu_uuid,
            utilization_percent=0.0,
            memory_used_mib=0.0,
            own_compute_memory_mib=0.0,
            foreign_processes=(),
            reason="idle",
            gpu_uuid=gpu_uuid,
            pci_bus_id="0000:01:00.0",
            name="GPU",
        ),
        full_grid.DiskStatus(
            safe=False,
            path=str(root),
            free_bytes=1,
            total_bytes=2,
            free_gib=0.0,
            minimum_free_gib=50.0,
            reason="below floor",
        ),
    )
    monkeypatch.setattr(full_grid, "validate_plan_semantics", lambda *unused, **kw: None)
    monkeypatch.setattr(
        full_grid,
        "load_preflight_attestation",
        lambda *unused, **unused_keywords: {"report_sha256": None},
    )
    monkeypatch.setattr(
        project_gpu_leases,
        "assert_gpu_lease",
        lambda unused: None,
    )
    try:
        with pytest.raises(full_grid.DiskUnavailable):
            full_grid.commit_job_output(
                root,
                formal,
                claim,
                metadata=observed_metadata,
                test_rows=np.asarray([2, 3], dtype=np.int64),
                probabilities=np.asarray(
                    [[0.8, 0.2], [0.3, 0.7]],
                    dtype=np.float64,
                ),
                cache_root=tmp_path / "cache",
                before_publish=lambda: unsafe_disk,
                gpu_lease=object(),  # type: ignore[arg-type]
            )
        assert not (root / "partials").exists()
    finally:
        full_grid.release_claim(claim)


def test_nested_outcome_alias_is_rejected(tmp_path: Path) -> None:
    root, plan = initialize(tmp_path)
    job = one_job(plan)
    claim = claim_job(root, plan, job)
    bad_metadata = metadata(plan, job)
    bad_metadata["fit"]["architecture"]["scalar_attributes"][
        "producer_accuracy"
    ] = 0.99
    try:
        with pytest.raises(full_grid.FullGridError, match="outcome aliases"):
            full_grid.commit_job_output(
                root,
                plan,
                claim,
                metadata=bad_metadata,
                test_rows=np.asarray([2, 3], dtype=np.int64),
                probabilities=np.asarray(
                    [[0.8, 0.2], [0.3, 0.7]],
                    dtype=np.float64,
                ),
            )
    finally:
        full_grid.release_claim(claim)


def test_documented_filterbank_f1_architecture_parameter_is_not_an_outcome_alias() -> None:
    base = {
        "metadata": {
            "fit": {
                "architecture": {
                    "scalar_attributes": {"F1": 8},
                },
            },
        },
    }
    assert full_grid._recursive_outcome_aliases(base) == []
    bad = copy.deepcopy(base)
    bad["metadata"]["fit"]["architecture"]["scalar_attributes"]["F1"] = 9
    assert full_grid._recursive_outcome_aliases(bad)


def test_lost_claim_after_rename_quarantines_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, plan = initialize(tmp_path)
    job = one_job(plan)
    claim = claim_job(root, plan, job)
    original = full_grid._recheck_owned_claim_path
    calls = 0

    def fail_second(selected: full_grid.Claim, descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise full_grid.FullGridError("synthetic post-rename loss")
        original(selected, descriptor)

    monkeypatch.setattr(full_grid, "_recheck_owned_claim_path", fail_second)
    try:
        with pytest.raises(full_grid.FullGridError, match="post-rename"):
            full_grid.commit_job_output(
                root,
                plan,
                claim,
                metadata=metadata(plan, job),
                test_rows=np.asarray([2, 3], dtype=np.int64),
                probabilities=np.asarray(
                    [[0.8, 0.2], [0.3, 0.7]],
                    dtype=np.float64,
                ),
            )
    finally:
        full_grid.release_claim(claim)
    assert not full_grid._record_directory(root, job).exists()
    quarantined = list((root / "quarantine" / "records").iterdir())
    assert len(quarantined) == 1


def test_release_uses_durable_tombstone_and_recovery(tmp_path: Path) -> None:
    root, plan = initialize(tmp_path)
    job = one_job(plan)
    tombstone_root = full_grid._ensure_real_directory(
        root,
        full_grid.CLAIM_TOMBSTONE_DIRECTORY,
    )
    owner = {
        "host": os.uname().nodename,
        "pid": 2_000_000_000,
        "boot_id": "dead-boot",
        "start_ticks": "dead-start",
    }
    nonce = "a" * 32
    value = {
        "schema": full_grid.CLAIM_SCHEMA,
        "created_at": full_grid._utc_now(),
        "plan_sha256": plan["plan_sha256"],
        "job": job.identity(),
        "job_id": job.job_id,
        "nonce": nonce,
        "owner": owner,
        "resource_guard": resource_status().as_dict(),
        "gpu_lease_receipt": None,
        "preflight_report_sha256": None,
    }
    path = tombstone_root / f"{job.job_id}.{nonce}.released"
    full_grid._write_json_exclusive(path, value)
    recovered = full_grid.recover_claim_tombstones(root, plan)
    assert recovered == (path,)
    assert not path.exists()


def test_tombstone_recovery_unlink_is_anchored_across_run_root_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, plan = initialize(tmp_path)
    displaced = tmp_path / "run-displaced"
    job = one_job(plan)
    tombstone_root = full_grid._ensure_real_directory(
        root,
        full_grid.CLAIM_TOMBSTONE_DIRECTORY,
    )
    nonce = "a" * 32
    value = {
        "schema": full_grid.CLAIM_SCHEMA,
        "created_at": full_grid._utc_now(),
        "plan_sha256": plan["plan_sha256"],
        "job": job.identity(),
        "job_id": job.job_id,
        "nonce": nonce,
        "owner": {
            "host": os.uname().nodename,
            "pid": 2_000_000_000,
            "boot_id": "dead-boot",
            "start_ticks": "dead-start",
        },
        "resource_guard": resource_status().as_dict(),
        "gpu_lease_receipt": None,
        "preflight_report_sha256": None,
    }
    name = f"{job.job_id}.{nonce}.released"
    full_grid._write_json_exclusive(tombstone_root / name, value)
    original_unlink = full_grid.os.unlink
    swapped = False

    def swap_before_anchored_unlink(
        selected: Any,
        *args: Any,
        dir_fd: int | None = None,
        **kwargs: Any,
    ) -> None:
        nonlocal swapped
        if selected == name and dir_fd is not None and not swapped:
            root.rename(displaced)
            root.mkdir()
            replacement_root = root / full_grid.CLAIM_TOMBSTONE_DIRECTORY
            replacement_root.mkdir()
            (replacement_root / name).write_bytes(b"replacement-must-survive")
            swapped = True
        original_unlink(
            selected,
            *args,
            dir_fd=dir_fd,
            **kwargs,
        )

    monkeypatch.setattr(full_grid.os, "unlink", swap_before_anchored_unlink)
    try:
        with pytest.raises(full_grid.FullGridError, match="directory path changed"):
            full_grid.recover_claim_tombstones(root, plan)
        assert (
            root / full_grid.CLAIM_TOMBSTONE_DIRECTORY / name
        ).read_bytes() == b"replacement-must-survive"
        assert not (
            displaced / full_grid.CLAIM_TOMBSTONE_DIRECTORY / name
        ).exists()
    finally:
        if swapped:
            original_unlink(
                root / full_grid.CLAIM_TOMBSTONE_DIRECTORY / name
            )
            (root / full_grid.CLAIM_TOMBSTONE_DIRECTORY).rmdir()
            root.rmdir()
            displaced.rename(root)


def test_full_completion_audit_rejects_extra_special_and_hardlinked_nodes(
    tmp_path: Path,
) -> None:
    root, plan = initialize(tmp_path)
    for model in plan["architectures"]:
        complete_job(root, plan, one_job(plan, str(model)))
    audit = full_grid.audit_grid(root, plan)
    assert audit["exact_cartesian_complete"]
    rogue = root / "records" / "rogue"
    os.mkfifo(rogue)
    audit = full_grid.audit_grid(root, plan)
    assert not audit["exact_cartesian_complete"]
    assert audit["unsafe_paths"] == 1


def test_audit_counts_unknown_quarantine_category_as_unresolved_forensic(
    tmp_path: Path,
) -> None:
    root, plan = initialize(tmp_path)
    for model in plan["architectures"]:
        complete_job(root, plan, one_job(plan, str(model)))
    unknown = full_grid._ensure_real_directory(
        root,
        "quarantine/unknown-category",
    )
    (unknown / "evidence").write_bytes(b"must-not-be-resolved")
    audit = full_grid.audit_grid(root, plan)
    assert not audit["exact_cartesian_complete"]
    assert audit["quarantine_artifacts"] == 1
    assert audit["resolved_forensic_artifacts"] == 0
    assert audit["invalid_forensic_artifacts"] == 1


def test_audit_rejects_broken_symlink_auxiliary_root(tmp_path: Path) -> None:
    root, plan = initialize(tmp_path)
    for model in plan["architectures"]:
        complete_job(root, plan, one_job(plan, str(model)))
    (root / "worker_logs").symlink_to(
        tmp_path / "absent-worker-logs",
        target_is_directory=True,
    )
    audit = full_grid.audit_grid(root, plan)
    assert not audit["exact_cartesian_complete"]
    assert audit["unsafe_paths"] == 1


def test_quarantine_category_reason_alias_and_no_replace_are_fail_closed(
    tmp_path: Path,
) -> None:
    root, _ = initialize(tmp_path)
    source = root / "source"
    source.write_bytes(b"x")
    with pytest.raises(full_grid.FullGridError, match="category/reason"):
        full_grid._quarantine(
            root,
            source,
            category="unknown",
            reason="unknown",
        )
    assert source.exists()

    original = root / "original"
    alias = root / "alias"
    original.write_bytes(b"x")
    os.link(original, alias)
    with pytest.raises(full_grid.FullGridError, match="aliased"):
        full_grid._quarantine(
            root,
            alias,
            category="claims",
            reason="stale",
        )
    assert alias.exists()

    first = root / "first"
    second = root / "second"
    destination = root / "destination"
    first.mkdir()
    second.mkdir()
    full_grid._atomic_rename_noreplace(first, destination)
    with pytest.raises(FileExistsError):
        full_grid._atomic_rename_noreplace(second, destination)
    assert second.is_dir()


def test_audit_rejects_quarantine_filename_outside_exact_schema(
    tmp_path: Path,
) -> None:
    root, plan = initialize(tmp_path)
    for model in plan["architectures"]:
        complete_job(root, plan, one_job(plan, str(model)))
    quarantine = full_grid._ensure_real_directory(root, "quarantine/claims")
    rogue = quarantine / "rogue"
    rogue.write_text("{}", encoding="utf-8")
    rogue.chmod(0o444)
    audit = full_grid.audit_grid(root, plan)
    assert not audit["exact_cartesian_complete"]
    assert audit["unsafe_paths"] == 1
    assert audit["quarantine_artifacts"] == 1
    assert audit["resolved_forensic_artifacts"] == 0
    assert audit["invalid_forensic_artifacts"] == 1


def test_audit_accepts_only_a_plan_bound_exact_quarantine_receipt(
    tmp_path: Path,
) -> None:
    root, plan = initialize(tmp_path)
    job = one_job(plan)
    owner = {
        "host": os.uname().nodename,
        "pid": 2_000_000_000,
        "boot_id": "dead-boot",
        "start_ticks": "dead-start",
    }
    value = {
        "schema": full_grid.CLAIM_SCHEMA,
        "created_at": full_grid._utc_now(),
        "plan_sha256": plan["plan_sha256"],
        "job": job.identity(),
        "job_id": job.job_id,
        "nonce": "b" * 32,
        "owner": owner,
        "resource_guard": resource_status().as_dict(),
        "gpu_lease_receipt": None,
        "preflight_report_sha256": None,
    }
    path = full_grid._claim_path(root, job)
    full_grid._ensure_real_directory(root, Path("claims") / job.job_id[-2:])
    full_grid._write_json_exclusive(path, value)
    full_grid._quarantine(
        root,
        path,
        category="claims",
        reason="stale",
    )
    for model in plan["architectures"]:
        complete_job(root, plan, one_job(plan, str(model)))
    audit = full_grid.audit_grid(root, plan)
    assert audit["exact_cartesian_complete"]
    assert audit["quarantine_artifacts"] == 1
    assert audit["resolved_forensic_artifacts"] == 1
    assert audit["invalid_forensic_artifacts"] == 0
    assert len(audit["details"]["forensic_ledger"]) == 1
    forensic = audit["details"]["forensic_ledger"][0]
    assert forensic["category"] == "claims"
    assert forensic["reason"] == "stale"
    assert forensic["identity"] == job.job_id
    assert full_grid.HEX_64_RE.fullmatch(
        forensic["artifact_sha256"]
    )
    assert audit["forensic_ledger_sha256"] == hashlib.sha256(
        full_grid._canonical_bytes(audit["details"]["forensic_ledger"])
    ).hexdigest()
    assert len(audit["details"]["quarantine_artifact_paths"]) == 1


def test_forensic_ledger_rejects_artifact_name_swap_after_initial_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, plan = initialize(tmp_path)
    job = one_job(plan)
    owner = {
        "host": os.uname().nodename,
        "pid": 2_000_000_000,
        "boot_id": "dead-boot",
        "start_ticks": "dead-start",
    }
    value = {
        "schema": full_grid.CLAIM_SCHEMA,
        "created_at": full_grid._utc_now(),
        "plan_sha256": plan["plan_sha256"],
        "job": job.identity(),
        "job_id": job.job_id,
        "nonce": "c" * 32,
        "owner": owner,
        "resource_guard": resource_status().as_dict(),
        "gpu_lease_receipt": None,
        "preflight_report_sha256": None,
    }
    claim_path = full_grid._claim_path(root, job)
    full_grid._ensure_real_directory(
        root,
        Path("claims") / job.job_id[-2:],
    )
    full_grid._write_json_exclusive(claim_path, value)
    quarantined = full_grid._quarantine(
        root,
        claim_path,
        category="claims",
        reason="stale",
    )
    for model in plan["architectures"]:
        complete_job(root, plan, one_job(plan, str(model)))

    payload = quarantined.read_bytes()
    displaced = quarantined.with_name(f"{quarantined.name}.displaced")
    original_digest = full_grid._forensic_tree_digest_at
    swapped = False

    def swap_name_after_digest(*args: Any, **kwargs: Any):
        nonlocal swapped
        result = original_digest(*args, **kwargs)
        if not swapped and str(args[1]) == quarantined.name:
            os.replace(quarantined, displaced)
            quarantined.write_bytes(payload)
            quarantined.chmod(0o444)
            swapped = True
        return result

    monkeypatch.setattr(
        full_grid,
        "_forensic_tree_digest_at",
        swap_name_after_digest,
    )
    audit = full_grid.audit_grid(root, plan)
    assert swapped
    assert not audit["exact_cartesian_complete"]
    assert audit["resolved_forensic_artifacts"] == 0
    assert audit["invalid_forensic_artifacts"] >= 1
    assert any(
        "changed during audit" in path
        for path in audit["details"]["unsafe_paths"]
    )


def test_failure_receipt_exact_schema(tmp_path: Path) -> None:
    root, plan = initialize(tmp_path)
    job = one_job(plan)
    claim = claim_job(root, plan, job)
    try:
        path = full_grid._record_failure(
            root,
            plan,
            claim,
            RuntimeError("synthetic"),
        )
        receipt = full_grid.strict_load(path)
        full_grid._validate_failure_receipt(receipt, plan=plan, job=job)
        receipt["hidden_score"] = 1.0
        with pytest.raises(full_grid.FullGridError, match="schema"):
            full_grid._validate_failure_receipt(receipt, plan=plan, job=job)
    finally:
        full_grid.release_claim(claim)


def test_gpu_resolution_enforces_distinct_physical_uuid_and_cap() -> None:
    identities = tuple(
        full_grid.GPUIdentity(
            index=str(index),
            uuid=f"GPU-0000000{index}-0000-0000-0000-00000000000{index}",
            pci_bus_id=f"0000:0{index}:00.0",
            name="GPU",
        )
        for index in range(4)
    )
    resolved = full_grid._resolve_gpus("0,1,2", identities=identities)
    assert [value.uuid for value in resolved] == [
        value.uuid for value in identities[:3]
    ]
    with pytest.raises(ValueError, match="1-3"):
        full_grid._resolve_gpus("0,1,2,3", identities=identities)
    with pytest.raises(full_grid.GPUUnavailable, match="outside formal"):
        full_grid._resolve_gpus("1,2,3", identities=identities)
    with pytest.raises(full_grid.GPUUnavailable, match="outside formal"):
        full_grid._resolve_gpus(identities[3].uuid, identities=identities)
    with pytest.raises(full_grid.GPUUnavailable, match="aliases"):
        full_grid._resolve_gpus(
            f"0,{identities[0].uuid}",
            identities=identities,
        )
    environment = {
        "nvidia_gpu_inventory": [
            value.as_dict() | {"driver_version": "1"} for value in identities
        ]
    }
    assert full_grid._formal_gpu_uuid_roster(environment) == [
        value.uuid for value in identities[:3]
    ]


def test_project_gpu_leases_are_cross_run_uuid_exclusive_and_globally_capped(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    runs = [tmp_path / f"run-{index}" for index in range(4)]
    for run in runs:
        run.mkdir()
    uuids = [
        f"GPU-0000000{index}-0000-0000-0000-00000000000{index}"
        for index in range(1, 5)
    ]
    leases: list[project_gpu_leases.GPULease] = []
    try:
        first = project_gpu_leases.acquire_gpu_lease(
            project_root=project,
            run_root=runs[0],
            plan_sha256="a" * 64,
            gpu_uuid=uuids[0],
            track_scope="common_recipe_43_only",
        )
        leases.append(first)
        assert set(first.value) == {
            "schema",
            "created_at",
            "project_root_sha256",
            "track_scope",
            "run_root_sha256",
            "plan_sha256",
            "gpu_uuid",
            "nonce",
            "owner",
        }
        assert isinstance(first.value["owner"]["pid"], int)
        assert not isinstance(first.value["owner"]["pid"], bool)
        assert isinstance(first.value["owner"]["start_ticks"], int)
        with pytest.raises(
            project_gpu_leases.GPUWorkerUnavailable,
            match="already has a lease",
        ):
            project_gpu_leases.acquire_gpu_lease(
                project_root=project,
                run_root=runs[1],
                plan_sha256="b" * 64,
                gpu_uuid=uuids[0],
                track_scope="another_track",
            )
        for index in (1, 2):
            leases.append(
                project_gpu_leases.acquire_gpu_lease(
                    project_root=project,
                    run_root=runs[index],
                    plan_sha256=f"{index + 1:x}" * 64,
                    gpu_uuid=uuids[index],
                    track_scope=f"track_{index}",
                )
            )
        with pytest.raises(
            project_gpu_leases.GPUWorkerUnavailable,
            match="cap is 3",
        ):
            project_gpu_leases.acquire_gpu_lease(
                project_root=project,
                run_root=runs[3],
                plan_sha256="f" * 64,
                gpu_uuid=uuids[3],
                track_scope="fourth_track",
            )
    finally:
        for lease in reversed(leases):
            project_gpu_leases.release_gpu_lease(lease)


def _formal_job_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    Path,
    dict[str, Any],
    full_grid.Job,
    project_gpu_leases.GPULease,
    Any,
]:
    root, plan = initialize(tmp_path)
    job = one_job(plan)
    gpu_uuid = "GPU-00000001-0000-0000-0000-000000000001"
    formal = copy.deepcopy(plan)
    formal["publication_mode"] = full_grid.FORMAL_PUBLICATION_MODE
    formal["execution_config"]["physical_gpu_uuid_roster"] = [gpu_uuid]
    formal["environment_identity"] = {
        "nvidia_driver_versions": None,
        "nvidia_gpu_inventory": [
            {
                "index": "0",
                "uuid": gpu_uuid,
                "pci_bus_id": "0000:01:00.0",
                "name": "Test GPU",
                "driver_version": "1",
            }
        ],
    }
    monkeypatch.setattr(
        full_grid,
        "validate_plan_semantics",
        lambda *unused, **unused_keywords: None,
    )
    monkeypatch.setattr(full_grid, "_verified_project_root", lambda: tmp_path)
    monkeypatch.setattr(
        full_grid,
        "load_preflight_attestation",
        lambda *unused, **unused_keywords: {"report_sha256": "f" * 64},
    )
    monkeypatch.setattr(
        full_grid,
        "_verify_job_runtime_identity",
        lambda *unused, **unused_keywords: None,
    )
    lease = project_gpu_leases.acquire_gpu_lease(
        project_root=tmp_path,
        run_root=root,
        plan_sha256=formal["plan_sha256"],
        gpu_uuid=gpu_uuid,
        track_scope=full_grid.TRACK_SCOPE,
    )

    def resources() -> full_grid.ResourceStatus:
        return full_grid.combine_resource_status(
            full_grid.GPUStatus(
                safe=True,
                gpu=gpu_uuid,
                utilization_percent=0.0,
                memory_used_mib=0.0,
                own_compute_memory_mib=0.0,
                foreign_processes=(),
                reason="idle",
                gpu_uuid=gpu_uuid,
                pci_bus_id="0000:01:00.0",
                name="Test GPU",
            ),
            full_grid.DiskStatus(
                safe=True,
                path=str(root),
                free_bytes=60 * 1024**3,
                total_bytes=100 * 1024**3,
                free_gib=60.0,
                minimum_free_gib=50.0,
                reason="above floor",
            ),
        )

    return root, formal, job, lease, resources


def test_claim_guard_exit_failure_quarantines_authoritative_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, plan, job, lease, resources = _formal_job_setup(
        tmp_path,
        monkeypatch,
    )
    original_guard = project_gpu_leases.guard_gpu_lease

    @contextmanager
    def fail_after_yield(selected: project_gpu_leases.GPULease):
        with original_guard(selected) as receipt:
            yield receipt
        raise project_gpu_leases.ProjectGPULeaseError(
            "synthetic post-yield claim lease loss"
        )

    monkeypatch.setattr(
        project_gpu_leases,
        "guard_gpu_lease",
        fail_after_yield,
    )
    try:
        with pytest.raises(full_grid.FullGridError, match="authority was lost"):
            full_grid.acquire_claim(
                root,
                plan,
                job,
                before_claim=resources,
                gpu_lease=lease,
            )
        assert not full_grid._claim_path(root, job).exists()
        quarantined = list(
            (root / "quarantine" / "claims").glob(
                f"{job.job_id}.json.lease-postcondition.*"
            )
        )
        assert len(quarantined) == 1
        assert quarantined[0].stat().st_mode & 0o222 == 0
    finally:
        project_gpu_leases.release_gpu_lease(lease)


@pytest.mark.parametrize(
    "boundary",
    ("write_return", "strict_load", "validate", "stat", "fence"),
)
def test_claim_construction_failure_quarantines_exact_published_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    root, plan = initialize(tmp_path)
    job = one_job(plan)
    claim_path = full_grid._claim_path(root, job)
    fired = False

    def inject() -> None:
        nonlocal fired
        fired = True
        raise full_grid.FullGridError(f"synthetic {boundary} boundary failure")

    if boundary == "write_return":
        original_write = full_grid._write_json_exclusive

        def fail_after_write(
            path: Path,
            value: Mapping[str, Any],
            *,
            on_publish: Any = None,
        ) -> os.stat_result:
            result = original_write(
                path,
                value,
                on_publish=on_publish,
            )
            if path == claim_path:
                inject()
            return result

        monkeypatch.setattr(
            full_grid,
            "_write_json_exclusive",
            fail_after_write,
        )
    elif boundary == "strict_load":
        original_load = full_grid.strict_load

        def fail_after_load(path: Path) -> dict[str, Any]:
            result = original_load(path)
            if path == claim_path:
                inject()
            return result

        monkeypatch.setattr(full_grid, "strict_load", fail_after_load)
    elif boundary == "validate":
        original_validate = full_grid._validate_claim_value

        def fail_after_validate(*args: Any, **kwargs: Any) -> None:
            original_validate(*args, **kwargs)
            if claim_path.exists():
                inject()

        monkeypatch.setattr(
            full_grid,
            "_validate_claim_value",
            fail_after_validate,
        )
    elif boundary == "stat":
        original_require = full_grid._require_unique_regular

        def fail_after_stat(
            path: Path,
            *,
            read_only: bool = False,
        ) -> os.stat_result:
            result = original_require(path, read_only=read_only)
            if path == claim_path:
                inject()
            return result

        monkeypatch.setattr(
            full_grid,
            "_require_unique_regular",
            fail_after_stat,
        )
    else:
        original_fence = full_grid._assert_publication_fence_descriptor

        def fail_after_fence(path: Path, descriptor: int) -> None:
            original_fence(path, descriptor)
            if claim_path.exists():
                inject()

        monkeypatch.setattr(
            full_grid,
            "_assert_publication_fence_descriptor",
            fail_after_fence,
        )

    with pytest.raises(full_grid.FullGridError, match=f"synthetic {boundary}"):
        full_grid.acquire_claim(
            root,
            plan,
            job,
            before_claim=resource_status,
        )
    assert fired
    assert not claim_path.exists()
    quarantined = list(
        (root / "quarantine" / "claims").glob(
            f"{job.job_id}.json.lease-postcondition.*"
        )
    )
    assert len(quarantined) == 1
    assert quarantined[0].stat().st_mode & 0o222 == 0


def test_claim_cleanup_never_quarantines_a_replacement_race_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, plan = initialize(tmp_path)
    job = one_job(plan)
    claim_path = full_grid._claim_path(root, job)
    displaced = tmp_path / "attempted-claim.displaced"
    replacement_payload = b'{"replacement":"race-winner"}\n'
    original_load = full_grid.strict_load

    def replace_after_read(path: Path) -> dict[str, Any]:
        result = original_load(path)
        if path == claim_path:
            os.replace(path, displaced)
            path.write_bytes(replacement_payload)
            path.chmod(0o444)
            raise full_grid.FullGridError(
                "synthetic replacement after claim read"
            )
        return result

    monkeypatch.setattr(full_grid, "strict_load", replace_after_read)
    with pytest.raises(full_grid.FullGridError, match="synthetic replacement"):
        full_grid.acquire_claim(
            root,
            plan,
            job,
            before_claim=resource_status,
        )
    assert claim_path.read_bytes() == replacement_payload
    assert displaced.is_file()
    quarantine = root / "quarantine" / "claims"
    assert not quarantine.exists() or not list(quarantine.iterdir())


def test_commit_guard_exit_failure_quarantines_authoritative_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, plan, job, lease, resources = _formal_job_setup(
        tmp_path,
        monkeypatch,
    )
    claim = full_grid.acquire_claim(
        root,
        plan,
        job,
        before_claim=resources,
        gpu_lease=lease,
    )
    original_guard = project_gpu_leases.guard_gpu_lease

    @contextmanager
    def fail_after_yield(selected: project_gpu_leases.GPULease):
        with original_guard(selected) as receipt:
            yield receipt
        raise project_gpu_leases.ProjectGPULeaseError(
            "synthetic post-yield commit lease loss"
        )

    monkeypatch.setattr(
        project_gpu_leases,
        "guard_gpu_lease",
        fail_after_yield,
    )
    observed_metadata = metadata(plan, job)
    observed_metadata["runtime"]["physical_gpu_uuid"] = str(
        lease.value["gpu_uuid"]
    )
    observed_metadata["runtime"]["cuda_visible_devices"] = str(
        lease.value["gpu_uuid"]
    )
    try:
        with pytest.raises(full_grid.FullGridError, match="authority was lost"):
            full_grid.commit_job_output(
                root,
                plan,
                claim,
                metadata=observed_metadata,
                test_rows=np.asarray([2, 3], dtype=np.int64),
                probabilities=np.asarray(
                    [[0.8, 0.2], [0.3, 0.7]],
                    dtype=np.float64,
                ),
                cache_root=tmp_path / "cache",
                before_publish=resources,
                gpu_lease=lease,
            )
        assert not full_grid._record_directory(root, job).exists()
        quarantined = list(
            (root / "quarantine" / "records").glob(
                "*.lease-postcondition.*"
            )
        )
        assert len(quarantined) == 1
        assert quarantined[0].is_dir()
        assert quarantined[0].stat().st_mode & 0o222 == 0
    finally:
        full_grid.release_claim(claim)
        project_gpu_leases.release_gpu_lease(lease)


def test_formal_claim_and_final_record_rename_are_guarded_and_receipt_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, plan = initialize(tmp_path)
    job = one_job(plan)
    gpu_uuid = "GPU-00000001-0000-0000-0000-000000000001"
    formal = copy.deepcopy(plan)
    formal["publication_mode"] = full_grid.FORMAL_PUBLICATION_MODE
    formal["execution_config"]["physical_gpu_uuid_roster"] = [gpu_uuid]
    formal["environment_identity"] = {
        "nvidia_driver_versions": None,
        "nvidia_gpu_inventory": [
            {
                "index": "0",
                "uuid": gpu_uuid,
                "pci_bus_id": "0000:01:00.0",
                "name": "Test GPU",
                "driver_version": "1",
            }
        ],
    }
    lease = project_gpu_leases.acquire_gpu_lease(
        project_root=tmp_path,
        run_root=root,
        plan_sha256=formal["plan_sha256"],
        gpu_uuid=gpu_uuid,
        track_scope=full_grid.TRACK_SCOPE,
    )
    monkeypatch.setattr(full_grid, "validate_plan_semantics", lambda *args, **kw: None)
    monkeypatch.setattr(full_grid, "_verified_project_root", lambda: tmp_path)
    monkeypatch.setattr(
        full_grid,
        "load_preflight_attestation",
        lambda *args, **kwargs: {"report_sha256": "f" * 64},
    )
    monkeypatch.setattr(
        full_grid,
        "_verify_job_runtime_identity",
        lambda *args, **kwargs: None,
    )

    def formal_resources() -> full_grid.ResourceStatus:
        return full_grid.combine_resource_status(
            full_grid.GPUStatus(
                safe=True,
                gpu=gpu_uuid,
                utilization_percent=0.0,
                memory_used_mib=0.0,
                own_compute_memory_mib=0.0,
                foreign_processes=(),
                reason="idle",
                gpu_uuid=gpu_uuid,
                pci_bus_id="0000:01:00.0",
                name="Test GPU",
            ),
            full_grid.DiskStatus(
                safe=True,
                path=str(root),
                free_bytes=60 * 1024**3,
                total_bytes=100 * 1024**3,
                free_gib=60.0,
                minimum_free_gib=50.0,
                reason="above floor",
            ),
        )

    state = {
        "guard_depth": 0,
        "claim_write_guarded": False,
        "record_rename_guarded": False,
    }
    original_guard = project_gpu_leases.guard_gpu_lease

    @contextmanager
    def observed_guard(
        selected: project_gpu_leases.GPULease,
    ):
        with original_guard(selected) as receipt:
            state["guard_depth"] += 1
            try:
                yield receipt
            finally:
                state["guard_depth"] -= 1

    original_write = full_grid._write_json_exclusive

    def observed_write(
        path: Path,
        value: dict[str, Any],
        *,
        on_publish: Any = None,
    ) -> os.stat_result:
        if path == full_grid._claim_path(root, job):
            state["claim_write_guarded"] = state["guard_depth"] > 0
        return original_write(path, value, on_publish=on_publish)

    record_name = full_grid._record_directory(root, job).name
    original_rename = full_grid._atomic_rename_noreplace_at

    def observed_rename(
        source_descriptor: int,
        source_name: str,
        destination_descriptor: int,
        destination_name: str,
    ) -> None:
        if destination_name == record_name:
            state["record_rename_guarded"] = state["guard_depth"] > 0
        original_rename(
            source_descriptor,
            source_name,
            destination_descriptor,
            destination_name,
        )

    monkeypatch.setattr(project_gpu_leases, "guard_gpu_lease", observed_guard)
    monkeypatch.setattr(full_grid, "_write_json_exclusive", observed_write)
    monkeypatch.setattr(full_grid, "_atomic_rename_noreplace_at", observed_rename)
    claim: full_grid.Claim | None = None
    try:
        claim = full_grid.acquire_claim(
            root,
            formal,
            job,
            before_claim=formal_resources,
            gpu_lease=lease,
        )
        observed_metadata = metadata(formal, job)
        observed_metadata["runtime"]["physical_gpu_uuid"] = gpu_uuid
        observed_metadata["runtime"]["cuda_visible_devices"] = gpu_uuid
        full_grid.commit_job_output(
            root,
            formal,
            claim,
            metadata=observed_metadata,
            test_rows=np.asarray([2, 3], dtype=np.int64),
            probabilities=np.asarray(
                [[0.8, 0.2], [0.3, 0.7]],
                dtype=np.float64,
            ),
            cache_root=tmp_path / "cache",
            before_publish=formal_resources,
            gpu_lease=lease,
        )
        completed = full_grid.validate_completion(root, formal, job)
        assert claim.gpu_lease_receipt == completed["record"][
            "gpu_lease_receipt"
        ]
        assert state["claim_write_guarded"]
        assert state["record_rename_guarded"]
    finally:
        if claim is not None:
            full_grid.release_claim(claim)
        project_gpu_leases.release_gpu_lease(lease)


def test_process_identity_is_non_null_and_bool_safe() -> None:
    identity = full_grid._process_identity()
    assert isinstance(identity["pid"], int) and not isinstance(identity["pid"], bool)
    assert identity["boot_id"]
    assert identity["start_ticks"]


def test_regular_file_modes_are_enforced(tmp_path: Path) -> None:
    path = tmp_path / "value"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(full_grid.FullGridError, match="read-only"):
        full_grid._require_unique_regular(path, read_only=True)
    path.chmod(stat.S_IRUSR)
    full_grid._require_unique_regular(path, read_only=True)
