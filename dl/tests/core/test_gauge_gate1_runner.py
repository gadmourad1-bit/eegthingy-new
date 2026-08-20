from __future__ import annotations

import copy
import os
import stat
import sys
import zipfile
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from benchmark import gauge_gate1_analysis as analysis
from benchmark import gauge_gate1_runner as gate


DUMMY_GPUS = (
    "GPU-00000000-0000-0000-0000-000000000001",
    "GPU-00000000-0000-0000-0000-000000000002",
    "GPU-00000000-0000-0000-0000-000000000003",
)


@pytest.fixture(autouse=True)
def _allow_darwin_synthetic_rename(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if sys.platform == "darwin":
        monkeypatch.setenv("EEG_MI_DARWIN_SYNTHETIC_TEST", "1")


def _fake_environment() -> dict[str, Any]:
    freeze = ["einops==1", "torch==2"]
    return {
        "schema": gate.ENVIRONMENT_SCHEMA,
        "python": "3.12",
        "python_executable": "/private/venv/bin/python",
        "python_prefix": "/private/venv",
        "platform": "test-platform",
        "uv_binding": {
            "kind": "required_absolute_environment_variable",
            "name": gate.UV_EXECUTABLE_ENVIRONMENT_VARIABLE,
        },
        "uv_executable": "/private/bin/uv",
        "uv_executable_sha256": "1" * 64,
        "uv_version": "uv 1",
        "pyvenv_cfg_sha256": "2" * 64,
        "uv_pip_freeze": freeze,
        "uv_pip_freeze_sha256": gate._sha256_bytes(
            "\n".join(freeze).encode()
        ),
        "required_packages": {
            name: "1" for name in gate.REQUIRED_PACKAGES
        },
        "torch_cuda_version": "12.4",
        "torch_cudnn_version": 90100,
        "nvidia_driver_versions": ["550.1"],
        "required_cublas_workspace_config": (
            gate.REQUIRED_CUBLAS_WORKSPACE_CONFIG
        ),
        "cpu_threads_per_worker": gate.CPU_THREADS,
        "minimum_free_gib": gate.MINIMUM_FREE_GIB,
    }


def _fake_reference(root: Path) -> dict[str, Any]:
    jobs = [
        gate.ScreenJob(
            dataset=job.dataset,
            model=model,
            subject=job.subject,
            fold=job.fold,
            seed=job.seed,
        )
        for model in gate.EXPECTED_REFERENCE_MODELS
        for job in gate._candidate_manifest()
    ]
    records = [
        {
            "relative_path": str(gate._reference_record_relative(job)),
            "file_sha256": f"{index + 100:064x}",
            "file_size_bytes": 100 + index,
            "job_id": job.job_id,
            **job.identity(),
        }
        for index, job in enumerate(jobs)
    ]
    plan = {
        "relative_path": "plan.json",
        "file_sha256": "a" * 64,
        "file_size_bytes": 100,
        "schema": gate.REFERENCE_PLAN_SCHEMA,
        "embedded_plan_sha256": gate.EXPECTED_REFERENCE_PLAN_SHA256,
    }
    return {
        "schema": "eeg-mi-gauge-gate1-reference-snapshot-v1",
        "access": "read_only_reuse_no_rerun_no_mutation",
        "root": str(root),
        "plan": plan,
        "records": records,
        "record_count": len(records),
        "snapshot_sha256": gate._mapping_sha256(
            {"plan": plan, "records": records}
        ),
    }


@pytest.fixture()
def plan_layout(tmp_path: Path) -> tuple[dict[str, Any], Path, Path]:
    project = tmp_path / "source"
    run_parent = tmp_path / "runs"
    run = run_parent / "gate"
    cache = tmp_path / "data_cache"
    reference = run_parent / "reference"
    for path in (project, run_parent, cache, reference):
        path.mkdir(parents=True, exist_ok=True)
    gate._ensure_run_layout(project, run)
    fence, fence_stat = gate._open_or_create_authority_fence(run)
    os.close(fence)
    sources = {
        name: {"sha256": f"{index + 1:064x}", "size_bytes": index + 1}
        for index, name in enumerate(gate.SOURCE_FILES)
    }
    caches: dict[str, Any] = {}
    splits: dict[str, Any] = {}
    for index, job in enumerate(gate._candidate_manifest()):
        caches[gate._subject_key(job)] = {
            "relative_path": (
                f"{gate.PREPROCESSING['schema']}/"
                f"{gate.DEFAULT_MONTAGE_PROFILE}/{job.dataset}/"
                f"subject_{job.subject:03d}.npz"
            ),
            "file_size_bytes": 100 + index,
            "file_sha256": f"{index + 500:064x}",
            "array_sha256": f"{index + 800:064x}",
        }
        splits[gate._job_key(job)] = {
            "train": {
                "count": 8,
                "rows_sha256": f"{index + 1000:064x}",
            },
            "validation": {
                "count": 4,
                "rows_sha256": f"{index + 1200:064x}",
            },
        }
    roster = [
        {
            "worker_index": index,
            "gpu_uuid": gpu,
            "pci_bus_id": f"0000:0{index + 1}:00.0",
            "name": "Dummy GPU",
        }
        for index, gpu in enumerate(DUMMY_GPUS)
    ]
    plan = gate.assemble_plan(
        project_root=project,
        run_root=run,
        cache_root=cache,
        authority_fence={
            "st_dev": int(fence_stat.st_dev),
            "st_ino": int(fence_stat.st_ino),
        },
        source_identity=sources,
        environment_identity=_fake_environment(),
        cache_identity=caches,
        split_identity=splits,
        reference_identity=_fake_reference(reference),
        gpu_roster=roster,
    )
    return plan, project, run


def _job(plan: dict[str, Any], index: int = 0) -> gate.ScreenJob:
    item = plan["jobs"][index]
    return gate._job_from_mapping(
        {key: item[key] for key in ("dataset", "model", "subject", "fold", "seed")}
    )


def _fake_receipt(plan: dict[str, Any], gpu: str = DUMMY_GPUS[0]) -> dict[str, Any]:
    owner = gate.project_gpu_leases.owner_identity()
    return {
        "schema": "eeg-mi-project-gpu-lease-receipt-v1",
        "lease": {
            "schema": "eeg-mi-project-gpu-lease-v1",
            "created_at": "2026-07-30T00:00:00+00:00",
            "project_root_sha256": "a" * 64,
            "track_scope": gate.TRACK_SCOPE,
            "run_root_sha256": "b" * 64,
            "plan_sha256": plan["plan_sha256"],
            "gpu_uuid": gpu,
            "nonce": "c" * 32,
            "owner": owner,
        },
        "lease_sha256": "d" * 64,
        "lease_st_dev": 1,
        "lease_st_ino": 1,
        "registry_fence_st_dev": 1,
        "registry_fence_st_ino": 1,
    }


def _resource(plan: dict[str, Any], gpu: str = DUMMY_GPUS[0]) -> dict[str, Any]:
    row = next(item for item in plan["gpu_roster"] if item["gpu_uuid"] == gpu)
    return {
        "gpu": {
            "gpu_uuid": gpu,
            "pci_bus_id": row["pci_bus_id"],
            "name": row["name"],
            "utilization_percent": 0.0,
            "memory_used_mib": 0.0,
            "own_pid": os.getpid(),
            "own_compute_memory_mib": 0.0,
            "foreign_compute_processes": [],
        },
        "disk": [
            {
                "path": plan["run_root"],
                "free_bytes": 100 * 1024**3,
                "free_gib": 100.0,
                "minimum_free_gib": gate.MINIMUM_FREE_GIB,
            },
            {
                "path": plan["cache_root"],
                "free_bytes": 100 * 1024**3,
                "free_gib": 100.0,
                "minimum_free_gib": gate.MINIMUM_FREE_GIB,
            },
        ],
    }


def _record(plan: dict[str, Any], job: gate.ScreenJob) -> dict[str, Any]:
    return {
        "schema": gate.RECORD_SCHEMA,
        "created_at": "2026-07-30T00:00:00+00:00",
        "plan_sha256": plan["plan_sha256"],
        "job": job.identity(),
        "job_id": job.job_id,
        "evaluation_scope": gate.EVALUATION_SCOPE,
        "confirmation_evidence": False,
        "opened_development_only": True,
        "candidate_only": True,
        "cache_identity": copy.deepcopy(
            plan["cache_identity"][gate._subject_key(job)]
        ),
        "split": copy.deepcopy(plan["split_identity"][gate._job_key(job)]),
        "source_identity_sha256": plan["source_identity_sha256"],
        "environment_identity_sha256": plan["environment_identity_sha256"],
        "reference_identity_sha256": plan["reference_identity"]["snapshot_sha256"],
        "scaler": {
            "fit_scope": "training_only",
            "mean_sha256": "1" * 64,
            "std_sha256": "2" * 64,
            "contract": dict(gate.CHANNEL_SCALING),
        },
        "model": {
            "identity": {
                "requested_name": gate.MODEL_NAME,
                "class": (
                    "benchmark.gauge_quotient."
                    "GaugeQuotientCrossMomentNet"
                ),
                "uses_positions": True,
                "config": {},
            },
            "parameter_count": 20_000,
            "constructed_state_sha256": "3" * 64,
            "initial_state_sha256": "3" * 64,
            "selected_state_sha256": "4" * 64,
            "source_statistics": {
                "available": False,
                "ran": False,
                "fit_scope": "training_rows_only",
                "outcome_aware": False,
            },
        },
        "fit": {
            "best_epoch": 4,
            "epochs_run": 10,
            "best_validation_loss": 0.4,
            "early_stopping": True,
            "config": copy.deepcopy(plan["train_config"]),
        },
        "validation_metrics": {
            "accuracy": 0.7,
            "balanced_accuracy": 0.7,
            "cohen_kappa": 0.4,
            "negative_log_likelihood": 0.5,
            "roc_auc": 0.8,
        },
        "timing": {
            "construction_seconds": 1.0,
            "fit_seconds": 2.0,
            "evaluation_seconds": 1.0,
            "total_seconds": 5.0,
        },
        "resource_guard": _resource(plan),
        "gpu_lease_receipt": _fake_receipt(plan),
    }


class _DummySnapshot:
    def revalidate(self) -> None:
        return None

    def close(self) -> None:
        return None


def _fake_runtime_stack() -> tuple[
    ExitStack,
    _DummySnapshot,
    _DummySnapshot,
    _DummySnapshot,
    _DummySnapshot,
]:
    return ExitStack(), *(_DummySnapshot() for _ in range(4))


def test_exact_candidate_only_plan_accepts_real_sibling_layout(
    plan_layout: tuple[dict[str, Any], Path, Path],
) -> None:
    plan, project, run = plan_layout
    gate.validate_plan(plan)
    assert Path(plan["run_root"]).parent.name == "runs"
    assert Path(plan["cache_root"]).name == "data_cache"
    assert Path(plan["reference_identity"]["root"]).parent.name == "runs"
    assert Path(plan["run_root"]).parent != project
    assert len(plan["jobs"]) == 17
    assert {item["model"] for item in plan["jobs"]} == {gate.MODEL_NAME}
    assert [item["worker_index"] for item in plan["jobs"]] == [
        index % 3 for index in range(17)
    ]


def test_read_only_layout_preflight_accepts_real_sibling_shape_without_writing(
    tmp_path: Path,
) -> None:
    project = tmp_path / "source"
    runs = tmp_path / "runs"
    reference = runs / "chsd_validation_screen_v2"
    cache = tmp_path / "data_cache"
    requested_run = runs / "gauge_gate1_opened17"
    for path in (project, reference, cache):
        path.mkdir(parents=True, exist_ok=True)
    before = {
        path: tuple(sorted(child.name for child in path.iterdir()))
        for path in (project, runs, reference, cache)
    }
    result = gate.preflight_launch_layout(
        project_root=project,
        run_root=requested_run,
        reference_root=reference,
        cache_root=cache,
    )
    after = {
        path: tuple(sorted(child.name for child in path.iterdir()))
        for path in (project, runs, reference, cache)
    }
    assert result["run_root"]["exists"] is False
    assert not requested_run.exists()
    assert before == after


def test_plan_binds_exact_approved_hashes(
    plan_layout: tuple[dict[str, Any], Path, Path],
) -> None:
    plan, _, _ = plan_layout
    assert plan["train_config_sha256"] == gate.TRAIN_CONFIG_SHA256
    assert plan["decision_rule_sha256"] == gate.OPENED17_RULE_SHA256
    assert (
        plan["reference_identity"]["plan"]["embedded_plan_sha256"]
        == gate.EXPECTED_REFERENCE_PLAN_SHA256
    )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("worker_count",), True),
        (("job_count",), 18),
        (("confirmation_evidence",), True),
        (("execution_contract", "comparator_training_allowed"), True),
    ],
)
def test_plan_rejects_type_scope_and_comparator_tampering(
    plan_layout: tuple[dict[str, Any], Path, Path],
    path: tuple[str, ...],
    value: Any,
) -> None:
    plan = copy.deepcopy(plan_layout[0])
    target: dict[str, Any] = plan
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    plan["plan_sha256"] = gate._payload_sha256(plan, "plan_sha256")
    with pytest.raises(gate.GaugeGate1Error):
        gate.validate_plan(plan)


def test_plan_source_closure_includes_novelty_and_analyzer(
    plan_layout: tuple[dict[str, Any], Path, Path],
) -> None:
    plan = plan_layout[0]
    assert "docs/GAUGE_QUOTIENT_NOVELTY_PLAN.md" in plan["source_identity"]
    assert "src/benchmark/gauge_gate1_analysis.py" in plan["source_identity"]


@pytest.mark.parametrize(
    "payload",
    [
        b'{"a":1,"a":2}',
        b'{"a":NaN}',
        b'[]',
        b"\xff",
    ],
)
def test_strict_json_rejects_duplicates_nonfinite_and_nonobjects(
    payload: bytes,
) -> None:
    with pytest.raises(gate.GaugeGate1Error):
        gate._strict_json_bytes(payload, source="fixture")


def test_npz_member_validator_rejects_duplicate_member(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.npz"
    with zipfile.ZipFile(path, "w") as archive:
        for name in gate.SUBJECT_CACHE_NPZ_MEMBERS:
            archive.writestr(f"{name}.npy", b"x")
        archive.writestr("x.npy", b"again")
    snapshot = gate._open_file_snapshot(path)
    try:
        with pytest.raises(gate.GaugeGate1Error, match="duplicated"):
            gate._validate_npz_members(snapshot)
    finally:
        snapshot.close()


def _valid_npz_cache(
    tmp_path: Path,
) -> tuple[
    Path,
    dict[str, Any],
    dict[str, np.ndarray[Any, Any]],
]:
    root = tmp_path / "cache"
    relative = Path("synthetic") / "subject_001.npz"
    path = root / relative
    path.parent.mkdir(parents=True)
    arrays = {
        "x": np.arange(36, dtype=np.float32).reshape(6, 2, 3),
        "y": np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64),
        "sessions": np.asarray(["s0", "s0", "s1", "s1", "s2", "s2"]),
        "runs": np.asarray(["r0", "r0", "r1", "r1", "r2", "r2"]),
        "positions": np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        ),
        "channel_names": np.asarray(["C3", "C4"]),
        "identity": np.asarray('{"fixture":"valid"}'),
    }
    np.savez(path, **arrays)
    payload = path.read_bytes()
    contract = {
        "relative_path": str(relative),
        "file_size_bytes": len(payload),
        "file_sha256": gate._sha256_bytes(payload),
        "array_sha256": "a" * 64,
    }
    return root, contract, arrays


def test_valid_npz_consumers_are_sequential_and_offset_independent(
    tmp_path: Path,
) -> None:
    root, contract, arrays = _valid_npz_cache(tmp_path)
    snapshot = gate._cache_snapshot(root, contract)
    try:
        assert os.lseek(snapshot.descriptor, 0, os.SEEK_CUR) == 0
        os.lseek(snapshot.descriptor, 7, os.SEEK_SET)
        gate._validate_npz_members(snapshot)
        assert os.lseek(snapshot.descriptor, 0, os.SEEK_CUR) == 7
        metadata = gate._npz_metadata(snapshot)
        assert metadata["channel_names"] == ("C3", "C4")
        assert os.lseek(snapshot.descriptor, 0, os.SEEK_CUR) == 7
        repeated = gate._npz_metadata(snapshot)
        assert repeated["identity_raw"] == metadata["identity_raw"]
        assert os.lseek(snapshot.descriptor, 0, os.SEEK_CUR) == 7
        rows = np.asarray([4, 1, 5], dtype=np.int64)
        selected_x, x_shape = gate._read_selected_npy_rows(
            snapshot,
            "x",
            rows,
            expected_dtype=np.dtype(np.float32),
            expected_ndim=3,
        )
        assert x_shape == arrays["x"].shape
        np.testing.assert_array_equal(selected_x, arrays["x"][rows])
        assert os.lseek(snapshot.descriptor, 0, os.SEEK_CUR) == 7
        selected_y, y_shape = gate._read_selected_npy_rows(
            snapshot,
            "y",
            rows,
            expected_dtype=np.dtype(np.int64),
            expected_ndim=1,
        )
        assert y_shape == arrays["y"].shape
        np.testing.assert_array_equal(selected_y, arrays["y"][rows])
        assert os.lseek(snapshot.descriptor, 0, os.SEEK_CUR) == 7
        gate._assert_file_snapshot(snapshot)
    finally:
        snapshot.close()


def test_npz_metadata_translates_eoferror_to_protocol_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, contract, _ = _valid_npz_cache(tmp_path)
    snapshot = gate._cache_snapshot(root, contract)

    def fail_load(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise EOFError("synthetic inherited-offset EOF")

    monkeypatch.setattr(gate.np, "load", fail_load)
    try:
        os.lseek(snapshot.descriptor, 11, os.SEEK_SET)
        with pytest.raises(gate.GaugeGate1Error, match="cannot read cache"):
            gate._npz_metadata(snapshot)
        assert os.lseek(snapshot.descriptor, 0, os.SEEK_CUR) == 11
        gate._assert_file_snapshot(snapshot)
    finally:
        snapshot.close()


def test_uv_executable_binding_is_explicit_absolute_and_path_independent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(gate.UV_EXECUTABLE_ENVIRONMENT_VARIABLE, raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(gate.GaugeGate1Error, match="must bind"):
        gate._bound_uv_executable()

    monkeypatch.setenv(gate.UV_EXECUTABLE_ENVIRONMENT_VARIABLE, "uv")
    with pytest.raises(gate.GaugeGate1Error, match="canonical absolute"):
        gate._bound_uv_executable()

    executable = tmp_path / "uv"
    executable.write_bytes(b"synthetic uv")
    executable.chmod(0o700)
    monkeypatch.setenv(
        gate.UV_EXECUTABLE_ENVIRONMENT_VARIABLE,
        str(executable),
    )
    path, snapshot = gate._bound_uv_executable()
    try:
        assert path == executable
        assert snapshot.payload == b"synthetic uv"
    finally:
        snapshot.close()


def test_package_reader_pins_all_children_and_rejects_directory_swap(
    tmp_path: Path,
) -> None:
    root = tmp_path / "staging"
    root.mkdir(mode=0o700)
    first, _ = gate._stage_package(
        staging_parent=root,
        basename="first",
        members={"a": b"one", "b": b"two", "c": b"three"},
    )
    second, _ = gate._stage_package(
        staging_parent=root,
        basename="second",
        members={"a": b"four", "b": b"five", "c": b"six"},
    )
    canonical = tmp_path / "canonical"
    gate._rename_noreplace(first, canonical)
    snapshot = gate._open_package(
        canonical, expected_names=frozenset({"a", "b", "c"})
    )
    displaced = tmp_path / "displaced"
    gate._rename_noreplace(canonical, displaced)
    gate._rename_noreplace(second, canonical)
    try:
        with pytest.raises(gate.GaugeGate1Error, match="changed|binding"):
            snapshot.revalidate()
    finally:
        snapshot.close()


def _same_parent_aba_restore(original: Path, replacement: Path) -> None:
    displaced_original = original.with_name(f".original-{original.name}")
    displaced_replacement = original.with_name(
        f".replacement-{original.name}"
    )
    gate._rename_noreplace(original, displaced_original)
    gate._rename_noreplace(replacement, original)
    gate._rename_noreplace(original, displaced_replacement)
    gate._rename_noreplace(displaced_original, original)


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="Linux ext4 namespace-history regression",
)
def test_linux_file_snapshot_rejects_same_parent_aba_restoration(
    tmp_path: Path,
) -> None:
    stable = tmp_path / "stable"
    stable.mkdir()
    original = stable / "evidence.json"
    replacement = stable / "replacement.json"
    original.write_bytes(b'{"identity":"A"}\n')
    replacement.write_bytes(b'{"identity":"B"}\n')
    snapshot = gate._open_file_snapshot(
        original,
        namespace_root=stable,
    )
    try:
        _same_parent_aba_restore(original, replacement)
        assert original.read_bytes() == snapshot.payload
        # This ext4 mount updates leaf ctime on rename, while the independent
        # audit fixture preserved it. Neutralize that extra signal so this
        # regression specifically proves parent namespace-history detection.
        neutralized = gate.FileSnapshot(
            path=snapshot.path,
            descriptor=snapshot.descriptor,
            observed=original.lstat(),
            payload=snapshot.payload,
            namespace=snapshot.namespace,
        )
        assert gate._stat_fingerprint(os.fstat(snapshot.descriptor)) == (
            gate._stat_fingerprint(neutralized.observed)
        )
        assert (
            neutralized.observed.st_dev,
            neutralized.observed.st_ino,
        ) == (
            snapshot.namespace.target_observed.st_dev,
            snapshot.namespace.target_observed.st_ino,
        )
        assert gate._read_descriptor(snapshot.descriptor) == snapshot.payload
        with pytest.raises(gate.GaugeGate1Error, match="namespace"):
            gate._assert_file_snapshot(neutralized)
    finally:
        snapshot.close()


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="Linux ext4 namespace-history regression",
)
def test_linux_package_snapshot_rejects_same_parent_aba_restoration(
    tmp_path: Path,
) -> None:
    stable = tmp_path / "stable"
    staging = stable / gate.STAGING_DIRECTORY
    stable.mkdir()
    staging.mkdir(mode=0o700)
    first, _ = gate._stage_package(
        staging_parent=staging,
        basename="package-a",
        members={"a": b"A", "b": b"2", "c": b"3"},
    )
    second, _ = gate._stage_package(
        staging_parent=staging,
        basename="package-b",
        members={"a": b"B", "b": b"5", "c": b"6"},
    )
    original = stable / "candidate"
    replacement = stable / "replacement"
    gate._rename_noreplace(first, original)
    gate._rename_noreplace(second, replacement)
    snapshot = gate._open_package(
        original,
        expected_names=frozenset({"a", "b", "c"}),
        namespace_root=stable,
    )
    initial_identity = (
        snapshot.observed.st_dev,
        snapshot.observed.st_ino,
    )
    try:
        _same_parent_aba_restore(original, replacement)
        assert (original / "a").read_bytes() == b"A"
        # Neutralize the package inode/directory ctime signal to force the
        # retained parent-chain mutation fingerprint to carry the rejection.
        snapshot.observed = original.lstat()
        snapshot.fingerprint = gate._directory_fingerprint(
            snapshot.descriptor
        )
        snapshot.namespace.target_observed = snapshot.observed
        assert gate._stat_fingerprint(os.fstat(snapshot.descriptor)) == (
            gate._stat_fingerprint(snapshot.observed)
        )
        assert gate._directory_fingerprint(snapshot.descriptor) == (
            snapshot.fingerprint
        )
        assert (
            original.lstat().st_dev,
            original.lstat().st_ino,
        ) == initial_identity
        for child in snapshot.children.values():
            gate._assert_file_snapshot_at(snapshot.descriptor, child)
        with pytest.raises(gate.GaugeGate1Error, match="namespace"):
            snapshot.revalidate()
    finally:
        snapshot.close()


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="Linux ext4 namespace-history regression",
)
def test_linux_snapshot_rejects_ancestor_directory_aba_restoration(
    tmp_path: Path,
) -> None:
    stable = tmp_path / "stable"
    original = stable / "branch"
    replacement = stable / "replacement"
    (original / "nested").mkdir(parents=True)
    (replacement / "nested").mkdir(parents=True)
    path = original / "nested" / "evidence.json"
    path.write_bytes(b'{"identity":"A"}\n')
    (replacement / "nested" / "evidence.json").write_bytes(
        b'{"identity":"B"}\n'
    )
    snapshot = gate._open_file_snapshot(
        path,
        namespace_root=stable,
    )
    try:
        _same_parent_aba_restore(original, replacement)
        assert path.read_bytes() == snapshot.payload
        with pytest.raises(gate.GaugeGate1Error, match="namespace"):
            gate._assert_file_snapshot(snapshot)
    finally:
        snapshot.close()


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="Linux namespace scope regression",
)
def test_namespace_snapshot_ignores_changes_inside_unrelated_sibling_subtree(
    tmp_path: Path,
) -> None:
    stable = tmp_path / "stable"
    evidence_parent = stable / "evidence"
    unrelated = stable / "unrelated"
    evidence_parent.mkdir(parents=True)
    unrelated.mkdir()
    path = evidence_parent / "input.json"
    path.write_bytes(b'{"identity":"A"}\n')
    snapshot = gate._open_file_snapshot(
        path,
        namespace_root=stable,
    )
    try:
        (unrelated / "ordinary-worker-output").write_bytes(b"unrelated")
        gate._assert_file_snapshot(snapshot)
    finally:
        snapshot.close()


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="Linux real-layout namespace publication regression",
)
def test_real_layout_allows_results_then_analysis_without_self_invalidation(
    tmp_path: Path,
) -> None:
    project = tmp_path / "source"
    run = tmp_path / "runs" / "gate"
    project.mkdir()
    run.parent.mkdir()
    gate._ensure_run_layout(project, run)
    staging = run / gate.STAGING_DIRECTORY

    def publish(
        basename: str,
        destination: Path,
        marker: bytes,
    ) -> gate.PackageSnapshot:
        stage, observed = gate._stage_package(
            staging_parent=staging,
            basename=basename,
            members={"a": marker, "b": b"2", "c": b"3"},
        )
        return gate._publish_package(
            stage=stage,
            stage_stat=observed,
            destination=destination,
            expected_names=frozenset({"a", "b", "c"}),
            validator=lambda _: None,
            namespace_root=run,
        )

    plan_snapshot = publish("plan-layout", run / gate.PLAN_DIRECTORY, b"plan")
    result_snapshot: gate.PackageSnapshot | None = None
    analysis_snapshot: gate.PackageSnapshot | None = None
    try:
        result_parent = gate._secure_mkdir_beneath(
            run / gate.RECORDS_DIRECTORY,
            Path("dataset/model/subject/fold"),
        )
        result_snapshot = publish(
            "result-layout",
            result_parent / "result",
            b"result",
        )
        plan_snapshot.revalidate()
        result_snapshot.revalidate()

        analysis_snapshot = publish(
            "analysis-layout",
            run
            / gate.ANALYSIS_ROOT_DIRECTORY
            / analysis.ANALYSIS_DIRECTORY,
            b"analysis",
        )
        plan_snapshot.revalidate()
        result_snapshot.revalidate()
        analysis_snapshot.revalidate()
    finally:
        if analysis_snapshot is not None:
            analysis_snapshot.close()
        if result_snapshot is not None:
            result_snapshot.close()
        plan_snapshot.close()


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="Linux three-worker namespace-isolation regression",
)
def test_precreated_three_worker_parents_isolate_sibling_publications(
    tmp_path: Path,
) -> None:
    project = tmp_path / "source"
    run = tmp_path / "runs" / "gate"
    project.mkdir()
    run.parent.mkdir()
    gate._ensure_run_layout(project, run)
    jobs = gate._candidate_manifest()
    gate._prepare_record_parents(run, jobs)
    for job in jobs:
        gate._validate_record_parent(run, job)

    worker_jobs = tuple(jobs[index] for index in range(gate.WORKER_COUNT))
    assert len({job.dataset for job in worker_jobs}) == 1
    assert len({job.subject for job in worker_jobs}) == gate.WORKER_COUNT
    staging = run / gate.STAGING_DIRECTORY
    snapshots: list[gate.PackageSnapshot] = []
    try:
        for worker_index, job in enumerate(worker_jobs):
            stage, observed = gate._stage_package(
                staging_parent=staging,
                basename=f"worker-{worker_index}",
                members={"a": str(worker_index).encode(), "b": b"2", "c": b"3"},
            )
            snapshot = gate._publish_package(
                stage=stage,
                stage_stat=observed,
                destination=gate._record_path(run, job),
                expected_names=frozenset({"a", "b", "c"}),
                validator=lambda _: None,
                namespace_root=run,
            )
            snapshots.append(snapshot)
            for held in snapshots:
                held.revalidate()
    finally:
        for snapshot in reversed(snapshots):
            snapshot.close()


def test_worker_record_parent_check_is_validation_only(tmp_path: Path) -> None:
    project = tmp_path / "source"
    run = tmp_path / "runs" / "gate"
    project.mkdir()
    run.parent.mkdir()
    gate._ensure_run_layout(project, run)
    jobs = gate._candidate_manifest()
    gate._prepare_record_parents(run, jobs)
    missing = run / gate._record_parent_relative(jobs[0])
    missing.rmdir()

    with pytest.raises(gate.GaugeGate1Error, match="absent or unsafe"):
        gate._validate_record_parent(run, jobs[0])
    assert not missing.exists()


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="Linux descriptor lifecycle regression",
)
def test_snapshot_close_releases_leaf_package_and_namespace_descriptors(
    tmp_path: Path,
) -> None:
    stable = tmp_path / "stable"
    stable.mkdir()
    leaf = stable / "leaf.json"
    leaf.write_bytes(b"leaf")
    baseline = len(os.listdir("/proc/self/fd"))
    file_snapshot = gate._open_file_snapshot(
        leaf,
        namespace_root=stable,
    )
    assert len(os.listdir("/proc/self/fd")) > baseline
    file_snapshot.close()
    file_snapshot.close()
    assert len(os.listdir("/proc/self/fd")) == baseline

    staging = stable / gate.STAGING_DIRECTORY
    staging.mkdir(mode=0o700)
    stage, _ = gate._stage_package(
        staging_parent=staging,
        basename="fd-lifecycle",
        members={"a": b"1", "b": b"2", "c": b"3"},
    )
    package = stable / "package"
    gate._rename_noreplace(stage, package)
    package_snapshot = gate._open_package(
        package,
        expected_names=frozenset({"a", "b", "c"}),
        namespace_root=stable,
    )
    assert len(os.listdir("/proc/self/fd")) > baseline
    package_snapshot.close()
    package_snapshot.close()
    assert len(os.listdir("/proc/self/fd")) == baseline


def test_package_reader_rejects_writable_child(tmp_path: Path) -> None:
    stage_root = tmp_path / "staging"
    stage_root.mkdir(mode=0o700)
    stage, _ = gate._stage_package(
        staging_parent=stage_root,
        basename="writable",
        members={"a": b"1", "b": b"2", "c": b"3"},
    )
    os.chmod(stage, 0o755)
    os.chmod(stage / "a", 0o644)
    with pytest.raises(gate.GaugeGate1Error, match="sealed"):
        gate._open_package(stage, expected_names=frozenset({"a", "b", "c"}))


def test_publish_package_move_then_raise_hides_exact_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir(mode=0o700)
    stage, observed = gate._stage_package(
        staging_parent=staging,
        basename="move-raise",
        members={"a": b"1", "b": b"2", "c": b"3"},
    )
    destination = tmp_path / "result"
    original = gate._rename_noreplace

    def move_then_raise(source: Path, target: Path) -> None:
        original(source, target)
        raise OSError("fault after successful rename")

    monkeypatch.setattr(gate, "_rename_noreplace", move_then_raise)
    with pytest.raises(OSError, match="fault"):
        gate._publish_package(
            stage=stage,
            stage_stat=observed,
            destination=destination,
            expected_names=frozenset({"a", "b", "c"}),
            validator=lambda _: None,
        )
    assert not destination.exists()
    assert any(
        path.name.startswith(".invalid-postrename.result")
        for path in tmp_path.iterdir()
    )


def test_publish_package_validator_failure_leaves_no_canonical_result(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir(mode=0o700)
    stage, observed = gate._stage_package(
        staging_parent=staging,
        basename="bad-validator",
        members={"a": b"1", "b": b"2", "c": b"3"},
    )
    destination = tmp_path / "result"

    def reject(_: gate.PackageSnapshot) -> None:
        raise gate.GaugeGate1Error("postrename validation fault")

    with pytest.raises(gate.GaugeGate1Error, match="postrename"):
        gate._publish_package(
            stage=stage,
            stage_stat=observed,
            destination=destination,
            expected_names=frozenset({"a", "b", "c"}),
            validator=reject,
        )
    assert not destination.exists()


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="Linux cross-parent directory rename permission semantics",
)
def test_linux_cross_parent_sealed_stage_is_resealed_after_publication(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir(mode=0o700)
    stage, observed = gate._stage_package(
        staging_parent=staging,
        basename="linux-cross-parent-success",
        members={"a": b"1", "b": b"2", "c": b"3"},
    )
    destination = tmp_path / "result"
    assert stat.S_IMODE(stage.lstat().st_mode) == 0o555

    gate._rename_noreplace(stage, destination)

    moved = destination.lstat()
    assert (moved.st_dev, moved.st_ino) == (observed.st_dev, observed.st_ino)
    assert stat.S_IMODE(moved.st_mode) == 0o555


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="Linux cross-parent directory rename permission semantics",
)
def test_linux_failed_noreplace_reseals_stage_and_preserves_destination(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir(mode=0o700)
    stage, observed = gate._stage_package(
        staging_parent=staging,
        basename="linux-cross-parent-exists",
        members={"a": b"stage", "b": b"2", "c": b"3"},
    )
    destination = tmp_path / "result"
    destination.mkdir()
    marker = destination / "existing"
    marker.write_bytes(b"do-not-replace")

    with pytest.raises(FileExistsError):
        gate._rename_noreplace(stage, destination)

    current = stage.lstat()
    assert (current.st_dev, current.st_ino) == (
        observed.st_dev,
        observed.st_ino,
    )
    assert stat.S_IMODE(current.st_mode) == 0o555
    assert marker.read_bytes() == b"do-not-replace"


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="Linux cross-parent directory rename permission semantics",
)
def test_linux_permission_recovery_rejects_nonprivate_stage_parent(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir(mode=0o755)
    stage, _ = gate._stage_package(
        staging_parent=staging,
        basename="linux-nonprivate",
        members={"a": b"1", "b": b"2", "c": b"3"},
    )
    destination = tmp_path / "result"

    with pytest.raises(gate.GaugeGate1Error, match="owner-private"):
        gate._rename_noreplace(stage, destination)

    assert stage.exists()
    assert stat.S_IMODE(stage.lstat().st_mode) == 0o555
    assert not destination.exists()


def test_plan_publication_is_sealed_resumable_and_no_replace(
    plan_layout: tuple[dict[str, Any], Path, Path],
) -> None:
    plan, project, run = plan_layout
    assert gate.publish_or_validate_plan(
        project_root=project, run_root=run, plan=plan
    )
    for job in gate._candidate_manifest():
        gate._validate_record_parent(run, job)
    assert not gate.publish_or_validate_plan(
        project_root=project, run_root=run, plan=plan
    )
    snapshot_plan, snapshot = gate.load_plan_snapshot(run)
    try:
        assert snapshot_plan == plan
        assert not (snapshot.observed.st_mode & 0o222)
        assert set(snapshot.children) == {"plan.json", "plan.sha256", "COMMITTED"}
    finally:
        snapshot.close()


def test_claim_open_failure_after_rename_is_rolled_back(
    plan_layout: tuple[dict[str, Any], Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, _, run = plan_layout
    job = _job(plan)
    monkeypatch.setattr(gate, "_validate_claim", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        gate,
        "_open_claim",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            gate.GaugeGate1Error("open validation fault")
        ),
    )
    with pytest.raises(gate.GaugeGate1Error, match="open validation"):
        gate._publish_claim(
            run_root=run,
            plan=plan,
            job=job,
            value={"arbitrary": True},
        )
    assert not gate._claim_path(run, job).exists()


def test_claim_post_guard_failure_hides_published_claim(
    plan_layout: tuple[dict[str, Any], Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, _, run = plan_layout
    job = _job(plan)
    receipt = _fake_receipt(plan)

    @contextmanager
    def failing_guard(_: Any):
        yield receipt
        raise gate.GaugeGate1Error("post-yield lease loss")

    monkeypatch.setattr(gate.project_gpu_leases, "guard_gpu_lease", failing_guard)
    monkeypatch.setattr(gate, "_validate_claim", lambda *args, **kwargs: None)
    monkeypatch.setattr(gate, "_runtime_binding_stack", lambda **_: _fake_runtime_stack())
    monkeypatch.setattr(gate, "_runtime_binding_revalidate", lambda **_: None)
    monkeypatch.setattr(gate, "_probe_gpu", lambda *args, **kwargs: {})
    monkeypatch.setattr(gate, "_probe_disk", lambda *args, **kwargs: {})
    with pytest.raises(gate.GaugeGate1Error, match="post-yield"):
        gate._claim_for_job(
            plan=plan,
            job=job,
            gpu_lease=object(),  # type: ignore[arg-type]
            resource_guard={},
        )
    assert not gate._claim_path(run, job).exists()
    assert any(
        path.name.startswith(f".invalid-claim-guard-exit.{job.job_id}")
        for path in (run / gate.CLAIMS_DIRECTORY).iterdir()
    )


def test_commit_post_guard_failure_hides_exact_result(
    plan_layout: tuple[dict[str, Any], Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, project, run = plan_layout
    job = _job(plan)
    monkeypatch.setattr(
        gate.project_gpu_leases,
        "validate_gpu_lease_receipt",
        lambda *args, **kwargs: None,
    )
    record = _record(plan, job)
    gate._prepare_record_parents(run, gate._candidate_manifest())
    gate._validate_record_parent(run, job)
    claim_path = gate._claim_path(run, job)
    claim_value = {
        "nonce": "a" * 32,
        "gpu_lease_receipt": _fake_receipt(plan),
    }
    payload = gate._canonical_bytes(claim_value) + b"\n"
    claim_stat = gate._write_file_exclusive(claim_path, payload, 0o444)
    claim_fd = os.open(claim_path, os.O_RDONLY)
    claim = gate.ClaimHandle(
        job=job,
        path=claim_path,
        descriptor=claim_fd,
        observed=claim_stat,
        value=claim_value,
    )

    @contextmanager
    def failing_guard(_: Any):
        yield claim_value["gpu_lease_receipt"]
        raise gate.GaugeGate1Error("post-yield result lease loss")

    monkeypatch.setattr(gate.project_gpu_leases, "guard_gpu_lease", failing_guard)
    monkeypatch.setattr(gate, "_assert_claim_handle", lambda *args, **kwargs: None)
    monkeypatch.setattr(gate, "_runtime_binding_stack", lambda **_: _fake_runtime_stack())
    monkeypatch.setattr(gate, "_runtime_binding_revalidate", lambda **_: None)
    monkeypatch.setattr(gate, "_probe_gpu", lambda *args, **kwargs: {})
    monkeypatch.setattr(gate, "_probe_disk", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        gate,
        "_validate_result_package",
        lambda snapshot, **_: record,
    )
    try:
        with pytest.raises(gate.GaugeGate1Error, match="post-yield"):
            gate._commit_result(
                plan=plan,
                job=job,
                claim=claim,
                record=record,
                gpu_lease=object(),  # type: ignore[arg-type]
            )
    finally:
        os.close(claim_fd)
    destination = gate._record_path(run, job)
    assert not destination.exists()
    assert any(
        path.name.startswith(f".invalid-result-guard-exit.{destination.name}")
        for path in destination.parent.iterdir()
    )


def test_quarantine_only_touches_exact_claim_nonce_stage(
    plan_layout: tuple[dict[str, Any], Path, Path],
) -> None:
    plan, _, run = plan_layout
    job = _job(plan)
    nonce = "a" * 32
    other = "b" * 32
    stage_root = run / gate.STAGING_DIRECTORY
    target = stage_root / (gate._stage_prefix(job, nonce) + "one")
    live = stage_root / (gate._stage_prefix(job, other) + "two")
    target.mkdir()
    live.mkdir()
    hidden = gate._quarantine_stages_for_claim(
        run_root=run,
        job=job,
        claim_nonce=nonce,
    )
    assert len(hidden) == 1
    assert not target.exists()
    assert live.exists()


def test_inode_bound_claim_hide_refuses_replacement(
    plan_layout: tuple[dict[str, Any], Path, Path],
) -> None:
    plan, _, run = plan_layout
    job = _job(plan)
    path = gate._claim_path(run, job)
    observed = gate._write_file_exclusive(path, b"old", 0o444)
    descriptor = os.open(path, os.O_RDONLY)
    displaced = path.with_name(".displaced")
    gate._rename_noreplace(path, displaced)
    gate._write_file_exclusive(path, b"replacement", 0o444)
    claim = gate.ClaimHandle(
        job=job,
        path=path,
        descriptor=descriptor,
        observed=observed,
        value={},
    )
    try:
        with pytest.raises(gate.GaugeGate1Error, match="binding"):
            gate._hide_claim(claim, label="should-not-move")
        assert path.read_bytes() == b"replacement"
    finally:
        os.close(descriptor)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("accuracy", True),
        ("balanced_accuracy", 1.1),
        ("cohen_kappa", -1.1),
        ("negative_log_likelihood", -0.1),
        ("roc_auc", float("nan")),
    ],
)
def test_record_rejects_impossible_metric_values(
    plan_layout: tuple[dict[str, Any], Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: Any,
) -> None:
    plan, _, run = plan_layout
    job = _job(plan)
    record = _record(plan, job)
    record["validation_metrics"][field] = value
    monkeypatch.setattr(
        gate.project_gpu_leases,
        "validate_gpu_lease_receipt",
        lambda *args, **kwargs: None,
    )
    with pytest.raises(gate.GaugeGate1Error):
        gate.validate_record(record, plan=plan, job=job, run_root=run)


def test_record_rejects_impossible_epoch_and_timing_aggregates(
    plan_layout: tuple[dict[str, Any], Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, _, run = plan_layout
    job = _job(plan)
    monkeypatch.setattr(
        gate.project_gpu_leases,
        "validate_gpu_lease_receipt",
        lambda *args, **kwargs: None,
    )
    record = _record(plan, job)
    record["fit"]["best_epoch"] = record["fit"]["epochs_run"]
    with pytest.raises(gate.GaugeGate1Error, match="epoch"):
        gate.validate_record(record, plan=plan, job=job, run_root=run)
    record = _record(plan, job)
    record["timing"]["total_seconds"] = 1.0
    with pytest.raises(gate.GaugeGate1Error, match="timing"):
        gate.validate_record(record, plan=plan, job=job, run_root=run)


def _score_fixture() -> tuple[
    dict[str, float], dict[str, dict[str, float]], list[gate.ScreenJob]
]:
    jobs = list(gate._candidate_manifest())
    candidate = {job.job_id: 0.8 for job in jobs}
    references = {
        model: {job.job_id: 0.7 for job in jobs}
        for model in gate.EXPECTED_REFERENCE_MODELS
    }
    return candidate, references, jobs


def test_frozen_decision_passes_only_when_all_four_conditions_pass() -> None:
    candidate, references, _ = _score_fixture()
    value = analysis.decision_from_balanced_accuracy(
        candidate=candidate,
        references=references,
        plan_sha256="a" * 64,
        reference_snapshot_sha256="b" * 64,
        created_at="2026-07-30T00:00:00+00:00",
    )
    assert value["passed"] is True
    assert value["strict_subject_wins"] == 17
    assert all(value["conditions"].values())


def _apply_dataset_deltas(
    candidate: dict[str, float],
    references: dict[str, dict[str, float]],
    jobs: list[gate.ScreenJob],
    deltas: dict[str, list[float]],
) -> None:
    for dataset, values in deltas.items():
        selected = [job for job in jobs if job.dataset == dataset]
        assert len(selected) == len(values)
        for job, delta in zip(selected, values, strict=True):
            candidate[job.job_id] = (
                references[analysis.PRIMARY_REFERENCE][job.job_id] + delta
            )


def test_frozen_decision_can_fail_only_equal_dataset_condition() -> None:
    candidate, references, jobs = _score_fixture()
    datasets = list(dict.fromkeys(job.dataset for job in jobs))
    deltas: dict[str, list[float]] = {}
    for dataset in datasets[:3]:
        count = sum(job.dataset == dataset for job in jobs)
        if count == 3:
            deltas[dataset] = [0.01, 0.01, -0.02]
        else:
            deltas[dataset] = [0.01, 0.01, -0.01, -0.01]
    for dataset in datasets[3:]:
        deltas[dataset] = [0.005, 0.005, -0.015, -0.015]
    _apply_dataset_deltas(candidate, references, jobs, deltas)
    value = analysis.decision_from_balanced_accuracy(
        candidate=candidate,
        references=references,
        plan_sha256="a" * 64,
        reference_snapshot_sha256="b" * 64,
        created_at="2026-07-30T00:00:00+00:00",
    )
    assert value["passed"] is False
    assert (
        value["conditions"][
            "strictly_exceeds_primary_equal_dataset_balanced_accuracy"
        ]
        is False
    )
    assert value["conditions"]["minimum_nonnegative_dataset_deltas"] is True
    assert value["conditions"]["maximum_dataset_deficit"] is True
    assert value["conditions"]["minimum_strict_subject_win_rate"] is True


def test_frozen_decision_can_fail_only_dataset_count_condition() -> None:
    candidate, references, jobs = _score_fixture()
    datasets = list(dict.fromkeys(job.dataset for job in jobs))
    deltas = {
        datasets[0]: [0.08, 0.08, 0.08],
        datasets[1]: [0.08, 0.08, 0.08],
        datasets[2]: [0.005, 0.005, -0.04],
        datasets[3]: [0.005, 0.005, -0.015, -0.015],
        datasets[4]: [0.005, 0.005, -0.015, -0.015],
    }
    _apply_dataset_deltas(candidate, references, jobs, deltas)
    value = analysis.decision_from_balanced_accuracy(
        candidate=candidate,
        references=references,
        plan_sha256="a" * 64,
        reference_snapshot_sha256="b" * 64,
        created_at="2026-07-30T00:00:00+00:00",
    )
    assert value["passed"] is False
    assert value["nonnegative_dataset_delta_count"] == 2
    assert value["strict_subject_wins"] >= 9
    failed = [name for name, passed in value["conditions"].items() if not passed]
    assert failed == ["minimum_nonnegative_dataset_deltas"]


def test_frozen_decision_can_fail_only_maximum_deficit_condition() -> None:
    candidate, references, jobs = _score_fixture()
    datasets = list(dict.fromkeys(job.dataset for job in jobs))
    deltas = {
        datasets[0]: [0.1, 0.1, 0.1],
        datasets[1]: [0.1, 0.1, 0.1],
        datasets[2]: [0.1, 0.1, 0.1],
        datasets[3]: [-0.04] * 4,
        datasets[4]: [-0.04] * 4,
    }
    _apply_dataset_deltas(candidate, references, jobs, deltas)
    value = analysis.decision_from_balanced_accuracy(
        candidate=candidate,
        references=references,
        plan_sha256="a" * 64,
        reference_snapshot_sha256="b" * 64,
        created_at="2026-07-30T00:00:00+00:00",
    )
    assert value["passed"] is False
    failed = [name for name, passed in value["conditions"].items() if not passed]
    assert failed == ["maximum_dataset_deficit"]


def test_frozen_decision_can_fail_only_subject_win_condition() -> None:
    candidate, references, jobs = _score_fixture()
    datasets = list(dict.fromkeys(job.dataset for job in jobs))
    deltas: dict[str, list[float]] = {}
    for dataset in datasets:
        count = sum(job.dataset == dataset for job in jobs)
        deltas[dataset] = [0.2] + [-0.01] * (count - 1)
    _apply_dataset_deltas(candidate, references, jobs, deltas)
    value = analysis.decision_from_balanced_accuracy(
        candidate=candidate,
        references=references,
        plan_sha256="a" * 64,
        reference_snapshot_sha256="b" * 64,
        created_at="2026-07-30T00:00:00+00:00",
    )
    assert value["passed"] is False
    assert value["strict_subject_wins"] == 5
    failed = [name for name, passed in value["conditions"].items() if not passed]
    assert failed == ["minimum_strict_subject_win_rate"]


def test_analysis_validator_rejects_inconsistent_counts() -> None:
    candidate, references, _ = _score_fixture()
    value = analysis.decision_from_balanced_accuracy(
        candidate=candidate,
        references=references,
        plan_sha256="a" * 64,
        reference_snapshot_sha256="b" * 64,
        created_at="2026-07-30T00:00:00+00:00",
    )
    value["strict_subject_wins"] = 18
    with pytest.raises(gate.GaugeGate1Error):
        analysis.validate_analysis(value)


def _existing_analysis_harness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    dict[str, Any],
    dict[str, float],
    dict[str, dict[str, float]],
    dict[str, Any],
]:
    candidate, references, _ = _score_fixture()
    plan = {
        "project_root": str(tmp_path / "source"),
        "run_root": str(tmp_path / "run"),
        "source_identity": {"source": "identity"},
        "environment_identity": {"environment": "identity"},
        "plan_sha256": "a" * 64,
        "reference_identity": {
            "root": str(tmp_path / "reference"),
            "snapshot_sha256": "b" * 64,
        },
        "authority_fence": {"st_dev": 1, "st_ino": 1},
    }
    Path(plan["project_root"]).mkdir()
    Path(plan["run_root"]).mkdir()
    analysis_root = (
        Path(plan["run_root"]) / gate.ANALYSIS_ROOT_DIRECTORY
    )
    analysis_root.mkdir()
    (analysis_root / analysis.ANALYSIS_DIRECTORY).mkdir()
    persisted = analysis.decision_from_balanced_accuracy(
        candidate=candidate,
        references=references,
        plan_sha256=plan["plan_sha256"],
        reference_snapshot_sha256=plan["reference_identity"][
            "snapshot_sha256"
        ],
        created_at="2026-07-30T00:00:00+00:00",
    )
    monkeypatch.setattr(
        gate,
        "load_plan_snapshot",
        lambda _: (plan, _DummySnapshot()),
    )
    monkeypatch.setattr(
        gate,
        "_source_snapshot",
        lambda *args, **kwargs: (plan["source_identity"], _DummySnapshot()),
    )
    monkeypatch.setattr(
        gate,
        "environment_identity",
        lambda: plan["environment_identity"],
    )
    monkeypatch.setattr(
        gate,
        "_open_package",
        lambda *args, **kwargs: _DummySnapshot(),
    )
    monkeypatch.setattr(
        analysis,
        "_validate_analysis_package",
        lambda *args, **kwargs: persisted,
    )
    return plan, candidate, references, persisted


def test_existing_analysis_replays_all_inputs_and_returns_exact_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, candidate, references, persisted = _existing_analysis_harness(
        tmp_path, monkeypatch
    )
    monkeypatch.setattr(
        analysis,
        "_coherent_inputs",
        lambda _: (
            _DummySnapshot(),
            [_DummySnapshot() for _ in range(17)],
            candidate,
            references,
        ),
    )
    observed, created = analysis.analyze(
        project_root=Path(plan["project_root"]),
        run_root=Path(plan["run_root"]),
    )
    assert created is False
    assert observed == persisted


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="Linux ext4 persisted-analysis namespace regression",
)
def test_persisted_analysis_rejects_candidate_package_aba_restoration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_open_package = gate._open_package
    plan, candidate, references, persisted = _existing_analysis_harness(
        tmp_path,
        monkeypatch,
    )
    run_root = Path(plan["run_root"])
    staging = run_root / gate.STAGING_DIRECTORY
    inputs = run_root / "candidate-inputs"
    staging.mkdir(mode=0o700)
    inputs.mkdir()
    first, _ = gate._stage_package(
        staging_parent=staging,
        basename="analysis-candidate-a",
        members={"a": b"A", "b": b"2", "c": b"3"},
    )
    second, _ = gate._stage_package(
        staging_parent=staging,
        basename="analysis-candidate-b",
        members={"a": b"B", "b": b"5", "c": b"6"},
    )
    original = inputs / "subject-result"
    replacement = inputs / "replacement-result"
    gate._rename_noreplace(first, original)
    gate._rename_noreplace(second, replacement)
    candidate_snapshot = real_open_package(
        original,
        expected_names=frozenset({"a", "b", "c"}),
        namespace_root=run_root,
    )
    candidate_identity = (
        candidate_snapshot.observed.st_dev,
        candidate_snapshot.observed.st_ino,
    )
    monkeypatch.setattr(
        analysis,
        "_coherent_inputs",
        lambda _: (
            _DummySnapshot(),
            [candidate_snapshot],
            candidate,
            references,
        ),
    )
    attacked = False

    def attack_candidate(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal attacked
        del args, kwargs
        if not attacked:
            _same_parent_aba_restore(original, replacement)
            # This ext4 mount also changes the restored package inode ctime.
            # Neutralize that independent leaf signal so analyze() must reject
            # through the retained run-root namespace history.
            candidate_snapshot.observed = original.lstat()
            candidate_snapshot.fingerprint = gate._directory_fingerprint(
                candidate_snapshot.descriptor
            )
            candidate_snapshot.namespace.target_observed = (
                candidate_snapshot.observed
            )
            assert gate._stat_fingerprint(
                os.fstat(candidate_snapshot.descriptor)
            ) == gate._stat_fingerprint(candidate_snapshot.observed)
            assert gate._directory_fingerprint(
                candidate_snapshot.descriptor
            ) == candidate_snapshot.fingerprint
            assert (original / "a").read_bytes() == b"A"
            assert (
                original.lstat().st_dev,
                original.lstat().st_ino,
            ) == candidate_identity
            for child in candidate_snapshot.children.values():
                gate._assert_file_snapshot_at(
                    candidate_snapshot.descriptor,
                    child,
                )
            attacked = True
        return persisted

    monkeypatch.setattr(
        analysis,
        "_validate_analysis_package",
        attack_candidate,
    )
    with pytest.raises(gate.GaugeGate1Error, match="namespace"):
        analysis.analyze(
            project_root=Path(plan["project_root"]),
            run_root=run_root,
        )


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="Linux ext4 persisted-analysis namespace regression",
)
def test_persisted_analysis_rejects_reference_file_aba_restoration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, candidate, references, persisted = _existing_analysis_harness(
        tmp_path,
        monkeypatch,
    )
    reference_root = Path(plan["reference_identity"]["root"])
    reference_parent = reference_root / "records"
    reference_parent.mkdir(parents=True)
    original = reference_parent / "reference.json"
    replacement = reference_parent / "replacement.json"
    original.write_bytes(b'{"identity":"A"}\n')
    replacement.write_bytes(b'{"identity":"B"}\n')
    reference_snapshot = gate._open_multi_file_snapshot(
        [original],
        namespace_root=reference_root,
    )
    reference_identity = (
        reference_snapshot.files[0].observed.st_dev,
        reference_snapshot.files[0].observed.st_ino,
    )
    monkeypatch.setattr(
        analysis,
        "_coherent_inputs",
        lambda _: (
            reference_snapshot,
            [_DummySnapshot()],
            candidate,
            references,
        ),
    )
    attacked = False

    def attack_reference(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal attacked
        del args, kwargs
        if not attacked:
            _same_parent_aba_restore(original, replacement)
            # Force the persisted-analysis path to rely on the pinned
            # reference-parent history, not this mount's leaf ctime signal.
            leaf = reference_snapshot.files[0]
            leaf.observed = original.lstat()
            assert leaf.namespace is not None
            leaf.namespace.target_observed = leaf.observed
            assert gate._stat_fingerprint(os.fstat(leaf.descriptor)) == (
                gate._stat_fingerprint(leaf.observed)
            )
            assert gate._read_descriptor(leaf.descriptor) == leaf.payload
            assert original.read_bytes() == leaf.payload
            assert (
                original.lstat().st_dev,
                original.lstat().st_ino,
            ) == reference_identity
            attacked = True
        return persisted

    monkeypatch.setattr(
        analysis,
        "_validate_analysis_package",
        attack_reference,
    )
    with pytest.raises(gate.GaugeGate1Error, match="namespace"):
        analysis.analyze(
            project_root=Path(plan["project_root"]),
            run_root=Path(plan["run_root"]),
        )


@pytest.mark.parametrize("substitute", ["candidate", "reference"])
def test_existing_analysis_rejects_coherent_cell_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    substitute: str,
) -> None:
    plan, candidate, references, _ = _existing_analysis_harness(
        tmp_path, monkeypatch
    )
    candidate = dict(candidate)
    references = {model: dict(values) for model, values in references.items()}
    job_id = next(iter(candidate))
    if substitute == "candidate":
        candidate[job_id] = 0.1
    else:
        references[analysis.PRIMARY_REFERENCE][job_id] = 0.95
    monkeypatch.setattr(
        analysis,
        "_coherent_inputs",
        lambda _: (
            _DummySnapshot(),
            [_DummySnapshot() for _ in range(17)],
            candidate,
            references,
        ),
    )
    with pytest.raises(gate.GaugeGate1Error, match="recomputation"):
        analysis.analyze(
            project_root=Path(plan["project_root"]),
            run_root=Path(plan["run_root"]),
        )


def test_existing_analysis_rejects_environment_drift_before_reuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, _, _, _ = _existing_analysis_harness(tmp_path, monkeypatch)
    monkeypatch.setattr(
        gate,
        "environment_identity",
        lambda: {"environment": "drifted"},
    )
    with pytest.raises(gate.GaugeGate1Error, match="environment"):
        analysis.analyze(
            project_root=Path(plan["project_root"]),
            run_root=Path(plan["run_root"]),
        )


def test_existing_analysis_rejects_source_drift_before_reuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, _, _, _ = _existing_analysis_harness(tmp_path, monkeypatch)
    monkeypatch.setattr(
        gate,
        "_source_snapshot",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            gate.GaugeGate1Error("source drift")
        ),
    )
    with pytest.raises(gate.GaugeGate1Error, match="source drift"):
        analysis.analyze(
            project_root=Path(plan["project_root"]),
            run_root=Path(plan["run_root"]),
        )


def test_disk_guard_rejects_low_water(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Fake:
        f_bavail = 1
        f_frsize = 1

    monkeypatch.setattr(os, "fstatvfs", lambda _: Fake())
    with pytest.raises(gate.ResourceUnavailable, match="below"):
        gate._probe_disk(tmp_path)


def test_remote_owner_is_never_stolen_automatically() -> None:
    assert gate._owner_is_live(
        {
            "host": "different-host",
            "pid": 999_999,
            "boot_id": "00000000-0000-0000-0000-000000000001",
            "start_ticks": 1,
        }
    )


def test_current_owner_is_live() -> None:
    assert gate._owner_is_live(gate.project_gpu_leases.owner_identity())


def test_no_runner_function_can_schedule_reference_model() -> None:
    jobs = gate._candidate_manifest()
    assert all(job.model == gate.MODEL_NAME for job in jobs)
    assert not hasattr(gate, "execute_reference_job")
    assert not hasattr(gate, "run_comparator")
