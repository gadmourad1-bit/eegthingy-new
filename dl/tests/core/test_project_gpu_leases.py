from __future__ import annotations

import multiprocessing
import os
import queue
import threading
from pathlib import Path

import pytest

from benchmark import project_gpu_leases as leases


def _uuid(index: int) -> str:
    return f"GPU-00000000-0000-0000-0000-{index:012x}"


def _roots(tmp_path: Path) -> tuple[Path, Path]:
    run = tmp_path / "run"
    run.mkdir()
    return tmp_path, run


def _acquire(tmp_path: Path, index: int) -> leases.GPULease:
    project, run = _roots(tmp_path) if not (tmp_path / "run").exists() else (
        tmp_path,
        tmp_path / "run",
    )
    return leases.acquire_gpu_lease(
        project_root=project,
        run_root=run,
        plan_sha256="a" * 64,
        gpu_uuid=_uuid(index),
        track_scope="test-track",
    )


def _concurrent_acquire_worker(
    project: str,
    run: str,
    index: int,
    start,
    release,
    results,
) -> None:
    start.wait(10.0)
    try:
        lease = leases.acquire_gpu_lease(
            project_root=Path(project),
            run_root=Path(run),
            plan_sha256="a" * 64,
            gpu_uuid=_uuid(index),
            track_scope="concurrency-test",
        )
    except leases.GPUWorkerUnavailable:
        results.put(("unavailable", index))
        return
    except Exception as error:  # pragma: no cover - returned to the parent
        results.put(("error", index, type(error).__name__, str(error)))
        return
    results.put(("acquired", index))
    release.wait(10.0)
    leases.release_gpu_lease(lease)


def _concurrent_release_worker(lease, start, results) -> None:
    start.wait(10.0)
    results.put(("attempting",))
    try:
        leases.release_gpu_lease(lease)
    except Exception as error:  # pragma: no cover - returned to the parent
        results.put(("error", type(error).__name__, str(error)))
        return
    results.put(("released",))


def test_shared_registry_enforces_distinct_uuid_and_global_cap_three(
    tmp_path: Path,
) -> None:
    acquired = [_acquire(tmp_path, index) for index in range(1, 4)]
    with pytest.raises(leases.GPUWorkerUnavailable, match="cap"):
        _acquire(tmp_path, 4)
    with pytest.raises(leases.GPUWorkerUnavailable, match="already"):
        leases.acquire_gpu_lease(
            project_root=tmp_path,
            run_root=tmp_path / "run",
            plan_sha256="a" * 64,
            gpu_uuid=_uuid(1),
            track_scope="another-track",
        )
    for lease in acquired:
        leases.release_gpu_lease(lease)
    assert {
        path.name
        for path in (
            tmp_path / leases.REGISTRY_DIRECTORY
        ).iterdir()
    } == {leases.REGISTRY_FENCE_FILENAME}


def test_four_processes_contend_atomically_for_three_global_slots(
    tmp_path: Path,
) -> None:
    project, run = _roots(tmp_path)
    context = multiprocessing.get_context("fork")
    start = context.Event()
    release = context.Event()
    results = context.Queue()
    processes = [
        context.Process(
            target=_concurrent_acquire_worker,
            args=(
                str(project),
                str(run),
                index,
                start,
                release,
                results,
            ),
        )
        for index in range(1, 5)
    ]
    for process in processes:
        process.start()
    start.set()
    try:
        observed = [results.get(timeout=15.0) for _ in processes]
    finally:
        release.set()
        for process in processes:
            process.join(timeout=15.0)
    assert all(process.exitcode == 0 for process in processes)
    assert sum(value[0] == "acquired" for value in observed) == 3
    assert sum(value[0] == "unavailable" for value in observed) == 1
    assert not [value for value in observed if value[0] == "error"]


def test_malformed_active_lease_fails_closed_without_stealing(
    tmp_path: Path,
) -> None:
    project, run = _roots(tmp_path)
    with leases._registry_lock(project):
        pass
    path = project / leases.REGISTRY_DIRECTORY / f"{_uuid(1)}.json"
    leases._write_json_exclusive(path, {})
    with pytest.raises(leases.ProjectGPULeaseError, match="schema"):
        leases.acquire_gpu_lease(
            project_root=project,
            run_root=run,
            plan_sha256="a" * 64,
            gpu_uuid=_uuid(2),
            track_scope="test-track",
        )
    assert path.exists()


def test_fence_bootstrap_power_cut_never_publishes_torn_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, run = _roots(tmp_path)
    original = leases._write_json_exclusive

    def torn(path: Path, value) -> None:
        if (
            leases.FORENSIC_DIRECTORY in path.parts
            and "registry-fence" in path.name
        ):
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                os.write(descriptor, b"{")
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            raise OSError("synthetic power cut")
        original(path, value)

    monkeypatch.setattr(leases, "_write_json_exclusive", torn)
    with pytest.raises(OSError, match="power cut"):
        with leases._registry_lock(project):
            pass
    fence = project / leases.REGISTRY_DIRECTORY / leases.REGISTRY_FENCE_FILENAME
    assert not fence.exists()
    monkeypatch.setattr(leases, "_write_json_exclusive", original)
    lease = leases.acquire_gpu_lease(
        project_root=project,
        run_root=run,
        plan_sha256="a" * 64,
        gpu_uuid=_uuid(1),
        track_scope="test-track",
    )
    leases.release_gpu_lease(lease)


def test_lease_power_cut_never_publishes_torn_active_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, run = _roots(tmp_path)
    with leases._registry_lock(project):
        pass
    original = leases._write_json_exclusive

    def torn(path: Path, value) -> None:
        if (
            leases.FORENSIC_DIRECTORY in path.parts
            and path.name.startswith("GPU-")
        ):
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                os.write(descriptor, b"{")
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            raise OSError("synthetic lease power cut")
        original(path, value)

    monkeypatch.setattr(leases, "_write_json_exclusive", torn)
    with pytest.raises(OSError, match="lease power cut"):
        leases.acquire_gpu_lease(
            project_root=project,
            run_root=run,
            plan_sha256="a" * 64,
            gpu_uuid=_uuid(1),
            track_scope="test-track",
        )
    active = project / leases.REGISTRY_DIRECTORY / f"{_uuid(1)}.json"
    assert not active.exists()
    monkeypatch.setattr(leases, "_write_json_exclusive", original)
    lease = leases.acquire_gpu_lease(
        project_root=project,
        run_root=run,
        plan_sha256="a" * 64,
        gpu_uuid=_uuid(1),
        track_scope="test-track",
    )
    leases.release_gpu_lease(lease)


def test_registry_fence_replacement_while_locked_is_detected(
    tmp_path: Path,
) -> None:
    project, _run = _roots(tmp_path)
    with pytest.raises(leases.RegistryFenceLost):
        with leases._registry_lock(project) as (root, _descriptor):
            path = root / leases.REGISTRY_FENCE_FILENAME
            path.unlink()
            leases._write_json_exclusive(
                path, leases._registry_value(project)
            )


def test_release_replacement_interleaving_cannot_archive_wrong_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lease = _acquire(tmp_path, 1)
    original_rename = leases._atomic_rename_noreplace
    replaced = {"done": False}

    def replace_then_rename(source: Path, destination: Path) -> None:
        if source == lease.path and not replaced["done"]:
            replaced["done"] = True
            hidden_root = leases._forensic_directory(
                tmp_path, "stale"
            )
            hidden = hidden_root / f"interleaved-{lease.nonce}.json"
            original_rename(source, hidden)
            leases._write_json_exclusive(source, lease.value)
        original_rename(source, destination)

    monkeypatch.setattr(
        leases, "_atomic_rename_noreplace", replace_then_rename
    )
    with pytest.raises(leases.ProjectGPULeaseError, match="snapshot"):
        leases.release_gpu_lease(lease)
    assert replaced["done"] is True


def test_symlinked_project_or_run_ancestor_is_rejected(
    tmp_path: Path,
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    run = real / "run"
    run.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(leases.ProjectGPULeaseError, match="symlink"):
        leases.acquire_gpu_lease(
            project_root=alias,
            run_root=alias / "run",
            plan_sha256="a" * 64,
            gpu_uuid=_uuid(1),
            track_scope="test-track",
        )


def test_assert_and_receipt_bind_exact_project_run_plan_track_and_uuid(
    tmp_path: Path,
) -> None:
    lease = _acquire(tmp_path, 1)
    leases.assert_gpu_lease(lease)
    receipt = leases.gpu_lease_receipt(lease)
    leases.validate_gpu_lease_receipt(
        receipt,
        project_root=tmp_path,
        run_root=tmp_path / "run",
        plan_sha256="a" * 64,
        gpu_uuid=_uuid(1),
        track_scope="test-track",
    )
    with pytest.raises(leases.ProjectGPULeaseError, match="identity"):
        leases.validate_gpu_lease_receipt(
            receipt,
            project_root=tmp_path,
            run_root=tmp_path / "run",
            plan_sha256="b" * 64,
            gpu_uuid=_uuid(1),
            track_scope="test-track",
        )
    leases.release_gpu_lease(lease)


def test_assert_rejects_same_value_republished_at_a_new_inode(
    tmp_path: Path,
) -> None:
    lease = _acquire(tmp_path, 1)
    hidden = (
        leases._forensic_directory(tmp_path, "stale")
        / f"assert-replaced-{lease.nonce}.json"
    )
    leases._atomic_rename_noreplace(lease.path, hidden)
    leases._write_json_exclusive(lease.path, lease.value)
    with pytest.raises(leases.ProjectGPULeaseError, match="ownership"):
        leases.assert_gpu_lease(lease)


def test_assert_rejects_registry_fence_replacement(
    tmp_path: Path,
) -> None:
    lease = _acquire(tmp_path, 1)
    fence = (
        tmp_path
        / leases.REGISTRY_DIRECTORY
        / leases.REGISTRY_FENCE_FILENAME
    )
    hidden = (
        leases._forensic_directory(tmp_path, "stale")
        / f"assert-fence-replaced-{lease.nonce}.json"
    )
    leases._atomic_rename_noreplace(fence, hidden)
    leases._write_json_exclusive(fence, leases._registry_value(tmp_path))
    with pytest.raises(leases.RegistryFenceLost, match="differs"):
        leases.assert_gpu_lease(lease)


def test_assert_detects_path_replacement_after_descriptor_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lease = _acquire(tmp_path, 1)
    original_open = leases._open_lease_snapshot
    replaced = {"done": False}

    def open_then_replace(path: Path, **kwargs):
        snapshot = original_open(path, **kwargs)
        if path == lease.path and not replaced["done"]:
            replaced["done"] = True
            hidden = (
                leases._forensic_directory(tmp_path, "stale")
                / f"assert-interleaved-{lease.nonce}.json"
            )
            leases._atomic_rename_noreplace(path, hidden)
            leases._write_json_exclusive(path, lease.value)
        return snapshot

    monkeypatch.setattr(leases, "_open_lease_snapshot", open_then_replace)
    with pytest.raises(leases.ProjectGPULeaseError, match="snapshot"):
        leases.assert_gpu_lease(lease)
    assert replaced["done"] is True


def test_acquire_rechecks_fence_immediately_after_active_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, run = _roots(tmp_path)
    with leases._registry_lock(project):
        pass
    original_rename = leases._atomic_rename_noreplace
    replaced = {"done": False}

    def publish_then_replace_fence(source: Path, destination: Path) -> None:
        original_rename(source, destination)
        if (
            destination.parent.name == leases.REGISTRY_DIRECTORY
            and destination.name.startswith("GPU-")
            and not replaced["done"]
        ):
            replaced["done"] = True
            fence = destination.parent / leases.REGISTRY_FENCE_FILENAME
            hidden = (
                leases._forensic_directory(project, "stale")
                / "acquire-post-mutation-fence.json"
            )
            original_rename(fence, hidden)
            leases._write_json_exclusive(
                fence, leases._registry_value(project)
            )

    monkeypatch.setattr(
        leases, "_atomic_rename_noreplace", publish_then_replace_fence
    )
    with pytest.raises(leases.RegistryFenceLost):
        leases.acquire_gpu_lease(
            project_root=project,
            run_root=run,
            plan_sha256="a" * 64,
            gpu_uuid=_uuid(1),
            track_scope="test-track",
        )
    assert replaced["done"] is True


def test_release_rechecks_fence_immediately_after_active_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lease = _acquire(tmp_path, 1)
    original_rename = leases._atomic_rename_noreplace
    replaced = {"done": False}

    def archive_then_replace_fence(source: Path, destination: Path) -> None:
        original_rename(source, destination)
        if source == lease.path and not replaced["done"]:
            replaced["done"] = True
            fence = (
                tmp_path
                / leases.REGISTRY_DIRECTORY
                / leases.REGISTRY_FENCE_FILENAME
            )
            hidden = (
                leases._forensic_directory(tmp_path, "stale")
                / "release-post-mutation-fence.json"
            )
            original_rename(fence, hidden)
            leases._write_json_exclusive(
                fence, leases._registry_value(tmp_path)
            )

    monkeypatch.setattr(
        leases, "_atomic_rename_noreplace", archive_then_replace_fence
    )
    with pytest.raises(leases.RegistryFenceLost):
        leases.release_gpu_lease(lease)
    assert replaced["done"] is True


def test_guard_holds_registry_against_concurrent_release(
    tmp_path: Path,
) -> None:
    lease = _acquire(tmp_path, 1)
    context = multiprocessing.get_context("fork")
    start = context.Event()
    results = context.Queue()
    process = context.Process(
        target=_concurrent_release_worker,
        args=(lease, start, results),
    )
    process.start()
    try:
        with leases.guard_gpu_lease(lease) as receipt:
            leases.validate_gpu_lease_receipt(
                receipt,
                project_root=tmp_path,
                run_root=tmp_path / "run",
                plan_sha256="a" * 64,
                gpu_uuid=_uuid(1),
                track_scope="test-track",
            )
            start.set()
            assert results.get(timeout=10.0) == ("attempting",)
            with pytest.raises(queue.Empty):
                results.get(timeout=0.25)
            assert lease.path.exists()
        outcome = results.get(timeout=10.0)
    finally:
        start.set()
        process.join(timeout=15.0)
    assert process.exitcode == 0
    assert outcome[0] == "error"
    assert "different process" in outcome[2]
    leases.release_gpu_lease(lease)


def test_guard_serializes_legitimate_same_owner_thread_release(
    tmp_path: Path,
) -> None:
    lease = _acquire(tmp_path, 1)
    results: queue.Queue[tuple[str, ...]] = queue.Queue()

    def release_from_same_process() -> None:
        results.put(("attempting",))
        try:
            leases.release_gpu_lease(lease)
        except Exception as error:  # pragma: no cover - reported to main thread
            results.put(("error", type(error).__name__, str(error)))
            return
        results.put(("released",))

    with leases.guard_gpu_lease(lease):
        thread = threading.Thread(target=release_from_same_process)
        thread.start()
        assert results.get(timeout=10.0) == ("attempting",)
        with pytest.raises(queue.Empty):
            results.get(timeout=0.25)
        assert lease.path.exists()
    assert results.get(timeout=10.0) == ("released",)
    thread.join(timeout=10.0)
    assert not thread.is_alive()
    assert {
        path.name
        for path in (
            tmp_path / leases.REGISTRY_DIRECTORY
        ).iterdir()
    } == {leases.REGISTRY_FENCE_FILENAME}


def test_guard_detects_active_lease_replacement_before_unlock(
    tmp_path: Path,
) -> None:
    lease = _acquire(tmp_path, 1)
    hidden = (
        leases._forensic_directory(tmp_path, "stale")
        / f"guard-replaced-{lease.nonce}.json"
    )
    with pytest.raises(leases.ProjectGPULeaseError, match="ownership"):
        with leases.guard_gpu_lease(lease):
            leases._atomic_rename_noreplace(lease.path, hidden)
            leases._write_json_exclusive(lease.path, lease.value)


def test_guard_detects_fence_replacement_before_unlock(
    tmp_path: Path,
) -> None:
    lease = _acquire(tmp_path, 1)
    fence = (
        tmp_path
        / leases.REGISTRY_DIRECTORY
        / leases.REGISTRY_FENCE_FILENAME
    )
    hidden = (
        leases._forensic_directory(tmp_path, "stale")
        / f"guard-fence-{lease.nonce}.json"
    )
    with pytest.raises(leases.RegistryFenceLost):
        with leases.guard_gpu_lease(lease):
            leases._atomic_rename_noreplace(fence, hidden)
            leases._write_json_exclusive(
                fence, leases._registry_value(tmp_path)
            )
