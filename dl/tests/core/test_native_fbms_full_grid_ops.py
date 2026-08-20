from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmark import native_fbms_full_grid_ops as ops
from benchmark import native_transfer_audit as audit_core


def _key(
    *,
    dataset: str = ops.CHO2017,
    subject: int = 16,
    fold: int = 0,
    seed: int = 7,
    condition: str = ops.EXACT_CONDITIONS[0],
) -> audit_core.TransferRecordKey:
    return audit_core.TransferRecordKey(
        dataset=dataset,
        subject=subject,
        fold=fold,
        seed=seed,
        condition=condition,
    )


def _validated_record(
    root: Path,
    *,
    key: audit_core.TransferRecordKey | None = None,
    source_manifest: dict[str, str] | None = None,
    corpus_hash: str = ops.PINNED_CORPUS_SHA256,
) -> audit_core.ValidatedTransferArtifact:
    record_key = _key() if key is None else key
    root.mkdir(parents=True, exist_ok=True)
    provenance = {
        "environment": ops.full_grid_audit.PINNED_EXECUTION_ENVIRONMENT,
        "protocol": {"train_config": {"device": "cuda"}},
        "source_checkpoint": {
            "file_sha256": ops.PINNED_CHECKPOINT_FILE_SHA256,
            "state_sha256": ops.PINNED_CHECKPOINT_STATE_SHA256,
            "corpus": {"sha256": corpus_hash},
            "pretraining": {
                "partition": {"sha256": ops.PINNED_PARTITION_SHA256},
                "state_hashes": {
                    "initial": ops.PINNED_SOURCE_INITIAL_STATE_SHA256
                },
            },
        }
    }
    (root / ops.transfer.PROVENANCE_FILENAME).write_text(
        json.dumps(provenance), encoding="utf-8"
    )
    values = np.asarray([0], dtype=np.int64)
    return audit_core.ValidatedTransferArtifact(
        key=record_key,
        path=root,
        predictions_file_sha256="1" * 64,
        provenance_file_sha256="2" * 64,
        checkpoint_file_sha256=ops.PINNED_CHECKPOINT_FILE_SHA256,
        cache_file_sha256="3" * 64,
        source_code_hashes=tuple(
            sorted((ops.PINNED_SOURCE_MANIFEST if source_manifest is None else source_manifest).items())
        ),
        test_rows=values,
        test_labels=values,
        predicted_labels=values,
        probabilities=np.asarray([[1.0, 0.0]], dtype=np.float64),
        sessions=np.asarray(["session"]),
        runs=np.asarray(["run"]),
    )


def _screen_with_current_caches(
    tmp_path: Path,
) -> tuple[SimpleNamespace, Path, dict[tuple[str, int], Path], str]:
    cache_root = tmp_path / "cache"
    records = []
    paths: dict[tuple[str, int], Path] = {}
    manifest: dict[str, str] = {}
    for dataset in ops.EXACT_DATASETS:
        for subject in ops.EXACT_SUBJECTS[dataset]:
            cache = ops.transfer_core._native_cache_path(
                cache_root, dataset, subject
            )
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(f"cache-{dataset}-{subject}".encode())
            digest = ops._sha256_file(cache)
            paths[(dataset, subject)] = cache
            manifest[f"{dataset}:s{subject}"] = digest
            result = tmp_path / "screen" / dataset / f"s{subject:03d}"
            result.mkdir(parents=True)
            (result / ops.transfer.PROVENANCE_FILENAME).write_text(
                json.dumps(
                    {
                        "target": {
                            "cache_path": (
                                f"/historical/eegthingy/data_cache/eeg-mi-cache-v2/"
                                f"native/{dataset}/subject_{subject:03d}.npz"
                            ),
                            "cache_file_sha256": digest,
                        }
                    }
                ),
                encoding="utf-8",
            )
            records.append(
                SimpleNamespace(
                    key=_key(dataset=dataset, subject=subject),
                    path=result,
                    cache_file_sha256=digest,
                )
            )
    manifest_hash = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return SimpleNamespace(records=tuple(records)), cache_root, paths, manifest_hash


def test_exact_grid_count_dimensions_and_uniform_names() -> None:
    grid = ops.exact_full_grid()

    assert len(grid) == 8_675
    assert ops.EXPECTED_RECORD_COUNTS == {
        ops.CHO2017: 4_625,
        ops.PHYSIONET_MI: 4_050,
    }
    assert sum(key.dataset == ops.CHO2017 for key in grid) == 4_625
    assert sum(key.dataset == ops.PHYSIONET_MI for key in grid) == 4_050
    names = {(key.dataset, ops.record_directory_name(key)) for key in grid}
    assert len(names) == len(grid)
    assert (
        ops.CHO2017,
        "s016_f0_seed7_pretrained_cardinal_fbms",
    ) in names
    assert (
        ops.PHYSIONET_MI,
        "s054_f2_seed47_pretrained_indexed_fbmsnet_spherical_spline",
    ) in names
    assert all("_f" in name and "_seed" in name for _, name in names)


def test_grid_identity_is_shared_with_full_grid_auditor() -> None:
    assert ops.full_grid_audit.EXPECTED_RECORD_COUNT == ops.EXPECTED_RECORD_COUNT
    assert (
        ops.full_grid_audit.FULL_RECORD_NAME_TEMPLATE
        == "s{subject:03d}_f{fold}_seed{seed}_{condition}"
    )
    assert ops.full_grid_audit.AUDIT_CONTRACT.conditions == ops.EXACT_CONDITIONS


def test_runner_has_no_grid_or_worker_override_surface(tmp_path: Path) -> None:
    parser = ops.build_parser()
    destinations = {action.dest for action in parser._actions}

    assert "workers" not in destinations
    assert "dataset" not in destinations
    assert "subjects" not in destinations
    assert "folds" not in destinations
    assert "seeds" not in destinations
    assert "conditions" not in destinations
    assert "checkpoint_sha256" not in destinations
    assert "device" not in destinations
    assert ops.MAX_GPU_WORKERS == 4

    with pytest.raises(SystemExit):
        parser.parse_args(["--workers", "8"])


def test_source_manifest_is_exact_and_rejects_optional_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert len(ops.PINNED_SOURCE_MANIFEST) == 11
    assert "requirements.txt" not in ops.PINNED_SOURCE_MANIFEST
    assert (
        ops.PINNED_SOURCE_MANIFEST["eeg_mi/native_fbms_transfer.py"]
        == ops.PINNED_TRANSFER_SOURCE_SHA256
    )
    monkeypatch.setattr(
        ops.transfer,
        "_source_file_hashes",
        lambda: dict(ops.PINNED_SOURCE_MANIFEST),
    )
    ops._validate_current_source_manifest()

    stale = dict(ops.PINNED_SOURCE_MANIFEST)
    stale["requirements.txt"] = "a" * 64
    monkeypatch.setattr(ops.transfer, "_source_file_hashes", lambda: stale)
    with pytest.raises(ops.FullGridOperationsError, match="extra=.*requirements.txt"):
        ops._validate_current_source_manifest()


def test_current_environment_preflight_rejects_device_or_package_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ops.transfer_core,
        "_environment_record",
        lambda unused: copy.deepcopy(
            ops.full_grid_audit.PINNED_EXECUTION_ENVIRONMENT
        ),
    )
    ops._validate_current_execution_environment()

    stale = copy.deepcopy(ops.full_grid_audit.PINNED_EXECUTION_ENVIRONMENT)
    stale["requested_device"] = "cpu"
    stale["packages"]["torch"] = "2.11.1+cu128"
    monkeypatch.setattr(
        ops.transfer_core, "_environment_record", lambda unused: stale
    )
    with pytest.raises(ops.FullGridOperationsError, match="changed=.*packages"):
        ops._validate_current_execution_environment()


def test_current_target_cache_preflight_accepts_all_91_matching_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    screen, cache_root, unused, manifest_hash = _screen_with_current_caches(tmp_path)
    del unused
    monkeypatch.setattr(
        ops.full_grid_audit,
        "PINNED_TARGET_CACHE_MANIFEST_SHA256",
        manifest_hash,
    )

    ops._validate_current_target_caches(screen, cache_root=cache_root)


def test_current_target_cache_preflight_rejects_hash_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    screen, cache_root, paths, manifest_hash = _screen_with_current_caches(tmp_path)
    monkeypatch.setattr(
        ops.full_grid_audit,
        "PINNED_TARGET_CACHE_MANIFEST_SHA256",
        manifest_hash,
    )
    paths[(ops.CHO2017, 16)].write_bytes(b"rebuilt-but-different-cache")

    with pytest.raises(ops.FullGridOperationsError, match="SHA-256 differs"):
        ops._validate_current_target_caches(screen, cache_root=cache_root)


def test_current_target_cache_preflight_rejects_missing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    screen, cache_root, paths, manifest_hash = _screen_with_current_caches(tmp_path)
    monkeypatch.setattr(
        ops.full_grid_audit,
        "PINNED_TARGET_CACHE_MANIFEST_SHA256",
        manifest_hash,
    )
    paths[(ops.PHYSIONET_MI, 54)].unlink()

    with pytest.raises(FileNotFoundError, match="available real file"):
        ops._validate_current_target_caches(screen, cache_root=cache_root)


def test_current_target_cache_preflight_rejects_relative_historical_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    screen, cache_root, unused, manifest_hash = _screen_with_current_caches(tmp_path)
    del unused
    monkeypatch.setattr(
        ops.full_grid_audit,
        "PINNED_TARGET_CACHE_MANIFEST_SHA256",
        manifest_hash,
    )
    first = screen.records[0]
    provenance_path = first.path / ops.transfer.PROVENANCE_FILENAME
    provenance = json.loads(provenance_path.read_text())
    provenance["target"]["cache_path"] = "relative/subject_016.npz"
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(ops.FullGridOperationsError, match="not absolute"):
        ops._validate_current_target_caches(screen, cache_root=cache_root)


def test_command_uses_fresh_module_process_and_all_pins(tmp_path: Path) -> None:
    job = ops.GridJob(
        key=_key(fold=3, seed=37, condition=ops.EXACT_CONDITIONS[-1]),
        output=tmp_path / "output",
    )
    command = ops.command_for_job(
        job,
        cache_root=tmp_path / "cache",
        checkpoint_path=tmp_path / "checkpoint.pt",
    )

    assert command[:3] == (
        sys.executable,
        "-m",
        "eeg_mi.native_fbms_transfer",
    )
    assert command[command.index("--checkpoint-sha256") + 1] == (
        ops.PINNED_CHECKPOINT_FILE_SHA256
    )
    assert command[command.index("--fold") + 1] == "3"
    assert command[command.index("--seed") + 1] == "37"
    assert command[command.index("--output") + 1] == str(job.output)


def test_pinned_record_validation_rejects_stale_corpus_and_manifest(
    tmp_path: Path,
) -> None:
    valid = _validated_record(tmp_path / "valid")
    ops._validate_pinned_record(valid)

    stale_corpus = _validated_record(
        tmp_path / "stale_corpus", corpus_hash="0" * 64
    )
    with pytest.raises(ops.FullGridOperationsError, match="corpus"):
        ops._validate_pinned_record(stale_corpus)

    stale_manifest = dict(ops.PINNED_SOURCE_MANIFEST)
    stale_manifest["eeg_mi/native_fbms_transfer.py"] = "0" * 64
    record = _validated_record(
        tmp_path / "stale_manifest", source_manifest=stale_manifest
    )
    with pytest.raises(ops.FullGridOperationsError, match="source manifest"):
        ops._validate_pinned_record(record)


def test_empty_resume_plan_covers_exact_grid_without_scoring(tmp_path: Path) -> None:
    cho = tmp_path / "cho"
    physionet = tmp_path / "physionet"
    cho.mkdir()
    physionet.mkdir()

    plan = ops.build_resume_plan(cho_root=cho, physionet_root=physionet)
    manifest = ops.plan_manifest(plan)

    assert not plan.complete
    assert len(plan.completed) == 0
    assert len(plan.pending) == 8_675
    assert manifest["scores_computed"] is False
    assert manifest["worker_count"] == 4
    assert manifest["pending_by_dataset"] == {
        ops.CHO2017: 4_625,
        ops.PHYSIONET_MI: 4_050,
    }
    assert 12.0 < manifest["ideal_pending_wall_hours_at_four_workers"] < 13.0


def test_resume_delegates_present_records_to_full_grid_auditor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cho = tmp_path / "cho"
    physionet = tmp_path / "physionet"
    cho.mkdir()
    physionet.mkdir()
    key = _key()
    output = cho / ops.record_directory_name(key)
    output.mkdir()
    artifact = _validated_record(tmp_path / "artifact", key=key)
    calls: list[tuple[Path, Path]] = []

    def fake_full_audit(*, cho_root: Path, physionet_root: Path) -> SimpleNamespace:
        calls.append((cho_root, physionet_root))
        return SimpleNamespace(records=(artifact,))

    monkeypatch.setattr(
        ops.full_grid_audit, "audit_locked_fbms_full_grid", fake_full_audit
    )
    monkeypatch.setattr(ops, "_validate_pinned_record", lambda unused: None)

    plan = ops.build_resume_plan(cho_root=cho, physionet_root=physionet)

    assert calls == [(cho, physionet)]
    assert [job.key for job in plan.completed] == [key]
    assert len(plan.pending) == 8_674


def test_resume_rejects_unknown_output_and_classifies_exact_transients(
    tmp_path: Path,
) -> None:
    cho = tmp_path / "cho"
    physionet = tmp_path / "physionet"
    cho.mkdir()
    physionet.mkdir()
    name = ops.record_directory_name(_key())
    lock = cho / f".{name}.native-transfer.lock"
    staging = cho / f".{name}.staging-powercut"
    lock.write_text("pid=123\n", encoding="utf-8")
    staging.mkdir()

    plan = ops.build_resume_plan(cho_root=cho, physionet_root=physionet)
    assert {(item.path, item.kind) for item in plan.recovery_artifacts} == {
        (lock, "lock"),
        (staging, "staging"),
    }

    (cho / "unrecognized-output").mkdir()
    with pytest.raises(ops.FullGridOperationsError, match="unknown outputs"):
        ops.build_resume_plan(cho_root=cho, physionet_root=physionet)


def test_recovery_removes_only_classified_writer_transients(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    lock = root / ".record.native-transfer.lock"
    staging = root / ".record.staging-dead"
    keep = root / "keep"
    lock.write_text(
        f"pid=99999999 output={(root / 'record').resolve()}\n",
        encoding="utf-8",
    )
    staging.mkdir()
    (staging / "partial").write_text("partial", encoding="utf-8")
    keep.write_text("keep", encoding="utf-8")
    plan = ops.ResumePlan(
        completed=(),
        pending=(),
        recovery_artifacts=(
            ops.RecoveryArtifact(lock, "lock"),
            ops.RecoveryArtifact(staging, "staging"),
        ),
    )

    ops.recover_interrupted_writes(plan)

    assert not lock.exists()
    assert not staging.exists()
    assert keep.read_text(encoding="utf-8") == "keep"


def test_recovery_refuses_a_live_manual_writer(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    lock = root / ".record.native-transfer.lock"
    staging = root / ".record.staging-active"
    lock.write_text(
        f"pid={os.getpid()} output={(root / 'record').resolve()}\n",
        encoding="utf-8",
    )
    staging.mkdir()
    plan = ops.ResumePlan(
        (),
        (),
        (
            ops.RecoveryArtifact(lock, "lock"),
            ops.RecoveryArtifact(staging, "staging"),
        ),
    )

    with pytest.raises(ops.FullGridOperationsError, match="active writer"):
        ops.recover_interrupted_writes(plan)

    assert lock.exists()
    assert staging.exists()


def test_recovery_keeps_stale_lock_if_staging_cleanup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    lock = root / ".record.native-transfer.lock"
    staging = root / ".record.staging-dead"
    lock.write_text(
        f"pid=99999999 output={(root / 'record').resolve()}\n",
        encoding="utf-8",
    )
    staging.mkdir()
    plan = ops.ResumePlan(
        (),
        (),
        (
            ops.RecoveryArtifact(lock, "lock"),
            ops.RecoveryArtifact(staging, "staging"),
        ),
    )

    def fail_cleanup(unused: Path) -> None:
        del unused
        raise OSError("simulated staging cleanup failure")

    monkeypatch.setattr(ops.shutil, "rmtree", fail_cleanup)
    with pytest.raises(OSError, match="simulated"):
        ops.recover_interrupted_writes(plan)

    assert staging.exists()
    assert lock.exists()


def test_atomic_seed_copy_is_byte_identical_and_preserves_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    predictions = b"locked prediction bytes\x00\x01"
    provenance = b'{"locked":true}\n'
    (source / ops.transfer.PREDICTIONS_FILENAME).write_bytes(predictions)
    (source / ops.transfer.PROVENANCE_FILENAME).write_bytes(provenance)
    record = _validated_record(tmp_path / "metadata")
    # Point the validated identity at byte fixtures with matching hashes.
    record = audit_core.ValidatedTransferArtifact(
        **{
            **record.__dict__,
            "path": source,
            "predictions_file_sha256": ops._sha256_file(
                source / ops.transfer.PREDICTIONS_FILENAME
            ),
            "provenance_file_sha256": ops._sha256_file(
                source / ops.transfer.PROVENANCE_FILENAME
            ),
        }
    )
    destination = tmp_path / "full" / "record"
    destination.parent.mkdir()

    ops._copy_screen_record_atomically(record, destination=destination)

    assert (source / ops.transfer.PREDICTIONS_FILENAME).read_bytes() == predictions
    assert (source / ops.transfer.PROVENANCE_FILENAME).read_bytes() == provenance
    assert (destination / ops.transfer.PREDICTIONS_FILENAME).read_bytes() == predictions
    assert (destination / ops.transfer.PROVENANCE_FILENAME).read_bytes() == provenance
    assert not list(destination.parent.glob(".*.staging-*"))


def test_seed_orchestration_uses_uniform_name_and_reaudits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "old_screen" / "s016_pretrained_cardinal_fbms"
    source.mkdir(parents=True)
    (source / ops.transfer.PREDICTIONS_FILENAME).write_bytes(b"predictions")
    (source / ops.transfer.PROVENANCE_FILENAME).write_bytes(b"provenance")
    record = _validated_record(tmp_path / "metadata")
    record = audit_core.ValidatedTransferArtifact(
        **{
            **record.__dict__,
            "path": source,
            "predictions_file_sha256": ops._sha256_file(
                source / ops.transfer.PREDICTIONS_FILENAME
            ),
            "provenance_file_sha256": ops._sha256_file(
                source / ops.transfer.PROVENANCE_FILENAME
            ),
        }
    )
    cho = tmp_path / "full_cho"
    physionet = tmp_path / "full_physionet"
    cho.mkdir()
    physionet.mkdir()
    empty = ops.ResumePlan((), (), ())
    destination = cho / ops.record_directory_name(record.key)
    seeded = ops.ResumePlan(
        (ops.GridJob(record.key, destination),),
        (),
        (),
    )
    audits = iter((empty, seeded))
    monkeypatch.setattr(
        ops,
        "_audit_seed_screen",
        lambda **unused: SimpleNamespace(records=(record,)),
    )
    monkeypatch.setattr(ops, "build_resume_plan", lambda **unused: next(audits))

    result = ops.seed_screen_into_full_roots(
        screen_cho_root=tmp_path / "old_screen",
        screen_physionet_root=tmp_path / "old_physionet",
        cache_root=tmp_path / "cache",
        cho_root=cho,
        physionet_root=physionet,
    )

    assert result is seeded
    assert destination.name == "s016_f0_seed7_pretrained_cardinal_fbms"
    assert (destination / ops.transfer.PREDICTIONS_FILENAME).read_bytes() == b"predictions"
    assert (source / ops.transfer.PREDICTIONS_FILENAME).read_bytes() == b"predictions"


def test_runner_claim_remains_held_by_inherited_child(tmp_path: Path) -> None:
    ops_root = tmp_path / "ops"
    ops_root.mkdir()
    with ops._RunnerClaim(ops_root) as claim:
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(0.35)"],
            close_fds=True,
            pass_fds=(claim.fileno(),),
        )
    try:
        with pytest.raises(ops.FullGridOperationsError, match="owns"):
            with ops._RunnerClaim(ops_root):
                pass
    finally:
        child.wait(timeout=5.0)
    with ops._RunnerClaim(ops_root):
        pass


def test_execute_failure_stops_without_spinning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = (tmp_path / "failed.log").open("wb")
    job = ops.GridJob(_key(), tmp_path / "missing-output")

    class FailedProcess:
        pid = 99
        returncode = 1

        @staticmethod
        def poll() -> int:
            return 1

    def fake_spawn(*unused: object, **unused_kwargs: object) -> ops._ActiveJob:
        del unused, unused_kwargs
        return ops._ActiveJob(  # type: ignore[arg-type]
            job,
            FailedProcess(),
            tmp_path / "failed.log",
            log,
            time.monotonic(),
        )

    monkeypatch.setattr(ops, "_spawn", fake_spawn)
    started = time.monotonic()
    with pytest.raises(ops.FullGridOperationsError, match="at least one record failed"):
        ops.execute_pending(
            ops.ResumePlan((), (job,), ()),
            cache_root=tmp_path,
            checkpoint_path=tmp_path / "checkpoint",
            ops_root=tmp_path,
            claim_fd=1,
        )
    assert time.monotonic() - started < 1.0


def test_execute_times_out_wedged_child_and_journals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = (tmp_path / "timeout.log").open("wb")
    job = ops.GridJob(_key(), tmp_path / "missing-output")

    class WedgedProcess:
        pid = 100
        returncode: int | None = None

        def poll(self) -> int | None:
            return self.returncode

        def send_signal(self, unused: int) -> None:
            del unused
            self.returncode = -15

        def wait(self, timeout: float) -> int:
            del timeout
            assert self.returncode is not None
            return self.returncode

        def kill(self) -> None:
            self.returncode = -9

    process = WedgedProcess()

    def fake_spawn(*unused: object, **unused_kwargs: object) -> ops._ActiveJob:
        del unused, unused_kwargs
        return ops._ActiveJob(  # type: ignore[arg-type]
            job,
            process,
            tmp_path / "timeout.log",
            log,
            time.monotonic() - 10.0,
        )

    monkeypatch.setattr(ops, "_spawn", fake_spawn)
    monkeypatch.setattr(ops, "JOB_TIMEOUT_SECONDS", 1.0)
    with pytest.raises(ops.FullGridOperationsError, match="at least one record failed"):
        ops.execute_pending(
            ops.ResumePlan((), (job,), ()),
            cache_root=tmp_path,
            checkpoint_path=tmp_path / "checkpoint",
            ops_root=tmp_path,
            claim_fd=1,
        )

    events = [json.loads(line) for line in (tmp_path / "journal.jsonl").read_text().splitlines()]
    assert [event["event"] for event in events] == [
        "record_started",
        "record_timed_out",
    ]
    assert events[-1]["timeout_seconds"] == 1.0
