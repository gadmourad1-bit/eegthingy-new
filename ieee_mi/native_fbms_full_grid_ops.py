"""Power-cut-safe operations runner for the exact CardinalFBMS full grid.

This module is deliberately separate from the model, fitter, record writer,
auditor, and decision gate.  It has one operational job: resume or execute the
prespecified development grid with four fresh Python processes, while refusing
to combine records from different checkpoints, source manifests, or protocols.

The runner never computes a score.  On every resume it fully validates every
present two-file record with the family auditor, validates cross-record
semantics, and treats the audited roots themselves as the source of truth.
Journal files and child-process logs live under a separate operations root.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO

from . import native_fbms_transfer as transfer
from . import native_fbms_full_grid_audit as full_grid_audit
from . import native_fbms_transfer_audit as family_audit
from . import native_transfer as transfer_core
from . import native_transfer_audit as audit_core


RUNNER_SCHEMA = "ieee-mi-native-cardinal-fbms-full-grid-ops-v1"
MAX_GPU_WORKERS = 4
JOB_TIMEOUT_SECONDS = 3_600.0

PINNED_CHECKPOINT_FILE_SHA256 = full_grid_audit.PINNED_CHECKPOINT_FILE_SHA256
PINNED_CHECKPOINT_STATE_SHA256 = full_grid_audit.PINNED_CHECKPOINT_STATE_SHA256
PINNED_SOURCE_INITIAL_STATE_SHA256 = (
    full_grid_audit.PINNED_SOURCE_INITIAL_STATE_SHA256
)
PINNED_CORPUS_SHA256 = full_grid_audit.PINNED_CORPUS_SHA256
PINNED_PARTITION_SHA256 = full_grid_audit.PINNED_PARTITION_SHA256
PINNED_TRANSFER_SOURCE_SHA256 = full_grid_audit.PINNED_TRANSFER_SOURCE_SHA256
PINNED_SOURCE_MANIFEST = full_grid_audit.PINNED_SOURCE_MANIFEST

CHO2017 = "cho2017"
PHYSIONET_MI = "physionet_mi"
EXACT_DATASETS: tuple[str, ...] = (CHO2017, PHYSIONET_MI)
EXACT_SUBJECTS = {
    CHO2017: tuple(range(16, 53)),
    PHYSIONET_MI: tuple(range(1, 55)),
}
EXACT_FOLDS = {
    CHO2017: tuple(range(5)),
    PHYSIONET_MI: tuple(range(3)),
}
EXACT_SEEDS: tuple[int, ...] = (7, 17, 27, 37, 47)
EXACT_CONDITIONS: tuple[str, ...] = (
    "pretrained_cardinal_fbms",
    "scratch_cardinal_fbms_canonical_seeded",
    "scratch_cardinal_fbms_native_projected",
    "scratch_fbmsnet_native",
    "pretrained_indexed_fbmsnet_spherical_spline",
)
EXPECTED_RECORD_COUNTS = {
    dataset: (
        len(EXACT_SUBJECTS[dataset])
        * len(EXACT_FOLDS[dataset])
        * len(EXACT_SEEDS)
        * len(EXACT_CONDITIONS)
    )
    for dataset in EXACT_DATASETS
}
EXPECTED_RECORD_COUNT = sum(EXPECTED_RECORD_COUNTS.values())

# Empirical evaluation-only timings from the fully audited fold-0/seed-7
# screen.  They were measured while using four concurrent GPU processes.
EMPIRICAL_SECONDS_PER_RECORD = {
    CHO2017: 35.89483212348104,
    PHYSIONET_MI: 3.524411125114764,
}


class FullGridOperationsError(RuntimeError):
    """The full-grid operational contract cannot be satisfied safely."""


@dataclass(frozen=True)
class GridJob:
    """One exact authorized record and its deterministic output location."""

    key: audit_core.TransferRecordKey
    output: Path


@dataclass(frozen=True)
class RecoveryArtifact:
    """A writer-owned lock or staging directory left by an interrupted job."""

    path: Path
    kind: str


@dataclass(frozen=True)
class ResumePlan:
    """Score-blind result of fully auditing the two exact roots."""

    completed: tuple[GridJob, ...]
    pending: tuple[GridJob, ...]
    recovery_artifacts: tuple[RecoveryArtifact, ...]

    @property
    def complete(self) -> bool:
        return not self.pending


@dataclass
class _ActiveJob:
    job: GridJob
    process: subprocess.Popen[bytes]
    log_path: Path
    log_handle: BinaryIO
    started_monotonic: float


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _assert_exact_scientific_contract() -> None:
    """Fail closed if an imported scientific grid constant has drifted."""

    observed = {
        "subjects": {
            dataset: tuple(transfer.NATIVE_TARGET_DEVELOPMENT_COHORTS[dataset])
            for dataset in EXACT_DATASETS
        },
        "folds": {
            dataset: tuple(transfer.NATIVE_TARGET_FOLDS[dataset])
            for dataset in EXACT_DATASETS
        },
        "seeds": tuple(transfer.FROZEN_DEVELOPMENT_SEEDS),
        "conditions": tuple(transfer.TRANSFER_CONDITIONS),
    }
    expected = {
        "subjects": EXACT_SUBJECTS,
        "folds": EXACT_FOLDS,
        "seeds": EXACT_SEEDS,
        "conditions": EXACT_CONDITIONS,
    }
    if observed != expected:
        raise FullGridOperationsError(
            "imported CardinalFBMS development grid differs from the pinned full grid"
        )
    full_audit_identity = {
        "subjects": dict(full_grid_audit.LOCKED_SUBJECTS),
        "folds": dict(full_grid_audit.LOCKED_FOLDS),
        "seeds": tuple(full_grid_audit.LOCKED_SEEDS),
        "conditions": tuple(full_grid_audit.LOCKED_CONDITIONS),
    }
    if full_audit_identity != expected:
        raise FullGridOperationsError(
            "full-grid auditor identity differs from the pinned operations grid"
        )
    if full_grid_audit.FULL_RECORD_NAME_TEMPLATE != (
        "s{subject:03d}_f{fold}_seed{seed}_{condition}"
    ):
        raise FullGridOperationsError("full-grid auditor record template changed")
    if EXPECTED_RECORD_COUNTS != {CHO2017: 4_625, PHYSIONET_MI: 4_050}:
        raise AssertionError("internal full-grid record counts are inconsistent")
    if EXPECTED_RECORD_COUNT != 8_675:
        raise AssertionError("internal total full-grid record count is inconsistent")


def exact_full_grid() -> tuple[audit_core.TransferRecordKey, ...]:
    """Authorize and return exactly 8,675 development record identities."""

    _assert_exact_scientific_contract()
    result = tuple(
        key
        for dataset in EXACT_DATASETS
        for key in audit_core.development_transfer_grid(
            dataset=dataset,
            subjects=EXACT_SUBJECTS[dataset],
            folds=EXACT_FOLDS[dataset],
            seeds=EXACT_SEEDS,
            conditions=EXACT_CONDITIONS,
            contract=full_grid_audit.AUDIT_CONTRACT,
        )
    )
    if len(result) != EXPECTED_RECORD_COUNT or len(set(result)) != len(result):
        raise AssertionError("authorized full grid has an invalid size or duplicates")
    return result


def record_directory_name(key: audit_core.TransferRecordKey) -> str:
    """Render the one immutable, collision-free full-grid directory name."""

    transfer.validate_target_record(
        key.dataset, key.subject, key.fold, key.condition, key.seed
    )
    return audit_core.record_directory_name(
        key, name_template=full_grid_audit.FULL_RECORD_NAME_TEMPLATE
    )


def _grid_jobs(roots: Mapping[str, Path]) -> tuple[GridJob, ...]:
    if set(roots) != set(EXACT_DATASETS):
        raise FullGridOperationsError("exactly one root per pinned dataset is required")
    jobs = tuple(
        GridJob(key=key, output=roots[key.dataset] / record_directory_name(key))
        for key in exact_full_grid()
    )
    names = {(job.key.dataset, job.output.name) for job in jobs}
    if len(names) != len(jobs):
        raise AssertionError("full-grid record names collide")
    return jobs


def _validate_current_source_manifest() -> None:
    observed = transfer._source_file_hashes()
    if observed != PINNED_SOURCE_MANIFEST:
        missing = sorted(set(PINNED_SOURCE_MANIFEST) - set(observed))
        extra = sorted(set(observed) - set(PINNED_SOURCE_MANIFEST))
        changed = sorted(
            name
            for name in set(observed) & set(PINNED_SOURCE_MANIFEST)
            if observed[name] != PINNED_SOURCE_MANIFEST[name]
        )
        raise FullGridOperationsError(
            "current writer source manifest differs from the audited screen: "
            f"missing={missing}, extra={extra}, changed={changed}"
        )
    if observed["ieee_mi/native_fbms_transfer.py"] != (
        PINNED_TRANSFER_SOURCE_SHA256
    ):
        raise AssertionError("pinned transfer source is inconsistent with manifest")


def _validate_current_execution_environment() -> None:
    """Reject host/device/package drift before producing an incompatible record."""

    observed = transfer_core._environment_record("cuda")
    try:
        encoded = json.dumps(
            observed,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        expected = json.dumps(
            full_grid_audit.PINNED_EXECUTION_ENVIRONMENT,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise FullGridOperationsError(
            "current CUDA execution environment is not canonical JSON"
        ) from error
    digest = hashlib.sha256(encoded).hexdigest()
    if encoded != expected or digest != (
        full_grid_audit.PINNED_EXECUTION_ENVIRONMENT_SHA256
    ):
        changed = sorted(
            name
            for name in set(observed) | set(full_grid_audit.PINNED_EXECUTION_ENVIRONMENT)
            if observed.get(name)
            != full_grid_audit.PINNED_EXECUTION_ENVIRONMENT.get(name)
        )
        raise FullGridOperationsError(
            "current CUDA execution environment differs from the frozen writer "
            f"host: changed={changed}"
        )


def _validate_checkpoint(checkpoint_path: Path) -> None:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint is unavailable: {checkpoint_path}")
    if _sha256_file(checkpoint_path) != PINNED_CHECKPOINT_FILE_SHA256:
        raise FullGridOperationsError("checkpoint bytes differ from the pinned source")
    checkpoint = transfer.load_immutable_native_checkpoint(
        checkpoint_path,
        expected_file_sha256=PINNED_CHECKPOINT_FILE_SHA256,
    )
    if checkpoint.file_sha256 != PINNED_CHECKPOINT_FILE_SHA256:
        raise FullGridOperationsError("checkpoint loader reported a different file hash")
    if checkpoint.state_sha256 != PINNED_CHECKPOINT_STATE_SHA256:
        raise FullGridOperationsError("checkpoint tensor state differs from the pin")
    if checkpoint.corpus.get("sha256") != PINNED_CORPUS_SHA256:
        raise FullGridOperationsError("checkpoint corpus differs from the pin")
    pretraining = checkpoint.pretraining
    partition = pretraining.get("partition")
    state_hashes = pretraining.get("state_hashes")
    if not isinstance(partition, Mapping) or (
        partition.get("sha256") != PINNED_PARTITION_SHA256
    ):
        raise FullGridOperationsError("checkpoint partition differs from the pin")
    if not isinstance(state_hashes, Mapping) or (
        state_hashes.get("initial") != PINNED_SOURCE_INITIAL_STATE_SHA256
    ):
        raise FullGridOperationsError("checkpoint source initialization differs from the pin")


def _nested_mapping(
    value: object, names: Sequence[str], *, context: str
) -> Mapping[str, object]:
    current = value
    for name in names:
        if not isinstance(current, Mapping):
            raise FullGridOperationsError(f"{context} is missing {'.'.join(names)}")
        current = current.get(name)
    if not isinstance(current, Mapping):
        raise FullGridOperationsError(f"{context} is missing {'.'.join(names)}")
    return current


def _validate_pinned_record(
    record: audit_core.ValidatedTransferArtifact,
) -> None:
    """Apply experiment-level pins after the complete family artifact audit."""

    if record.checkpoint_file_sha256 != PINNED_CHECKPOINT_FILE_SHA256:
        raise FullGridOperationsError(
            f"stale checkpoint in existing record {record.path}"
        )
    source_manifest = dict(record.source_code_hashes)
    if source_manifest != PINNED_SOURCE_MANIFEST:
        raise FullGridOperationsError(
            f"stale source manifest in existing record {record.path}"
        )
    provenance = audit_core._load_json_object(
        record.path / transfer.PROVENANCE_FILENAME
    )
    try:
        environment_hash = full_grid_audit._validate_execution_environment(
            provenance
        )
    except audit_core.TransferArtifactError as error:
        raise FullGridOperationsError(
            f"stale execution environment in existing record {record.path}"
        ) from error
    if environment_hash != full_grid_audit.PINNED_EXECUTION_ENVIRONMENT_SHA256:
        raise AssertionError("full-grid environment validator returned a stale digest")
    checkpoint = _nested_mapping(
        provenance, ("source_checkpoint",), context=str(record.path)
    )
    corpus = _nested_mapping(
        checkpoint, ("corpus",), context=str(record.path)
    )
    pretraining = _nested_mapping(
        checkpoint, ("pretraining",), context=str(record.path)
    )
    partition = _nested_mapping(
        pretraining, ("partition",), context=str(record.path)
    )
    state_hashes = _nested_mapping(
        pretraining, ("state_hashes",), context=str(record.path)
    )
    protocol = _nested_mapping(
        provenance, ("protocol",), context=str(record.path)
    )
    train_config = _nested_mapping(
        protocol, ("train_config",), context=str(record.path)
    )
    exact = {
        "checkpoint file": (
            checkpoint.get("file_sha256"),
            PINNED_CHECKPOINT_FILE_SHA256,
        ),
        "checkpoint state": (
            checkpoint.get("state_sha256"),
            PINNED_CHECKPOINT_STATE_SHA256,
        ),
        "corpus": (corpus.get("sha256"), PINNED_CORPUS_SHA256),
        "partition": (partition.get("sha256"), PINNED_PARTITION_SHA256),
        "source initial state": (
            state_hashes.get("initial"),
            PINNED_SOURCE_INITIAL_STATE_SHA256,
        ),
    }
    changed = [name for name, values in exact.items() if values[0] != values[1]]
    if changed:
        raise FullGridOperationsError(
            f"stale pinned provenance {changed} in existing record {record.path}"
        )
    if train_config.get("device") != "cuda":
        raise FullGridOperationsError(
            f"existing record did not use the pinned CUDA device: {record.path}"
        )


def _classify_recovery_entry(
    entry: Path,
    *,
    names: set[str],
) -> RecoveryArtifact | None:
    candidate = entry.name
    lock_suffix = ".native-transfer.lock"
    if candidate.startswith(".") and candidate.endswith(lock_suffix):
        output_name = candidate[1 : -len(lock_suffix)]
        if output_name in names:
            if entry.is_symlink() or not entry.is_file():
                raise FullGridOperationsError(
                    f"writer lock has an unsafe type: {entry}"
                )
            return RecoveryArtifact(path=entry, kind="lock")
    if candidate.startswith(".") and ".staging-" in candidate:
        output_name = candidate[1:].split(".staging-", 1)[0]
        if output_name in names:
            if entry.is_symlink() or not entry.is_dir():
                raise FullGridOperationsError(
                    f"writer staging path has an unsafe type: {entry}"
                )
            kind = (
                "seed_staging"
                if candidate.startswith(f".{output_name}.staging-seedcopy-")
                else "staging"
            )
            return RecoveryArtifact(path=entry, kind=kind)
    return None


def build_resume_plan(
    *, cho_root: Path, physionet_root: Path
) -> ResumePlan:
    """Fully validate all present records and identify exact missing work."""

    roots = {CHO2017: cho_root, PHYSIONET_MI: physionet_root}
    jobs = _grid_jobs(roots)
    jobs_by_dataset: dict[str, dict[str, GridJob]] = {
        dataset: {} for dataset in EXACT_DATASETS
    }
    for job in jobs:
        jobs_by_dataset[job.key.dataset][job.output.name] = job

    recovery: list[RecoveryArtifact] = []
    for dataset in EXACT_DATASETS:
        root = roots[dataset]
        if root.is_symlink() or not root.is_dir():
            raise FileNotFoundError(f"audited root must be a real directory: {root}")
        expected = jobs_by_dataset[dataset]
        unknown: list[str] = []
        names = set(expected)
        for entry in root.iterdir():
            if entry.name in expected:
                continue
            recovery_entry = _classify_recovery_entry(
                entry, names=names
            )
            if recovery_entry is not None:
                recovery.append(recovery_entry)
            else:
                unknown.append(entry.name)
        if unknown:
            raise FullGridOperationsError(
                f"audited root {root} contains unknown outputs: {sorted(unknown)}"
            )

    # Delegate the complete record, cross-record, per-seed, and outer-fold
    # validation to the frozen full-grid protocol.  It remains score blind.
    full_audit = full_grid_audit.audit_locked_fbms_full_grid(
        cho_root=cho_root,
        physionet_root=physionet_root,
    )
    for artifact in full_audit.records:
        _validate_pinned_record(artifact)
    jobs_by_key = {job.key: job for job in jobs}
    completed = [jobs_by_key[record.key] for record in full_audit.records]
    completed_set = {job.key for job in completed}
    pending = [job for job in jobs if job.key not in completed_set]
    completed.sort(key=lambda job: job.key)
    pending.sort(key=lambda job: job.key)
    recovery.sort(key=lambda item: str(item.path))
    if len(completed) + len(pending) != EXPECTED_RECORD_COUNT:
        raise AssertionError("resume plan does not cover the exact full grid")
    return ResumePlan(
        completed=tuple(completed),
        pending=tuple(pending),
        recovery_artifacts=tuple(recovery),
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _current_boot_time_epoch() -> float | None:
    """Return Linux boot time when available, without shelling out."""

    try:
        with Path("/proc/stat").open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("btime "):
                    return float(int(line.split()[1]))
    except (FileNotFoundError, OSError, ValueError, IndexError):
        return None
    return None


def _predates_current_boot(path: Path, boot_time: float | None) -> bool:
    return boot_time is not None and path.stat().st_mtime < boot_time


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _writer_lock_is_provably_stale(
    path: Path, *, boot_time: float | None
) -> bool:
    """Reject live/manual publishers before deleting their output claim."""

    if _predates_current_boot(path, boot_time):
        return True
    try:
        if path.stat().st_size > 4096:
            raise FullGridOperationsError(f"writer lock is unexpectedly large: {path}")
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise FullGridOperationsError(f"cannot inspect writer lock {path}") from error
    if not text.endswith("\n") or text.count("\n") != 1:
        raise FullGridOperationsError(
            f"writer lock is malformed and not proven pre-boot: {path}"
        )
    fields = text[:-1].split(" output=", 1)
    if len(fields) != 2 or not fields[0].startswith("pid="):
        raise FullGridOperationsError(f"writer lock has stale semantics: {path}")
    try:
        pid = int(fields[0].removeprefix("pid="))
    except ValueError as error:
        raise FullGridOperationsError(f"writer lock PID is malformed: {path}") from error
    if pid <= 0:
        raise FullGridOperationsError(f"writer lock PID is invalid: {path}")
    suffix = ".native-transfer.lock"
    output_name = path.name[1 : -len(suffix)]
    expected_output = (path.parent / output_name).resolve()
    observed_output = Path(fields[1]).expanduser().resolve()
    if observed_output != expected_output:
        raise FullGridOperationsError(
            f"writer lock output identity differs from its filename: {path}"
        )
    if _pid_is_alive(pid):
        raise FullGridOperationsError(
            f"refusing to recover active writer PID {pid} claimed by {path}"
        )
    return True


def _validate_recovery_safety(plan: ResumePlan) -> None:
    """Prove every transient stale without mutating an audited root."""

    boot_time = _current_boot_time_epoch()
    stale_locks: set[str] = set()
    for artifact in plan.recovery_artifacts:
        if artifact.kind != "lock":
            continue
        if not _writer_lock_is_provably_stale(artifact.path, boot_time=boot_time):
            raise AssertionError("writer-lock liveness check returned an invalid result")
        suffix = ".native-transfer.lock"
        stale_locks.add(artifact.path.name[1 : -len(suffix)])
    for artifact in plan.recovery_artifacts:
        if artifact.kind not in {"staging", "seed_staging"}:
            continue
        output_name = artifact.path.name[1:].split(".staging-", 1)[0]
        if artifact.kind == "seed_staging":
            continue
        if output_name not in stale_locks and not _predates_current_boot(
            artifact.path, boot_time
        ):
            raise FullGridOperationsError(
                "writer staging directory has no proven-stale paired lock: "
                f"{artifact.path}"
            )


def recover_interrupted_writes(plan: ResumePlan) -> None:
    """Remove only exact writer-owned transient paths after taking run.lock."""

    _validate_recovery_safety(plan)

    # No mutation happens until every transient in the set is proven safe.
    touched: set[Path] = set()
    ordered = sorted(
        plan.recovery_artifacts,
        key=lambda artifact: (artifact.kind == "lock", str(artifact.path)),
    )
    for artifact in ordered:
        path = artifact.path
        if artifact.kind == "lock":
            if path.is_symlink() or not path.is_file():
                raise FullGridOperationsError(
                    f"writer lock changed type before recovery: {path}"
                )
            path.unlink()
        elif artifact.kind in {"staging", "seed_staging"}:
            if path.is_symlink() or not path.is_dir():
                raise FullGridOperationsError(
                    f"writer staging path changed type before recovery: {path}"
                )
            shutil.rmtree(path)
        else:
            raise AssertionError(f"unknown recovery artifact kind {artifact.kind!r}")
        touched.add(path.parent)
    for root in touched:
        _fsync_directory(root)


def _validate_current_target_caches(
    screen: family_audit.LockedFBMSScreenAudit,
    *,
    cache_root: Path,
) -> None:
    """Rehash one live cache per target subject before any outcome job."""

    if cache_root.is_symlink():
        resolved_root = cache_root.resolve()
    else:
        resolved_root = cache_root.resolve()
    if not resolved_root.is_dir():
        raise FileNotFoundError(f"resolved target cache root is unavailable: {resolved_root}")

    identities: dict[tuple[str, int], tuple[Path, str]] = {}
    for record in screen.records:
        provenance = audit_core._load_json_object(
            record.path / transfer.PROVENANCE_FILENAME
        )
        target = audit_core._mapping(provenance.get("target"), name="target")
        declared_value = target.get("cache_path")
        if not isinstance(declared_value, str) or not declared_value:
            raise FullGridOperationsError(
                f"screen record has no absolute target cache path: {record.path}"
            )
        declared = Path(declared_value).expanduser()
        if not declared.is_absolute():
            raise FullGridOperationsError(
                f"screen target cache path is not absolute: {record.path}"
            )
        expected = transfer_core._native_cache_path(
            resolved_root, record.key.dataset, record.key.subject
        ).resolve()
        recorded_hash = target.get("cache_file_sha256")
        if recorded_hash != record.cache_file_sha256:
            raise FullGridOperationsError(
                f"screen cache provenance differs from audited identity: {record.path}"
            )
        key = (record.key.dataset, record.key.subject)
        identity = (expected, record.cache_file_sha256)
        previous = identities.setdefault(key, identity)
        if previous != identity:
            raise FullGridOperationsError(
                f"screen cache path/hash changed across conditions for {key}"
            )

    expected_keys = {
        (dataset, subject)
        for dataset in EXACT_DATASETS
        for subject in EXACT_SUBJECTS[dataset]
    }
    if set(identities) != expected_keys or len(identities) != 91:
        raise FullGridOperationsError(
            "screen did not resolve exactly one cache for each of 91 target subjects"
        )

    current_manifest: dict[str, str] = {}
    for (dataset, subject), (path, expected_hash) in sorted(identities.items()):
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(
                f"current target cache must be an available real file: {path}"
            )
        before = path.stat()
        observed_hash = _sha256_file(path)
        after = path.stat()
        stable_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) == (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        if not stable_identity:
            raise FullGridOperationsError(
                f"target cache changed while it was being hashed: {path}"
            )
        if observed_hash != expected_hash:
            raise FullGridOperationsError(
                "current target cache SHA-256 differs from the audited screen: "
                f"{dataset} S{subject}, path={path}"
            )
        current_manifest[f"{dataset}:s{subject}"] = observed_hash
    manifest_hash = hashlib.sha256(
        json.dumps(
            current_manifest,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if manifest_hash != full_grid_audit.PINNED_TARGET_CACHE_MANIFEST_SHA256:
        raise FullGridOperationsError(
            "current 91-subject target cache manifest differs from the frozen screen"
        )


def _audit_seed_screen(
    *,
    screen_cho_root: Path,
    screen_physionet_root: Path,
    cache_root: Path,
) -> family_audit.LockedFBMSScreenAudit:
    """Fully audit and pin all 455 source records before any byte copy."""

    screen = family_audit.audit_locked_fbms_screen(
        cho_root=screen_cho_root,
        physionet_root=screen_physionet_root,
    )
    if not screen.complete or len(screen.records) != family_audit.EXPECTED_RECORD_COUNT:
        raise FullGridOperationsError(
            "fold0/seed7 source screen is incomplete; refusing to seed full roots"
        )
    if len(screen.records) != 455:
        raise AssertionError("locked seed screen has an impossible record count")
    for record in screen.records:
        _validate_pinned_record(record)
    roots = {CHO2017: screen_cho_root, PHYSIONET_MI: screen_physionet_root}
    expected_names = {
        dataset: {
            audit_core.record_directory_name(record.key)
            for record in screen.records
            if record.key.dataset == dataset
        }
        for dataset in EXACT_DATASETS
    }
    for dataset, root in roots.items():
        observed = {entry.name for entry in root.iterdir()}
        unknown = sorted(observed - expected_names[dataset])
        if unknown:
            raise FullGridOperationsError(
                f"source screen root {root} contains unknown outputs: {unknown}"
            )
    _validate_current_target_caches(screen, cache_root=cache_root)
    return screen


def _copy_file_exact(source: Path, destination: Path) -> str:
    """Copy one file exclusively, fsync it, and verify byte identity."""

    source_hash = _sha256_file(source)
    with source.open("rb") as read_handle, destination.open("xb") as write_handle:
        shutil.copyfileobj(read_handle, write_handle, length=1024 * 1024)
        write_handle.flush()
        os.fsync(write_handle.fileno())
    if destination.stat().st_size != source.stat().st_size:
        raise FullGridOperationsError(
            f"seed copy size differs from source: {source} -> {destination}"
        )
    if _sha256_file(destination) != source_hash:
        raise FullGridOperationsError(
            f"seed copy SHA-256 differs from source: {source} -> {destination}"
        )
    return source_hash


def _copy_screen_record_atomically(
    record: audit_core.ValidatedTransferArtifact,
    *,
    destination: Path,
) -> None:
    """Byte-copy one audited record without modifying or moving its source."""

    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite seeded record {destination}")
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-seedcopy-",
            dir=destination.parent,
        )
    )
    try:
        observed_hashes: dict[str, str] = {}
        for filename in (transfer.PREDICTIONS_FILENAME, transfer.PROVENANCE_FILENAME):
            observed_hashes[filename] = _copy_file_exact(
                record.path / filename, staging / filename
            )
        expected_hashes = {
            transfer.PREDICTIONS_FILENAME: record.predictions_file_sha256,
            transfer.PROVENANCE_FILENAME: record.provenance_file_sha256,
        }
        if observed_hashes != expected_hashes:
            raise FullGridOperationsError(
                f"audited source hashes changed during seed copy from {record.path}"
            )
        _fsync_directory(staging)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"seed destination appeared concurrently: {destination}")
        os.replace(staging, destination)
        _fsync_directory(destination.parent)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def seed_screen_into_full_roots(
    *,
    screen_cho_root: Path,
    screen_physionet_root: Path,
    cache_root: Path,
    cho_root: Path,
    physionet_root: Path,
) -> ResumePlan:
    """Safely reuse the 455 screen records under the uniform full-grid names.

    Sources are fully audited before copying.  Existing destinations must be
    independently valid and byte-identical, which makes the seeding operation
    resumable after a power cut.  Sources are never moved, renamed, or deleted.
    The full roots are re-audited before this function returns.
    """

    screen = _audit_seed_screen(
        screen_cho_root=screen_cho_root,
        screen_physionet_root=screen_physionet_root,
        cache_root=cache_root,
    )
    roots = {CHO2017: cho_root, PHYSIONET_MI: physionet_root}
    existing_plan = build_resume_plan(
        cho_root=cho_root, physionet_root=physionet_root
    )
    if existing_plan.recovery_artifacts:
        raise FullGridOperationsError(
            "recover writer transients before seeding the full-grid roots"
        )
    existing_by_key = {job.key: job for job in existing_plan.completed}
    for source_record in sorted(screen.records, key=lambda record: record.key):
        destination = roots[source_record.key.dataset] / record_directory_name(
            source_record.key
        )
        existing = existing_by_key.get(source_record.key)
        if existing is None:
            _copy_screen_record_atomically(
                source_record, destination=destination
            )
        else:
            destination_record = audit_core.validate_transfer_artifact(
                existing.output,
                expected_key=source_record.key,
                contract=full_grid_audit.AUDIT_CONTRACT,
            )
            _validate_pinned_record(destination_record)
            if (
                destination_record.predictions_file_sha256
                != source_record.predictions_file_sha256
                or destination_record.provenance_file_sha256
                != source_record.provenance_file_sha256
            ):
                raise FullGridOperationsError(
                    "existing seeded destination is valid but not byte-identical "
                    f"to its audited source: {destination}"
                )

    seeded_plan = build_resume_plan(
        cho_root=cho_root, physionet_root=physionet_root
    )
    completed_keys = {job.key for job in seeded_plan.completed}
    missing_seed_keys = sorted(
        record.key for record in screen.records if record.key not in completed_keys
    )
    if missing_seed_keys:
        raise FullGridOperationsError(
            f"full roots failed post-seed audit for {len(missing_seed_keys)} records"
        )
    return seeded_plan


class _RunnerClaim:
    """An OS-released exclusive claim inherited by every active child.

    The flock remains held if the scheduler dies but child records are still
    running.  A new runner can enter only after those children exit.  A machine
    power cut closes every descriptor automatically, making the next resume
    unambiguous without trusting a stale PID file.
    """

    def __init__(self, ops_root: Path) -> None:
        self.path = ops_root / "run.lock"
        self._handle: BinaryIO | None = None

    def __enter__(self) -> "_RunnerClaim":
        handle = self.path.open("a+b")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            handle.close()
            raise FullGridOperationsError(
                f"another full-grid runner or inherited child owns {self.path}"
            ) from error
        handle.seek(0)
        handle.truncate()
        payload = {
            "schema": RUNNER_SCHEMA,
            "pid": os.getpid(),
            "started_utc": _utc_now(),
            "worker_count": MAX_GPU_WORKERS,
        }
        handle.write((json.dumps(payload, sort_keys=True) + "\n").encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())
        self._handle = handle
        return self

    def fileno(self) -> int:
        if self._handle is None:
            raise RuntimeError("runner claim is not active")
        return self._handle.fileno()

    def __exit__(self, *unused: object) -> None:
        del unused
        if self._handle is not None:
            # Do not issue LOCK_UN: an abruptly orphaned child may still hold
            # this same open-file description.  Closing releases the flock
            # only when the final inherited descriptor is gone.
            self._handle.close()
            self._handle = None


def _paths_overlap(left: Path, right: Path) -> bool:
    try:
        left.relative_to(right)
        return True
    except ValueError:
        pass
    try:
        right.relative_to(left)
        return True
    except ValueError:
        return False


def _prepare_directories(
    *,
    cho_root: Path,
    physionet_root: Path,
    ops_root: Path,
    create: bool,
) -> tuple[Path, Path, Path]:
    supplied = tuple(path.expanduser() for path in (cho_root, physionet_root, ops_root))
    for path in supplied:
        if path.is_symlink():
            raise FullGridOperationsError(
                f"operational path must not be a symlink: {path}"
            )
    paths = tuple(path.resolve() for path in supplied)
    cho, physionet, ops = paths
    if _paths_overlap(cho, physionet):
        raise FullGridOperationsError("the two audited roots must be disjoint")
    if _paths_overlap(cho, ops) or _paths_overlap(physionet, ops):
        raise FullGridOperationsError(
            "operations logs/state must be outside and disjoint from audited roots"
        )
    for path in paths:
        if path.exists():
            if path.is_symlink() or not path.is_dir():
                raise FullGridOperationsError(f"operational path is not a real directory: {path}")
            continue
        if not create:
            raise FileNotFoundError(f"operational directory is unavailable: {path}")
        parent = path.parent
        if parent.is_symlink() or not parent.is_dir():
            raise FileNotFoundError(f"operational parent must be a real directory: {parent}")
        path.mkdir()
        _fsync_directory(parent)
    return cho, physionet, ops


def _prepare_screen_roots(
    *,
    screen_cho_root: Path,
    screen_physionet_root: Path,
    disjoint_from: Sequence[Path],
) -> tuple[Path, Path]:
    supplied = (screen_cho_root.expanduser(), screen_physionet_root.expanduser())
    for path in supplied:
        if path.is_symlink():
            raise FullGridOperationsError(
                f"screen root must not be a symlink: {path}"
            )
    roots = tuple(path.resolve() for path in supplied)
    if _paths_overlap(roots[0], roots[1]):
        raise FullGridOperationsError("the two source screen roots must be disjoint")
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(f"source screen root is unavailable: {root}")
        if any(_paths_overlap(root, other) for other in disjoint_from):
            raise FullGridOperationsError(
                "source screen roots must be disjoint from full-grid and operations roots"
            )
    return roots


def _append_journal(path: Path, payload: Mapping[str, object]) -> None:
    record = {"schema": RUNNER_SCHEMA, **payload}
    encoded = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def command_for_job(
    job: GridJob,
    *,
    cache_root: Path,
    checkpoint_path: Path,
) -> tuple[str, ...]:
    """Build one fresh-process command with no mutable scientific arguments."""

    return (
        sys.executable,
        "-m",
        "ieee_mi.native_fbms_transfer",
        "--dataset",
        job.key.dataset,
        "--subject",
        str(job.key.subject),
        "--fold",
        str(job.key.fold),
        "--condition",
        job.key.condition,
        "--seed",
        str(job.key.seed),
        "--cache-root",
        str(cache_root),
        "--checkpoint",
        str(checkpoint_path),
        "--checkpoint-sha256",
        PINNED_CHECKPOINT_FILE_SHA256,
        "--output",
        str(job.output),
        "--device",
        "cuda",
    )


def _allocate_log(ops_root: Path, job: GridJob) -> tuple[Path, BinaryIO]:
    parent = ops_root / "logs" / job.key.dataset / job.output.name
    parent.mkdir(parents=True, exist_ok=True)
    for sequence in range(10_000):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        path = parent / f"attempt-{stamp}-p{os.getpid()}-{sequence:04d}.log"
        try:
            return path, path.open("xb")
        except FileExistsError:
            continue
    raise FullGridOperationsError(f"could not allocate a unique log under {parent}")


def _spawn(
    job: GridJob,
    *,
    cache_root: Path,
    checkpoint_path: Path,
    ops_root: Path,
    claim_fd: int,
) -> _ActiveJob:
    _validate_current_source_manifest()
    _validate_current_execution_environment()
    log_path, log_handle = _allocate_log(ops_root, job)
    command = command_for_job(
        job,
        cache_root=cache_root,
        checkpoint_path=checkpoint_path,
    )
    project_root = Path(__file__).resolve().parent.parent
    try:
        process = subprocess.Popen(
            command,
            cwd=project_root,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            close_fds=True,
            pass_fds=(claim_fd,),
        )
    except BaseException:
        log_handle.close()
        raise
    return _ActiveJob(
        job=job,
        process=process,
        log_path=log_path,
        log_handle=log_handle,
        started_monotonic=time.monotonic(),
    )


def _terminate_active(active: Sequence[_ActiveJob]) -> None:
    for item in active:
        if item.process.poll() is None:
            item.process.send_signal(signal.SIGTERM)
    deadline = time.monotonic() + 30.0
    while any(item.process.poll() is None for item in active) and time.monotonic() < deadline:
        time.sleep(0.1)
    for item in active:
        if item.process.poll() is None:
            item.process.kill()
    for item in active:
        with contextlib.suppress(Exception):
            item.process.wait(timeout=5.0)
        item.log_handle.close()


def _terminate_timed_out(item: _ActiveJob) -> None:
    if item.process.poll() is None:
        item.process.send_signal(signal.SIGTERM)
        try:
            item.process.wait(timeout=30.0)
        except subprocess.TimeoutExpired:
            item.process.kill()
            item.process.wait(timeout=5.0)
    item.log_handle.close()


def execute_pending(
    plan: ResumePlan,
    *,
    cache_root: Path,
    checkpoint_path: Path,
    ops_root: Path,
    claim_fd: int,
) -> None:
    """Run pending records in four fresh processes and validate every publish."""

    pending = deque(plan.pending)
    active: list[_ActiveJob] = []
    journal = ops_root / "journal.jsonl"
    failed = False
    try:
        while pending or active:
            while pending and len(active) < MAX_GPU_WORKERS and not failed:
                item = _spawn(
                    pending.popleft(),
                    cache_root=cache_root,
                    checkpoint_path=checkpoint_path,
                    ops_root=ops_root,
                    claim_fd=claim_fd,
                )
                active.append(item)
                _append_journal(
                    journal,
                    {
                        "event": "record_started",
                        "created_utc": _utc_now(),
                        "record": item.job.key.as_dict(),
                        "output": str(item.job.output),
                        "log": str(item.log_path),
                        "pid": item.process.pid,
                    },
                )
            now = time.monotonic()
            timed_out = [
                item
                for item in active
                if item.process.poll() is None
                and now - item.started_monotonic >= JOB_TIMEOUT_SECONDS
            ]
            for item in timed_out:
                active.remove(item)
                _terminate_timed_out(item)
                failed = True
                _append_journal(
                    journal,
                    {
                        "event": "record_timed_out",
                        "created_utc": _utc_now(),
                        "record": item.job.key.as_dict(),
                        "output": str(item.job.output),
                        "log": str(item.log_path),
                        "timeout_seconds": JOB_TIMEOUT_SECONDS,
                        "return_code": item.process.returncode,
                        "validated": False,
                    },
                )
            if failed and not active:
                break
            finished = [item for item in active if item.process.poll() is not None]
            if not finished:
                time.sleep(0.2)
                continue
            for item in finished:
                active.remove(item)
                return_code = item.process.returncode
                item.log_handle.close()
                validation_error: str | None = None
                if return_code == 0:
                    try:
                        artifact = audit_core.validate_transfer_artifact(
                            item.job.output,
                            expected_key=item.job.key,
                            contract=full_grid_audit.AUDIT_CONTRACT,
                        )
                        _validate_pinned_record(artifact)
                        _validate_current_source_manifest()
                    except Exception as error:  # fail closed; journal the reason
                        validation_error = f"{type(error).__name__}: {error}"
                        failed = True
                else:
                    failed = True
                _append_journal(
                    journal,
                    {
                        "event": "record_finished",
                        "created_utc": _utc_now(),
                        "record": item.job.key.as_dict(),
                        "output": str(item.job.output),
                        "log": str(item.log_path),
                        "return_code": return_code,
                        "validated": return_code == 0 and validation_error is None,
                        "validation_error": validation_error,
                    },
                )
        if failed:
            raise FullGridOperationsError(
                "at least one record failed; no further jobs were launched; resume after inspection"
            )
    except BaseException:
        _terminate_active(active)
        raise


def plan_manifest(plan: ResumePlan) -> dict[str, object]:
    """Return score-free counts and timing estimates for operator review."""

    pending_by_dataset = {
        dataset: sum(job.key.dataset == dataset for job in plan.pending)
        for dataset in EXACT_DATASETS
    }
    worker_seconds = sum(
        pending_by_dataset[dataset] * EMPIRICAL_SECONDS_PER_RECORD[dataset]
        for dataset in EXACT_DATASETS
    )
    ideal_hours = worker_seconds / MAX_GPU_WORKERS / 3600.0
    return {
        "schema": RUNNER_SCHEMA,
        "decision": "GO",
        "mode": "development",
        "confirmation_access": False,
        "scores_computed": False,
        "worker_count": MAX_GPU_WORKERS,
        "per_record_timeout_seconds": JOB_TIMEOUT_SECONDS,
        "worker_isolation": "one fresh Python process per record",
        "expected_record_count": EXPECTED_RECORD_COUNT,
        "expected_by_dataset": dict(EXPECTED_RECORD_COUNTS),
        "fully_validated_record_count": len(plan.completed),
        "pending_record_count": len(plan.pending),
        "pending_by_dataset": pending_by_dataset,
        "recoverable_transient_count": len(plan.recovery_artifacts),
        "empirical_pending_worker_hours": worker_seconds / 3600.0,
        "ideal_pending_wall_hours_at_four_workers": ideal_hours,
        "planning_wall_hours_with_process_and_audit_overhead": [
            ideal_hours * 1.08,
            ideal_hours * 1.33,
        ],
        "pins": {
            "checkpoint_file_sha256": PINNED_CHECKPOINT_FILE_SHA256,
            "checkpoint_state_sha256": PINNED_CHECKPOINT_STATE_SHA256,
            "corpus_sha256": PINNED_CORPUS_SHA256,
            "partition_sha256": PINNED_PARTITION_SHA256,
            "transfer_source_sha256": PINNED_TRANSFER_SOURCE_SHA256,
            "source_manifest": dict(PINNED_SOURCE_MANIFEST),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Preflight or run the exact 8,675-record CardinalFBMS development grid."
        )
    )
    parser.add_argument("--cho-root", required=True, type=Path)
    parser.add_argument("--physionet-root", required=True, type=Path)
    parser.add_argument("--screen-cho-root", required=True, type=Path)
    parser.add_argument("--screen-physionet-root", required=True, type=Path)
    parser.add_argument("--ops-root", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="recover exact writer transients and launch pending records",
    )
    return parser


def run(argv: Sequence[str] | None = None) -> dict[str, object]:
    arguments = build_parser().parse_args(argv)
    cho, physionet, ops = _prepare_directories(
        cho_root=arguments.cho_root,
        physionet_root=arguments.physionet_root,
        ops_root=arguments.ops_root,
        create=arguments.execute,
    )
    screen_cho, screen_physionet = _prepare_screen_roots(
        screen_cho_root=arguments.screen_cho_root,
        screen_physionet_root=arguments.screen_physionet_root,
        disjoint_from=(cho, physionet, ops),
    )
    cache_root = arguments.cache_root.expanduser().resolve()
    checkpoint_path = arguments.checkpoint.expanduser().resolve()
    if not cache_root.is_dir():
        raise FileNotFoundError(f"cache root is unavailable: {cache_root}")
    _assert_exact_scientific_contract()
    _validate_current_source_manifest()
    _validate_current_execution_environment()
    _validate_checkpoint(checkpoint_path)

    if not arguments.execute:
        with _RunnerClaim(ops):
            _audit_seed_screen(
                screen_cho_root=screen_cho,
                screen_physionet_root=screen_physionet,
                cache_root=cache_root,
            )
            plan = build_resume_plan(cho_root=cho, physionet_root=physionet)
            _validate_recovery_safety(plan)
            return plan_manifest(plan)

    with _RunnerClaim(ops) as claim:
        plan = build_resume_plan(cho_root=cho, physionet_root=physionet)
        recover_interrupted_writes(plan)
        if plan.recovery_artifacts:
            plan = build_resume_plan(cho_root=cho, physionet_root=physionet)
        plan = seed_screen_into_full_roots(
            screen_cho_root=screen_cho,
            screen_physionet_root=screen_physionet,
            cache_root=cache_root,
            cho_root=cho,
            physionet_root=physionet,
        )
        manifest = plan_manifest(plan)
        _append_journal(
            ops / "journal.jsonl",
            {
                "event": "run_preflight_complete",
                "created_utc": _utc_now(),
                "manifest": manifest,
            },
        )
        execute_pending(
            plan,
            cache_root=cache_root,
            checkpoint_path=checkpoint_path,
            ops_root=ops,
            claim_fd=claim.fileno(),
        )
        final_plan = build_resume_plan(cho_root=cho, physionet_root=physionet)
        if not final_plan.complete:
            raise FullGridOperationsError(
                f"runner stopped with {len(final_plan.pending)} records pending"
            )
        final_manifest = plan_manifest(final_plan)
        _append_journal(
            ops / "journal.jsonl",
            {
                "event": "run_complete",
                "created_utc": _utc_now(),
                "manifest": final_manifest,
            },
        )
        return final_manifest


def main() -> None:
    try:
        manifest = run()
    except Exception as error:
        payload = {
            "schema": RUNNER_SCHEMA,
            "decision": "NO-GO",
            "error_type": type(error).__name__,
            "error": str(error),
            "scores_computed": False,
        }
        print(json.dumps(payload, sort_keys=True, indent=2), file=sys.stderr)
        raise SystemExit(1) from error
    print(json.dumps(manifest, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
