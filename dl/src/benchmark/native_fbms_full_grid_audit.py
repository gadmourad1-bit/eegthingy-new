"""Fail-closed audit of the prespecified CardinalFBMS development expansion.

The only caller-controlled values are two filesystem roots.  Dataset cohorts,
outer folds, training seeds, conditions, record names, and artifact semantics
are fixed here.  The full grid remains development-only and never authorizes a
sealed confirmation subject.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from . import native_fbms_transfer as transfer
from . import native_fbms_transfer_audit as screen_audit
from . import native_transfer_audit as _core
from .native_fbms_pretraining_cli import CHECKPOINT_SCHEMA


AUDIT_SCHEMA = "eeg-mi-native-cardinal-fbms-full-grid-audit-v1"
SUMMARY_SCHEMA = "eeg-mi-native-cardinal-fbms-full-grid-summary-v1"
GRID_SCHEMA = "eeg-mi-native-cardinal-fbms-full-development-grid-v1"
SUMMARY_ANALYSIS_POLICY = (
    "locked_fbms_full_development_grid_subject_unit_descriptive_no_inference"
)
MODE = "development"

CHO2017 = screen_audit.CHO2017
PHYSIONET_MI = screen_audit.PHYSIONET_MI
LOCKED_DATASETS: tuple[str, ...] = (CHO2017, PHYSIONET_MI)
LOCKED_SUBJECTS = {
    CHO2017: tuple(range(16, 53)),
    PHYSIONET_MI: tuple(range(1, 55)),
}
LOCKED_FOLDS = {
    CHO2017: (0, 1, 2, 3, 4),
    PHYSIONET_MI: (0, 1, 2),
}
LOCKED_SEEDS: tuple[int, ...] = (7, 17, 27, 37, 47)
LOCKED_CONDITIONS: tuple[str, ...] = transfer.TRANSFER_CONDITIONS

# Generated score-blind on the x86_64 GPU host before any full-grid outcome
# jobs. Torch module initialization is not bit-identical on Apple arm64, so
# artifact acceptance must use these frozen writer-platform digests rather
# than reconstructing a state on the machine that happens to run the audit.
CANONICAL_SCRATCH_STATE_SHA256_BY_SEED = {
    7: "382d4e18385904b885e4140963705939b7d935ae95731ac402d3a447eabef507",
    17: "876e995189902d14e1f9424edeafd132be8185f0caf8a60ff6dd8304824f09a3",
    27: "76b023e24ac63260e3ae3fa1d22709787f7b52d78e496c0a761342aed82ab079",
    37: "486aa5fe51783925228d0fc4438ad3415ce82ca714aee23341bbc4c76f0c006b",
    47: "2a2943e16de9117d0f2bfbfe6ea957cb7be67710aae385f4b20a90b87b5b6af1",
}

PINNED_CHECKPOINT_FILE_SHA256 = (
    "7378450eee169a56f0bfc1bf6ad78ad8014b5c63ae998267b4cc0ddee0c712cb"
)
PINNED_CHECKPOINT_STATE_SHA256 = (
    "e045e8c15c1b8319da0282e5cfa3342f1ffc105fd42a4197e99b8ebd5ea0fff5"
)
PINNED_SOURCE_INITIAL_STATE_SHA256 = (
    "382d4e18385904b885e4140963705939b7d935ae95731ac402d3a447eabef507"
)
PINNED_CORPUS_SHA256 = (
    "e5ce98c41184ff42db6e9976d3369834ac83adb2c8dadacd560ba422ea2a360e"
)
PINNED_PARTITION_SHA256 = (
    "7636dee344ff619ea725b78ba7645ed7d23564459256cadebd4117e260e2a1d1"
)
PINNED_TRANSFER_SOURCE_SHA256 = (
    "8270ead4f0d3e825f90638f752c25784c5dd2c0cd9012863de9c70598569aad6"
)
PINNED_SOURCE_MANIFEST = {
    "eeg_mi/baselines.py": (
        "a2270d0affb552315211f8360fea4a804c27a712c4a533907a49881731864379"
    ),
    "eeg_mi/config.py": (
        "b1be4418d8e28f4c58b2ea716ade766955dd5ea80630e1a1af9d5c8a4213bea6"
    ),
    "eeg_mi/data.py": (
        "00cf9b5570d5a9563ede010d2a141c69359d3f227d39f67e19b334702aaa2937"
    ),
    "eeg_mi/models.py": (
        "b4df65012d1df9f8979207dd28518ead0a396cd3ddae215a87cbf95e21534400"
    ),
    "eeg_mi/native_fbms_pretraining_cli.py": (
        "9fc025f2597743748a8674c7dc97c2ab82ec67d384d233c655eeb27c785388f8"
    ),
    "eeg_mi/native_fbms_transfer.py": PINNED_TRANSFER_SOURCE_SHA256,
    "eeg_mi/native_pretraining.py": (
        "138ae2156a749136081518ea7f3571596ed8ac7d145290c08a0595ef1c6d1ff9"
    ),
    "eeg_mi/native_pretraining_cli.py": (
        "795c0485f2a39ae346ea19e1118c15cf9eb2953485dcdcd0e2cccdb9003cfd2e"
    ),
    "eeg_mi/native_transfer.py": (
        "c7755acc379d270bf7188ca9923ace392c4f31ab232cbb0dc60e9d75c2a07f34"
    ),
    "eeg_mi/requirements-cu128.txt": (
        "06171d32aaef187b2ba9047322a8e4a93e5f16d04b743cb75095beda9f858a16"
    ),
    "eeg_mi/training.py": (
        "33906fdd99821df6f847bd42bb2e55604f34a2d9d0ba6c34183d038056260f16"
    ),
}
PINNED_SOURCE_MANIFEST_SHA256 = (
    "f83b9ed261fa44d01ba6cd09542413f9d82caa26974af0754b612f4e6c967b5f"
)
PINNED_TARGET_CACHE_MANIFEST_SHA256 = (
    "e6ed4800c6bb1eff284a80fc2b1c240b1b63cecbbe427af5321784990354935e"
)

# Exact environment recorded by every audited fold-0/seed-7 source record on
# the frozen x86_64 RTX 5070 writer host.  The expansion must not mix devices,
# package builds, Python executables, kernels, or CUDA/cuDNN identities.
PINNED_EXECUTION_ENVIRONMENT = {
    "cuda_available": True,
    "cuda_devices": [
        {
            "compute_capability": [12, 0],
            "index": 0,
            "name": "NVIDIA GeForce RTX 5070",
        }
    ],
    "cudnn": 91900,
    "machine": "x86_64",
    "packages": {
        "braindecode": "1.6.1",
        "mne": "1.12.1",
        "moabb": "1.5.0",
        "numpy": "2.5.1",
        "scipy": "1.18.0",
        "torch": "2.11.0+cu128",
    },
    "platform": "Linux-7.0.0-28-generic-x86_64-with-glibc2.39",
    "python": "3.12.3",
    "python_executable": (
        "/home/user/Desktop/eegthingy_eeg_arch_v7_20260719/.venv/bin/python"
    ),
    "python_implementation": "CPython",
    "requested_device": "cuda",
    "torch": "2.11.0+cu128",
    "torch_cuda_build": "12.8",
}
PINNED_EXECUTION_ENVIRONMENT_SHA256 = (
    "7e8da881ff0284ef00a6f947225e4bcb7f7eb930ac664c7dfcd5080a18a8841d"
)

# The fold0/seed7 screen could omit fold and seed from its directory names.
# The expansion cannot: this template is part of the immutable grid identity.
FULL_RECORD_NAME_TEMPLATE = (
    "s{subject:03d}_f{fold}_seed{seed}_{condition}"
)

EXPECTED_RECORD_COUNT_BY_DATASET = {
    dataset: (
        len(LOCKED_SUBJECTS[dataset])
        * len(LOCKED_FOLDS[dataset])
        * len(LOCKED_SEEDS)
        * len(LOCKED_CONDITIONS)
    )
    for dataset in LOCKED_DATASETS
}
EXPECTED_RECORD_COUNT = sum(EXPECTED_RECORD_COUNT_BY_DATASET.values())


def _assert_writer_grid_identity() -> None:
    """Detect writer-authorization drift before inspecting either root."""

    expected_cohorts = {
        dataset: LOCKED_SUBJECTS[dataset] for dataset in LOCKED_DATASETS
    }
    expected_folds = {
        dataset: LOCKED_FOLDS[dataset] for dataset in LOCKED_DATASETS
    }
    if dict(transfer.NATIVE_TARGET_DEVELOPMENT_COHORTS) != expected_cohorts:
        raise RuntimeError("FBMS writer development cohorts differ from full grid")
    if dict(transfer.NATIVE_TARGET_FOLDS) != expected_folds:
        raise RuntimeError("FBMS writer outer folds differ from full grid")
    if tuple(transfer.FROZEN_DEVELOPMENT_SEEDS) != LOCKED_SEEDS:
        raise RuntimeError("FBMS writer seeds differ from full grid")
    if tuple(transfer.TRANSFER_CONDITIONS) != LOCKED_CONDITIONS:
        raise RuntimeError("FBMS writer conditions differ from full grid")


def _expected_canonical_scratch_state_sha256(seed: int) -> str:
    """Return the prespecified x86/GPU writer-state digest for one seed."""

    if seed not in LOCKED_SEEDS:
        raise ValueError(f"seed must be one of {LOCKED_SEEDS}")
    return CANONICAL_SCRATCH_STATE_SHA256_BY_SEED[seed]


def _replace_source_initial_for_base_validation(
    provenance: Mapping[str, object],
    initialization_hash: str,
) -> Mapping[str, object]:
    """Make a shallow structural copy for the seed-7 screen validator.

    The screen contract intentionally binds canonical scratch to the source
    checkpoint's seed-7 initial state.  For another frozen seed, the full-grid
    validator first checks the frozen x86/GPU writer-state digest above, then
    substitutes only this field while reusing every other screen semantic
    check.  The returned semantic identity is restored to the real source
    initial hash by :func:`_validate_full_artifact_semantics`.
    """

    adjusted = dict(provenance)
    checkpoint = dict(
        _core._mapping(provenance.get("source_checkpoint"), name="source_checkpoint")
    )
    pretraining = dict(
        _core._mapping(
            checkpoint.get("pretraining"), name="source_checkpoint.pretraining"
        )
    )
    state_hashes = dict(
        _core._mapping(
            pretraining.get("state_hashes"),
            name="source_checkpoint.pretraining.state_hashes",
        )
    )
    state_hashes["initial"] = initialization_hash
    pretraining["state_hashes"] = state_hashes
    checkpoint["pretraining"] = pretraining
    adjusted["source_checkpoint"] = checkpoint
    return adjusted


def _validate_target_seed(
    provenance: Mapping[str, object],
    key: _core.TransferRecordKey,
) -> int:
    """Bind the writer's real target seed to the authorized full-grid key."""

    protocol = _core._mapping(provenance.get("protocol"), name="protocol")
    train_config = _core._mapping(
        protocol.get("train_config"), name="protocol.train_config"
    )
    observed = _core._integer(
        train_config.get("seed"), name="protocol.train_config.seed"
    )
    if observed not in LOCKED_SEEDS or key.seed not in LOCKED_SEEDS:
        raise _core.TransferArtifactError(
            "protocol seed is outside the frozen full-development seed grid"
        )
    if observed != key.seed:
        raise _core.TransferArtifactError("protocol seed differs from record")
    return observed


def _replace_target_seed_for_screen_validation(
    provenance: Mapping[str, object],
) -> Mapping[str, object]:
    """Copy only the nested target seed needed by the seed-7 screen checker.

    The real seed is validated independently before this compatibility copy is
    made and restored to the returned full-grid semantic identity afterwards.
    No caller-owned mapping is mutated.
    """

    adjusted = dict(provenance)
    protocol = dict(_core._mapping(provenance.get("protocol"), name="protocol"))
    train_config = dict(
        _core._mapping(
            protocol.get("train_config"), name="protocol.train_config"
        )
    )
    train_config["seed"] = screen_audit.LOCKED_SEEDS[0]
    protocol["train_config"] = train_config
    adjusted["protocol"] = protocol
    return adjusted


def _target_row_count(provenance: Mapping[str, object]) -> int:
    target = _core._mapping(provenance.get("target"), name="target")
    raw_shape = target.get("raw_shape")
    if (
        not isinstance(raw_shape, list)
        or len(raw_shape) != 3
        or any(isinstance(value, bool) or not isinstance(value, int) for value in raw_shape)
        or any(value <= 0 for value in raw_shape)
    ):
        raise _core.TransferArtifactError("target.raw_shape must contain three positive integers")
    channels = tuple(str(value) for value in target.get("channels", ()))
    if raw_shape[1] != len(channels):
        raise _core.TransferArtifactError(
            "target raw channel dimension differs from target channel identity"
        )
    return int(raw_shape[0])


def _validate_execution_environment(
    provenance: Mapping[str, object],
) -> str:
    environment = _core._mapping(
        provenance.get("environment"), name="environment"
    )
    try:
        encoded = json.dumps(
            environment,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        expected_encoded = json.dumps(
            PINNED_EXECUTION_ENVIRONMENT,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise _core.TransferArtifactError(
            "execution environment is not canonical JSON"
        ) from error
    digest = hashlib.sha256(encoded).hexdigest()
    expected_digest = hashlib.sha256(expected_encoded).hexdigest()
    if expected_digest != PINNED_EXECUTION_ENVIRONMENT_SHA256:
        raise AssertionError("pinned execution-environment constant is inconsistent")
    if encoded != expected_encoded or digest != PINNED_EXECUTION_ENVIRONMENT_SHA256:
        raise _core.TransferArtifactError(
            "execution environment differs from the frozen RTX 5070 writer host"
        )
    protocol = _core._mapping(provenance.get("protocol"), name="protocol")
    train_config = _core._mapping(
        protocol.get("train_config"), name="protocol.train_config"
    )
    if train_config.get("device") != "cuda":
        raise _core.TransferArtifactError(
            "protocol device differs from the frozen CUDA writer"
        )
    return digest


def _validate_experiment_pins(
    provenance: Mapping[str, object],
) -> Mapping[str, str]:
    """Require THE frozen checkpoint, corpus, partition and writer manifest."""

    checkpoint = _core._mapping(
        provenance.get("source_checkpoint"), name="source_checkpoint"
    )
    corpus = _core._mapping(checkpoint.get("corpus"), name="source_checkpoint.corpus")
    pretraining = _core._mapping(
        checkpoint.get("pretraining"), name="source_checkpoint.pretraining"
    )
    partition = _core._mapping(
        pretraining.get("partition"), name="source_checkpoint.pretraining.partition"
    )
    state_hashes = _core._mapping(
        pretraining.get("state_hashes"),
        name="source_checkpoint.pretraining.state_hashes",
    )
    observed = {
        "checkpoint_file_sha256": _core._digest(
            checkpoint.get("file_sha256"), name="source_checkpoint.file_sha256"
        ),
        "checkpoint_state_sha256": _core._digest(
            checkpoint.get("state_sha256"), name="source_checkpoint.state_sha256"
        ),
        "corpus_sha256": _core._digest(
            corpus.get("sha256"), name="source_checkpoint.corpus.sha256"
        ),
        "partition_sha256": _core._digest(
            partition.get("sha256"),
            name="source_checkpoint.pretraining.partition.sha256",
        ),
        "source_initial_state_sha256": _core._digest(
            state_hashes.get("initial"),
            name="source_checkpoint.pretraining.state_hashes.initial",
        ),
    }
    expected = {
        "checkpoint_file_sha256": PINNED_CHECKPOINT_FILE_SHA256,
        "checkpoint_state_sha256": PINNED_CHECKPOINT_STATE_SHA256,
        "corpus_sha256": PINNED_CORPUS_SHA256,
        "partition_sha256": PINNED_PARTITION_SHA256,
        "source_initial_state_sha256": PINNED_SOURCE_INITIAL_STATE_SHA256,
    }
    changed = sorted(name for name in expected if observed[name] != expected[name])
    if changed:
        raise _core.TransferArtifactError(
            f"full-grid artifact differs from frozen experiment pins {changed}"
        )

    source_code = _core._mapping(provenance.get("source_code"), name="source_code")
    if source_code.get("hash_algorithm") != "sha256":
        raise _core.TransferArtifactError("writer manifest is not SHA-256")
    files = _core._mapping(source_code.get("files"), name="source_code.files")
    canonical_files = {
        str(name): _core._digest(value, name=f"source_code.files[{name!r}]")
        for name, value in files.items()
    }
    try:
        encoded = json.dumps(
            canonical_files,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        expected_encoded = json.dumps(
            PINNED_SOURCE_MANIFEST,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise _core.TransferArtifactError("writer manifest is not canonical JSON") from error
    manifest_hash = hashlib.sha256(encoded).hexdigest()
    if hashlib.sha256(expected_encoded).hexdigest() != PINNED_SOURCE_MANIFEST_SHA256:
        raise AssertionError("pinned source-manifest constant is inconsistent")
    if encoded != expected_encoded or manifest_hash != PINNED_SOURCE_MANIFEST_SHA256:
        raise _core.TransferArtifactError(
            "writer manifest differs from the frozen 11-file source identity"
        )
    return {**observed, "source_manifest_sha256": manifest_hash}


def _validate_full_artifact_semantics(
    provenance: Mapping[str, object],
    key: _core.TransferRecordKey,
) -> Mapping[str, str]:
    """Apply all family checks plus independent per-seed scratch identity."""

    pinned_identity = _validate_experiment_pins(provenance)
    environment_hash = _validate_execution_environment(provenance)
    target_row_count = _target_row_count(provenance)
    actual_target_seed = _validate_target_seed(provenance, key)
    adjusted = _replace_target_seed_for_screen_validation(provenance)
    actual_source_initial: str | None = None
    if key.condition == transfer.SCRATCH_CARDINAL_FBMS_CANONICAL:
        condition = _core._mapping(provenance.get("condition"), name="condition")
        observed = _core._digest(
            condition.get("initialization_state_sha256"),
            name="condition.initialization_state_sha256",
        )
        expected = _expected_canonical_scratch_state_sha256(key.seed)
        if observed != expected:
            raise _core.TransferArtifactError(
                "canonical scratch initialization differs from frozen x86/GPU "
                f"seed {key.seed} digest"
            )
        checkpoint = _core._mapping(
            provenance.get("source_checkpoint"), name="source_checkpoint"
        )
        pretraining = _core._mapping(
            checkpoint.get("pretraining"), name="source_checkpoint.pretraining"
        )
        hashes = _core._mapping(
            pretraining.get("state_hashes"),
            name="source_checkpoint.pretraining.state_hashes",
        )
        actual_source_initial = _core._digest(
            hashes.get("initial"),
            name="source_checkpoint.pretraining.state_hashes.initial",
        )
        if actual_source_initial != observed:
            adjusted = _replace_source_initial_for_base_validation(
                adjusted, observed
            )

    screen_key = replace(key, seed=screen_audit.LOCKED_SEEDS[0])
    identity = dict(
        screen_audit._validate_family_artifact_semantics(  # noqa: SLF001
            adjusted, screen_key
        )
    )
    identity.update(pinned_identity)
    if actual_source_initial is not None:
        identity["source_initial_state_sha256"] = actual_source_initial
    identity["execution_environment_sha256"] = environment_hash
    identity["target_row_count"] = str(target_row_count)
    identity["target_seed"] = str(actual_target_seed)
    return identity


def _semantic_identity(
    record: _core.ValidatedTransferArtifact,
) -> dict[str, str]:
    identity = dict(record.semantic_identity)
    required = {
        "checkpoint_file_sha256",
        "checkpoint_state_sha256",
        "corpus_sha256",
        "partition_sha256",
        "source_initial_state_sha256",
        "source_manifest_sha256",
        "initialization_state_sha256",
        "selection_start_sha256",
        "target_geometry_sha256",
        "native_channel_count",
        "target_row_count",
        "target_seed",
        "execution_environment_sha256",
    }
    missing = sorted(required - set(identity))
    if missing:
        raise _core.TransferArtifactError(
            f"full-grid semantic identity misses fields {missing}"
        )
    try:
        row_count = int(identity["target_row_count"])
    except ValueError as error:
        raise _core.TransferArtifactError(
            "full-grid target row count is not an integer"
        ) from error
    if row_count <= 0 or str(row_count) != identity["target_row_count"]:
        raise _core.TransferArtifactError(
            "full-grid target row count is invalid"
        )
    try:
        target_seed = int(identity["target_seed"])
    except ValueError as error:
        raise _core.TransferArtifactError(
            "full-grid target seed is not an integer"
        ) from error
    if (
        target_seed not in LOCKED_SEEDS
        or str(target_seed) != identity["target_seed"]
    ):
        raise _core.TransferArtifactError("full-grid target seed is invalid")
    if target_seed != record.key.seed:
        raise _core.TransferArtifactError(
            "full-grid target seed differs from record"
        )
    return identity


def _require_constant_state(
    records: Sequence[_core.ValidatedTransferArtifact],
    *,
    condition: str,
    grouping_fields: tuple[str, ...],
) -> None:
    groups: dict[tuple[str, ...], set[tuple[str, str]]] = {}
    for record in records:
        if record.key.condition != condition:
            continue
        identity = _semantic_identity(record)
        group = tuple(
            str(record.key.seed) if field == "seed" else identity[field]
            for field in grouping_fields
        )
        groups.setdefault(group, set()).add(
            (
                identity["initialization_state_sha256"],
                identity["selection_start_sha256"],
            )
        )
    if any(len(states) != 1 for states in groups.values()):
        raise _core.TransferArtifactError(
            f"{condition} changed initialization/reset within "
            f"{grouping_fields} group"
        )


def _require_seed_effect(
    records: Sequence[_core.ValidatedTransferArtifact],
    *,
    condition: str,
    structural_field: str | None,
) -> None:
    by_structure: dict[str, dict[int, str]] = {}
    for record in records:
        if record.key.condition != condition:
            continue
        identity = _semantic_identity(record)
        structure = "global" if structural_field is None else identity[structural_field]
        seed_states = by_structure.setdefault(structure, {})
        state = identity["initialization_state_sha256"]
        previous = seed_states.setdefault(record.key.seed, state)
        if previous != state:
            raise _core.TransferArtifactError(
                f"{condition} changed initialization within seed/structure group"
            )
    for structure, seed_states in by_structure.items():
        if len(set(seed_states.values())) != len(seed_states):
            raise _core.TransferArtifactError(
                f"{condition} ignored a frozen seed for structure {structure}"
            )


def _validate_outer_fold_coverage(
    records: Sequence[_core.ValidatedTransferArtifact],
) -> None:
    """Verify disjoint OOF rows and exact cache-row coverage when folds are complete."""

    subject_row_counts: dict[tuple[str, int], int] = {}
    grouped: dict[
        tuple[str, int, int, str], list[_core.ValidatedTransferArtifact]
    ] = {}
    for record in records:
        identity = _semantic_identity(record)
        row_count = int(identity["target_row_count"])
        subject_key = (record.key.dataset, record.key.subject)
        previous = subject_row_counts.setdefault(subject_key, row_count)
        if previous != row_count:
            raise _core.TransferArtifactError(
                f"target row count changed within subject {subject_key}"
            )
        if np.any(record.test_rows >= row_count):
            raise _core.TransferArtifactError(
                f"test row lies outside target cache for {record.key.as_dict()}"
            )
        grouped.setdefault(
            (
                record.key.dataset,
                record.key.subject,
                record.key.seed,
                record.key.condition,
            ),
            [],
        ).append(record)

    for group, fold_records in grouped.items():
        dataset = group[0]
        expected_folds = set(LOCKED_FOLDS[dataset])
        observed_folds = {record.key.fold for record in fold_records}
        if not observed_folds <= expected_folds:
            raise _core.TransferArtifactError(
                f"outer-fold group contains unauthorized folds: {group}"
            )
        rows_seen: set[int] = set()
        for record in sorted(fold_records, key=lambda item: item.key.fold):
            overlap = rows_seen.intersection(record.test_rows.tolist())
            if overlap:
                raise _core.TransferArtifactError(
                    f"outer test folds overlap for {group}: {sorted(overlap)}"
                )
            rows_seen.update(record.test_rows.tolist())
        if observed_folds == expected_folds:
            row_count = int(_semantic_identity(fold_records[0])["target_row_count"])
            expected_rows = set(range(row_count))
            if rows_seen != expected_rows:
                missing = sorted(expected_rows - rows_seen)
                extra = sorted(rows_seen - expected_rows)
                raise _core.TransferArtifactError(
                    "outer test-fold union does not cover every cached target row "
                    f"exactly once for {group}; missing={missing}, extra={extra}"
                )


def _validate_full_family_grid_semantics(
    records: Sequence[_core.ValidatedTransferArtifact],
) -> None:
    if not records:
        return
    identities = [_semantic_identity(record) for record in records]
    exact_pins = {
        "checkpoint_file_sha256": PINNED_CHECKPOINT_FILE_SHA256,
        "checkpoint_state_sha256": PINNED_CHECKPOINT_STATE_SHA256,
        "corpus_sha256": PINNED_CORPUS_SHA256,
        "partition_sha256": PINNED_PARTITION_SHA256,
        "source_initial_state_sha256": PINNED_SOURCE_INITIAL_STATE_SHA256,
        "source_manifest_sha256": PINNED_SOURCE_MANIFEST_SHA256,
    }
    for field, expected in exact_pins.items():
        if {identity[field] for identity in identities} != {expected}:
            raise _core.TransferArtifactError(
                f"full FBMS grid differs from frozen semantic identity {field}"
            )
    environments = {
        identity["execution_environment_sha256"] for identity in identities
    }
    if environments != {PINNED_EXECUTION_ENVIRONMENT_SHA256}:
        raise _core.TransferArtifactError(
            "full FBMS grid changed the frozen execution environment"
        )

    # Pretrained starts are seed-independent. Scratch starts are deterministic
    # within seed and model geometry, and must actually change across seeds.
    _require_constant_state(
        records,
        condition=transfer.PRETRAINED_CARDINAL_FBMS,
        grouping_fields=(),
    )
    _require_constant_state(
        records,
        condition=transfer.PRETRAINED_INDEXED_FBMSNET_SPLINE,
        grouping_fields=(),
    )
    _require_constant_state(
        records,
        condition=transfer.SCRATCH_CARDINAL_FBMS_CANONICAL,
        grouping_fields=("seed",),
    )
    _require_constant_state(
        records,
        condition=transfer.SCRATCH_CARDINAL_FBMS_NATIVE,
        grouping_fields=("seed", "target_geometry_sha256"),
    )
    _require_constant_state(
        records,
        condition=transfer.SCRATCH_FBMSNET_NATIVE,
        grouping_fields=("seed", "native_channel_count"),
    )
    _require_seed_effect(
        records,
        condition=transfer.SCRATCH_CARDINAL_FBMS_CANONICAL,
        structural_field=None,
    )
    _require_seed_effect(
        records,
        condition=transfer.SCRATCH_CARDINAL_FBMS_NATIVE,
        structural_field="target_geometry_sha256",
    )
    _require_seed_effect(
        records,
        condition=transfer.SCRATCH_FBMSNET_NATIVE,
        structural_field="native_channel_count",
    )
    _validate_outer_fold_coverage(records)


def _validate_pinned_target_cache_manifest(
    records: Sequence[_core.ValidatedTransferArtifact],
) -> None:
    """Bind the complete 91-subject target corpus to the audited screen."""

    caches: dict[str, str] = {}
    for record in records:
        name = f"{record.key.dataset}:s{record.key.subject}"
        previous = caches.setdefault(name, record.cache_file_sha256)
        if previous != record.cache_file_sha256:
            raise _core.TransferArtifactError(
                f"target cache changed within pinned subject {name}"
            )
    expected_subject_count = sum(
        len(LOCKED_SUBJECTS[dataset]) for dataset in LOCKED_DATASETS
    )
    if len(caches) < expected_subject_count:
        return
    if len(caches) != expected_subject_count:
        raise _core.TransferArtifactError(
            "target cache manifest contains an unauthorized subject"
        )
    encoded = json.dumps(
        caches, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    if hashlib.sha256(encoded).hexdigest() != PINNED_TARGET_CACHE_MANIFEST_SHA256:
        raise _core.TransferArtifactError(
            "target cache manifest differs from the frozen 91-subject screen"
        )


AUDIT_CONTRACT = _core.TransferAuditContract(
    audit_schema=AUDIT_SCHEMA,
    summary_schema=SUMMARY_SCHEMA,
    summary_analysis_policy=SUMMARY_ANALYSIS_POLICY,
    artifact_schema=transfer.ARTIFACT_SCHEMA,
    prediction_schema=transfer.PREDICTION_SCHEMA,
    predictions_filename=transfer.PREDICTIONS_FILENAME,
    provenance_filename=transfer.PROVENANCE_FILENAME,
    evidence_scope=transfer.EVIDENCE_SCOPE,
    analysis_policy=transfer.ANALYSIS_POLICY,
    conditions=LOCKED_CONDITIONS,
    validator=transfer.validate_target_record,
    checkpoint_schema=CHECKPOINT_SCHEMA,
    checkpoint_model_identity="cardinal_fbms",
    required_source_files=(
        "eeg_mi/native_fbms_pretraining_cli.py",
        "eeg_mi/native_fbms_transfer.py",
    ),
    artifact_semantics_validator=_validate_full_artifact_semantics,
    grid_semantics_validator=_validate_full_family_grid_semantics,
)


@dataclass(frozen=True)
class LockedFBMSFullGridAudit:
    cho2017: _core.TransferDirectoryAudit
    physionet_mi: _core.TransferDirectoryAudit

    @property
    def audits(self) -> tuple[_core.TransferDirectoryAudit, ...]:
        return (self.cho2017, self.physionet_mi)

    @property
    def complete(self) -> bool:
        return all(directory.complete for directory in self.audits)

    @property
    def records(self) -> tuple[_core.ValidatedTransferArtifact, ...]:
        return tuple(
            record for directory in self.audits for record in directory.records
        )


def locked_full_grid() -> dict[str, object]:
    return {
        "schema": GRID_SCHEMA,
        "record_name_template": FULL_RECORD_NAME_TEMPLATE,
        "datasets": {
            dataset: {
                "subjects": list(LOCKED_SUBJECTS[dataset]),
                "folds": list(LOCKED_FOLDS[dataset]),
                "seeds": list(LOCKED_SEEDS),
                "conditions": list(LOCKED_CONDITIONS),
                "expected_record_count": EXPECTED_RECORD_COUNT_BY_DATASET[dataset],
            }
            for dataset in LOCKED_DATASETS
        },
        "expected_record_count": EXPECTED_RECORD_COUNT,
    }


def locked_scientific_pins() -> dict[str, object]:
    """Expose the exact artifact identity enforced before any gate analysis."""

    return {
        "checkpoint_file_sha256": PINNED_CHECKPOINT_FILE_SHA256,
        "checkpoint_state_sha256": PINNED_CHECKPOINT_STATE_SHA256,
        "source_initial_state_sha256": PINNED_SOURCE_INITIAL_STATE_SHA256,
        "corpus_sha256": PINNED_CORPUS_SHA256,
        "partition_sha256": PINNED_PARTITION_SHA256,
        "source_manifest_file_count": len(PINNED_SOURCE_MANIFEST),
        "source_manifest_sha256": PINNED_SOURCE_MANIFEST_SHA256,
        "target_cache_subject_count": sum(
            len(LOCKED_SUBJECTS[dataset]) for dataset in LOCKED_DATASETS
        ),
        "target_cache_manifest_sha256": PINNED_TARGET_CACHE_MANIFEST_SHA256,
        "execution_environment_sha256": PINNED_EXECUTION_ENVIRONMENT_SHA256,
    }


def _expected_directory_keys(
    dataset: str,
) -> tuple[_core.TransferRecordKey, ...]:
    return _core.development_transfer_grid(
        dataset=dataset,
        subjects=LOCKED_SUBJECTS[dataset],
        folds=LOCKED_FOLDS[dataset],
        seeds=LOCKED_SEEDS,
        conditions=LOCKED_CONDITIONS,
        contract=AUDIT_CONTRACT,
    )


def _validate_cross_dataset_provenance(
    full_grid: LockedFBMSFullGridAudit,
) -> None:
    records = full_grid.records
    if not records:
        return
    if len({record.checkpoint_file_sha256 for record in records}) != 1:
        raise _core.TransferArtifactError(
            "full FBMS datasets do not share one source checkpoint"
        )
    if len({record.source_code_hashes for record in records}) != 1:
        raise _core.TransferArtifactError(
            "full FBMS datasets do not share one source-code manifest"
        )
    _validate_full_family_grid_semantics(records)
    _validate_pinned_target_cache_manifest(records)


def audit_locked_fbms_full_grid(
    *,
    cho_root: str | Path,
    physionet_root: str | Path,
) -> LockedFBMSFullGridAudit:
    """Audit exactly 8,675 authorized development records."""

    _assert_writer_grid_identity()
    # Authorize and collision-check both complete grids before touching paths.
    for dataset in LOCKED_DATASETS:
        _expected_directory_keys(dataset)
    full_grid = LockedFBMSFullGridAudit(
        cho2017=_core.audit_transfer_directory(
            cho_root,
            dataset=CHO2017,
            subjects=LOCKED_SUBJECTS[CHO2017],
            folds=LOCKED_FOLDS[CHO2017],
            seeds=LOCKED_SEEDS,
            conditions=LOCKED_CONDITIONS,
            name_template=FULL_RECORD_NAME_TEMPLATE,
            contract=AUDIT_CONTRACT,
        ),
        physionet_mi=_core.audit_transfer_directory(
            physionet_root,
            dataset=PHYSIONET_MI,
            subjects=LOCKED_SUBJECTS[PHYSIONET_MI],
            folds=LOCKED_FOLDS[PHYSIONET_MI],
            seeds=LOCKED_SEEDS,
            conditions=LOCKED_CONDITIONS,
            name_template=FULL_RECORD_NAME_TEMPLATE,
            contract=AUDIT_CONTRACT,
        ),
    )
    _validate_cross_dataset_provenance(full_grid)
    if full_grid.complete and len(full_grid.records) != EXPECTED_RECORD_COUNT:
        raise RuntimeError("complete FBMS full grid has an impossible record count")
    return full_grid


def audit_manifest(full_grid: LockedFBMSFullGridAudit) -> dict[str, object]:
    missing = tuple(
        key for directory in full_grid.audits for key in directory.missing_keys
    )
    return {
        "schema": AUDIT_SCHEMA,
        "mode": MODE,
        "confirmation_access": False,
        "confirmatory_evidence": False,
        "analysis_policy": SUMMARY_ANALYSIS_POLICY,
        "locked_grid": locked_full_grid(),
        "locked_scientific_pins": locked_scientific_pins(),
        "complete": full_grid.complete,
        "expected_record_count": EXPECTED_RECORD_COUNT,
        "validated_record_count": len(full_grid.records),
        "missing_record_count": len(missing),
        "missing_records": [key.as_dict() for key in missing],
        "datasets": {
            CHO2017: _core.audit_manifest(full_grid.cho2017),
            PHYSIONET_MI: _core.audit_manifest(full_grid.physionet_mi),
        },
    }


def descriptive_full_grid_summary(
    full_grid: LockedFBMSFullGridAudit,
    *,
    require_complete: bool = True,
) -> dict[str, object]:
    """Compute OOF subject scores without treating folds/seeds as subjects."""

    if require_complete and not full_grid.complete:
        raise RuntimeError("refusing to summarize an incomplete FBMS full grid")
    dataset_summaries = {
        CHO2017: _core.descriptive_transfer_summary(
            full_grid.cho2017, require_complete=require_complete
        ),
        PHYSIONET_MI: _core.descriptive_transfer_summary(
            full_grid.physionet_mi, require_complete=require_complete
        ),
    }
    condition_means = {
        dataset: {
            condition: float(
                dataset_summaries[dataset]["conditions"][condition][
                    "equal_subject_mean_balanced_accuracy"
                ]
            )
            for condition in LOCKED_CONDITIONS
        }
        for dataset in LOCKED_DATASETS
    }
    equal_dataset_means = {
        condition: float(
            np.mean(
                [condition_means[dataset][condition] for dataset in LOCKED_DATASETS]
            )
        )
        for condition in LOCKED_CONDITIONS
    }
    for dataset in LOCKED_DATASETS:
        conditions = dataset_summaries[dataset]["conditions"]
        for condition in LOCKED_CONDITIONS:
            summary = conditions[condition]
            if summary["n_records_observed"] != (
                len(LOCKED_SUBJECTS[dataset])
                * len(LOCKED_FOLDS[dataset])
                * len(LOCKED_SEEDS)
            ):
                raise RuntimeError("full-grid summary record-unit count is inconsistent")
            if summary["n_subject_seed_units_observed"] != (
                len(LOCKED_SUBJECTS[dataset]) * len(LOCKED_SEEDS)
            ):
                raise RuntimeError(
                    "full-grid summary subject/seed-unit count is inconsistent"
                )
            if summary["n_subjects_observed"] != len(LOCKED_SUBJECTS[dataset]):
                raise RuntimeError("full-grid summary subject count is inconsistent")
    return {
        "schema": SUMMARY_SCHEMA,
        "mode": MODE,
        "confirmation_access": False,
        "confirmatory_evidence": False,
        "analysis_policy": SUMMARY_ANALYSIS_POLICY,
        "inferential_statistics": False,
        "scientific_unit": "subject",
        "aggregation": {
            "folds": (
                "concatenate disjoint outer-test predictions within "
                "dataset/subject/seed/condition before scoring"
            ),
            "fold_coverage": "every cached target row exactly once",
            "seeds": "arithmetic mean of OOF seed scores within subject/condition",
            "subjects": "equal-subject arithmetic mean within dataset/condition",
            "datasets": "equal-dataset arithmetic mean",
            "metric": "binary balanced accuracy",
            "pseudoreplicates": "folds and seeds are never inferential units",
        },
        "complete_grid": full_grid.complete,
        "locked_grid": locked_full_grid(),
        "expected_record_count": EXPECTED_RECORD_COUNT,
        "validated_record_count": len(full_grid.records),
        "dataset_condition_mean_balanced_accuracy": condition_means,
        "equal_dataset_condition_mean_balanced_accuracy": equal_dataset_means,
        "datasets": dataset_summaries,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cho-root", required=True, type=Path)
    parser.add_argument("--physionet-root", required=True, type=Path)
    return parser


def run(argv: Sequence[str] | None = None) -> dict[str, object]:
    arguments = build_parser().parse_args(argv)
    full_grid = audit_locked_fbms_full_grid(
        cho_root=arguments.cho_root,
        physionet_root=arguments.physionet_root,
    )
    return {
        "audit": audit_manifest(full_grid),
        "summary": descriptive_full_grid_summary(full_grid),
    }


def main() -> None:
    print(json.dumps(run(), indent=2, sort_keys=True, allow_nan=False))


__all__ = [
    "AUDIT_CONTRACT",
    "AUDIT_SCHEMA",
    "CANONICAL_SCRATCH_STATE_SHA256_BY_SEED",
    "CHO2017",
    "EXPECTED_RECORD_COUNT",
    "EXPECTED_RECORD_COUNT_BY_DATASET",
    "FULL_RECORD_NAME_TEMPLATE",
    "GRID_SCHEMA",
    "LOCKED_CONDITIONS",
    "LOCKED_DATASETS",
    "LOCKED_FOLDS",
    "LOCKED_SEEDS",
    "LOCKED_SUBJECTS",
    "LockedFBMSFullGridAudit",
    "PINNED_CHECKPOINT_FILE_SHA256",
    "PINNED_CHECKPOINT_STATE_SHA256",
    "PINNED_CORPUS_SHA256",
    "PINNED_EXECUTION_ENVIRONMENT",
    "PINNED_EXECUTION_ENVIRONMENT_SHA256",
    "PINNED_PARTITION_SHA256",
    "PINNED_SOURCE_INITIAL_STATE_SHA256",
    "PINNED_SOURCE_MANIFEST",
    "PINNED_SOURCE_MANIFEST_SHA256",
    "PINNED_TARGET_CACHE_MANIFEST_SHA256",
    "PINNED_TRANSFER_SOURCE_SHA256",
    "PHYSIONET_MI",
    "SUMMARY_SCHEMA",
    "audit_locked_fbms_full_grid",
    "audit_manifest",
    "descriptive_full_grid_summary",
    "locked_full_grid",
    "locked_scientific_pins",
]


if __name__ == "__main__":  # pragma: no cover
    main()
