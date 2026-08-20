from __future__ import annotations

import copy
import hashlib
import inspect
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from benchmark import native_fbms_full_grid_audit as audit
from benchmark import native_fbms_full_grid_gate as gate
from benchmark import native_fbms_transfer as transfer
from benchmark import native_fbms_transfer_audit as screen_audit
from benchmark import native_transfer_audit as core


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _pinned_provenance() -> dict[str, object]:
    return {
        "environment": audit.PINNED_EXECUTION_ENVIRONMENT,
        "protocol": {"train_config": {"device": "cuda"}},
        "source_checkpoint": {
            "file_sha256": audit.PINNED_CHECKPOINT_FILE_SHA256,
            "state_sha256": audit.PINNED_CHECKPOINT_STATE_SHA256,
            "corpus": {"sha256": audit.PINNED_CORPUS_SHA256},
            "pretraining": {
                "partition": {"sha256": audit.PINNED_PARTITION_SHA256},
                "state_hashes": {
                    "initial": audit.PINNED_SOURCE_INITIAL_STATE_SHA256
                },
            },
        },
        "source_code": {
            "hash_algorithm": "sha256",
            "files": audit.PINNED_SOURCE_MANIFEST,
        },
    }


def _condition_state(
    condition: str,
    *,
    dataset: str,
    seed: int,
) -> tuple[str, str, str, str]:
    geometry = _digest(f"geometry-{dataset}")
    if condition == transfer.PRETRAINED_CARDINAL_FBMS:
        return _digest("candidate-init"), _digest("candidate-reset"), geometry, "2"
    if condition == transfer.PRETRAINED_INDEXED_FBMSNET_SPLINE:
        return _digest("indexed-pretrained"), _digest("indexed-reset"), geometry, "2"
    if condition == transfer.SCRATCH_CARDINAL_FBMS_CANONICAL:
        state = _digest(f"canonical-{seed}")
        return state, state, geometry, "2"
    if condition == transfer.SCRATCH_CARDINAL_FBMS_NATIVE:
        state = _digest(f"native-cardinal-{seed}-{geometry}")
        return state, state, geometry, "2"
    if condition == transfer.SCRATCH_FBMSNET_NATIVE:
        state = _digest(f"native-indexed-{seed}-2")
        return state, state, geometry, "2"
    raise AssertionError(condition)


def _predicted(condition: str, labels: np.ndarray) -> np.ndarray:
    if condition == transfer.PRETRAINED_CARDINAL_FBMS:
        return labels.copy()
    if condition == transfer.SCRATCH_FBMSNET_NATIVE:
        return 1 - labels
    return np.asarray([0, 0], dtype=np.int64)


def _record(key: core.TransferRecordKey) -> core.ValidatedTransferArtifact:
    fold_count = len(audit.LOCKED_FOLDS[key.dataset])
    row_count = 2 * fold_count
    rows = np.asarray([2 * key.fold, 2 * key.fold + 1], dtype=np.int64)
    labels = np.asarray([0, 1], dtype=np.int64)
    predicted = _predicted(key.condition, labels)
    probabilities = np.full((2, 2), 0.1, dtype=np.float64)
    probabilities[np.arange(2), predicted] = 0.9
    initialization, selection, geometry, channels = _condition_state(
        key.condition, dataset=key.dataset, seed=key.seed
    )
    identity = {
        "checkpoint_file_sha256": audit.PINNED_CHECKPOINT_FILE_SHA256,
        "checkpoint_state_sha256": audit.PINNED_CHECKPOINT_STATE_SHA256,
        "corpus_sha256": audit.PINNED_CORPUS_SHA256,
        "partition_sha256": audit.PINNED_PARTITION_SHA256,
        "source_initial_state_sha256": audit.PINNED_SOURCE_INITIAL_STATE_SHA256,
        "source_manifest_sha256": audit.PINNED_SOURCE_MANIFEST_SHA256,
        "execution_environment_sha256": audit.PINNED_EXECUTION_ENVIRONMENT_SHA256,
        "initialization_state_sha256": initialization,
        "selection_start_sha256": selection,
        "target_geometry_sha256": geometry,
        "native_channel_count": channels,
        "target_row_count": str(row_count),
        "target_seed": str(key.seed),
    }
    return core.ValidatedTransferArtifact(
        key=key,
        path=Path("/synthetic") / core.record_directory_name(
            key, name_template=audit.FULL_RECORD_NAME_TEMPLATE
        ),
        predictions_file_sha256=_digest(f"predictions-{key}"),
        provenance_file_sha256=_digest(f"provenance-{key}"),
        checkpoint_file_sha256=_digest("checkpoint-file"),
        cache_file_sha256=_digest(f"cache-{key.dataset}-{key.subject}"),
        source_code_hashes=(("eeg_mi/native_fbms_transfer.py", _digest("source")),),
        test_rows=rows,
        test_labels=labels,
        predicted_labels=predicted,
        probabilities=probabilities,
        sessions=np.asarray(["session", "session"]),
        runs=np.asarray([f"fold-{key.fold}", f"fold-{key.fold}"]),
        semantic_identity=tuple(sorted(identity.items())),
    )


def _directory(tmp_path: Path, dataset: str) -> core.TransferDirectoryAudit:
    root = tmp_path / dataset
    root.mkdir()
    expected = audit._expected_directory_keys(dataset)
    records = tuple(_record(key) for key in expected)
    return core.TransferDirectoryAudit(
        root=root,
        name_template=audit.FULL_RECORD_NAME_TEMPLATE,
        expected_keys=expected,
        records=records,
        missing_keys=(),
        contract=audit.AUDIT_CONTRACT,
    )


def _full_grid(tmp_path: Path) -> audit.LockedFBMSFullGridAudit:
    return audit.LockedFBMSFullGridAudit(
        cho2017=_directory(tmp_path, audit.CHO2017),
        physionet_mi=_directory(tmp_path, audit.PHYSIONET_MI),
    )


def _mutate_identity(
    record: core.ValidatedTransferArtifact,
    **updates: str,
) -> core.ValidatedTransferArtifact:
    identity = dict(record.semantic_identity)
    identity.update(updates)
    return replace(record, semantic_identity=tuple(sorted(identity.items())))


def test_full_grid_is_exact_and_has_no_subset_surface() -> None:
    grid = audit.locked_full_grid()
    assert audit.EXPECTED_RECORD_COUNT_BY_DATASET == {
        audit.CHO2017: 4_625,
        audit.PHYSIONET_MI: 4_050,
    }
    assert audit.EXPECTED_RECORD_COUNT == 8_675
    assert grid["record_name_template"] == (
        "s{subject:03d}_f{fold}_seed{seed}_{condition}"
    )
    assert grid["datasets"][audit.CHO2017]["folds"] == [0, 1, 2, 3, 4]
    assert grid["datasets"][audit.PHYSIONET_MI]["folds"] == [0, 1, 2]
    assert grid["datasets"][audit.CHO2017]["seeds"] == [7, 17, 27, 37, 47]
    assert audit.CANONICAL_SCRATCH_STATE_SHA256_BY_SEED == {
        7: "382d4e18385904b885e4140963705939b7d935ae95731ac402d3a447eabef507",
        17: "876e995189902d14e1f9424edeafd132be8185f0caf8a60ff6dd8304824f09a3",
        27: "76b023e24ac63260e3ae3fa1d22709787f7b52d78e496c0a761342aed82ab079",
        37: "486aa5fe51783925228d0fc4438ad3415ce82ca714aee23341bbc4c76f0c006b",
        47: "2a2943e16de9117d0f2bfbfe6ea957cb7be67710aae385f4b20a90b87b5b6af1",
    }
    assert tuple(grid["datasets"][audit.CHO2017]["conditions"]) == (
        transfer.TRANSFER_CONDITIONS
    )
    for parser in (audit.build_parser(), gate.build_parser()):
        destinations = {action.dest for action in parser._actions}
        assert destinations == {"help", "cho_root", "physionet_root"}
    assert tuple(
        inspect.signature(audit.audit_locked_fbms_full_grid).parameters
    ) == ("cho_root", "physionet_root")
    assert tuple(inspect.signature(gate.evaluate_fbms_full_grid_gate).parameters) == (
        "full_grid",
    )
    with pytest.raises(TypeError, match="LockedFBMSFullGridAudit"):
        gate.evaluate_fbms_full_grid_gate({})  # type: ignore[arg-type]
    assert audit.locked_scientific_pins() == {
        "checkpoint_file_sha256": audit.PINNED_CHECKPOINT_FILE_SHA256,
        "checkpoint_state_sha256": audit.PINNED_CHECKPOINT_STATE_SHA256,
        "source_initial_state_sha256": audit.PINNED_SOURCE_INITIAL_STATE_SHA256,
        "corpus_sha256": audit.PINNED_CORPUS_SHA256,
        "partition_sha256": audit.PINNED_PARTITION_SHA256,
        "source_manifest_file_count": 11,
        "source_manifest_sha256": audit.PINNED_SOURCE_MANIFEST_SHA256,
        "target_cache_subject_count": 91,
        "target_cache_manifest_sha256": audit.PINNED_TARGET_CACHE_MANIFEST_SHA256,
        "execution_environment_sha256": audit.PINNED_EXECUTION_ENVIRONMENT_SHA256,
    }


def test_full_artifact_pins_reject_forged_identity_manifest_and_environment() -> None:
    provenance = _pinned_provenance()
    identity = audit._validate_experiment_pins(provenance)
    assert identity["checkpoint_file_sha256"] == audit.PINNED_CHECKPOINT_FILE_SHA256
    assert identity["source_manifest_sha256"] == audit.PINNED_SOURCE_MANIFEST_SHA256
    assert (
        audit._validate_execution_environment(provenance)
        == audit.PINNED_EXECUTION_ENVIRONMENT_SHA256
    )

    mutations = []
    forged_checkpoint = copy.deepcopy(provenance)
    forged_checkpoint["source_checkpoint"]["file_sha256"] = "0" * 64
    mutations.append(forged_checkpoint)
    forged_corpus = copy.deepcopy(provenance)
    forged_corpus["source_checkpoint"]["corpus"]["sha256"] = "0" * 64
    mutations.append(forged_corpus)
    forged_partition = copy.deepcopy(provenance)
    forged_partition["source_checkpoint"]["pretraining"]["partition"][
        "sha256"
    ] = "0" * 64
    mutations.append(forged_partition)
    forged_initial = copy.deepcopy(provenance)
    forged_initial["source_checkpoint"]["pretraining"]["state_hashes"][
        "initial"
    ] = "0" * 64
    mutations.append(forged_initial)
    forged_manifest = copy.deepcopy(provenance)
    forged_manifest["source_code"]["files"]["requirements.txt"] = "0" * 64
    mutations.append(forged_manifest)
    for forged in mutations:
        with pytest.raises(core.TransferArtifactError):
            audit._validate_experiment_pins(forged)

    forged_environment = copy.deepcopy(provenance)
    forged_environment["environment"]["packages"]["torch"] = "2.11.1+cu128"
    with pytest.raises(core.TransferArtifactError, match="execution environment"):
        audit._validate_execution_environment(forged_environment)
    forged_device = copy.deepcopy(provenance)
    forged_device["protocol"]["train_config"]["device"] = "cpu"
    with pytest.raises(core.TransferArtifactError, match="protocol device"):
        audit._validate_execution_environment(forged_device)


def test_complete_target_cache_manifest_rejects_forged_subject(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = tuple(
        _record(
            core.TransferRecordKey(
                dataset=dataset,
                subject=subject,
                fold=0,
                seed=7,
                condition=transfer.PRETRAINED_CARDINAL_FBMS,
            )
        )
        for dataset in audit.LOCKED_DATASETS
        for subject in audit.LOCKED_SUBJECTS[dataset]
    )
    manifest = {
        f"{record.key.dataset}:s{record.key.subject}": record.cache_file_sha256
        for record in records
    }
    synthetic_hash = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    monkeypatch.setattr(
        audit, "PINNED_TARGET_CACHE_MANIFEST_SHA256", synthetic_hash
    )
    audit._validate_pinned_target_cache_manifest(records)

    forged = list(records)
    forged[0] = replace(forged[0], cache_file_sha256="0" * 64)
    with pytest.raises(core.TransferArtifactError, match="target cache manifest"):
        audit._validate_pinned_target_cache_manifest(forged)

    assert (
        "e6ed4800c6bb1eff284a80fc2b1c240b1b63cecbbe427af5321784990354935e"
        != synthetic_hash
    )


def test_full_summary_uses_oof_subject_unit_without_pseudoreplication(
    tmp_path: Path,
) -> None:
    full_grid = _full_grid(tmp_path)
    assert len(full_grid.records) == 8_675
    assert audit.audit_manifest(full_grid)["locked_scientific_pins"] == (
        audit.locked_scientific_pins()
    )
    audit._validate_full_family_grid_semantics(full_grid.records)
    summary = audit.descriptive_full_grid_summary(full_grid)
    assert summary["scientific_unit"] == "subject"
    assert summary["aggregation"]["fold_coverage"] == (
        "every cached target row exactly once"
    )
    assert summary["aggregation"]["pseudoreplicates"] == (
        "folds and seeds are never inferential units"
    )
    for dataset in audit.LOCKED_DATASETS:
        conditions = summary["datasets"][dataset]["conditions"]
        candidate = conditions[transfer.PRETRAINED_CARDINAL_FBMS]
        assert candidate["n_records_observed"] == (
            len(audit.LOCKED_SUBJECTS[dataset])
            * len(audit.LOCKED_FOLDS[dataset])
            * len(audit.LOCKED_SEEDS)
        )
        assert candidate["n_subject_seed_units_observed"] == (
            len(audit.LOCKED_SUBJECTS[dataset]) * len(audit.LOCKED_SEEDS)
        )
        assert candidate["n_subjects_observed"] == len(
            audit.LOCKED_SUBJECTS[dataset]
        )
        assert candidate["equal_subject_mean_balanced_accuracy"] == 1.0
        assert conditions[transfer.SCRATCH_CARDINAL_FBMS_CANONICAL][
            "equal_subject_mean_balanced_accuracy"
        ] == 0.5
    assert summary["equal_dataset_condition_mean_balanced_accuracy"][
        transfer.PRETRAINED_CARDINAL_FBMS
    ] == 1.0


def test_gate_revalidates_and_computes_its_own_subject_bootstrap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full_grid = _full_grid(tmp_path)
    calls: list[tuple[Path, Path]] = []

    def revalidate(*, cho_root: Path, physionet_root: Path):
        calls.append((Path(cho_root), Path(physionet_root)))
        return full_grid

    monkeypatch.setattr(audit, "audit_locked_fbms_full_grid", revalidate)
    decision = gate.evaluate_fbms_full_grid_gate(full_grid)
    assert calls == [(full_grid.cho2017.root, full_grid.physionet_mi.root)]
    assert decision["overall_pass"] is True
    assert all(check["pass"] for check in decision["gate_checks"].values())
    bootstrap = decision["inputs"][
        "internally_computed_paired_subject_bootstrap"
    ]
    assert bootstrap["resampling_unit"] == "subject"
    assert bootstrap["repetitions"] == 200_000
    assert bootstrap["seed"] == 20_260_720
    assert bootstrap["datasets"][audit.CHO2017]["subject_count"] == 37
    assert bootstrap["datasets"][audit.PHYSIONET_MI]["subject_count"] == 54
    assert bootstrap["equal_dataset_macro"]["one_sided_95_lower"] == 0.5
    primary = decision["inputs"][
        "internally_computed_primary_checkpoint_matched_paired_subject_bootstrap"
    ]
    assert primary["comparator_condition"] == (
        transfer.PRETRAINED_INDEXED_FBMSNET_SPLINE
    )
    assert primary["seed"] == gate.PRIMARY_COMPARATOR_BOOTSTRAP_SEED
    assert primary["equal_dataset_macro"]["one_sided_95_lower"] == 0.5
    primary_check = decision["gate_checks"][
        "candidate_minus_primary_checkpoint_matched_comparator_equal_dataset_macro_one_sided_95_lower_above_zero"
    ]
    assert primary_check["pass"] is True
    assert primary_check["reported_dataset_intervals_are_gate_criteria"] is False
    assert decision["locked_contract"]["scientific_pins"] == (
        audit.locked_scientific_pins()
    )
    assert decision["locked_contract"][
        "primary_checkpoint_matched_comparator_scope"
    ] == (
        "development-only checkpoint-matched indexed projection control; "
        "not an author-faithful FBMSNet reproduction"
    )
    assert len(
        decision["inputs"]["freshly_revalidated_full_grid_audit_sha256"]
    ) == 64


@pytest.mark.parametrize("seed", (17, 27, 37, 47))
def test_nondefault_canonical_seed_is_checked_against_frozen_gpu_digest(
    monkeypatch: pytest.MonkeyPatch,
    seed: int,
) -> None:
    expected = _digest(f"expected-{seed}")
    source_initial = _digest("source-seed-7-initial")
    provenance: dict[str, object] = {
        "target": {"raw_shape": [10, 2, 320], "channels": ["C0", "C1"]},
        "condition": {"initialization_state_sha256": expected},
        "protocol": {"train_config": {"seed": seed}},
        "source_checkpoint": {
            "pretraining": {"state_hashes": {"initial": source_initial}}
        },
    }
    key = core.TransferRecordKey(
        dataset=audit.CHO2017,
        subject=16,
        fold=0,
        seed=seed,
        condition=transfer.SCRATCH_CARDINAL_FBMS_CANONICAL,
    )
    monkeypatch.setattr(
        audit,
        "_expected_canonical_scratch_state_sha256",
        lambda observed_seed: expected if observed_seed == seed else _digest("other"),
    )
    monkeypatch.setattr(
        audit,
        "_validate_experiment_pins",
        lambda unused: {
            "checkpoint_file_sha256": audit.PINNED_CHECKPOINT_FILE_SHA256,
            "checkpoint_state_sha256": audit.PINNED_CHECKPOINT_STATE_SHA256,
            "corpus_sha256": audit.PINNED_CORPUS_SHA256,
            "partition_sha256": audit.PINNED_PARTITION_SHA256,
            "source_initial_state_sha256": source_initial,
            "source_manifest_sha256": audit.PINNED_SOURCE_MANIFEST_SHA256,
        },
    )
    monkeypatch.setattr(
        audit,
        "_validate_execution_environment",
        lambda unused: audit.PINNED_EXECUTION_ENVIRONMENT_SHA256,
    )

    def base_validator(adjusted, observed_key):
        assert observed_key == replace(key, seed=7)
        assert adjusted["protocol"]["train_config"]["seed"] == 7
        hashes = adjusted["source_checkpoint"]["pretraining"]["state_hashes"]
        assert hashes["initial"] == expected
        return {
            "checkpoint_state_sha256": _digest("checkpoint"),
            "source_initial_state_sha256": expected,
            "initialization_state_sha256": expected,
            "selection_start_sha256": expected,
            "target_geometry_sha256": _digest("geometry"),
            "native_channel_count": "2",
        }

    monkeypatch.setattr(
        screen_audit, "_validate_family_artifact_semantics", base_validator
    )
    identity = audit._validate_full_artifact_semantics(provenance, key)
    assert identity["source_initial_state_sha256"] == source_initial
    assert identity["target_row_count"] == "10"
    assert identity["target_seed"] == str(seed)
    provenance["condition"]["initialization_state_sha256"] = _digest("forged")
    with pytest.raises(core.TransferArtifactError, match="frozen x86/GPU"):
        audit._validate_full_artifact_semantics(provenance, key)


@pytest.mark.parametrize("condition", transfer.TRANSFER_CONDITIONS)
def test_every_condition_accepts_a_locked_nondefault_target_seed_structurally(
    monkeypatch: pytest.MonkeyPatch,
    condition: str,
) -> None:
    seed = 17
    expected = audit.CANONICAL_SCRATCH_STATE_SHA256_BY_SEED[seed]
    source_initial = audit.PINNED_SOURCE_INITIAL_STATE_SHA256
    provenance: dict[str, object] = {
        "target": {"raw_shape": [10, 2, 320], "channels": ["C0", "C1"]},
        "condition": {"initialization_state_sha256": expected},
        "protocol": {
            "train_config": {"seed": seed, "device": "cuda"},
            "sentinel": "must-survive",
        },
        "source_checkpoint": {
            "pretraining": {"state_hashes": {"initial": source_initial}}
        },
    }
    original = copy.deepcopy(provenance)
    key = core.TransferRecordKey(
        dataset=audit.CHO2017,
        subject=16,
        fold=1,
        seed=seed,
        condition=condition,
    )
    monkeypatch.setattr(
        audit,
        "_validate_experiment_pins",
        lambda unused: {
            "checkpoint_file_sha256": audit.PINNED_CHECKPOINT_FILE_SHA256,
            "checkpoint_state_sha256": audit.PINNED_CHECKPOINT_STATE_SHA256,
            "corpus_sha256": audit.PINNED_CORPUS_SHA256,
            "partition_sha256": audit.PINNED_PARTITION_SHA256,
            "source_initial_state_sha256": source_initial,
            "source_manifest_sha256": audit.PINNED_SOURCE_MANIFEST_SHA256,
        },
    )
    monkeypatch.setattr(
        audit,
        "_validate_execution_environment",
        lambda unused: audit.PINNED_EXECUTION_ENVIRONMENT_SHA256,
    )

    def base_validator(adjusted, observed_key):
        assert observed_key == replace(key, seed=7)
        assert adjusted["protocol"]["train_config"] == {
            "seed": 7,
            "device": "cuda",
        }
        assert adjusted["protocol"]["sentinel"] == "must-survive"
        adjusted_source_initial = adjusted["source_checkpoint"]["pretraining"][
            "state_hashes"
        ]["initial"]
        if condition == transfer.SCRATCH_CARDINAL_FBMS_CANONICAL:
            assert adjusted_source_initial == expected
        else:
            assert adjusted_source_initial == source_initial
        return {
            "checkpoint_state_sha256": _digest("checkpoint"),
            "source_initial_state_sha256": adjusted_source_initial,
            "initialization_state_sha256": expected,
            "selection_start_sha256": expected,
            "target_geometry_sha256": _digest("geometry"),
            "native_channel_count": "2",
        }

    monkeypatch.setattr(
        screen_audit, "_validate_family_artifact_semantics", base_validator
    )
    identity = audit._validate_full_artifact_semantics(provenance, key)
    assert provenance == original
    assert identity["target_seed"] == str(seed)
    assert identity["source_initial_state_sha256"] == source_initial


def test_target_protocol_seed_must_equal_record_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provenance: dict[str, object] = {
        "target": {"raw_shape": [10, 2, 320], "channels": ["C0", "C1"]},
        "protocol": {"train_config": {"seed": 7}},
    }
    key = core.TransferRecordKey(
        dataset=audit.CHO2017,
        subject=16,
        fold=0,
        seed=17,
        condition=transfer.PRETRAINED_CARDINAL_FBMS,
    )
    monkeypatch.setattr(audit, "_validate_experiment_pins", lambda unused: {})
    monkeypatch.setattr(
        audit, "_validate_execution_environment", lambda unused: _digest("environment")
    )
    with pytest.raises(core.TransferArtifactError, match="seed differs from record"):
        audit._validate_full_artifact_semantics(provenance, key)


@pytest.mark.parametrize("observed", (True, 17.0, "17"))
def test_target_protocol_seed_must_be_an_exact_integer(
    observed: object,
) -> None:
    provenance = {"protocol": {"train_config": {"seed": observed}}}
    key = core.TransferRecordKey(
        dataset=audit.CHO2017,
        subject=16,
        fold=0,
        seed=17,
        condition=transfer.PRETRAINED_CARDINAL_FBMS,
    )
    with pytest.raises(core.TransferArtifactError, match="must be an integer"):
        audit._validate_target_seed(provenance, key)


def test_target_protocol_seed_must_belong_to_the_locked_grid() -> None:
    provenance = {"protocol": {"train_config": {"seed": 57}}}
    key = core.TransferRecordKey(
        dataset=audit.CHO2017,
        subject=16,
        fold=0,
        seed=57,
        condition=transfer.PRETRAINED_CARDINAL_FBMS,
    )
    with pytest.raises(core.TransferArtifactError, match="outside the frozen"):
        audit._validate_target_seed(provenance, key)


def test_semantic_identity_cannot_ignore_the_actual_target_seed() -> None:
    key = core.TransferRecordKey(
        dataset=audit.CHO2017,
        subject=16,
        fold=0,
        seed=17,
        condition=transfer.PRETRAINED_CARDINAL_FBMS,
    )
    ignored = _mutate_identity(_record(key), target_seed="7")
    with pytest.raises(core.TransferArtifactError, match="seed differs from record"):
        audit._semantic_identity(ignored)


@pytest.mark.parametrize(
    "condition",
    (
        transfer.SCRATCH_CARDINAL_FBMS_CANONICAL,
        transfer.SCRATCH_CARDINAL_FBMS_NATIVE,
        transfer.SCRATCH_FBMSNET_NATIVE,
    ),
)
def test_grid_rejects_a_scratch_condition_that_ignores_seed(
    tmp_path: Path,
    condition: str,
) -> None:
    full_grid = _full_grid(tmp_path)
    selected = [
        record
        for record in full_grid.cho2017.records
        if record.key.subject == 16
        and record.key.fold == 0
        and record.key.condition == condition
    ]
    seed7 = next(record for record in selected if record.key.seed == 7)
    seed17 = next(record for record in selected if record.key.seed == 17)
    seed7_identity = dict(seed7.semantic_identity)
    forged = _mutate_identity(
        seed17,
        initialization_state_sha256=seed7_identity[
            "initialization_state_sha256"
        ],
        selection_start_sha256=seed7_identity["selection_start_sha256"],
    )
    records = [forged if record is seed17 else record for record in selected]
    with pytest.raises(core.TransferArtifactError, match="ignored a frozen seed"):
        audit._validate_full_family_grid_semantics(records)


def test_outer_fold_union_must_cover_every_cache_row_exactly_once(
    tmp_path: Path,
) -> None:
    full_grid = _full_grid(tmp_path)
    group = [
        record
        for record in full_grid.cho2017.records
        if record.key.subject == 16
        and record.key.seed == 7
        and record.key.condition == transfer.PRETRAINED_CARDINAL_FBMS
    ]
    assert len(group) == 5
    missing_last_row = replace(
        group[-1], test_rows=np.asarray([8], dtype=np.int64)
    )
    broken = [missing_last_row if record is group[-1] else record for record in group]
    with pytest.raises(core.TransferArtifactError, match="does not cover every"):
        audit._validate_outer_fold_coverage(broken)

    overlapping = replace(
        group[-1], test_rows=np.asarray([0, 9], dtype=np.int64)
    )
    broken = [overlapping if record is group[-1] else record for record in group]
    with pytest.raises(core.TransferArtifactError, match="overlap"):
        audit._validate_outer_fold_coverage(broken)


def test_forged_or_stale_full_grid_cannot_reach_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full_grid = _full_grid(tmp_path)
    forged_directory = replace(
        full_grid.cho2017,
        name_template=core.DEFAULT_RECORD_NAME_TEMPLATE,
    )
    forged = replace(full_grid, cho2017=forged_directory)
    with pytest.raises(ValueError, match="name template"):
        gate.evaluate_fbms_full_grid_gate(forged)

    def stale(**_kwargs):
        raise core.TransferArtifactError("stale artifact bytes")

    monkeypatch.setattr(audit, "audit_locked_fbms_full_grid", stale)
    with pytest.raises(core.TransferArtifactError, match="stale artifact bytes"):
        gate.evaluate_fbms_full_grid_gate(full_grid)
