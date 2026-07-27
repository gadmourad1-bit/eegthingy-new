from __future__ import annotations

import hashlib
import inspect
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

from ieee_mi import native_fbms_pretraining_cli as pretrain
from ieee_mi import native_fbms_transfer as transfer
from ieee_mi import native_fbms_transfer_audit as audit
from ieee_mi import native_fbms_transfer_gate as gate
from ieee_mi import native_transfer_audit as core
from ieee_mi.training import TrainConfig


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _key(dataset: str, subject: int, condition: str) -> core.TransferRecordKey:
    return core.TransferRecordKey(
        dataset=dataset,
        subject=subject,
        fold=0,
        seed=7,
        condition=condition,
    )


def _predictions_for_condition(
    condition: str, labels: np.ndarray
) -> np.ndarray:
    if condition == transfer.PRETRAINED_CARDINAL_FBMS:
        return labels.copy()
    if condition == transfer.SCRATCH_FBMSNET_NATIVE:
        return 1 - labels
    # Two correct and two incorrect: binary balanced accuracy 0.5.
    return np.asarray([0, 1, 1, 0], dtype=np.int64)


def _write_record(
    root: Path,
    key: core.TransferRecordKey,
    *,
    checkpoint_digest: str | None = None,
    cache_digest: str | None = None,
) -> Path:
    path = root / core.record_directory_name(key)
    path.mkdir()
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    predicted = _predictions_for_condition(key.condition, labels)
    rows = np.arange(len(labels), dtype=np.int64)
    probabilities = np.full((len(labels), 2), 0.1, dtype=np.float64)
    probabilities[np.arange(len(labels)), predicted] = 0.9
    predictions_path = path / transfer.PREDICTIONS_FILENAME
    np.savez_compressed(
        predictions_path,
        schema=np.asarray(transfer.PREDICTION_SCHEMA),
        dataset=np.asarray(key.dataset),
        subject=np.asarray(key.subject, dtype=np.int64),
        fold=np.asarray(key.fold, dtype=np.int64),
        condition=np.asarray(key.condition),
        test_rows=rows,
        test_labels=labels,
        predicted_labels=predicted,
        probabilities=probabilities,
        sessions=np.asarray(["session"] * len(labels)),
        runs=np.asarray(["run"] * len(labels)),
    )
    if checkpoint_digest is None:
        checkpoint_digest = hashlib.sha256(b"locked-fbms-checkpoint").hexdigest()
    if cache_digest is None:
        cache_digest = hashlib.sha256(
            f"cache-{key.dataset}-{key.subject}".encode()
        ).hexdigest()
    source_files = {
        "ieee_mi/native_fbms_pretraining_cli.py": hashlib.sha256(
            b"pretraining-source"
        ).hexdigest(),
        "ieee_mi/native_fbms_transfer.py": hashlib.sha256(
            b"transfer-source"
        ).hexdigest(),
        "ieee_mi/models.py": hashlib.sha256(b"model-source").hexdigest(),
    }
    checkpoint_state = hashlib.sha256(b"state").hexdigest()
    source_initial = hashlib.sha256(b"source-initial").hexdigest()
    if key.condition == transfer.PRETRAINED_CARDINAL_FBMS:
        initialization_hash = checkpoint_state
    elif key.condition == transfer.SCRATCH_CARDINAL_FBMS_CANONICAL:
        initialization_hash = source_initial
    elif key.condition == transfer.PRETRAINED_INDEXED_FBMSNET_SPLINE:
        initialization_hash = hashlib.sha256(b"indexed-derived-state").hexdigest()
    elif key.condition == transfer.SCRATCH_CARDINAL_FBMS_NATIVE:
        initialization_hash = hashlib.sha256(
            b"native-cardinal-scratch-two-channel-geometry"
        ).hexdigest()
    elif key.condition == transfer.SCRATCH_FBMSNET_NATIVE:
        initialization_hash = hashlib.sha256(
            b"native-indexed-fbms-scratch-two-channels"
        ).hexdigest()
    else:
        initialization_hash = hashlib.sha256(
            f"initial-{key.dataset}-{key.subject}-{key.condition}".encode()
        ).hexdigest()
    indexed = key.condition == transfer.PRETRAINED_INDEXED_FBMSNET_SPLINE
    effective_model = (
        "fbmsnet"
        if key.condition
        in {
            transfer.SCRATCH_FBMSNET_NATIVE,
            transfer.PRETRAINED_INDEXED_FBMSNET_SPLINE,
        }
        else "cardinal_fbms"
    )
    initialization_names = {
        transfer.PRETRAINED_CARDINAL_FBMS: (
            "immutable_cardinal_fbms_pretraining_checkpoint"
        ),
        transfer.SCRATCH_CARDINAL_FBMS_CANONICAL: (
            "fresh_seeded_canonical21_cardinal_fbms"
        ),
        transfer.SCRATCH_CARDINAL_FBMS_NATIVE: (
            "fresh_seeded_native_geometry_projection"
        ),
        transfer.SCRATCH_FBMSNET_NATIVE: (
            "fresh_seeded_native_indexed_fbmsnet"
        ),
        transfer.PRETRAINED_INDEXED_FBMSNET_SPLINE: (
            "derived_from_pretrained_cardinal_fbms_on_canonical21"
        ),
    }
    channels = (
        tuple(transfer.CANONICAL_21_CHANNELS)
        if indexed
        else ("C0", "C1")
    )
    target_channels = ("C0", "C1")
    target_positions_hash = hashlib.sha256(b"positions").hexdigest()
    condition_positions_hash = (
        audit.CANONICAL_21_POSITIONS_SHA256
        if indexed
        else target_positions_hash
    )
    if effective_model == "cardinal_fbms":
        model_class = "ieee_mi.models.CardinalFBMSNet"
        model_config = {
            "reference_architecture": "Braindecode FBMSNet",
            "n_temporal_views": 36,
            "dilatability": 8,
            "stride_factor": 4,
            "extended_atlas": False,
            "spatial_field": "regularized_cardinal_rbf",
            "spatial_field_norm": "max",
            "spatial_max_norm": 2.0,
            "replaced_component": "grouped_channel_indexed_spatial_conv_only",
            "dynamic_fbms_input_channels": True,
            "dynamic_fbms_input_times": True,
            "paired_fbms_initialization": (
                key.condition == transfer.SCRATCH_CARDINAL_FBMS_NATIVE
            ),
            "paired_fbms_initialization_fit": (
                "exact_minimum_norm"
                if key.condition == transfer.SCRATCH_CARDINAL_FBMS_NATIVE
                else None
            ),
            "paired_fbms_observed_channels": (
                len(channels)
                if key.condition == transfer.SCRATCH_CARDINAL_FBMS_NATIVE
                else None
            ),
            "paired_fbms_basis_rank": (
                len(channels)
                if key.condition == transfer.SCRATCH_CARDINAL_FBMS_NATIVE
                else None
            ),
        }
        parameter_count = 11_441
    else:
        model_class = "braindecode.models.fbmsnet.FBMSNet"
        model_config = {}
        parameter_count = 5_393 + 288 * len(channels)
    train_rows_hash = hashlib.sha256(
        f"train-rows-{key.dataset}-{key.subject}".encode()
    ).hexdigest()
    interpolation = None
    if indexed:
        interpolation = {
            "method": "Perrin spherical spline",
            "implementation": (
                "order-4 Legendre series with constant-potential constraint"
            ),
            "solver": "Moore-Penrose pseudoinverse matching the MNE convention",
            "fit_partition": "training call only",
            "fit_inputs": "native channel geometry; no labels or EEG amplitudes",
            "application": (
                "one fixed matrix applied unchanged to train, validation, and test"
            ),
            "stiffness": 4,
            "n_legendre_terms": 50,
            "regularization": 1e-5,
            "source_channels": ["C0", "C1"],
            "target_channels": list(transfer.CANONICAL_21_CHANNELS),
            "source_positions_sha256": target_positions_hash,
            "target_positions_sha256": audit.CANONICAL_21_POSITIONS_SHA256,
            "matrix_shape": [21, 2],
            "matrix_sha256": hashlib.sha256(b"matrix").hexdigest(),
            "training_rows_sha256": train_rows_hash,
        }

    def scaler(*, fit_rows: str, row_count: int) -> dict[str, object]:
        return {
            "schema": "ieee-mi-channel-scaling-v1",
            "fit_rows": fit_rows,
            "row_count": row_count,
            "channel_names": list(channels),
            "mean": [0.0] * len(channels),
            "std": [1.0] * len(channels),
            "clip_standard_deviations": 12.0,
        }

    if key.condition in {
        transfer.SCRATCH_CARDINAL_FBMS_CANONICAL,
        transfer.SCRATCH_CARDINAL_FBMS_NATIVE,
        transfer.SCRATCH_FBMSNET_NATIVE,
    }:
        reset_start = initialization_hash
    else:
        reset_start = hashlib.sha256(
            f"reset-{key.condition}".encode()
        ).hexdigest()
    first_lr = 8e-4 * (1.0 + math.cos(math.pi / 200.0)) / 2.0
    target_train_config = asdict(TrainConfig())
    target_train_config["device"] = "cpu"
    provenance = {
        "schema": transfer.ARTIFACT_SCHEMA,
        "mode": "development",
        "confirmation_access": False,
        "evidence_scope": transfer.EVIDENCE_SCOPE,
        "analysis_policy": transfer.ANALYSIS_POLICY,
        "record": key.as_dict(),
        "target": {
            "role": "development_target_transfer",
            "montage_profile": "native",
            "cache_file_sha256": cache_digest,
            "raw_array_sha256": hashlib.sha256(b"raw").hexdigest(),
            "channels": list(target_channels),
            "positions_sha256": target_positions_hash,
            "split": {
                "train": {
                    "count": 10,
                    "rows_sha256": train_rows_hash,
                    "labels_sha256": hashlib.sha256(b"train-labels").hexdigest(),
                },
                "validation": {
                    "count": 4,
                    "rows_sha256": hashlib.sha256(b"validation-rows").hexdigest(),
                    "labels_sha256": hashlib.sha256(
                        b"validation-labels"
                    ).hexdigest(),
                },
                "test": {
                    "count": len(labels),
                    "rows_sha256": transfer._core._array_sha256(rows),
                    "labels_sha256": transfer._core._array_sha256(labels),
                }
            },
        },
        "source_checkpoint": {
            "schema": pretrain.CHECKPOINT_SCHEMA,
            "file_sha256": checkpoint_digest,
            "state_sha256": checkpoint_state,
            "size_bytes": 123,
            "model": {"identity": "cardinal_fbms"},
            "pretraining": {
                "state_hashes": {
                    "initial": source_initial,
                    "selection": hashlib.sha256(b"selection").hexdigest(),
                    "checkpoint": checkpoint_state,
                }
            },
        },
        "condition": {
            "requested": key.condition,
            "effective_model": effective_model,
            "initialization": initialization_names[key.condition],
            "initialization_state_sha256": initialization_hash,
            "derived_state_sha256": initialization_hash if indexed else None,
            "model_class": model_class,
            "model_config": model_config,
            "trainable_parameter_count": parameter_count,
            "channels": list(channels),
            "positions_sha256": condition_positions_hash,
            "interpolation": interpolation,
        },
        "protocol": {
            "phase_a": {
                "scaler_fit": "target train rows only",
                "optimization": "target train rows only",
                "selection": "minimum target validation cross-entropy",
                "selected_checkpoint_disposition": "discarded",
                "best_epoch_zero_based": 0,
                "selected_epoch_count": 1,
                "selection_history": [
                    {
                        "epoch": float(epoch),
                        "train_loss": 0.7,
                        "validation_loss": 0.5,
                        "validation_accuracy": 0.5,
                        "learning_rate": 8e-4
                        * (1.0 + math.cos(math.pi * (epoch + 1) / 200.0))
                        / 2.0,
                    }
                    for epoch in range(36)
                ],
                "selection_scaler": scaler(
                    fit_rows="training only", row_count=10
                ),
                "selection_start_sha256": reset_start,
                "selection_state_sha256": hashlib.sha256(
                    b"selection-state"
                ).hexdigest(),
            },
            "phase_b": {
                "initialization": (
                    "fresh model restored from identical condition state"
                ),
                "batch_norm": (
                    "running statistics reset; affine parameters retained"
                ),
                "scaler_fit": (
                    "target train plus validation rows; test excluded"
                ),
                "optimization": "target train plus validation rows",
                "epoch_count": 1,
                "refit_history": [
                    {
                        "epoch": 0.0,
                        "train_loss": 0.68,
                        "learning_rate": first_lr,
                    }
                ],
                "refit_scaler": scaler(
                    fit_rows="training plus validation; test excluded",
                    row_count=14,
                ),
                "refit_start_sha256": reset_start,
                "refit_state_sha256": hashlib.sha256(
                    b"refit-state"
                ).hexdigest(),
                "reset_batch_norm_modules": (
                    ["mix_conv.1", "batch_norm"]
                    if effective_model == "cardinal_fbms"
                    else ["mix_conv.1", "spatial_conv.1"]
                ),
            },
            "test": {
                "aggregate_metrics_emitted": False,
                "inferential_statistics_emitted": False,
            },
            "cross_record_state_carry": False,
            "train_config": target_train_config,
        },
        "predictions": {
            "schema": transfer.PREDICTION_SCHEMA,
            "filename": transfer.PREDICTIONS_FILENAME,
            "file_sha256": _sha256(predictions_path),
            "size_bytes": predictions_path.stat().st_size,
            "row_count": len(labels),
            "test_rows_sha256": transfer._core._array_sha256(rows),
            "test_labels_sha256": transfer._core._array_sha256(labels),
            "predicted_labels_sha256": transfer._core._array_sha256(predicted),
            "probabilities_sha256": transfer._core._array_sha256(probabilities),
        },
        "source_code": {
            "hash_algorithm": "sha256",
            "files": source_files,
        },
    }
    (path / transfer.PROVENANCE_FILENAME).write_text(
        json.dumps(provenance, sort_keys=True), encoding="utf-8"
    )
    return path


def _write_locked_screen(tmp_path: Path) -> tuple[Path, Path]:
    cho_root = tmp_path / "cho"
    physionet_root = tmp_path / "physionet"
    cho_root.mkdir()
    physionet_root.mkdir()
    for dataset, root in (
        (audit.CHO2017, cho_root),
        (audit.PHYSIONET_MI, physionet_root),
    ):
        for subject in audit.LOCKED_SUBJECTS[dataset]:
            for condition in audit.LOCKED_CONDITIONS:
                _write_record(root, _key(dataset, subject, condition))
    return cho_root, physionet_root


def test_locked_grid_and_cli_have_no_subset_or_condition_surface() -> None:
    grid = audit.locked_screen_grid()
    assert audit.EXPECTED_RECORD_COUNT == 455
    assert grid["expected_record_count"] == 455
    assert grid["datasets"][audit.CHO2017]["subjects"] == list(range(16, 53))
    assert grid["datasets"][audit.PHYSIONET_MI]["subjects"] == list(
        range(1, 55)
    )
    assert tuple(grid["datasets"][audit.CHO2017]["conditions"]) == (
        transfer.TRANSFER_CONDITIONS
    )
    destinations = {action.dest for action in audit.build_parser()._actions}
    assert destinations.isdisjoint(
        {"dataset", "subjects", "folds", "seeds", "conditions"}
    )
    assert "conditions=" not in inspect.signature(
        audit.audit_locked_fbms_screen
    ).parameters
    gate_destinations = {
        action.dest for action in gate.build_parser()._actions
    }
    assert gate_destinations == {"help", "cho_root", "physionet_root"}
    assert tuple(inspect.signature(gate.evaluate_fbms_transfer_gate).parameters) == (
        "screen",
    )
    with pytest.raises(TypeError, match="LockedFBMSScreenAudit"):
        gate.evaluate_fbms_transfer_gate({})  # type: ignore[arg-type]


def test_full_455_record_screen_validates_and_equal_weights_subjects(
    tmp_path: Path,
) -> None:
    cho_root, physionet_root = _write_locked_screen(tmp_path)
    screen = audit.audit_locked_fbms_screen(
        cho_root=cho_root, physionet_root=physionet_root
    )
    assert screen.complete
    assert len(screen.records) == 455
    assert all(
        directory.contract == audit.AUDIT_CONTRACT
        for directory in screen.audits
    )
    manifest = audit.audit_manifest(screen)
    assert manifest["complete"] is True
    assert manifest["expected_record_count"] == 455
    assert manifest["validated_record_count"] == 455

    summary = audit.descriptive_screen_summary(screen)
    assert summary["complete_grid"] is True
    assert summary["inferential_statistics"] is False
    means = summary["dataset_condition_mean_balanced_accuracy"]
    for dataset in audit.LOCKED_DATASETS:
        assert means[dataset][transfer.PRETRAINED_CARDINAL_FBMS] == 1.0
        assert means[dataset][transfer.SCRATCH_CARDINAL_FBMS_CANONICAL] == 0.5
        assert means[dataset][transfer.SCRATCH_CARDINAL_FBMS_NATIVE] == 0.5
        assert means[dataset][transfer.SCRATCH_FBMSNET_NATIVE] == 0.0
        assert means[dataset][transfer.PRETRAINED_INDEXED_FBMSNET_SPLINE] == 0.5
        candidate = summary["datasets"][dataset]["conditions"][
            transfer.PRETRAINED_CARDINAL_FBMS
        ]
        assert candidate["n_subjects_observed"] == len(
            audit.LOCKED_SUBJECTS[dataset]
        )

    decision = gate.evaluate_fbms_transfer_gate(screen)
    assert decision["overall_pass"] is True
    assert all(check["pass"] for check in decision["gate_checks"].values())
    bootstrap = decision["inputs"][
        "internally_computed_physionet_paired_subject_bootstrap"
    ]
    assert bootstrap["repetitions"] == 200_000
    assert bootstrap["seed"] == 20_260_719
    assert bootstrap["subject_count"] == 54
    assert bootstrap["lower_bound_balanced_accuracy_delta"] == 0.5
    assert len(
        decision["inputs"]["freshly_revalidated_screen_audit_sha256"]
    ) == 64

    def mutate_semantic(
        record: core.ValidatedTransferArtifact,
        **updates: str,
    ) -> core.ValidatedTransferArtifact:
        values = dict(record.semantic_identity)
        values.update(updates)
        return replace(record, semantic_identity=tuple(sorted(values.items())))

    native_group_mutation = list(screen.records)
    native_index = next(
        index
        for index, record in enumerate(native_group_mutation)
        if record.key.condition == transfer.SCRATCH_CARDINAL_FBMS_NATIVE
    )
    native_group_mutation[native_index] = mutate_semantic(
        native_group_mutation[native_index],
        initialization_state_sha256="0" * 64,
        selection_start_sha256="0" * 64,
    )
    with pytest.raises(core.TransferArtifactError, match="seeded init/reset"):
        audit._validate_family_grid_semantics(native_group_mutation)

    cross_dataset_mutation = [
        mutate_semantic(record, selection_start_sha256="f" * 64)
        if record.key.dataset == audit.PHYSIONET_MI
        and record.key.condition == transfer.PRETRAINED_CARDINAL_FBMS
        else record
        for record in screen.records
    ]
    with pytest.raises(core.TransferArtifactError, match="across datasets"):
        audit._validate_family_grid_semantics(cross_dataset_mutation)

    # A once-valid in-memory audit cannot hide later byte changes: the gate
    # reopens every artifact and reruns family semantic validation.
    stale_path = (
        cho_root
        / core.record_directory_name(
            _key(
                audit.CHO2017,
                16,
                transfer.PRETRAINED_CARDINAL_FBMS,
            )
        )
        / transfer.PROVENANCE_FILENAME
    )
    original = stale_path.read_text(encoding="utf-8")
    stale = json.loads(original)
    stale["condition"]["effective_model"] = "fbmsnet"
    stale_path.write_text(json.dumps(stale), encoding="utf-8")
    try:
        with pytest.raises(core.TransferArtifactError, match="effective model"):
            gate.evaluate_fbms_transfer_gate(screen)
    finally:
        stale_path.write_text(original, encoding="utf-8")


def test_family_schema_checkpoint_and_cross_record_consistency_fail_closed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "small"
    root.mkdir()
    first_key = _key(
        audit.CHO2017, 16, transfer.PRETRAINED_CARDINAL_FBMS
    )
    first_path = _write_record(root, first_key)
    provenance_path = first_path / transfer.PROVENANCE_FILENAME
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["source_checkpoint"]["model"]["identity"] = "cardinal_fbc"
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(core.TransferArtifactError, match="model differs"):
        core.validate_transfer_artifact(
            first_path, expected_key=first_key, contract=audit.AUDIT_CONTRACT
        )

    semantic_root = tmp_path / "semantic"
    semantic_root.mkdir()
    semantic_path = _write_record(semantic_root, first_key)
    semantic_provenance_path = semantic_path / transfer.PROVENANCE_FILENAME
    semantic_provenance = json.loads(
        semantic_provenance_path.read_text(encoding="utf-8")
    )
    semantic_provenance["condition"]["effective_model"] = "fbmsnet"
    semantic_provenance_path.write_text(
        json.dumps(semantic_provenance), encoding="utf-8"
    )
    with pytest.raises(core.TransferArtifactError, match="effective model"):
        core.validate_transfer_artifact(
            semantic_path,
            expected_key=first_key,
            contract=audit.AUDIT_CONTRACT,
        )

    # Build a consistent two-condition grid, then give one record a different
    # checkpoint identity.  Both records are individually valid; the grid is not.
    root2 = tmp_path / "cross-record"
    root2.mkdir()
    conditions = (
        transfer.PRETRAINED_CARDINAL_FBMS,
        transfer.SCRATCH_CARDINAL_FBMS_CANONICAL,
    )
    _write_record(root2, _key(audit.CHO2017, 16, conditions[0]))
    _write_record(
        root2,
        _key(audit.CHO2017, 16, conditions[1]),
        checkpoint_digest=hashlib.sha256(b"different-checkpoint").hexdigest(),
    )
    with pytest.raises(core.TransferArtifactError, match="one source checkpoint"):
        core.audit_transfer_directory(
            root2,
            dataset=audit.CHO2017,
            subjects=(16,),
            folds=(0,),
            seeds=(7,),
            conditions=conditions,
            contract=audit.AUDIT_CONTRACT,
        )


@pytest.mark.parametrize(
    ("case", "condition", "message"),
    (
        ("initialization", transfer.PRETRAINED_CARDINAL_FBMS, "initialization semantics"),
        ("derived", transfer.PRETRAINED_CARDINAL_FBMS, "non-derived"),
        (
            "interpolation",
            transfer.PRETRAINED_INDEXED_FBMSNET_SPLINE,
            "interpolation",
        ),
        ("phase_start", transfer.PRETRAINED_CARDINAL_FBMS, "identical reset state"),
        ("scaler_role", transfer.PRETRAINED_CARDINAL_FBMS, "fitting rows"),
        ("model_class", transfer.PRETRAINED_CARDINAL_FBMS, "model class"),
        ("model_config", transfer.PRETRAINED_CARDINAL_FBMS, "config changed"),
        ("parameter_count", transfer.SCRATCH_FBMSNET_NATIVE, "parameter count"),
        ("channels", transfer.PRETRAINED_CARDINAL_FBMS, "channels differ"),
        ("positions", transfer.PRETRAINED_CARDINAL_FBMS, "positions differ"),
        (
            "canonical_spline_positions",
            transfer.PRETRAINED_INDEXED_FBMSNET_SPLINE,
            "frozen canonical21",
        ),
        (
            "checkpoint_state_record",
            transfer.PRETRAINED_CARDINAL_FBMS,
            "pretraining state record",
        ),
        ("train_config", transfer.PRETRAINED_CARDINAL_FBMS, "frozen field"),
        ("bn_reset", transfer.PRETRAINED_CARDINAL_FBMS, "reset modules"),
        ("history_lr", transfer.PRETRAINED_CARDINAL_FBMS, "schedule differs"),
        ("history_epoch", transfer.PRETRAINED_CARDINAL_FBMS, "epoch sequence"),
        ("history_premature", transfer.PRETRAINED_CARDINAL_FBMS, "prematurely"),
        (
            "history_after_patience",
            transfer.PRETRAINED_CARDINAL_FBMS,
            "after patience",
        ),
        ("reported_best", transfer.PRETRAINED_CARDINAL_FBMS, "reported best"),
        (
            "scratch_init_start",
            transfer.SCRATCH_CARDINAL_FBMS_CANONICAL,
            "reset start differs",
        ),
    ),
)
def test_condition_and_phase_semantic_tampering_is_rejected(
    tmp_path: Path,
    case: str,
    condition: str,
    message: str,
) -> None:
    root = tmp_path / case
    root.mkdir()
    key = _key(audit.CHO2017, 16, condition)
    path = _write_record(root, key)
    provenance_path = path / transfer.PROVENANCE_FILENAME
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if case == "initialization":
        provenance["condition"]["initialization"] = "wrong"
    elif case == "derived":
        provenance["condition"]["derived_state_sha256"] = "0" * 64
    elif case == "interpolation":
        provenance["condition"]["interpolation"] = None
    elif case == "phase_start":
        provenance["protocol"]["phase_b"]["refit_start_sha256"] = "0" * 64
    elif case == "scaler_role":
        provenance["protocol"]["phase_a"]["selection_scaler"][
            "fit_rows"
        ] = "training plus validation"
    elif case == "model_class":
        provenance["condition"]["model_class"] = "synthetic.Wrong"
    elif case == "model_config":
        provenance["condition"]["model_config"]["stride_factor"] = 99
    elif case == "parameter_count":
        provenance["condition"]["trainable_parameter_count"] += 1
    elif case == "channels":
        provenance["condition"]["channels"] = ["wrong"]
        provenance["condition"]["model_config"] = {
            **provenance["condition"]["model_config"],
        }
    elif case == "positions":
        provenance["condition"]["positions_sha256"] = "0" * 64
    elif case == "canonical_spline_positions":
        provenance["condition"]["positions_sha256"] = "0" * 64
        provenance["condition"]["interpolation"][
            "target_positions_sha256"
        ] = "0" * 64
    elif case == "checkpoint_state_record":
        provenance["source_checkpoint"]["pretraining"]["state_hashes"][
            "checkpoint"
        ] = "0" * 64
    elif case == "train_config":
        provenance["protocol"]["train_config"]["learning_rate"] = 0.1
    elif case == "bn_reset":
        provenance["protocol"]["phase_b"]["reset_batch_norm_modules"] = []
    elif case == "history_lr":
        provenance["protocol"]["phase_a"]["selection_history"][0][
            "learning_rate"
        ] = 0.1
    elif case == "history_epoch":
        provenance["protocol"]["phase_a"]["selection_history"][0][
            "epoch"
        ] = 1.0
    elif case == "history_premature":
        provenance["protocol"]["phase_a"]["selection_history"].pop()
    elif case == "history_after_patience":
        provenance["protocol"]["phase_a"]["selection_history"].append(
            {
                "epoch": 36.0,
                "train_loss": 0.7,
                "validation_loss": 0.5,
                "validation_accuracy": 0.5,
                "learning_rate": 8e-4
                * (1.0 + math.cos(math.pi * 37.0 / 200.0))
                / 2.0,
            }
        )
    elif case == "reported_best":
        provenance["protocol"]["phase_a"]["best_epoch_zero_based"] = 1
        provenance["protocol"]["phase_a"]["selected_epoch_count"] = 2
        provenance["protocol"]["phase_b"]["epoch_count"] = 2
        provenance["protocol"]["phase_b"]["refit_history"].append(
            {
                "epoch": 1.0,
                "train_loss": 0.67,
                "learning_rate": 8e-4
                * (1.0 + math.cos(math.pi * 2.0 / 200.0))
                / 2.0,
            }
        )
    elif case == "scratch_init_start":
        provenance["protocol"]["phase_a"]["selection_start_sha256"] = "0" * 64
        provenance["protocol"]["phase_b"]["refit_start_sha256"] = "0" * 64
    else:  # pragma: no cover
        raise AssertionError(case)
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(core.TransferArtifactError, match=message):
        core.validate_transfer_artifact(
            path, expected_key=key, contract=audit.AUDIT_CONTRACT
        )


def test_locked_screen_authorizes_full_grid_before_touching_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0
    original = core.development_transfer_grid

    def guarded_grid(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise PermissionError("synthetic grid preflight refusal")
        return original(*args, **kwargs)

    monkeypatch.setattr(core, "development_transfer_grid", guarded_grid)
    with pytest.raises(PermissionError, match="preflight refusal"):
        audit.audit_locked_fbms_screen(
            cho_root=tmp_path / "must-not-be-read-cho",
            physionet_root=tmp_path / "must-not-be-read-physionet",
        )
    assert calls == 2
