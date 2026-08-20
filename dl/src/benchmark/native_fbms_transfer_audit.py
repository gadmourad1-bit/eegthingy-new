"""Locked five-condition auditor for the CardinalFBMS fold0/seed7 screen.

Only the two caller-supplied development roots are variable.  Cohorts, fold,
seed, conditions, schemas, checkpoint family, and record naming are immutable;
there is no subset-selection surface capable of producing an overall pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from . import native_fbms_transfer as transfer
from . import native_transfer_audit as _core
from .config import CANONICAL_21_CHANNELS, CHANNEL_SCALING
from .native_fbms_pretraining_cli import CHECKPOINT_SCHEMA
from .training import TrainConfig


AUDIT_SCHEMA = "eeg-mi-native-cardinal-fbms-transfer-screen-audit-v1"
SUMMARY_SCHEMA = "eeg-mi-native-cardinal-fbms-transfer-screen-summary-v1"
SCREEN_SCHEMA = "eeg-mi-native-cardinal-fbms-fold0-seed7-grid-v1"
SUMMARY_ANALYSIS_POLICY = "locked_fbms_development_screen_descriptive_no_inference"

CHO2017 = "cho2017"
PHYSIONET_MI = "physionet_mi"
LOCKED_DATASETS: tuple[str, ...] = (CHO2017, PHYSIONET_MI)
LOCKED_SUBJECTS = {
    CHO2017: tuple(range(16, 53)),
    PHYSIONET_MI: tuple(range(1, 55)),
}
LOCKED_FOLDS: tuple[int, ...] = (0,)
LOCKED_SEEDS: tuple[int, ...] = (7,)
LOCKED_CONDITIONS = transfer.TRANSFER_CONDITIONS
EXPECTED_RECORD_COUNT = sum(
    len(LOCKED_SUBJECTS[dataset])
    * len(LOCKED_FOLDS)
    * len(LOCKED_SEEDS)
    * len(LOCKED_CONDITIONS)
    for dataset in LOCKED_DATASETS
)
CANONICAL_21_POSITIONS_SHA256 = (
    "121a8433e69a6468ebb3dd6d717d5fcc17f3a11efe70ea2cf512d1ae70336a94"
)


_CONDITION_SEMANTICS = {
    transfer.PRETRAINED_CARDINAL_FBMS: (
        "cardinal_fbms",
        "immutable_cardinal_fbms_pretraining_checkpoint",
        False,
        False,
    ),
    transfer.SCRATCH_CARDINAL_FBMS_CANONICAL: (
        "cardinal_fbms",
        "fresh_seeded_canonical21_cardinal_fbms",
        False,
        False,
    ),
    transfer.SCRATCH_CARDINAL_FBMS_NATIVE: (
        "cardinal_fbms",
        "fresh_seeded_native_geometry_projection",
        False,
        False,
    ),
    transfer.SCRATCH_FBMSNET_NATIVE: (
        "fbmsnet",
        "fresh_seeded_native_indexed_fbmsnet",
        False,
        False,
    ),
    transfer.PRETRAINED_INDEXED_FBMSNET_SPLINE: (
        "fbmsnet",
        "derived_from_pretrained_cardinal_fbms_on_canonical21",
        True,
        True,
    ),
}


def _scaler_semantics(
    value: object,
    *,
    name: str,
    expected_fit_rows: str,
    expected_row_count: int,
    expected_channels: tuple[str, ...],
) -> None:
    scaler = _core._mapping(value, name=name)
    exact_fields = {
        "schema",
        "fit_rows",
        "row_count",
        "channel_names",
        "mean",
        "std",
        "clip_standard_deviations",
    }
    if set(scaler) != exact_fields:
        raise _core.TransferArtifactError(f"{name} fields differ from fitter")
    if scaler.get("schema") != CHANNEL_SCALING["schema"]:
        raise _core.TransferArtifactError(f"{name} schema differs from fitter")
    if scaler.get("fit_rows") != expected_fit_rows:
        raise _core.TransferArtifactError(f"{name} fitting rows are invalid")
    if _core._integer(scaler.get("row_count"), name=f"{name}.row_count") != (
        expected_row_count
    ):
        raise _core.TransferArtifactError(f"{name} row count differs from split")
    if tuple(scaler.get("channel_names", ())) != expected_channels:
        raise _core.TransferArtifactError(f"{name} channels differ from condition")
    try:
        mean = np.asarray(scaler.get("mean"), dtype=np.float64)
        std = np.asarray(scaler.get("std"), dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise _core.TransferArtifactError(f"{name} statistics are malformed") from error
    if mean.shape != (len(expected_channels),) or std.shape != mean.shape:
        raise _core.TransferArtifactError(f"{name} statistic shape is invalid")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise _core.TransferArtifactError(f"{name} statistics are invalid")
    if scaler.get("clip_standard_deviations") != CHANNEL_SCALING[
        "clip_standard_deviations"
    ]:
        raise _core.TransferArtifactError(f"{name} clipping policy differs")


def _validate_train_config(value: object, *, expected_seed: int) -> TrainConfig:
    config = _core._mapping(value, name="protocol.train_config")
    expected = asdict(TrainConfig())
    if set(config) != set(expected):
        raise _core.TransferArtifactError("target TrainConfig fields differ")
    device = str(config.get("device", ""))
    valid_device = device in {"cpu", "mps", "cuda"}
    if device.startswith("cuda:"):
        suffix = device.removeprefix("cuda:")
        valid_device = suffix.isdigit() and 0 <= int(suffix) <= 31
    if not valid_device:
        raise _core.TransferArtifactError("target TrainConfig device is invalid")
    for name, expected_value in expected.items():
        observed_value = config.get(name)
        if name != "device" and (
            type(observed_value) is not type(expected_value)
            or observed_value != expected_value
        ):
            raise _core.TransferArtifactError(
                f"target TrainConfig changed frozen field {name}"
            )
    if config.get("seed") != expected_seed:
        raise _core.TransferArtifactError("protocol seed differs from record")
    return TrainConfig(**dict(config))


def _history_semantics(
    value: object,
    *,
    name: str,
    config: TrainConfig,
    expected_length: int | None,
    selection: bool,
) -> tuple[list[Mapping[str, object]], int | None]:
    if not isinstance(value, list) or not value:
        raise _core.TransferArtifactError(f"{name} must be a nonempty list")
    if expected_length is not None and len(value) != expected_length:
        raise _core.TransferArtifactError(f"{name} differs from selected duration")
    if len(value) > config.epochs:
        raise _core.TransferArtifactError(f"{name} exceeds frozen epoch budget")
    expected_fields = (
        {"epoch", "train_loss", "validation_loss", "validation_accuracy", "learning_rate"}
        if selection
        else {"epoch", "train_loss", "learning_rate"}
    )
    result: list[Mapping[str, object]] = []
    best_loss = float("inf")
    computed_best_epoch = -1
    stale = 0
    for index, item in enumerate(value):
        row = _core._mapping(item, name=f"{name}[{index}]")
        if set(row) != expected_fields:
            raise _core.TransferArtifactError(f"{name}[{index}] fields differ")
        epoch = row.get("epoch")
        if isinstance(epoch, bool) or not isinstance(epoch, (int, float)) or epoch != float(index):
            raise _core.TransferArtifactError(f"{name} epoch sequence is invalid")
        for field in expected_fields - {"epoch"}:
            observed = row.get(field)
            if isinstance(observed, bool) or not isinstance(observed, (int, float)):
                raise _core.TransferArtifactError(f"{name} {field} is not numeric")
            number = float(observed)
            if not math.isfinite(number):
                raise _core.TransferArtifactError(f"{name} {field} is non-finite")
            if field in {"train_loss", "validation_loss"} and number < 0.0:
                raise _core.TransferArtifactError(f"{name} {field} is negative")
            if field == "validation_accuracy" and not 0.0 <= number <= 1.0:
                raise _core.TransferArtifactError(
                    f"{name} validation_accuracy lies outside [0, 1]"
                )
        expected_lr = config.learning_rate * (
            1.0 + math.cos(math.pi * float(index + 1) / config.epochs)
        ) / 2.0
        if not math.isclose(
            float(row["learning_rate"]), expected_lr, rel_tol=1e-10, abs_tol=1e-15
        ):
            raise _core.TransferArtifactError(
                f"{name} learning-rate schedule differs from fitter"
            )
        if selection:
            validation_loss = float(row["validation_loss"])
            if validation_loss < best_loss - config.min_delta:
                best_loss = validation_loss
                computed_best_epoch = index
                stale = 0
            else:
                stale += 1
                if stale >= config.patience and index != len(value) - 1:
                    raise _core.TransferArtifactError(
                        f"{name} contains rows after patience was exhausted"
                    )
        result.append(row)
    if selection:
        if computed_best_epoch < 0:
            raise _core.TransferArtifactError(f"{name} has no selectable epoch")
        if len(result) < config.epochs and stale != config.patience:
            raise _core.TransferArtifactError(
                f"{name} stopped prematurely before patience was exhausted"
            )
        return result, computed_best_epoch
    return result, None


def _validate_family_artifact_semantics(
    provenance: Mapping[str, object],
    key: _core.TransferRecordKey,
) -> Mapping[str, str]:
    """Bind each condition label to its actual model, state and protocol."""

    checkpoint = _core._mapping(
        provenance.get("source_checkpoint"), name="source_checkpoint"
    )
    checkpoint_state = _core._digest(
        checkpoint.get("state_sha256"), name="source_checkpoint.state_sha256"
    )
    pretraining = _core._mapping(
        checkpoint.get("pretraining"), name="source_checkpoint.pretraining"
    )
    source_state_hashes = _core._mapping(
        pretraining.get("state_hashes"),
        name="source_checkpoint.pretraining.state_hashes",
    )
    source_initial = _core._digest(
        source_state_hashes.get("initial"),
        name="source_checkpoint.pretraining.state_hashes.initial",
    )
    source_checkpoint_recorded = _core._digest(
        source_state_hashes.get("checkpoint"),
        name="source_checkpoint.pretraining.state_hashes.checkpoint",
    )
    if source_checkpoint_recorded != checkpoint_state:
        raise _core.TransferArtifactError(
            "source checkpoint state differs from pretraining state record"
        )

    condition = _core._mapping(provenance.get("condition"), name="condition")
    expected_model, expected_initialization, needs_derived, needs_interpolation = (
        _CONDITION_SEMANTICS[key.condition]
    )
    if condition.get("requested") != key.condition:
        raise _core.TransferArtifactError("condition requested identity changed")
    if condition.get("effective_model") != expected_model:
        raise _core.TransferArtifactError("condition effective model is incompatible")
    if condition.get("initialization") != expected_initialization:
        raise _core.TransferArtifactError("condition initialization semantics changed")
    initialization_hash = _core._digest(
        condition.get("initialization_state_sha256"),
        name="condition.initialization_state_sha256",
    )
    derived_value = condition.get("derived_state_sha256")
    if needs_derived:
        derived_hash = _core._digest(
            derived_value, name="condition.derived_state_sha256"
        )
        if derived_hash != initialization_hash:
            raise _core.TransferArtifactError(
                "derived indexed state differs from fitting initialization"
            )
    elif derived_value is not None:
        raise _core.TransferArtifactError(
            "non-derived condition reports a derived state"
        )
    else:
        derived_hash = ""
    if key.condition == transfer.PRETRAINED_CARDINAL_FBMS and (
        initialization_hash != checkpoint_state
    ):
        raise _core.TransferArtifactError(
            "pretrained CardinalFBMS did not initialize from source checkpoint"
        )
    if key.condition == transfer.SCRATCH_CARDINAL_FBMS_CANONICAL and (
        initialization_hash != source_initial
    ):
        raise _core.TransferArtifactError(
            "canonical-seeded scratch did not use source initial state"
        )
    target_record = _core._mapping(provenance.get("target"), name="target")
    target_channels = tuple(str(value) for value in target_record.get("channels", ()))
    if not target_channels or len(target_channels) != len(set(target_channels)):
        raise _core.TransferArtifactError("target channel identity is invalid")
    target_positions_hash = _core._digest(
        target_record.get("positions_sha256"), name="target.positions_sha256"
    )
    model_class = condition.get("model_class")
    expected_class = (
        "eeg_mi.models.CardinalFBMSNet"
        if expected_model == "cardinal_fbms"
        else "braindecode.models.fbmsnet.FBMSNet"
    )
    if model_class != expected_class:
        raise _core.TransferArtifactError("condition model class is incompatible")
    model_config = condition.get("model_config")
    if not isinstance(model_config, Mapping):
        raise _core.TransferArtifactError("condition model config is missing")
    condition_parameter_count = _core._integer(
        condition.get("trainable_parameter_count"),
        name="condition.trainable_parameter_count",
    )
    channels = tuple(str(value) for value in condition.get("channels", ()))
    if not channels or len(channels) != len(set(channels)):
        raise _core.TransferArtifactError("condition channels are invalid")
    condition_positions_hash = _core._digest(
        condition.get("positions_sha256"), name="condition.positions_sha256"
    )
    if expected_model == "cardinal_fbms":
        expected_config = {
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
        }
        if key.condition == transfer.SCRATCH_CARDINAL_FBMS_NATIVE:
            expected_config.update(
                {
                    "paired_fbms_initialization": True,
                    "paired_fbms_initialization_fit": (
                        "exact_minimum_norm"
                        if len(channels) <= 21
                        else "least_squares_projection"
                    ),
                    "paired_fbms_observed_channels": len(channels),
                    "paired_fbms_basis_rank": min(len(channels), 21),
                }
            )
        else:
            expected_config.update(
                {
                    "paired_fbms_initialization": False,
                    "paired_fbms_initialization_fit": None,
                    "paired_fbms_observed_channels": None,
                    "paired_fbms_basis_rank": None,
                }
            )
        if json.dumps(
            dict(model_config), sort_keys=True, separators=(",", ":")
        ) != json.dumps(expected_config, sort_keys=True, separators=(",", ":")):
            raise _core.TransferArtifactError("condition CardinalFBMS config changed")
        expected_parameter_count = 11_441
    else:
        if dict(model_config):
            raise _core.TransferArtifactError("condition indexed FBMS config changed")
        expected_parameter_count = 5_393 + 288 * len(channels)
    if condition_parameter_count != expected_parameter_count:
        raise _core.TransferArtifactError("condition parameter count is incompatible")

    interpolation = condition.get("interpolation")
    if needs_interpolation:
        interpolation_record = _core._mapping(
            interpolation, name="condition.interpolation"
        )
        expected_interpolation = {
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
        }
        for name, expected in expected_interpolation.items():
            if interpolation_record.get(name) != expected:
                raise _core.TransferArtifactError(
                    f"interpolation changed locked field {name}"
                )
        if tuple(interpolation_record.get("target_channels", ())) != (
            CANONICAL_21_CHANNELS
        ) or channels != CANONICAL_21_CHANNELS:
            raise _core.TransferArtifactError(
                "indexed interpolation target is not canonical21"
            )
        source_channels = tuple(interpolation_record.get("source_channels", ()))
        if source_channels != target_channels:
            raise _core.TransferArtifactError(
                "interpolation source channels differ from native target"
            )
        if tuple(interpolation_record.get("matrix_shape", ())) != (
            21,
            len(source_channels),
        ):
            raise _core.TransferArtifactError("interpolation matrix shape is invalid")
        for name, expected in (
            ("stiffness", 4),
            ("n_legendre_terms", 50),
            ("regularization", 1e-5),
        ):
            if interpolation_record.get(name) != expected:
                raise _core.TransferArtifactError(
                    f"interpolation changed locked field {name}"
                )
        for digest_name in (
            "source_positions_sha256",
            "target_positions_sha256",
            "matrix_sha256",
            "training_rows_sha256",
        ):
            _core._digest(
                interpolation_record.get(digest_name),
                name=f"condition.interpolation.{digest_name}",
            )
        if condition_positions_hash != interpolation_record.get(
            "target_positions_sha256"
        ):
            raise _core.TransferArtifactError(
                "indexed condition positions differ from interpolation target"
            )
        if condition_positions_hash != CANONICAL_21_POSITIONS_SHA256:
            raise _core.TransferArtifactError(
                "indexed interpolation target coordinates are not frozen canonical21"
            )
        if target_positions_hash != interpolation_record.get(
            "source_positions_sha256"
        ):
            raise _core.TransferArtifactError(
                "interpolation source positions differ from native target"
            )
    elif interpolation is not None:
        raise _core.TransferArtifactError(
            "native condition unexpectedly reports interpolation"
        )
    else:
        if channels != target_channels:
            raise _core.TransferArtifactError(
                "native condition channels differ from target cache"
            )
        if condition_positions_hash != target_positions_hash:
            raise _core.TransferArtifactError(
                "native condition positions differ from target cache"
            )

    split = _core._mapping(target_record.get("split"), name="target.split")
    train_split = _core._mapping(split.get("train"), name="target.split.train")
    validation_split = _core._mapping(
        split.get("validation"), name="target.split.validation"
    )
    train_count = _core._integer(
        train_split.get("count"), name="target.split.train.count"
    )
    validation_count = _core._integer(
        validation_split.get("count"), name="target.split.validation.count"
    )
    if train_count <= 0 or validation_count <= 0:
        raise _core.TransferArtifactError("target fitting splits must be nonempty")
    if needs_interpolation and interpolation_record.get(
        "training_rows_sha256"
    ) != train_split.get("rows_sha256"):
        raise _core.TransferArtifactError(
            "interpolation was not fitted from the declared training rows"
        )

    protocol = _core._mapping(provenance.get("protocol"), name="protocol")
    if protocol.get("cross_record_state_carry") is not False:
        raise _core.TransferArtifactError("protocol permits cross-record state")
    train_config = _validate_train_config(
        protocol.get("train_config"), expected_seed=key.seed
    )
    phase_a = _core._mapping(protocol.get("phase_a"), name="protocol.phase_a")
    phase_b = _core._mapping(protocol.get("phase_b"), name="protocol.phase_b")
    exact_phase_a = {
        "scaler_fit": "target train rows only",
        "optimization": "target train rows only",
        "selection": "minimum target validation cross-entropy",
        "selected_checkpoint_disposition": "discarded",
    }
    for name, expected in exact_phase_a.items():
        if phase_a.get(name) != expected:
            raise _core.TransferArtifactError(f"Phase A changed locked field {name}")
    exact_phase_b = {
        "initialization": "fresh model restored from identical condition state",
        "batch_norm": "running statistics reset; affine parameters retained",
        "scaler_fit": "target train plus validation rows; test excluded",
        "optimization": "target train plus validation rows",
    }
    for name, expected in exact_phase_b.items():
        if phase_b.get(name) != expected:
            raise _core.TransferArtifactError(f"Phase B changed locked field {name}")
    selection_start = _core._digest(
        phase_a.get("selection_start_sha256"),
        name="protocol.phase_a.selection_start_sha256",
    )
    refit_start = _core._digest(
        phase_b.get("refit_start_sha256"),
        name="protocol.phase_b.refit_start_sha256",
    )
    if selection_start != refit_start:
        raise _core.TransferArtifactError(
            "Phase A and Phase B did not start from identical reset state"
        )
    if key.condition in {
        transfer.SCRATCH_CARDINAL_FBMS_CANONICAL,
        transfer.SCRATCH_CARDINAL_FBMS_NATIVE,
        transfer.SCRATCH_FBMSNET_NATIVE,
    } and selection_start != initialization_hash:
        raise _core.TransferArtifactError(
            "scratch condition reset start differs from initialization"
        )
    _core._digest(
        phase_a.get("selection_state_sha256"),
        name="protocol.phase_a.selection_state_sha256",
    )
    _core._digest(
        phase_b.get("refit_state_sha256"),
        name="protocol.phase_b.refit_state_sha256",
    )
    best_epoch = _core._integer(
        phase_a.get("best_epoch_zero_based"),
        name="protocol.phase_a.best_epoch_zero_based",
    )
    selected_count = _core._integer(
        phase_a.get("selected_epoch_count"),
        name="protocol.phase_a.selected_epoch_count",
    )
    refit_count = _core._integer(
        phase_b.get("epoch_count"), name="protocol.phase_b.epoch_count"
    )
    if best_epoch < 0 or selected_count != best_epoch + 1 or refit_count != selected_count:
        raise _core.TransferArtifactError("Phase A/B selected duration is inconsistent")
    selection_history, computed_best_epoch = _history_semantics(
        phase_a.get("selection_history"),
        name="protocol.phase_a.selection_history",
        config=train_config,
        expected_length=None,
        selection=True,
    )
    _history_semantics(
        phase_b.get("refit_history"),
        name="protocol.phase_b.refit_history",
        config=train_config,
        expected_length=refit_count,
        selection=False,
    )
    if best_epoch >= len(selection_history):
        raise _core.TransferArtifactError(
            "Phase A selected epoch lies outside selection history"
        )
    if best_epoch != computed_best_epoch:
        raise _core.TransferArtifactError(
            "Phase A reported best epoch differs from validation-loss selection"
        )
    reset_modules = phase_b.get("reset_batch_norm_modules")
    expected_reset_modules = (
        ["mix_conv.1", "batch_norm"]
        if expected_model == "cardinal_fbms"
        else ["mix_conv.1", "spatial_conv.1"]
    )
    if reset_modules != expected_reset_modules:
        raise _core.TransferArtifactError(
            "Phase B BatchNorm reset modules differ from model"
        )
    _scaler_semantics(
        phase_a.get("selection_scaler"),
        name="protocol.phase_a.selection_scaler",
        expected_fit_rows=str(CHANNEL_SCALING["fit_rows"]["selection"]),
        expected_row_count=train_count,
        expected_channels=channels,
    )
    _scaler_semantics(
        phase_b.get("refit_scaler"),
        name="protocol.phase_b.refit_scaler",
        expected_fit_rows=str(CHANNEL_SCALING["fit_rows"]["final_refit"]),
        expected_row_count=train_count + validation_count,
        expected_channels=channels,
    )
    return {
        "checkpoint_state_sha256": checkpoint_state,
        "source_initial_state_sha256": source_initial,
        "initialization_state_sha256": initialization_hash,
        "derived_state_sha256": derived_hash,
        "selection_start_sha256": selection_start,
        "effective_model": expected_model,
        "target_geometry_sha256": hashlib.sha256(
            json.dumps(
                {
                    "channels": list(target_channels),
                    "positions_sha256": target_positions_hash,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "native_channel_count": str(len(target_channels)),
    }


def _validate_family_grid_semantics(
    records: Sequence[_core.ValidatedTransferArtifact],
) -> None:
    identities = [dict(record.semantic_identity) for record in records]
    for name in ("checkpoint_state_sha256", "source_initial_state_sha256"):
        if len({identity[name] for identity in identities}) != 1:
            raise _core.TransferArtifactError(
                f"FBMS grid changed semantic source identity {name}"
            )
    canonical_hashes = {
        identity["initialization_state_sha256"]
        for record, identity in zip(records, identities)
        if record.key.condition == transfer.SCRATCH_CARDINAL_FBMS_CANONICAL
    }
    if len(canonical_hashes) > 1:
        raise _core.TransferArtifactError(
            "canonical-seeded scratch initialization changed across records"
        )
    for condition in (
        transfer.PRETRAINED_CARDINAL_FBMS,
        transfer.SCRATCH_CARDINAL_FBMS_CANONICAL,
        transfer.PRETRAINED_INDEXED_FBMSNET_SPLINE,
    ):
        condition_identities = [
            identity
            for record, identity in zip(records, identities)
            if record.key.condition == condition
        ]
        for name in ("initialization_state_sha256", "selection_start_sha256"):
            if len({identity[name] for identity in condition_identities}) > 1:
                raise _core.TransferArtifactError(
                    f"{condition} changed {name} across datasets or subjects"
                )
    for condition, grouping_field in (
        (
            transfer.SCRATCH_CARDINAL_FBMS_NATIVE,
            "target_geometry_sha256",
        ),
        (transfer.SCRATCH_FBMSNET_NATIVE, "native_channel_count"),
    ):
        groups: dict[str, set[tuple[str, str]]] = {}
        for record, identity in zip(records, identities):
            if record.key.condition != condition:
                continue
            groups.setdefault(identity[grouping_field], set()).add(
                (
                    identity["initialization_state_sha256"],
                    identity["selection_start_sha256"],
                )
            )
        if any(len(states) != 1 for states in groups.values()):
            raise _core.TransferArtifactError(
                f"{condition} changed seeded init/reset within {grouping_field} group"
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
    artifact_semantics_validator=_validate_family_artifact_semantics,
    grid_semantics_validator=_validate_family_grid_semantics,
)


@dataclass(frozen=True)
class LockedFBMSScreenAudit:
    cho2017: _core.TransferDirectoryAudit
    physionet_mi: _core.TransferDirectoryAudit

    @property
    def audits(self) -> tuple[_core.TransferDirectoryAudit, ...]:
        return (self.cho2017, self.physionet_mi)

    @property
    def complete(self) -> bool:
        return all(audit.complete for audit in self.audits)

    @property
    def records(self) -> tuple[_core.ValidatedTransferArtifact, ...]:
        return tuple(record for audit in self.audits for record in audit.records)


def locked_screen_grid() -> dict[str, object]:
    """Return the immutable grid identity used by both audit and decision gate."""

    return {
        "schema": SCREEN_SCHEMA,
        "datasets": {
            dataset: {
                "subjects": list(LOCKED_SUBJECTS[dataset]),
                "folds": list(LOCKED_FOLDS),
                "seeds": list(LOCKED_SEEDS),
                "conditions": list(LOCKED_CONDITIONS),
                "expected_record_count": (
                    len(LOCKED_SUBJECTS[dataset])
                    * len(LOCKED_FOLDS)
                    * len(LOCKED_SEEDS)
                    * len(LOCKED_CONDITIONS)
                ),
            }
            for dataset in LOCKED_DATASETS
        },
        "expected_record_count": EXPECTED_RECORD_COUNT,
    }


def _validate_cross_dataset_provenance(screen: LockedFBMSScreenAudit) -> None:
    records = screen.records
    if not records:
        return
    if len({record.checkpoint_file_sha256 for record in records}) != 1:
        raise _core.TransferArtifactError(
            "locked FBMS datasets do not share one source checkpoint"
        )
    if len({record.source_code_hashes for record in records}) != 1:
        raise _core.TransferArtifactError(
            "locked FBMS datasets do not share one source-code manifest"
        )
    _validate_family_grid_semantics(records)


def audit_locked_fbms_screen(
    *,
    cho_root: str | Path,
    physionet_root: str | Path,
) -> LockedFBMSScreenAudit:
    """Audit exactly 455 fold0/seed7 records; no caller-selected dimensions."""

    # Authorize the complete fixed grid before either caller path is touched.
    for dataset in LOCKED_DATASETS:
        _core.development_transfer_grid(
            dataset=dataset,
            subjects=LOCKED_SUBJECTS[dataset],
            folds=LOCKED_FOLDS,
            seeds=LOCKED_SEEDS,
            conditions=LOCKED_CONDITIONS,
            contract=AUDIT_CONTRACT,
        )
    screen = LockedFBMSScreenAudit(
        cho2017=_core.audit_transfer_directory(
            cho_root,
            dataset=CHO2017,
            subjects=LOCKED_SUBJECTS[CHO2017],
            folds=LOCKED_FOLDS,
            seeds=LOCKED_SEEDS,
            conditions=LOCKED_CONDITIONS,
            contract=AUDIT_CONTRACT,
        ),
        physionet_mi=_core.audit_transfer_directory(
            physionet_root,
            dataset=PHYSIONET_MI,
            subjects=LOCKED_SUBJECTS[PHYSIONET_MI],
            folds=LOCKED_FOLDS,
            seeds=LOCKED_SEEDS,
            conditions=LOCKED_CONDITIONS,
            contract=AUDIT_CONTRACT,
        ),
    )
    _validate_cross_dataset_provenance(screen)
    return screen


def audit_manifest(screen: LockedFBMSScreenAudit) -> dict[str, object]:
    records = screen.records
    missing = tuple(
        key for audit in screen.audits for key in audit.missing_keys
    )
    manifest = {
        "schema": AUDIT_SCHEMA,
        "mode": "development",
        "confirmation_access": False,
        "confirmatory_evidence": False,
        "analysis_policy": SUMMARY_ANALYSIS_POLICY,
        "locked_grid": locked_screen_grid(),
        "complete": screen.complete,
        "expected_record_count": EXPECTED_RECORD_COUNT,
        "validated_record_count": len(records),
        "missing_record_count": len(missing),
        "missing_records": [key.as_dict() for key in missing],
        "datasets": {
            CHO2017: _core.audit_manifest(screen.cho2017),
            PHYSIONET_MI: _core.audit_manifest(screen.physionet_mi),
        },
    }
    if screen.complete and len(records) != EXPECTED_RECORD_COUNT:
        raise RuntimeError("complete FBMS screen has an impossible record count")
    return manifest


def descriptive_screen_summary(
    screen: LockedFBMSScreenAudit,
    *,
    require_complete: bool = True,
) -> dict[str, object]:
    if require_complete and not screen.complete:
        raise RuntimeError("refusing to summarize an incomplete locked FBMS screen")
    dataset_summaries = {
        CHO2017: _core.descriptive_transfer_summary(
            screen.cho2017, require_complete=require_complete
        ),
        PHYSIONET_MI: _core.descriptive_transfer_summary(
            screen.physionet_mi, require_complete=require_complete
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
    return {
        "schema": SUMMARY_SCHEMA,
        "mode": "development",
        "confirmation_access": False,
        "confirmatory_evidence": False,
        "analysis_policy": SUMMARY_ANALYSIS_POLICY,
        "inferential_statistics": False,
        "complete_grid": screen.complete,
        "locked_grid": locked_screen_grid(),
        "expected_record_count": EXPECTED_RECORD_COUNT,
        "validated_record_count": len(screen.records),
        "dataset_condition_mean_balanced_accuracy": condition_means,
        "datasets": dataset_summaries,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cho-root", required=True, type=Path)
    parser.add_argument("--physionet-root", required=True, type=Path)
    parser.add_argument(
        "--allow-incomplete-summary",
        action="store_true",
        help="emit explicitly marked partial descriptive metrics",
    )
    return parser


def run(argv: Sequence[str] | None = None) -> dict[str, object]:
    arguments = build_parser().parse_args(argv)
    screen = audit_locked_fbms_screen(
        cho_root=arguments.cho_root,
        physionet_root=arguments.physionet_root,
    )
    result = audit_manifest(screen)
    if screen.complete or arguments.allow_incomplete_summary:
        result["descriptive_summary"] = descriptive_screen_summary(
            screen,
            require_complete=not arguments.allow_incomplete_summary,
        )
    return result


def main() -> None:
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
