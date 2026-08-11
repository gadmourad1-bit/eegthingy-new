"""Development-only CardinalFBMS native-montage transfer family.

The five conditions are fixed and share one byte-pinned CardinalFBMS source
checkpoint.  This module reuses the legacy transfer implementation's target
authorization, cache validation, interpolation, two-phase fitter, and atomic
publisher, while retaining an independent condition tuple and artifact schema.
It has no cache-building, confirmation, aggregate-scoring, or model-search
surface.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from . import native_transfer as _core
from .baselines import make_model
from .config import CANONICAL_21_CHANNELS
from .data import apply_channel_scaler
from .models import CANONICAL_21_POSITIONS, CardinalFBMSNet, parameter_count
from .native_fbms_pretraining_cli import CHECKPOINT_SCHEMA
from .native_pretraining import fine_tune_pretrained_target, state_dict_sha256
from .native_pretraining import (
    PRIMARY_NATIVE_DEVELOPMENT_SOURCES,
    NativePretrainConfig,
)
from .training import TrainConfig, configure_determinism, predict_probabilities


FloatArray = _core.FloatArray
IntArray = _core.IntArray
ModelFactory = Callable[[], nn.Module]
ImmutableNativeCheckpoint = _core.ImmutableNativeCheckpoint
NativeTargetSubject = _core.NativeTargetSubject
TransferEvaluation = _core.TransferEvaluation

ARTIFACT_SCHEMA = "ieee-mi-native-cardinal-fbms-transfer-development-v1"
PREDICTION_SCHEMA = "ieee-mi-native-cardinal-fbms-transfer-predictions-v1"
PREDICTIONS_FILENAME = _core.PREDICTIONS_FILENAME
PROVENANCE_FILENAME = _core.PROVENANCE_FILENAME
EVIDENCE_SCOPE = "development_only_cardinal_fbms_nonconfirmatory_raw_predictions"
ANALYSIS_POLICY = _core.ANALYSIS_POLICY

PRETRAINED_CARDINAL_FBMS = "pretrained_cardinal_fbms"
SCRATCH_CARDINAL_FBMS_CANONICAL = "scratch_cardinal_fbms_canonical_seeded"
SCRATCH_CARDINAL_FBMS_NATIVE = "scratch_cardinal_fbms_native_projected"
SCRATCH_FBMSNET_NATIVE = "scratch_fbmsnet_native"
PRETRAINED_INDEXED_FBMSNET_SPLINE = (
    "pretrained_indexed_fbmsnet_spherical_spline"
)
TRANSFER_CONDITIONS: tuple[str, ...] = (
    PRETRAINED_CARDINAL_FBMS,
    SCRATCH_CARDINAL_FBMS_CANONICAL,
    SCRATCH_CARDINAL_FBMS_NATIVE,
    SCRATCH_FBMSNET_NATIVE,
    PRETRAINED_INDEXED_FBMSNET_SPLINE,
)

NATIVE_TARGET_DEVELOPMENT_COHORTS = _core.NATIVE_TARGET_DEVELOPMENT_COHORTS
NATIVE_TARGET_FOLDS = _core.NATIVE_TARGET_FOLDS
FROZEN_DEVELOPMENT_SEEDS = _core.FROZEN_DEVELOPMENT_SEEDS
FROZEN_SOURCE_PRETRAIN_CONFIG = NativePretrainConfig()


def validate_target_record(
    dataset: str,
    subject: int,
    fold: int,
    condition: str,
    seed: int,
) -> tuple[str, int, int, str, int]:
    """Authorize one FBMS-family record before any caller path is inspected."""

    key, subject_id, fold_id, seed_id = _core.validate_target_identity(
        dataset, subject, fold, seed
    )
    condition_key = str(condition).strip().lower().replace("-", "_")
    if condition_key not in TRANSFER_CONDITIONS:
        raise ValueError(f"condition must be one of {TRANSFER_CONDITIONS}")
    return key, subject_id, fold_id, condition_key, seed_id


def _make_factory(
    model_name: str,
    *,
    channel_names: tuple[str, ...],
    positions: FloatArray,
    n_times: int,
    paired_geometry: bool = True,
) -> ModelFactory:
    positions_tensor = torch.as_tensor(positions, dtype=torch.float32)

    def factory() -> nn.Module:
        return make_model(
            model_name,
            n_channels=len(channel_names),
            n_outputs=2,
            n_times=n_times,
            sfreq=128.0,
            channel_names=channel_names,
            channel_positions=positions_tensor if paired_geometry else None,
        )

    return factory


def _canonical_cardinal_factory() -> ModelFactory:
    return _make_factory(
        "cardinal_fbms",
        channel_names=CANONICAL_21_CHANNELS,
        positions=np.asarray(CANONICAL_21_POSITIONS, dtype=np.float32),
        n_times=320,
        paired_geometry=True,
    )


def _validate_checkpoint_model_record(
    checkpoint: ImmutableNativeCheckpoint,
) -> None:
    model = checkpoint.model
    if model.get("identity") != "cardinal_fbms":
        raise RuntimeError("checkpoint model identity is not CardinalFBMS")
    construction = model.get("construction")
    if not isinstance(construction, Mapping) or float(
        construction.get("sfreq_hz", -1.0)
    ) != 128.0:
        raise RuntimeError("CardinalFBMS checkpoint has stale construction metadata")
    config = model.get("config")
    required_config = {
        "reference_architecture": "Braindecode FBMSNet",
        "n_temporal_views": 36,
        "dilatability": 8,
        "stride_factor": 4,
        "extended_atlas": False,
        "spatial_field": "regularized_cardinal_rbf",
        "paired_fbms_initialization": True,
        "paired_fbms_initialization_fit": "exact_minimum_norm",
        "paired_fbms_observed_channels": 21,
        "paired_fbms_basis_rank": 21,
        "dynamic_fbms_input_channels": True,
        "dynamic_fbms_input_times": True,
    }
    if not isinstance(config, Mapping) or any(
        config.get(name) != expected for name, expected in required_config.items()
    ):
        raise RuntimeError("CardinalFBMS checkpoint has stale model configuration")
    if model.get("uses_positions") is not True:
        raise RuntimeError("CardinalFBMS checkpoint is not coordinate-conditioned")

    factory = _canonical_cardinal_factory()
    reconstructed = factory().cpu()
    if not isinstance(reconstructed, CardinalFBMSNet):
        raise TypeError("canonical CardinalFBMS factory returned an unexpected class")
    expected_class = f"{type(reconstructed).__module__}.{type(reconstructed).__qualname__}"
    if model.get("class") != expected_class:
        raise RuntimeError("CardinalFBMS checkpoint class identity is stale")
    if int(model.get("trainable_parameter_count", -1)) != parameter_count(reconstructed):
        raise RuntimeError("CardinalFBMS checkpoint parameter count is stale")
    before = state_dict_sha256(checkpoint.state)
    reconstructed.load_state_dict(_core._clone_cpu_state(checkpoint.state), strict=True)
    if state_dict_sha256(reconstructed.state_dict()) != before:
        raise RuntimeError("strict CardinalFBMS checkpoint reconstruction changed state")
    if state_dict_sha256(checkpoint.state) != before:
        raise RuntimeError("checkpoint changed during strict reconstruction")


def _validate_checkpoint_source_provenance(
    checkpoint: ImmutableNativeCheckpoint,
) -> None:
    """Validate hashes captured from the loader's one pinned byte snapshot."""

    source_code = checkpoint.source_code
    if not isinstance(source_code, Mapping) or source_code.get("hash_algorithm") != "sha256":
        raise RuntimeError("CardinalFBMS checkpoint source provenance is missing")
    files = source_code.get("files")
    required = {
        "ieee_mi/baselines.py",
        "ieee_mi/config.py",
        "ieee_mi/data.py",
        "ieee_mi/models.py",
        "ieee_mi/native_pretraining.py",
        "ieee_mi/native_pretraining_cli.py",
        "ieee_mi/native_fbms_pretraining_cli.py",
        "ieee_mi/training.py",
    }
    if not isinstance(files, Mapping) or not required <= set(files):
        raise RuntimeError("CardinalFBMS checkpoint source provenance is incomplete")
    for name, digest in files.items():
        _core._validate_hex_sha256(str(digest), name=f"source hash {name}")
    project = Path(__file__).resolve().parent.parent
    for name in sorted(required):
        current_path = project / name
        if not current_path.is_file():
            raise FileNotFoundError(f"checkpoint source file is unavailable: {name}")
        if _core._sha256_file(current_path) != files[name]:
            raise RuntimeError(
                f"current source file differs from CardinalFBMS checkpoint: {name}"
            )


def _frozen_partition_record() -> dict[str, object]:
    """Recompute the prespecified subject partition without reading EEG."""

    config = FROZEN_SOURCE_PRETRAIN_CONFIG
    by_dataset = {
        dataset: [(dataset, str(subject)) for subject in subjects]
        for dataset, subjects in PRIMARY_NATIVE_DEVELOPMENT_SOURCES.items()
    }
    train: list[tuple[str, str]] = []
    validation: list[tuple[str, str]] = []
    for dataset, keys in sorted(by_dataset.items()):
        ranked = sorted(
            keys,
            key=lambda key: hashlib.sha256(
                f"{config.partition_salt}\0{key[0]}\0{key[1]}".encode("utf-8")
            ).digest(),
        )
        validation_count = max(
            1,
            min(
                len(ranked) - 1,
                math.ceil(config.validation_fraction * len(ranked)),
            ),
        )
        validation.extend(ranked[:validation_count])
        train.extend(ranked[validation_count:])
    train_tuple = tuple(sorted(train))
    validation_tuple = tuple(sorted(validation))
    payload = {
        "salt": config.partition_salt,
        "validation_fraction": config.validation_fraction,
        "train": [list(key) for key in train_tuple],
        "validation": [list(key) for key in validation_tuple],
    }
    return {
        "train": train_tuple,
        "validation": validation_tuple,
        "validation_fraction": config.validation_fraction,
        "salt": config.partition_salt,
        "sha256": hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest(),
    }


def _subject_pairs(value: object, *, name: str) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, (tuple, list)):
        raise RuntimeError(f"{name} must be a subject-pair sequence")
    result: list[tuple[str, str]] = []
    for pair in value:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise RuntimeError(f"{name} contains a malformed subject pair")
        try:
            result.append((str(pair[0]), str(pair[1])))
        except (TypeError, ValueError) as error:
            raise RuntimeError(f"{name} contains a malformed subject pair") from error
    return tuple(result)


def _validate_frozen_pretraining_contract(
    checkpoint: ImmutableNativeCheckpoint,
) -> None:
    """Bind the source to the frozen config, partition, corpus and scratch state."""

    training_config = checkpoint.pretraining.get("training_config")
    if not isinstance(training_config, Mapping):
        raise RuntimeError("CardinalFBMS source training configuration is missing")
    expected_config = asdict(FROZEN_SOURCE_PRETRAIN_CONFIG)
    if set(training_config) != set(expected_config):
        raise RuntimeError("CardinalFBMS source training configuration fields differ")
    observed_device = str(training_config.get("device", ""))
    valid_device = observed_device in {"cpu", "mps", "cuda"}
    if observed_device.startswith("cuda:"):
        suffix = observed_device.removeprefix("cuda:")
        valid_device = suffix.isdigit() and 0 <= int(suffix) <= 31
    if not valid_device:
        raise RuntimeError("CardinalFBMS source execution device is invalid")
    for name, expected in expected_config.items():
        if name != "device" and training_config.get(name) != expected:
            raise RuntimeError(
                f"CardinalFBMS source training configuration changed {name}"
            )
    if training_config.get("seed") != 7:
        raise RuntimeError("CardinalFBMS source seed must equal frozen seed 7")

    partition = checkpoint.pretraining.get("partition")
    if not isinstance(partition, Mapping):
        raise RuntimeError("CardinalFBMS source partition is missing")
    expected_partition = _frozen_partition_record()
    if set(partition) != set(expected_partition):
        raise RuntimeError("CardinalFBMS source partition fields differ")
    if _subject_pairs(partition.get("train"), name="partition.train") != (
        expected_partition["train"]
    ):
        raise RuntimeError("CardinalFBMS source training subjects changed")
    if _subject_pairs(
        partition.get("validation"), name="partition.validation"
    ) != expected_partition["validation"]:
        raise RuntimeError("CardinalFBMS source validation subjects changed")
    for name in ("validation_fraction", "salt", "sha256"):
        if partition.get(name) != expected_partition[name]:
            raise RuntimeError(f"CardinalFBMS source partition changed {name}")

    identities = checkpoint.corpus.get("source_identities")
    if not isinstance(identities, list):
        raise RuntimeError("CardinalFBMS source identities are missing")
    expected_keys = tuple(
        (dataset, int(subject))
        for dataset, subjects in PRIMARY_NATIVE_DEVELOPMENT_SOURCES.items()
        for subject in subjects
    )
    observed_keys = tuple(
        (str(identity.get("dataset", "")), int(identity.get("subject", -1)))
        for identity in identities
        if isinstance(identity, Mapping)
    )
    if len(observed_keys) != len(identities) or observed_keys != expected_keys:
        raise RuntimeError("CardinalFBMS source identity order differs from frozen corpus")
    digest_identities = [
        {
            "dataset": str(identity["dataset"]),
            "subject": int(identity["subject"]),
            "cache_file_sha256": str(identity["cache_file_sha256"]),
            "cache_identity_sha256": str(identity["cache_identity_sha256"]),
            "cache_array_sha256": str(identity["cache_array_sha256"]),
            "authorized_rows_sha256": str(identity["authorized_rows_sha256"]),
        }
        for identity in identities
    ]
    corpus_hash = hashlib.sha256(
        json.dumps(
            digest_identities, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    if checkpoint.corpus.get("sha256") != corpus_hash:
        raise RuntimeError("CardinalFBMS source corpus hash is not reproducible")

    state_hashes = checkpoint.pretraining.get("state_hashes")
    if not isinstance(state_hashes, Mapping):
        raise RuntimeError("CardinalFBMS source state hashes are missing")
    initial_hash = _core._validate_hex_sha256(
        str(state_hashes.get("initial", "")),
        name="CardinalFBMS source initial-state sha256",
    )
    canonical_scratch = _scratch_state(
        _canonical_cardinal_factory(), seed=FROZEN_SOURCE_PRETRAIN_CONFIG.seed
    )
    if state_dict_sha256(canonical_scratch) != initial_hash:
        raise RuntimeError(
            "CardinalFBMS canonical scratch state differs from recorded source initial state"
        )


def load_immutable_native_checkpoint(
    path: str | Path,
    *,
    expected_file_sha256: str,
) -> ImmutableNativeCheckpoint:
    expected_hash = _core._validate_hex_sha256(
        expected_file_sha256,
        name="expected checkpoint file sha256",
    )
    checkpoint = _core.load_immutable_native_checkpoint(
        path,
        expected_file_sha256=expected_hash,
        expected_checkpoint_schema=CHECKPOINT_SCHEMA,
        expected_model_identity="cardinal_fbms",
    )
    _validate_checkpoint_source_provenance(checkpoint)
    _validate_checkpoint_model_record(checkpoint)
    _validate_frozen_pretraining_contract(checkpoint)
    return checkpoint


def load_native_target_subject(
    cache_root: str | Path,
    *,
    dataset: str,
    subject: int,
    fold: int,
    condition: str = PRETRAINED_CARDINAL_FBMS,
    seed: int = 7,
) -> NativeTargetSubject:
    return _core._load_native_target_subject_for_record(
        cache_root,
        dataset=dataset,
        subject=subject,
        fold=fold,
        condition=condition,
        seed=seed,
        validator=validate_target_record,
    )


def cardinal_fbms_checkpoint_to_indexed_fbmsnet(
    pretrained_state: Mapping[str, Tensor],
) -> dict[str, Tensor]:
    """Evaluate a CardinalFBMS checkpoint on canonical21 without mutation."""

    source_hash_before = state_dict_sha256(pretrained_state)
    positions = np.asarray(CANONICAL_21_POSITIONS, dtype=np.float32)
    cardinal = _canonical_cardinal_factory()().cpu()
    cardinal.load_state_dict(_core._clone_cpu_state(pretrained_state), strict=True)
    if not isinstance(cardinal, CardinalFBMSNet):
        raise TypeError("CardinalFBMS checkpoint factory returned an unexpected class")
    indexed = _make_factory(
        "fbmsnet",
        channel_names=CANONICAL_21_CHANNELS,
        positions=positions,
        n_times=320,
    )().cpu()
    spatial = getattr(indexed, "spatial_conv", None)
    if not isinstance(spatial, nn.Sequential) or len(spatial) < 3:
        raise TypeError("indexed FBMSNet has an unexpected spatial block")
    indexed_spatial = spatial[0]

    pairs = (
        (indexed.spectral_filtering, cardinal.spectral_filtering, "spectral_filtering"),
        (indexed.mix_conv, cardinal.mix_conv, "mixed_temporal"),
        (spatial[1], cardinal.batch_norm, "spatial_batch_norm"),
        (spatial[2], cardinal.activation, "spatial_activation"),
        (indexed.padding_layer, cardinal.padding_layer, "padding_layer"),
        (indexed.temporal_layer, cardinal.temporal_layer, "temporal_layer"),
        (indexed.flatten_layer, cardinal.flatten_layer, "flatten_layer"),
        (indexed.final_layer, cardinal.final_layer, "final_layer"),
    )
    for target, source, name in pairs:
        try:
            target.load_state_dict(source.state_dict(), strict=True)
        except RuntimeError as error:
            raise RuntimeError(
                f"cannot transfer CardinalFBMS {name} to indexed FBMSNet"
            ) from error

    canonical_tensor = torch.as_tensor(positions, dtype=torch.float32)
    with torch.no_grad():
        basis = cardinal.spatial_field.basis(canonical_tensor)
        fields = torch.einsum(
            "fsa,ca->fsc",
            cardinal.spatial_field.coefficients,
            basis,
        )
        fields = cardinal.spatial_field._apply_field_norm(fields)
        if fields.numel() != indexed_spatial.weight.numel():
            raise RuntimeError("continuous and indexed FBMS spatial budgets differ")
        parametrizations = getattr(indexed_spatial, "parametrizations", None)
        weight_parametrization = (
            getattr(parametrizations, "weight", None)
            if parametrizations is not None
            else None
        )
        original_weight = getattr(weight_parametrization, "original", None)
        if not isinstance(original_weight, Tensor):
            raise TypeError("indexed FBMSNet spatial max-norm state is missing")
        original_weight.copy_(fields.reshape_as(original_weight))
        if indexed_spatial.bias is None:
            raise RuntimeError("indexed FBMSNet unexpectedly has no spatial bias")
        indexed_spatial.bias.copy_(
            cardinal.spatial_bias.reshape_as(indexed_spatial.bias)
        )
    result = _core._clone_cpu_state(indexed.state_dict())
    if state_dict_sha256(pretrained_state) != source_hash_before:
        raise RuntimeError("source checkpoint changed during indexed conversion")
    return result


def _scratch_state(factory: ModelFactory, *, seed: int) -> dict[str, Tensor]:
    configure_determinism(seed)
    return _core._clone_cpu_state(factory().state_dict())


def _interpolation_record(
    interpolator: _core.SphericalSplineInterpolator,
    target: NativeTargetSubject,
) -> Mapping[str, object]:
    return {
        "method": "Perrin spherical spline",
        "implementation": "order-4 Legendre series with constant-potential constraint",
        "solver": "Moore-Penrose pseudoinverse matching the MNE convention",
        "fit_partition": "training call only",
        "fit_inputs": "native channel geometry; no labels or EEG amplitudes",
        "application": "one fixed matrix applied unchanged to train, validation, and test",
        "stiffness": interpolator.stiffness,
        "n_legendre_terms": interpolator.n_legendre_terms,
        "regularization": interpolator.regularization,
        "source_channels": list(interpolator.source_channel_names),
        "target_channels": list(interpolator.target_channel_names),
        # Bind the record to the exact coordinate arrays supplied at the
        # transfer boundary.  The interpolator stores normalized float64
        # copies internally, whereas the target/cache and canonical model use
        # their raw float32 arrays; hashing those internal copies here would
        # make two representations of the same montage appear unrelated.
        "source_positions_sha256": _core._array_sha256(target.positions),
        "target_positions_sha256": _core._array_sha256(
            np.asarray(CANONICAL_21_POSITIONS, dtype=np.float32)
        ),
        "matrix_shape": list(interpolator.matrix.shape),
        "matrix_sha256": interpolator.matrix_sha256,
        "training_rows_sha256": _core._array_sha256(target.train_rows),
    }


def evaluate_transfer_condition(
    target: NativeTargetSubject,
    checkpoint: ImmutableNativeCheckpoint,
    *,
    condition: str,
    train_config: TrainConfig,
) -> TransferEvaluation:
    """Fit and predict one condition without cross-record or test-to-fit state."""

    _, _, _, condition_key, seed = validate_target_record(
        target.dataset,
        target.subject,
        target.fold,
        condition,
        train_config.seed,
    )
    if checkpoint.model.get("identity") != "cardinal_fbms":
        raise RuntimeError("FBMS-family condition requires a CardinalFBMS checkpoint")
    if set(target.y.tolist()) != {0, 1}:
        raise ValueError("target labels must be binary")

    x_train_raw = target.x[target.train_rows].copy()
    y_train = target.y[target.train_rows].copy()
    x_validation_raw = target.x[target.validation_rows].copy()
    y_validation = target.y[target.validation_rows].copy()
    x_test_raw = target.x[target.test_rows].copy()
    test_labels = target.y[target.test_rows].copy()
    channels = target.channel_names
    positions = target.positions.copy()
    interpolation_record: Mapping[str, object] | None = None
    derived_state_hash: str | None = None

    if condition_key == PRETRAINED_INDEXED_FBMSNET_SPLINE:
        interpolator = _core.fit_spherical_spline_interpolator(
            x_train_raw,
            positions,
            channels,
        )
        x_train_raw = interpolator.transform(x_train_raw)
        x_validation_raw = interpolator.transform(x_validation_raw)
        x_test_raw = interpolator.transform(x_test_raw)
        channels = CANONICAL_21_CHANNELS
        positions = np.asarray(CANONICAL_21_POSITIONS, dtype=np.float32)
        factory = _make_factory(
            "fbmsnet",
            channel_names=channels,
            positions=positions,
            n_times=x_train_raw.shape[2],
        )
        start_state = cardinal_fbms_checkpoint_to_indexed_fbmsnet(checkpoint.state)
        derived_state_hash = state_dict_sha256(start_state)
        effective_model = "fbmsnet"
        initialization = "derived_from_pretrained_cardinal_fbms_on_canonical21"
        interpolation_record = _interpolation_record(interpolator, target)
    elif condition_key == PRETRAINED_CARDINAL_FBMS:
        factory = _make_factory(
            "cardinal_fbms",
            channel_names=channels,
            positions=positions,
            n_times=x_train_raw.shape[2],
            paired_geometry=False,
        )
        start_state = _core._clone_cpu_state(checkpoint.state)
        effective_model = "cardinal_fbms"
        initialization = "immutable_cardinal_fbms_pretraining_checkpoint"
    elif condition_key == SCRATCH_CARDINAL_FBMS_CANONICAL:
        factory = _make_factory(
            "cardinal_fbms",
            channel_names=channels,
            positions=positions,
            n_times=x_train_raw.shape[2],
            paired_geometry=False,
        )
        start_state = _scratch_state(_canonical_cardinal_factory(), seed=seed)
        effective_model = "cardinal_fbms"
        initialization = "fresh_seeded_canonical21_cardinal_fbms"
    elif condition_key == SCRATCH_CARDINAL_FBMS_NATIVE:
        factory = _make_factory(
            "cardinal_fbms",
            channel_names=channels,
            positions=positions,
            n_times=x_train_raw.shape[2],
            paired_geometry=True,
        )
        start_state = _scratch_state(factory, seed=seed)
        effective_model = "cardinal_fbms"
        initialization = "fresh_seeded_native_geometry_projection"
    elif condition_key == SCRATCH_FBMSNET_NATIVE:
        factory = _make_factory(
            "fbmsnet",
            channel_names=channels,
            positions=positions,
            n_times=x_train_raw.shape[2],
        )
        start_state = _scratch_state(factory, seed=seed)
        effective_model = "fbmsnet"
        initialization = "fresh_seeded_native_indexed_fbmsnet"
    else:  # pragma: no cover - guarded by the record validator
        raise RuntimeError("unreachable FBMS-family condition")

    initialization_hash = state_dict_sha256(start_state)
    immutable_before = state_dict_sha256(checkpoint.state)
    result = fine_tune_pretrained_target(
        factory,
        start_state,
        x_train_raw,
        y_train,
        x_validation_raw,
        y_validation,
        positions,
        channel_names=channels,
        config=train_config,
    )
    if state_dict_sha256(checkpoint.state) != immutable_before:
        raise RuntimeError("immutable CardinalFBMS checkpoint changed during fitting")
    if result.pretrained_checkpoint_sha256 != initialization_hash:
        raise RuntimeError("target fitter did not start from the declared condition state")
    refit_mean, refit_std = _core._scaler_arrays(
        result.refit_scaler, len(channels)
    )
    x_test = apply_channel_scaler(x_test_raw, refit_mean, refit_std)
    probabilities = np.asarray(
        predict_probabilities(
            result.model,
            x_test,
            positions,
            device=train_config.device,
        ),
        dtype=np.float64,
    )
    if probabilities.shape != (len(target.test_rows), 2):
        raise RuntimeError("target prediction shape is invalid")
    if not np.isfinite(probabilities).all() or not np.allclose(
        probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-6
    ):
        raise RuntimeError("target probabilities are invalid")
    model = result.model
    return TransferEvaluation(
        probabilities=probabilities,
        predicted_labels=probabilities.argmax(axis=1).astype(np.int64),
        test_labels=test_labels,
        fine_tune=result,
        requested_condition=condition_key,
        effective_model=effective_model,
        initialization=initialization,
        initialization_state_sha256=initialization_hash,
        derived_state_sha256=derived_state_hash,
        model_class=f"{type(model).__module__}.{type(model).__qualname__}",
        model_config=copy.deepcopy(getattr(model, "config", {})),
        trainable_parameter_count=parameter_count(model),
        channels=channels,
        positions=positions,
        interpolation=interpolation_record,
    )


def _source_paths() -> dict[str, Path]:
    paths = dict(_core._source_paths())
    package = Path(__file__).resolve().parent
    additions = {
        "ieee_mi/native_fbms_pretraining_cli.py": package
        / "native_fbms_pretraining_cli.py",
        "ieee_mi/native_fbms_transfer.py": Path(__file__).resolve(),
    }
    overlap = set(paths) & set(additions)
    if overlap:
        raise RuntimeError(f"duplicate FBMS transfer source provenance keys {overlap}")
    paths.update(additions)
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required FBMS transfer sources are missing: {missing}")
    return paths


def _source_file_hashes() -> dict[str, str]:
    return {
        name: _core._sha256_file(path)
        for name, path in sorted(_source_paths().items())
    }


def run_record(
    *,
    dataset: str,
    subject: int,
    fold: int,
    condition: str,
    seed: int,
    cache_root: str | Path,
    checkpoint_path: str | Path,
    expected_checkpoint_file_sha256: str,
    output: str | Path,
    device: str = "cuda",
) -> Path:
    """Run and atomically publish one authorized FBMS-family record."""

    key, subject_id, fold_id, condition_key, seed_id = validate_target_record(
        dataset, subject, fold, condition, seed
    )
    expected_hash = _core._validate_hex_sha256(
        expected_checkpoint_file_sha256,
        name="expected checkpoint file sha256",
    )
    output_path = Path(output).resolve()
    with _core._OutputClaim(output_path):
        source_hashes_before = _source_file_hashes()
        # Family/schema compatibility is validated before target-cache I/O.
        checkpoint = load_immutable_native_checkpoint(
            checkpoint_path,
            expected_file_sha256=expected_hash,
        )
        target = load_native_target_subject(
            cache_root,
            dataset=key,
            subject=subject_id,
            fold=fold_id,
            condition=condition_key,
            seed=seed_id,
        )
        config = replace(TrainConfig(), seed=seed_id, device=str(device))
        started_utc = _core._utc_now()
        started = time.perf_counter()
        evaluation = evaluate_transfer_condition(
            target,
            checkpoint,
            condition=condition_key,
            train_config=config,
        )
        elapsed = time.perf_counter() - started
        if _source_file_hashes() != source_hashes_before:
            raise RuntimeError("FBMS transfer source code changed during the record")
        _core._publish_record(
            output_path,
            target=target,
            checkpoint=checkpoint,
            evaluation=evaluation,
            train_config=config,
            source_hashes=source_hashes_before,
            started_utc=started_utc,
            elapsed_seconds=elapsed,
            artifact_schema=ARTIFACT_SCHEMA,
            prediction_schema=PREDICTION_SCHEMA,
            predictions_filename=PREDICTIONS_FILENAME,
            provenance_filename=PROVENANCE_FILENAME,
            evidence_scope=EVIDENCE_SCOPE,
            analysis_policy=ANALYSIS_POLICY,
        )
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate one locked CardinalFBMS development transfer record."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=tuple(NATIVE_TARGET_DEVELOPMENT_COHORTS),
    )
    parser.add_argument("--subject", required=True, type=int)
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--condition", required=True, choices=TRANSFER_CONDITIONS)
    parser.add_argument(
        "--seed", required=True, type=int, choices=FROZEN_DEVELOPMENT_SEEDS
    )
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    return parser


def run(argv: Sequence[str] | None = None) -> Path:
    arguments = build_parser().parse_args(argv)
    return run_record(
        dataset=arguments.dataset,
        subject=arguments.subject,
        fold=arguments.fold,
        condition=arguments.condition,
        seed=arguments.seed,
        cache_root=arguments.cache_root,
        checkpoint_path=arguments.checkpoint,
        expected_checkpoint_file_sha256=arguments.checkpoint_sha256,
        output=arguments.output,
        device=arguments.device,
    )


def main() -> None:
    print(run())


if __name__ == "__main__":
    main()
