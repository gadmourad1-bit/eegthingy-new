"""Development-only native-montage target-transfer record generator.

This module evaluates exactly one ``(dataset, subject, fold, seed, condition)``
record at a time.  It intentionally has no cache-building path, confirmation
mode, aggregate scorer, significance test, or model-search surface.  The only
authorized target cohorts are Cho2017 S16--S52 and PhysionetMI S1--S54.

Every condition follows the same source-only two-phase protocol implemented by
``fine_tune_pretrained_target``:

* Phase A fits channel scaling on target-train rows, uses validation rows only
  to select an epoch count, and then discards the selected weights.
* Phase B constructs a fresh model from the identical immutable initialization
  or pretrained checkpoint, resets BatchNorm running state, fits a new scaler
  on train plus validation, refits for the selected duration, and predicts the
  held-out test rows once.

The indexed FBCNet control uses a label-free spherical-spline map fitted from
the training call's native channel geometry.  The map is subsequently applied
unchanged to validation and test amplitudes.  Its canonical indexed checkpoint
is derived by evaluating the immutable pretrained CardinalFBC spatial field at
the same 21 canonical coordinates; it is therefore a montage-normalization
control for the continuous field rather than an independently pretrained net.

Artifacts contain row-level predictions and complete development provenance,
but deliberately contain no test-set aggregate metrics or inferential
statistics.  A directory-level atomic rename publishes both files together,
and an exclusive claim file prevents cooperating writers from racing.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import os
import platform
import shutil
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from types import MappingProxyType

import numpy as np
import torch
from numpy.polynomial.legendre import legval
from numpy.typing import NDArray
from scipy.linalg import pinv
from torch import Tensor, nn

from .baselines import make_model
from .config import (
    CANONICAL_21_CHANNELS,
    CHANNEL_SCALING,
    DATASETS,
    PREPROCESSING,
    preprocessing_for_dataset,
)
from .data import apply_channel_scaler, load_subject_cache, split_indices
from .models import CANONICAL_21_POSITIONS, CardinalFBCNet, parameter_count
from .native_pretraining import (
    PRIMARY_NATIVE_DEVELOPMENT_SOURCES,
    TargetFineTuneResult,
    fine_tune_pretrained_target,
    state_dict_sha256,
)
from .native_pretraining_cli import CHECKPOINT_SCHEMA
from .training import TrainConfig, configure_determinism, predict_probabilities


FloatArray = NDArray[np.float32]
Float64Array = NDArray[np.float64]
IntArray = NDArray[np.int64]
ModelFactory = Callable[[], nn.Module]

ARTIFACT_SCHEMA = "eeg-mi-native-transfer-development-v1"
PREDICTION_SCHEMA = "eeg-mi-native-transfer-predictions-v1"
PREDICTIONS_FILENAME = "predictions.npz"
PROVENANCE_FILENAME = "provenance.json"
MONTAGE_PROFILE = "native"
EVIDENCE_SCOPE = "development_only_nonconfirmatory_raw_predictions"
ANALYSIS_POLICY = "no_test_aggregate_metrics_or_inferential_statistics"

PRETRAINED_CARDINAL_FBC = "pretrained_cardinal_fbc"
SCRATCH_CARDINAL_FBC = "scratch_cardinal_fbc"
SCRATCH_FBMSNET = "scratch_fbmsnet"
PRETRAINED_INDEXED_FBCNET_SPLINE = (
    "pretrained_indexed_fbcnet_spherical_spline"
)
TRANSFER_CONDITIONS: tuple[str, ...] = (
    PRETRAINED_CARDINAL_FBC,
    SCRATCH_CARDINAL_FBC,
    SCRATCH_FBMSNET,
    PRETRAINED_INDEXED_FBCNET_SPLINE,
)

# Cho S1--S15 belong to the locked source pretraining corpus, so target
# transfer starts at S16.  Physionet S55--S109 remain sealed confirmation.
NATIVE_TARGET_DEVELOPMENT_COHORTS = MappingProxyType(
    {
        "cho2017": tuple(range(16, 53)),
        "physionet_mi": tuple(range(1, 55)),
    }
)
NATIVE_TARGET_FOLDS = MappingProxyType(
    {
        "cho2017": tuple(range(5)),
        "physionet_mi": tuple(range(3)),
    }
)
FROZEN_DEVELOPMENT_SEEDS: tuple[int, ...] = (7, 17, 27, 37, 47)


@dataclass(frozen=True)
class ImmutableNativeCheckpoint:
    """One byte-pinned, validated in-memory pretraining checkpoint snapshot."""

    path: str
    file_sha256: str
    size_bytes: int
    state_sha256: str
    state: Mapping[str, Tensor]
    corpus: Mapping[str, object]
    model: Mapping[str, object]
    pretraining: Mapping[str, object]
    source_code: Mapping[str, object] | None = None
    checkpoint_schema: str = CHECKPOINT_SCHEMA


@dataclass(frozen=True)
class NativeTargetSubject:
    """One validated native target cache and its prespecified row split."""

    dataset: str
    subject: int
    fold: int
    x: FloatArray
    y: IntArray
    positions: FloatArray
    channel_names: tuple[str, ...]
    sessions: NDArray[np.str_]
    runs: NDArray[np.str_]
    train_rows: IntArray
    validation_rows: IntArray
    test_rows: IntArray
    cache_path: str
    cache_file_sha256: str
    cache_identity: Mapping[str, object]


@dataclass(frozen=True)
class SphericalSplineInterpolator:
    """Fixed native-to-canonical spherical-spline voltage operator."""

    matrix: Float64Array
    source_channel_names: tuple[str, ...]
    source_positions: Float64Array
    target_channel_names: tuple[str, ...]
    target_positions: Float64Array
    stiffness: int
    n_legendre_terms: int
    regularization: float
    matrix_sha256: str

    def transform(self, values: FloatArray) -> FloatArray:
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 3:
            raise ValueError("spherical-spline input must have shape (trials, channels, time)")
        if values.shape[1] != self.matrix.shape[1]:
            raise ValueError("spherical-spline input channels differ from fitted geometry")
        if not np.isfinite(values).all():
            raise ValueError("spherical-spline input contains non-finite values")
        transformed = np.einsum(
            "oc,nct->not",
            self.matrix,
            values.astype(np.float64, copy=False),
            optimize=True,
        )
        if not np.isfinite(transformed).all():
            raise RuntimeError("spherical-spline interpolation produced non-finite values")
        return transformed.astype(np.float32)


@dataclass
class TransferEvaluation:
    """Raw prediction result for one independently initialized condition."""

    probabilities: Float64Array
    predicted_labels: IntArray
    test_labels: IntArray
    fine_tune: TargetFineTuneResult
    requested_condition: str
    effective_model: str
    initialization: str
    initialization_state_sha256: str
    derived_state_sha256: str | None
    model_class: str
    model_config: object
    trainable_parameter_count: int
    channels: tuple[str, ...]
    positions: FloatArray
    interpolation: Mapping[str, object] | None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _sha256_bytes(values: bytes) -> str:
    return hashlib.sha256(values).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _path_exists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _jsonable(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (torch.device, torch.dtype)):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"value is not JSON serializable: {type(value)!r}")


def _clone_cpu_state(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in state.items()
    }


def _validate_registry_contract() -> None:
    expected_cohorts = {
        "cho2017": tuple(range(16, 53)),
        "physionet_mi": tuple(range(1, 55)),
    }
    expected_folds = {
        "cho2017": tuple(range(5)),
        "physionet_mi": tuple(range(3)),
    }
    if dict(NATIVE_TARGET_DEVELOPMENT_COHORTS) != expected_cohorts:
        raise RuntimeError("native target cohort differs from the locked pilot")
    if dict(NATIVE_TARGET_FOLDS) != expected_folds:
        raise RuntimeError("native target folds differ from the locked pilot")
    if PREPROCESSING.get("schema") != "eeg-mi-cache-v2":
        raise RuntimeError("native target transfer requires the locked v2 cache schema")
    for dataset, subjects in NATIVE_TARGET_DEVELOPMENT_COHORTS.items():
        if dataset not in DATASETS:
            raise RuntimeError(f"unregistered target dataset {dataset}")
        spec = DATASETS[dataset]
        selected = set(subjects)
        if not selected or not selected <= set(spec.development_subjects):
            raise RuntimeError(f"target cohort is not development-only for {dataset}")
        if selected & set(spec.confirmation_subjects):
            raise PermissionError(f"target cohort reaches sealed data for {dataset}")
    source_cho = set(PRIMARY_NATIVE_DEVELOPMENT_SOURCES.get("cho2017", ()))
    if source_cho & set(NATIVE_TARGET_DEVELOPMENT_COHORTS["cho2017"]):
        raise RuntimeError("Cho source-pretraining and target-transfer subjects overlap")


def validate_target_record(
    dataset: str,
    subject: int,
    fold: int,
    condition: str,
    seed: int,
) -> tuple[str, int, int, str, int]:
    """Authorize one target record without touching a path or external data."""

    key, subject_id, fold_id, seed_id = validate_target_identity(
        dataset, subject, fold, seed
    )
    condition_key = str(condition).strip().lower().replace("-", "_")
    if condition_key not in TRANSFER_CONDITIONS:
        raise ValueError(f"condition must be one of {TRANSFER_CONDITIONS}")
    return key, subject_id, fold_id, condition_key, seed_id


def validate_target_identity(
    dataset: str,
    subject: int,
    fold: int,
    seed: int,
) -> tuple[str, int, int, int]:
    """Authorize the shared development target identity without any path I/O."""

    _validate_registry_contract()
    key = str(dataset).strip().lower().replace("-", "_")
    try:
        subject_id = int(subject)
        fold_id = int(fold)
        seed_id = int(seed)
    except (TypeError, ValueError) as error:
        raise TypeError("subject, fold, and seed must be integers") from error
    if key not in NATIVE_TARGET_DEVELOPMENT_COHORTS:
        raise PermissionError(
            "native target transfer is restricted to Cho2017 and PhysionetMI development"
        )
    if subject_id not in NATIVE_TARGET_DEVELOPMENT_COHORTS[key]:
        raise PermissionError(
            f"{key} S{subject_id} is outside the locked native target development cohort"
        )
    spec = DATASETS[key]
    if subject_id in spec.confirmation_subjects:
        raise PermissionError(f"{key} S{subject_id} is sealed confirmation data")
    if fold_id not in NATIVE_TARGET_FOLDS[key]:
        raise ValueError(f"fold {fold_id} is invalid for the locked {key} protocol")
    if seed_id not in FROZEN_DEVELOPMENT_SEEDS:
        raise ValueError(
            f"seed must be one of the frozen development seeds {FROZEN_DEVELOPMENT_SEEDS}"
        )
    return key, subject_id, fold_id, seed_id


class _OutputClaim:
    """Reject overwrite/races before checkpoint or target-cache access."""

    def __init__(self, output: Path) -> None:
        self.output = output
        self.lock = output.parent / f".{output.name}.native-transfer.lock"
        self._descriptor: int | None = None

    def __enter__(self) -> "_OutputClaim":
        if _path_exists(self.output):
            raise FileExistsError(f"refusing to overwrite output {self.output}")
        if not self.output.parent.is_dir():
            raise FileNotFoundError(
                f"output parent must already exist: {self.output.parent}"
            )
        try:
            self._descriptor = os.open(
                self.lock,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except FileExistsError as error:
            raise FileExistsError(
                f"another native-transfer publisher has claimed {self.output}"
            ) from error
        try:
            os.write(
                self._descriptor,
                f"pid={os.getpid()} output={self.output}\n".encode("utf-8"),
            )
            os.fsync(self._descriptor)
            if _path_exists(self.output):
                raise FileExistsError(
                    f"output appeared while acquiring publication claim: {self.output}"
                )
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise
        return self

    def __exit__(self, *unused: object) -> None:
        del unused
        if self._descriptor is not None:
            os.close(self._descriptor)
            self._descriptor = None
        try:
            self.lock.unlink()
        except FileNotFoundError:
            pass


def _validate_hex_sha256(value: str, *, name: str) -> str:
    result = str(value).strip().lower()
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256 digest")
    return result


def _validate_pretraining_corpus_identity(corpus: Mapping[str, object]) -> None:
    if corpus.get("cache_schema") != PREPROCESSING["schema"]:
        raise RuntimeError("pretraining checkpoint references a stale cache schema")
    if corpus.get("montage_profile") != MONTAGE_PROFILE:
        raise PermissionError("pretraining checkpoint is not native-montage")
    identities = corpus.get("source_identities")
    if not isinstance(identities, list):
        raise RuntimeError("pretraining checkpoint has no source identities")
    observed: set[tuple[str, int]] = set()
    for identity in identities:
        if not isinstance(identity, Mapping):
            raise RuntimeError("pretraining source identity is malformed")
        dataset = str(identity.get("dataset", ""))
        try:
            subject = int(identity.get("subject", -1))
        except (TypeError, ValueError) as error:
            raise RuntimeError("pretraining source subject is malformed") from error
        if dataset not in PRIMARY_NATIVE_DEVELOPMENT_SOURCES:
            raise PermissionError(f"checkpoint contains unauthorized source {dataset}")
        if subject not in PRIMARY_NATIVE_DEVELOPMENT_SOURCES[dataset]:
            raise PermissionError(
                f"checkpoint contains unauthorized source {dataset} S{subject}"
            )
        key = (dataset, subject)
        if key in observed:
            raise RuntimeError("checkpoint repeats a pretraining source identity")
        observed.add(key)
        for digest_name in (
            "cache_file_sha256",
            "cache_identity_sha256",
            "cache_array_sha256",
            "authorized_rows_sha256",
        ):
            _validate_hex_sha256(str(identity.get(digest_name, "")), name=digest_name)
    expected = {
        (dataset, subject)
        for dataset, subjects in PRIMARY_NATIVE_DEVELOPMENT_SOURCES.items()
        for subject in subjects
    }
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise RuntimeError(
            f"checkpoint source corpus differs from the locked corpus; missing={missing}, extra={extra}"
        )
    if int(corpus.get("source_record_count", -1)) != len(expected):
        raise RuntimeError("checkpoint source record count is stale")
    _validate_hex_sha256(str(corpus.get("sha256", "")), name="corpus sha256")


def load_immutable_native_checkpoint(
    path: str | Path,
    *,
    expected_file_sha256: str,
    expected_checkpoint_schema: str = CHECKPOINT_SCHEMA,
    expected_model_identity: str = "cardinal_fbc",
) -> ImmutableNativeCheckpoint:
    """Read and validate one pretraining checkpoint as a byte-pinned snapshot."""

    expected_hash = _validate_hex_sha256(
        expected_file_sha256, name="expected checkpoint file sha256"
    )
    checkpoint_path = Path(path).resolve(strict=True)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint is not a regular file: {checkpoint_path}")
    raw = checkpoint_path.read_bytes()
    observed_hash = _sha256_bytes(raw)
    if observed_hash != expected_hash:
        raise RuntimeError(
            "pretraining checkpoint bytes differ from the caller-pinned SHA-256"
        )
    payload = torch.load(
        io.BytesIO(raw),
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(payload, Mapping):
        raise RuntimeError("pretraining checkpoint payload is not a mapping")
    if payload.get("schema") != expected_checkpoint_schema:
        raise RuntimeError("pretraining checkpoint schema is not supported")
    if payload.get("mode") != "development" or payload.get("confirmation_access") is not False:
        raise PermissionError("checkpoint is not an authorized development artifact")
    model = payload.get("model")
    if not isinstance(model, Mapping) or model.get("identity") != expected_model_identity:
        raise RuntimeError(
            f"checkpoint is not canonical {expected_model_identity}"
        )
    construction = model.get("construction")
    if not isinstance(construction, Mapping):
        raise RuntimeError("checkpoint model construction record is missing")
    expected_construction = {
        "n_channels": 21,
        "n_outputs": 2,
        "n_times": 320,
    }
    for name, expected in expected_construction.items():
        if int(construction.get(name, -1)) != expected:
            raise RuntimeError(f"checkpoint model construction has stale {name}")
    if tuple(construction.get("channel_names", ())) != CANONICAL_21_CHANNELS:
        raise RuntimeError("checkpoint canonical channel order is stale")
    recorded_positions = np.asarray(
        construction.get("channel_positions", ()), dtype=np.float64
    )
    if recorded_positions.shape != (21, 3) or not np.allclose(
        recorded_positions,
        np.asarray(CANONICAL_21_POSITIONS, dtype=np.float64),
        rtol=0.0,
        atol=1e-7,
    ):
        raise RuntimeError("checkpoint canonical channel coordinates are stale")
    state_value = payload.get("model_state_dict")
    if not isinstance(state_value, Mapping) or not state_value:
        raise RuntimeError("checkpoint model state is missing")
    if any(not isinstance(name, str) for name in state_value):
        raise RuntimeError("checkpoint state keys must be strings")
    if any(not isinstance(value, Tensor) for value in state_value.values()):
        raise RuntimeError("checkpoint state values must be tensors")
    state = _clone_cpu_state(state_value)  # type: ignore[arg-type]
    if any(value.device.type != "cpu" for value in state.values()):
        raise RuntimeError("checkpoint snapshot retained non-CPU tensors")
    if any(
        (value.is_floating_point() or value.is_complex())
        and not bool(torch.isfinite(value).all())
        for value in state.values()
    ):
        raise RuntimeError("checkpoint state contains non-finite values")
    state_hash = state_dict_sha256(state)
    state_hashes = payload.get("state_hashes")
    if not isinstance(state_hashes, Mapping):
        raise RuntimeError("checkpoint state hashes are missing")
    if state_hash != state_hashes.get("checkpoint"):
        raise RuntimeError("checkpoint state digest does not match its payload")
    corpus = payload.get("corpus")
    if not isinstance(corpus, Mapping):
        raise RuntimeError("checkpoint corpus identity is missing")
    _validate_pretraining_corpus_identity(corpus)
    training_config = payload.get("training_config")
    partition = payload.get("partition")
    if not isinstance(training_config, Mapping):
        raise RuntimeError("checkpoint pretraining configuration is missing")
    if not isinstance(partition, Mapping):
        raise RuntimeError("checkpoint pretraining partition is missing")
    partition_hash = str(partition.get("sha256", ""))
    _validate_hex_sha256(partition_hash, name="pretraining partition sha256")
    best_epoch = int(payload.get("best_epoch_zero_based", -1))
    selected_epoch_count = int(payload.get("selected_epoch_count", -1))
    if best_epoch < 0 or selected_epoch_count != best_epoch + 1:
        raise RuntimeError("checkpoint selected pretraining duration is inconsistent")
    pretraining = {
        "training_config": copy.deepcopy(dict(training_config)),
        "partition": copy.deepcopy(dict(partition)),
        "state_hashes": copy.deepcopy(dict(state_hashes)),
        "best_epoch_zero_based": best_epoch,
        "selected_epoch_count": selected_epoch_count,
    }
    source_code_value = payload.get("source_code")
    source_code = (
        MappingProxyType(copy.deepcopy(dict(source_code_value)))
        if isinstance(source_code_value, Mapping)
        else None
    )
    return ImmutableNativeCheckpoint(
        path=str(checkpoint_path),
        file_sha256=observed_hash,
        size_bytes=len(raw),
        state_sha256=state_hash,
        state=MappingProxyType(state),
        corpus=MappingProxyType(copy.deepcopy(dict(corpus))),
        model=MappingProxyType(copy.deepcopy(dict(model))),
        pretraining=MappingProxyType(pretraining),
        source_code=source_code,
        checkpoint_schema=str(payload["schema"]),
    )


def _native_cache_path(cache_root: Path, dataset: str, subject: int) -> Path:
    return (
        cache_root
        / str(PREPROCESSING["schema"])
        / MONTAGE_PROFILE
        / dataset
        / f"subject_{subject:03d}.npz"
    )


def load_native_target_subject(
    cache_root: str | Path,
    *,
    dataset: str,
    subject: int,
    fold: int,
    condition: str = PRETRAINED_CARDINAL_FBC,
    seed: int = 7,
) -> NativeTargetSubject:
    """Load one authorized target cache after a fail-before-path preflight."""

    return _load_native_target_subject_for_record(
        cache_root,
        dataset=dataset,
        subject=subject,
        fold=fold,
        condition=condition,
        seed=seed,
        validator=validate_target_record,
    )


def _load_native_target_subject_for_record(
    cache_root: str | Path,
    *,
    dataset: str,
    subject: int,
    fold: int,
    condition: str,
    seed: int,
    validator: Callable[[str, int, int, str, int], tuple[str, int, int, str, int]],
) -> NativeTargetSubject:
    """Shared loader whose caller supplies a fail-before-path family validator."""

    key, subject_id, fold_id, _, _ = validator(
        dataset, subject, fold, condition, seed
    )
    root = Path(cache_root).resolve()
    cache_path = _native_cache_path(root, key, subject_id)
    before_hash = _sha256_file(cache_path)
    cached = load_subject_cache(
        key,
        subject_id,
        cache_root=root,
        montage_profile=MONTAGE_PROFILE,
    )
    after_hash = _sha256_file(cache_path)
    if before_hash != after_hash:
        raise RuntimeError("target cache bytes changed while being loaded")
    identity = copy.deepcopy(cached["identity"])
    if identity.get("montage_profile") != MONTAGE_PROFILE:
        raise PermissionError("target cache is not native-montage")
    coordinates = identity.get("coordinates")
    if not isinstance(coordinates, Mapping) or not bool(
        coordinates.get("point_electrode_continuity_evidence", False)
    ):
        raise PermissionError("target cache lacks point-electrode coordinate evidence")
    values = np.asarray(cached["x"], dtype=np.float32)
    labels = np.asarray(cached["y"], dtype=np.int64)
    positions = np.asarray(cached["positions"], dtype=np.float32)
    names = tuple(str(value) for value in cached["channel_names"].tolist())
    sessions = np.asarray(cached["sessions"]).astype(str)
    runs = np.asarray(cached["runs"]).astype(str)
    if set(labels.tolist()) != {0, 1}:
        raise RuntimeError("native target cache is not binary left/right")
    train, validation, test = split_indices(
        key,
        labels,
        sessions,
        runs,
        fold=fold_id,
        subject=subject_id,
    )
    return NativeTargetSubject(
        dataset=key,
        subject=subject_id,
        fold=fold_id,
        x=values,
        y=labels,
        positions=positions,
        channel_names=names,
        sessions=sessions,
        runs=runs,
        train_rows=train,
        validation_rows=validation,
        test_rows=test,
        cache_path=str(cache_path),
        cache_file_sha256=after_hash,
        cache_identity=MappingProxyType(identity),
    )


def _normalize_unit_positions(values: np.ndarray, *, name: str) -> Float64Array:
    positions = np.asarray(values, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"{name} positions must have shape (channels, 3)")
    if not np.isfinite(positions).all():
        raise ValueError(f"{name} positions contain non-finite values")
    norms = np.linalg.norm(positions, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError(f"{name} positions contain a zero vector")
    return positions / norms


def _spherical_spline_g(
    cosine: Float64Array,
    *,
    stiffness: int,
    n_legendre_terms: int,
) -> Float64Array:
    if stiffness <= 0 or n_legendre_terms <= 0:
        raise ValueError("spherical-spline order and term count must be positive")
    orders = np.arange(1, n_legendre_terms + 1, dtype=np.float64)
    coefficients = (2.0 * orders + 1.0) / (
        np.power(orders, stiffness)
        * np.power(orders + 1.0, stiffness)
        * (4.0 * np.pi)
    )
    return legval(
        np.asarray(cosine, dtype=np.float64),
        np.concatenate((np.zeros(1, dtype=np.float64), coefficients)),
    )


def fit_spherical_spline_interpolator(
    x_train: FloatArray,
    source_positions: FloatArray,
    source_channel_names: Sequence[str],
    *,
    target_positions: FloatArray = np.asarray(
        CANONICAL_21_POSITIONS, dtype=np.float32
    ),
    target_channel_names: Sequence[str] = CANONICAL_21_CHANNELS,
    stiffness: int = 4,
    n_legendre_terms: int = 50,
    regularization: float = 1e-5,
) -> SphericalSplineInterpolator:
    """Fit the standard regularized spherical-spline map from train geometry.

    ``x_train`` is accepted solely to prove the fit call is made from the
    training partition and to validate its montage.  The interpolation matrix
    is label-free and amplitude-free: it depends only on source/target unit
    coordinates and the prespecified spline constants.
    """

    train = np.asarray(x_train, dtype=np.float32)
    if train.ndim != 3 or len(train) == 0:
        raise ValueError("spherical-spline fit requires nonempty training EEG")
    if not np.isfinite(train).all():
        raise ValueError("spherical-spline training EEG contains non-finite values")
    source_names = tuple(source_channel_names)
    target_names = tuple(target_channel_names)
    if (
        len(source_names) != train.shape[1]
        or not source_names
        or any(not isinstance(name, str) or not name for name in source_names)
        or len(set(source_names)) != len(source_names)
    ):
        raise ValueError("source channel names do not match the training montage")
    if (
        not target_names
        or any(not isinstance(name, str) or not name for name in target_names)
        or len(set(target_names)) != len(target_names)
    ):
        raise ValueError("target channel names must be nonempty and unique")
    if regularization <= 0.0:
        raise ValueError("spherical-spline regularization must be positive")
    source = _normalize_unit_positions(source_positions, name="source")
    target = _normalize_unit_positions(target_positions, name="target")
    if len(source) != len(source_names) or len(target) != len(target_names):
        raise ValueError("spherical-spline coordinates and channel names differ")
    if len(source) < 4:
        raise ValueError("spherical-spline interpolation requires at least four electrodes")

    source_kernel = _spherical_spline_g(
        source @ source.T,
        stiffness=stiffness,
        n_legendre_terms=n_legendre_terms,
    )
    source_kernel.flat[:: len(source) + 1] += float(regularization)
    augmented = np.empty((len(source) + 1, len(source) + 1), dtype=np.float64)
    augmented[:-1, :-1] = source_kernel
    augmented[:-1, -1] = 1.0
    augmented[-1, :-1] = 1.0
    augmented[-1, -1] = 0.0
    evaluation = np.concatenate(
        (
            _spherical_spline_g(
                target @ source.T,
                stiffness=stiffness,
                n_legendre_terms=n_legendre_terms,
            ),
            np.ones((len(target), 1), dtype=np.float64),
        ),
        axis=1,
    )
    # The Moore-Penrose inverse follows the established MNE/Perrin numerical
    # convention and remains stable for nearly redundant electrode geometry.
    coefficients = pinv(augmented)[:, :-1]
    matrix = np.ascontiguousarray(evaluation @ coefficients, dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise RuntimeError("spherical-spline matrix contains non-finite values")
    np.testing.assert_allclose(
        matrix @ np.ones(len(source), dtype=np.float64),
        np.ones(len(target), dtype=np.float64),
        rtol=0.0,
        atol=1e-8,
        err_msg="spherical-spline map does not preserve constant potentials",
    )
    return SphericalSplineInterpolator(
        matrix=matrix,
        source_channel_names=source_names,
        source_positions=source,
        target_channel_names=target_names,
        target_positions=target,
        stiffness=int(stiffness),
        n_legendre_terms=int(n_legendre_terms),
        regularization=float(regularization),
        matrix_sha256=_array_sha256(matrix),
    )


def _make_factory(
    model_name: str,
    *,
    channel_names: tuple[str, ...],
    positions: FloatArray,
    n_times: int,
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
            channel_positions=positions_tensor,
        )

    return factory


def cardinal_checkpoint_to_indexed_fbcnet(
    pretrained_state: Mapping[str, Tensor],
) -> dict[str, Tensor]:
    """Evaluate a CardinalFBC checkpoint on the canonical indexed montage."""

    source_hash_before = state_dict_sha256(pretrained_state)
    positions = np.asarray(CANONICAL_21_POSITIONS, dtype=np.float32)
    cardinal_factory = _make_factory(
        "cardinal_fbc",
        channel_names=CANONICAL_21_CHANNELS,
        positions=positions,
        n_times=320,
    )
    indexed_factory = _make_factory(
        "fbcnet",
        channel_names=CANONICAL_21_CHANNELS,
        positions=positions,
        n_times=320,
    )
    cardinal = cardinal_factory().cpu()
    cardinal.load_state_dict(_clone_cpu_state(pretrained_state), strict=True)
    if not isinstance(cardinal, CardinalFBCNet):
        raise TypeError("Cardinal checkpoint factory did not return CardinalFBCNet")
    indexed = indexed_factory().cpu()
    spatial = getattr(indexed, "spatial_conv", None)
    if not isinstance(spatial, nn.Sequential) or len(spatial) < 2:
        raise TypeError("indexed FBCNet has an unexpected spatial block")
    indexed_spatial = spatial[0]
    if not hasattr(indexed_spatial, "weight") or not hasattr(indexed_spatial, "bias"):
        raise TypeError("indexed FBCNet spatial convolution is missing parameters")

    # Copy every shared reference component explicitly; this fails closed if a
    # future Braindecode release changes the architecture/state contract.
    pairs = (
        (indexed.spectral_filtering, cardinal.spectral_filtering, "spectral_filtering"),
        (spatial[1], cardinal.batch_norm, "spatial_batch_norm"),
        (indexed.padding_layer, cardinal.padding_layer, "padding_layer"),
        (indexed.temporal_layer, cardinal.temporal_layer, "temporal_layer"),
        (indexed.flatten_layer, cardinal.flatten_layer, "flatten_layer"),
        (indexed.final_layer, cardinal.final_layer, "final_layer"),
    )
    for target, source, name in pairs:
        try:
            target.load_state_dict(source.state_dict(), strict=True)
        except RuntimeError as error:
            raise RuntimeError(f"cannot transfer CardinalFBC {name} to indexed FBCNet") from error

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
            raise RuntimeError("continuous and indexed FBC spatial budgets differ")
        parametrizations = getattr(indexed_spatial, "parametrizations", None)
        weight_parametrization = (
            getattr(parametrizations, "weight", None)
            if parametrizations is not None
            else None
        )
        original_weight = getattr(weight_parametrization, "original", None)
        if not isinstance(original_weight, Tensor):
            raise TypeError("indexed FBCNet spatial max-norm parameterization is missing")
        # ``Conv2dWithConstraint.weight`` is a computed parametrized view;
        # copying into that view is a no-op.  Install the evaluated field in
        # the persistent original parameter, after which the same max-norm
        # parametrization used by FBCNet is applied on every forward pass.
        original_weight.copy_(fields.reshape_as(original_weight))
        if indexed_spatial.bias is None:
            raise RuntimeError("indexed FBCNet unexpectedly has no spatial bias")
        indexed_spatial.bias.copy_(cardinal.spatial_bias.reshape_as(indexed_spatial.bias))
    result = _clone_cpu_state(indexed.state_dict())
    # Prove the conversion did not mutate the reusable source checkpoint.
    if state_dict_sha256(pretrained_state) != source_hash_before:
        raise RuntimeError("source checkpoint changed during indexed conversion")
    return result


def _scratch_state(factory: ModelFactory, *, seed: int) -> dict[str, Tensor]:
    configure_determinism(seed)
    return _clone_cpu_state(factory().state_dict())


def _scaler_arrays(record: Mapping[str, object], n_channels: int) -> tuple[FloatArray, FloatArray]:
    try:
        mean = np.asarray(record["mean"], dtype=np.float32).reshape(1, n_channels, 1)
        std = np.asarray(record["std"], dtype=np.float32).reshape(1, n_channels, 1)
    except (KeyError, ValueError) as error:
        raise RuntimeError("refit scaler record is malformed") from error
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0.0):
        raise RuntimeError("refit scaler record contains invalid values")
    return mean, std


def evaluate_transfer_condition(
    target: NativeTargetSubject,
    checkpoint: ImmutableNativeCheckpoint,
    *,
    condition: str,
    train_config: TrainConfig,
) -> TransferEvaluation:
    """Fit and predict one condition with no state shared across records."""

    _, _, _, condition_key, seed = validate_target_record(
        target.dataset,
        target.subject,
        target.fold,
        condition,
        train_config.seed,
    )
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

    if condition_key == PRETRAINED_INDEXED_FBCNET_SPLINE:
        interpolator = fit_spherical_spline_interpolator(
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
            "fbcnet", channel_names=channels, positions=positions, n_times=x_train_raw.shape[2]
        )
        start_state = cardinal_checkpoint_to_indexed_fbcnet(checkpoint.state)
        derived_state_hash = state_dict_sha256(start_state)
        effective_model = "fbcnet"
        initialization = "derived_from_pretrained_cardinal_field_on_canonical21"
        interpolation_record = {
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
            "source_positions_sha256": _array_sha256(interpolator.source_positions),
            "target_positions_sha256": _array_sha256(interpolator.target_positions),
            "matrix_shape": list(interpolator.matrix.shape),
            "matrix_sha256": interpolator.matrix_sha256,
            "training_rows_sha256": _array_sha256(target.train_rows),
        }
    elif condition_key == PRETRAINED_CARDINAL_FBC:
        factory = _make_factory(
            "cardinal_fbc", channel_names=channels, positions=positions, n_times=x_train_raw.shape[2]
        )
        start_state = _clone_cpu_state(checkpoint.state)
        effective_model = "cardinal_fbc"
        initialization = "immutable_native_pretraining_checkpoint"
    elif condition_key == SCRATCH_CARDINAL_FBC:
        factory = _make_factory(
            "cardinal_fbc", channel_names=channels, positions=positions, n_times=x_train_raw.shape[2]
        )
        start_state = _scratch_state(factory, seed=seed)
        effective_model = "cardinal_fbc"
        initialization = "fresh_seeded_scratch"
    elif condition_key == SCRATCH_FBMSNET:
        factory = _make_factory(
            "fbmsnet", channel_names=channels, positions=positions, n_times=x_train_raw.shape[2]
        )
        start_state = _scratch_state(factory, seed=seed)
        effective_model = "fbmsnet"
        initialization = "fresh_seeded_scratch"
    else:  # pragma: no cover - guarded above
        raise RuntimeError("unreachable transfer condition")

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
        raise RuntimeError("immutable pretraining checkpoint was mutated during target fitting")
    if result.pretrained_checkpoint_sha256 != initialization_hash:
        raise RuntimeError("target fitter did not start from the declared condition state")
    refit_mean, refit_std = _scaler_arrays(result.refit_scaler, len(channels))
    x_test = apply_channel_scaler(x_test_raw, refit_mean, refit_std)
    probabilities = predict_probabilities(
        result.model,
        x_test,
        positions,
        device=train_config.device,
    )
    probabilities = np.asarray(probabilities, dtype=np.float64)
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
    package = Path(__file__).resolve().parent
    project = package.parent
    paths = {
        "eeg_mi/baselines.py": package / "baselines.py",
        "eeg_mi/config.py": package / "config.py",
        "eeg_mi/data.py": package / "data.py",
        "eeg_mi/models.py": package / "models.py",
        "eeg_mi/native_pretraining.py": package / "native_pretraining.py",
        "eeg_mi/native_pretraining_cli.py": package / "native_pretraining_cli.py",
        "eeg_mi/native_transfer.py": Path(__file__).resolve(),
        "eeg_mi/training.py": package / "training.py",
        "eeg_mi/requirements-cu128.txt": package / "requirements-cu128.txt",
    }
    root_requirements = project / "requirements.txt"
    if root_requirements.is_file():
        paths["requirements.txt"] = root_requirements
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required transfer source files are missing: {missing}")
    return paths


def _source_file_hashes() -> dict[str, str]:
    return {name: _sha256_file(path) for name, path in sorted(_source_paths().items())}


def _distribution_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _environment_record(device: str) -> dict[str, object]:
    cuda_devices: list[dict[str, object]] = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            cuda_devices.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "compute_capability": list(torch.cuda.get_device_capability(index)),
                }
            )
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "requested_device": device,
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_devices": cuda_devices,
        "packages": {
            name: _distribution_version(name)
            for name in ("braindecode", "mne", "moabb", "numpy", "scipy", "torch")
        },
    }


def _split_record(target: NativeTargetSubject) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, rows in (
        ("train", target.train_rows),
        ("validation", target.validation_rows),
        ("test", target.test_rows),
    ):
        result[name] = {
            "count": len(rows),
            "rows_sha256": _array_sha256(rows),
            "labels_sha256": _array_sha256(target.y[rows]),
            "sessions": sorted(set(target.sessions[rows].tolist())),
            "runs": sorted(set(target.runs[rows].tolist())),
        }
    return result


def _provenance_payload(
    *,
    target: NativeTargetSubject,
    checkpoint: ImmutableNativeCheckpoint,
    evaluation: TransferEvaluation,
    train_config: TrainConfig,
    source_hashes: Mapping[str, str],
    predictions_file_sha256: str,
    predictions_size_bytes: int,
    started_utc: str,
    elapsed_seconds: float,
    artifact_schema: str = ARTIFACT_SCHEMA,
    prediction_schema: str = PREDICTION_SCHEMA,
    predictions_filename: str = PREDICTIONS_FILENAME,
    evidence_scope: str = EVIDENCE_SCOPE,
    analysis_policy: str = ANALYSIS_POLICY,
) -> dict[str, object]:
    fine = evaluation.fine_tune
    return {
        "schema": artifact_schema,
        "mode": "development",
        "confirmation_access": False,
        "evidence_scope": evidence_scope,
        "analysis_policy": analysis_policy,
        "created_utc": _utc_now(),
        "started_utc": started_utc,
        "elapsed_seconds": float(elapsed_seconds),
        "record": {
            "dataset": target.dataset,
            "subject": target.subject,
            "fold": target.fold,
            "seed": train_config.seed,
            "condition": evaluation.requested_condition,
        },
        "target": {
            "role": "development_target_transfer",
            "montage_profile": MONTAGE_PROFILE,
            "cache_path": target.cache_path,
            "cache_file_sha256": target.cache_file_sha256,
            "cache_identity": target.cache_identity,
            "raw_shape": list(target.x.shape),
            "raw_array_sha256": _array_sha256(target.x),
            "channels": list(target.channel_names),
            "positions_sha256": _array_sha256(target.positions),
            "split": _split_record(target),
        },
        "source_checkpoint": {
            "schema": checkpoint.checkpoint_schema,
            "path": checkpoint.path,
            "file_sha256": checkpoint.file_sha256,
            "size_bytes": checkpoint.size_bytes,
            "state_sha256": checkpoint.state_sha256,
            "model": checkpoint.model,
            "corpus": checkpoint.corpus,
            "pretraining": checkpoint.pretraining,
            "read_policy": "single byte snapshot matched to caller-pinned SHA-256",
        },
        "condition": {
            "requested": evaluation.requested_condition,
            "effective_model": evaluation.effective_model,
            "initialization": evaluation.initialization,
            "initialization_state_sha256": evaluation.initialization_state_sha256,
            "derived_state_sha256": evaluation.derived_state_sha256,
            "model_class": evaluation.model_class,
            "model_config": evaluation.model_config,
            "trainable_parameter_count": evaluation.trainable_parameter_count,
            "channels": list(evaluation.channels),
            "positions_sha256": _array_sha256(evaluation.positions),
            "interpolation": evaluation.interpolation,
        },
        "protocol": {
            "phase_a": {
                "scaler_fit": "target train rows only",
                "optimization": "target train rows only",
                "selection": "minimum target validation cross-entropy",
                "selected_checkpoint_disposition": "discarded",
                "best_epoch_zero_based": fine.best_epoch,
                "selected_epoch_count": fine.best_epoch + 1,
                "selection_history": fine.selection_history,
                "selection_scaler": fine.selection_scaler,
                "selection_start_sha256": fine.selection_start_sha256,
                "selection_state_sha256": fine.selection_state_sha256,
            },
            "phase_b": {
                "initialization": "fresh model restored from identical condition state",
                "batch_norm": "running statistics reset; affine parameters retained",
                "scaler_fit": "target train plus validation rows; test excluded",
                "optimization": "target train plus validation rows",
                "epoch_count": fine.best_epoch + 1,
                "refit_history": fine.refit_history,
                "refit_scaler": fine.refit_scaler,
                "refit_start_sha256": fine.refit_start_sha256,
                "refit_state_sha256": fine.refit_state_sha256,
                "reset_batch_norm_modules": list(fine.reset_batch_norm_modules),
            },
            "test": {
                "access": "one prediction pass after Phase B",
                "scaler": "Phase B train-plus-validation scaler",
                "aggregate_metrics_emitted": False,
                "inferential_statistics_emitted": False,
            },
            "cross_record_state_carry": False,
            "train_config": asdict(train_config),
            "channel_scaling_contract": CHANNEL_SCALING,
            "preprocessing": preprocessing_for_dataset(target.dataset),
        },
        "predictions": {
            "schema": prediction_schema,
            "filename": predictions_filename,
            "file_sha256": predictions_file_sha256,
            "size_bytes": predictions_size_bytes,
            "row_count": len(target.test_rows),
            "test_rows_sha256": _array_sha256(target.test_rows),
            "test_labels_sha256": _array_sha256(evaluation.test_labels),
            "probabilities_sha256": _array_sha256(evaluation.probabilities),
            "predicted_labels_sha256": _array_sha256(evaluation.predicted_labels),
        },
        "source_code": {
            "hash_algorithm": "sha256",
            "files": dict(source_hashes),
        },
        "environment": _environment_record(train_config.device),
    }


def _write_predictions(
    path: Path,
    *,
    target: NativeTargetSubject,
    evaluation: TransferEvaluation,
    prediction_schema: str = PREDICTION_SCHEMA,
) -> None:
    with path.open("xb") as handle:
        np.savez_compressed(
            handle,
            schema=np.asarray(prediction_schema),
            dataset=np.asarray(target.dataset),
            subject=np.asarray(target.subject, dtype=np.int64),
            fold=np.asarray(target.fold, dtype=np.int64),
            condition=np.asarray(evaluation.requested_condition),
            test_rows=target.test_rows.astype(np.int64, copy=False),
            test_labels=evaluation.test_labels.astype(np.int64, copy=False),
            predicted_labels=evaluation.predicted_labels.astype(np.int64, copy=False),
            probabilities=evaluation.probabilities.astype(np.float64, copy=False),
            sessions=target.sessions[target.test_rows].astype(str),
            runs=target.runs[target.test_rows].astype(str),
        )
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    encoded = (
        json.dumps(_jsonable(payload), sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _publish_record(
    output: Path,
    *,
    target: NativeTargetSubject,
    checkpoint: ImmutableNativeCheckpoint,
    evaluation: TransferEvaluation,
    train_config: TrainConfig,
    source_hashes: Mapping[str, str],
    started_utc: str,
    elapsed_seconds: float,
    artifact_schema: str = ARTIFACT_SCHEMA,
    prediction_schema: str = PREDICTION_SCHEMA,
    predictions_filename: str = PREDICTIONS_FILENAME,
    provenance_filename: str = PROVENANCE_FILENAME,
    evidence_scope: str = EVIDENCE_SCOPE,
    analysis_policy: str = ANALYSIS_POLICY,
) -> None:
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    try:
        predictions_path = staging / predictions_filename
        _write_predictions(
            predictions_path,
            target=target,
            evaluation=evaluation,
            prediction_schema=prediction_schema,
        )
        provenance = _provenance_payload(
            target=target,
            checkpoint=checkpoint,
            evaluation=evaluation,
            train_config=train_config,
            source_hashes=source_hashes,
            predictions_file_sha256=_sha256_file(predictions_path),
            predictions_size_bytes=predictions_path.stat().st_size,
            started_utc=started_utc,
            elapsed_seconds=elapsed_seconds,
            artifact_schema=artifact_schema,
            prediction_schema=prediction_schema,
            predictions_filename=predictions_filename,
            evidence_scope=evidence_scope,
            analysis_policy=analysis_policy,
        )
        _write_json(staging / provenance_filename, provenance)
        _fsync_directory(staging)
        if _path_exists(output):
            raise FileExistsError(f"refusing to overwrite output {output}")
        os.replace(staging, output)
        _fsync_directory(output.parent)
    finally:
        if _path_exists(staging):
            shutil.rmtree(staging)


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
    """Run and atomically publish one authorized development record."""

    key, subject_id, fold_id, condition_key, seed_id = validate_target_record(
        dataset, subject, fold, condition, seed
    )
    expected_hash = _validate_hex_sha256(
        expected_checkpoint_file_sha256,
        name="expected checkpoint file sha256",
    )
    output_path = Path(output).resolve()
    with _OutputClaim(output_path):
        source_hashes_before = _source_file_hashes()
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
        started_utc = _utc_now()
        started = time.perf_counter()
        evaluation = evaluate_transfer_condition(
            target,
            checkpoint,
            condition=condition_key,
            train_config=config,
        )
        elapsed = time.perf_counter() - started
        source_hashes_after = _source_file_hashes()
        if source_hashes_after != source_hashes_before:
            raise RuntimeError("transfer source code changed while the record was running")
        _publish_record(
            output_path,
            target=target,
            checkpoint=checkpoint,
            evaluation=evaluation,
            train_config=config,
            source_hashes=source_hashes_before,
            started_utc=started_utc,
            elapsed_seconds=elapsed,
        )
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate one development-only native target-transfer record."
    )
    parser.add_argument("--dataset", required=True, choices=tuple(NATIVE_TARGET_DEVELOPMENT_COHORTS))
    parser.add_argument("--subject", required=True, type=int)
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--condition", required=True, choices=TRANSFER_CONDITIONS)
    parser.add_argument("--seed", required=True, type=int, choices=FROZEN_DEVELOPMENT_SEEDS)
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
    output = run()
    print(output)


if __name__ == "__main__":  # pragma: no cover
    main()
