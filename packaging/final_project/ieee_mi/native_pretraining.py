"""Leakage-safe native-montage pretraining for binary CardinalFBC models.

This module deliberately contains no dataset loading or benchmark CLI.  It
operates on already constructed, source-authorized subject arrays and enforces
the parts of the pilot protocol that are easy to violate accidentally:

* the primary corpus excludes BNCI2014-004's supplied bipolar derivations;
* the pretraining split is an order-independent, hashed subject split;
* every optimizer macro-step receives one class-balanced batch per dataset and
  averages dataset losses equally;
* selection uses equal-dataset validation cross-entropy;
* the selected duration is refit deterministically from the exact initial
  state using all pretraining subjects; and
* every target selection/refit starts from a fresh copy of one pretrained
  checkpoint, with BatchNorm running statistics reset from source data only.

The model factory is structural rather than tied to a concrete class so the
protocol can be unit-tested with tiny coordinate-conditioned networks.  A
production factory should return a binary :class:`CardinalFBCNet` compatible
model whose ``uses_positions`` attribute is true.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from .config import CHANNEL_SCALING, DATASETS, PREPROCESSING
from .data import (
    apply_channel_scaler,
    fit_channel_scaler,
    load_subject_cache,
)
from .training import (
    TrainConfig,
    _augment as _common_augment,
    configure_determinism,
    fit_model,
    refit_model,
)


FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int64]
SubjectKey = tuple[str, str]
ModelFactory = Callable[[], nn.Module]

PRIMARY_EXCLUDED_DATASETS = frozenset({"bnci2014_004"})
DEFAULT_PARTITION_SALT = "ieee-mi-native-pretraining-v1"
PRIMARY_NATIVE_DEVELOPMENT_SOURCES = MappingProxyType(
    {
        "bnci2014_001": tuple(range(1, 10)),
        "cho2017": tuple(range(1, 16)),
        "local_exp4": (1, 3, 4, 5, 6, 7, 8, 10),
    }
)


def _dataset_key(value: str) -> str:
    return value.strip().lower().replace("-", "_")


@dataclass(frozen=True)
class NativeMontageSubject:
    """One binary left/right subject represented on one native montage."""

    dataset: str
    subject: str | int
    x: FloatArray
    y: IntArray
    positions: FloatArray
    channel_names: tuple[str, ...]

    @property
    def key(self) -> SubjectKey:
        return (_dataset_key(self.dataset), str(self.subject))


@dataclass(frozen=True)
class SubjectPartition:
    """Stable subject-disjoint pretraining selection partition."""

    train: tuple[SubjectKey, ...]
    validation: tuple[SubjectKey, ...]
    validation_fraction: float
    salt: str
    sha256: str


@dataclass(frozen=True)
class NativePretrainConfig:
    """Optimization policy for dataset-balanced pretraining."""

    macro_epochs: int = 200
    batch_size_per_dataset: int = 64
    steps_per_macro_epoch: int | None = None
    learning_rate: float = 8e-4
    weight_decay: float = 5e-4
    patience: int = 35
    min_delta: float = 1e-4
    label_smoothing: float = 0.05
    segment_probability: float = 0.5
    segment_count: int = 8
    time_shift_samples: int = 8
    noise_std: float = 0.01
    gradient_clip: float = 5.0
    validation_batch_size: int = 128
    validation_fraction: float = 0.2
    partition_salt: str = DEFAULT_PARTITION_SALT
    seed: int = 7
    device: str = "cuda"


@dataclass
class NativePretrainingResult:
    """Selected/refit pretraining state and its reproducibility record."""

    model: nn.Module
    checkpoint_state: dict[str, Tensor]
    partition: SubjectPartition
    best_epoch: int
    best_validation_equal_dataset_ce: float
    selection_history: list[dict[str, object]]
    refit_history: list[dict[str, object]]
    initial_state_sha256: str
    selection_state_sha256: str
    checkpoint_sha256: str
    selection_steps_per_epoch: int
    refit_steps_per_epoch: int
    dataset_subject_counts: dict[str, int]
    selection_scalers: dict[str, dict[str, object]]
    refit_scalers: dict[str, dict[str, object]]


@dataclass
class TargetFineTuneResult:
    """One independent source-only target fine-tune/refit result."""

    model: nn.Module
    best_epoch: int
    pretrained_checkpoint_sha256: str
    selection_start_sha256: str
    refit_start_sha256: str
    selection_state_sha256: str
    refit_state_sha256: str
    reset_batch_norm_modules: tuple[str, ...]
    selection_scaler: dict[str, object]
    refit_scaler: dict[str, object]
    selection_history: list[dict[str, float]]
    refit_history: list[dict[str, float]]


@dataclass(frozen=True)
class LoadedNativePretrainingCorpus:
    """Raw-volts locked source subjects plus immutable cache provenance."""

    subjects: tuple[NativeMontageSubject, ...]
    sources: tuple[dict[str, object], ...]
    sha256: str
    cache_schema: str
    montage_profile: str


@dataclass(frozen=True)
class _DatasetPool:
    dataset: str
    x: Tensor
    y: Tensor
    positions: Tensor
    class_rows: tuple[Tensor, Tensor]


def _cpu_state_dict(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in state.items()
    }


def state_dict_sha256(state: Mapping[str, Tensor]) -> str:
    """Hash a complete persistent model state independently of device."""

    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        # Reshape first because scalar buffers such as BatchNorm's
        # ``num_batches_tracked`` cannot be dtype-viewed directly.
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def model_state_sha256(model: nn.Module) -> str:
    """Hash every persistent tensor in ``model``."""

    return state_dict_sha256(model.state_dict())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _native_cache_path(cache_root: Path, dataset: str, subject: int) -> Path:
    return (
        cache_root
        / str(PREPROCESSING["schema"])
        / "native"
        / dataset
        / f"subject_{subject:03d}.npz"
    )


def _validate_locked_native_source_registry() -> None:
    if PREPROCESSING.get("schema") != "ieee-mi-cache-v2":
        raise RuntimeError("native pretraining requires the locked v2 cache schema")
    if set(PRIMARY_NATIVE_DEVELOPMENT_SOURCES) & PRIMARY_EXCLUDED_DATASETS:
        raise PermissionError(
            "the locked native source registry contains forbidden BNCI2014-004"
        )
    for dataset, subjects in PRIMARY_NATIVE_DEVELOPMENT_SOURCES.items():
        if dataset not in DATASETS:
            raise RuntimeError(f"unregistered native pretraining dataset {dataset}")
        spec = DATASETS[dataset]
        selected = set(subjects)
        if selected & set(spec.confirmation_subjects):
            raise PermissionError(
                f"native pretraining source registry reaches sealed subjects for {dataset}"
            )
        if not selected or not selected <= set(spec.development_subjects):
            raise RuntimeError(
                f"native pretraining source subjects are not development-only for {dataset}"
            )
    expected = {
        "bnci2014_001": tuple(range(1, 10)),
        "cho2017": tuple(range(1, 16)),
        "local_exp4": (1, 3, 4, 5, 6, 7, 8, 10),
    }
    if dict(PRIMARY_NATIVE_DEVELOPMENT_SOURCES) != expected:
        raise RuntimeError("native pretraining source registry differs from the frozen pilot")


def load_primary_native_pretraining_corpus(
    cache_root: str | Path,
    *,
    montage_profile: str = "native",
) -> LoadedNativePretrainingCorpus:
    """Load and source-scale the exact development-only native corpus.

    This function never builds a cache. It validates the fixed role registry
    before probing any path, then delegates archive/schema validation to
    :func:`load_subject_cache`. BNCI2014-001 contributes left/right rows only;
    Cho S1--15 and all eight local subjects contribute all binary rows. Each
    subject scaler is fitted independently using every row authorized for that
    subject's pretraining role.
    """

    profile = str(montage_profile).strip().lower()
    if profile != "native":
        raise PermissionError(
            "native pretraining accepts only montage_profile='native'; "
            "harmonized/profile-mixed caches are forbidden"
        )
    _validate_locked_native_source_registry()
    root = Path(cache_root).resolve()
    records: list[NativeMontageSubject] = []
    sources: list[dict[str, object]] = []
    digest_sources: list[dict[str, object]] = []
    for dataset, subject_ids in PRIMARY_NATIVE_DEVELOPMENT_SOURCES.items():
        spec = DATASETS[dataset]
        for subject in subject_ids:
            # Repeat the authorization immediately before the non-stage-aware
            # cache loader. No caller-supplied dataset/subject can reach here.
            if subject not in spec.development_subjects:
                raise PermissionError(f"S{subject} is not development data for {dataset}")
            if subject in spec.confirmation_subjects:
                raise PermissionError(f"S{subject} is sealed confirmation data for {dataset}")
            cache_path = _native_cache_path(root, dataset, subject)
            cached = load_subject_cache(
                dataset,
                subject,
                cache_root=root,
                montage_profile="native",
            )
            identity = copy.deepcopy(cached["identity"])
            # Cache identities have passed through JSON, which represents the
            # registry's subject/event tuples as arrays. Compare against the
            # same canonical JSON value rather than raw dataclass Python types.
            expected_dataset_identity = json.loads(
                json.dumps(asdict(spec), sort_keys=True)
            )
            if identity.get("dataset") != expected_dataset_identity:
                raise RuntimeError(
                    f"cache stores a stale full dataset registry for {dataset} S{subject}"
                )
            if identity.get("montage_profile") != "native":
                raise RuntimeError(f"cache is not native for {dataset} S{subject}")
            coordinates = identity.get("coordinates", {})
            if not isinstance(coordinates, dict) or not bool(
                coordinates.get("point_electrode_continuity_evidence", False)
            ):
                raise PermissionError(
                    f"cache lacks point-electrode continuity authorization for {dataset} S{subject}"
                )
            labels = np.asarray(cached["y"], dtype=np.int64)
            if dataset == "bnci2014_001":
                rows = np.flatnonzero(np.isin(labels, (0, 1))).astype(np.int64)
            else:
                rows = np.arange(len(labels), dtype=np.int64)
            selected_y = labels[rows]
            if set(selected_y.tolist()) != {0, 1}:
                raise RuntimeError(
                    f"authorized native rows are not binary left/right for {dataset} S{subject}"
                )
            channel_names = tuple(
                str(value) for value in cached["channel_names"].tolist()
            )
            selected_raw = np.asarray(cached["x"][rows], dtype=np.float32)
            row_hash = hashlib.sha256(np.ascontiguousarray(rows).tobytes()).hexdigest()
            cache_file_hash = _sha256_file(cache_path)
            identity_hash = hashlib.sha256(
                json.dumps(identity, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                )
            ).hexdigest()
            source = {
                "dataset": dataset,
                "subject": subject,
                "role": "development_pretraining_source",
                "montage_profile": "native",
                "cache_path": str(cache_path),
                "cache_size_bytes": cache_path.stat().st_size,
                "cache_file_sha256": cache_file_hash,
                "cache_identity_sha256": identity_hash,
                "cache_array_sha256": identity["array_sha256"],
                "cache_identity": identity,
                "raw_shape": list(cached["x"].shape),
                "authorized_shape": list(selected_raw.shape),
                "authorized_rows": rows.tolist(),
                "authorized_rows_sha256": row_hash,
                "authorized_label_mapping": {"0": "left_hand", "1": "right_hand"},
                "channel_names": list(channel_names),
                "scaling": "deferred to partition-aware pretraining",
            }
            sources.append(source)
            digest_sources.append(
                {
                    "dataset": dataset,
                    "subject": subject,
                    "cache_file_sha256": cache_file_hash,
                    "cache_identity_sha256": identity_hash,
                    "cache_array_sha256": identity["array_sha256"],
                    "authorized_rows_sha256": row_hash,
                }
            )
            records.append(
                NativeMontageSubject(
                    dataset=dataset,
                    subject=subject,
                    x=selected_raw,
                    y=selected_y,
                    positions=np.asarray(cached["positions"], dtype=np.float32),
                    channel_names=channel_names,
                )
            )
    validated = validate_primary_pretraining_corpus(records)
    expected_count = sum(len(value) for value in PRIMARY_NATIVE_DEVELOPMENT_SOURCES.values())
    if len(validated) != expected_count:
        raise RuntimeError("native pretraining loader did not materialize the exact corpus")
    corpus_hash = hashlib.sha256(
        json.dumps(
            digest_sources, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return LoadedNativePretrainingCorpus(
        subjects=validated,
        sources=tuple(sources),
        sha256=corpus_hash,
        cache_schema=str(PREPROCESSING["schema"]),
        montage_profile="native",
    )


def _validate_config(config: NativePretrainConfig) -> None:
    if config.macro_epochs <= 0:
        raise ValueError("macro_epochs must be positive")
    if config.batch_size_per_dataset <= 0 or config.batch_size_per_dataset % 2:
        raise ValueError("batch_size_per_dataset must be a positive even integer")
    if config.steps_per_macro_epoch is not None and config.steps_per_macro_epoch <= 0:
        raise ValueError("steps_per_macro_epoch must be positive when supplied")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if config.weight_decay < 0.0:
        raise ValueError("weight_decay must be nonnegative")
    if config.patience <= 0:
        raise ValueError("patience must be positive")
    if config.min_delta < 0.0:
        raise ValueError("min_delta must be nonnegative")
    if not 0.0 <= config.label_smoothing < 1.0:
        raise ValueError("label_smoothing must lie in [0, 1)")
    if not 0.0 <= config.segment_probability <= 1.0:
        raise ValueError("segment_probability must lie in [0, 1]")
    if config.segment_count <= 0:
        raise ValueError("segment_count must be positive")
    if config.time_shift_samples < 0:
        raise ValueError("time_shift_samples must be nonnegative")
    if config.noise_std < 0.0:
        raise ValueError("noise_std must be nonnegative")
    if config.gradient_clip <= 0.0:
        raise ValueError("gradient_clip must be positive")
    if config.validation_batch_size <= 0:
        raise ValueError("validation_batch_size must be positive")
    if not 0.0 < config.validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie in (0, 1)")
    if not config.partition_salt:
        raise ValueError("partition_salt must not be empty")


def validate_primary_pretraining_corpus(
    subjects: Sequence[NativeMontageSubject],
) -> tuple[NativeMontageSubject, ...]:
    """Validate binary point-electrode subjects for the primary corpus.

    BNCI2014-004 is rejected even if its arrays otherwise look valid because
    the locked preprocessing contract identifies its channels as supplied
    bipolar derivations. Treating those derivations as point samples at C3,
    Cz, and C4 would invalidate the spatial-field interpretation.
    """

    records = tuple(subjects)
    if not records:
        raise ValueError("the pretraining corpus is empty")
    seen: set[SubjectKey] = set()
    for record in records:
        dataset = _dataset_key(record.dataset)
        if dataset in PRIMARY_EXCLUDED_DATASETS:
            raise PermissionError(
                "primary native-montage pretraining excludes BNCI2014-004 "
                "because it contains supplied bipolar derivations"
            )
        if not dataset:
            raise ValueError("dataset names must not be empty")
        if record.key in seen:
            raise ValueError(f"duplicate pretraining subject {record.key}")
        seen.add(record.key)
        x = np.asarray(record.x)
        y = np.asarray(record.y)
        positions = np.asarray(record.positions)
        if x.ndim != 3 or x.shape[0] == 0:
            raise ValueError(f"{record.key} x must have shape (trials, channels, time)")
        if y.shape != (x.shape[0],):
            raise ValueError(f"{record.key} labels do not match trials")
        if positions.shape != (x.shape[1], 3):
            raise ValueError(f"{record.key} positions do not match channels")
        names = tuple(record.channel_names)
        if len(names) != x.shape[1] or any(not name for name in names):
            raise ValueError(f"{record.key} channel names do not match channels")
        if len(set(names)) != len(names):
            raise ValueError(f"{record.key} channel names are not unique")
        if x.shape[2] <= 0:
            raise ValueError(f"{record.key} epochs must contain samples")
        if not np.issubdtype(x.dtype, np.floating):
            raise ValueError(f"{record.key} EEG must be floating point")
        if not np.issubdtype(y.dtype, np.integer):
            raise ValueError(f"{record.key} labels must be integers")
        if not np.issubdtype(positions.dtype, np.floating):
            raise ValueError(f"{record.key} positions must be floating point")
        if not np.isfinite(x).all() or not np.isfinite(positions).all():
            raise ValueError(f"{record.key} contains non-finite values")
        if set(y.astype(np.int64).tolist()) != {0, 1}:
            raise ValueError(
                f"{record.key} must contain both remapped left/right labels 0 and 1"
            )
        if np.any(np.linalg.norm(positions, axis=1) <= 0.0):
            raise ValueError(f"{record.key} contains a zero electrode coordinate")

    datasets = defaultdict(int)
    for record in records:
        datasets[_dataset_key(record.dataset)] += 1
    too_small = sorted(dataset for dataset, count in datasets.items() if count < 2)
    if too_small:
        raise ValueError(
            "hashed subject validation requires at least two subjects per dataset; "
            f"too small={too_small}"
        )
    return records


def hashed_subject_partition(
    subjects: Sequence[NativeMontageSubject],
    *,
    validation_fraction: float = 0.2,
    salt: str = DEFAULT_PARTITION_SALT,
) -> SubjectPartition:
    """Make an order-independent per-dataset hashed subject split.

    Subjects are ranked by SHA-256 separately within every dataset. The first
    ``ceil(validation_fraction * n_subjects)`` hashes enter validation, with
    at least one subject retained in each role. This is a deterministic
    approximately 80/20 split even for small datasets and cannot separate
    trials from the same subject across roles.
    """

    records = validate_primary_pretraining_corpus(subjects)
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie in (0, 1)")
    if not salt:
        raise ValueError("salt must not be empty")
    by_dataset: dict[str, list[SubjectKey]] = defaultdict(list)
    for record in records:
        by_dataset[record.key[0]].append(record.key)

    train: list[SubjectKey] = []
    validation: list[SubjectKey] = []
    for dataset, keys in sorted(by_dataset.items()):
        ranked = sorted(
            keys,
            key=lambda key: hashlib.sha256(
                f"{salt}\0{key[0]}\0{key[1]}".encode("utf-8")
            ).digest(),
        )
        validation_count = max(
            1,
            min(len(ranked) - 1, math.ceil(validation_fraction * len(ranked))),
        )
        validation.extend(ranked[:validation_count])
        train.extend(ranked[validation_count:])

    train_tuple = tuple(sorted(train))
    validation_tuple = tuple(sorted(validation))
    payload = {
        "salt": salt,
        "validation_fraction": validation_fraction,
        "train": [list(key) for key in train_tuple],
        "validation": [list(key) for key in validation_tuple],
    }
    partition_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return SubjectPartition(
        train=train_tuple,
        validation=validation_tuple,
        validation_fraction=float(validation_fraction),
        salt=salt,
        sha256=partition_hash,
    )


def _fit_dataset_scalers(
    subjects: Sequence[NativeMontageSubject],
    included_keys: set[SubjectKey],
    *,
    fit_rows: str,
) -> dict[str, dict[str, object]]:
    """Fit one channel scaler per dataset using only authorized subject rows."""

    grouped: dict[str, list[NativeMontageSubject]] = defaultdict(list)
    for record in subjects:
        if record.key in included_keys:
            grouped[record.key[0]].append(record)
    if not grouped:
        raise ValueError("scaler fitting received no subjects")
    scalers: dict[str, dict[str, object]] = {}
    for dataset, records in sorted(grouped.items()):
        ordered = sorted(records, key=lambda record: record.key)
        names = tuple(ordered[0].channel_names)
        shape = tuple(ordered[0].x.shape[1:])
        for record in ordered[1:]:
            if tuple(record.channel_names) != names or tuple(record.x.shape[1:]) != shape:
                raise ValueError(
                    f"dataset {dataset} subjects do not share one ordered montage/time grid"
                )
        values = np.concatenate(
            [np.asarray(record.x, dtype=np.float32) for record in ordered], axis=0
        )
        mean, std = fit_channel_scaler(values, names)
        payload: dict[str, object] = {
            "schema": CHANNEL_SCALING["schema"],
            "dataset": dataset,
            "fit_rows": fit_rows,
            "fit_subjects": [list(record.key) for record in ordered],
            "row_count": int(len(values)),
            "channel_names": list(names),
            "mean": mean.reshape(-1).tolist(),
            "std": std.reshape(-1).tolist(),
            "clip_standard_deviations": CHANNEL_SCALING[
                "clip_standard_deviations"
            ],
        }
        payload["sha256"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        scalers[dataset] = payload
    return scalers


def _apply_dataset_scalers(
    subjects: Sequence[NativeMontageSubject],
    scalers: Mapping[str, Mapping[str, object]],
) -> tuple[NativeMontageSubject, ...]:
    """Return scaled copies without mutating the raw-volts corpus."""

    scaled: list[NativeMontageSubject] = []
    for record in subjects:
        dataset = record.key[0]
        if dataset not in scalers:
            raise RuntimeError(f"no fitted scaler for dataset {dataset}")
        scaler = scalers[dataset]
        names = tuple(str(value) for value in scaler["channel_names"])
        if names != tuple(record.channel_names):
            raise RuntimeError(f"scaler montage differs for dataset {dataset}")
        channels = len(names)
        mean = np.asarray(scaler["mean"], dtype=np.float32).reshape(1, channels, 1)
        std = np.asarray(scaler["std"], dtype=np.float32).reshape(1, channels, 1)
        scaled.append(
            NativeMontageSubject(
                dataset=record.dataset,
                subject=record.subject,
                x=apply_channel_scaler(record.x, mean, std),
                y=np.asarray(record.y, dtype=np.int64),
                positions=np.asarray(record.positions, dtype=np.float32),
                channel_names=tuple(record.channel_names),
            )
        )
    return tuple(scaled)


def _make_dataset_pools(
    subjects: Sequence[NativeMontageSubject],
    included_keys: set[SubjectKey],
) -> dict[str, _DatasetPool]:
    grouped: dict[str, list[NativeMontageSubject]] = defaultdict(list)
    for record in subjects:
        if record.key in included_keys:
            grouped[record.key[0]].append(record)
    if not grouped:
        raise ValueError("subject selection produced no dataset pools")

    pools: dict[str, _DatasetPool] = {}
    for dataset, records in sorted(grouped.items()):
        # Corpus input order must not change trial-row identities sampled by a
        # seeded generator. Subject keys provide a stable concatenation order.
        records = sorted(records, key=lambda record: record.key)
        reference_shape = tuple(records[0].x.shape[1:])
        reference_positions = np.asarray(records[0].positions, dtype=np.float32)
        for record in records[1:]:
            if tuple(record.x.shape[1:]) != reference_shape:
                raise ValueError(
                    f"dataset {dataset} subjects do not share one channel/time shape"
                )
            if not np.array_equal(
                np.asarray(record.positions, dtype=np.float32), reference_positions
            ):
                raise ValueError(
                    f"dataset {dataset} subjects do not share one coordinate montage"
                )
        x = torch.as_tensor(
            np.concatenate(
                [np.asarray(record.x, dtype=np.float32) for record in records], axis=0
            ),
            dtype=torch.float32,
        )
        y = torch.as_tensor(
            np.concatenate(
                [np.asarray(record.y, dtype=np.int64) for record in records], axis=0
            ),
            dtype=torch.long,
        )
        class_rows = tuple(
            torch.nonzero(y == label, as_tuple=False).flatten()
            for label in (0, 1)
        )
        if any(len(rows) == 0 for rows in class_rows):
            raise RuntimeError(f"dataset {dataset} pool lost a binary class")
        pools[dataset] = _DatasetPool(
            dataset=dataset,
            x=x,
            y=y,
            positions=torch.as_tensor(reference_positions, dtype=torch.float32),
            class_rows=(class_rows[0], class_rows[1]),
        )
    return pools


def _balanced_batch(
    pool: _DatasetPool,
    *,
    batch_size: int,
    config: NativePretrainConfig,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    """Sample one balanced dataset batch and augment it deterministically."""

    half = batch_size // 2
    rows = []
    for class_rows in pool.class_rows:
        sampled = torch.randint(
            len(class_rows), (half,), generator=generator
        )
        rows.append(class_rows[sampled])
    selected = torch.cat(rows)
    selected = selected[torch.randperm(len(selected), generator=generator)]
    batch_x = pool.x[selected]
    batch_y = pool.y[selected]
    # Keep the complete sampling/augmentation stream on CPU. This makes the
    # same seed reproducible across target devices and applies the shared EEG
    # recipe independently to every dataset-homogeneous batch.
    batch_x = _common_augment(batch_x, batch_y, config, generator)
    return (
        batch_x.to(device),
        batch_y.to(device),
        pool.positions.to(device),
    )


def _forward_binary(model: nn.Module, x: Tensor, positions: Tensor) -> Tensor:
    if not bool(getattr(model, "uses_positions", False)):
        raise TypeError("native-montage pretraining requires uses_positions=True")
    logits = model(x, positions)
    if isinstance(logits, tuple):
        logits = logits[0]
    if logits.shape != (len(x), 2):
        raise RuntimeError(
            "shared native-montage model must return exactly two left/right logits; "
            f"received {tuple(logits.shape)}"
        )
    return logits


def _derived_steps(pools: Mapping[str, _DatasetPool], batch_size: int) -> int:
    return max(math.ceil(len(pool.y) / batch_size) for pool in pools.values())


def _equal_dataset_validation_ce(
    model: nn.Module,
    pools: Mapping[str, _DatasetPool],
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[float, dict[str, float]]:
    model.eval()
    per_dataset: dict[str, float] = {}
    with torch.no_grad():
        for dataset, pool in sorted(pools.items()):
            total = 0.0
            seen = 0
            positions = pool.positions.to(device)
            for start in range(0, len(pool.y), batch_size):
                stop = min(start + batch_size, len(pool.y))
                x = pool.x[start:stop].to(device)
                y = pool.y[start:stop].to(device)
                logits = _forward_binary(model, x, positions)
                total += float(
                    nn.functional.cross_entropy(logits, y, reduction="sum")
                )
                seen += len(y)
            per_dataset[dataset] = total / seen
    return float(np.mean(list(per_dataset.values()))), per_dataset


def _optimization_epoch(
    model: nn.Module,
    pools: Mapping[str, _DatasetPool],
    *,
    optimizer: torch.optim.Optimizer,
    config: NativePretrainConfig,
    generator: torch.Generator,
    device: torch.device,
    steps: int,
) -> tuple[float, dict[str, float]]:
    model.train()
    dataset_totals = {dataset: 0.0 for dataset in pools}
    macro_total = 0.0
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        losses: list[Tensor] = []
        for dataset, pool in sorted(pools.items()):
            x, y, positions = _balanced_batch(
                pool,
                batch_size=config.batch_size_per_dataset,
                config=config,
                generator=generator,
                device=device,
            )
            loss = nn.functional.cross_entropy(
                _forward_binary(model, x, positions),
                y,
                label_smoothing=config.label_smoothing,
            )
            losses.append(loss)
            dataset_totals[dataset] += float(loss.detach())
        macro_loss = torch.stack(losses).mean()
        macro_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()
        macro_total += float(macro_loss.detach())
    return (
        macro_total / steps,
        {dataset: value / steps for dataset, value in dataset_totals.items()},
    )


def _selection_fit(
    model: nn.Module,
    train_pools: Mapping[str, _DatasetPool],
    validation_pools: Mapping[str, _DatasetPool],
    *,
    config: NativePretrainConfig,
    steps: int,
) -> tuple[dict[str, Tensor], int, float, list[dict[str, object]]]:
    configure_determinism(config.seed)
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model.to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("pretraining model has no trainable parameters")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.macro_epochs
    )
    generator = torch.Generator(device="cpu").manual_seed(config.seed + 104729)
    best_state: dict[str, Tensor] | None = None
    best_epoch = -1
    best_loss = float("inf")
    stale = 0
    history: list[dict[str, object]] = []
    for epoch in range(config.macro_epochs):
        train_loss, dataset_train_loss = _optimization_epoch(
            model,
            train_pools,
            optimizer=optimizer,
            config=config,
            generator=generator,
            device=device,
            steps=steps,
        )
        scheduler.step()
        validation_loss, dataset_validation_loss = _equal_dataset_validation_ce(
            model,
            validation_pools,
            device=device,
            batch_size=config.validation_batch_size,
        )
        if not math.isfinite(validation_loss):
            raise RuntimeError("pretraining produced non-finite validation loss")
        history.append(
            {
                "epoch": epoch,
                "training_equal_dataset_loss": train_loss,
                "training_dataset_loss": dataset_train_loss,
                "validation_equal_dataset_ce": validation_loss,
                "validation_dataset_ce": dataset_validation_loss,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if validation_loss < best_loss - config.min_delta:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = _cpu_state_dict(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("pretraining selection produced no finite checkpoint")
    model.load_state_dict(best_state)
    return best_state, best_epoch, best_loss, history


def _deterministic_refit(
    model: nn.Module,
    pools: Mapping[str, _DatasetPool],
    *,
    epochs: int,
    config: NativePretrainConfig,
    steps: int,
) -> list[dict[str, object]]:
    if not 1 <= epochs <= config.macro_epochs:
        raise ValueError("refit epochs must lie inside the pretraining horizon")
    configure_determinism(config.seed)
    device = torch.device(config.device)
    model.to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.macro_epochs
    )
    generator = torch.Generator(device="cpu").manual_seed(config.seed + 104729)
    history: list[dict[str, object]] = []
    for epoch in range(epochs):
        train_loss, dataset_train_loss = _optimization_epoch(
            model,
            pools,
            optimizer=optimizer,
            config=config,
            generator=generator,
            device=device,
            steps=steps,
        )
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "training_equal_dataset_loss": train_loss,
                "training_dataset_loss": dataset_train_loss,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
    return history


def fit_native_montage_pretraining(
    model_factory: ModelFactory,
    subjects: Sequence[NativeMontageSubject],
    *,
    config: NativePretrainConfig = NativePretrainConfig(),
) -> NativePretrainingResult:
    """Select and deterministically refit one shared binary native model."""

    _validate_config(config)
    records = validate_primary_pretraining_corpus(subjects)
    partition = hashed_subject_partition(
        records,
        validation_fraction=config.validation_fraction,
        salt=config.partition_salt,
    )
    selection_scalers = _fit_dataset_scalers(
        records,
        set(partition.train),
        fit_rows="pretraining selection subjects: training role only",
    )
    selection_records = _apply_dataset_scalers(records, selection_scalers)
    train_pools = _make_dataset_pools(selection_records, set(partition.train))
    validation_pools = _make_dataset_pools(
        selection_records, set(partition.validation)
    )
    refit_scalers = _fit_dataset_scalers(
        records,
        {record.key for record in records},
        fit_rows="pretraining final refit subjects: training plus validation roles",
    )
    refit_records = _apply_dataset_scalers(records, refit_scalers)
    all_pools = _make_dataset_pools(
        refit_records, {record.key for record in refit_records}
    )
    if train_pools.keys() != validation_pools.keys():
        raise RuntimeError("every pretraining dataset must appear in both split roles")

    configure_determinism(config.seed)
    model = model_factory()
    initial_state = _cpu_state_dict(model.state_dict())
    initial_hash = state_dict_sha256(initial_state)
    selection_steps = (
        config.steps_per_macro_epoch
        if config.steps_per_macro_epoch is not None
        else _derived_steps(train_pools, config.batch_size_per_dataset)
    )
    selected_state, best_epoch, best_loss, selection_history = _selection_fit(
        model,
        train_pools,
        validation_pools,
        config=config,
        steps=selection_steps,
    )
    selection_hash = state_dict_sha256(selected_state)

    # The validation-selected checkpoint is discarded. The final checkpoint
    # is regenerated from the exact seeded initialization using all authorized
    # pretraining subjects for precisely the selected duration.
    model.load_state_dict(initial_state)
    refit_steps = (
        config.steps_per_macro_epoch
        if config.steps_per_macro_epoch is not None
        else _derived_steps(all_pools, config.batch_size_per_dataset)
    )
    refit_history = _deterministic_refit(
        model,
        all_pools,
        epochs=best_epoch + 1,
        config=config,
        steps=refit_steps,
    )
    checkpoint = _cpu_state_dict(model.state_dict())
    return NativePretrainingResult(
        model=model,
        checkpoint_state=checkpoint,
        partition=partition,
        best_epoch=best_epoch,
        best_validation_equal_dataset_ce=best_loss,
        selection_history=selection_history,
        refit_history=refit_history,
        initial_state_sha256=initial_hash,
        selection_state_sha256=selection_hash,
        checkpoint_sha256=state_dict_sha256(checkpoint),
        selection_steps_per_epoch=selection_steps,
        refit_steps_per_epoch=refit_steps,
        dataset_subject_counts={
            dataset: sum(record.key[0] == dataset for record in records)
            for dataset in sorted({record.key[0] for record in records})
        },
        selection_scalers=copy.deepcopy(selection_scalers),
        refit_scalers=copy.deepcopy(refit_scalers),
    )


def reset_batch_norm_running_stats(model: nn.Module) -> tuple[str, ...]:
    """Reset BN running state while proving affine parameters are retained."""

    reset: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.modules.batchnorm._BatchNorm):
            continue
        weight = module.weight.detach().clone() if module.weight is not None else None
        bias = module.bias.detach().clone() if module.bias is not None else None
        module.reset_running_stats()
        if weight is not None and not torch.equal(module.weight.detach(), weight):
            raise RuntimeError(f"BatchNorm reset changed affine weight for {name}")
        if bias is not None and not torch.equal(module.bias.detach(), bias):
            raise RuntimeError(f"BatchNorm reset changed affine bias for {name}")
        reset.append(name)
    return tuple(reset)


def _validate_target_arrays(
    x_train: FloatArray,
    y_train: IntArray,
    x_validation: FloatArray,
    y_validation: IntArray,
    positions: FloatArray,
    channel_names: tuple[str, ...],
) -> None:
    if x_train.ndim != 3 or x_validation.ndim != 3:
        raise ValueError("target EEG must have shape (trials, channels, time)")
    if x_train.shape[1:] != x_validation.shape[1:]:
        raise ValueError("target train and validation shapes are incompatible")
    if y_train.shape != (len(x_train),) or y_validation.shape != (len(x_validation),):
        raise ValueError("target labels do not match target trials")
    if positions.shape != (x_train.shape[1], 3):
        raise ValueError("target positions do not match target channels")
    if len(channel_names) != x_train.shape[1]:
        raise ValueError("target channel_names do not match target channels")
    if any(not name for name in channel_names) or len(set(channel_names)) != len(
        channel_names
    ):
        raise ValueError("target channel_names must be nonempty and unique")
    if set(np.asarray(y_train, dtype=np.int64).tolist()) != {0, 1}:
        raise ValueError("target training rows must contain both binary classes")
    if set(np.asarray(y_validation, dtype=np.int64).tolist()) != {0, 1}:
        raise ValueError("target validation rows must contain both binary classes")
    if not (
        np.isfinite(x_train).all()
        and np.isfinite(x_validation).all()
        and np.isfinite(positions).all()
    ):
        raise ValueError("target arrays contain non-finite values")


def _target_scaler_record(
    mean: FloatArray,
    std: FloatArray,
    channel_names: tuple[str, ...],
    *,
    fit_rows: str,
    row_count: int,
) -> dict[str, object]:
    return {
        "schema": CHANNEL_SCALING["schema"],
        "fit_rows": fit_rows,
        "row_count": int(row_count),
        "channel_names": list(channel_names),
        "mean": mean.reshape(-1).tolist(),
        "std": std.reshape(-1).tolist(),
        "clip_standard_deviations": CHANNEL_SCALING["clip_standard_deviations"],
    }


def fine_tune_pretrained_target(
    model_factory: ModelFactory,
    pretrained_state: Mapping[str, Tensor],
    x_train: FloatArray,
    y_train: IntArray,
    x_validation: FloatArray,
    y_validation: IntArray,
    positions: FloatArray,
    *,
    channel_names: Sequence[str],
    config: TrainConfig = TrainConfig(),
) -> TargetFineTuneResult:
    """Fine-tune one target without target-test input or cross-target state.

    Inputs are raw volts. Selection fits its channel scaler on ``x_train``
    only and reads selection-scaled ``x_validation`` only for checkpoint
    duration. The selected model is discarded. A fresh model then restores
    the identical pretrained checkpoint, resets BatchNorm running statistics,
    fits a new scaler on the raw train-plus-validation rows, and refits for the
    selected duration. There is intentionally no test-array argument.
    """

    x_train = np.asarray(x_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)
    x_validation = np.asarray(x_validation, dtype=np.float32)
    y_validation = np.asarray(y_validation, dtype=np.int64)
    positions = np.asarray(positions, dtype=np.float32)
    if isinstance(channel_names, (str, bytes)) or any(
        not isinstance(value, str) for value in channel_names
    ):
        raise TypeError("channel_names must be a sequence of strings")
    names = tuple(channel_names)
    _validate_target_arrays(
        x_train, y_train, x_validation, y_validation, positions, names
    )
    selection_mean, selection_std = fit_channel_scaler(x_train, names)
    x_train_selection = apply_channel_scaler(
        x_train, selection_mean, selection_std
    )
    x_validation_selection = apply_channel_scaler(
        x_validation, selection_mean, selection_std
    )
    selection_scaler = _target_scaler_record(
        selection_mean,
        selection_std,
        names,
        fit_rows=str(CHANNEL_SCALING["fit_rows"]["selection"]),
        row_count=len(x_train),
    )
    checkpoint = _cpu_state_dict(pretrained_state)
    checkpoint_hash = state_dict_sha256(checkpoint)

    configure_determinism(config.seed)
    selection_model = model_factory()
    selection_model.load_state_dict(checkpoint, strict=True)
    selection_reset = reset_batch_norm_running_stats(selection_model)
    selection_start_hash = model_state_sha256(selection_model)
    selection = fit_model(
        selection_model,
        x_train_selection,
        y_train,
        x_validation_selection,
        y_validation,
        positions,
        config=config,
    )

    # Constructing a second model makes it impossible for selected weights or
    # BN state to leak into refit. It must match the exact reset start hash.
    configure_determinism(config.seed)
    refit_model_instance = model_factory()
    if refit_model_instance is selection_model:
        raise RuntimeError("target model_factory must return a fresh model per call")
    refit_model_instance.load_state_dict(checkpoint, strict=True)
    refit_reset = reset_batch_norm_running_stats(refit_model_instance)
    refit_start_hash = model_state_sha256(refit_model_instance)
    if selection_reset != refit_reset or selection_start_hash != refit_start_hash:
        raise RuntimeError("target selection and refit did not start identically")
    # Refit scaling is fitted afresh from the raw arrays. Validation rows must
    # never be concatenated to train-scaled arrays from model selection.
    x_source_raw = np.concatenate((x_train, x_validation), axis=0)
    refit_mean, refit_std = fit_channel_scaler(x_source_raw, names)
    x_source = apply_channel_scaler(x_source_raw, refit_mean, refit_std)
    refit_scaler = _target_scaler_record(
        refit_mean,
        refit_std,
        names,
        fit_rows=str(CHANNEL_SCALING["fit_rows"]["final_refit"]),
        row_count=len(x_source_raw),
    )
    y_source = np.concatenate((y_train, y_validation), axis=0)
    refit = refit_model(
        refit_model_instance,
        x_source,
        y_source,
        positions,
        epochs=int(selection["best_epoch"]) + 1,
        config=config,
    )
    return TargetFineTuneResult(
        model=refit["model"],
        best_epoch=int(selection["best_epoch"]),
        pretrained_checkpoint_sha256=checkpoint_hash,
        selection_start_sha256=selection_start_hash,
        refit_start_sha256=refit_start_hash,
        selection_state_sha256=model_state_sha256(selection["model"]),
        refit_state_sha256=model_state_sha256(refit["model"]),
        reset_batch_norm_modules=selection_reset,
        selection_scaler=selection_scaler,
        refit_scaler=refit_scaler,
        selection_history=copy.deepcopy(selection["history"]),
        refit_history=copy.deepcopy(refit["history"]),
    )
