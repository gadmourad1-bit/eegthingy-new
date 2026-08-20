"""Leakage-safe paired preprocessing and training for CSDV-FBC.

This module is intentionally isolated from the production CLI and artifact
writers.  It operates only on explicit in-memory development arrays and keeps
the two deterministic views paired throughout preprocessing and optimization:

* the native view is standardized on its observed channels;
* the canonical view is first transported from raw volts with the pinned
  Perrin spherical-spline implementation and then standardized separately;
* both scalers are fitted from an explicit source-subject or target-row fit
  set, never from validation or test inputs;
* segment donors and temporal shifts are shared across the paired views;
* pretraining retains the existing hashed subject partition, class-balanced
  per-dataset batches, and equal-dataset macro loss/validation semantics; and
* target fitting exposes no test argument.  A returned one-shot predictor can
  transform and predict one test array exactly once without fitting anything.

The fixed candidate objective is imported from :mod:`benchmark.dual_view` and is
called with its frozen weights, temperature, and label smoothing explicitly.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from .config import CANONICAL_21_CHANNELS, CHANNEL_SCALING
from .data import apply_channel_scaler, fit_channel_scaler
from .dual_view import (
    CardinalSplineDualViewLoss,
    CardinalSplineDualViewOutput,
    cardinal_spline_dual_view_loss,
)
from .native_pretraining import (
    NativeMontageSubject,
    NativePretrainConfig,
    SubjectKey,
    SubjectPartition,
    hashed_subject_partition,
    model_state_sha256,
    reset_batch_norm_running_stats,
    state_dict_sha256,
    validate_primary_pretraining_corpus,
)
from .native_transfer import (
    SphericalSplineInterpolator,
    fit_spherical_spline_interpolator,
)
from .training import TrainConfig, configure_determinism


FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int64]
ModelFactory = Callable[[], nn.Module]

FROZEN_AUXILIARY_VIEW_WEIGHT = 0.25
FROZEN_PREFERENCE_WEIGHT = 0.05
FROZEN_PREFERENCE_TEMPERATURE = 0.25
FROZEN_LABEL_SMOOTHING = 0.05


def _cpu_state_dict(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in state.items()
    }


def _json_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class ChannelScalerSnapshot:
    """One immutable, auditable channel-standardization fit."""

    channel_names: tuple[str, ...]
    mean: FloatArray
    std: FloatArray
    fit_rows: str
    fit_subjects: tuple[SubjectKey, ...]
    row_count: int
    sha256: str

    def apply(self, values: FloatArray) -> FloatArray:
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 3 or values.shape[1] != len(self.channel_names):
            raise ValueError("values do not match the fitted scaler montage")
        return apply_channel_scaler(values, self.mean, self.std)


@dataclass(frozen=True)
class PairedDatasetTransform:
    """Spline plus distinct native/canonical scalers for one montage."""

    dataset: str
    interpolator: SphericalSplineInterpolator
    native_scaler: ChannelScalerSnapshot
    canonical_scaler: ChannelScalerSnapshot
    fit_rows: str
    fit_subjects: tuple[SubjectKey, ...]
    row_count: int
    sha256: str

    def transform(self, raw_values: FloatArray) -> tuple[FloatArray, FloatArray]:
        raw = np.asarray(raw_values, dtype=np.float32)
        native = self.native_scaler.apply(raw)
        canonical_raw = self.interpolator.transform(raw)
        canonical = self.canonical_scaler.apply(canonical_raw)
        return native, canonical


@dataclass(frozen=True)
class PairedMontageSubject:
    dataset: str
    subject: str | int
    native_x: FloatArray
    canonical_x: FloatArray
    y: IntArray
    positions: FloatArray
    native_channel_names: tuple[str, ...]

    @property
    def key(self) -> SubjectKey:
        return (self.dataset.strip().lower().replace("-", "_"), str(self.subject))


@dataclass(frozen=True)
class PairedAugmentationTrace:
    """Serializable evidence that temporal choices were shared by both views."""

    segment_applied: bool
    segment_sources: tuple[tuple[int, int, tuple[int, ...], tuple[int, ...]], ...]
    shifts: tuple[int, ...]


@dataclass(frozen=True)
class _PairedDatasetPool:
    dataset: str
    native_x: Tensor
    canonical_x: Tensor
    y: Tensor
    positions: Tensor
    class_rows: tuple[Tensor, Tensor]


@dataclass
class DualViewPretrainingResult:
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
    selection_transforms: dict[str, PairedDatasetTransform]
    refit_transforms: dict[str, PairedDatasetTransform]


@dataclass
class DualViewTargetResult:
    """Fitted target model plus a fail-closed one-shot raw-test predictor."""

    model: nn.Module
    positions: FloatArray
    native_channel_names: tuple[str, ...]
    n_times: int
    refit_transform: PairedDatasetTransform
    best_epoch: int
    pretrained_checkpoint_sha256: str
    selection_start_sha256: str
    refit_start_sha256: str
    selection_state_sha256: str
    refit_state_sha256: str
    reset_batch_norm_modules: tuple[str, ...]
    selection_transform: PairedDatasetTransform
    selection_history: list[dict[str, float]]
    refit_history: list[dict[str, float]]
    device: str
    _test_consumed: bool = field(default=False, init=False, repr=False)

    @property
    def test_consumed(self) -> bool:
        return self._test_consumed

    def predict_test_once(
        self,
        raw_test: FloatArray,
        *,
        batch_size: int = 128,
    ) -> NDArray[np.float64]:
        """Transform and predict one unlabeled test array exactly once."""

        if self._test_consumed:
            raise RuntimeError("target test prediction has already been consumed")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        raw = np.asarray(raw_test, dtype=np.float32)
        expected = (len(self.native_channel_names), self.n_times)
        if raw.ndim != 3 or tuple(raw.shape[1:]) != expected:
            raise ValueError(
                f"raw_test must have shape (trials, {expected[0]}, {expected[1]})"
            )
        if len(raw) == 0 or not np.isfinite(raw).all():
            raise ValueError("raw_test must be nonempty and finite")

        # Claim before any model call so an exception cannot permit adaptive
        # repeated probing of the same fitted target object.
        self._test_consumed = True
        native, canonical = self.refit_transform.transform(raw)
        device = torch.device(self.device)
        positions = torch.as_tensor(
            self.positions, dtype=torch.float32, device=device
        )
        self.model.to(device)
        self.model.eval()
        rows: list[Tensor] = []
        with torch.inference_mode():
            for start in range(0, len(raw), batch_size):
                stop = min(start + batch_size, len(raw))
                output = _forward_views(
                    self.model,
                    torch.as_tensor(
                        native[start:stop], dtype=torch.float32, device=device
                    ),
                    torch.as_tensor(
                        canonical[start:stop], dtype=torch.float32, device=device
                    ),
                    positions,
                )
                rows.append(torch.softmax(output.fused_logits, dim=1).cpu())
        probabilities = torch.cat(rows, dim=0).numpy().astype(np.float64, copy=False)
        if probabilities.shape != (len(raw), 2) or not np.isfinite(probabilities).all():
            raise RuntimeError("one-shot target prediction produced invalid probabilities")
        if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-6):
            raise RuntimeError("one-shot target probabilities do not sum to one")
        return probabilities


def _scaler_snapshot(
    values: FloatArray,
    channel_names: tuple[str, ...],
    *,
    fit_rows: str,
    fit_subjects: tuple[SubjectKey, ...],
) -> ChannelScalerSnapshot:
    mean, std = fit_channel_scaler(values, channel_names)
    payload: dict[str, object] = {
        "schema": CHANNEL_SCALING["schema"],
        "channel_names": list(channel_names),
        "fit_rows": fit_rows,
        "fit_subjects": [list(key) for key in fit_subjects],
        "row_count": int(len(values)),
        "mean": mean.reshape(-1).tolist(),
        "std": std.reshape(-1).tolist(),
        "clip_standard_deviations": CHANNEL_SCALING[
            "clip_standard_deviations"
        ],
    }
    return ChannelScalerSnapshot(
        channel_names=channel_names,
        mean=np.asarray(mean, dtype=np.float32).copy(),
        std=np.asarray(std, dtype=np.float32).copy(),
        fit_rows=fit_rows,
        fit_subjects=fit_subjects,
        row_count=len(values),
        sha256=_json_sha256(payload),
    )


def _fit_one_transform(
    dataset: str,
    records: Sequence[NativeMontageSubject],
    *,
    fit_rows: str,
) -> PairedDatasetTransform:
    ordered = tuple(sorted(records, key=lambda record: record.key))
    if not ordered:
        raise ValueError("paired transform fitting received no subjects")
    names = tuple(ordered[0].channel_names)
    shape = tuple(ordered[0].x.shape[1:])
    positions = np.asarray(ordered[0].positions, dtype=np.float32)
    for record in ordered[1:]:
        if tuple(record.channel_names) != names or tuple(record.x.shape[1:]) != shape:
            raise ValueError(f"dataset {dataset} does not share one native shape")
        if not np.array_equal(np.asarray(record.positions, dtype=np.float32), positions):
            raise ValueError(f"dataset {dataset} does not share one native geometry")
    raw = np.concatenate(
        [np.asarray(record.x, dtype=np.float32) for record in ordered], axis=0
    )
    fit_subjects = tuple(record.key for record in ordered)
    # The existing validated API accepts amplitudes only as proof of an
    # explicit fit call; its matrix is determined solely by geometry.
    interpolator = fit_spherical_spline_interpolator(raw, positions, names)
    native_scaler = _scaler_snapshot(
        raw,
        names,
        fit_rows=fit_rows,
        fit_subjects=fit_subjects,
    )
    canonical_raw = interpolator.transform(raw)
    canonical_names = tuple(interpolator.target_channel_names)
    if canonical_names != tuple(CANONICAL_21_CHANNELS):
        raise RuntimeError("spline target order is not canonical21")
    canonical_scaler = _scaler_snapshot(
        canonical_raw,
        canonical_names,
        fit_rows=fit_rows,
        fit_subjects=fit_subjects,
    )
    payload: dict[str, object] = {
        "dataset": dataset,
        "fit_rows": fit_rows,
        "fit_subjects": [list(key) for key in fit_subjects],
        "row_count": int(len(raw)),
        "spline_matrix_sha256": interpolator.matrix_sha256,
        "native_scaler_sha256": native_scaler.sha256,
        "canonical_scaler_sha256": canonical_scaler.sha256,
    }
    return PairedDatasetTransform(
        dataset=dataset,
        interpolator=interpolator,
        native_scaler=native_scaler,
        canonical_scaler=canonical_scaler,
        fit_rows=fit_rows,
        fit_subjects=fit_subjects,
        row_count=len(raw),
        sha256=_json_sha256(payload),
    )


def fit_paired_dataset_transforms(
    subjects: Sequence[NativeMontageSubject],
    included_keys: set[SubjectKey],
    *,
    fit_rows: str,
) -> dict[str, PairedDatasetTransform]:
    """Fit both views using exactly ``included_keys`` and no other subject."""

    records = validate_primary_pretraining_corpus(subjects)
    known = {record.key for record in records}
    if not included_keys or not included_keys <= known:
        raise ValueError("included_keys must be a nonempty subset of the corpus")
    grouped: dict[str, list[NativeMontageSubject]] = defaultdict(list)
    for record in records:
        if record.key in included_keys:
            grouped[record.key[0]].append(record)
    all_datasets = {record.key[0] for record in records}
    if set(grouped) != all_datasets:
        raise ValueError("every corpus dataset must contribute scaler-fit subjects")
    return {
        dataset: _fit_one_transform(dataset, selected, fit_rows=fit_rows)
        for dataset, selected in sorted(grouped.items())
    }


def _apply_transforms(
    subjects: Sequence[NativeMontageSubject],
    transforms: Mapping[str, PairedDatasetTransform],
) -> tuple[PairedMontageSubject, ...]:
    paired: list[PairedMontageSubject] = []
    for record in subjects:
        dataset = record.key[0]
        if dataset not in transforms:
            raise RuntimeError(f"no paired transform for dataset {dataset}")
        transform = transforms[dataset]
        if tuple(record.channel_names) != transform.native_scaler.channel_names:
            raise RuntimeError(f"native channel order changed for dataset {dataset}")
        native, canonical = transform.transform(record.x)
        paired.append(
            PairedMontageSubject(
                dataset=dataset,
                subject=record.subject,
                native_x=native,
                canonical_x=canonical,
                y=np.asarray(record.y, dtype=np.int64).copy(),
                positions=np.asarray(record.positions, dtype=np.float32).copy(),
                native_channel_names=tuple(record.channel_names),
            )
        )
    return tuple(paired)


def _make_pools(
    subjects: Sequence[PairedMontageSubject],
    included_keys: set[SubjectKey],
) -> dict[str, _PairedDatasetPool]:
    grouped: dict[str, list[PairedMontageSubject]] = defaultdict(list)
    for record in subjects:
        if record.key in included_keys:
            grouped[record.key[0]].append(record)
    if not grouped:
        raise ValueError("paired subject selection produced no pools")
    pools: dict[str, _PairedDatasetPool] = {}
    for dataset, selected in sorted(grouped.items()):
        ordered = sorted(selected, key=lambda record: record.key)
        native_shape = tuple(ordered[0].native_x.shape[1:])
        canonical_shape = tuple(ordered[0].canonical_x.shape[1:])
        positions = np.asarray(ordered[0].positions, dtype=np.float32)
        for record in ordered[1:]:
            if (
                tuple(record.native_x.shape[1:]) != native_shape
                or tuple(record.canonical_x.shape[1:]) != canonical_shape
                or not np.array_equal(record.positions, positions)
            ):
                raise ValueError(f"paired dataset {dataset} does not share one grid")
        native = torch.as_tensor(
            np.concatenate([record.native_x for record in ordered]),
            dtype=torch.float32,
        )
        canonical = torch.as_tensor(
            np.concatenate([record.canonical_x for record in ordered]),
            dtype=torch.float32,
        )
        labels = torch.as_tensor(
            np.concatenate([record.y for record in ordered]), dtype=torch.long
        )
        class_rows = tuple(
            torch.nonzero(labels == label, as_tuple=False).flatten()
            for label in (0, 1)
        )
        if any(len(rows) == 0 for rows in class_rows):
            raise RuntimeError(f"paired dataset {dataset} lost a class")
        pools[dataset] = _PairedDatasetPool(
            dataset=dataset,
            native_x=native,
            canonical_x=canonical,
            y=labels,
            positions=torch.as_tensor(positions, dtype=torch.float32),
            class_rows=(class_rows[0], class_rows[1]),
        )
    return pools


def paired_augment(
    native_x: Tensor,
    canonical_x: Tensor,
    labels: Tensor,
    config: NativePretrainConfig | TrainConfig,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor, PairedAugmentationTrace]:
    """Apply shared segment donors/shifts and deterministic sensor noise."""

    if native_x.ndim != 3 or canonical_x.ndim != 3:
        raise ValueError("paired augmentation inputs must be three-dimensional")
    if (
        native_x.shape[0] != canonical_x.shape[0]
        or native_x.shape[2] != canonical_x.shape[2]
        or labels.shape != (native_x.shape[0],)
    ):
        raise ValueError("paired augmentation inputs are not row/time aligned")
    if native_x.device.type != "cpu" or canonical_x.device.type != "cpu":
        raise ValueError("paired augmentation must run on CPU for device-independent RNG")
    native = native_x.clone()
    canonical = canonical_x.clone()
    sources: list[tuple[int, int, tuple[int, ...], tuple[int, ...]]] = []
    segment_applied = False
    if config.segment_probability > 0.0:
        segment_applied = bool(
            torch.rand((), generator=generator) < config.segment_probability
        )
    if segment_applied and config.segment_count > 1:
        boundaries = torch.linspace(
            0, native.shape[-1], config.segment_count + 1
        ).long()
        original_native = native.clone()
        original_canonical = canonical.clone()
        for label_tensor in torch.unique(labels, sorted=True):
            label = int(label_tensor)
            rows = torch.nonzero(labels == label_tensor, as_tuple=False).flatten()
            if len(rows) < 2:
                continue
            for segment in range(config.segment_count):
                sampled = rows[
                    torch.randint(len(rows), (len(rows),), generator=generator)
                ]
                start = int(boundaries[segment])
                stop = int(boundaries[segment + 1])
                native[rows, :, start:stop] = original_native[
                    sampled, :, start:stop
                ]
                canonical[rows, :, start:stop] = original_canonical[
                    sampled, :, start:stop
                ]
                sources.append(
                    (
                        label,
                        segment,
                        tuple(int(value) for value in rows.tolist()),
                        tuple(int(value) for value in sampled.tolist()),
                    )
                )

    if config.time_shift_samples > 0:
        shifts_tensor = torch.randint(
            -config.time_shift_samples,
            config.time_shift_samples + 1,
            (len(native),),
            generator=generator,
        )
    else:
        shifts_tensor = torch.zeros(len(native), dtype=torch.long)
    shifted_native = torch.empty_like(native)
    shifted_canonical = torch.empty_like(canonical)
    for row, shift_value in enumerate(shifts_tensor):
        shift = int(shift_value)
        for source, destination in (
            (native, shifted_native),
            (canonical, shifted_canonical),
        ):
            if shift == 0:
                destination[row] = source[row]
            elif shift > 0:
                destination[row, :, :shift] = 0.0
                destination[row, :, shift:] = source[row, :, :-shift]
            else:
                destination[row, :, shift:] = 0.0
                destination[row, :, :shift] = source[row, :, -shift:]
    native = shifted_native
    canonical = shifted_canonical

    if config.noise_std > 0.0:
        native_noise = torch.randn(native.shape, generator=generator)
        if canonical.shape == native.shape:
            canonical_noise = native_noise
        else:
            canonical_noise = torch.randn(canonical.shape, generator=generator)
        native = native + config.noise_std * native_noise
        canonical = canonical + config.noise_std * canonical_noise
    trace = PairedAugmentationTrace(
        segment_applied=segment_applied,
        segment_sources=tuple(sources),
        shifts=tuple(int(value) for value in shifts_tensor.tolist()),
    )
    return native, canonical, trace


def _forward_views(
    model: nn.Module,
    native_x: Tensor,
    canonical_x: Tensor,
    positions: Tensor,
) -> CardinalSplineDualViewOutput:
    if not bool(getattr(model, "uses_positions", False)) or not bool(
        getattr(model, "uses_canonical_view", False)
    ):
        raise TypeError("dual-view training requires position and canonical-view support")
    method = getattr(model, "forward_views", None)
    if not callable(method):
        raise TypeError("dual-view model must implement forward_views")
    output = method(native_x, canonical_x, positions)
    if not isinstance(output, CardinalSplineDualViewOutput):
        raise TypeError("forward_views returned an unsupported output type")
    expected = (len(native_x), 2)
    for name in ("fused_logits", "direct_logits", "canonical_logits"):
        if getattr(output, name).shape != expected:
            raise RuntimeError(f"{name} must have shape {expected}")
    if output.canonical_weight.shape != (len(native_x), 1):
        raise RuntimeError("canonical_weight must have shape (batch, 1)")
    return output


def _frozen_loss(
    output: CardinalSplineDualViewOutput,
    labels: Tensor,
) -> CardinalSplineDualViewLoss:
    return cardinal_spline_dual_view_loss(
        output,
        labels,
        auxiliary_view_weight=FROZEN_AUXILIARY_VIEW_WEIGHT,
        preference_weight=FROZEN_PREFERENCE_WEIGHT,
        preference_temperature=FROZEN_PREFERENCE_TEMPERATURE,
        label_smoothing=FROZEN_LABEL_SMOOTHING,
    )


def _validate_common_config(config: NativePretrainConfig | TrainConfig) -> None:
    epochs = (
        config.macro_epochs
        if isinstance(config, NativePretrainConfig)
        else config.epochs
    )
    batch_size = (
        config.batch_size_per_dataset
        if isinstance(config, NativePretrainConfig)
        else config.batch_size
    )
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("training epochs and batch size must be positive")
    if isinstance(config, NativePretrainConfig):
        if batch_size % 2:
            raise ValueError("pretraining batch size must be even")
        if (
            config.steps_per_macro_epoch is not None
            and config.steps_per_macro_epoch <= 0
        ):
            raise ValueError("steps_per_macro_epoch must be positive")
        if config.validation_batch_size <= 0:
            raise ValueError("validation_batch_size must be positive")
    if config.learning_rate <= 0.0 or config.weight_decay < 0.0:
        raise ValueError("optimizer configuration is invalid")
    if config.patience <= 0 or config.min_delta < 0.0:
        raise ValueError("early-stopping configuration is invalid")
    if not math.isclose(config.label_smoothing, FROZEN_LABEL_SMOOTHING):
        raise ValueError("CSDV-FBC label smoothing is frozen at 0.05")
    if not 0.0 <= config.segment_probability <= 1.0:
        raise ValueError("segment_probability must lie in [0, 1]")
    if config.segment_count <= 0 or config.time_shift_samples < 0:
        raise ValueError("temporal augmentation configuration is invalid")
    if config.noise_std < 0.0 or config.gradient_clip <= 0.0:
        raise ValueError("noise/gradient configuration is invalid")


def _balanced_batch(
    pool: _PairedDatasetPool,
    *,
    batch_size: int,
    config: NativePretrainConfig,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if batch_size <= 0 or batch_size % 2:
        raise ValueError("pretraining batch size must be a positive even integer")
    half = batch_size // 2
    selected_parts = []
    for class_rows in pool.class_rows:
        sampled = torch.randint(len(class_rows), (half,), generator=generator)
        selected_parts.append(class_rows[sampled])
    selected = torch.cat(selected_parts)
    selected = selected[torch.randperm(len(selected), generator=generator)]
    native, canonical, _ = paired_augment(
        pool.native_x[selected],
        pool.canonical_x[selected],
        pool.y[selected],
        config,
        generator,
    )
    return (
        native.to(device),
        canonical.to(device),
        pool.y[selected].to(device),
        pool.positions.to(device),
    )


def _equal_dataset_validation_ce(
    model: nn.Module,
    pools: Mapping[str, _PairedDatasetPool],
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[float, dict[str, float]]:
    model.eval()
    per_dataset: dict[str, float] = {}
    with torch.inference_mode():
        for dataset, pool in sorted(pools.items()):
            total = 0.0
            positions = pool.positions.to(device)
            for start in range(0, len(pool.y), batch_size):
                stop = min(start + batch_size, len(pool.y))
                output = _forward_views(
                    model,
                    pool.native_x[start:stop].to(device),
                    pool.canonical_x[start:stop].to(device),
                    positions,
                )
                total += float(
                    nn.functional.cross_entropy(
                        output.fused_logits,
                        pool.y[start:stop].to(device),
                        reduction="sum",
                    )
                )
            per_dataset[dataset] = total / len(pool.y)
    return float(np.mean(list(per_dataset.values()))), per_dataset


def _pretrain_epoch(
    model: nn.Module,
    pools: Mapping[str, _PairedDatasetPool],
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
            native, canonical, labels, positions = _balanced_batch(
                pool,
                batch_size=config.batch_size_per_dataset,
                config=config,
                generator=generator,
                device=device,
            )
            loss = _frozen_loss(
                _forward_views(model, native, canonical, positions), labels
            ).total
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
    train_pools: Mapping[str, _PairedDatasetPool],
    validation_pools: Mapping[str, _PairedDatasetPool],
    *,
    config: NativePretrainConfig,
    steps: int,
) -> tuple[dict[str, Tensor], int, float, list[dict[str, object]]]:
    configure_determinism(config.seed)
    device = torch.device(config.device)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
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
        train_loss, dataset_train = _pretrain_epoch(
            model,
            train_pools,
            optimizer=optimizer,
            config=config,
            generator=generator,
            device=device,
            steps=steps,
        )
        scheduler.step()
        validation_loss, dataset_validation = _equal_dataset_validation_ce(
            model,
            validation_pools,
            device=device,
            batch_size=config.validation_batch_size,
        )
        history.append(
            {
                "epoch": epoch,
                "training_equal_dataset_loss": train_loss,
                "training_dataset_loss": dataset_train,
                "validation_equal_dataset_fused_ce": validation_loss,
                "validation_dataset_fused_ce": dataset_validation,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if validation_loss < best_loss - config.min_delta:
            best_state = _cpu_state_dict(model.state_dict())
            best_epoch = epoch
            best_loss = validation_loss
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("dual-view pretraining selected no finite checkpoint")
    return best_state, best_epoch, best_loss, history


def _pretrain_refit(
    model: nn.Module,
    pools: Mapping[str, _PairedDatasetPool],
    *,
    epochs: int,
    config: NativePretrainConfig,
    steps: int,
) -> list[dict[str, object]]:
    configure_determinism(config.seed)
    device = torch.device(config.device)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.macro_epochs
    )
    generator = torch.Generator(device="cpu").manual_seed(config.seed + 104729)
    history: list[dict[str, object]] = []
    for epoch in range(epochs):
        train_loss, dataset_train = _pretrain_epoch(
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
                "training_dataset_loss": dataset_train,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
    return history


def fit_dual_view_native_pretraining(
    model_factory: ModelFactory,
    subjects: Sequence[NativeMontageSubject],
    *,
    config: NativePretrainConfig = NativePretrainConfig(),
) -> DualViewPretrainingResult:
    """Select/refit CSDV-FBC with existing partition and dataset weighting."""

    _validate_common_config(config)
    records = validate_primary_pretraining_corpus(subjects)
    partition = hashed_subject_partition(
        records,
        validation_fraction=config.validation_fraction,
        salt=config.partition_salt,
    )
    selection_transforms = fit_paired_dataset_transforms(
        records,
        set(partition.train),
        fit_rows="pretraining selection subjects: training role only",
    )
    selection_records = _apply_transforms(records, selection_transforms)
    train_pools = _make_pools(selection_records, set(partition.train))
    validation_pools = _make_pools(selection_records, set(partition.validation))
    if train_pools.keys() != validation_pools.keys():
        raise RuntimeError("every dataset must appear in both pretraining roles")

    all_keys = {record.key for record in records}
    refit_transforms = fit_paired_dataset_transforms(
        records,
        all_keys,
        fit_rows="pretraining final refit subjects: all authorized source roles",
    )
    refit_records = _apply_transforms(records, refit_transforms)
    all_pools = _make_pools(refit_records, all_keys)

    configure_determinism(config.seed)
    model = model_factory()
    reset_batch_norm_running_stats(model)
    initial_state = _cpu_state_dict(model.state_dict())
    initial_hash = state_dict_sha256(initial_state)
    selection_steps = (
        config.steps_per_macro_epoch
        if config.steps_per_macro_epoch is not None
        else max(
            math.ceil(len(pool.y) / config.batch_size_per_dataset)
            for pool in train_pools.values()
        )
    )
    selected_state, best_epoch, best_loss, selection_history = _selection_fit(
        model,
        train_pools,
        validation_pools,
        config=config,
        steps=selection_steps,
    )
    selection_hash = state_dict_sha256(selected_state)

    # Discard selected weights and regenerate from the exact seeded state.
    model.load_state_dict(initial_state, strict=True)
    if state_dict_sha256(model.state_dict()) != initial_hash:
        raise RuntimeError("dual-view refit did not restore the exact initial state")
    refit_steps = (
        config.steps_per_macro_epoch
        if config.steps_per_macro_epoch is not None
        else max(
            math.ceil(len(pool.y) / config.batch_size_per_dataset)
            for pool in all_pools.values()
        )
    )
    refit_history = _pretrain_refit(
        model,
        all_pools,
        epochs=best_epoch + 1,
        config=config,
        steps=refit_steps,
    )
    checkpoint = _cpu_state_dict(model.state_dict())
    return DualViewPretrainingResult(
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
        selection_transforms=copy.deepcopy(selection_transforms),
        refit_transforms=copy.deepcopy(refit_transforms),
    )


def _fit_target_transform(
    raw_fit: FloatArray,
    positions: FloatArray,
    channel_names: tuple[str, ...],
    *,
    fit_rows: str,
) -> PairedDatasetTransform:
    pseudo = NativeMontageSubject(
        dataset="target",
        subject="fit_rows",
        x=np.asarray(raw_fit, dtype=np.float32),
        y=np.resize(np.asarray((0, 1), dtype=np.int64), len(raw_fit)),
        positions=np.asarray(positions, dtype=np.float32),
        channel_names=channel_names,
    )
    return _fit_one_transform("target", (pseudo,), fit_rows=fit_rows)


def _target_validation_ce(
    model: nn.Module,
    native: Tensor,
    canonical: Tensor,
    labels: Tensor,
    positions: Tensor,
    *,
    batch_size: int,
) -> float:
    model.eval()
    total = 0.0
    with torch.inference_mode():
        for start in range(0, len(labels), batch_size):
            stop = min(start + batch_size, len(labels))
            output = _forward_views(
                model,
                native[start:stop],
                canonical[start:stop],
                positions,
            )
            total += float(
                nn.functional.cross_entropy(
                    output.fused_logits,
                    labels[start:stop],
                    reduction="sum",
                )
            )
    return total / len(labels)


def _target_epoch(
    model: nn.Module,
    native: Tensor,
    canonical: Tensor,
    labels: Tensor,
    positions: Tensor,
    *,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    generator: torch.Generator,
) -> float:
    model.train()
    order = torch.randperm(len(labels), generator=generator)
    total = 0.0
    seen = 0
    for start in range(0, len(order), config.batch_size):
        rows = order[start : start + config.batch_size]
        batch_native, batch_canonical, _ = paired_augment(
            native[rows],
            canonical[rows],
            labels[rows],
            config,
            generator,
        )
        batch_native = batch_native.to(positions.device)
        batch_canonical = batch_canonical.to(positions.device)
        batch_labels = labels[rows].to(positions.device)
        optimizer.zero_grad(set_to_none=True)
        loss = _frozen_loss(
            _forward_views(
                model, batch_native, batch_canonical, positions
            ),
            batch_labels,
        ).total
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()
        total += float(loss.detach()) * len(rows)
        seen += len(rows)
    return total / seen


def _target_selection(
    model: nn.Module,
    train_pair: tuple[FloatArray, FloatArray],
    y_train: IntArray,
    validation_pair: tuple[FloatArray, FloatArray],
    y_validation: IntArray,
    positions_array: FloatArray,
    *,
    config: TrainConfig,
) -> tuple[dict[str, Tensor], int, list[dict[str, float]]]:
    device = torch.device(config.device)
    native_train = torch.as_tensor(train_pair[0], dtype=torch.float32)
    canonical_train = torch.as_tensor(train_pair[1], dtype=torch.float32)
    train_labels = torch.as_tensor(y_train, dtype=torch.long)
    native_validation = torch.as_tensor(
        validation_pair[0], dtype=torch.float32, device=device
    )
    canonical_validation = torch.as_tensor(
        validation_pair[1], dtype=torch.float32, device=device
    )
    validation_labels = torch.as_tensor(y_validation, dtype=torch.long, device=device)
    positions = torch.as_tensor(positions_array, dtype=torch.float32, device=device)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )
    generator = torch.Generator(device="cpu").manual_seed(config.seed + 7919)
    best_state: dict[str, Tensor] | None = None
    best_epoch = -1
    best_loss = float("inf")
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(config.epochs):
        train_loss = _target_epoch(
            model,
            native_train,
            canonical_train,
            train_labels,
            positions,
            optimizer=optimizer,
            config=config,
            generator=generator,
        )
        scheduler.step()
        validation_loss = _target_validation_ce(
            model,
            native_validation,
            canonical_validation,
            validation_labels,
            positions,
            batch_size=config.batch_size,
        )
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "validation_fused_ce": validation_loss,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if validation_loss < best_loss - config.min_delta:
            best_state = _cpu_state_dict(model.state_dict())
            best_epoch = epoch
            best_loss = validation_loss
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("target dual-view selection produced no checkpoint")
    return best_state, best_epoch, history


def _target_refit(
    model: nn.Module,
    paired: tuple[FloatArray, FloatArray],
    labels_array: IntArray,
    positions_array: FloatArray,
    *,
    epochs: int,
    config: TrainConfig,
) -> list[dict[str, float]]:
    device = torch.device(config.device)
    native = torch.as_tensor(paired[0], dtype=torch.float32)
    canonical = torch.as_tensor(paired[1], dtype=torch.float32)
    labels = torch.as_tensor(labels_array, dtype=torch.long)
    positions = torch.as_tensor(positions_array, dtype=torch.float32, device=device)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )
    generator = torch.Generator(device="cpu").manual_seed(config.seed + 7919)
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        loss = _target_epoch(
            model,
            native,
            canonical,
            labels,
            positions,
            optimizer=optimizer,
            config=config,
            generator=generator,
        )
        scheduler.step()
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": loss,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
    return history


def fit_dual_view_target(
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
) -> DualViewTargetResult:
    """Fit target selection/refit without accepting or observing test input."""

    _validate_common_config(config)
    x_train = np.asarray(x_train, dtype=np.float32)
    x_validation = np.asarray(x_validation, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)
    y_validation = np.asarray(y_validation, dtype=np.int64)
    positions = np.asarray(positions, dtype=np.float32)
    names = tuple(channel_names)
    if x_train.ndim != 3 or x_validation.shape[1:] != x_train.shape[1:]:
        raise ValueError("target train/validation arrays have incompatible shapes")
    if y_train.shape != (len(x_train),) or y_validation.shape != (len(x_validation),):
        raise ValueError("target labels do not match target rows")
    if set(y_train.tolist()) != {0, 1} or set(y_validation.tolist()) != {0, 1}:
        raise ValueError("both target source roles must contain both classes")
    if positions.shape != (x_train.shape[1], 3) or len(names) != x_train.shape[1]:
        raise ValueError("target native montage metadata are incompatible")
    if (
        len(names) != len(set(names))
        or any(not name for name in names)
        or not np.isfinite(x_train).all()
        or not np.isfinite(x_validation).all()
        or not np.isfinite(positions).all()
    ):
        raise ValueError("target source arrays or metadata are invalid")

    selection_transform = _fit_target_transform(
        x_train,
        positions,
        names,
        fit_rows="target selection: training rows only",
    )
    train_pair = selection_transform.transform(x_train)
    validation_pair = selection_transform.transform(x_validation)
    checkpoint = _cpu_state_dict(pretrained_state)
    checkpoint_hash = state_dict_sha256(checkpoint)

    configure_determinism(config.seed)
    selection_model = model_factory()
    selection_model.load_state_dict(checkpoint, strict=True)
    selection_reset = reset_batch_norm_running_stats(selection_model)
    selection_start = model_state_sha256(selection_model)
    selected_state, best_epoch, selection_history = _target_selection(
        selection_model,
        train_pair,
        y_train,
        validation_pair,
        y_validation,
        positions,
        config=config,
    )

    source_raw = np.concatenate((x_train, x_validation), axis=0)
    source_labels = np.concatenate((y_train, y_validation), axis=0)
    refit_transform = _fit_target_transform(
        source_raw,
        positions,
        names,
        fit_rows="target final refit: training plus validation rows",
    )
    source_pair = refit_transform.transform(source_raw)
    configure_determinism(config.seed)
    refit_model = model_factory()
    if refit_model is selection_model:
        raise RuntimeError("target model_factory did not return a fresh model")
    refit_model.load_state_dict(checkpoint, strict=True)
    refit_reset = reset_batch_norm_running_stats(refit_model)
    refit_start = model_state_sha256(refit_model)
    if selection_reset != refit_reset or selection_start != refit_start:
        raise RuntimeError("target selection/refit starts are not identical")
    refit_history = _target_refit(
        refit_model,
        source_pair,
        source_labels,
        positions,
        epochs=best_epoch + 1,
        config=config,
    )
    return DualViewTargetResult(
        model=refit_model,
        positions=positions.copy(),
        native_channel_names=names,
        n_times=x_train.shape[2],
        refit_transform=copy.deepcopy(refit_transform),
        best_epoch=best_epoch,
        pretrained_checkpoint_sha256=checkpoint_hash,
        selection_start_sha256=selection_start,
        refit_start_sha256=refit_start,
        selection_state_sha256=state_dict_sha256(selected_state),
        refit_state_sha256=model_state_sha256(refit_model),
        reset_batch_norm_modules=selection_reset,
        selection_transform=copy.deepcopy(selection_transform),
        selection_history=copy.deepcopy(selection_history),
        refit_history=copy.deepcopy(refit_history),
        device=config.device,
    )
