"""Leakage-safe reference-recipe training for the strongest neural baselines.

This module is deliberately separate from :mod:`ieee_mi.training`.  The latter
is the common-recipe comparison, whereas this file reproduces the material
optimization choices from the official TCFormer and FBCNet releases.  Neither
entry point accepts test arrays: test prediction remains a later, read-only
benchmark step.

The implementations are recipe-faithful but not claims of bit-for-bit
reproduction.  They train the pinned PyTorch/Braindecode model adapters on this
study's harmonized inputs rather than the authors' original data loaders.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from scipy import signal
from torch import Tensor, nn

from .training import configure_determinism


REFERENCE_TRAINING_SCHEMA = "ieee-mi-reference-recipe-v1"
TCFORMER_SOURCE_COMMIT = "74c89b7ab8c64e4eb51e0f748dd87dd4c94e68c5"
FBCNET_SOURCE_COMMIT = "de1bbdd8a54cb1e466830e3d47070e0e56761a37"
BRAIDECODE_VERSION = "1.6.1"
FBCNET_BANDS: tuple[tuple[float, float], ...] = tuple(
    (float(low), float(low + 4)) for low in range(4, 40, 4)
)


@dataclass(frozen=True)
class TCFormerReferenceConfig:
    """Released within-subject optimization recipe for TCFormer."""

    epochs: int = 1000
    batch_size: int = 48
    learning_rate: float = 9e-4
    beta_1: float = 0.5
    beta_2: float = 0.999
    weight_decay: float = 1e-3
    warmup_epochs: int = 20
    segment_count: int = 8
    seed: int = 7
    device: str = "cuda"
    horizon_provenance: str = "explicit_configuration"

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate < 0.0 or self.weight_decay < 0.0:
            raise ValueError("learning rate and weight decay must be nonnegative")
        if not 0 < self.warmup_epochs < self.epochs:
            raise ValueError("warmup_epochs must be in (0, epochs)")
        if self.segment_count <= 1:
            raise ValueError("segment_count must exceed one")
        if not self.horizon_provenance:
            raise ValueError("horizon_provenance must be nonempty")


@dataclass(frozen=True)
class FBCNetReferenceConfig:
    """Original FBCNet optimizer and two-stage stopping recipe."""

    stage1_max_epochs: int = 1500
    stage1_patience: int = 200
    stage2_max_epochs: int = 600
    batch_size: int = 16
    learning_rate: float = 1e-3
    beta_1: float = 0.9
    beta_2: float = 0.999
    weight_decay: float = 0.0
    patience_min_relative_change: float = 1e-6
    seed: int = 7
    device: str = "cuda"

    def __post_init__(self) -> None:
        if self.stage1_max_epochs <= 0 or self.stage2_max_epochs <= 0:
            raise ValueError("both stage epoch caps must be positive")
        if self.stage1_patience <= 0 or self.batch_size <= 0:
            raise ValueError("patience and batch size must be positive")
        if self.learning_rate < 0.0 or self.weight_decay < 0.0:
            raise ValueError("learning rate and weight decay must be nonnegative")
        if not 0.0 <= self.patience_min_relative_change < 1.0:
            raise ValueError("patience_min_relative_change must be in [0, 1)")


@dataclass(frozen=True)
class Cheby2BandDesign:
    """One original FBCNet Chebyshev-II band-pass design."""

    passband_hz: tuple[float, float]
    stopband_hz: tuple[float, float]
    order: int
    critical_frequencies: NDArray[np.float64]
    numerator: NDArray[np.float64]
    denominator: NDArray[np.float64]


@dataclass
class RelativeNoDecrease:
    """Released relative-change patience counter, including its zero behavior.

    A tied zero is considered a sufficient relative decrease because
    ``0 <= (1 - min_relative_change) * 0``.  It therefore resets patience,
    matching the FBCNet release instead of eventually stopping on a perfect
    validation inaccuracy plateau.
    """

    patience: int
    min_relative_change: float = 1e-6
    minimum: float = float("inf")
    stale_epochs: int = 0

    def __post_init__(self) -> None:
        if self.patience <= 0:
            raise ValueError("patience must be positive")
        if not 0.0 <= self.min_relative_change < 1.0:
            raise ValueError("min_relative_change must be in [0, 1)")

    def update(self, value: float) -> bool:
        """Consume one metric and return whether patience is exhausted."""

        value = float(value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("patience metric must be finite and nonnegative")
        if value <= (1.0 - self.min_relative_change) * self.minimum:
            self.minimum = value
            self.stale_epochs = 0
        else:
            self.stale_epochs += 1
        return self.stale_epochs >= self.patience


class PrefilteredFBCNetAdapter(nn.Module):
    """Run a Braindecode FBCNet after an external causal Cheby-II bank.

    ``apply_fbcnet_cheby2_filterbank`` emits ``(batch, band, channel, time)``.
    This adapter bypasses only the base model's built-in spectral layer and
    keeps its spatial, temporal, and constrained classification layers.
    """

    uses_positions = False

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        required = (
            "spatial_conv",
            "padding_layer",
            "temporal_layer",
            "flatten_layer",
            "final_layer",
            "stride_factor",
            "n_times_padded",
            "n_bands",
        )
        missing = [name for name in required if not hasattr(base, name)]
        if missing:
            raise TypeError(f"FBCNet adapter is missing attributes {missing}")
        self.base = base

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1] != int(self.base.n_bands):
            raise ValueError(
                "prefiltered FBCNet input must have shape "
                f"(batch, {int(self.base.n_bands)}, channel, time)"
            )
        values = self.base.spatial_conv(x)
        batch, channels, _, _ = values.shape
        values = self.base.padding_layer(values)
        values = values.reshape(
            batch,
            channels,
            int(self.base.stride_factor),
            int(self.base.n_times_padded) // int(self.base.stride_factor),
        )
        values = self.base.temporal_layer(values)
        return self.base.final_layer(self.base.flatten_layer(values))


def _update_state_hash(digest: Any, value: Any) -> None:
    """Add one nested state value to a deterministic, device-agnostic hash."""

    if isinstance(value, Tensor):
        tensor = value.detach().cpu()
        if tensor.layout != torch.strided:
            tensor = tensor.to_dense()
        if tensor.is_quantized:
            tensor = tensor.int_repr()
        tensor = tensor.contiguous().reshape(-1)
        digest.update(b"tensor\0")
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(b"\0")
        digest.update(repr(tuple(value.shape)).encode("ascii"))
        digest.update(b"\0")
        try:
            payload = tensor.numpy().tobytes(order="C")
        except TypeError:  # NumPy has no native bfloat16 representation.
            payload = tensor.view(torch.uint8).numpy().tobytes(order="C")
        digest.update(payload)
        return
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(b"ndarray\0")
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
        return
    if isinstance(value, Mapping):
        digest.update(b"mapping\0")
        for key in sorted(
            value,
            key=lambda item: (type(item).__qualname__, repr(item)),
        ):
            _update_state_hash(digest, key)
            _update_state_hash(digest, value[key])
        digest.update(b"end-mapping\0")
        return
    if isinstance(value, (list, tuple)):
        digest.update(type(value).__name__.encode("ascii") + b"\0")
        for item in value:
            _update_state_hash(digest, item)
        digest.update(b"end-sequence\0")
        return
    digest.update(type(value).__qualname__.encode("utf-8") + b"\0")
    digest.update(repr(value).encode("utf-8") + b"\0")


def _state_sha256(state: Mapping[Any, Any]) -> str:
    digest = hashlib.sha256()
    _update_state_hash(digest, state)
    return digest.hexdigest()


def model_state_sha256(model: nn.Module) -> str:
    """Return a deterministic SHA-256 of model parameters and buffers."""

    return _state_sha256(model.state_dict())


def optimizer_state_sha256(
    optimizer_or_state: torch.optim.Optimizer | Mapping[Any, Any],
) -> str:
    """Return a deterministic SHA-256 of an optimizer or copied state dict."""

    if isinstance(optimizer_or_state, torch.optim.Optimizer):
        state = optimizer_or_state.state_dict()
    else:
        state = optimizer_or_state
    return _state_sha256(state)


def _seeded_model_from_factory(
    model_factory: Callable[[], nn.Module] | nn.Module,
    *,
    seed: int,
) -> nn.Module:
    """Seed before initialization and reject already-constructed modules."""

    if isinstance(model_factory, nn.Module) or not callable(model_factory):
        raise TypeError(
            "pass a zero-argument model factory, not a preconstructed model; "
            "the reference seed must be installed before parameter initialization"
        )
    configure_determinism(seed)
    model = model_factory()
    if not isinstance(model, nn.Module):
        raise TypeError("model factory must return torch.nn.Module")
    return model


def _verify_braindecode_version() -> None:
    try:
        installed = importlib.metadata.version("braindecode")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(
            f"FBCNet reference training requires braindecode=={BRAIDECODE_VERSION}"
        ) from error
    if installed != BRAIDECODE_VERSION:
        raise RuntimeError(
            "FBCNet reference training requires "
            f"braindecode=={BRAIDECODE_VERSION}, found {installed}"
        )


def tcformer_reference_config(
    dataset_key: str,
    **overrides: Any,
) -> TCFormerReferenceConfig:
    """Return a declared horizon and its provenance for a development dataset.

    BCI Competition IV-2a and IV-2b use their released 1000- and 500-epoch
    settings.  The local, Cho2017, and PhysioNet 1000-epoch settings are study
    adaptations derived from IV-2a; they were not released by TCFormer's
    authors.  Unknown keys are rejected so a cohort cannot silently inherit a
    recipe.
    """

    horizons = {
        "bnci2014_001": (1000, "released_bcic_iv_2a"),
        "bnci2014_004": (500, "released_bcic_iv_2b"),
        "local_exp4": (1000, "adapted_from_released_bcic_iv_2a_not_released"),
        "cho2017": (1000, "adapted_from_released_bcic_iv_2a_not_released"),
        "physionet_mi": (1000, "adapted_from_released_bcic_iv_2a_not_released"),
    }
    if dataset_key not in horizons:
        raise ValueError(f"no declared TCFormer reference horizon for {dataset_key!r}")
    if "epochs" in overrides or "horizon_provenance" in overrides:
        raise ValueError(
            "tcformer_reference_config owns epochs and horizon_provenance; "
            "construct TCFormerReferenceConfig directly for an explicit horizon"
        )
    epochs, provenance = horizons[dataset_key]
    return TCFormerReferenceConfig(
        epochs=epochs,
        horizon_provenance=provenance,
        **overrides,
    )


def tcformer_lr_multiplier(
    epoch: int, *, total_epochs: int, warmup_epochs: int = 20
) -> float:
    """Official linear-warmup/cosine multiplier, indexed from epoch zero."""

    if total_epochs <= 0:
        raise ValueError("total_epochs must be positive")
    if not 0 < warmup_epochs < total_epochs:
        raise ValueError("warmup_epochs must be in (0, total_epochs)")
    if not 0 <= epoch <= total_epochs:
        raise ValueError("epoch must be in [0, total_epochs]")
    if epoch < warmup_epochs:
        return float(epoch) / float(warmup_epochs)
    progress = float(epoch - warmup_epochs) / float(total_epochs - warmup_epochs)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def tcformer_segment_reconstruct(
    x: Tensor,
    y: Tensor,
    *,
    segments: int = 8,
    generator: torch.Generator,
) -> Tensor:
    """Construct exactly one same-class S&R trial for every real trial.

    Each output segment independently selects a donor from the current
    mini-batch with the same label.  Unlike the common-recipe augmentation,
    this is applied to every batch and returns only the synthetic half.
    """

    if x.ndim != 3:
        raise ValueError("TCFormer S&R expects (batch, channel, time)")
    if y.ndim != 1 or len(y) != len(x):
        raise ValueError("labels must be a vector aligned with x")
    if segments <= 1 or x.shape[-1] % segments:
        raise ValueError("time samples must divide exactly into the requested segments")
    reconstructed = torch.empty_like(x)
    width = x.shape[-1] // segments
    for label in torch.unique(y):
        rows = torch.nonzero(y == label, as_tuple=False).flatten()
        if len(rows) == 0:  # pragma: no cover - impossible after torch.unique
            continue
        donors = rows[
            torch.randint(
                len(rows),
                (len(rows), segments),
                device=x.device,
                generator=generator,
            )
        ]
        for segment in range(segments):
            start = segment * width
            stop = start + width
            reconstructed[rows, :, start:stop] = x[
                donors[:, segment], :, start:stop
            ]
    return reconstructed


def tcformer_augmented_batch(
    x: Tensor,
    y: Tensor,
    *,
    segments: int = 8,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    """Concatenate and shuffle real plus one-for-one S&R trials (48 -> 96)."""

    synthetic = tcformer_segment_reconstruct(
        x, y, segments=segments, generator=generator
    )
    combined_x = torch.cat((x, synthetic), dim=0)
    combined_y = torch.cat((y, y), dim=0)
    order = torch.randperm(len(combined_x), device=x.device, generator=generator)
    return combined_x[order], combined_y[order]


def design_fbcnet_cheby2_filterbank(
    sfreq: float,
    *,
    bands: Sequence[tuple[float, float]] = FBCNET_BANDS,
    transition_hz: float = 2.0,
    passband_attenuation_db: float = 3.0,
    stopband_attenuation_db: float = 30.0,
) -> tuple[Cheby2BandDesign, ...]:
    """Design the original nine causal Chebyshev-II FBCNet filters."""

    nyquist = float(sfreq) / 2.0
    if nyquist <= 0.0 or transition_hz <= 0.0:
        raise ValueError("sfreq and transition_hz must be positive")
    designs: list[Cheby2BandDesign] = []
    for low, high in bands:
        low, high = float(low), float(high)
        stop = (low - transition_hz, high + transition_hz)
        if not 0.0 < stop[0] < low < high < stop[1] < nyquist:
            raise ValueError(
                f"band {(low, high)} with transition {transition_hz} is invalid "
                f"at sfreq={sfreq}"
            )
        wp = np.asarray((low / nyquist, high / nyquist), dtype=np.float64)
        ws = np.asarray((stop[0] / nyquist, stop[1] / nyquist), dtype=np.float64)
        order, _natural = signal.cheb2ord(
            wp, ws, passband_attenuation_db, stopband_attenuation_db
        )
        # Preserve the released FBCNet implementation literally: it uses
        # cheb2ord only to select N, then passes the declared stop-band vector
        # (fStop), not cheb2ord's returned natural frequencies, to cheby2.
        numerator, denominator = signal.cheby2(
            order, stopband_attenuation_db, ws, btype="bandpass"
        )
        designs.append(
            Cheby2BandDesign(
                passband_hz=(low, high),
                stopband_hz=stop,
                order=int(order),
                critical_frequencies=np.asarray(ws, dtype=np.float64),
                numerator=np.asarray(numerator, dtype=np.float64),
                denominator=np.asarray(denominator, dtype=np.float64),
            )
        )
    if not designs:
        raise ValueError("at least one filter band is required")
    return tuple(designs)


def fbcnet_filterbank_frequency_response(
    sfreq: float,
    *,
    designs: Sequence[Cheby2BandDesign] | None = None,
    frequencies: int = 4096,
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    """Return frequencies in Hz and the complex response of every band."""

    if frequencies <= 1:
        raise ValueError("frequencies must exceed one")
    selected = tuple(
        design_fbcnet_cheby2_filterbank(sfreq) if designs is None else designs
    )
    responses: list[NDArray[np.complex128]] = []
    frequency_axis: NDArray[np.float64] | None = None
    for design in selected:
        hz, response = signal.freqz(
            design.numerator,
            design.denominator,
            worN=frequencies,
            fs=float(sfreq),
        )
        frequency_axis = np.asarray(hz, dtype=np.float64)
        responses.append(np.asarray(response, dtype=np.complex128))
    if frequency_axis is None:
        raise ValueError("at least one filter design is required")
    return frequency_axis, np.stack(responses, axis=0)


def apply_fbcnet_cheby2_filterbank(
    x: NDArray[np.floating[Any]],
    sfreq: float,
    *,
    designs: Sequence[Cheby2BandDesign] | None = None,
    time_axis: int = -1,
    band_axis: int = 1,
) -> NDArray[np.float32]:
    """Apply the original one-pass causal ``scipy.signal.lfilter`` bank.

    For standard ``(trial, channel, time)`` input, the default output is
    ``(trial, band, channel, time)``, ready for ``PrefilteredFBCNetAdapter``.
    Filtering is performed independently on each supplied epoch, as in the
    released FBCNet transform.
    """

    values = np.asarray(x)
    if values.ndim < 2:
        raise ValueError("filter-bank input must have at least two dimensions")
    selected = tuple(
        design_fbcnet_cheby2_filterbank(sfreq) if designs is None else designs
    )
    if not selected:
        raise ValueError("at least one filter design is required")
    filtered = [
        signal.lfilter(
            design.numerator,
            design.denominator,
            values,
            axis=time_axis,
        )
        for design in selected
    ]
    return np.stack(filtered, axis=band_axis).astype(np.float32, copy=False)


def tcformer_reference_metadata(config: TCFormerReferenceConfig) -> dict[str, Any]:
    return {
        "schema": REFERENCE_TRAINING_SCHEMA,
        "model": "tcformer",
        "faithfulness": (
            "released optimization recipe on a study-adapted model/input protocol"
        ),
        "source": {
            "repository": "https://github.com/Altaheri/TCFormer",
            "commit": TCFORMER_SOURCE_COMMIT,
        },
        "optimizer": "torch.optim.Adam (coupled L2 weight decay)",
        "loss": "un-smoothed cross entropy",
        "scheduler": "epoch-indexed linear warmup then cosine decay",
        "augmentation": (
            "one 8-segment same-class S&R trial per real trial; combined batch shuffled"
        ),
        "checkpoint": "final fixed-horizon epoch; no early stopping",
        "gradient_clipping": None,
        "validation_during_fit": None,
        "test_access_during_fit": False,
        "adaptations": (
            "pure PyTorch loop instead of Lightning",
            "study-harmonized input window, sampling rate, montage, and scaling",
            (
                "study-controlled seed installed before model-factory initialization, "
                "plus a deterministic dedicated augmentation generator"
            ),
        ),
        "horizon_provenance": config.horizon_provenance,
        "config": asdict(config),
    }


def fbcnet_reference_metadata(config: FBCNetReferenceConfig) -> dict[str, Any]:
    return {
        "schema": REFERENCE_TRAINING_SCHEMA,
        "model": "fbcnet",
        "faithfulness": (
            "released optimizer, filter, and stopping recipe on a "
            "Braindecode/study-adapted architecture and input protocol"
        ),
        "source": {
            "repository": "https://github.com/ravikiran-mane/FBCNet",
            "commit": FBCNET_SOURCE_COMMIT,
            "training": "codes/centralRepo/baseModel.py",
            "filter": "codes/centralRepo/transforms.py",
        },
        "implementation": {
            "package": "braindecode",
            "version": BRAIDECODE_VERSION,
        },
        "optimizer": "torch.optim.Adam",
        "loss": "cross entropy on logits (equivalent to released NLL on log-softmax)",
        "scheduler": None,
        "augmentation": None,
        "checkpoint": "best stage-1 validation inaccuracy, then final stage-2 state",
        "gradient_clipping": None,
        "stage2_validation_role": (
            "the original validation subset is included in optimization and is used "
            "only for the released loss-threshold stopping rule, never for test "
            "prediction"
        ),
        "filterbank": {
            "bands_hz": FBCNET_BANDS,
            "transition_hz": 2.0,
            "passband_attenuation_db": 3.0,
            "stopband_attenuation_db": 30.0,
            "coefficient_rule": (
                "cheb2ord selects N; released fStop vector is passed literally "
                "to scipy.signal.cheby2"
            ),
            "application": "one-pass causal scipy.signal.lfilter per epoch",
        },
        "epoch_count_semantics": {
            "implemented_default_stage1_epoch_count": 1500,
            "released_stage1_MaxEpoch_literal": 1500,
            "released_stage1_effective_epoch_count_without_patience_stop": 1501,
            "released_literal_stage1_epoch_count_without_patience_stop": 1501,
            "implemented_default_stage2_epoch_count": 600,
            "released_stage2_effective_epoch_count": 600,
            "released_literal_stage2_epoch_count": 600,
            "explanation": (
                "the released stage-1 epoch counter starts at zero and checks "
                "epoch > MaxEpoch after optimization; stage 2 has a distinct "
                "counter transition that yields exactly 600 optimization epochs"
            ),
        },
        "test_access_during_fit": False,
        "adaptations": (
            "Braindecode FBCNet layers behind an external original causal filter bank",
            (
                "study-harmonized input window, sampling rate, montage, and "
                "leakage-safe scaling"
            ),
            (
                "study-controlled seed installed before model-factory initialization "
                "rather than the release's single fixed initialization seed"
            ),
            (
                "the configured default stage-1 cap is 1500 optimization epochs; "
                "the release's literal counter path could execute 1501"
            ),
        ),
        "config": asdict(config),
    }


def _reference_forward(model: nn.Module, x: Tensor, positions: Tensor) -> Tensor:
    if bool(getattr(model, "uses_positions", False)):
        output = model(x, positions)
    else:
        output = model(x)
    if isinstance(output, tuple):
        output = output[0]
    if output.ndim > 2:
        output = output.flatten(start_dim=2).mean(dim=2)
    if output.ndim != 2:
        raise RuntimeError(f"model returned invalid logit shape {tuple(output.shape)}")
    return output


def _device(config_device: str) -> torch.device:
    device = torch.device(config_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _as_training_tensors(
    x: NDArray[np.float32],
    y: NDArray[np.int64],
    positions: NDArray[np.float32],
    *,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    x_array = np.asarray(x, dtype=np.float32)
    y_array = np.asarray(y, dtype=np.int64)
    positions_array = np.asarray(positions, dtype=np.float32)
    if x_array.ndim not in (3, 4) or len(x_array) == 0:
        raise ValueError(
            "x must be a nonempty trial tensor with three or four dimensions"
        )
    if y_array.ndim != 1 or len(y_array) != len(x_array):
        raise ValueError("labels must be a vector aligned with x")
    if positions_array.ndim != 2 or positions_array.shape[1] != 3:
        raise ValueError("positions must have shape (channel, 3)")
    channel_axis = 1 if x_array.ndim == 3 else 2
    if len(positions_array) != x_array.shape[channel_axis]:
        raise ValueError("positions length does not match the trial channel axis")
    if np.any(y_array < 0) or len(np.unique(y_array)) < 2:
        raise ValueError(
            "training labels must contain at least two nonnegative classes"
        )
    if not np.isfinite(x_array).all() or not np.isfinite(positions_array).all():
        raise ValueError("training inputs must be finite")
    return (
        torch.as_tensor(x_array, dtype=torch.float32, device=device),
        torch.as_tensor(y_array, dtype=torch.long, device=device),
        torch.as_tensor(positions_array, dtype=torch.float32, device=device),
    )


def _set_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def fit_tcformer_reference(
    model_factory: Callable[[], nn.Module],
    x_train: NDArray[np.float32],
    y_train: NDArray[np.int64],
    positions: NDArray[np.float32],
    *,
    config: TCFormerReferenceConfig = TCFormerReferenceConfig(),
) -> dict[str, Any]:
    """Fit TCFormer for its fixed horizon using source data only.

    ``model_factory`` is intentionally required: ``config.seed`` is installed
    immediately before the factory is called, so the initialization is part of
    the reproducibility contract.  Passing an already initialized module is
    rejected.
    """

    model = _seeded_model_from_factory(model_factory, seed=config.seed)
    initial_state_sha256 = model_state_sha256(model)
    device = _device(config.device)
    model = model.to(device)
    x, y, positions_t = _as_training_tensors(
        x_train, y_train, positions, device=device
    )
    if x.ndim != 3:
        raise ValueError("TCFormer reference training requires unfiltered 3-D trials")
    if x.shape[-1] % config.segment_count:
        raise ValueError("TCFormer S&R requires time length divisible by segment_count")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta_1, config.beta_2),
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator(device=device).manual_seed(config.seed + 7919)
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    for epoch in range(config.epochs):
        learning_rate = config.learning_rate * tcformer_lr_multiplier(
            epoch,
            total_epochs=config.epochs,
            warmup_epochs=config.warmup_epochs,
        )
        _set_learning_rate(optimizer, learning_rate)
        model.train()
        order = torch.randperm(len(x), device=device, generator=generator)
        loss_sum = 0.0
        seen = 0
        for start in range(0, len(order), config.batch_size):
            rows = order[start : start + config.batch_size]
            batch_x, batch_y = tcformer_augmented_batch(
                x[rows],
                y[rows],
                segments=config.segment_count,
                generator=generator,
            )
            optimizer.zero_grad(set_to_none=True)
            logits = _reference_forward(model, batch_x, positions_t)
            loss = nn.functional.cross_entropy(logits, batch_y)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(batch_y)
            seen += len(batch_y)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": loss_sum / max(seen, 1),
                "learning_rate": learning_rate,
                "real_rows": float(len(x)),
                "optimized_rows": float(2 * len(x)),
            }
        )
    # The official warmup-cosine curve reaches zero at the post-training
    # endpoint.  Expose that terminal optimizer state rather than leaving the
    # last epoch's pre-step rate installed.
    terminal_learning_rate = config.learning_rate * tcformer_lr_multiplier(
        config.epochs,
        total_epochs=config.epochs,
        warmup_epochs=config.warmup_epochs,
    )
    _set_learning_rate(optimizer, terminal_learning_rate)
    return {
        "model": model,
        "optimizer": optimizer,
        "initial_state_sha256": initial_state_sha256,
        "final_state_sha256": model_state_sha256(model),
        "terminal_learning_rate": terminal_learning_rate,
        "epochs_run": config.epochs,
        "fit_seconds": time.perf_counter() - started,
        "history": history,
        "recipe": tcformer_reference_metadata(config),
    }


def _plain_optimization_epoch(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
    positions: Tensor,
    optimizer: torch.optim.Optimizer,
    *,
    batch_size: int,
    generator: torch.Generator,
) -> float:
    model.train()
    order = torch.randperm(len(x), device=x.device, generator=generator)
    loss_sum = 0.0
    for start in range(0, len(order), batch_size):
        rows = order[start : start + batch_size]
        optimizer.zero_grad(set_to_none=True)
        logits = _reference_forward(model, x[rows], positions)
        loss = nn.functional.cross_entropy(logits, y[rows])
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.detach()) * len(rows)
    return loss_sum / len(x)


@torch.no_grad()
def _evaluate_loss_accuracy(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
    positions: Tensor,
    *,
    batch_size: int,
) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    for start in range(0, len(x), batch_size):
        stop = min(start + batch_size, len(x))
        logits = _reference_forward(model, x[start:stop], positions)
        loss_sum += float(
            nn.functional.cross_entropy(logits, y[start:stop], reduction="sum")
        )
        correct += int((logits.argmax(dim=1) == y[start:stop]).sum())
    return loss_sum / len(x), correct / len(x)


def fbcnet_stage1_transition_threshold(
    stage1_history: Sequence[Mapping[str, float]],
) -> float:
    """Return the released stage-2 target from the last stage-1 epoch.

    This deliberately does not use the best-checkpoint epoch: the release
    captures the final stage-1 training loss first and restores the best model
    and Adam state afterward.
    """

    if not stage1_history:
        raise ValueError("stage1_history must contain at least one epoch")
    try:
        threshold = float(stage1_history[-1]["train_loss"])
    except KeyError as error:
        raise ValueError("last stage-1 record is missing train_loss") from error
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("stage-1 transition threshold must be finite and nonnegative")
    return threshold


def _require_prefiltered_fbcnet_trials(
    name: str,
    values: NDArray[np.float32],
) -> None:
    array = np.asarray(values)
    if array.ndim != 4 or array.shape[1] != len(FBCNET_BANDS):
        raise ValueError(
            f"{name} must have shape (batch, {len(FBCNET_BANDS)}, channel, time); "
            "apply_fbcnet_cheby2_filterbank before reference fitting"
        )


def fit_fbcnet_reference(
    model_factory: Callable[[], PrefilteredFBCNetAdapter],
    x_train: NDArray[np.float32],
    y_train: NDArray[np.int64],
    x_validation: NDArray[np.float32],
    y_validation: NDArray[np.int64],
    positions: NDArray[np.float32],
    *,
    config: FBCNetReferenceConfig = FBCNetReferenceConfig(),
) -> dict[str, Any]:
    """Fit the pinned prefiltered FBCNet reference procedure.

    The public boundary accepts only nine-band causal-filtered arrays and a
    factory that returns :class:`PrefilteredFBCNetAdapter`.  The configuration
    seed is installed immediately before factory initialization.  Generic
    three-dimensional models are intentionally outside this reference API.
    """

    _require_prefiltered_fbcnet_trials("x_train", x_train)
    _require_prefiltered_fbcnet_trials("x_validation", x_validation)
    _verify_braindecode_version()
    model = _seeded_model_from_factory(model_factory, seed=config.seed)
    if not isinstance(model, PrefilteredFBCNetAdapter):
        raise TypeError(
            "FBCNet reference model factory must return PrefilteredFBCNetAdapter"
        )
    if int(model.base.n_bands) != len(FBCNET_BANDS):
        raise ValueError(
            f"FBCNet reference adapter must expose exactly {len(FBCNET_BANDS)} bands"
        )
    initial_state_sha256 = model_state_sha256(model)
    return _fit_fbcnet_two_stage(
        model,
        x_train,
        y_train,
        x_validation,
        y_validation,
        positions,
        config=config,
        initial_state_sha256=initial_state_sha256,
    )


def _fit_fbcnet_two_stage(
    model: nn.Module,
    x_train: NDArray[np.float32],
    y_train: NDArray[np.int64],
    x_validation: NDArray[np.float32],
    y_validation: NDArray[np.int64],
    positions: NDArray[np.float32],
    *,
    config: FBCNetReferenceConfig,
    initial_state_sha256: str,
) -> dict[str, Any]:
    """Generic mechanics for the released best-restore/combined-data recipe.

    Stage 1 stops after the validation inaccuracy fails its relative-improvement
    rule for 200 epochs (or the cap), preserving both the best model and Adam
    state.  Stage 2 resumes that optimizer on train+validation and stops when
    the *original validation subset* loss reaches the final stage-1 training
    loss, or after 600 epochs.  No test tensor is accepted or observed.
    """

    device = _device(config.device)
    model = model.to(device)
    train_x, train_y, positions_t = _as_training_tensors(
        x_train, y_train, positions, device=device
    )
    validation_x, validation_y, validation_positions = _as_training_tensors(
        x_validation, y_validation, positions, device=device
    )
    if train_x.ndim != validation_x.ndim or train_x.shape[1:] != validation_x.shape[1:]:
        raise ValueError("train and validation trial shapes must match")
    if not torch.equal(positions_t, validation_positions):
        raise RuntimeError("internal position conversion was inconsistent")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta_1, config.beta_2),
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator(device=device).manual_seed(config.seed + 7919)
    best_model_state = copy.deepcopy(model.state_dict())
    best_optimizer_state = copy.deepcopy(optimizer.state_dict())
    best_validation_inaccuracy = float("inf")
    best_epoch = -1
    patience = RelativeNoDecrease(
        patience=config.stage1_patience,
        min_relative_change=config.patience_min_relative_change,
    )
    stage1_history: list[dict[str, float]] = []
    stage1_stop = "max_epochs"
    started = time.perf_counter()

    for epoch in range(config.stage1_max_epochs):
        optimization_loss = _plain_optimization_epoch(
            model,
            train_x,
            train_y,
            positions_t,
            optimizer,
            batch_size=config.batch_size,
            generator=generator,
        )
        train_loss, train_accuracy = _evaluate_loss_accuracy(
            model,
            train_x,
            train_y,
            positions_t,
            batch_size=config.batch_size,
        )
        validation_loss, validation_accuracy = _evaluate_loss_accuracy(
            model,
            validation_x,
            validation_y,
            positions_t,
            batch_size=config.batch_size,
        )
        validation_inaccuracy = 1.0 - validation_accuracy
        if validation_inaccuracy < best_validation_inaccuracy:
            best_validation_inaccuracy = validation_inaccuracy
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())
        patience_exhausted = patience.update(validation_inaccuracy)
        stage1_history.append(
            {
                "epoch": float(epoch),
                "optimization_loss": optimization_loss,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
            }
        )
        if patience_exhausted:
            stage1_stop = "relative_no_decrease_patience"
            break

    # The released code captures this threshold at the stopping epoch, before
    # restoring the best model/optimizer pair.
    stage1_train_loss_threshold = fbcnet_stage1_transition_threshold(stage1_history)
    stage1_threshold_origin_epoch = int(stage1_history[-1]["epoch"])
    best_model_state_sha256 = _state_sha256(best_model_state)
    best_optimizer_state_sha256 = optimizer_state_sha256(best_optimizer_state)
    model.load_state_dict(best_model_state)
    optimizer.load_state_dict(best_optimizer_state)
    stage2_start_model_state_sha256 = model_state_sha256(model)
    stage2_start_optimizer_state_sha256 = optimizer_state_sha256(optimizer)
    if stage2_start_model_state_sha256 != best_model_state_sha256:
        raise RuntimeError("best stage-1 model state was not restored exactly")
    if stage2_start_optimizer_state_sha256 != best_optimizer_state_sha256:
        raise RuntimeError("best stage-1 Adam state was not restored exactly")
    source_x = torch.cat((train_x, validation_x), dim=0)
    source_y = torch.cat((train_y, validation_y), dim=0)
    stage2_history: list[dict[str, float]] = []
    stage2_stop = "max_epochs"

    for epoch in range(config.stage2_max_epochs):
        optimization_loss = _plain_optimization_epoch(
            model,
            source_x,
            source_y,
            positions_t,
            optimizer,
            batch_size=config.batch_size,
            generator=generator,
        )
        source_loss, source_accuracy = _evaluate_loss_accuracy(
            model,
            source_x,
            source_y,
            positions_t,
            batch_size=config.batch_size,
        )
        (
            original_validation_loss,
            original_validation_accuracy,
        ) = _evaluate_loss_accuracy(
            model,
            validation_x,
            validation_y,
            positions_t,
            batch_size=config.batch_size,
        )
        stage2_history.append(
            {
                "epoch": float(epoch),
                "optimization_loss": optimization_loss,
                "source_loss": source_loss,
                "source_accuracy": source_accuracy,
                "original_validation_loss": original_validation_loss,
                "original_validation_accuracy": original_validation_accuracy,
                "source_rows": float(len(source_x)),
                "original_validation_rows": float(len(validation_x)),
            }
        )
        if original_validation_loss <= stage1_train_loss_threshold:
            stage2_stop = "original_validation_loss_reached_stage1_train_loss"
            break

    return {
        "model": model,
        "optimizer": optimizer,
        "initial_state_sha256": initial_state_sha256,
        "final_state_sha256": model_state_sha256(model),
        "final_optimizer_state_sha256": optimizer_state_sha256(optimizer),
        "stage1_best_epoch": best_epoch,
        "stage1_stop": stage1_stop,
        "stage1_epochs_run": len(stage1_history),
        "stage2_epochs_run": len(stage2_history),
        "stage1_train_loss_threshold": stage1_train_loss_threshold,
        "stage1_threshold_origin": "last_stage1_epoch_before_best_restore",
        "stage1_threshold_origin_epoch": stage1_threshold_origin_epoch,
        "stage1_best_model_state_sha256": best_model_state_sha256,
        "stage1_best_optimizer_state_sha256": best_optimizer_state_sha256,
        "stage2_start_model_state_sha256": stage2_start_model_state_sha256,
        "stage2_start_optimizer_state_sha256": stage2_start_optimizer_state_sha256,
        "stage2_source_count": len(source_x),
        "stage2_original_validation_count": len(validation_x),
        "stage2_stop": stage2_stop,
        "fit_seconds": time.perf_counter() - started,
        "stage1_history": stage1_history,
        "stage2_history": stage2_history,
        "recipe": fbcnet_reference_metadata(config),
    }
