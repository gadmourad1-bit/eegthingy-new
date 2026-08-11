"""Gauge-quotient cross-moment continuation for motor-imagery EEG.

This module contains one experimental configuration only.  The continuation
evaluates continuous spatial potentials at the supplied channel coordinates
and projects each potential through the valid-channel centering projector

``P_m = diag(m) - m m.T / (m.T m)``.

For a valid-channel mask ``m``, every resulting spatial filter sums to zero.
Its source signal is therefore invariant (up to floating-point roundoff) to
adding an arbitrary time-varying common signal to every valid model-input
channel.  The same source can equivalently be written as a normalized sum of
all pairwise channel differences.

The invariant sources feed fixed, predeclared band/window log-variance and
shrinkage cross-moment features.  Those features extend the single constrained
head of ``cardinal_fbc_micro_extended`` with exactly zero columns, preserving
the complete native predictor at construction.

Important boundary: the project's point-electrode caches are already common
average referenced, and their source-fitted per-channel scaling need not
commute with a raw-voltage reference change.  The exact invariance proved here
is therefore an invariance of the continuation at its model input, not a claim
that the complete predictor or preprocessing pipeline is reference invariant.
"""

from __future__ import annotations

import math
from typing import Any, Final

import torch
from torch import Tensor, nn

from . import baselines
from .models import (
    EXTENDED_31_POSITIONS,
    CardinalFBCMicroDynamicsNet,
    CardinalScalpField,
    ExpandedMaxNormLinear,
)


MODEL_NAME: Final = "gauge_quotient_crossmoment_v1"
NATIVE_MODEL_NAME: Final = "cardinal_fbc_micro_extended"
BANDS_HZ: Final = (
    (4.0, 8.0),
    (8.0, 12.0),
    (12.0, 16.0),
    (16.0, 22.0),
    (22.0, 30.0),
    (30.0, 40.0),
)
SFREQ_HZ: Final = 128.0
FIR_KERNEL_SAMPLES: Final = 65
FIR_ZERO_DC_TOLERANCE: Final = 1e-7
FIR_UNIT_NORM_TOLERANCE: Final = 1e-6
N_QUOTIENT_SOURCES: Final = 4
N_WINDOWS: Final = 4
LAGS_SAMPLES: Final = (1, 2, 4, 8)
SHRINKAGE: Final = 0.10
VARIANCE_FLOOR: Final = 1e-6
FISHER_CLIP: Final = 1.0 - 1e-4
MAXIMUM_PARAMETERS: Final = 30_000


def _fixed_bandpass_kernels(
    *,
    bands_hz: tuple[tuple[float, float], ...] = BANDS_HZ,
    sfreq: float = SFREQ_HZ,
    kernel_size: int = FIR_KERNEL_SAMPLES,
) -> Tensor:
    """Construct deterministic, zero-DC, unit-norm windowed-sinc filters."""

    if kernel_size <= 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be an odd integer greater than one")
    if not bands_hz:
        raise ValueError("at least one band is required")
    if any(
        not 0.0 < low < high < sfreq / 2.0
        for low, high in bands_hz
    ):
        raise ValueError("every band must lie strictly inside Nyquist")
    samples = torch.arange(
        -(kernel_size // 2),
        kernel_size // 2 + 1,
        dtype=torch.float64,
    )
    window = torch.hamming_window(
        kernel_size,
        periodic=False,
        dtype=torch.float64,
    )
    kernels: list[Tensor] = []
    for low, high in bands_hz:
        high_pass = (2.0 * high / sfreq) * torch.sinc(
            2.0 * high * samples / sfreq
        )
        low_pass = (2.0 * low / sfreq) * torch.sinc(
            2.0 * low * samples / sfreq
        )
        kernel = (high_pass - low_pass) * window
        kernel = kernel - kernel.mean()
        kernel = kernel / torch.linalg.vector_norm(kernel).clamp_min(1e-12)

        # Conversion to the stored float32 representation can reintroduce a
        # small DC residual.  Correct that residual in float64 on the centre
        # tap after conversion.  The unit-norm check is deliberately after the
        # correction so both numerical contracts describe the actual buffer
        # consumed by conv1d, rather than its float64 design precursor.
        stored = kernel.to(torch.float32)
        dc_residual = stored.to(torch.float64).sum()
        centre = kernel_size // 2
        stored[centre] = (
            stored[centre].to(torch.float64) - dc_residual
        ).to(torch.float32)
        stored_dc = abs(float(stored.to(torch.float64).sum().item()))
        stored_norm = float(
            torch.linalg.vector_norm(stored.to(torch.float64)).item()
        )
        if stored_dc > FIR_ZERO_DC_TOLERANCE:
            raise RuntimeError(
                "stored FIR coefficients exceed the frozen zero-DC tolerance"
            )
        if abs(stored_norm - 1.0) > FIR_UNIT_NORM_TOLERANCE:
            raise RuntimeError(
                "stored FIR coefficients exceed the frozen unit-norm tolerance"
            )
        kernels.append(stored)
    return torch.stack(kernels)[:, None, :]


class FixedSensorimotorBandBank(nn.Module):
    """One immutable six-band FIR bank shared across all EEG channels."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("kernels", _fixed_bandpass_kernels())

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError("x must have shape (batch, channels, time)")
        batch, channels, n_times = x.shape
        filtered = nn.functional.conv1d(
            x.reshape(batch * channels, 1, n_times),
            self.kernels.to(x),
            padding=FIR_KERNEL_SAMPLES // 2,
        )
        return filtered.reshape(
            batch,
            channels,
            len(BANDS_HZ),
            n_times,
        )


class GaugeQuotientSpatialField(nn.Module):
    """Continuous spatial potentials projected into the voltage quotient.

    The stored parameters are potentials rather than direct channel weights.
    A regularized cardinal RBF field evaluates those potentials at arbitrary
    supplied coordinates.  The mask-dependent centering projector then removes
    the constant spatial mode separately for every batch item, band, and
    source.  Invalid channels have exactly zero weight.
    """

    def __init__(self) -> None:
        super().__init__()
        self.potential_field = CardinalScalpField(
            n_filters=len(BANDS_HZ),
            n_sources=N_QUOTIENT_SOURCES,
            anchor_positions=EXTENDED_31_POSITIONS,
            field_norm="none",
        )

    @staticmethod
    def _valid_mask(
        *,
        batch: int,
        channels: int,
        device: torch.device,
        channel_mask: Tensor | None,
    ) -> Tensor:
        if channel_mask is None:
            return torch.ones(
                batch,
                channels,
                dtype=torch.bool,
                device=device,
            )
        if channel_mask.shape != (batch, channels):
            raise ValueError("channel_mask must have shape (batch, channels)")
        if channel_mask.dtype is not torch.bool:
            raise TypeError("channel_mask must be boolean")
        valid = channel_mask.to(device=device)
        if torch.any(valid.sum(dim=-1) == 0):
            raise ValueError("every trial must contain at least one valid channel")
        return valid

    def potentials(self, positions: Tensor) -> Tensor:
        """Return unprojected continuous potentials as ``(band, source, ch)``."""

        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions must have shape (channels, 3)")
        if not positions.is_floating_point():
            raise TypeError("positions must be a real floating-point tensor")
        if not torch.isfinite(positions).all():
            raise ValueError("positions must be finite")
        if torch.any(torch.linalg.vector_norm(positions, dim=-1) <= 0.0):
            raise ValueError("positions must be nonzero")
        basis = self.potential_field.basis(positions).to(
            self.potential_field.coefficients
        )
        return torch.einsum(
            "fsa,ca->fsc",
            self.potential_field.coefficients,
            basis,
        )

    def projected_potentials(
        self,
        positions: Tensor,
        *,
        batch: int,
        channel_mask: Tensor | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> Tensor:
        """Return normalized valid-channel sum-zero filters.

        The output shape is ``(batch, band, source, channel)``.  With one valid
        channel the quotient is the trivial zero space and the returned filter
        is exactly zero.
        """

        target_device = positions.device if device is None else device
        potentials = self.potentials(positions).to(
            device=target_device,
            dtype=dtype,
        )
        valid = self._valid_mask(
            batch=batch,
            channels=positions.shape[0],
            device=target_device,
            channel_mask=channel_mask,
        )
        mask = valid[:, None, None, :].to(potentials.dtype)
        counts = mask.sum(dim=-1, keepdim=True)
        masked_mean = (potentials[None] * mask).sum(
            dim=-1,
            keepdim=True,
        ) / counts
        quotient = (potentials[None] - masked_mean) * mask
        norms = torch.linalg.vector_norm(
            quotient,
            dim=-1,
            keepdim=True,
        )
        return quotient / norms.clamp_min(1e-7)

    def forward(
        self,
        filtered: Tensor,
        positions: Tensor,
        channel_mask: Tensor | None = None,
    ) -> Tensor:
        if filtered.ndim != 4:
            raise ValueError(
                "filtered must have shape (batch, channels, bands, time)"
            )
        batch, channels, bands, _ = filtered.shape
        if positions.shape != (channels, 3):
            raise ValueError("positions do not match the filtered channel axis")
        if bands != len(BANDS_HZ):
            raise ValueError("filtered band count does not match the fixed bank")
        weights = self.projected_potentials(
            positions,
            batch=batch,
            channel_mask=channel_mask,
            dtype=filtered.dtype,
            device=filtered.device,
        )
        return torch.einsum("bfsc,bcft->bfst", weights, filtered)


class GaugeQuotientCrossMomentNet(nn.Module):
    """Function-preserving CardinalFBC floor plus one quotient continuation."""

    uses_positions = True

    def __init__(
        self,
        native_backbone: CardinalFBCMicroDynamicsNet,
    ) -> None:
        super().__init__()
        if not isinstance(native_backbone, CardinalFBCMicroDynamicsNet):
            raise TypeError(
                "native_backbone must be CardinalFBCMicroDynamicsNet"
            )
        native_config = getattr(native_backbone, "config", None)
        required_native_config = {
            "extended_atlas": True,
            "continuation_kind": "gabor_fir",
            "dynamic_fbc_floor_input_channels": True,
            "dynamic_fbc_floor_input_times": True,
            "zero_initialized_continuation_head": True,
            "single_expanded_head": True,
        }
        if not isinstance(native_config, dict):
            raise TypeError("native backbone must expose dictionary metadata")
        for key, expected in required_native_config.items():
            if native_config.get(key) != expected:
                raise ValueError(
                    f"native backbone metadata {key!r} must equal {expected!r}"
                )
        native_head = native_backbone.final_layer
        if not isinstance(native_head, nn.Linear):
            raise TypeError("native backbone must expose one linear head")
        native_feature_count = (
            native_backbone.floor_feature_count
            + native_backbone.continuation_feature_count
        )
        if native_head.in_features != native_feature_count:
            raise ValueError("native head does not match its encoded features")

        pair_indices = torch.triu_indices(
            N_QUOTIENT_SOURCES,
            N_QUOTIENT_SOURCES,
            offset=1,
        )
        self.register_buffer("pair_rows", pair_indices[0])
        self.register_buffer("pair_columns", pair_indices[1])
        self.band_bank = FixedSensorimotorBandBank()
        self.quotient_field = GaugeQuotientSpatialField()
        self.native_backbone = native_backbone
        self.native_backbone.final_layer = nn.Identity()
        self.native_feature_count = int(native_feature_count)
        self.log_variance_feature_count = (
            len(BANDS_HZ) * N_WINDOWS * N_QUOTIENT_SOURCES
        )
        self.cross_moment_feature_count = (
            len(BANDS_HZ)
            * N_WINDOWS
            * len(LAGS_SAMPLES)
            * N_QUOTIENT_SOURCES
            * (N_QUOTIENT_SOURCES - 1)
            // 2
        )
        self.quotient_feature_count = (
            self.log_variance_feature_count
            + self.cross_moment_feature_count
        )
        self.final_layer = ExpandedMaxNormLinear.extend(
            native_head,
            self.quotient_feature_count,
        )
        self.config: dict[str, Any] = {
            "architecture": MODEL_NAME,
            "architecture_family": "gauge_quotient_cross_moment",
            "single_configuration": True,
            "native_model": NATIVE_MODEL_NAME,
            "native_config": dict(native_config),
            "native_feature_count": self.native_feature_count,
            "native_head_removed_from_backbone": True,
            "single_expanded_head": True,
            "zero_initialized_quotient_head_columns": True,
            "bands_hz": BANDS_HZ,
            "sfreq_hz": SFREQ_HZ,
            "fir_kernel_samples": FIR_KERNEL_SAMPLES,
            "fir_trainable": False,
            "quotient_sources": N_QUOTIENT_SOURCES,
            "windows": N_WINDOWS,
            "lag_samples": LAGS_SAMPLES,
            "lag_moment": (
                "normalized_antisymmetric_cross_covariance_time_reversal_odd"
            ),
            "quotient_continuation_global_voltage_sign_invariant": True,
            "complete_predictor_global_voltage_sign_invariant": False,
            "source_permutation_invariant": False,
            "source_order": (
                "learned_potential_index_bound_to_expanded_head_columns"
            ),
            "covariance_shrinkage": SHRINKAGE,
            "variance_floor": VARIANCE_FLOOR,
            "fisher_clip": FISHER_CLIP,
            "continuous_potential": "regularized_cardinal_rbf_extended_31",
            "stored_potential_coefficients_sum_zero": False,
            "mask_projection": "diag(m)-m*m.T/(m.T*m)",
            "projected_spatial_weights_sum_zero_up_to_floating_point": True,
            "complete_graph_edge_equivalent": True,
            "invariance_scope": (
                "arbitrary_time_varying_common_offset_at_model_input_"
                "over_valid_channels_for_quotient_continuation_only"
            ),
            "complete_predictor_invariant": False,
            "complete_predictor_common_mode_invariant": False,
            "preprocessing_commutation_claim": False,
            "log_variance_feature_count": self.log_variance_feature_count,
            "cross_moment_feature_count": self.cross_moment_feature_count,
            "quotient_feature_count": self.quotient_feature_count,
            "maximum_parameter_budget": MAXIMUM_PARAMETERS,
        }
        if torch.count_nonzero(
            self.final_layer.weight[:, self.native_feature_count :]
        ):
            raise RuntimeError("quotient head columns were not initialized to zero")

    @staticmethod
    def _validate_input(
        x: Tensor,
        positions: Tensor,
        channel_mask: Tensor | None,
    ) -> Tensor:
        if not isinstance(x, Tensor):
            raise TypeError("x must be a tensor")
        if x.ndim != 3:
            raise ValueError("x must have shape (batch, channels, time)")
        if any(dimension <= 0 for dimension in x.shape):
            raise ValueError("x batch, channel, and time dimensions must be nonempty")
        if not x.is_floating_point():
            raise TypeError("x must be a real floating-point tensor")
        if not torch.isfinite(x).all():
            raise ValueError("x must be finite")
        if not isinstance(positions, Tensor):
            raise TypeError("positions must be a tensor")
        if positions.ndim != 2 or positions.shape != (x.shape[1], 3):
            raise ValueError("positions do not match the EEG channel axis")
        if not positions.is_floating_point():
            raise TypeError("positions must be a real floating-point tensor")
        if not torch.isfinite(positions).all():
            raise ValueError("positions must be finite")
        if torch.any(torch.linalg.vector_norm(positions, dim=-1) <= 0.0):
            raise ValueError("positions must be nonzero")
        if x.shape[-1] % N_WINDOWS:
            raise ValueError(
                f"input time length must be divisible by {N_WINDOWS}"
            )
        if max(LAGS_SAMPLES) >= x.shape[-1] // N_WINDOWS:
            raise ValueError("a fixed lag is not shorter than each time window")
        if channel_mask is None:
            return torch.ones(
                x.shape[:2],
                dtype=torch.bool,
                device=x.device,
            )
        if not isinstance(channel_mask, Tensor):
            raise TypeError("channel_mask must be a tensor")
        if channel_mask.shape != x.shape[:2]:
            raise ValueError("channel_mask must have shape (batch, channels)")
        if channel_mask.dtype is not torch.bool:
            raise TypeError("channel_mask must be boolean")
        valid = channel_mask.to(device=x.device)
        if torch.any(valid.sum(dim=-1) == 0):
            raise ValueError("every trial must contain at least one valid channel")
        return valid

    def _encode_native_validated(
        self,
        x: Tensor,
        positions: Tensor,
        channel_mask: Tensor | None,
    ) -> Tensor:
        """Encode native features after the pure candidate boundary check."""

        floor, _signed_sources = (
            self.native_backbone.encode_floor_with_signed_sources(
                x,
                positions,
                channel_mask,
            )
        )
        micro = self.native_backbone.encode_continuation(
            x,
            positions,
            channel_mask,
        )
        return torch.cat((floor, micro), dim=1)

    def encode_native(
        self,
        x: Tensor,
        positions: Tensor,
        channel_mask: Tensor | None = None,
    ) -> Tensor:
        """Return the native floor-plus-micro vector with one floor BN call.

        ``encode_floor_with_signed_sources`` is invoked exactly once.  The
        signed tensor is intentionally not recomputed: doing so would advance
        the stateful floor batch normalization twice during one forward pass.
        The gauge branch has its own fixed band bank and no batch normalization.
        """

        valid = self._validate_input(x, positions, channel_mask)
        native_mask = None if channel_mask is None else valid
        return self._encode_native_validated(x, positions, native_mask)

    def _quotient_sources_validated(
        self,
        x: Tensor,
        positions: Tensor,
        valid: Tensor,
    ) -> Tensor:
        """Return quotient sources after the pure candidate boundary check."""

        masked_x = torch.where(valid[..., None], x, torch.zeros_like(x))
        filtered = self.band_bank(masked_x)
        return self.quotient_field(filtered, positions, valid)

    def quotient_sources(
        self,
        x: Tensor,
        positions: Tensor,
        channel_mask: Tensor | None = None,
    ) -> Tensor:
        """Return quotient sources as ``(batch, band, source, time)``."""

        valid = self._validate_input(x, positions, channel_mask)
        return self._quotient_sources_validated(x, positions, valid)

    def _encode_quotient_validated(
        self,
        x: Tensor,
        positions: Tensor,
        valid: Tensor,
    ) -> Tensor:
        """Encode quotient features after the pure candidate boundary check."""

        sources = self._quotient_sources_validated(x, positions, valid)
        log_variance, cross_moments = self.quotient_statistics(sources)
        return torch.cat(
            (
                log_variance.flatten(start_dim=1),
                cross_moments.flatten(start_dim=1),
            ),
            dim=1,
        )

    def encode_quotient(
        self,
        x: Tensor,
        positions: Tensor,
        channel_mask: Tensor | None = None,
    ) -> Tensor:
        """Encode fixed log-variance and shrinkage-normalized lag wedges."""

        valid = self._validate_input(x, positions, channel_mask)
        return self._encode_quotient_validated(x, positions, valid)

    def quotient_statistics(
        self,
        sources: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return log-variance and time-reversal-odd lag cross-moments.

        ``sources`` must have shape ``(batch, band, source, time)``.  The
        returned tensors have shapes ``(batch, band, window, source)`` and
        ``(batch, band, window, lag, source_pair)``.  For lag ``q`` the signed
        numerator is

        ``0.5 * E[z_i(t) z_j(t+q) - z_j(t) z_i(t+q)]``.

        It is the skew-symmetric (wedge) part of the lagged cross-covariance:
        it vanishes at lag zero and changes sign under time reversal.  A fixed
        diagonal-shrinkage scale stabilizes normalization without converting
        the moment into a same-time correlation feature.
        """

        if sources.ndim != 4:
            raise ValueError(
                "sources must have shape (batch, band, source, time)"
            )
        batch, bands, source_count, n_times = sources.shape
        if batch <= 0 or n_times <= 0:
            raise ValueError("source batch and time dimensions must be nonempty")
        if not sources.is_floating_point():
            raise TypeError("sources must be a real floating-point tensor")
        if not torch.isfinite(sources).all():
            raise ValueError("sources must be finite")
        if bands != len(BANDS_HZ) or source_count != N_QUOTIENT_SOURCES:
            raise ValueError("sources have incompatible band/source dimensions")
        if n_times % N_WINDOWS:
            raise ValueError(
                f"source time length must be divisible by {N_WINDOWS}"
            )
        window_length = n_times // N_WINDOWS
        if max(LAGS_SAMPLES) >= window_length:
            raise ValueError("a fixed lag is not shorter than each time window")
        windows = sources.reshape(
            batch,
            bands,
            source_count,
            N_WINDOWS,
            window_length,
        ).permute(0, 1, 3, 2, 4)
        centered = windows - windows.mean(dim=-1, keepdim=True)
        covariance = centered @ centered.transpose(-1, -2)
        covariance = covariance / max(window_length - 1, 1)
        variance = covariance.diagonal(dim1=-2, dim2=-1)
        log_variance = torch.log(variance.clamp_min(VARIANCE_FLOOR))

        mean_variance = variance.mean(dim=-1, keepdim=True)
        identity = torch.eye(
            source_count,
            dtype=covariance.dtype,
            device=covariance.device,
        )
        shrunk = (
            (1.0 - SHRINKAGE) * covariance
            + SHRINKAGE
            * mean_variance[..., None]
            * identity[None, None, None]
        )
        shrunk_variance = shrunk.diagonal(dim1=-2, dim2=-1).clamp_min(
            VARIANCE_FLOOR
        )
        pair_scale = torch.sqrt(
            shrunk_variance[..., :, None]
            * shrunk_variance[..., None, :]
        ).clamp_min(VARIANCE_FLOOR)
        lag_moments: list[Tensor] = []
        for lag in LAGS_SAMPLES:
            earlier = centered[..., :-lag]
            later = centered[..., lag:]
            lagged = torch.einsum(
                "...it,...jt->...ij",
                earlier,
                later,
            ) / (window_length - lag)
            wedge = 0.5 * (lagged - lagged.transpose(-1, -2))
            normalized = (wedge / pair_scale)[
                ...,
                self.pair_rows,
                self.pair_columns,
            ].clamp(-FISHER_CLIP, FISHER_CLIP)
            lag_moments.append(torch.atanh(normalized))
        return log_variance, torch.stack(lag_moments, dim=-2)

    def forward(
        self,
        x: Tensor,
        positions: Tensor,
        channel_mask: Tensor | None = None,
    ) -> Tensor:
        # All candidate-level rejection conditions are pure and run before the
        # first native filter/batch-normalization call.  This matters in train
        # mode: a rejected candidate call must not partially update native
        # buffers, fire native module hooks, or consume RNG state.
        valid = self._validate_input(x, positions, channel_mask)
        native_mask = None if channel_mask is None else valid
        native = self._encode_native_validated(x, positions, native_mask)
        quotient = self._encode_quotient_validated(x, positions, valid)
        return self.final_layer(torch.cat((native, quotient), dim=1))


def make_gauge_quotient_model(
    *,
    requested_model: str = MODEL_NAME,
    n_channels: int,
    n_outputs: int,
    n_times: int,
    sfreq: float,
    channel_names: tuple[str, ...] | None = None,
    channel_positions: Tensor | None = None,
) -> GaugeQuotientCrossMomentNet:
    """Build the sole candidate directly, without touching formal registries."""

    if requested_model != MODEL_NAME:
        raise ValueError(f"unsupported requested model {requested_model!r}")
    if isinstance(n_channels, bool) or not isinstance(n_channels, int):
        raise TypeError("n_channels must be an integer")
    if n_channels <= 0:
        raise ValueError("n_channels must be positive")
    if isinstance(n_outputs, bool) or not isinstance(n_outputs, int):
        raise TypeError("n_outputs must be an integer")
    if n_outputs <= 0:
        raise ValueError("n_outputs must be positive")
    if isinstance(n_times, bool) or not isinstance(n_times, int):
        raise TypeError("n_times must be an integer")
    if n_times <= 0:
        raise ValueError("n_times must be positive")
    if n_times % N_WINDOWS:
        raise ValueError(f"n_times must be divisible by {N_WINDOWS}")
    if max(LAGS_SAMPLES) >= n_times // N_WINDOWS:
        raise ValueError("a fixed lag is not shorter than each time window")
    try:
        sfreq_value = float(sfreq)
    except (TypeError, ValueError) as error:
        raise TypeError("sfreq must be a real finite scalar") from error
    if not math.isfinite(sfreq_value):
        raise ValueError("sfreq must be finite")
    if not math.isclose(sfreq_value, SFREQ_HZ, rel_tol=0.0, abs_tol=0.0):
        raise ValueError(f"{MODEL_NAME} is frozen at {SFREQ_HZ} Hz")
    if channel_positions is None:
        raise ValueError(f"{MODEL_NAME} requires channel_positions")
    if not isinstance(channel_positions, Tensor):
        raise TypeError("channel_positions must be a tensor")
    if channel_positions.ndim != 2 or channel_positions.shape != (n_channels, 3):
        raise ValueError(
            "channel_positions must have shape (n_channels, 3)"
        )
    if not channel_positions.is_floating_point():
        raise TypeError(
            "channel_positions must be a real floating-point tensor"
        )
    if not torch.isfinite(channel_positions).all():
        raise ValueError("channel_positions must be finite")
    if torch.any(
        torch.linalg.vector_norm(channel_positions, dim=-1) <= 0.0
    ):
        raise ValueError("channel_positions must be nonzero")
    if channel_names is not None and len(channel_names) != n_channels:
        raise ValueError("channel_names must match n_channels")
    native = baselines.make_model(
        NATIVE_MODEL_NAME,
        n_channels=n_channels,
        n_outputs=n_outputs,
        n_times=n_times,
        sfreq=sfreq_value,
        channel_names=channel_names,
        channel_positions=channel_positions,
    )
    if not isinstance(native, CardinalFBCMicroDynamicsNet):
        raise RuntimeError("native model factory returned an unexpected class")
    model = GaugeQuotientCrossMomentNet(native)
    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if parameter_count > MAXIMUM_PARAMETERS:
        raise RuntimeError(
            f"{MODEL_NAME} has {parameter_count} parameters, exceeding "
            f"the frozen budget of {MAXIMUM_PARAMETERS}"
        )
    return model


__all__ = [
    "BANDS_HZ",
    "FIR_KERNEL_SAMPLES",
    "FIR_UNIT_NORM_TOLERANCE",
    "FIR_ZERO_DC_TOLERANCE",
    "FISHER_CLIP",
    "FixedSensorimotorBandBank",
    "GaugeQuotientCrossMomentNet",
    "GaugeQuotientSpatialField",
    "LAGS_SAMPLES",
    "MAXIMUM_PARAMETERS",
    "MODEL_NAME",
    "NATIVE_MODEL_NAME",
    "N_QUOTIENT_SOURCES",
    "N_WINDOWS",
    "SFREQ_HZ",
    "SHRINKAGE",
    "VARIANCE_FLOOR",
    "make_gauge_quotient_model",
]
