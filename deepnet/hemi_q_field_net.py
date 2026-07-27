"""HemiQ-FieldNet: a compact parity-field decoder for binary motor imagery.

The network is deliberately specialised to the C3--Cz--C4 strip.  Sagittal
reflection swaps C3 and C4 and, for left/right imagery, must negate the binary
logit.  HemiQ-FieldNet enforces that action throughout the computation:

* ``e_h=(C3+C4)/sqrt(2)`` and ``Cz`` are even, while
  ``o=(C3-C4)/sqrt(2)`` is odd;
* a shared bank of constrained, ordered quadrature Gabor filters produces
  local complex coefficients without mixing parity types;
* local powers and coherence magnitudes form an invariant context, while the
  real/imaginary cross-products of ``o`` with ``e_h`` and ``Cz`` form an odd
  signed field;
* bounded first, signed-second, and third moments of the C3-vs-C4 log-power
  asymmetry augment that field without introducing outlier-sensitive ratios;
* one full-epoch token per learned frequency complements the local tokens,
  with an invariant coordinate identifying the temporal scale;
* two even-conditioned blocks apply only bias-free maps to the odd field; and
* invariant attention takes a weighted sum of bias-free odd token scores.

Consequently the sole output obeys ``f(Mx)=-f(x)`` by construction in both
training and evaluation modes.  There is no dropout, BatchNorm, candidate
selection, expert router, or inference-time tangent path.

The sklearn-like classifier optionally uses a projected source-only tangent
logistic model as a *training teacher*.  Its coefficient decays linearly to
zero during the first half of fitting; validation checkpointing and inference
always use the neural logit alone.
"""

from __future__ import annotations

import copy
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.metrics import balanced_accuracy_score
from torch import Tensor, nn

from .cameo_net import FrozenTangentAnchor, _mirror_covariances
from .config import CHANNELS, SFREQ


HEMI_Q_CHANNELS = ("C3", "Cz", "C4")


def _binary_loss(logit: Tensor, labels: Tensor) -> Tensor:
    return nn.functional.binary_cross_entropy_with_logits(
        logit, labels.to(dtype=logit.dtype)
    )


def _resolve_device(value: str) -> torch.device:
    if value in ("", "auto", None):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _configure_torch_determinism(enabled: bool) -> None:
    if enabled:
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(bool(enabled), warn_only=False)


class ConstrainedQuadratureGaborBank(nn.Module):
    """Learned ordered Gabor pairs constrained to a frequency interval.

    ``K+1`` positive gaps partition the requested interval; the first ``K``
    cumulative boundaries are the filter centres.  This makes the centres
    strictly ordered and strictly interior without sorting learned parameters.
    Every centre has a learned positive half-power bandwidth.  Cosine and sine
    kernels share the same Gaussian envelope and differ only by a quarter-cycle
    phase, so their ordering is always ``(cos_0, sin_0, ..., cos_K, sin_K)``.
    """

    def __init__(
        self,
        *,
        n_filters: int = 12,
        kernel_size: int = 63,
        sfreq: float = 125.0,
        frequency_low: float = 8.0,
        frequency_high: float = 30.0,
        bandwidth_low: float = 1.5,
        bandwidth_high: float = 8.0,
        temporal_stride: int = 2,
    ) -> None:
        super().__init__()
        if n_filters <= 0:
            raise ValueError("n_filters must be positive")
        if kernel_size <= 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer greater than one")
        if temporal_stride <= 0:
            raise ValueError("temporal_stride must be positive")
        if sfreq <= 0.0:
            raise ValueError("sfreq must be positive")
        if not 0.0 < frequency_low < frequency_high < sfreq / 2.0:
            raise ValueError("Gabor frequencies must lie inside the Nyquist interval")
        if not 0.0 < bandwidth_low < bandwidth_high:
            raise ValueError("bandwidth bounds must be positive and ordered")

        self.n_filters = int(n_filters)
        self.kernel_size = int(kernel_size)
        self.sfreq = float(sfreq)
        self.frequency_low = float(frequency_low)
        self.frequency_high = float(frequency_high)
        self.bandwidth_low = float(bandwidth_low)
        self.bandwidth_high = float(bandwidth_high)
        self.temporal_stride = int(temporal_stride)

        # Equal positive gaps initialise linearly spaced interior centres.
        self.raw_frequency_gaps = nn.Parameter(torch.zeros(n_filters + 1))
        initial_bandwidth = 3.5
        fraction = (initial_bandwidth - bandwidth_low) / (
            bandwidth_high - bandwidth_low
        )
        fraction = min(max(fraction, 1e-4), 1.0 - 1e-4)
        raw_bandwidth = math.log(fraction / (1.0 - fraction))
        self.raw_bandwidths = nn.Parameter(
            torch.full((n_filters,), float(raw_bandwidth))
        )

    def frequencies_hz(self) -> Tensor:
        gaps = nn.functional.softplus(self.raw_frequency_gaps) + 1e-4
        fractions = torch.cumsum(gaps[:-1], dim=0) / gaps.sum()
        return self.frequency_low + (
            self.frequency_high - self.frequency_low
        ) * fractions

    def bandwidths_hz(self) -> Tensor:
        return self.bandwidth_low + (
            self.bandwidth_high - self.bandwidth_low
        ) * torch.sigmoid(self.raw_bandwidths)

    def kernels(self) -> Tensor:
        frequencies = self.frequencies_hz()
        bandwidths = self.bandwidths_hz()
        positions = torch.arange(
            self.kernel_size,
            dtype=frequencies.dtype,
            device=frequencies.device,
        )
        positions = (positions - self.kernel_size // 2) / self.sfreq

        # For a Gaussian-windowed sinusoid this maps the learned frequency
        # bandwidth monotonically to a positive temporal width.
        sigma = math.sqrt(math.log(2.0)) / (math.pi * bandwidths)
        envelope = torch.exp(
            -0.5 * torch.square(positions.unsqueeze(0) / sigma.unsqueeze(1))
        )
        phase = 2.0 * math.pi * frequencies.unsqueeze(1) * positions.unsqueeze(0)
        cosine = envelope * torch.cos(phase)
        sine = envelope * torch.sin(phase)

        # Removing the finite-window DC component prevents a learned low-pass
        # shortcut.  Symmetric cosine and antisymmetric sine kernels remain
        # orthogonal, after which each component receives unit L2 norm.
        cosine = cosine - cosine.mean(dim=1, keepdim=True)
        sine = sine - sine.mean(dim=1, keepdim=True)
        cosine = cosine / torch.linalg.vector_norm(
            cosine, dim=1, keepdim=True
        ).clamp_min(1e-8)
        sine = sine / torch.linalg.vector_norm(
            sine, dim=1, keepdim=True
        ).clamp_min(1e-8)
        return torch.stack((cosine, sine), dim=1).reshape(
            2 * self.n_filters, 1, self.kernel_size
        )

    def forward(self, signals: Tensor) -> Tensor:
        """Return complex coefficients with shape ``(B,C,K,2,T')``."""

        if signals.ndim != 3:
            raise ValueError("signals must have shape (batch, channels, time)")
        batch, channels, n_times = signals.shape
        if n_times < self.kernel_size:
            raise ValueError("signal is shorter than the Gabor kernel")
        flattened = signals.reshape(batch * channels, 1, n_times)
        filtered = nn.functional.conv1d(
            flattened,
            self.kernels(),
            stride=self.temporal_stride,
            padding=self.kernel_size // 2,
        )
        return filtered.reshape(
            batch, channels, self.n_filters, 2, filtered.shape[-1]
        )


@dataclass
class HemiQFieldFeatures:
    """Auditable parity-field tensors; only ``logit`` is a model output."""

    logit: Tensor
    invariant_context: Tensor
    odd_field: Tensor
    hidden_odd: Tensor
    attention: Tensor


class EvenConditionedOddBlock(nn.Module):
    """An odd map modulated exclusively by an even context.

    The context network may contain biases.  Every path carrying an odd value
    is bias-free and uses an odd activation, so negating ``odd`` negates the
    result while holding ``even`` fixed.
    """

    def __init__(
        self,
        *,
        odd_input: int,
        even_width: int,
        width: int = 32,
        residual: bool = False,
    ) -> None:
        super().__init__()
        if min(odd_input, even_width, width) <= 0:
            raise ValueError("block dimensions must be positive")
        if residual and odd_input != width:
            raise ValueError("a residual odd block requires odd_input == width")
        self.residual = bool(residual)
        self.primary = nn.Linear(odd_input, width, bias=False)
        self.modulated = nn.Linear(odd_input, width, bias=False)
        self.conditioner = nn.Sequential(
            nn.Linear(even_width, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.output = nn.Linear(width, width, bias=False)

    def forward(self, odd: Tensor, even: Tensor) -> Tensor:
        if odd.shape[:-1] != even.shape[:-1]:
            raise ValueError("odd field and even context token axes must match")
        primary = torch.tanh(self.primary(odd))
        modulated = torch.tanh(self.modulated(odd))
        gate = torch.sigmoid(self.conditioner(even))
        update = torch.tanh(self.output(primary + gate * modulated))
        return odd + update if self.residual else update


class HemiQFieldNet(nn.Module):
    """Exact reflection-anti-equivariant C3--Cz--C4 raw-EEG decoder.

    For complex Gabor coefficients ``q_3`` and ``q_4``, define the bounded
    instantaneous hemispheric contrast

    ``u = tanh((log(|q_3|^2 + eps) - log(|q_4|^2 + eps)) / 2)``.

    Each temporal scale contributes ``E[u]``, ``E[u |u|]``, ``E[u^3]``, and
    the correspondingly bounded contrast of its pooled powers.  All four are
    odd under C3/C4 exchange.  They accompany the original four odd complex
    cross-spectral coordinates at both local-window and full-epoch scales.
    """

    input_channels = HEMI_Q_CHANNELS

    def __init__(
        self,
        *,
        n_times: int = 251,
        sfreq: float = 125.0,
        n_filters: int = 12,
        kernel_size: int = 63,
        temporal_stride: int = 2,
        local_window: int = 25,
        local_stride: int = 10,
        width: int = 32,
        frequency_low: float = 8.0,
        frequency_high: float = 30.0,
        bandwidth_low: float = 1.5,
        bandwidth_high: float = 8.0,
        spectral_epsilon: float = 1e-5,
    ) -> None:
        super().__init__()
        if n_times <= 0:
            raise ValueError("n_times must be positive")
        if local_window <= 0 or local_stride <= 0:
            raise ValueError("local pooling dimensions must be positive")
        if width <= 0:
            raise ValueError("width must be positive")
        if spectral_epsilon <= 0.0:
            raise ValueError("spectral_epsilon must be positive")
        convolution_length = (n_times - 1) // temporal_stride + 1
        if convolution_length < local_window:
            raise ValueError("local_window exceeds the filtered time dimension")

        self.n_times = int(n_times)
        self.n_filters = int(n_filters)
        self.local_window = int(local_window)
        self.local_stride = int(local_stride)
        self.width = int(width)
        self.spectral_epsilon = float(spectral_epsilon)
        self.filter_bank = ConstrainedQuadratureGaborBank(
            n_filters=n_filters,
            kernel_size=kernel_size,
            sfreq=sfreq,
            frequency_low=frequency_low,
            frequency_high=frequency_high,
            bandwidth_low=bandwidth_low,
            bandwidth_high=bandwidth_high,
            temporal_stride=temporal_stride,
        )

        convolution_length = (n_times - 1) // temporal_stride + 1
        self.local_token_count = (
            convolution_length - local_window
        ) // local_stride + 1
        self.token_count_per_frequency = self.local_token_count + 1

        # I = three log powers, three coherence magnitudes, and frequency,
        # time, and temporal-scale coordinates.
        self.invariant_dim = 9
        # F = four original complex cross-products plus four robust signed
        # C3-vs-C4 log-power/asymmetry moments.
        self.odd_field_dim = 8
        self.context_encoder = nn.Sequential(
            nn.Linear(self.invariant_dim, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
        )
        self.odd_block_1 = EvenConditionedOddBlock(
            odd_input=self.odd_field_dim,
            even_width=width,
            width=width,
        )
        self.odd_block_2 = EvenConditionedOddBlock(
            odd_input=width,
            even_width=width,
            width=width,
            residual=True,
        )
        self.attention_score = nn.Sequential(
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, 1),
        )
        self.odd_readout = nn.Linear(width, 1, bias=False)

    @staticmethod
    def reflect(raw: Tensor) -> Tensor:
        """Swap C3/C4 for tensors ordered as ``(C3, Cz, C4)``."""

        if raw.ndim != 3 or raw.shape[1] != 3:
            raise ValueError("raw must have shape (batch, 3, time)")
        return raw[:, (2, 1, 0), :]

    def _local_mean(self, values: Tensor) -> Tensor:
        if values.ndim != 3:
            raise ValueError("spectral values must have shape (batch, filter, time)")
        batch, filters, n_times = values.shape
        pooled = nn.functional.avg_pool1d(
            values.reshape(batch * filters, 1, n_times),
            kernel_size=self.local_window,
            stride=self.local_stride,
        )
        return pooled.reshape(batch, filters, pooled.shape[-1])

    def _cross(
        self,
        first: tuple[Tensor, Tensor],
        second: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, Tensor]:
        """Local mean of ``first * conj(second)``."""

        first_real, first_imag = first
        second_real, second_imag = second
        real = self._local_mean(
            first_real * second_real + first_imag * second_imag
        )
        imag = self._local_mean(
            first_imag * second_real - first_real * second_imag
        )
        return real, imag

    def _normalized_cross(
        self,
        first: tuple[Tensor, Tensor],
        second: tuple[Tensor, Tensor],
        first_power: Tensor,
        second_power: Tensor,
    ) -> tuple[Tensor, Tensor]:
        real, imag = self._cross(first, second)
        denominator = torch.sqrt(
            (first_power * second_power).clamp_min(self.spectral_epsilon)
        )
        return real / denominator, imag / denominator

    @staticmethod
    def _global_mean(values: Tensor) -> Tensor:
        """Mean over the complete filtered epoch, retaining a token axis."""

        if values.ndim != 3:
            raise ValueError("spectral values must have shape (batch, filter, time)")
        return values.mean(dim=-1, keepdim=True)

    @staticmethod
    def _cross_from_mean(
        first: tuple[Tensor, Tensor],
        second: tuple[Tensor, Tensor],
        mean: Any,
    ) -> tuple[Tensor, Tensor]:
        """Return ``mean(first * conj(second))`` for a chosen token scale."""

        first_real, first_imag = first
        second_real, second_imag = second
        real = mean(first_real * second_real + first_imag * second_imag)
        imag = mean(first_imag * second_real - first_real * second_imag)
        return real, imag

    def _scale_tokens(
        self,
        *,
        even_h: tuple[Tensor, Tensor],
        centre_z: tuple[Tensor, Tensor],
        odd_h: tuple[Tensor, Tensor],
        c3: tuple[Tensor, Tensor],
        c4: tuple[Tensor, Tensor],
        mean: Any,
        frequency_coordinate: Tensor,
        time_coordinate: Tensor,
        scale_coordinate: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Construct invariant context and odd field at one temporal scale.

        ``mean`` is either local average pooling or a whole-epoch mean.  No
        learned operation occurs here, which keeps the transformation law
        auditable independently of downstream weights.
        """

        def power(pair: tuple[Tensor, Tensor]) -> Tensor:
            return mean(torch.square(pair[0]) + torch.square(pair[1])).clamp_min(
                self.spectral_epsilon
            )

        p_even = power(even_h)
        p_centre = power(centre_z)
        p_odd = power(odd_h)
        p_c3 = power(c3)
        p_c4 = power(c4)

        def normalized_cross(
            first: tuple[Tensor, Tensor],
            second: tuple[Tensor, Tensor],
            first_power: Tensor,
            second_power: Tensor,
        ) -> tuple[Tensor, Tensor]:
            real, imag = self._cross_from_mean(first, second, mean)
            denominator = torch.sqrt(
                (first_power * second_power).clamp_min(self.spectral_epsilon)
            )
            return real / denominator, imag / denominator

        even_centre = normalized_cross(even_h, centre_z, p_even, p_centre)
        odd_even = normalized_cross(odd_h, even_h, p_odd, p_even)
        odd_centre = normalized_cross(odd_h, centre_z, p_odd, p_centre)

        def magnitude(pair: tuple[Tensor, Tensor]) -> Tensor:
            return torch.sqrt(
                torch.square(pair[0])
                + torch.square(pair[1])
                + self.spectral_epsilon
            )

        invariant_context = torch.stack(
            (
                torch.log(p_even),
                torch.log(p_centre),
                torch.log(p_odd),
                magnitude(even_centre),
                magnitude(odd_even),
                magnitude(odd_centre),
                frequency_coordinate,
                time_coordinate,
                scale_coordinate,
            ),
            dim=-1,
        )

        # The log transform limits multiplicative amplitude excursions, and
        # tanh bounds the instantaneous contrast before any temporal moment is
        # formed.  u, u|u|, and u^3 are all odd, so their pooled values remain
        # exactly odd under the C3/C4 swap.  The last coordinate captures the
        # aggregate (rather than pointwise) log-power contrast at this scale.
        instantaneous_c3 = torch.square(c3[0]) + torch.square(c3[1])
        instantaneous_c4 = torch.square(c4[0]) + torch.square(c4[1])
        bounded_log_asymmetry = torch.tanh(
            0.5
            * (
                torch.log(instantaneous_c3 + self.spectral_epsilon)
                - torch.log(instantaneous_c4 + self.spectral_epsilon)
            )
        )
        asymmetry_first = mean(bounded_log_asymmetry)
        asymmetry_signed_second = mean(
            bounded_log_asymmetry * torch.abs(bounded_log_asymmetry)
        )
        asymmetry_third = mean(torch.pow(bounded_log_asymmetry, 3))
        pooled_log_asymmetry = torch.tanh(
            0.5 * (torch.log(p_c3) - torch.log(p_c4))
        )

        odd_field = torch.tanh(
            torch.stack(
                (
                    odd_even[0],
                    odd_even[1],
                    odd_centre[0],
                    odd_centre[1],
                    asymmetry_first,
                    asymmetry_signed_second,
                    asymmetry_third,
                    pooled_log_asymmetry,
                ),
                dim=-1,
            )
        )
        return invariant_context, odd_field

    def _parity_field(self, raw: Tensor) -> tuple[Tensor, Tensor]:
        if raw.ndim != 3 or raw.shape[1] != 3:
            raise ValueError("raw must have shape (batch, 3, time) in C3,Cz,C4 order")
        if raw.shape[2] != self.n_times:
            raise ValueError(f"expected {self.n_times} samples, got {raw.shape[2]}")

        scale = math.sqrt(0.5)
        even_hemisphere = (raw[:, 0] + raw[:, 2]) * scale
        centre = raw[:, 1]
        odd_hemisphere = (raw[:, 0] - raw[:, 2]) * scale
        # Filtering the physical C3/C4 signals in the same shared call makes
        # their exchange literal (not a numerically reconstructed identity).
        # This is what gives the added asymmetry moments bit-exact parity.
        parity_signals = torch.stack(
            (
                even_hemisphere,
                centre,
                odd_hemisphere,
                raw[:, 0],
                raw[:, 2],
            ),
            dim=1,
        )
        coefficients = self.filter_bank(parity_signals)
        even_h = (coefficients[:, 0, :, 0], coefficients[:, 0, :, 1])
        centre_z = (coefficients[:, 1, :, 0], coefficients[:, 1, :, 1])
        odd_h = (coefficients[:, 2, :, 0], coefficients[:, 2, :, 1])
        c3 = (coefficients[:, 3, :, 0], coefficients[:, 3, :, 1])
        c4 = (coefficients[:, 4, :, 0], coefficients[:, 4, :, 1])

        batch, _, filters, _, filtered_times = coefficients.shape
        local_times = self.local_token_count
        frequencies = self.filter_bank.frequencies_hz()
        midpoint = 0.5 * (
            self.filter_bank.frequency_low + self.filter_bank.frequency_high
        )
        half_span = 0.5 * (
            self.filter_bank.frequency_high - self.filter_bank.frequency_low
        )
        frequency_coordinate = ((frequencies - midpoint) / half_span).reshape(
            1, filters, 1
        )
        frequency_coordinate = frequency_coordinate.expand(
            batch, filters, local_times
        )
        time_coordinate = torch.linspace(
            -1.0,
            1.0,
            local_times,
            dtype=raw.dtype,
            device=raw.device,
        ).reshape(1, 1, local_times)
        time_coordinate = time_coordinate.expand(batch, filters, local_times)
        local_scale = math.log(self.local_window / filtered_times)
        local_scale_coordinate = torch.full_like(time_coordinate, local_scale)
        local_context, local_odd = self._scale_tokens(
            even_h=even_h,
            centre_z=centre_z,
            odd_h=odd_h,
            c3=c3,
            c4=c4,
            mean=self._local_mean,
            frequency_coordinate=frequency_coordinate,
            time_coordinate=time_coordinate,
            scale_coordinate=local_scale_coordinate,
        )

        global_frequency = frequency_coordinate[:, :, :1]
        global_time = torch.zeros_like(global_frequency)
        global_scale = torch.zeros_like(global_frequency)
        global_context, global_odd = self._scale_tokens(
            even_h=even_h,
            centre_z=centre_z,
            odd_h=odd_h,
            c3=c3,
            c4=c4,
            mean=self._global_mean,
            frequency_coordinate=global_frequency,
            time_coordinate=global_time,
            scale_coordinate=global_scale,
        )
        invariant_context = torch.cat((local_context, global_context), dim=2)
        odd_field = torch.cat((local_odd, global_odd), dim=2)
        return invariant_context, odd_field

    def extract_features(self, raw: Tensor) -> HemiQFieldFeatures:
        invariant_context, odd_field = self._parity_field(raw)
        even_hidden = self.context_encoder(invariant_context)
        hidden_odd = self.odd_block_1(odd_field, even_hidden)
        hidden_odd = self.odd_block_2(hidden_odd, even_hidden)

        batch = raw.shape[0]
        attention_logits = self.attention_score(even_hidden).reshape(batch, -1)
        attention = torch.softmax(attention_logits, dim=1)
        token_scores = self.odd_readout(hidden_odd).reshape(batch, -1)
        logit = torch.sum(attention * token_scores, dim=1)
        return HemiQFieldFeatures(
            logit=logit,
            invariant_context=invariant_context,
            odd_field=odd_field,
            hidden_odd=hidden_odd,
            attention=attention.reshape(invariant_context.shape[:-1]),
        )

    def forward(self, raw: Tensor) -> Tensor:
        return self.extract_features(raw).logit


@dataclass(frozen=True)
class HemiQFieldConfig:
    n_times: int = 251
    sfreq: float = SFREQ
    epochs: int = 240
    batch_size: int = 64
    learning_rate: float = 7e-4
    weight_decay: float = 5e-4
    patience: int = 40
    min_delta: float = 1e-4
    n_filters: int = 12
    kernel_size: int = 63
    temporal_stride: int = 2
    local_window: int = 25
    local_stride: int = 10
    width: int = 32
    frequency_low: float = 8.0
    frequency_high: float = 30.0
    bandwidth_low: float = 1.5
    bandwidth_high: float = 8.0
    spectral_epsilon: float = 1e-5
    teacher_weight: float = 0.25
    teacher_fraction: float = 0.5
    teacher_regularization: float = 1.0
    gradient_clip: float = 5.0
    seed: int = 7
    device: str = "cuda"
    deterministic: bool = True

    def __post_init__(self) -> None:
        positive_integers = (
            self.n_times,
            self.epochs,
            self.batch_size,
            self.patience,
            self.n_filters,
            self.kernel_size,
            self.temporal_stride,
            self.local_window,
            self.local_stride,
            self.width,
        )
        if any(value <= 0 for value in positive_integers):
            raise ValueError("all count and dimension settings must be positive")
        if self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if self.min_delta < 0.0 or self.gradient_clip <= 0.0:
            raise ValueError("min_delta must be non-negative and gradient_clip positive")
        if self.teacher_weight < 0.0:
            raise ValueError("teacher_weight must be non-negative")
        if not 0.0 < self.teacher_fraction <= 1.0:
            raise ValueError("teacher_fraction must be in (0, 1]")
        if self.teacher_regularization <= 0.0:
            raise ValueError("teacher_regularization must be positive")


class HemiQFieldClassifier:
    """Source-only preprocessing and validation-BCE fitting for HemiQ-FieldNet."""

    def __init__(self, config: HemiQFieldConfig = HemiQFieldConfig()) -> None:
        self.config = config

    @staticmethod
    def _validate_raw_labels(
        raw: NDArray[np.floating],
        labels: NDArray[np.integer],
        *,
        name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(raw)
        y = np.asarray(labels, dtype=np.int64)
        if values.ndim != 3:
            raise ValueError(f"{name} raw data must have shape (N, channels, time)")
        if len(values) != len(y):
            raise ValueError(f"{name} raw data and labels have inconsistent lengths")
        if set(np.unique(y).tolist()) != {0, 1}:
            raise ValueError(f"{name} split must contain labels 0 and 1")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} raw data contain non-finite values")
        return values, y

    @staticmethod
    def _channel_selection(channels: Sequence[str]) -> np.ndarray:
        names = tuple(channels)
        if len(set(names)) != len(names):
            raise ValueError("channel names must be unique")
        missing = [name for name in HEMI_Q_CHANNELS if name not in names]
        if missing:
            raise ValueError(f"missing HemiQ channels: {missing}")
        return np.asarray([names.index(name) for name in HEMI_Q_CHANNELS], dtype=np.int64)

    def _select_raw(self, raw: NDArray[np.floating]) -> np.ndarray:
        values = np.asarray(raw, dtype=np.float32)
        if values.ndim != 3 or values.shape[1] != len(self.channels_):
            raise ValueError(
                f"raw must have shape (N, {len(self.channels_)}, {self.config.n_times})"
            )
        if values.shape[2] != self.config.n_times:
            raise ValueError(
                f"expected {self.config.n_times} time samples, got {values.shape[2]}"
            )
        return values[:, self.channel_selection_, :]

    @staticmethod
    def _select_covariances(
        covariances: NDArray[np.floating],
        channel_selection: NDArray[np.int64],
        *,
        n_rows: int,
        n_channels: int,
    ) -> np.ndarray:
        values = np.asarray(covariances)
        if (
            values.ndim != 4
            or values.shape[-2:] != (n_channels, n_channels)
            or len(values) != n_rows
        ):
            raise ValueError(
                "teacher covariances must have shape (N, bands, channels, channels)"
            )
        return values[..., channel_selection, :][..., :, channel_selection]

    def _fit_raw_scaler(self, selected_source: NDArray[np.floating]) -> None:
        source = np.asarray(selected_source, dtype=np.float32)
        # Pool C3 and C4 explicitly.  The resulting statistics are bitwise
        # identical for the reflected channel pair and see source rows only.
        hemispheric = source[:, (0, 2), :].reshape(-1).astype(np.float64)
        centre = source[:, 1, :].reshape(-1).astype(np.float64)
        hemispheric_mean = float(hemispheric.mean())
        centre_mean = float(centre.mean())
        hemispheric_std = float(hemispheric.std()) + 1e-6
        centre_std = float(centre.std()) + 1e-6
        self.raw_mean_ = np.asarray(
            [[[hemispheric_mean], [centre_mean], [hemispheric_mean]]],
            dtype=np.float32,
        )
        self.raw_std_ = np.asarray(
            [[[hemispheric_std], [centre_std], [hemispheric_std]]],
            dtype=np.float32,
        )

    def _prepare_raw(self, raw: NDArray[np.floating]) -> np.ndarray:
        selected = self._select_raw(raw)
        return ((selected - self.raw_mean_) / self.raw_std_).astype(
            np.float32, copy=False
        )

    def _model_kwargs(self) -> dict[str, Any]:
        config = self.config
        return {
            "n_times": config.n_times,
            "sfreq": config.sfreq,
            "n_filters": config.n_filters,
            "kernel_size": config.kernel_size,
            "temporal_stride": config.temporal_stride,
            "local_window": config.local_window,
            "local_stride": config.local_stride,
            "width": config.width,
            "frequency_low": config.frequency_low,
            "frequency_high": config.frequency_high,
            "bandwidth_low": config.bandwidth_low,
            "bandwidth_high": config.bandwidth_high,
            "spectral_epsilon": config.spectral_epsilon,
        }

    def fit(
        self,
        raw_train: NDArray[np.floating],
        cov_train: NDArray[np.floating] | None,
        y_train: NDArray[np.integer],
        raw_validation: NDArray[np.floating],
        cov_validation: NDArray[np.floating] | None,
        y_validation: NDArray[np.integer],
        *,
        channels: Sequence[str] = CHANNELS,
    ) -> "HemiQFieldClassifier":
        """Fit on source rows; ``cov_validation`` is intentionally never read."""

        del cov_validation  # The optional teacher is strictly source-only.
        train_values, y_train_array = self._validate_raw_labels(
            raw_train, y_train, name="training"
        )
        validation_values, y_validation_array = self._validate_raw_labels(
            raw_validation, y_validation, name="validation"
        )
        if train_values.shape[1:] != validation_values.shape[1:]:
            raise ValueError("training and validation raw shapes disagree")
        if train_values.shape[2] != self.config.n_times:
            raise ValueError(
                f"configured n_times={self.config.n_times}, got {train_values.shape[2]}"
            )
        self.channels_ = tuple(channels)
        if len(self.channels_) != train_values.shape[1]:
            raise ValueError("channel-name count does not match raw data")
        self.channel_selection_ = self._channel_selection(self.channels_)

        config = self.config
        _configure_torch_determinism(config.deterministic)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        self.device_ = _resolve_device(config.device)
        started = time.perf_counter()

        selected_train = np.asarray(train_values, dtype=np.float32)[
            :, self.channel_selection_, :
        ]
        self._fit_raw_scaler(selected_train)
        prepared_train = self._prepare_raw(train_values)
        prepared_validation = self._prepare_raw(validation_values)

        teacher_logits: np.ndarray | None = None
        self.teacher_was_used_ = bool(config.teacher_weight > 0.0)
        self.teacher_param_count_ = 0
        if self.teacher_was_used_:
            if cov_train is None:
                raise ValueError("cov_train is required when teacher_weight is positive")
            selected_covariances = self._select_covariances(
                cov_train,
                self.channel_selection_,
                n_rows=len(train_values),
                n_channels=len(self.channels_),
            )
            mirror_index = np.asarray((2, 1, 0), dtype=np.int64)
            teacher = FrozenTangentAnchor(
                regularization=config.teacher_regularization
            ).fit(selected_covariances, y_train_array, mirror_index)
            reflected_covariances = _mirror_covariances(
                selected_covariances, mirror_index
            )
            # Projection removes any finite-sample even component of the convex
            # teacher and makes its soft target respect the same signed action.
            teacher_logits = 0.5 * (
                teacher.signed_logit(selected_covariances)
                - teacher.signed_logit(reflected_covariances)
            )
            if hasattr(teacher, "model_"):
                self.teacher_param_count_ = int(
                    teacher.model_.coef_.size + teacher.model_.intercept_.size
                )

        train_tensor = torch.from_numpy(prepared_train).to(self.device_)
        validation_tensor = torch.from_numpy(prepared_validation).to(self.device_)
        y_tr = torch.from_numpy(y_train_array).to(self.device_)
        y_va = torch.from_numpy(y_validation_array).to(self.device_)
        teacher_tensor = (
            None
            if teacher_logits is None
            else torch.from_numpy(teacher_logits.astype(np.float32)).to(self.device_)
        )

        self.model_kwargs_ = self._model_kwargs()
        self.model_ = HemiQFieldNet(**self.model_kwargs_).to(self.device_)
        self.param_count_ = sum(parameter.numel() for parameter in self.model_.parameters())
        self.trainable_param_count_ = sum(
            parameter.numel()
            for parameter in self.model_.parameters()
            if parameter.requires_grad
        )
        if self.trainable_param_count_ >= 50_000:
            raise RuntimeError(
                "HemiQ-FieldNet exceeded its 50k trainable-parameter contract: "
                f"{self.trainable_param_count_}"
            )
        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        generator = torch.Generator(device=self.device_).manual_seed(config.seed + 1)

        # The untrained network is a legitimate neural checkpoint.  No teacher,
        # accuracy metric, threshold, or alternative output participates in
        # checkpoint selection: the sole criterion is validation BCE.
        self.model_.eval()
        with torch.no_grad():
            initial_validation_logit = self.model_(validation_tensor)
            best_loss = float(_binary_loss(initial_validation_logit, y_va))
            initial_prediction = (initial_validation_logit >= 0.0).long()
            best_balanced = float(
                balanced_accuracy_score(
                    y_validation_array, initial_prediction.cpu().numpy()
                )
            )
        best_state = copy.deepcopy(self.model_.state_dict())
        best_epoch = -1
        stale = 0
        self.history_: list[dict[str, float]] = [
            {
                "epoch": -1.0,
                "train_loss": float("nan"),
                "validation_loss": best_loss,
                "validation_balanced_accuracy": best_balanced,
                "teacher_coefficient": 0.0,
            }
        ]
        teacher_decay_epochs = max(
            1, int(math.ceil(config.epochs * config.teacher_fraction))
        )

        for epoch in range(config.epochs):
            self.model_.train()
            order = torch.randperm(
                len(y_tr), device=self.device_, generator=generator
            )
            teacher_coefficient = config.teacher_weight * max(
                0.0, 1.0 - epoch / teacher_decay_epochs
            )
            total_loss = 0.0
            for start in range(0, len(order), config.batch_size):
                rows = order[start : start + config.batch_size]
                logit = self.model_(train_tensor[rows])
                loss = _binary_loss(logit, y_tr[rows])
                if teacher_tensor is not None and teacher_coefficient > 0.0:
                    soft_target = torch.sigmoid(teacher_tensor[rows])
                    loss = loss + teacher_coefficient * _binary_loss(
                        logit, soft_target
                    )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model_.parameters(), config.gradient_clip)
                optimizer.step()
                total_loss += float(loss.detach()) * len(rows)

            self.model_.eval()
            with torch.no_grad():
                validation_logit = self.model_(validation_tensor)
                validation_loss = float(_binary_loss(validation_logit, y_va))
                validation_prediction = (validation_logit >= 0.0).long()
                validation_balanced = float(
                    balanced_accuracy_score(
                        y_validation_array, validation_prediction.cpu().numpy()
                    )
                )
            self.history_.append(
                {
                    "epoch": float(epoch),
                    "train_loss": total_loss / max(1, len(y_tr)),
                    "validation_loss": validation_loss,
                    "validation_balanced_accuracy": validation_balanced,
                    "teacher_coefficient": float(teacher_coefficient),
                }
            )
            if validation_loss < best_loss - config.min_delta:
                best_loss = validation_loss
                best_balanced = validation_balanced
                best_epoch = epoch
                best_state = copy.deepcopy(self.model_.state_dict())
                stale = 0
            else:
                stale += 1
                if stale >= config.patience:
                    break

        self.model_.load_state_dict(best_state)
        self.model_.eval()
        self.best_epoch_ = int(best_epoch)
        self.epochs_run_ = len(self.history_) - 1
        self.best_validation_loss_ = float(best_loss)
        self.best_validation_balanced_accuracy_ = float(best_balanced)
        self.train_seconds_ = time.perf_counter() - started
        self.decision_threshold_ = 0.0
        self.config_ = asdict(config)
        return self

    def _check_fitted(self) -> None:
        if not hasattr(self, "model_"):
            raise RuntimeError("fit must be called before prediction")

    def decision_function(self, raw: NDArray[np.floating]) -> np.ndarray:
        self._check_fitted()
        prepared = self._prepare_raw(raw)
        results: list[np.ndarray] = []
        self.model_.eval()
        with torch.no_grad():
            for start in range(0, len(prepared), 256):
                tensor = torch.from_numpy(prepared[start : start + 256]).to(
                    self.device_
                )
                results.append(self.model_(tensor).cpu().numpy())
        if not results:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(results).astype(np.float32, copy=False)

    def predict_proba(self, raw: NDArray[np.floating]) -> np.ndarray:
        logit = self.decision_function(raw)
        positive = 1.0 / (1.0 + np.exp(-np.clip(logit, -40.0, 40.0)))
        return np.column_stack((1.0 - positive, positive))

    def predict(self, raw: NDArray[np.floating]) -> np.ndarray:
        return (self.decision_function(raw) >= self.decision_threshold_).astype(
            np.int64
        )

    def max_equivariance_error(self, raw: NDArray[np.floating]) -> float:
        self._check_fitted()
        prepared = self._prepare_raw(raw)
        tensor = torch.from_numpy(prepared).to(self.device_)
        self.model_.eval()
        with torch.no_grad():
            direct = self.model_(tensor)
            reflected = self.model_(HemiQFieldNet.reflect(tensor))
        if len(direct) == 0:
            return 0.0
        return float(torch.max(torch.abs(direct + reflected)).cpu())

    def filter_diagnostics(self) -> dict[str, list[float]]:
        self._check_fitted()
        return {
            "frequencies_hz": self.model_.filter_bank.frequencies_hz()
            .detach()
            .cpu()
            .tolist(),
            "bandwidths_hz": self.model_.filter_bank.bandwidths_hz()
            .detach()
            .cpu()
            .tolist(),
        }


__all__ = [
    "ConstrainedQuadratureGaborBank",
    "EvenConditionedOddBlock",
    "HEMI_Q_CHANNELS",
    "HemiQFieldClassifier",
    "HemiQFieldConfig",
    "HemiQFieldFeatures",
    "HemiQFieldNet",
]
