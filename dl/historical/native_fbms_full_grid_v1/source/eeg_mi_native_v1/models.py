"""Experimental neural architectures for montage-heterogeneous MI decoding.

The lead architecture, :class:`ScopeNet`, does not assign learned parameters to
channel indices. Instead it evaluates smooth, frequency-conditioned spatial
fields at the supplied electrode coordinates and approximates their scalp
integrals with geometry-derived quadrature weights. Consequently a single
backbone accepts different channel counts and is invariant to channel order.

The temporal and readout path deliberately retains the log-energy mechanism
that is consistently strong for sensorimotor rhythms. This is not a generic
Transformer: ordered physical-frequency filters, continuous scalp fields,
multi-resolution energy/dynamics, and an axial convolutional mixer are the
model's defining operations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor, nn


# Unit vectors for the fixed 21-electrode sensorimotor atlas used by the
# common-montage protocol.  They are standard_1005 head coordinates, not
# estimates from a held-out participant.  A cardinal radial basis evaluated at
# these anchors is exactly channel-indexed on the atlas yet remains a continuous
# function that can be sampled at a different montage.
CANONICAL_21_POSITIONS = (
    (-0.008863090, 0.672512174, 0.740033031),
    (-0.490799546, 0.453353763, 0.744033754),
    (0.482264251, 0.459051639, 0.746118486),
    (-0.259081185, 0.447670907, 0.855843246),
    (0.243200347, 0.453856409, 0.857244372),
    (-0.751242995, 0.176574886, 0.635968029),
    (0.749052346, 0.180679828, 0.637397349),
    (-0.531238556, 0.184795663, 0.826822937),
    (0.523457944, 0.188883215, 0.830851912),
    (-0.275423110, 0.191206098, 0.942115903),
    (0.264082938, 0.193931669, 0.944801927),
    (-0.009616030, 0.193261296, 0.981100202),
    (-0.516889632, -0.093921199, 0.850884199),
    (0.512567878, -0.095002279, 0.853374898),
    (-0.269217432, -0.077864096, 0.959926665),
    (0.263983428, -0.078921460, 0.961292982),
    (-0.010742731, -0.074100412, 0.997192919),
    (-0.233516648, -0.342253029, 0.910127938),
    (0.225912407, -0.340402722, 0.912737429),
    (-0.012708604, -0.336128980, 0.941730201),
    (-0.016152671, -0.578608930, 0.815445125),
)

# Union atlas spanning the common public montage and every additional electrode
# in the project's 15-channel deployment montage (T5/T6, F7/F8, F3/F4,
# T3/T4, P3/P4).  The ordering is fixed and derived from standard_1005.
EXTENDED_31_POSITIONS = CANONICAL_21_POSITIONS + (
    # These are normalized MNE head-frame ``info['chs'].loc`` coordinates,
    # matching both the local FIF recordings and the first 21 anchors above.
    # Do not substitute normalized raw montage ``ch_pos`` values: those use a
    # different origin and silently corrupt the geometry.
    (-0.783904334825, -0.443478838280, 0.434534824657),
    (0.769150510786, -0.457597514611, 0.446107617487),
    (-0.679913644766, 0.691519471553, 0.243963636890),
    (0.672480568422, 0.701379729219, 0.236297186897),
    (-0.404623827681, 0.677024521875, 0.614749831112),
    (0.395679829501, 0.688177104733, 0.608152731678),
    (-0.927931611701, 0.160496972261, 0.336427772191),
    (0.923725109685, 0.169278653344, 0.343622844497),
    (-0.449902040176, -0.361390720831, 0.816691435698),
    (0.438728944768, -0.362645096252, 0.822195504237),
)


class OrderedSincFilterBank(nn.Module):
    """Strictly ordered, bounded physical-frequency band-pass filters."""

    def __init__(
        self,
        *,
        n_filters: int = 16,
        kernel_size: int = 65,
        sfreq: float = 128.0,
        frequency_low: float = 4.0,
        frequency_high: float = 40.0,
        bandwidth_low: float = 1.5,
        bandwidth_high: float = 10.0,
        stride: int = 2,
    ) -> None:
        super().__init__()
        if n_filters <= 0:
            raise ValueError("n_filters must be positive")
        if kernel_size <= 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer greater than one")
        if not 0.0 < frequency_low < frequency_high < sfreq / 2.0:
            raise ValueError("frequency interval must lie inside Nyquist")
        if not 0.0 < bandwidth_low < bandwidth_high:
            raise ValueError("bandwidth bounds must be positive and ordered")
        self.n_filters = int(n_filters)
        self.kernel_size = int(kernel_size)
        self.sfreq = float(sfreq)
        self.frequency_low = float(frequency_low)
        self.frequency_high = float(frequency_high)
        self.bandwidth_low = float(bandwidth_low)
        self.bandwidth_high = float(bandwidth_high)
        self.stride = int(stride)

        self.raw_frequency_gaps = nn.Parameter(torch.zeros(n_filters + 1))
        initial_bandwidth = min(max(4.0, bandwidth_low + 1e-3), bandwidth_high - 1e-3)
        fraction = (initial_bandwidth - bandwidth_low) / (bandwidth_high - bandwidth_low)
        self.raw_bandwidths = nn.Parameter(
            torch.full((n_filters,), math.log(fraction / (1.0 - fraction)))
        )

    def frequencies_hz(self) -> Tensor:
        gaps = nn.functional.softplus(self.raw_frequency_gaps) + 1e-4
        fractions = torch.cumsum(gaps[:-1], dim=0) / gaps.sum()
        return self.frequency_low + (self.frequency_high - self.frequency_low) * fractions

    def bandwidths_hz(self) -> Tensor:
        return self.bandwidth_low + (self.bandwidth_high - self.bandwidth_low) * torch.sigmoid(
            self.raw_bandwidths
        )

    def kernels(self) -> Tensor:
        centers = self.frequencies_hz()
        bandwidths = self.bandwidths_hz()
        low = torch.clamp(centers - 0.5 * bandwidths, min=0.5)
        high = torch.clamp(centers + 0.5 * bandwidths, max=self.sfreq / 2.0 - 0.5)
        positions = torch.arange(
            -(self.kernel_size // 2),
            self.kernel_size // 2 + 1,
            device=centers.device,
            dtype=centers.dtype,
        )
        low_pass_high = (2.0 * high[:, None] / self.sfreq) * torch.sinc(
            2.0 * high[:, None] * positions[None, :] / self.sfreq
        )
        low_pass_low = (2.0 * low[:, None] / self.sfreq) * torch.sinc(
            2.0 * low[:, None] * positions[None, :] / self.sfreq
        )
        window = torch.hamming_window(
            self.kernel_size,
            periodic=False,
            dtype=centers.dtype,
            device=centers.device,
        )
        kernels = (low_pass_high - low_pass_low) * window[None, :]
        kernels = kernels - kernels.mean(dim=1, keepdim=True)
        kernels = kernels / torch.linalg.vector_norm(kernels, dim=1, keepdim=True).clamp_min(
            1e-7
        )
        return kernels[:, None, :]

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError("x must have shape (batch, channels, time)")
        batch, channels, n_times = x.shape
        filtered = nn.functional.conv1d(
            x.reshape(batch * channels, 1, n_times),
            self.kernels(),
            stride=self.stride,
            padding=self.kernel_size // 2,
        )
        return filtered.reshape(batch, channels, self.n_filters, filtered.shape[-1])


class AdaptiveSincFilterBank(OrderedSincFilterBank):
    """Sinc-initialized filter bank with a fully learnable zero-DC FIR residual.

    The ordered physical-frequency filters provide a stable initialization and
    interpretable nominal center frequencies.  The residual makes the class
    expressive enough to recover an unconstrained temporal convolution when a
    dataset's useful passbands are not well represented by ideal sinc kernels.
    """

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.fir_residual = nn.Parameter(
            torch.zeros(self.n_filters, 1, self.kernel_size)
        )

    def kernels(self) -> Tensor:
        base = super().kernels()
        residual = self.fir_residual - self.fir_residual.mean(dim=-1, keepdim=True)
        kernels = base + residual
        kernels = kernels - kernels.mean(dim=-1, keepdim=True)
        return kernels / torch.linalg.vector_norm(
            kernels, dim=-1, keepdim=True
        ).clamp_min(1e-7)


class LearnableGaborFIRBank(nn.Module):
    """Compact zero-DC FIR bank initialized at sensorimotor frequencies.

    Unlike :class:`OrderedSincFilterBank`, this branch deliberately imposes no
    ordering constraint after initialization.  Each short filter is a free FIR
    whose zero-mean and unit-norm projection is applied on every forward pass.
    The Gabor initialization covers mu/beta and low-gamma rhythms without
    duplicating FBCNet's longer fixed filter bank.
    """

    def __init__(
        self,
        *,
        n_filters: int = 8,
        kernel_size: int = 25,
        sfreq: float = 128.0,
        frequency_low: float = 6.0,
        frequency_high: float = 36.0,
    ) -> None:
        super().__init__()
        if n_filters <= 0:
            raise ValueError("n_filters must be positive")
        if kernel_size <= 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer greater than one")
        if not 0.0 < frequency_low < frequency_high < sfreq / 2.0:
            raise ValueError("frequency interval must lie inside Nyquist")
        self.n_filters = int(n_filters)
        self.kernel_size = int(kernel_size)
        self.sfreq = float(sfreq)
        centers = torch.linspace(frequency_low, frequency_high, n_filters)
        self.register_buffer("initial_frequencies_hz", centers)

        samples = torch.arange(
            -(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32
        )
        seconds = samples / self.sfreq
        # A compact Gaussian taper avoids a hard truncation while preserving
        # enough cycles for the lowest initialized center frequency.
        sigma_seconds = (kernel_size / self.sfreq) / 4.0
        envelope = torch.exp(-0.5 * (seconds / sigma_seconds).square())
        kernels = envelope[None, :] * torch.cos(
            2.0 * math.pi * centers[:, None] * seconds[None, :]
        )
        kernels = kernels - kernels.mean(dim=-1, keepdim=True)
        kernels = kernels / torch.linalg.vector_norm(
            kernels, dim=-1, keepdim=True
        ).clamp_min(1e-7)
        self.raw_kernels = nn.Parameter(kernels[:, None, :])

    def kernels(self) -> Tensor:
        kernels = self.raw_kernels - self.raw_kernels.mean(dim=-1, keepdim=True)
        return kernels / torch.linalg.vector_norm(
            kernels, dim=-1, keepdim=True
        ).clamp_min(1e-7)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError("x must have shape (batch, channels, time)")
        batch, channels, n_times = x.shape
        filtered = nn.functional.conv1d(
            x.reshape(batch * channels, 1, n_times),
            self.kernels(),
            padding=self.kernel_size // 2,
        )
        return filtered.reshape(
            batch, channels, self.n_filters, filtered.shape[-1]
        )


class LowRankSpectralTemporalResidual(nn.Module):
    """Tiny multiscale correction that preserves the physical filter floor.

    A rank-four filter projection is convolved at four physical scales and
    projected back to the original ordered bands.  Near-zero residual gates
    make the initial mapping effectively the identity instead of replacing the
    successful Sinc/cardinal representation.
    """

    def __init__(
        self,
        *,
        n_filters: int,
        kernel_sizes: tuple[int, ...] = (7, 15, 31, 63),
        gate_initial: float = 1e-3,
    ) -> None:
        super().__init__()
        if not kernel_sizes or any(k <= 1 or k % 2 == 0 for k in kernel_sizes):
            raise ValueError("residual kernels must be odd integers greater than one")
        rank = len(kernel_sizes)
        self.down = nn.Parameter(torch.empty(rank, n_filters))
        self.up = nn.Parameter(torch.empty(n_filters, rank))
        nn.init.orthogonal_(self.down)
        nn.init.orthogonal_(self.up)
        self.kernels = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, 1, size)) for size in kernel_sizes]
        )
        for kernel in self.kernels:
            center = kernel.shape[-1] // 2
            with torch.no_grad():
                kernel[..., center] = 1.0
        bounded_fraction = max(min(float(gate_initial) / 0.25, 0.999), -0.999)
        self.raw_gates = nn.Parameter(
            torch.full((rank,), math.atanh(bounded_fraction))
        )

    def forward(self, filtered: Tensor) -> Tensor:
        if filtered.ndim != 4:
            raise ValueError("filtered must have shape (batch, channels, filters, time)")
        batch, channels, _, n_times = filtered.shape
        down = self.down / torch.linalg.vector_norm(
            self.down, dim=1, keepdim=True
        ).clamp_min(1e-7)
        up = self.up / torch.linalg.vector_norm(
            self.up, dim=0, keepdim=True
        ).clamp_min(1e-7)
        latent = torch.einsum("rf,bcft->bcrt", down, filtered)
        convolved: list[Tensor] = []
        for index, kernel in enumerate(self.kernels):
            normalized_kernel = kernel - kernel.mean(dim=-1, keepdim=True)
            normalized_kernel = normalized_kernel / torch.linalg.vector_norm(
                normalized_kernel, dim=-1, keepdim=True
            ).clamp_min(1e-7)
            values = nn.functional.conv1d(
                latent[:, :, index, :].reshape(batch * channels, 1, n_times),
                normalized_kernel,
                padding=kernel.shape[-1] // 2,
            ).reshape(batch, channels, n_times)
            convolved.append(values)
        gates = 0.25 * torch.tanh(self.raw_gates)
        residual = torch.stack(convolved, dim=2) * gates[None, None, :, None]
        return filtered + torch.einsum("fr,bcrt->bcft", up, residual)


def spherical_polynomial_basis(positions: Tensor) -> Tensor:
    """Return the real degree-0..3 spherical polynomial basis (16 modes)."""

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (channels, 3)")
    norm = torch.linalg.vector_norm(positions, dim=1, keepdim=True).clamp_min(1e-8)
    x, y, z = (positions / norm).unbind(dim=1)
    one = torch.ones_like(x)
    columns = (
        one,
        x,
        y,
        z,
        x * y,
        y * z,
        z * x,
        x.square() - y.square(),
        3.0 * z.square() - one,
        y * (3.0 * x.square() - y.square()),
        x * y * z,
        y * (5.0 * z.square() - one),
        z * (x.square() - y.square()),
        x * (5.0 * z.square() - one),
        x * (x.square() - 3.0 * y.square()),
        z * (5.0 * z.square() - 3.0),
    )
    return torch.stack(columns, dim=1)


def scalp_quadrature_weights(positions: Tensor) -> Tensor:
    """Approximate sensor Voronoi areas from local angular spacing."""

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (channels, 3)")
    channels = positions.shape[0]
    if channels == 1:
        return torch.ones(1, dtype=positions.dtype, device=positions.device)
    unit = positions / torch.linalg.vector_norm(positions, dim=1, keepdim=True).clamp_min(1e-8)
    cosine = torch.clamp(unit @ unit.T, -1.0, 1.0)
    angular = torch.acos(cosine)
    angular = angular + torch.eye(channels, device=positions.device, dtype=positions.dtype) * 10.0
    neighbours = min(3, channels - 1)
    local = torch.topk(angular, k=neighbours, dim=1, largest=False).values.mean(dim=1)
    areas = local.square().clamp_min(1e-4)
    return areas / areas.sum()


class ContinuousScalpField(nn.Module):
    """Frequency-conditioned smooth spatial filters on an irregular montage."""

    basis_dim = 16

    def __init__(
        self,
        *,
        n_filters: int,
        n_sources: int = 8,
        frequency_polynomial_degree: int = 2,
    ) -> None:
        super().__init__()
        self.n_filters = int(n_filters)
        self.n_sources = int(n_sources)
        self.frequency_polynomial_degree = int(frequency_polynomial_degree)
        self.coefficients = nn.Parameter(
            torch.empty(frequency_polynomial_degree + 1, n_sources, self.basis_dim)
        )
        nn.init.xavier_uniform_(self.coefficients, gain=0.6)

    def _fields(self, positions: Tensor, normalized_frequencies: Tensor) -> Tensor:
        basis = spherical_polynomial_basis(positions)
        powers = torch.stack(
            [normalized_frequencies.pow(degree) for degree in range(self.frequency_polynomial_degree + 1)],
            dim=1,
        )
        coefficients = torch.einsum("fd,dsq->fsq", powers, self.coefficients)
        return torch.einsum("fsq,cq->fsc", coefficients, basis)

    def forward(
        self,
        filtered: Tensor,
        positions: Tensor,
        normalized_frequencies: Tensor,
        channel_mask: Tensor | None = None,
    ) -> Tensor:
        if filtered.ndim != 4:
            raise ValueError("filtered must have shape (batch, channels, filters, time)")
        if positions.shape != (filtered.shape[1], 3):
            raise ValueError("positions do not match the filtered channel axis")
        field = self._fields(positions.to(filtered), normalized_frequencies.to(filtered))
        quadrature = scalp_quadrature_weights(positions.to(filtered))
        weighted = field * quadrature[None, None, :]
        if channel_mask is None:
            weighted = weighted / torch.linalg.vector_norm(
                weighted, dim=-1, keepdim=True
            ).clamp_min(1e-7)
            return torch.einsum("fsc,bcft->bsft", weighted, filtered)
        if channel_mask.shape != filtered.shape[:2]:
            raise ValueError("channel_mask must have shape (batch, channels)")
        batch_weighted = weighted[None, :, :, :] * channel_mask[:, None, None, :].to(
            filtered.dtype
        )
        batch_weighted = batch_weighted / torch.linalg.vector_norm(
            batch_weighted, dim=-1, keepdim=True
        ).clamp_min(1e-7)
        return torch.einsum("bfsc,bcft->bsft", batch_weighted, filtered)


class FreeSpatialFilter(nn.Module):
    """Channel-indexed control with the same downstream energy decoder."""

    def __init__(self, *, n_filters: int, n_sources: int, n_channels: int) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.empty(n_filters, n_sources, n_channels))
        nn.init.xavier_uniform_(self.weights, gain=0.7)

    def forward(self, filtered: Tensor) -> Tensor:
        weights = self.weights / torch.linalg.vector_norm(
            self.weights, dim=-1, keepdim=True
        ).clamp_min(1e-7)
        return torch.einsum("fsc,bcft->bsft", weights, filtered)


def spherical_gaussian_kernel(first: Tensor, second: Tensor, length_scale: float) -> Tensor:
    """Gaussian radial kernel of great-circle distance on the unit scalp."""

    first_unit = first / torch.linalg.vector_norm(first, dim=1, keepdim=True).clamp_min(1e-8)
    second_unit = second / torch.linalg.vector_norm(second, dim=1, keepdim=True).clamp_min(1e-8)
    cosine = torch.clamp(first_unit @ second_unit.T, -1.0 + 1e-6, 1.0 - 1e-6)
    angle = torch.acos(cosine)
    return torch.exp(-0.5 * (angle / length_scale).square())


class CardinalScalpField(nn.Module):
    """Full-rank continuous spatial fields induced from a fixed scalp atlas.

    The regularized cardinal basis is approximately the identity at the 21
    inducing electrodes.  The layer therefore contains an ordinary free
    per-band spatial filter on the common montage, while interpolation to any
    supplied coordinates is defined without channel names or channel order.
    """

    def __init__(
        self,
        *,
        n_filters: int,
        n_sources: int,
        length_scale: float = 0.35,
        ridge: float = 1e-4,
        anchor_positions: tuple[tuple[float, float, float], ...] = CANONICAL_21_POSITIONS,
        field_norm: str = "unit",
        max_norm: float = 2.0,
    ) -> None:
        super().__init__()
        if field_norm not in {"unit", "max", "none"}:
            raise ValueError("field_norm must be one of: unit, max, none")
        if max_norm <= 0.0:
            raise ValueError("max_norm must be positive")
        anchors = torch.tensor(anchor_positions, dtype=torch.float32)
        anchor_kernel = spherical_gaussian_kernel(anchors, anchors, length_scale)
        regularized = anchor_kernel + ridge * torch.eye(len(anchors))
        self.register_buffer("anchors", anchors)
        self.register_buffer("cardinal_inverse", torch.linalg.inv(regularized))
        self.length_scale = float(length_scale)
        self.field_norm = field_norm
        self.max_norm = float(max_norm)
        self.coefficients = nn.Parameter(
            torch.empty(n_filters, n_sources, len(anchor_positions))
        )
        nn.init.xavier_uniform_(self.coefficients, gain=0.7)

    def basis(self, positions: Tensor) -> Tensor:
        kernel = spherical_gaussian_kernel(
            positions.to(self.anchors), self.anchors, self.length_scale
        )
        return kernel @ self.cardinal_inverse

    def _apply_field_norm(self, fields: Tensor) -> Tensor:
        if self.field_norm == "none":
            return fields
        if self.field_norm == "unit":
            return fields / torch.linalg.vector_norm(
                fields, dim=-1, keepdim=True
            ).clamp_min(1e-7)
        n_channels = fields.shape[-1]
        # Conv2dWithConstraint uses the same row-wise p=2 renormalization.
        # Values below the bound are untouched, unlike unit normalization.
        return fields.reshape(-1, n_channels).renorm(
            p=2, dim=0, maxnorm=self.max_norm
        ).reshape_as(fields)

    def forward(
        self,
        filtered: Tensor,
        positions: Tensor,
        channel_mask: Tensor | None = None,
    ) -> Tensor:
        if filtered.ndim != 4:
            raise ValueError("filtered must have shape (batch, channels, filters, time)")
        if positions.shape != (filtered.shape[1], 3):
            raise ValueError("positions do not match the filtered channel axis")
        basis = self.basis(positions).to(filtered)
        fields = torch.einsum("fsa,ca->fsc", self.coefficients, basis)
        if channel_mask is None:
            fields = self._apply_field_norm(fields)
            return torch.einsum("fsc,bcft->bsft", fields, filtered)
        if channel_mask.shape != filtered.shape[:2]:
            raise ValueError("channel_mask must have shape (batch, channels)")
        batch_fields = fields[None] * channel_mask[:, None, None, :].to(filtered.dtype)
        batch_fields = self._apply_field_norm(batch_fields)
        return torch.einsum("bfsc,bcft->bsft", batch_fields, filtered)


class JointCardinalScalpField(CardinalScalpField):
    """Tensor-product cardinal operator over scalp position and frequency."""

    def __init__(
        self,
        *,
        n_filters: int,
        n_sources: int,
        frequency_low: float = 4.0,
        frequency_high: float = 40.0,
        frequency_length_scale: float = 4.0,
        **kwargs: float,
    ) -> None:
        super().__init__(n_filters=n_filters, n_sources=n_sources, **kwargs)
        anchors = torch.linspace(
            frequency_low,
            frequency_high,
            n_filters + 2,
            dtype=torch.float32,
        )[1:-1]
        kernel = self._frequency_kernel(anchors, anchors, frequency_length_scale)
        ridge = float(kwargs.get("ridge", 1e-4))
        self.register_buffer("frequency_anchors", anchors)
        self.register_buffer(
            "frequency_cardinal_inverse",
            torch.linalg.inv(kernel + ridge * torch.eye(n_filters)),
        )
        self.frequency_length_scale = float(frequency_length_scale)

    @staticmethod
    def _frequency_kernel(first: Tensor, second: Tensor, length_scale: float) -> Tensor:
        return torch.exp(
            -0.5 * ((first[:, None] - second[None, :]) / length_scale).square()
        )

    def frequency_basis(self, frequencies_hz: Tensor) -> Tensor:
        kernel = self._frequency_kernel(
            frequencies_hz.to(self.frequency_anchors),
            self.frequency_anchors,
            self.frequency_length_scale,
        )
        return kernel @ self.frequency_cardinal_inverse

    def forward(
        self,
        filtered: Tensor,
        positions: Tensor,
        frequencies_hz: Tensor,
        channel_mask: Tensor | None = None,
    ) -> Tensor:
        if filtered.ndim != 4:
            raise ValueError("filtered must have shape (batch, channels, filters, time)")
        spatial_basis = self.basis(positions).to(filtered)
        frequency_basis = self.frequency_basis(frequencies_hz).to(filtered)
        coefficients = torch.einsum(
            "fj,jsa->fsa", frequency_basis, self.coefficients
        )
        fields = torch.einsum("fsa,ca->fsc", coefficients, spatial_basis)
        if channel_mask is None:
            fields = fields / torch.linalg.vector_norm(
                fields, dim=-1, keepdim=True
            ).clamp_min(1e-7)
            return torch.einsum("fsc,bcft->bsft", fields, filtered)
        if channel_mask.shape != filtered.shape[:2]:
            raise ValueError("channel_mask must have shape (batch, channels)")
        batch_fields = fields[None] * channel_mask[:, None, None, :].to(filtered.dtype)
        batch_fields = batch_fields / torch.linalg.vector_norm(
            batch_fields, dim=-1, keepdim=True
        ).clamp_min(1e-7)
        return torch.einsum("bfsc,bcft->bsft", batch_fields, filtered)


@dataclass(frozen=True)
class CardinalFieldConfig:
    n_filters: int = 16
    n_sources: int = 32
    sinc_kernel: int = 65
    segments: int = 4
    dropout: float = 0.0
    length_scale: float = 0.35
    ridge: float = 1e-4
    adaptive_fir: bool = False
    frequency_cardinal: bool = False
    frequency_length_scale: float = 4.0
    contrast_bins: int = 0
    contrast_width: int = 8
    extended_atlas: bool = False
    spectral_residual: bool = False
    learned_windows: bool = False
    minimum_window_fraction: float = 0.10
    soft_window_temperature_samples: float = 1.0


class _CardinalEnergyDecoder(nn.Module):
    """One-head segmented log-variance decoder with no lossy token norm."""

    def __init__(self, *, n_outputs: int, config: CardinalFieldConfig) -> None:
        super().__init__()
        self.config = config
        features = config.n_filters * config.n_sources
        self.batch_norm = nn.BatchNorm1d(features)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(features * config.segments, n_outputs)
        if config.learned_windows:
            if not 0.0 <= config.minimum_window_fraction < 1.0 / config.segments:
                raise ValueError(
                    "minimum_window_fraction must be in [0, 1 / segments)"
                )
            if config.soft_window_temperature_samples <= 0.0:
                raise ValueError("soft_window_temperature_samples must be positive")
            # Equal logits recover equal-duration windows at initialization.
            # Their common offset is unidentifiable, leaving segments - 1
            # effective degrees of freedom.
            self.window_logits = nn.Parameter(torch.zeros(config.segments))
        else:
            self.register_parameter("window_logits", None)
        if config.contrast_bins > 0:
            if config.contrast_bins < 2:
                raise ValueError("contrast_bins must be zero or at least two")
            self.contrast_embedding = nn.Sequential(
                nn.Linear(2 * config.contrast_bins - 1, config.contrast_width),
                nn.GELU(),
            )
            self.contrast_classifier = nn.Linear(
                features * config.contrast_width, n_outputs
            )
            nn.init.zeros_(self.contrast_classifier.weight)
            nn.init.zeros_(self.contrast_classifier.bias)
        else:
            self.contrast_embedding = None
            self.contrast_classifier = None

    @staticmethod
    def _segmented_log_variance(values: Tensor, segments: int) -> Tensor:
        batch, features, n_times = values.shape
        if n_times % segments:
            pad = segments - n_times % segments
            values = nn.functional.pad(values, (0, pad))
            n_times += pad
        values = values.reshape(batch, features, segments, n_times // segments)
        return torch.log(values.var(dim=-1, unbiased=False).clamp_min(1e-6))

    def temporal_boundaries(self) -> Tensor:
        """Return ordered normalized boundaries for the energy windows."""

        if self.window_logits is None:
            return torch.linspace(
                0.0,
                1.0,
                self.config.segments + 1,
                device=self.classifier.weight.device,
                dtype=self.classifier.weight.dtype,
            )
        remaining = 1.0 - self.config.segments * self.config.minimum_window_fraction
        widths = self.config.minimum_window_fraction + remaining * torch.softmax(
            self.window_logits - self.window_logits.mean(), dim=0
        )
        zero = torch.zeros(1, device=widths.device, dtype=widths.dtype)
        return torch.cat((zero, torch.cumsum(widths, dim=0)))

    def _soft_segmented_log_variance(self, values: Tensor) -> Tensor:
        """Differentiable ordered windows initialized at equal quarters."""

        _, _, n_times = values.shape
        boundaries = self.temporal_boundaries().to(values)
        time = (
            torch.arange(n_times, device=values.device, dtype=values.dtype) + 0.5
        ) / n_times
        if self.config.segments == 1:
            weights = torch.ones_like(time)[None, :]
        else:
            temperature = self.config.soft_window_temperature_samples / n_times
            # Only internal thresholds are softened.  Using sigmoids at zero
            # and one would attenuate the trial endpoints and give the outer
            # windows less mass than the interior windows.
            steps = torch.sigmoid(
                (time[None, :] - boundaries[1:-1, None]) / temperature
            )
            weights = torch.cat(
                (
                    1.0 - steps[:1],
                    steps[:-1] - steps[1:],
                    steps[-1:],
                ),
                dim=0,
            )
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-7)
        mean = torch.einsum("bft,st->bfs", values, weights)
        second = torch.einsum("bft,st->bfs", values.square(), weights)
        variance = second - mean.square()
        return torch.log(variance.clamp_min(1e-6))

    def _temporal_log_variance(self, values: Tensor) -> Tensor:
        if self.window_logits is None:
            return self._segmented_log_variance(values, self.config.segments)
        return self._soft_segmented_log_variance(values)

    def decode(self, sources: Tensor) -> Tensor:
        batch, spatial, filters, n_times = sources.shape
        values = sources.permute(0, 2, 1, 3).reshape(batch, filters * spatial, n_times)
        values = self.activation(self.batch_norm(values))
        log_variance = self._temporal_log_variance(values)
        logits = self.classifier(self.dropout(log_variance.flatten(start_dim=1)))
        if self.contrast_embedding is not None and self.contrast_classifier is not None:
            fine = self._segmented_log_variance(values, self.config.contrast_bins)
            centered = fine - fine.mean(dim=-1, keepdim=True)
            differences = fine[..., 1:] - fine[..., :-1]
            contrasts = torch.cat((centered, differences), dim=-1)
            hidden = self.contrast_embedding(contrasts)
            logits = logits + self.contrast_classifier(
                self.dropout(hidden.flatten(start_dim=1))
            )
        return logits


class CardinalFieldNet(_CardinalEnergyDecoder):
    """Continuous cardinal scalp-field filter bank with log-variance readout."""

    uses_positions = True

    def __init__(
        self,
        *,
        n_outputs: int,
        sfreq: float = 128.0,
        config: CardinalFieldConfig = CardinalFieldConfig(),
    ) -> None:
        super().__init__(n_outputs=n_outputs, config=config)
        bank_type = AdaptiveSincFilterBank if config.adaptive_fir else OrderedSincFilterBank
        self.filter_bank = bank_type(
            n_filters=config.n_filters,
            kernel_size=config.sinc_kernel,
            sfreq=sfreq,
            stride=1,
        )
        self.spectral_residual = (
            LowRankSpectralTemporalResidual(n_filters=config.n_filters)
            if config.spectral_residual
            else nn.Identity()
        )
        field_type = (
            JointCardinalScalpField
            if config.frequency_cardinal
            else CardinalScalpField
        )
        field_kwargs: dict[str, float | int] = {
            "n_filters": config.n_filters,
            "n_sources": config.n_sources,
            "length_scale": config.length_scale,
            "ridge": config.ridge,
            "anchor_positions": (
                EXTENDED_31_POSITIONS
                if config.extended_atlas
                else CANONICAL_21_POSITIONS
            ),
        }
        if config.frequency_cardinal:
            field_kwargs["frequency_length_scale"] = config.frequency_length_scale
        self.spatial_field = field_type(**field_kwargs)

    def forward(
        self,
        x: Tensor,
        positions: Tensor,
        channel_mask: Tensor | None = None,
    ) -> Tensor:
        filtered = self.spectral_residual(self.filter_bank(x))
        if isinstance(self.spatial_field, JointCardinalScalpField):
            sources = self.spatial_field(
                filtered,
                positions,
                self.filter_bank.frequencies_hz(),
                channel_mask,
            )
        else:
            sources = self.spatial_field(filtered, positions, channel_mask)
        return self.decode(sources)


class FreeCardinalFieldNet(_CardinalEnergyDecoder):
    """Channel-indexed control for the cardinal spatial operator."""

    uses_positions = False

    def __init__(
        self,
        *,
        n_channels: int,
        n_outputs: int,
        sfreq: float = 128.0,
        config: CardinalFieldConfig = CardinalFieldConfig(),
    ) -> None:
        super().__init__(n_outputs=n_outputs, config=config)
        bank_type = AdaptiveSincFilterBank if config.adaptive_fir else OrderedSincFilterBank
        self.filter_bank = bank_type(
            n_filters=config.n_filters,
            kernel_size=config.sinc_kernel,
            sfreq=sfreq,
            stride=1,
        )
        self.spatial_filter = FreeSpatialFilter(
            n_filters=config.n_filters,
            n_sources=config.n_sources,
            n_channels=n_channels,
        )

    def forward(self, x: Tensor, positions: Tensor | None = None) -> Tensor:
        del positions
        return self.decode(self.spatial_filter(self.filter_bank(x)))


class CardinalMixedTemporalNet(_CardinalEnergyDecoder):
    """Multi-scale temporal views followed by a continuous cardinal field.

    ``spectral_filtering`` and ``mixed_temporal`` are injected from the pinned
    FBMSNet reimplementation for a controlled comparison.  Its channel-indexed
    spatial convolution is replaced by a full-rank coordinate operator; the
    log-variance decoder remains a single fixed head.
    """

    uses_positions = True

    def __init__(
        self,
        *,
        spectral_filtering: nn.Module,
        mixed_temporal: nn.Module,
        n_outputs: int,
        dropout: float = 0.25,
        n_temporal_views: int = 36,
        n_sources_per_view: int = 8,
    ) -> None:
        config = CardinalFieldConfig(
            n_filters=n_temporal_views,
            n_sources=n_sources_per_view,
            segments=4,
            dropout=dropout,
        )
        super().__init__(n_outputs=n_outputs, config=config)
        self.spectral_filtering = spectral_filtering
        self.mixed_temporal = mixed_temporal
        self.spatial_field = CardinalScalpField(
            n_filters=n_temporal_views,
            n_sources=n_sources_per_view,
        )

    def forward(
        self,
        x: Tensor,
        positions: Tensor,
        channel_mask: Tensor | None = None,
    ) -> Tensor:
        bands = self.spectral_filtering(x)
        mixed = self.mixed_temporal(bands)
        if mixed.ndim != 4 or mixed.shape[2] != x.shape[1]:
            raise RuntimeError(
                f"mixed temporal block returned unexpected shape {tuple(mixed.shape)}"
            )
        filtered = mixed.permute(0, 2, 1, 3)
        return self.decode(self.spatial_field(filtered, positions, channel_mask))


class CardinalFBMSNet(nn.Module):
    """Function-preserving montage-continuous FBMSNet spatial replacement.

    FBMSNet's filter bank, mixed temporal convolution, spatial normalization
    and activation, temporal log-variance views, and constrained classifier are
    retained exactly. Only the grouped channel-indexed spatial convolution is
    replaced by a regularized cardinal-RBF field with the same coefficient and
    bias budget on the 21-electrode inducing montage.
    """

    uses_positions = True

    def __init__(
        self,
        *,
        spectral_filtering: nn.Module,
        mix_conv: nn.Module,
        batch_norm: nn.Module,
        activation: nn.Module,
        padding_layer: nn.Module,
        temporal_layer: nn.Module,
        flatten_layer: nn.Module,
        final_layer: nn.Module,
        n_times: int,
        n_temporal_views: int = 36,
        dilatability: int = 8,
        stride_factor: int = 4,
        extended_atlas: bool = False,
        indexed_spatial_weights: Tensor | None = None,
        indexed_spatial_bias: Tensor | None = None,
        observed_positions: Tensor | None = None,
        spatial_max_norm: float = 2.0,
    ) -> None:
        super().__init__()
        if n_temporal_views <= 0 or dilatability <= 0:
            raise ValueError("temporal views and dilatability must be positive")
        if stride_factor <= 0:
            raise ValueError("stride_factor must be positive")
        self.n_times = int(n_times)
        self.n_temporal_views = int(n_temporal_views)
        self.dilatability = int(dilatability)
        self.stride_factor = int(stride_factor)
        self.n_times_padded = self.n_times + (-self.n_times % self.stride_factor)
        self.out_channels_spatial = self.n_temporal_views * self.dilatability
        self.spectral_filtering = spectral_filtering
        self.mix_conv = mix_conv
        self.spatial_field = CardinalScalpField(
            n_filters=self.n_temporal_views,
            n_sources=self.dilatability,
            anchor_positions=(
                EXTENDED_31_POSITIONS
                if extended_atlas
                else CANONICAL_21_POSITIONS
            ),
            field_norm="max",
            max_norm=spatial_max_norm,
        )
        self.spatial_bias = nn.Parameter(
            torch.zeros(self.n_temporal_views, self.dilatability)
        )
        paired_values = (
            indexed_spatial_weights,
            indexed_spatial_bias,
            observed_positions,
        )
        paired_basis_rank: int | None = None
        paired_fit: str | None = None
        paired_observed_channels: int | None = None
        if any(value is not None for value in paired_values):
            if any(value is None for value in paired_values):
                raise ValueError(
                    "indexed spatial weights, bias, and observed positions must be supplied together"
                )
            assert indexed_spatial_weights is not None
            assert indexed_spatial_bias is not None
            assert observed_positions is not None
            expected_weights = (
                self.n_temporal_views,
                self.dilatability,
                observed_positions.shape[0],
            )
            if indexed_spatial_weights.shape != expected_weights:
                raise ValueError(
                    "indexed spatial weights have shape "
                    f"{tuple(indexed_spatial_weights.shape)}; expected {expected_weights}"
                )
            expected_bias = (self.n_temporal_views, self.dilatability)
            if indexed_spatial_bias.shape != expected_bias:
                raise ValueError(
                    f"indexed spatial bias has shape {tuple(indexed_spatial_bias.shape)}; "
                    f"expected {expected_bias}"
                )
            # The float64 pseudoinverse gives an exact minimum-norm continuation
            # when the observed basis has full row rank (including the paired
            # canonical 21-channel construction).  Overdetermined montages use
            # the corresponding least-squares coordinate-field projection.
            with torch.no_grad():
                target = indexed_spatial_weights.to(torch.float64)
                basis = self.spatial_field.basis(
                    observed_positions.to(self.spatial_field.anchors)
                ).to(torch.float64)
                paired_observed_channels = int(basis.shape[0])
                paired_basis_rank = int(torch.linalg.matrix_rank(basis))
                paired_fit = (
                    "exact_minimum_norm"
                    if paired_basis_rank == paired_observed_channels
                    else "least_squares_projection"
                )
                coefficients = torch.einsum(
                    "ac,fsc->fsa", torch.linalg.pinv(basis), target
                )
                self.spatial_field.coefficients.copy_(
                    coefficients.to(self.spatial_field.coefficients)
                )
                self.spatial_bias.copy_(indexed_spatial_bias.to(self.spatial_bias))
        self.batch_norm = batch_norm
        self.activation = activation
        self.padding_layer = padding_layer
        self.temporal_layer = temporal_layer
        self.flatten_layer = flatten_layer
        self.final_layer = final_layer
        self.config = {
            "reference_architecture": "Braindecode FBMSNet",
            "n_temporal_views": self.n_temporal_views,
            "dilatability": self.dilatability,
            "stride_factor": self.stride_factor,
            "extended_atlas": bool(extended_atlas),
            "spatial_field": "regularized_cardinal_rbf",
            "spatial_field_norm": "max",
            "spatial_max_norm": float(spatial_max_norm),
            "paired_fbms_initialization": indexed_spatial_weights is not None,
            "paired_fbms_initialization_fit": paired_fit,
            "paired_fbms_observed_channels": paired_observed_channels,
            "paired_fbms_basis_rank": paired_basis_rank,
            "replaced_component": "grouped_channel_indexed_spatial_conv_only",
            "dynamic_fbms_input_channels": True,
            "dynamic_fbms_input_times": True,
        }

    def _apply_shared_spectral_filter(self, x: Tensor) -> Tensor:
        """Apply FBMSNet's shared FIR bank to the current native montage.

        Braindecode's :class:`FilterBankLayer` stores one FIR per band, but its
        forward method repeats each FIR according to the construction-time
        ``n_chans`` value.  That value is not persistent model state.  Restoring
        it after each call lets one immutable CardinalFBMS checkpoint alternate
        between native montages without changing any learned tensor or the
        channel-shared filtering operation.
        """

        configured_channels = getattr(self.spectral_filtering, "n_chans", None)
        if configured_channels is None or int(configured_channels) == x.shape[1]:
            return self.spectral_filtering(x)
        self.spectral_filtering.n_chans = int(x.shape[1])
        try:
            return self.spectral_filtering(x)
        finally:
            self.spectral_filtering.n_chans = configured_channels

    def encode(
        self,
        x: Tensor,
        positions: Tensor,
        channel_mask: Tensor | None = None,
    ) -> Tensor:
        filtered = self._apply_shared_spectral_filter(x)
        mixed = self.mix_conv(filtered)
        if (
            mixed.ndim != 4
            or mixed.shape[1] != self.n_temporal_views
            or mixed.shape[2] != x.shape[1]
        ):
            raise RuntimeError(
                f"FBMS mixed convolution returned unexpected shape {tuple(mixed.shape)}"
            )
        sources = self.spatial_field(
            mixed.permute(0, 2, 1, 3), positions, channel_mask
        )
        sources = sources + self.spatial_bias.T[None, :, :, None]
        batch, spatial, temporal_views, n_times = sources.shape
        values = sources.permute(0, 2, 1, 3).reshape(
            batch, temporal_views * spatial, 1, n_times
        )
        values = self.activation(self.batch_norm(values))
        values = self.padding_layer(values)
        current_n_times_padded = n_times + (-n_times % self.stride_factor)
        if values.shape[-1] != current_n_times_padded:
            raise RuntimeError("FBMS padding layer returned an unexpected time length")
        values = values.reshape(
            batch,
            self.out_channels_spatial,
            self.stride_factor,
            current_n_times_padded // self.stride_factor,
        )
        return self.flatten_layer(self.temporal_layer(values))

    def forward(
        self,
        x: Tensor,
        positions: Tensor,
        channel_mask: Tensor | None = None,
    ) -> Tensor:
        return self.final_layer(self.encode(x, positions, channel_mask))


class CardinalFBCNet(nn.Module):
    """FBCNet's reference decoder with a montage-continuous spatial layer.

    The fixed filter bank, batch normalization, SiLU, four log-variance views,
    and constrained classifier are taken unchanged from the pinned Braindecode
    FBCNet implementation.  Only its grouped channel-indexed spatial
    convolution is replaced by a regularized cardinal-RBF weight field with
    the same 9 x 32 x 21 coefficient budget on the canonical montage.
    """

    uses_positions = True

    def __init__(
        self,
        *,
        spectral_filtering: nn.Module,
        batch_norm: nn.Module,
        activation: nn.Module,
        padding_layer: nn.Module,
        temporal_layer: nn.Module,
        flatten_layer: nn.Module,
        final_layer: nn.Module,
        n_times: int,
        n_bands: int = 9,
        n_sources: int = 32,
        stride_factor: int = 4,
        extended_atlas: bool = False,
        indexed_spatial_weights: Tensor | None = None,
        indexed_spatial_bias: Tensor | None = None,
        observed_positions: Tensor | None = None,
    ) -> None:
        super().__init__()
        if stride_factor <= 0:
            raise ValueError("stride_factor must be positive")
        self.n_times = int(n_times)
        self.n_bands = int(n_bands)
        self.n_sources = int(n_sources)
        self.stride_factor = int(stride_factor)
        self.n_times_padded = self.n_times + (-self.n_times % self.stride_factor)
        self.spectral_filtering = spectral_filtering
        self.spatial_field = CardinalScalpField(
            n_filters=self.n_bands,
            n_sources=self.n_sources,
            anchor_positions=(
                EXTENDED_31_POSITIONS
                if extended_atlas
                else CANONICAL_21_POSITIONS
            ),
            field_norm="max",
            max_norm=2.0,
        )
        self.spatial_bias = nn.Parameter(torch.zeros(self.n_bands, self.n_sources))
        paired_values = (
            indexed_spatial_weights,
            indexed_spatial_bias,
            observed_positions,
        )
        if any(value is not None for value in paired_values):
            if any(value is None for value in paired_values):
                raise ValueError(
                    "indexed spatial weights, bias, and observed positions must be supplied together"
                )
            assert indexed_spatial_weights is not None
            assert indexed_spatial_bias is not None
            assert observed_positions is not None
            expected = (self.n_bands, self.n_sources, observed_positions.shape[0])
            if indexed_spatial_weights.shape != expected:
                raise ValueError(
                    f"indexed spatial weights have shape {tuple(indexed_spatial_weights.shape)}; "
                    f"expected {expected}"
                )
            expected_bias = (self.n_bands, self.n_sources)
            if indexed_spatial_bias.shape != expected_bias:
                raise ValueError(
                    f"indexed spatial bias has shape {tuple(indexed_spatial_bias.shape)}; "
                    f"expected {expected_bias}"
                )
            # Reproduce the freshly seeded channel-indexed FBC layer exactly
            # on the observed montage. The float64 pseudoinverse gives the
            # minimum Euclidean coefficient-norm continuation when the atlas
            # has additional inducing positions; training is then free to use
            # the full continuous field.
            with torch.no_grad():
                target = indexed_spatial_weights.to(torch.float64)
                basis = self.spatial_field.basis(
                    observed_positions.to(self.spatial_field.anchors)
                ).to(torch.float64)
                coefficients = torch.einsum(
                    "ac,fsc->fsa", torch.linalg.pinv(basis), target
                )
                self.spatial_field.coefficients.copy_(
                    coefficients.to(self.spatial_field.coefficients)
                )
                self.spatial_bias.copy_(indexed_spatial_bias.to(self.spatial_bias))
        self.batch_norm = batch_norm
        self.activation = activation
        self.padding_layer = padding_layer
        self.temporal_layer = temporal_layer
        self.flatten_layer = flatten_layer
        self.final_layer = final_layer
        self.config = {
            "n_bands": self.n_bands,
            "n_sources": self.n_sources,
            "stride_factor": self.stride_factor,
            "extended_atlas": bool(extended_atlas),
            "paired_fbc_initialization": indexed_spatial_weights is not None,
            "dynamic_fbc_floor_input_channels": True,
            "dynamic_fbc_floor_input_times": True,
        }

    def _apply_shared_spectral_filter(self, x: Tensor) -> Tensor:
        """Apply one channel-shared filter bank to an arbitrary montage.

        Braindecode's :class:`FilterBankLayer` stores only one FIR per band,
        but its forward method repeats that FIR according to the ``n_chans``
        constructor argument.  Temporarily supplying the current channel
        count preserves exactly the same per-channel filtering while allowing
        one CardinalFBC instance to alternate between native montages.  The
        attribute is restored immediately and is not part of the state dict.
        Custom spectral modules without ``n_chans`` are called unchanged.
        """

        configured_channels = getattr(self.spectral_filtering, "n_chans", None)
        if configured_channels is None or int(configured_channels) == x.shape[1]:
            return self.spectral_filtering(x)
        self.spectral_filtering.n_chans = int(x.shape[1])
        try:
            return self.spectral_filtering(x)
        finally:
            self.spectral_filtering.n_chans = configured_channels

    def encode_floor_with_signed_sources(
        self,
        x: Tensor,
        positions: Tensor,
        channel_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return the FBC feature vector and its signed BN source signals.

        The second tensor has shape ``(batch, bands, sources, time)`` and is
        taken immediately after FBCNet's spatial batch normalization, before
        its SiLU activation.  Keeping both outputs in this one path prevents
        residual feature branches from silently applying the filter bank,
        spatial field, or stateful batch normalization a second time.
        """

        filtered = self._apply_shared_spectral_filter(x)
        if filtered.ndim != 4 or filtered.shape[1] != self.n_bands:
            raise RuntimeError(
                f"spectral filter returned unexpected shape {tuple(filtered.shape)}"
            )
        filtered = filtered.permute(0, 2, 1, 3)
        sources = self.spatial_field(filtered, positions, channel_mask)
        sources = sources + self.spatial_bias[None, :, :, None].permute(0, 2, 1, 3)
        batch, spatial, bands, n_times = sources.shape
        values = sources.permute(0, 2, 1, 3).reshape(
            batch, bands * spatial, 1, n_times
        )
        signed_sources = self.batch_norm(values).reshape(
            batch, bands, spatial, n_times
        )
        values = self.activation(
            signed_sources.reshape(batch, bands * spatial, 1, n_times)
        )
        values = self.padding_layer(values)
        # FBC's temporal representation always contains ``stride_factor``
        # log-variance views, so its feature count is independent of the
        # epoch duration.  Compute the padding contract from the current
        # batch rather than the construction-time duration.  This lets one
        # shared coordinate-field encoder alternate between, for example,
        # the 256-sample local montage and 320-sample public montages without
        # padding or cropping the underlying EEG.
        current_n_times_padded = n_times + (-n_times % self.stride_factor)
        if values.shape[-1] != current_n_times_padded:
            raise RuntimeError("FBC padding layer returned an unexpected time length")
        values = values.reshape(
            batch,
            bands * spatial,
            self.stride_factor,
            current_n_times_padded // self.stride_factor,
        )
        values = self.temporal_layer(values)
        return self.flatten_layer(values), signed_sources

    def encode_floor(
        self,
        x: Tensor,
        positions: Tensor,
        channel_mask: Tensor | None = None,
    ) -> Tensor:
        """Return the exact FBCNet log-variance feature vector."""

        features, _ = self.encode_floor_with_signed_sources(
            x, positions, channel_mask
        )
        return features

    def forward(
        self,
        x: Tensor,
        positions: Tensor,
        channel_mask: Tensor | None = None,
    ) -> Tensor:
        return self.final_layer(self.encode_floor(x, positions, channel_mask))


class CardinalFBCCorrelationNet(CardinalFBCNet):
    """Function-preserving FBC reference path plus source correlations.

    Each of FBCNet's nine signed, batch-normalized spatial-source groups is
    projected from 32 sources to a small six-dimensional subspace.  Four
    contiguous temporal views are summarized by shrinkage correlations and a
    Fisher transform.  These 9 x 4 x C(6, 2) features enter the same constrained
    classifier as the ordinary FBC log-variance features. All added classifier
    columns start at zero, so initialization exactly preserves the paired FBC
    mapping while allowing the correlation branch to begin learning after the
    first classifier update.
    """

    uses_positions = True

    def __init__(
        self,
        *,
        spectral_filtering: nn.Module,
        batch_norm: nn.Module,
        activation: nn.Module,
        padding_layer: nn.Module,
        temporal_layer: nn.Module,
        flatten_layer: nn.Module,
        final_layer: nn.Module,
        n_times: int,
        n_bands: int = 9,
        n_sources: int = 32,
        stride_factor: int = 4,
        extended_atlas: bool = False,
        indexed_spatial_weights: Tensor | None = None,
        indexed_spatial_bias: Tensor | None = None,
        observed_positions: Tensor | None = None,
        correlation_rank: int = 6,
        correlation_windows: int = 4,
        covariance_shrinkage: float = 0.1,
        variance_floor: float = 1e-6,
        fisher_clip: float = 1.0 - 1e-4,
    ) -> None:
        if not 1 < correlation_rank <= n_sources:
            raise ValueError("correlation_rank must lie in [2, n_sources]")
        if correlation_windows <= 0:
            raise ValueError("correlation_windows must be positive")
        if not 0.0 <= covariance_shrinkage < 1.0:
            raise ValueError("covariance_shrinkage must lie in [0, 1)")
        if variance_floor <= 0.0:
            raise ValueError("variance_floor must be positive")
        if not 0.0 < fisher_clip < 1.0:
            raise ValueError("fisher_clip must lie in (0, 1)")
        super().__init__(
            spectral_filtering=spectral_filtering,
            batch_norm=batch_norm,
            activation=activation,
            padding_layer=padding_layer,
            temporal_layer=temporal_layer,
            flatten_layer=flatten_layer,
            final_layer=final_layer,
            n_times=n_times,
            n_bands=n_bands,
            n_sources=n_sources,
            stride_factor=stride_factor,
            extended_atlas=extended_atlas,
            indexed_spatial_weights=indexed_spatial_weights,
            indexed_spatial_bias=indexed_spatial_bias,
            observed_positions=observed_positions,
        )
        if self.n_times_padded % correlation_windows:
            raise ValueError(
                "padded input length must be divisible by correlation_windows"
            )
        self.correlation_rank = int(correlation_rank)
        self.correlation_windows = int(correlation_windows)
        self.covariance_shrinkage = float(covariance_shrinkage)
        self.variance_floor = float(variance_floor)
        self.fisher_clip = float(fisher_clip)
        self.correlation_projection = nn.Parameter(
            torch.empty(self.n_bands, self.correlation_rank, self.n_sources)
        )
        with torch.no_grad():
            for projection in self.correlation_projection:
                nn.init.orthogonal_(projection)
        pairs = torch.triu_indices(
            self.correlation_rank, self.correlation_rank, offset=1
        )
        self.register_buffer("correlation_pair_rows", pairs[0])
        self.register_buffer("correlation_pair_columns", pairs[1])
        self.floor_feature_count = int(self.final_layer.in_features)
        self.correlation_feature_count = (
            self.n_bands
            * self.correlation_windows
            * self.correlation_rank
            * (self.correlation_rank - 1)
            // 2
        )
        self.final_layer = ExpandedMaxNormLinear.extend(
            self.final_layer, self.correlation_feature_count
        )
        self.config = {
            **self.config,
            "correlation_rank": self.correlation_rank,
            "correlation_windows": self.correlation_windows,
            "covariance_shrinkage": self.covariance_shrinkage,
            "variance_floor": self.variance_floor,
            "fisher_clip": self.fisher_clip,
            "correlation_feature_count": self.correlation_feature_count,
            "orthogonal_projection_initialization": True,
            "single_expanded_head": True,
        }

    def normalized_correlation_projection(self) -> Tensor:
        """Return row-unit projection matrices without constraining storage."""

        return nn.functional.normalize(
            self.correlation_projection, p=2, dim=-1, eps=1e-7
        )

    def encode_correlations(self, signed_sources: Tensor) -> Tensor:
        """Encode shrinkage Fisher-z correlations from signed FBC sources."""

        if signed_sources.ndim != 4:
            raise ValueError(
                "signed_sources must have shape (batch, bands, sources, time)"
            )
        batch, bands, sources, n_times = signed_sources.shape
        if bands != self.n_bands or sources != self.n_sources:
            raise ValueError("signed_sources have incompatible band/source dimensions")
        padded = self.padding_layer(
            signed_sources.reshape(batch, bands * sources, 1, n_times)
        )
        if padded.shape[-1] != self.n_times_padded:
            raise RuntimeError("FBC padding layer returned an unexpected time length")
        padded = padded.reshape(batch, bands, sources, self.n_times_padded)
        projected = torch.einsum(
            "bfst,frs->bfrt",
            padded,
            self.normalized_correlation_projection(),
        )
        window_length = self.n_times_padded // self.correlation_windows
        windows = projected.reshape(
            batch,
            bands,
            self.correlation_rank,
            self.correlation_windows,
            window_length,
        ).permute(0, 1, 3, 2, 4)
        centered = windows - windows.mean(dim=-1, keepdim=True)
        covariance = centered @ centered.transpose(-1, -2)
        covariance = covariance / max(window_length - 1, 1)
        variances = covariance.diagonal(dim1=-2, dim2=-1).clamp_min(
            self.variance_floor
        )
        shrinkage_diagonal = torch.diag_embed(variances)
        covariance = (
            (1.0 - self.covariance_shrinkage) * covariance
            + self.covariance_shrinkage * shrinkage_diagonal
        )
        scales = torch.sqrt(variances.unsqueeze(-1) * variances.unsqueeze(-2))
        correlations = covariance / scales.clamp_min(self.variance_floor)
        off_diagonal = correlations[
            ...,
            self.correlation_pair_rows,
            self.correlation_pair_columns,
        ]
        fisher_z = torch.atanh(
            off_diagonal.clamp(-self.fisher_clip, self.fisher_clip)
        )
        return fisher_z.flatten(start_dim=1)

    def forward(
        self,
        x: Tensor,
        positions: Tensor,
        channel_mask: Tensor | None = None,
    ) -> Tensor:
        floor, signed_sources = self.encode_floor_with_signed_sources(
            x, positions, channel_mask
        )
        correlations = self.encode_correlations(signed_sources)
        return self.final_layer(torch.cat((floor, correlations), dim=1))


class CardinalSpatiotemporalField(nn.Module):
    """Continuous full-rank field that mixes temporal filters and sensors."""

    def __init__(
        self,
        *,
        n_input_filters: int,
        n_output_sources: int,
        length_scale: float = 0.35,
        ridge: float = 1e-4,
        anchor_positions: tuple[tuple[float, float, float], ...] = CANONICAL_21_POSITIONS,
    ) -> None:
        super().__init__()
        atlas = CardinalScalpField(
            n_filters=1,
            n_sources=1,
            length_scale=length_scale,
            ridge=ridge,
            anchor_positions=anchor_positions,
        )
        self.register_buffer("anchors", atlas.anchors.detach().clone())
        self.register_buffer(
            "cardinal_inverse", atlas.cardinal_inverse.detach().clone()
        )
        self.length_scale = float(length_scale)
        self.coefficients = nn.Parameter(
            torch.empty(
                n_output_sources,
                n_input_filters,
                len(anchor_positions),
            )
        )
        nn.init.xavier_uniform_(self.coefficients, gain=0.7)

    def basis(self, positions: Tensor) -> Tensor:
        kernel = spherical_gaussian_kernel(
            positions.to(self.anchors), self.anchors, self.length_scale
        )
        return kernel @ self.cardinal_inverse

    def forward(
        self,
        filtered: Tensor,
        positions: Tensor,
        channel_mask: Tensor | None = None,
    ) -> Tensor:
        if filtered.ndim != 4:
            raise ValueError("filtered must have shape (batch, channels, filters, time)")
        basis = self.basis(positions).to(filtered)
        fields = torch.einsum("ofa,ca->ofc", self.coefficients, basis)
        if channel_mask is None:
            fields = fields / torch.linalg.vector_norm(
                fields.flatten(start_dim=1), dim=1, keepdim=True
            )[:, None].clamp_min(1e-7)
            return torch.einsum("ofc,bcft->bot", fields, filtered)
        if channel_mask.shape != filtered.shape[:2]:
            raise ValueError("channel_mask must have shape (batch, channels)")
        batch_fields = fields[None] * channel_mask[:, None, None, :].to(filtered.dtype)
        norms = torch.linalg.vector_norm(
            batch_fields.flatten(start_dim=2), dim=2, keepdim=True
        )
        batch_fields = batch_fields / norms[..., None].clamp_min(1e-7)
        return torch.einsum("bofc,bcft->bot", batch_fields, filtered)


class ExpandedMaxNormLinear(nn.Linear):
    """A row-max-norm head that exactly extends a constrained reference head."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        max_norm: float | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        if max_norm is not None and max_norm <= 0.0:
            raise ValueError("max_norm must be positive")
        self.max_norm = max_norm

    def constrained_weight(self) -> Tensor:
        if self.max_norm is None:
            return self.weight
        return self.weight.renorm(p=2, dim=0, maxnorm=self.max_norm)

    def forward(self, values: Tensor) -> Tensor:
        return nn.functional.linear(values, self.constrained_weight(), self.bias)

    @classmethod
    def extend(cls, floor: nn.Module, extra_features: int) -> "ExpandedMaxNormLinear":
        """Append zero columns while retaining a reference head's exact mapping."""

        if not isinstance(floor, nn.Linear):
            raise TypeError("floor classifier must be a linear layer")
        if extra_features <= 0:
            raise ValueError("extra_features must be positive")
        source_weight = floor.weight.detach()
        max_norm: float | None = None
        parametrizations = getattr(floor, "parametrizations", None)
        if parametrizations is not None and hasattr(parametrizations, "weight"):
            weight_stack = parametrizations.weight
            source_weight = weight_stack.original.detach()
            if len(weight_stack) != 1 or not hasattr(weight_stack[0], "max_norm"):
                raise TypeError("unsupported floor weight parametrization")
            max_norm = float(weight_stack[0].max_norm)

        expanded = cls(
            floor.in_features + extra_features,
            floor.out_features,
            bias=floor.bias is not None,
            max_norm=max_norm,
            device=source_weight.device,
            dtype=source_weight.dtype,
        )
        with torch.no_grad():
            expanded.weight.zero_()
            expanded.weight[:, : floor.in_features].copy_(source_weight)
            if floor.bias is not None:
                assert expanded.bias is not None
                expanded.bias.copy_(floor.bias.detach())
        return expanded


class _CardinalFBCEnergyDynamicsContinuation(CardinalFBCNet):
    """Shared function-preserving energy/dynamics continuation for CardinalFBC.

    Subclasses choose only the temporal filter family and branch dimensions.
    The reference FBC feature encoder remains untouched, while a single
    max-norm classifier is extended with zero columns. This makes every
    continuation exactly equal to its paired FBC mapping at initialization.
    """

    uses_positions = True

    def __init__(
        self,
        *,
        continuation_kind: str,
        continuation_temporal_filters: int,
        continuation_temporal_kernel: int,
        continuation_sources: int,
        energy_windows: int,
        dynamics_channels: int,
        dynamics_kernel: int,
        continuation_stride: int = 1,
        continuation_scale: float = 1.0,
        sfreq: float = 128.0,
        **floor_kwargs: object,
    ) -> None:
        if continuation_temporal_filters <= 0:
            raise ValueError("continuation_temporal_filters must be positive")
        if continuation_sources <= 0:
            raise ValueError("continuation_sources must be positive")
        if energy_windows <= 0:
            raise ValueError("energy_windows must be positive")
        if dynamics_channels <= 0:
            raise ValueError("dynamics_channels must be positive")
        if dynamics_kernel <= 1 or dynamics_kernel % 2 == 0:
            raise ValueError("dynamics_kernel must be an odd integer greater than one")
        if continuation_stride <= 0:
            raise ValueError("continuation_stride must be positive")
        if not math.isfinite(continuation_scale) or continuation_scale <= 0.0:
            raise ValueError("continuation_scale must be finite and positive")
        super().__init__(**floor_kwargs)

        self.continuation_kind = str(continuation_kind)
        self.continuation_scale = float(continuation_scale)
        self.continuation_temporal_filters = int(continuation_temporal_filters)
        self.continuation_sources = int(continuation_sources)
        self.energy_windows = int(energy_windows)
        self.dynamics_channels = int(dynamics_channels)
        if self.continuation_kind == "gabor_fir":
            if continuation_stride != 1:
                raise ValueError("the Gabor FIR continuation requires stride one")
            self.continuation_filter_bank: nn.Module = LearnableGaborFIRBank(
                n_filters=self.continuation_temporal_filters,
                kernel_size=continuation_temporal_kernel,
                sfreq=sfreq,
            )
        elif self.continuation_kind == "ordered_sinc":
            self.continuation_filter_bank = OrderedSincFilterBank(
                n_filters=self.continuation_temporal_filters,
                kernel_size=continuation_temporal_kernel,
                sfreq=sfreq,
                stride=continuation_stride,
            )
        else:
            raise ValueError(f"unknown continuation kind {continuation_kind!r}")

        extended_atlas = bool(self.config["extended_atlas"])
        self.continuation_spatiotemporal_field = CardinalSpatiotemporalField(
            n_input_filters=self.continuation_temporal_filters,
            n_output_sources=self.continuation_sources,
            anchor_positions=(
                EXTENDED_31_POSITIONS
                if extended_atlas
                else CANONICAL_21_POSITIONS
            ),
        )
        self.continuation_batch_norm = nn.BatchNorm1d(self.continuation_sources)
        self.continuation_dynamics_depthwise = nn.Conv1d(
            self.continuation_sources,
            self.continuation_sources,
            kernel_size=dynamics_kernel,
            padding=dynamics_kernel // 2,
            groups=self.continuation_sources,
            bias=False,
        )
        self.continuation_dynamics_pointwise = nn.Conv1d(
            self.continuation_sources,
            self.dynamics_channels,
            kernel_size=1,
            bias=False,
        )
        self.continuation_dynamics_activation = nn.GELU()
        self.floor_feature_count = int(self.final_layer.in_features)
        self.continuation_feature_count = (
            self.continuation_sources * self.energy_windows
            + 2 * self.dynamics_channels
        )
        self.final_layer = ExpandedMaxNormLinear.extend(
            self.final_layer, self.continuation_feature_count
        )
        self.config = {
            **self.config,
            "continuation_kind": self.continuation_kind,
            "continuation_temporal_filters": self.continuation_temporal_filters,
            "continuation_temporal_kernel": int(continuation_temporal_kernel),
            "continuation_stride": int(continuation_stride),
            "continuation_sources": self.continuation_sources,
            "energy_windows": self.energy_windows,
            "dynamics_channels": self.dynamics_channels,
            "dynamics_kernel": int(dynamics_kernel),
            "continuation_feature_count": self.continuation_feature_count,
            "continuation_scale": self.continuation_scale,
            "zero_initialized_continuation_head": True,
            "single_expanded_head": True,
        }

    def _windowed_log_energy(self, values: Tensor) -> Tensor:
        n_times = values.shape[-1]
        if n_times < self.energy_windows:
            raise ValueError("input is shorter than the requested energy windows")
        boundaries = [
            round(index * n_times / self.energy_windows)
            for index in range(self.energy_windows + 1)
        ]
        energies = [
            torch.log(
                values[..., boundaries[index] : boundaries[index + 1]]
                .square()
                .mean(dim=-1)
                .clamp_min(1e-6)
            )
            for index in range(self.energy_windows)
        ]
        return torch.stack(energies, dim=-1)

    def encode_continuation(
        self,
        x: Tensor,
        positions: Tensor,
        channel_mask: Tensor | None = None,
    ) -> Tensor:
        filtered = self.continuation_filter_bank(x)
        latent = self.continuation_batch_norm(
            self.continuation_spatiotemporal_field(
                filtered, positions, channel_mask
            )
        )
        energy = self._windowed_log_energy(latent).flatten(start_dim=1)
        dynamics = self.continuation_dynamics_activation(
            self.continuation_dynamics_pointwise(
                self.continuation_dynamics_depthwise(latent)
            )
        )
        summary = torch.cat(
            (
                dynamics.mean(dim=-1),
                dynamics.std(dim=-1, unbiased=False),
            ),
            dim=1,
        )
        return self.continuation_scale * torch.cat((energy, summary), dim=1)

    def forward(
        self,
        x: Tensor,
        positions: Tensor,
        channel_mask: Tensor | None = None,
    ) -> Tensor:
        floor = self.encode_floor(x, positions, channel_mask)
        continuation = self.encode_continuation(x, positions, channel_mask)
        return self.final_layer(torch.cat((floor, continuation), dim=1))


class CardinalFBCMicroDynamicsNet(_CardinalFBCEnergyDynamicsContinuation):
    """CardinalFBC reference path plus a compact free-FIR continuation."""

    def __init__(
        self,
        *,
        spectral_filtering: nn.Module,
        batch_norm: nn.Module,
        activation: nn.Module,
        padding_layer: nn.Module,
        temporal_layer: nn.Module,
        flatten_layer: nn.Module,
        final_layer: nn.Module,
        n_times: int,
        sfreq: float = 128.0,
        n_bands: int = 9,
        n_sources: int = 32,
        stride_factor: int = 4,
        extended_atlas: bool = False,
        indexed_spatial_weights: Tensor | None = None,
        indexed_spatial_bias: Tensor | None = None,
        observed_positions: Tensor | None = None,
        micro_temporal_filters: int = 8,
        micro_temporal_kernel: int = 25,
        micro_sources: int = 8,
        energy_windows: int = 8,
        dynamics_channels: int = 8,
        dynamics_kernel: int = 15,
        continuation_scale: float = 1.0,
    ) -> None:
        super().__init__(
            continuation_kind="gabor_fir",
            continuation_temporal_filters=micro_temporal_filters,
            continuation_temporal_kernel=micro_temporal_kernel,
            continuation_sources=micro_sources,
            energy_windows=energy_windows,
            dynamics_channels=dynamics_channels,
            dynamics_kernel=dynamics_kernel,
            continuation_scale=continuation_scale,
            sfreq=sfreq,
            spectral_filtering=spectral_filtering,
            batch_norm=batch_norm,
            activation=activation,
            padding_layer=padding_layer,
            temporal_layer=temporal_layer,
            flatten_layer=flatten_layer,
            final_layer=final_layer,
            n_times=n_times,
            n_bands=n_bands,
            n_sources=n_sources,
            stride_factor=stride_factor,
            extended_atlas=extended_atlas,
            indexed_spatial_weights=indexed_spatial_weights,
            indexed_spatial_bias=indexed_spatial_bias,
            observed_positions=observed_positions,
        )
        self.micro_temporal_filters = self.continuation_temporal_filters
        self.micro_sources = self.continuation_sources
        self.micro_feature_count = self.continuation_feature_count
        self.config = {
            **self.config,
            "micro_temporal_filters": self.micro_temporal_filters,
            "micro_temporal_kernel": int(micro_temporal_kernel),
            "micro_sources": self.micro_sources,
        }

    @property
    def micro_filter_bank(self) -> nn.Module:
        return self.continuation_filter_bank

    @property
    def micro_spatiotemporal_field(self) -> CardinalSpatiotemporalField:
        return self.continuation_spatiotemporal_field

    @property
    def micro_batch_norm(self) -> nn.BatchNorm1d:
        return self.continuation_batch_norm

    @property
    def micro_dynamics_depthwise(self) -> nn.Conv1d:
        return self.continuation_dynamics_depthwise

    @property
    def micro_dynamics_pointwise(self) -> nn.Conv1d:
        return self.continuation_dynamics_pointwise

    @property
    def micro_dynamics_activation(self) -> nn.GELU:
        return self.continuation_dynamics_activation

    def encode_micro(
        self,
        x: Tensor,
        positions: Tensor,
        channel_mask: Tensor | None = None,
    ) -> Tensor:
        return self.encode_continuation(x, positions, channel_mask)


class CardinalFBCPhysicalDynamicsNet(_CardinalFBCEnergyDynamicsContinuation):
    """FBC reference path plus an ordered physical-frequency continuation.

    Sixteen learnable ordered Sinc bands are jointly projected with electrode
    coordinates into twelve continuous sources. Eight local log-energy views
    retain FBCNet's MI-specific inductive bias, while a depthwise/pointwise dynamics
    path adds signed mean and standard-deviation summaries. The only classifier
    is the function-preserved FBC head with zero-initialized new columns.
    """

    def __init__(
        self,
        *,
        spectral_filtering: nn.Module,
        batch_norm: nn.Module,
        activation: nn.Module,
        padding_layer: nn.Module,
        temporal_layer: nn.Module,
        flatten_layer: nn.Module,
        final_layer: nn.Module,
        n_times: int,
        sfreq: float = 128.0,
        n_bands: int = 9,
        n_sources: int = 32,
        stride_factor: int = 4,
        extended_atlas: bool = False,
        indexed_spatial_weights: Tensor | None = None,
        indexed_spatial_bias: Tensor | None = None,
        observed_positions: Tensor | None = None,
        physical_temporal_filters: int = 16,
        physical_temporal_kernel: int = 65,
        physical_stride: int = 2,
        physical_sources: int = 12,
        energy_windows: int = 8,
        dynamics_channels: int = 8,
        dynamics_kernel: int = 15,
    ) -> None:
        super().__init__(
            continuation_kind="ordered_sinc",
            continuation_temporal_filters=physical_temporal_filters,
            continuation_temporal_kernel=physical_temporal_kernel,
            continuation_sources=physical_sources,
            continuation_stride=physical_stride,
            energy_windows=energy_windows,
            dynamics_channels=dynamics_channels,
            dynamics_kernel=dynamics_kernel,
            sfreq=sfreq,
            spectral_filtering=spectral_filtering,
            batch_norm=batch_norm,
            activation=activation,
            padding_layer=padding_layer,
            temporal_layer=temporal_layer,
            flatten_layer=flatten_layer,
            final_layer=final_layer,
            n_times=n_times,
            n_bands=n_bands,
            n_sources=n_sources,
            stride_factor=stride_factor,
            extended_atlas=extended_atlas,
            indexed_spatial_weights=indexed_spatial_weights,
            indexed_spatial_bias=indexed_spatial_bias,
            observed_positions=observed_positions,
        )
        self.physical_temporal_filters = self.continuation_temporal_filters
        self.physical_sources = self.continuation_sources
        self.physical_feature_count = self.continuation_feature_count
        self.config = {
            **self.config,
            "physical_temporal_filters": self.physical_temporal_filters,
            "physical_temporal_kernel": int(physical_temporal_kernel),
            "physical_stride": int(physical_stride),
            "physical_sources": self.physical_sources,
            "ordered_physical_frequency_bank": True,
        }

    @property
    def physical_filter_bank(self) -> OrderedSincFilterBank:
        assert isinstance(self.continuation_filter_bank, OrderedSincFilterBank)
        return self.continuation_filter_bank

    @property
    def physical_spatiotemporal_field(self) -> CardinalSpatiotemporalField:
        return self.continuation_spatiotemporal_field

    @property
    def physical_dynamics_depthwise(self) -> nn.Conv1d:
        return self.continuation_dynamics_depthwise

    @property
    def physical_dynamics_pointwise(self) -> nn.Conv1d:
        return self.continuation_dynamics_pointwise


@dataclass(frozen=True)
class CardinalDynamicsConfig:
    temporal_filters: int = 32
    temporal_kernel: int = 25
    physical_filter_bank: bool = False
    adaptive_fir: bool = False
    spectral_residual: bool = False
    latent_sources: int = 32
    pool_kernel: int = 75
    pool_stride: int = 15
    dynamics_channels: int = 16
    dynamics_kernel: int = 15
    dropout: float = 0.4
    length_scale: float = 0.35
    ridge: float = 1e-4
    extended_atlas: bool = False


class CardinalDynamicsNet(nn.Module):
    """Continuous spatiotemporal energy-and-dynamics MI network.

    On the inducing montage, the cardinal tensor spans every ordinary combined
    temporal/spatial convolution.  At a new montage the same tensor is sampled
    continuously at the observed coordinates.  Energy and signed dynamics are
    concatenated into one fixed classifier—there is no expert routing.
    """

    uses_positions = True

    def __init__(
        self,
        *,
        n_outputs: int | None,
        n_times: int = 320,
        config: CardinalDynamicsConfig = CardinalDynamicsConfig(),
    ) -> None:
        super().__init__()
        latent_times = (
            n_times
            if config.physical_filter_bank
            else n_times - config.temporal_kernel + 1
        )
        pooled_times = (latent_times - config.pool_kernel) // config.pool_stride + 1
        if latent_times <= 0 or pooled_times <= 0:
            raise ValueError("temporal and pooling kernels do not fit the epoch")
        self.config = config
        if config.physical_filter_bank:
            bank_type = (
                AdaptiveSincFilterBank if config.adaptive_fir else OrderedSincFilterBank
            )
            self.temporal = bank_type(
                n_filters=config.temporal_filters,
                kernel_size=config.temporal_kernel,
                stride=1,
            )
            self.spectral_residual = (
                LowRankSpectralTemporalResidual(
                    n_filters=config.temporal_filters
                )
                if config.spectral_residual
                else nn.Identity()
            )
        else:
            if config.adaptive_fir or config.spectral_residual:
                raise ValueError(
                    "adaptive_fir/spectral_residual require physical_filter_bank"
                )
            self.temporal = nn.Conv1d(
                1,
                config.temporal_filters,
                kernel_size=config.temporal_kernel,
                bias=False,
            )
            self.spectral_residual = nn.Identity()
        self.spatiotemporal_field = CardinalSpatiotemporalField(
            n_input_filters=config.temporal_filters,
            n_output_sources=config.latent_sources,
            length_scale=config.length_scale,
            ridge=config.ridge,
            anchor_positions=(
                EXTENDED_31_POSITIONS
                if config.extended_atlas
                else CANONICAL_21_POSITIONS
            ),
        )
        self.batch_norm = nn.BatchNorm1d(config.latent_sources)
        self.energy_pool = nn.AvgPool1d(
            config.pool_kernel, stride=config.pool_stride
        )
        padding = config.dynamics_kernel // 2
        self.dynamics = nn.Sequential(
            nn.Conv1d(
                config.latent_sources,
                config.latent_sources,
                kernel_size=config.dynamics_kernel,
                padding=padding,
                groups=config.latent_sources,
                bias=False,
            ),
            nn.Conv1d(
                config.latent_sources,
                config.dynamics_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                max(1, math.gcd(4, config.dynamics_channels)),
                config.dynamics_channels,
            ),
            nn.GELU(),
        )
        self.feature_count = (
            config.latent_sources * pooled_times + 2 * config.dynamics_channels
        )
        self.dropout = nn.Dropout(config.dropout)
        # ``None`` exposes the exact same module as a reusable feature encoder
        # without retaining an unused branch classifier.  Standalone models
        # continue to supply an output count and therefore keep their original
        # classifier behavior.
        self.classifier: nn.Module = (
            nn.Identity()
            if n_outputs is None
            else nn.Linear(self.feature_count, n_outputs)
        )

    def encode(
        self,
        x: Tensor,
        positions: Tensor,
        channel_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if x.ndim != 3:
            raise ValueError("x must have shape (batch, channels, time)")
        batch, channels, n_times = x.shape
        if isinstance(self.temporal, OrderedSincFilterBank):
            filtered = self.spectral_residual(self.temporal(x))
        else:
            temporal = self.temporal(x.reshape(batch * channels, 1, n_times))
            filtered = temporal.reshape(
                batch,
                channels,
                self.config.temporal_filters,
                temporal.shape[-1],
            )
        latent = self.batch_norm(
            self.spatiotemporal_field(filtered, positions, channel_mask)
        )
        energy = torch.log(
            self.energy_pool(latent.square()).clamp_min(1e-6)
        ).flatten(start_dim=1)
        dynamics = self.dynamics(latent)
        dynamics_summary = torch.cat(
            (
                dynamics.mean(dim=-1),
                dynamics.std(dim=-1, unbiased=False),
            ),
            dim=1,
        )
        return energy, dynamics_summary

    def encode_features(
        self,
        x: Tensor,
        positions: Tensor,
        channel_mask: Tensor | None = None,
    ) -> Tensor:
        """Return the complete energy-and-dynamics feature vector."""

        energy, dynamics = self.encode(x, positions, channel_mask)
        return torch.cat((energy, dynamics), dim=1)

    def forward(
        self,
        x: Tensor,
        positions: Tensor,
        channel_mask: Tensor | None = None,
    ) -> Tensor:
        features = self.encode_features(x, positions, channel_mask)
        return self.classifier(self.dropout(features))


class CardinalFBCCompactDynamicsNet(CardinalFBCNet):
    """FBC reference path plus the full compact dynamics feature encoder.

    The continuation is the same high-capacity encoder used by
    :class:`CardinalDynamicsNet`'s compact configuration: 24 unconstrained
    length-25 temporal FIRs, a 24-source cardinal spatiotemporal field,
    75-sample energy pooling at stride 15, and a 12-channel signed-dynamics
    path. Its features are concatenated with FBC log-variance features and
    enter one expanded max-norm head.  New head columns start at zero, so the
    paired FBC/Cardinal reference mapping is preserved exactly at
    initialization without a second classifier or logit ensemble.
    """

    uses_positions = True

    def __init__(
        self,
        *,
        spectral_filtering: nn.Module,
        batch_norm: nn.Module,
        activation: nn.Module,
        padding_layer: nn.Module,
        temporal_layer: nn.Module,
        flatten_layer: nn.Module,
        final_layer: nn.Module,
        n_times: int,
        n_bands: int = 9,
        n_sources: int = 32,
        stride_factor: int = 4,
        extended_atlas: bool = False,
        indexed_spatial_weights: Tensor | None = None,
        indexed_spatial_bias: Tensor | None = None,
        observed_positions: Tensor | None = None,
        continuation_scale: float = 1.0,
    ) -> None:
        if not math.isfinite(continuation_scale) or continuation_scale <= 0.0:
            raise ValueError("continuation_scale must be finite and positive")
        super().__init__(
            spectral_filtering=spectral_filtering,
            batch_norm=batch_norm,
            activation=activation,
            padding_layer=padding_layer,
            temporal_layer=temporal_layer,
            flatten_layer=flatten_layer,
            final_layer=final_layer,
            n_times=n_times,
            n_bands=n_bands,
            n_sources=n_sources,
            stride_factor=stride_factor,
            extended_atlas=extended_atlas,
            indexed_spatial_weights=indexed_spatial_weights,
            indexed_spatial_bias=indexed_spatial_bias,
            observed_positions=observed_positions,
        )
        self.continuation_scale = float(continuation_scale)
        dynamics_config = CardinalDynamicsConfig(
            temporal_filters=24,
            temporal_kernel=25,
            physical_filter_bank=False,
            latent_sources=24,
            pool_kernel=75,
            pool_stride=15,
            dynamics_channels=12,
            dynamics_kernel=15,
            dropout=0.0,
            extended_atlas=extended_atlas,
        )
        self.compact_dynamics = CardinalDynamicsNet(
            n_outputs=None,
            n_times=n_times,
            config=dynamics_config,
        )
        if not isinstance(self.compact_dynamics.classifier, nn.Identity):
            raise RuntimeError("compact dynamics continuation retained a classifier")
        self.floor_feature_count = int(self.final_layer.in_features)
        self.compact_dynamics_feature_count = self.compact_dynamics.feature_count
        self.final_layer = ExpandedMaxNormLinear.extend(
            self.final_layer, self.compact_dynamics_feature_count
        )
        self.config = {
            **self.config,
            "compact_dynamics_temporal_filters": 24,
            "compact_dynamics_temporal_kernel": 25,
            "compact_dynamics_latent_sources": 24,
            "compact_dynamics_pool_kernel": 75,
            "compact_dynamics_pool_stride": 15,
            "compact_dynamics_channels": 12,
            "compact_dynamics_kernel": 15,
            "compact_dynamics_dropout": 0.0,
            "compact_dynamics_feature_count": self.compact_dynamics_feature_count,
            "continuation_scale": self.continuation_scale,
            "zero_initialized_continuation_head": True,
            "single_expanded_head": True,
            "separate_branch_classifier": False,
        }

    def encode_compact_dynamics(
        self,
        x: Tensor,
        positions: Tensor,
        channel_mask: Tensor | None = None,
    ) -> Tensor:
        """Reuse the standalone dynamics network's complete feature encoder."""

        features = self.compact_dynamics.encode_features(
            x, positions, channel_mask
        )
        return self.continuation_scale * features

    def forward(
        self,
        x: Tensor,
        positions: Tensor,
        channel_mask: Tensor | None = None,
    ) -> Tensor:
        floor = self.encode_floor(x, positions, channel_mask)
        dynamics = self.encode_compact_dynamics(x, positions, channel_mask)
        return self.final_layer(torch.cat((floor, dynamics), dim=1))


class MultiResolutionMoments(nn.Module):
    """Log-energy at nested physical scales plus low-order dynamics."""

    def __init__(self, bins: Iterable[int] = (1, 2, 4, 8)) -> None:
        super().__init__()
        self.bins = tuple(int(value) for value in bins)
        if not self.bins or any(value <= 0 for value in self.bins):
            raise ValueError("all multi-resolution bin counts must be positive")
        self.output_dim = sum(self.bins) + 4

    def forward(self, sources: Tensor) -> Tensor:
        if sources.ndim != 4:
            raise ValueError("sources must have shape (batch, sources, filters, time)")
        batch, spatial, filters, n_times = sources.shape
        flattened = sources.reshape(batch * spatial * filters, 1, n_times)
        energies: list[Tensor] = []
        squared = flattened.square()
        for bins in self.bins:
            if n_times % bins == 0:
                pooled = squared.reshape(
                    batch * spatial * filters, 1, bins, n_times // bins
                ).mean(dim=-1)
            else:
                # Explicit half-open segment means avoid the nondeterministic
                # CUDA adaptive-pooling backward kernel.
                boundaries = torch.linspace(
                    0, n_times, bins + 1, device=sources.device
                ).round().long()
                pooled = torch.cat(
                    [
                        squared[..., int(boundaries[i]) : int(boundaries[i + 1])].mean(
                            dim=-1, keepdim=True
                        )
                        for i in range(bins)
                    ],
                    dim=-1,
                )
            energies.append(torch.log(pooled.clamp_min(1e-6)).reshape(batch, spatial, filters, bins))
        difference = sources[..., 1:] - sources[..., :-1]
        dynamics = torch.stack(
            (
                sources.mean(dim=-1),
                sources.std(dim=-1, unbiased=False),
                difference.abs().mean(dim=-1),
                difference.std(dim=-1, unbiased=False),
            ),
            dim=-1,
        )
        return torch.cat((*energies, dynamics), dim=-1)


class AxialMixerBlock(nn.Module):
    """Small frequency/source mixer without quadratic token attention."""

    def __init__(self, *, width: int, n_sources: int, dropout: float) -> None:
        super().__init__()
        self.frequency_norm = nn.LayerNorm(width)
        self.frequency_depthwise = nn.Conv1d(
            width, width, kernel_size=3, padding=1, groups=width, bias=False
        )
        self.frequency_pointwise = nn.Conv1d(width, width, kernel_size=1)
        self.source_norm = nn.LayerNorm(width)
        self.source_mix = nn.Linear(n_sources, n_sources, bias=False)
        self.feature_norm = nn.LayerNorm(width)
        self.feature_mlp = nn.Sequential(
            nn.Linear(width, 2 * width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * width, width),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: Tensor) -> Tensor:
        batch, sources, filters, width = values.shape
        frequency = self.frequency_norm(values).reshape(batch * sources, filters, width)
        frequency = frequency.transpose(1, 2)
        frequency = self.frequency_pointwise(self.frequency_depthwise(frequency))
        frequency = frequency.transpose(1, 2).reshape(batch, sources, filters, width)
        values = values + self.dropout(frequency)

        source = self.source_norm(values).permute(0, 2, 3, 1)
        source = self.source_mix(source).permute(0, 3, 1, 2)
        values = values + self.dropout(source)
        return values + self.feature_mlp(self.feature_norm(values))


@dataclass(frozen=True)
class ScopeConfig:
    n_filters: int = 16
    n_sources: int = 8
    sinc_kernel: int = 65
    sinc_stride: int = 2
    width: int = 48
    mixer_blocks: int = 2
    dropout: float = 0.25
    moment_bins: tuple[int, ...] = (1, 2, 4, 8)
    adaptive_fir: bool = False


class _EnergyDecoder(nn.Module):
    def __init__(self, *, n_outputs: int, config: ScopeConfig) -> None:
        super().__init__()
        self.config = config
        self.moments = MultiResolutionMoments(config.moment_bins)
        # This direct path is deliberately unnormalized.  A multiplicative
        # ERD/ERS change becomes an additive shift in log energy; token-wise
        # LayerNorm would erase that primary MI signal.  The path is a strict
        # log-variance linear decoder analogous to the successful floor in
        # ShallowFBCSPNet/FBCNet.
        self.floor_classifier = nn.Linear(
            config.n_sources * config.n_filters, n_outputs
        )
        self.embedding = nn.Sequential(
            nn.Linear(self.moments.output_dim, config.width),
            nn.GELU(),
        )
        self.source_embedding = nn.Parameter(
            torch.zeros(1, config.n_sources, 1, config.width)
        )
        self.frequency_embedding = nn.Parameter(
            torch.zeros(1, 1, config.n_filters, config.width)
        )
        nn.init.normal_(self.source_embedding, std=0.02)
        nn.init.normal_(self.frequency_embedding, std=0.02)
        self.mixers = nn.ModuleList(
            AxialMixerBlock(
                width=config.width,
                n_sources=config.n_sources,
                dropout=config.dropout,
            )
            for _ in range(config.mixer_blocks)
        )
        self.output_norm = nn.LayerNorm(config.width)
        # A single fixed head retains source/frequency identity. Attention
        # pooling erased precisely the band-specific spatial evidence that
        # makes CSP/FBCNet effective on MI.
        self.classifier = nn.Linear(
            config.n_sources * config.n_filters * config.width, n_outputs
        )
        # Start as the transparent log-energy model. The higher-capacity
        # multiscale branch is learned as a residual rather than perturbing the
        # known-good inductive bias at initialization.
        nn.init.zeros_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def decode(self, sources: Tensor) -> Tensor:
        log_energy = torch.log(sources.square().mean(dim=-1).clamp_min(1e-6))
        floor_logits = self.floor_classifier(log_energy.flatten(start_dim=1))
        hidden = (
            self.embedding(self.moments(sources))
            + self.source_embedding
            + self.frequency_embedding
        )
        for mixer in self.mixers:
            hidden = mixer(hidden)
        hidden = self.output_norm(hidden)
        return floor_logits + self.classifier(hidden.flatten(start_dim=1))


def _make_filter_bank(config: ScopeConfig, sfreq: float) -> OrderedSincFilterBank:
    bank_type = AdaptiveSincFilterBank if config.adaptive_fir else OrderedSincFilterBank
    return bank_type(
        n_filters=config.n_filters,
        kernel_size=config.sinc_kernel,
        sfreq=sfreq,
        stride=config.sinc_stride,
    )


class ScopeNet(_EnergyDecoder):
    """Scalp-Coordinate Operator with Physical-frequency Energy (SCOPE-Net)."""

    uses_positions = True

    def __init__(
        self,
        *,
        n_outputs: int,
        sfreq: float = 128.0,
        config: ScopeConfig = ScopeConfig(),
    ) -> None:
        super().__init__(n_outputs=n_outputs, config=config)
        self.filter_bank = _make_filter_bank(config, sfreq)
        self.spatial_field = ContinuousScalpField(
            n_filters=config.n_filters,
            n_sources=config.n_sources,
        )

    def encode_sources(
        self,
        x: Tensor,
        positions: Tensor,
        channel_mask: Tensor | None = None,
    ) -> Tensor:
        filtered = self.filter_bank(x)
        frequencies = self.filter_bank.frequencies_hz()
        normalized = 2.0 * (
            (frequencies - self.filter_bank.frequency_low)
            / (self.filter_bank.frequency_high - self.filter_bank.frequency_low)
        ) - 1.0
        return self.spatial_field(filtered, positions, normalized, channel_mask)

    def forward(
        self,
        x: Tensor,
        positions: Tensor,
        channel_mask: Tensor | None = None,
    ) -> Tensor:
        return self.decode(self.encode_sources(x, positions, channel_mask))


class FreeScopeNet(_EnergyDecoder):
    """Parameter-matched channel-indexed control for the continuous field."""

    uses_positions = False

    def __init__(
        self,
        *,
        n_channels: int,
        n_outputs: int,
        sfreq: float = 128.0,
        config: ScopeConfig = ScopeConfig(),
    ) -> None:
        super().__init__(n_outputs=n_outputs, config=config)
        self.filter_bank = _make_filter_bank(config, sfreq)
        self.spatial_filter = FreeSpatialFilter(
            n_filters=config.n_filters,
            n_sources=config.n_sources,
            n_channels=n_channels,
        )

    def forward(self, x: Tensor, positions: Tensor | None = None) -> Tensor:
        del positions
        return self.decode(self.spatial_filter(self.filter_bank(x)))


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
