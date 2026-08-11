"""Complex-Hermitian surface-differential network for motor imagery EEG.

This module is an isolated research prototype.  It intentionally does not
register itself in the benchmark model factory; the frozen benchmark can add
that integration only after the primitives and protocol are reviewed.

The formal subject-specific factory retains each dataset's native montage;
an optional coordinate projection remains available as a robustness ablation.
Overlapping DPSS windows then produce positive-frequency multitaper
cross-spectral density (CSD) estimates. A strictly positive shrinkage term
turns complex coherency into Hermitian positive-definite (HPD) matrices, which
are represented by their matrix logarithms. The trainable path evaluates the
logarithm with a matrix-atanh series built from complex linear solves, avoiding
eigenvector derivatives at the repeated eigenvalues caused by low-channel
montages. An eigendecomposition implementation remains available as an
independent audit reference.

The Hermitian Surface-Differential (HSD) layer forms four *identity-anchored
log-domain finite-difference fields*: state, time difference, frequency
difference, and mixed time-frequency difference.  These are ordinary finite
differences in one shared matrix-log coordinate system; no Riemannian-geometry
claim is made for them.  Temporal and frequency differences are divided by
their physical spacing in seconds and hertz.

The main branch applies a compact factorized time/frequency decoder to the HSD
fields.  A separate branch retains log band power and an early-window-relative
ERD-like feature so that amplitude information discarded by coherency remains
available to the classifier.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


Band = tuple[float, float]


def fibonacci_hemisphere(n_points: int, *, min_z: float = 0.20) -> Tensor:
    """Return deterministic approximately uniform unit vectors on a hemisphere."""

    if n_points <= 0:
        raise ValueError("n_points must be positive")
    if not 0.0 <= min_z < 1.0:
        raise ValueError("min_z must lie in [0, 1)")
    indices = torch.arange(n_points, dtype=torch.float64)
    # Midpoint sampling avoids placing a virtual sensor exactly at either pole.
    z = min_z + (1.0 - min_z) * (indices + 0.5) / n_points
    radius = torch.sqrt(torch.clamp(1.0 - z.square(), min=0.0))
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    azimuth = indices * golden_angle
    points = torch.stack(
        (radius * torch.cos(azimuth), radius * torch.sin(azimuth), z),
        dim=-1,
    )
    return points.to(torch.float32)


def dpss_tapers(
    n_times: int,
    *,
    time_bandwidth: float = 2.5,
    n_tapers: int = 3,
) -> Tensor:
    """Construct orthonormal discrete prolate spheroidal (DPSS) tapers.

    The tapers are the leading eigenvectors of the discrete spectral
    concentration operator with half bandwidth ``time_bandwidth / n_times``.
    They are generated once in float64, sign-canonicalized for deterministic
    tests/checkpoints, and returned in float32.
    """

    if n_times < 4:
        raise ValueError("n_times must be at least four")
    if not 0.5 < time_bandwidth < n_times / 2.0:
        raise ValueError("time_bandwidth must lie in (0.5, n_times / 2)")
    if not 1 <= n_tapers <= n_times:
        raise ValueError("n_tapers must lie in [1, n_times]")
    # More than floor(2*NW) tapers have poor spectral concentration and are not
    # useful for this estimator.
    if n_tapers > max(1, math.floor(2.0 * time_bandwidth)):
        raise ValueError("n_tapers must not exceed floor(2 * time_bandwidth)")

    indices = torch.arange(n_times, dtype=torch.float64)
    offsets = indices[:, None] - indices[None, :]
    half_bandwidth = time_bandwidth / n_times
    concentration = 2.0 * half_bandwidth * torch.sinc(
        2.0 * half_bandwidth * offsets
    )
    _, eigenvectors = torch.linalg.eigh(concentration)
    tapers = eigenvectors[:, -n_tapers:].transpose(0, 1).flip(0).contiguous()

    # Eigenvector signs are arbitrary.  Make the entry with greatest magnitude
    # positive so serialized tapers are stable across repeated construction.
    peak_indices = tapers.abs().argmax(dim=1)
    peaks = tapers.gather(1, peak_indices[:, None]).squeeze(1)
    tapers = tapers * torch.where(peaks < 0.0, -1.0, 1.0)[:, None]
    return tapers.to(torch.float32)


class CoordinateProjector(nn.Module):
    """Smooth coordinate-only interpolation onto fixed virtual scalp sensors."""

    def __init__(
        self,
        n_virtual_sensors: int,
        *,
        initial_concentration: float = 12.0,
        minimum_concentration: float = 0.5,
    ) -> None:
        super().__init__()
        if initial_concentration <= minimum_concentration:
            raise ValueError(
                "initial_concentration must exceed minimum_concentration"
            )
        self.minimum_concentration = float(minimum_concentration)
        self.register_buffer(
            "virtual_positions", fibonacci_hemisphere(n_virtual_sensors)
        )
        shifted = initial_concentration - minimum_concentration
        self.raw_concentration = nn.Parameter(
            torch.tensor(math.log(math.expm1(shifted)), dtype=torch.float32)
        )

    def concentration(self) -> Tensor:
        return self.minimum_concentration + nn.functional.softplus(
            self.raw_concentration
        )

    def interpolation_weights(self, positions: Tensor) -> Tensor:
        """Return weights with shape ``(virtual, channels)`` or ``(B, V, C)``."""

        if positions.ndim not in (2, 3) or positions.shape[-1] != 3:
            raise ValueError("positions must have shape (channels, 3) or (B, C, 3)")
        positions = positions / torch.linalg.vector_norm(
            positions, dim=-1, keepdim=True
        ).clamp_min(1e-7)
        virtual = self.virtual_positions.to(device=positions.device, dtype=positions.dtype)
        if positions.ndim == 2:
            similarities = virtual @ positions.transpose(0, 1)
        else:
            similarities = torch.einsum("vd,bcd->bvc", virtual, positions)
        return torch.softmax(self.concentration().to(similarities) * similarities, dim=-1)

    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError("x must have shape (batch, channels, time)")
        if positions.shape[-2] != x.shape[1]:
            raise ValueError("positions and x must contain the same channel count")
        weights = self.interpolation_weights(positions)
        if weights.ndim == 2:
            return torch.einsum("vc,bct->bvt", weights.to(x), x)
        if weights.shape[0] != x.shape[0]:
            raise ValueError("batched positions must match the input batch size")
        return torch.einsum("bvc,bct->bvt", weights.to(x), x)


class NativeMontageProjector(nn.Module):
    """Identity adapter retaining every native channel and its ordering."""

    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError("x must have shape (batch, channels, time)")
        if positions.ndim not in (2, 3) or positions.shape[-1] != 3:
            raise ValueError("positions must have shape (channels, 3) or (B, C, 3)")
        if positions.shape[-2] != x.shape[1]:
            raise ValueError("positions and x must contain the same channel count")
        if positions.ndim == 3 and positions.shape[0] != x.shape[0]:
            raise ValueError("batched positions must match the input batch size")
        return x


class MultitaperCSD(nn.Module):
    """Overlapping DPSS estimator for positive-frequency band CSD matrices."""

    def __init__(
        self,
        *,
        sfreq: float = 128.0,
        window_length: int = 128,
        hop_length: int = 64,
        time_bandwidth: float = 2.5,
        n_tapers: int = 3,
        bands: tuple[Band, ...] = (
            (4.0, 8.0),
            (8.0, 12.0),
            (12.0, 16.0),
            (16.0, 24.0),
            (24.0, 32.0),
            (32.0, 40.0),
        ),
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        if sfreq != 128.0:
            raise ValueError(
                "CHSDNet is locked to 128 Hz for the frozen common benchmark"
            )
        if window_length < 4:
            raise ValueError("window_length must be at least four")
        if not 1 <= hop_length <= window_length:
            raise ValueError("hop_length must lie in [1, window_length]")
        if not bands:
            raise ValueError("at least one frequency band is required")
        previous_high: float | None = None
        for low, high in bands:
            if not 0.0 < low < high <= 40.0:
                raise ValueError("bands must be positive, ordered, and at most 40 Hz")
            if previous_high is not None and low < previous_high:
                raise ValueError("frequency bands must be non-overlapping and ordered")
            previous_high = high
        if eps <= 0.0:
            raise ValueError("eps must be positive")

        self.sfreq = float(sfreq)
        self.window_length = int(window_length)
        self.hop_length = int(hop_length)
        self.bands = tuple((float(low), float(high)) for low, high in bands)
        self.eps = float(eps)
        self.register_buffer(
            "tapers",
            dpss_tapers(
                window_length,
                time_bandwidth=time_bandwidth,
                n_tapers=n_tapers,
            ),
        )
        self.register_buffer(
            "frequencies_hz",
            torch.fft.rfftfreq(window_length, d=1.0 / sfreq),
        )

        # Validate bin coverage at construction rather than failing in a long
        # benchmark after the first forward pass.
        for band_index, (low, high) in enumerate(self.bands):
            inclusive_high = band_index == len(self.bands) - 1
            mask = (self.frequencies_hz >= low) & (
                self.frequencies_hz <= high
                if inclusive_high
                else self.frequencies_hz < high
            )
            if not bool(mask.any()):
                raise ValueError(f"band {(low, high)} contains no FFT bins")

    def _pad_for_complete_windows(self, x: Tensor) -> Tensor:
        n_times = x.shape[-1]
        target = max(n_times, self.window_length)
        remainder = (target - self.window_length) % self.hop_length
        if remainder:
            target += self.hop_length - remainder
        if target == n_times:
            return x
        return nn.functional.pad(x, (0, target - n_times))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return ``(csd, power, reliability)``.

        Shapes are ``(B, W, F, V, V)``, ``(B, W, F, V)``, and
        ``(B, W, F)``.  Reliability is an inverse dispersion score derived
        from agreement of the taper-wise band powers.
        """

        if x.ndim != 3:
            raise ValueError("x must have shape (batch, virtual_sensors, time)")
        if x.shape[-1] < 2:
            raise ValueError("x must contain at least two time samples")
        if not x.is_floating_point():
            raise TypeError("x must have a floating-point dtype")
        # CPU and CUDA FFTs are well supported in float32/64; explicitly escape
        # half precision under mixed precision for numerical stability.
        work = x if x.dtype in (torch.float32, torch.float64) else x.float()
        work = self._pad_for_complete_windows(work)
        windows = work.unfold(-1, self.window_length, self.hop_length)
        windows = windows - windows.mean(dim=-1, keepdim=True)
        tapered = windows.unsqueeze(-2) * self.tapers.to(work)[None, None, None]
        spectra = torch.fft.rfft(tapered, dim=-1, norm="ortho")
        # (B, virtual, windows, tapers, frequencies) -> (B, W, K, R, V)
        spectra = spectra.permute(0, 2, 3, 4, 1)

        csd_by_band: list[Tensor] = []
        reliability_by_band: list[Tensor] = []
        for band_index, (low, high) in enumerate(self.bands):
            inclusive_high = band_index == len(self.bands) - 1
            mask = (self.frequencies_hz >= low) & (
                self.frequencies_hz <= high
                if inclusive_high
                else self.frequencies_hz < high
            )
            selected = spectra[..., mask, :]
            n_atoms = selected.shape[-3] * selected.shape[-2]
            csd = torch.einsum(
                "bwksv,bwksu->bwvu", selected, selected.conj()
            ) / float(n_atoms)
            csd = 0.5 * (csd + csd.mH)
            csd_by_band.append(csd)

            taper_power = selected.abs().square().mean(dim=(-2, -1))
            log_taper_power = torch.log(taper_power.clamp_min(self.eps))
            dispersion = log_taper_power.var(dim=-1, unbiased=False)
            reliability_by_band.append(1.0 / (1.0 + dispersion))

        band_csd = torch.stack(csd_by_band, dim=2)
        power = band_csd.diagonal(dim1=-2, dim2=-1).real.clamp_min(self.eps)
        reliability = torch.stack(reliability_by_band, dim=2).clamp(0.0, 1.0)
        return band_csd, power, reliability


def shrink_complex_coherency(
    csd: Tensor,
    shrinkage: float | Tensor,
    *,
    eps: float = 1e-7,
) -> tuple[Tensor, Tensor]:
    """Convert complex CSD to shrinkage coherency and return its diagonal power."""

    if csd.ndim < 2 or csd.shape[-1] != csd.shape[-2]:
        raise ValueError("csd must end in square matrix dimensions")
    if not csd.is_complex():
        raise TypeError("csd must be a complex tensor")
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    csd = 0.5 * (csd + csd.mH)
    power = csd.diagonal(dim1=-2, dim2=-1).real.clamp_min(eps)
    scales = torch.sqrt(power.unsqueeze(-1) * power.unsqueeze(-2)).clamp_min(eps)
    coherency = csd / scales
    coherency = 0.5 * (coherency + coherency.mH)
    size = csd.shape[-1]
    identity = torch.eye(size, device=csd.device, dtype=csd.dtype)
    # Enforce the exact unit diagonal before shrinkage; FFT roundoff can
    # otherwise leave tiny imaginary or non-unit diagonal components.
    coherency = (
        coherency
        - torch.diag_embed(coherency.diagonal(dim1=-2, dim2=-1))
        + identity
    )
    amount = torch.as_tensor(shrinkage, device=csd.device, dtype=power.dtype)
    if amount.numel() != 1 or not bool(torch.isfinite(amount).all()):
        raise ValueError("shrinkage must be a finite scalar")
    if not bool((amount > 0.0) & (amount <= 1.0)):
        raise ValueError("shrinkage must lie in (0, 1]")
    coherency = (1.0 - amount) * coherency + amount * identity
    coherency = 0.5 * (coherency + coherency.mH)
    return coherency, power


def hermitian_matrix_log(matrix: Tensor, *, eps: float = 1e-6) -> Tensor:
    """Spectral matrix logarithm retained as an audit/reference implementation.

    This eigendecomposition path is accurate in the forward direction, but its
    eigenvector derivative is not defined within a repeated eigenspace.  Use
    :func:`hermitian_matrix_log_series` in trainable model paths.
    """

    if matrix.ndim < 2 or matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError("matrix must end in square matrix dimensions")
    if not matrix.is_complex():
        raise TypeError("matrix must be a complex tensor")
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    hermitian = 0.5 * (matrix + matrix.mH)
    eigenvalues, eigenvectors = torch.linalg.eigh(hermitian)
    log_eigenvalues = torch.log(eigenvalues.clamp_min(eps))
    logged = (eigenvectors * log_eigenvalues.unsqueeze(-2)) @ eigenvectors.mH
    return 0.5 * (logged + logged.mH)


def hermitian_log_series_error_bound(
    matrix_size: int,
    shrinkage: float,
    n_terms: int,
) -> float:
    """Bound the atanh-series error for a shrinkage coherency matrix.

    A unit-diagonal coherency matrix has trace ``matrix_size`` and eigenvalues
    in ``[0, matrix_size]``.  Shrinkage by ``alpha`` therefore places its
    eigenvalues in ``[alpha, alpha + (1-alpha)*matrix_size]``.  Mapping an
    eigenvalue with ``y=(lambda-1)/(lambda+1)`` gives a uniform spectral-radius
    bound ``rho < 1``.  For ``K`` retained terms, the omitted matrix-log tail is

    ``2 * rho**(2K+1) / ((2K+1) * (1-rho**2))``.
    """

    if matrix_size <= 0:
        raise ValueError("matrix_size must be positive")
    if not math.isfinite(shrinkage) or not 0.0 < shrinkage <= 1.0:
        raise ValueError("shrinkage must lie in (0, 1]")
    if n_terms <= 0:
        raise ValueError("n_terms must be positive")
    minimum_eigenvalue = shrinkage
    maximum_eigenvalue = (
        shrinkage + (1.0 - shrinkage) * float(matrix_size)
    )
    lower_radius = abs(
        (minimum_eigenvalue - 1.0) / (minimum_eigenvalue + 1.0)
    )
    upper_radius = abs(
        (maximum_eigenvalue - 1.0) / (maximum_eigenvalue + 1.0)
    )
    radius = max(lower_radius, upper_radius)
    if radius == 0.0:
        return 0.0
    exponent = 2 * n_terms + 1
    return (
        2.0
        * radius**exponent
        / (float(exponent) * (1.0 - radius * radius))
    )


def hermitian_matrix_log_series(
    matrix: Tensor,
    *,
    n_terms: int = 32,
) -> Tensor:
    """Differentiable Hermitian matrix logarithm using a matrix-atanh series.

    For HPD ``A``, define ``Y = (A-I)(A+I)^-1``.  Since both factors are
    polynomials in ``A``, the implementation may use the numerically convenient
    equivalent solve ``(A+I)Y = A-I``.  It then evaluates

    ``log(A) = 2 * sum(Y**(2k+1)/(2k+1), k=0..K-1)``.

    Only complex solves and matrix products participate in autograd, so
    repeated eigenvalues do not introduce undefined eigenvector gradients.
    HSDLayer separately validates a conservative truncation-error bound from
    its matrix size and fixed shrinkage.
    """

    if matrix.ndim < 2 or matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError("matrix must end in square matrix dimensions")
    if not matrix.is_complex():
        raise TypeError("matrix must be a complex tensor")
    if n_terms <= 0:
        raise ValueError("n_terms must be positive")
    hermitian = 0.5 * (matrix + matrix.mH)
    size = matrix.shape[-1]
    identity = torch.eye(size, device=matrix.device, dtype=matrix.dtype)
    y = torch.linalg.solve(hermitian + identity, hermitian - identity)
    y = 0.5 * (y + y.mH)
    y_squared = y @ y
    odd_power = y
    series = y
    for term_index in range(1, n_terms):
        odd_power = odd_power @ y_squared
        series = series + odd_power / float(2 * term_index + 1)
    logged = 2.0 * series
    return 0.5 * (logged + logged.mH)


def hermitian_vech(matrix: Tensor) -> Tensor:
    """Isometrically vectorize a complex Hermitian matrix into real coordinates.

    Diagonal entries are retained once.  Real and imaginary parts of the strict
    upper triangle are multiplied by ``sqrt(2)``, preserving the Frobenius norm.
    An ``n x n`` Hermitian matrix therefore maps to exactly ``n**2`` reals.
    """

    if matrix.ndim < 2 or matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError("matrix must end in square matrix dimensions")
    if not matrix.is_complex():
        raise TypeError("matrix must be a complex tensor")
    size = matrix.shape[-1]
    rows, columns = torch.triu_indices(
        size, size, offset=1, device=matrix.device
    )
    upper = matrix[..., rows, columns]
    scale = math.sqrt(2.0)
    return torch.cat(
        (
            matrix.diagonal(dim1=-2, dim2=-1).real,
            scale * upper.real,
            scale * upper.imag,
        ),
        dim=-1,
    )


def hermitian_unvech(vector: Tensor, size: int) -> Tensor:
    """Inverse of :func:`hermitian_vech`, primarily for audit tests."""

    if size <= 0:
        raise ValueError("size must be positive")
    if vector.shape[-1] != size * size:
        raise ValueError("the vector length must equal size**2")
    pair_count = size * (size - 1) // 2
    diagonal = vector[..., :size]
    real_upper = vector[..., size : size + pair_count] / math.sqrt(2.0)
    imag_upper = vector[..., size + pair_count :] / math.sqrt(2.0)
    rows, columns = torch.triu_indices(
        size, size, offset=1, device=vector.device
    )
    complex_diagonal = torch.complex(diagonal, torch.zeros_like(diagonal))
    matrix = torch.diag_embed(complex_diagonal)
    upper = torch.zeros(
        *vector.shape[:-1],
        size,
        size,
        device=vector.device,
        dtype=complex_diagonal.dtype,
    )
    upper[..., rows, columns] = torch.complex(real_upper, imag_upper)
    return matrix + upper + upper.mH


def log_domain_difference_fields(
    state: Tensor,
    reliability: Tensor,
    *,
    time_step_seconds: float | Tensor,
    band_centers_hz: Tensor,
    eps: float = 1e-7,
) -> tuple[Tensor, Tensor]:
    """Build physically scaled state/difference fields and reliability.

    ``state`` has shape ``(B, W, F, D)`` and ``reliability`` has shape
    ``(B, W, F)``.  The returned fields have shape ``(B, W, F, 4, D)`` in the
    order state, time difference, frequency difference, mixed difference.
    Field reliability has shape ``(B, W, F, 4)``.  Undefined boundary
    differences are represented by zeros with zero reliability.

    Temporal differences are divided by ``time_step_seconds``.  Frequency
    differences use the adjacent (potentially nonuniform) gaps in
    ``band_centers_hz``.  The mixed numerator is divided by both spacings.
    """

    if state.ndim != 4:
        raise ValueError("state must have shape (batch, windows, bands, features)")
    if reliability.shape != state.shape[:-1]:
        raise ValueError("reliability must match the first three state dimensions")
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    batch, n_windows, n_bands, n_features = state.shape
    time_step = torch.as_tensor(
        time_step_seconds, device=state.device, dtype=state.dtype
    )
    if time_step.numel() != 1 or not bool(torch.isfinite(time_step).all()):
        raise ValueError("time_step_seconds must be a finite scalar")
    if not bool(time_step > 0.0):
        raise ValueError("time_step_seconds must be positive")
    centers = torch.as_tensor(
        band_centers_hz, device=state.device, dtype=state.dtype
    )
    if centers.ndim != 1 or centers.numel() != n_bands:
        raise ValueError("band_centers_hz must have one entry per state band")
    if not bool(torch.isfinite(centers).all()):
        raise ValueError("band_centers_hz must be finite")
    frequency_steps = centers[1:] - centers[:-1]
    if frequency_steps.numel() and not bool((frequency_steps > 0.0).all()):
        raise ValueError("band_centers_hz must be strictly increasing")

    zero_time = state.new_zeros(batch, 1, n_bands, n_features)
    time_difference = torch.cat(
        (zero_time, (state[:, 1:] - state[:, :-1]) / time_step), dim=1
    )
    zero_frequency = state.new_zeros(batch, n_windows, 1, n_features)
    frequency_core = (state[:, :, 1:] - state[:, :, :-1]) / frequency_steps[
        None, None, :, None
    ]
    frequency_difference = torch.cat(
        (zero_frequency, frequency_core), dim=2
    )
    mixed_core = (
        state[:, 1:, 1:]
        - state[:, :-1, 1:]
        - state[:, 1:, :-1]
        + state[:, :-1, :-1]
    ) / (time_step * frequency_steps[None, None, :, None])
    mixed_with_frequency_boundary = torch.cat(
        (
            state.new_zeros(batch, max(n_windows - 1, 0), 1, n_features),
            mixed_core,
        ),
        dim=2,
    )
    mixed_difference = torch.cat(
        (zero_time, mixed_with_frequency_boundary), dim=1
    )

    reliability = reliability.clamp(0.0, 1.0)
    time_weight = torch.cat(
        (
            reliability.new_zeros(batch, 1, n_bands),
            torch.sqrt((reliability[:, 1:] * reliability[:, :-1]).clamp_min(eps)),
        ),
        dim=1,
    )
    frequency_weight = torch.cat(
        (
            reliability.new_zeros(batch, n_windows, 1),
            torch.sqrt(
                (reliability[:, :, 1:] * reliability[:, :, :-1]).clamp_min(eps)
            ),
        ),
        dim=2,
    )
    mixed_weight_core = (
        reliability[:, 1:, 1:]
        * reliability[:, :-1, 1:]
        * reliability[:, 1:, :-1]
        * reliability[:, :-1, :-1]
    ).clamp_min(eps).pow(0.25)
    mixed_weight = torch.cat(
        (
            reliability.new_zeros(batch, max(n_windows - 1, 0), 1),
            mixed_weight_core,
        ),
        dim=2,
    )
    mixed_weight = torch.cat(
        (reliability.new_zeros(batch, 1, n_bands), mixed_weight), dim=1
    )
    fields = torch.stack(
        (state, time_difference, frequency_difference, mixed_difference), dim=-2
    )
    weights = torch.stack(
        (reliability, time_weight, frequency_weight, mixed_weight), dim=-1
    )
    return fields, weights


class HSDLayer(nn.Module):
    """Hermitian Surface-Differential log-domain feature layer."""

    def __init__(
        self,
        n_virtual_sensors: int,
        *,
        fixed_shrinkage: float = 0.10,
        matrix_log_terms: int = 32,
        maximum_series_error: float = 1e-5,
    ) -> None:
        super().__init__()
        if n_virtual_sensors <= 0:
            raise ValueError("n_virtual_sensors must be positive")
        if not math.isfinite(fixed_shrinkage) or not 0.0 < fixed_shrinkage <= 1.0:
            raise ValueError("fixed_shrinkage must lie in (0, 1]")
        if matrix_log_terms <= 0:
            raise ValueError("matrix_log_terms must be positive")
        if not math.isfinite(maximum_series_error) or maximum_series_error <= 0.0:
            raise ValueError("maximum_series_error must be finite and positive")
        self.n_virtual_sensors = int(n_virtual_sensors)
        self.matrix_log_terms = int(matrix_log_terms)
        self.maximum_series_error = float(maximum_series_error)
        self.series_error_bound = hermitian_log_series_error_bound(
            self.n_virtual_sensors,
            float(fixed_shrinkage),
            self.matrix_log_terms,
        )
        if self.series_error_bound > self.maximum_series_error:
            raise ValueError(
                "matrix-log series bound "
                f"{self.series_error_bound:.3e} exceeds "
                f"{self.maximum_series_error:.3e}; increase matrix_log_terms "
                "or use a supported matrix-size/shrinkage combination"
            )
        self.register_buffer(
            "fixed_shrinkage",
            torch.tensor(float(fixed_shrinkage), dtype=torch.float32),
        )

    @property
    def feature_dim(self) -> int:
        return self.n_virtual_sensors**2

    def shrinkage(self) -> Tensor:
        return self.fixed_shrinkage

    def forward(
        self,
        csd: Tensor,
        reliability: Tensor,
        *,
        time_step_seconds: float | Tensor,
        band_centers_hz: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        coherency, power = shrink_complex_coherency(csd, self.shrinkage())
        log_coherency = hermitian_matrix_log_series(
            coherency, n_terms=self.matrix_log_terms
        )
        state = hermitian_vech(log_coherency)
        fields, field_reliability = log_domain_difference_fields(
            state,
            reliability,
            time_step_seconds=time_step_seconds,
            band_centers_hz=band_centers_hz,
        )
        return fields, field_reliability, power


class _FactorizedTFBlock(nn.Module):
    """Small residual decoder factored into temporal and frequency axes."""

    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(1, width)
        self.temporal = nn.Conv2d(
            width,
            width,
            kernel_size=(3, 1),
            padding=(1, 0),
            groups=width,
            bias=False,
        )
        self.temporal_mix = nn.Conv2d(width, width, kernel_size=1)
        self.frequency = nn.Conv2d(
            width,
            width,
            kernel_size=(1, 3),
            padding=(0, 1),
            groups=width,
            bias=False,
        )
        self.frequency_mix = nn.Conv2d(width, width, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout)
        self.residual_scale = nn.Parameter(torch.tensor(0.10))

    def forward(self, values: Tensor) -> Tensor:
        residual = values
        values = self.norm(values)
        values = nn.functional.gelu(self.temporal_mix(self.temporal(values)))
        values = self.frequency_mix(self.frequency(values))
        values = self.dropout(values)
        return residual + self.residual_scale * values


@dataclass(frozen=True)
class CHSDConfig:
    """Configuration for the common-protocol CHSDNet prototype."""

    sfreq: float = 128.0
    n_virtual_sensors: int = 8
    window_length: int = 128
    hop_length: int = 64
    time_bandwidth: float = 2.5
    n_tapers: int = 3
    bands: tuple[Band, ...] = (
        (4.0, 8.0),
        (8.0, 12.0),
        (12.0, 16.0),
        (16.0, 24.0),
        (24.0, 32.0),
        (32.0, 40.0),
    )
    projection_concentration: float = 12.0
    use_coordinate_projection: bool = True
    shrinkage: float = 0.10
    matrix_log_terms: int = 32
    maximum_series_error: float = 1e-5
    hsd_width: int = 32
    power_width: int = 16
    decoder_blocks: int = 2
    dropout: float = 0.10
    bandwise_pooling: bool = True
    eps: float = 1e-7

    def __post_init__(self) -> None:
        if self.sfreq != 128.0:
            raise ValueError(
                "CHSDNet is locked to 128 Hz for the frozen common benchmark"
            )
        if self.n_virtual_sensors < 2:
            raise ValueError("n_virtual_sensors must be at least two")
        if self.hsd_width <= 0 or self.power_width <= 0:
            raise ValueError("branch widths must be positive")
        if self.decoder_blocks <= 0:
            raise ValueError("decoder_blocks must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if not 0.02 < self.shrinkage < 0.50:
            raise ValueError("shrinkage must lie strictly between 0.02 and 0.50")
        if self.matrix_log_terms <= 0:
            raise ValueError("matrix_log_terms must be positive")
        if (
            not math.isfinite(self.maximum_series_error)
            or self.maximum_series_error <= 0.0
        ):
            raise ValueError("maximum_series_error must be finite and positive")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive")


class CHSDNet(nn.Module):
    """Complex-Hermitian Surface-Differential Network.

    The model accepts ``(batch, channels, time)`` EEG and unit-scaled electrode
    coordinates.  ``n_outputs`` is deliberately restricted to the two- and
    four-class motor-imagery tasks in the study.
    """

    uses_positions = True

    def __init__(
        self,
        n_outputs: int,
        *,
        config: CHSDConfig | None = None,
    ) -> None:
        super().__init__()
        if n_outputs not in (2, 4):
            raise ValueError("CHSDNet supports exactly two or four outputs")
        self.n_outputs = int(n_outputs)
        self.config = config or CHSDConfig()
        n_virtual = self.config.n_virtual_sensors

        if self.config.use_coordinate_projection:
            self.projector: nn.Module = CoordinateProjector(
                n_virtual,
                initial_concentration=self.config.projection_concentration,
            )
        else:
            self.projector = NativeMontageProjector()
        self.spectral_estimator = MultitaperCSD(
            sfreq=self.config.sfreq,
            window_length=self.config.window_length,
            hop_length=self.config.hop_length,
            time_bandwidth=self.config.time_bandwidth,
            n_tapers=self.config.n_tapers,
            bands=self.config.bands,
            eps=self.config.eps,
        )
        self.hsd = HSDLayer(
            n_virtual,
            fixed_shrinkage=self.config.shrinkage,
            matrix_log_terms=self.config.matrix_log_terms,
            maximum_series_error=self.config.maximum_series_error,
        )
        self.time_step_seconds = self.config.hop_length / self.config.sfreq
        self.register_buffer(
            "band_centers_hz",
            torch.tensor(
                [(low + high) / 2.0 for low, high in self.config.bands],
                dtype=torch.float32,
            ),
        )

        feature_dim = self.hsd.feature_dim
        self.field_norms = nn.ModuleList(
            nn.LayerNorm(feature_dim) for _ in range(4)
        )
        self.field_projections = nn.ModuleList(
            nn.Linear(feature_dim, self.config.hsd_width) for _ in range(4)
        )
        # Begin from a state-dominant representation. The directional fields
        # are residual evidence channels and can grow when validation data
        # support them, without overwhelming the state at initialization.
        self.field_gates = nn.Parameter(
            torch.tensor((2.0, -2.0, -2.0, -2.0), dtype=torch.float32)
        )
        self.hsd_decoder = nn.Sequential(
            *(
                _FactorizedTFBlock(
                    self.config.hsd_width, self.config.dropout
                )
                for _ in range(self.config.decoder_blocks)
            )
        )

        power_features = 2 * n_virtual
        self.power_norm = nn.LayerNorm(power_features)
        self.power_projection = nn.Linear(
            power_features, self.config.power_width
        )
        self.power_decoder = _FactorizedTFBlock(
            self.config.power_width, self.config.dropout
        )

        pooled_bands = len(self.config.bands) if self.config.bandwise_pooling else 1
        combined_width = pooled_bands * (
            self.config.hsd_width + self.config.power_width
        ) + 2
        self.classifier = nn.Sequential(
            nn.LayerNorm(combined_width),
            nn.Linear(combined_width, self.config.hsd_width),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hsd_width, self.n_outputs),
        )

    @staticmethod
    def _weighted_pool(
        values: Tensor,
        reliability: Tensor,
        *,
        preserve_frequency: bool,
    ) -> Tensor:
        """Reliability-pool a ``(B, width, W, F)`` grid.

        The default architecture pools over time but keeps the ordered band
        axis before flattening.  Motor-imagery effects are strongly
        frequency-localized, so collapsing mu and beta locations into one
        translation-invariant global average would discard useful identity.
        The global mode remains available as a prespecified ablation.
        """

        weights = reliability[:, None].to(values)
        if preserve_frequency:
            numerator = (values * weights).sum(dim=-2)
            denominator = weights.sum(dim=-2).clamp_min(1e-7)
            return (numerator / denominator).flatten(start_dim=1)
        numerator = (values * weights).sum(dim=(-2, -1))
        denominator = weights.sum(dim=(-2, -1)).clamp_min(1e-7)
        return numerator / denominator

    def forward_features(self, x: Tensor, positions: Tensor) -> dict[str, Tensor]:
        if x.ndim != 3:
            raise ValueError("x must have shape (batch, channels, time)")
        projected = self.projector(x, positions)
        csd, _, reliability = self.spectral_estimator(projected)
        fields, field_reliability, power = self.hsd(
            csd,
            reliability,
            time_step_seconds=self.time_step_seconds,
            band_centers_hz=self.band_centers_hz,
        )

        embedded: Tensor | None = None
        gates = torch.sigmoid(self.field_gates)
        for field_index, (normalizer, projection) in enumerate(
            zip(self.field_norms, self.field_projections, strict=True)
        ):
            field = fields[..., field_index, :]
            field_embedding = projection(normalizer(field))
            field_embedding = field_embedding * field_reliability[
                ..., field_index, None
            ]
            contribution = gates[field_index] * field_embedding
            embedded = contribution if embedded is None else embedded + contribution
        assert embedded is not None
        hsd_grid = self.hsd_decoder(embedded.permute(0, 3, 1, 2))
        hsd_summary = self._weighted_pool(
            hsd_grid,
            reliability,
            preserve_frequency=self.config.bandwise_pooling,
        )

        log_power = torch.log(power.clamp_min(self.config.eps))
        # This is an early-window-relative ERD-like contrast.  It is not called
        # a prestimulus baseline unless a dataset actually supplies one.
        reference_windows = max(1, (log_power.shape[1] + 3) // 4)
        early_reference = log_power[:, :reference_windows].mean(
            dim=1, keepdim=True
        )
        relative_power = log_power - early_reference
        power_features = torch.cat((log_power, relative_power), dim=-1)
        power_grid = self.power_projection(self.power_norm(power_features))
        power_grid = self.power_decoder(power_grid.permute(0, 3, 1, 2))
        power_summary = self._weighted_pool(
            power_grid,
            reliability,
            preserve_frequency=self.config.bandwise_pooling,
        )

        reliability_mean = reliability.mean(dim=(1, 2))
        reliability_std = reliability.std(dim=(1, 2), unbiased=False)
        summary = torch.cat(
            (
                hsd_summary,
                power_summary,
                reliability_mean[:, None],
                reliability_std[:, None],
            ),
            dim=-1,
        )
        return {
            "summary": summary,
            "projected": projected,
            "csd": csd,
            "reliability": reliability,
            "fields": fields,
            "field_reliability": field_reliability,
            "log_power": log_power,
            "relative_power": relative_power,
        }

    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        return self.classifier(self.forward_features(x, positions)["summary"])


def make_chsd_model(
    *,
    requested_model: str,
    n_channels: int,
    n_outputs: int,
    n_times: int,
    sfreq: float,
    channel_names: tuple[str, ...],
    channel_positions: Tensor,
) -> CHSDNet:
    """Construct the frozen common-protocol CHSDNet for the grid runner.

    The explicit signature is part of the full-grid factory contract.  Input
    metadata are validated here even though the coordinate-aware model receives
    the actual positions at every forward call.
    """

    if requested_model != "chsdnet":
        raise ValueError(f"unsupported requested model {requested_model!r}")
    if n_channels <= 0 or n_times <= 1:
        raise ValueError("n_channels and n_times must describe a nonempty trial")
    if len(channel_names) != n_channels:
        raise ValueError("channel_names length must equal n_channels")
    if channel_positions.shape != (n_channels, 3):
        raise ValueError("channel_positions must have shape (n_channels, 3)")
    if not bool(torch.isfinite(channel_positions).all()):
        raise ValueError("channel_positions must be finite")
    return CHSDNet(
        n_outputs,
        config=CHSDConfig(
            sfreq=float(sfreq),
            n_virtual_sensors=n_channels,
            use_coordinate_projection=False,
            # The conservative atanh tail bound remains below 1e-5 at the
            # largest locked montage (21 channels).
            matrix_log_terms=64,
        ),
    )


__all__ = [
    "Band",
    "CHSDConfig",
    "CHSDNet",
    "CoordinateProjector",
    "HSDLayer",
    "MultitaperCSD",
    "NativeMontageProjector",
    "dpss_tapers",
    "fibonacci_hemisphere",
    "hermitian_log_series_error_bound",
    "hermitian_matrix_log",
    "hermitian_matrix_log_series",
    "hermitian_unvech",
    "hermitian_vech",
    "log_domain_difference_fields",
    "make_chsd_model",
    "shrink_complex_coherency",
]
