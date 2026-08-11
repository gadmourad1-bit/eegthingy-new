"""Direct additive decoder for native-montage CHSD fields.

This experimental variant keeps every time, frequency, and channel-pair
coordinate instead of compressing each Hermitian matrix through a shared
low-rank embedding.  A state head is active at initialization; temporal,
spectral, mixed, and power heads start at exactly zero and can earn residual
influence during source-only optimization.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from .chsd import HSDLayer, MultitaperCSD


DIRECT_MODEL_NAME = "chsdnet_direct"


class _FrozenFeatureStandardizer(nn.Module):
    """Train-row-only per-coordinate standardization stored in model buffers."""

    def __init__(self, n_features: int, *, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        self.register_buffer("mean", torch.zeros(n_features))
        self.register_buffer("inverse_std", torch.ones(n_features))
        self.register_buffer("fitted", torch.tensor(False))

    def install(self, mean: Tensor, variance: Tensor) -> None:
        if mean.shape != self.mean.shape or variance.shape != self.mean.shape:
            raise ValueError("feature statistics have the wrong shape")
        if not bool(torch.isfinite(mean).all()) or not bool(
            torch.isfinite(variance).all()
        ):
            raise ValueError("feature statistics must be finite")
        with torch.no_grad():
            self.mean.copy_(mean.to(self.mean))
            self.inverse_std.copy_(
                torch.rsqrt(variance.to(self.inverse_std).clamp_min(self.eps))
            )
            self.fitted.fill_(True)

    def forward(self, values: Tensor) -> Tensor:
        return (values - self.mean.to(values)) * self.inverse_std.to(values)


@dataclass(frozen=True)
class CHSDDirectConfig:
    sfreq: float = 128.0
    window_length: int = 128
    hop_length: int = 64
    time_bandwidth: float = 2.5
    n_tapers: int = 3
    bands: tuple[tuple[float, float], ...] = (
        (4.0, 8.0),
        (8.0, 12.0),
        (12.0, 16.0),
        (16.0, 24.0),
        (24.0, 32.0),
        (32.0, 40.0),
    )
    shrinkage: float = 0.10
    matrix_log_terms: int = 64
    maximum_series_error: float = 1e-5
    residual_scale: float = 0.10
    eps: float = 1e-7

    def __post_init__(self) -> None:
        if self.sfreq != 128.0:
            raise ValueError("CHSDDirectNet is locked to 128 Hz")
        if self.window_length <= 1 or not 1 <= self.hop_length <= self.window_length:
            raise ValueError("invalid window/hop configuration")
        if not 0.0 < self.residual_scale <= 1.0:
            raise ValueError("residual_scale must lie in (0, 1]")


class CHSDDirectNet(nn.Module):
    """Native complex-Hermitian state plus zero-started additive field heads."""

    uses_positions = True

    def __init__(
        self,
        *,
        n_channels: int,
        n_outputs: int,
        n_times: int,
        config: CHSDDirectConfig | None = None,
    ) -> None:
        super().__init__()
        if n_channels <= 0 or n_times <= 1:
            raise ValueError("n_channels and n_times must be positive")
        if n_outputs not in (2, 4):
            raise ValueError("CHSDDirectNet supports exactly two or four outputs")
        self.n_channels = int(n_channels)
        self.n_outputs = int(n_outputs)
        self.n_times = int(n_times)
        self.config = config or CHSDDirectConfig()
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
            self.n_channels,
            fixed_shrinkage=self.config.shrinkage,
            matrix_log_terms=self.config.matrix_log_terms,
            maximum_series_error=self.config.maximum_series_error,
        )
        self.register_buffer(
            "band_centers_hz",
            torch.tensor(
                [(low + high) / 2.0 for low, high in self.config.bands],
                dtype=torch.float32,
            ),
        )
        self.time_step_seconds = self.config.hop_length / self.config.sfreq
        padded_times = max(self.n_times, self.config.window_length)
        remainder = (padded_times - self.config.window_length) % self.config.hop_length
        if remainder:
            padded_times += self.config.hop_length - remainder
        self.n_windows = (
            1
            + (padded_times - self.config.window_length)
            // self.config.hop_length
        )
        self.n_bands = len(self.config.bands)
        state_features = (
            self.n_windows * self.n_bands * self.n_channels * self.n_channels
        )
        power_features = (
            self.n_windows * self.n_bands * 2 * self.n_channels
        )
        self.state_norm = _FrozenFeatureStandardizer(state_features)
        self.state_head = nn.Linear(state_features, self.n_outputs)
        self.residual_norms = nn.ModuleList(
            _FrozenFeatureStandardizer(state_features) for _ in range(3)
        )
        self.residual_heads = nn.ModuleList(
            nn.Linear(state_features, self.n_outputs) for _ in range(3)
        )
        self.power_norm = _FrozenFeatureStandardizer(power_features)
        self.power_head = nn.Linear(power_features, self.n_outputs)
        with torch.no_grad():
            for head in (*self.residual_heads, self.power_head):
                head.weight.zero_()
                if head.bias is not None:
                    head.bias.zero_()

    def forward_features(self, x: Tensor, positions: Tensor) -> dict[str, Tensor]:
        if x.ndim != 3 or x.shape[1:] != (self.n_channels, self.n_times):
            raise ValueError(
                "x must match the construction-time channel/time dimensions"
            )
        if positions.ndim != 2 or positions.shape != (self.n_channels, 3):
            raise ValueError("positions must have shape (n_channels, 3)")
        csd, _, reliability = self.spectral_estimator(x)
        fields, field_reliability, power = self.hsd(
            csd,
            reliability,
            time_step_seconds=self.time_step_seconds,
            band_centers_hz=self.band_centers_hz,
        )
        flattened_fields = tuple(
            fields[..., index, :].flatten(start_dim=1) for index in range(4)
        )
        log_power = torch.log(power.clamp_min(self.config.eps))
        reference_windows = max(1, (log_power.shape[1] + 3) // 4)
        early_reference = log_power[:, :reference_windows].mean(
            dim=1, keepdim=True
        )
        relative_power = log_power - early_reference
        power_features = torch.cat((log_power, relative_power), dim=-1).flatten(
            start_dim=1
        )
        return {
            "state": flattened_fields[0],
            "temporal": flattened_fields[1],
            "spectral": flattened_fields[2],
            "mixed": flattened_fields[3],
            "power": power_features,
            "field_reliability": field_reliability,
        }

    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        features = self.forward_features(x, positions)
        logits = self.state_head(self.state_norm(features["state"]))
        residual = self.power_head(self.power_norm(features["power"]))
        for name, normalizer, head in zip(
            ("temporal", "spectral", "mixed"),
            self.residual_norms,
            self.residual_heads,
            strict=True,
        ):
            residual = residual + head(normalizer(features[name]))
        return logits + self.config.residual_scale * residual

    @torch.no_grad()
    def fit_source_statistics(
        self,
        x_source: Tensor,
        positions: Tensor,
        *,
        batch_size: int = 64,
    ) -> None:
        """Fit feature location/scale using source rows only.

        This hook is called before optimization for model selection and is
        refitted after reset on train-plus-validation rows. It never uses a
        validation row during selection or a prediction-only test row.
        """

        if x_source.ndim != 3 or x_source.shape[1:] != (
            self.n_channels,
            self.n_times,
        ):
            raise ValueError("x_source has the wrong trial shape")
        if len(x_source) < 2 or batch_size <= 0:
            raise ValueError("at least two source trials and a positive batch are required")
        normalizers = (self.state_norm, *self.residual_norms, self.power_norm)
        sums = [
            normalizer.mean.new_zeros(normalizer.mean.shape)
            for normalizer in normalizers
        ]
        square_sums = [value.clone() for value in sums]
        count = 0
        was_training = self.training
        self.eval()
        for start in range(0, len(x_source), batch_size):
            features = self.forward_features(
                x_source[start : start + batch_size],
                positions,
            )
            values = (
                features["state"],
                features["temporal"],
                features["spectral"],
                features["mixed"],
                features["power"],
            )
            for index, value in enumerate(values):
                sums[index].add_(value.sum(dim=0))
                square_sums[index].add_(value.square().sum(dim=0))
            count += len(values[0])
        for normalizer, total, square_total in zip(
            normalizers, sums, square_sums, strict=True
        ):
            mean = total / float(count)
            variance = square_total / float(count) - mean.square()
            normalizer.install(mean, variance.clamp_min(0.0))
        self.train(was_training)


def make_chsd_direct_model(
    *,
    requested_model: str,
    n_channels: int,
    n_outputs: int,
    n_times: int,
    sfreq: float,
    channel_names: tuple[str, ...],
    channel_positions: Tensor,
) -> CHSDDirectNet:
    """Construct the direct CHSD decoder through the grid factory contract."""

    if requested_model != DIRECT_MODEL_NAME:
        raise ValueError(f"unsupported requested model {requested_model!r}")
    if len(channel_names) != n_channels or len(set(channel_names)) != n_channels:
        raise ValueError("channel_names must be unique and match n_channels")
    positions = torch.as_tensor(channel_positions, dtype=torch.float32)
    if positions.shape != (n_channels, 3) or not bool(torch.isfinite(positions).all()):
        raise ValueError("channel_positions must be finite with shape (n_channels, 3)")
    config = CHSDDirectConfig(sfreq=float(sfreq))
    return CHSDDirectNet(
        n_channels=n_channels,
        n_outputs=n_outputs,
        n_times=n_times,
        config=config,
    )


__all__ = [
    "CHSDDirectConfig",
    "CHSDDirectNet",
    "DIRECT_MODEL_NAME",
    "make_chsd_direct_model",
]
