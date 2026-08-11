"""PARITY-Fuse: compact, internally equivariant raw/tangent EEG fusion.

The binary left/right motor-imagery problem admits a useful order-two action:
sagittal channel reflection exchanges the two labels.  PARITY-Fuse builds that
``Z2`` action into its latent representation.  A shared encoder processes an
epoch and its reflected partner, after which the two encodings are decomposed
into invariant (even) and sign-changing (odd) parts.  The tangent path performs
the same decomposition after a shared, band-structured encoder.

Fusion happens *per sample and per latent coordinate*.  Its gate may observe
even latents, squared odd latents, and products of raw/tangent odd latents; all
of these are invariant to exchanging the paired views.  It therefore cannot
break the signed output action.  A bias-free residual readout is added to the
odd projection of a frozen source-only tangent logistic classifier.  The
residual readout starts at exactly zero, so the untrained neural model is
exactly the projected convex anchor rather than a random perturbation of it.

There is deliberately no dropout, BatchNorm, or other batch/stochastic state in
the paired encoders.  Original/reflected examples are concatenated and encoded
in one call, eliminating stochastic orbit mismatch during both training and
evaluation.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.metrics import balanced_accuracy_score
from torch import Tensor, nn

from .augment import left_right_swap_index
from .cameo_net import (
    FrozenTangentAnchor,
    _mirror_covariances,
    _mirror_raw,
)
from .config import CHANNELS


def _group_count(channels: int, maximum: int = 4) -> int:
    """Choose the largest small GroupNorm divisor."""

    return max(1, math.gcd(int(channels), int(maximum)))


@dataclass
class ParityFuseOutput:
    """Outputs and auditable internal parity representations."""

    logit: Tensor
    anchor_logit: Tensor
    residual_logit: Tensor
    fusion_weights: Tensor
    raw_even: Tensor
    raw_odd: Tensor
    tangent_even: Tensor
    tangent_odd: Tensor
    fused_odd: Tensor


class HeadlessRawEncoder(nn.Module):
    """Deterministic compact temporal/spatial encoder with no scalar head.

    GroupNorm is sample-local, unlike BatchNorm, and this module contains no
    stochastic layer.  Calling it on the concatenated two-view batch therefore
    gives an orbit-consistent feature map in training mode as well as eval mode.
    """

    def __init__(
        self,
        *,
        n_channels: int,
        temporal_filters: int = 16,
        temporal_kernel: int = 31,
        dynamics_channels: int = 24,
        dynamics_kernel: int = 15,
        output_dim: int = 40,
    ) -> None:
        super().__init__()
        if n_channels <= 0 or temporal_filters <= 0 or dynamics_channels <= 0:
            raise ValueError("channel dimensions must be positive")
        if temporal_kernel <= 0 or temporal_kernel % 2 == 0:
            raise ValueError("temporal_kernel must be a positive odd integer")
        if dynamics_kernel <= 0 or dynamics_kernel % 2 == 0:
            raise ValueError("dynamics_kernel must be a positive odd integer")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")

        self.n_channels = int(n_channels)
        self.output_dim = int(output_dim)
        self.temporal = nn.Conv2d(
            1,
            temporal_filters,
            kernel_size=(1, temporal_kernel),
            padding=(0, temporal_kernel // 2),
            bias=False,
        )
        self.spatial = nn.Conv2d(
            temporal_filters,
            temporal_filters,
            kernel_size=(n_channels, 1),
            groups=temporal_filters,
            bias=False,
        )
        self.spatial_norm = nn.GroupNorm(
            _group_count(temporal_filters), temporal_filters
        )
        self.dynamics = nn.Sequential(
            nn.Conv1d(
                temporal_filters,
                temporal_filters,
                kernel_size=dynamics_kernel,
                padding=dynamics_kernel // 2,
                groups=temporal_filters,
                bias=False,
            ),
            nn.Conv1d(temporal_filters, dynamics_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(dynamics_channels), dynamics_channels),
            nn.GELU(),
        )
        # Three stable summaries retain signed level, power, and local dynamics.
        summary_dim = 3 * dynamics_channels
        self.projection = nn.Sequential(
            nn.Linear(summary_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, epochs: Tensor) -> Tensor:
        if epochs.ndim != 3:
            raise ValueError("epochs must have shape (batch, channels, time)")
        if epochs.shape[1] != self.n_channels:
            raise ValueError(
                f"expected {self.n_channels} channels, got {epochs.shape[1]}"
            )
        latent = self.temporal(epochs.unsqueeze(1))
        latent = self.spatial(latent).squeeze(2)
        latent = nn.functional.gelu(self.spatial_norm(latent))
        latent = self.dynamics(latent)

        mean = latent.mean(dim=-1)
        log_power = torch.log(torch.square(latent).mean(dim=-1).clamp_min(1e-6))
        if latent.shape[-1] > 1:
            difference = latent[..., 1:] - latent[..., :-1]
            log_difference_power = torch.log(
                torch.square(difference).mean(dim=-1).clamp_min(1e-6)
            )
        else:  # pragma: no cover - real EEG windows contain many samples
            log_difference_power = torch.zeros_like(mean)
        return self.projection(
            torch.cat((mean, log_power, log_difference_power), dim=1)
        )


class SharedBandTangentEncoder(nn.Module):
    """Apply one learned tangent map to every covariance band, then mix bands."""

    def __init__(
        self,
        *,
        n_bands: int,
        n_tangent_features: int,
        hidden_dim: int = 24,
        band_dim: int = 12,
        output_dim: int = 32,
    ) -> None:
        super().__init__()
        if n_bands <= 0 or n_tangent_features <= 0:
            raise ValueError("tangent dimensions must be positive")
        if n_tangent_features % n_bands:
            raise ValueError("tangent feature count must be divisible by n_bands")
        if min(hidden_dim, band_dim, output_dim) <= 0:
            raise ValueError("encoder dimensions must be positive")
        self.n_bands = int(n_bands)
        self.n_tangent_features = int(n_tangent_features)
        self.features_per_band = n_tangent_features // n_bands
        self.output_dim = int(output_dim)

        # These layers are shared over the explicit band axis.
        self.shared_band_map = nn.Sequential(
            nn.Linear(self.features_per_band, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, band_dim),
            nn.GELU(),
        )
        self.band_mixer = nn.Sequential(
            nn.Linear(n_bands * band_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, tangent: Tensor) -> Tensor:
        if tangent.ndim != 2 or tangent.shape[1] != self.n_tangent_features:
            raise ValueError(
                "tangent must have shape "
                f"(batch, {self.n_tangent_features})"
            )
        bands = tangent.reshape(
            tangent.shape[0], self.n_bands, self.features_per_band
        )
        encoded_bands = self.shared_band_map(bands)
        return self.band_mixer(encoded_bands.flatten(start_dim=1))


class InvariantLatentFusion(nn.Module):
    """Fuse two odd latents with a reflection-invariant coordinate-wise gate."""

    def __init__(
        self,
        *,
        raw_dim: int,
        tangent_dim: int,
        fusion_dim: int = 32,
        gate_hidden: int = 24,
    ) -> None:
        super().__init__()
        if min(raw_dim, tangent_dim, fusion_dim, gate_hidden) <= 0:
            raise ValueError("fusion dimensions must be positive")
        self.fusion_dim = int(fusion_dim)

        # Bias is permitted on even paths, but forbidden on odd paths.
        self.raw_even_projection = nn.Linear(raw_dim, fusion_dim)
        self.tangent_even_projection = nn.Linear(tangent_dim, fusion_dim)
        self.raw_odd_projection = nn.Linear(raw_dim, fusion_dim, bias=False)
        self.tangent_odd_projection = nn.Linear(
            tangent_dim, fusion_dim, bias=False
        )

        # Five blocks are invariant: two even contexts, two squared odd
        # magnitudes, and the raw/tangent odd product (two sign changes).
        self.gate = nn.Sequential(
            nn.Linear(5 * fusion_dim, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 2 * fusion_dim),
        )

    def forward(
        self,
        raw_even: Tensor,
        raw_odd: Tensor,
        tangent_even: Tensor,
        tangent_odd: Tensor,
    ) -> tuple[Tensor, Tensor]:
        raw_even_context = nn.functional.gelu(
            self.raw_even_projection(raw_even)
        )
        tangent_even_context = nn.functional.gelu(
            self.tangent_even_projection(tangent_even)
        )
        raw_odd_projected = torch.tanh(self.raw_odd_projection(raw_odd))
        tangent_odd_projected = torch.tanh(
            self.tangent_odd_projection(tangent_odd)
        )
        invariant_context = torch.cat(
            (
                raw_even_context,
                tangent_even_context,
                torch.square(raw_odd_projected),
                torch.square(tangent_odd_projected),
                raw_odd_projected * tangent_odd_projected,
            ),
            dim=1,
        )
        gate_logits = self.gate(invariant_context).reshape(
            -1, 2, self.fusion_dim
        )
        fusion_weights = torch.softmax(gate_logits, dim=1)
        fused_odd = (
            fusion_weights[:, 0] * raw_odd_projected
            + fusion_weights[:, 1] * tangent_odd_projected
        )
        return fused_odd, fusion_weights


class ParityFuseNet(nn.Module):
    """Exact signed-``Z2`` raw/tangent network with a frozen anchor skip."""

    def __init__(
        self,
        *,
        n_channels: int,
        n_times: int,
        n_bands: int,
        n_tangent_features: int,
        tangent_anchor_weight: NDArray[np.floating] | Tensor,
        temporal_filters: int = 16,
        temporal_kernel: int = 31,
        dynamics_channels: int = 24,
        dynamics_kernel: int = 15,
        raw_dim: int = 40,
        tangent_hidden: int = 24,
        tangent_band_dim: int = 12,
        tangent_dim: int = 32,
        fusion_dim: int = 32,
        gate_hidden: int = 24,
    ) -> None:
        super().__init__()
        if n_times <= 0:
            raise ValueError("n_times must be positive")
        self.n_times = int(n_times)
        self.raw_encoder = HeadlessRawEncoder(
            n_channels=n_channels,
            temporal_filters=temporal_filters,
            temporal_kernel=temporal_kernel,
            dynamics_channels=dynamics_channels,
            dynamics_kernel=dynamics_kernel,
            output_dim=raw_dim,
        )
        self.tangent_encoder = SharedBandTangentEncoder(
            n_bands=n_bands,
            n_tangent_features=n_tangent_features,
            hidden_dim=tangent_hidden,
            band_dim=tangent_band_dim,
            output_dim=tangent_dim,
        )
        self.fusion = InvariantLatentFusion(
            raw_dim=raw_dim,
            tangent_dim=tangent_dim,
            fusion_dim=fusion_dim,
            gate_hidden=gate_hidden,
        )

        anchor = torch.as_tensor(tangent_anchor_weight, dtype=torch.float32).reshape(
            1, -1
        )
        if anchor.shape[1] != n_tangent_features:
            raise ValueError("tangent anchor has the wrong feature dimension")
        # A buffer, not a Parameter: the convex source-only floor never trains.
        self.register_buffer("tangent_anchor_weight", anchor.clone())
        self.residual_readout = nn.Linear(fusion_dim, 1, bias=False)
        nn.init.zeros_(self.residual_readout.weight)

    @staticmethod
    def _parity(first: Tensor, reflected: Tensor) -> tuple[Tensor, Tensor]:
        return 0.5 * (first + reflected), 0.5 * (first - reflected)

    def forward(
        self,
        raw: Tensor,
        raw_reflected: Tensor,
        tangent: Tensor,
        tangent_reflected: Tensor,
    ) -> ParityFuseOutput:
        if raw.shape != raw_reflected.shape:
            raise ValueError("raw view shapes disagree")
        if tangent.shape != tangent_reflected.shape:
            raise ValueError("tangent view shapes disagree")
        if raw.ndim != 3 or raw.shape[2] != self.n_times:
            raise ValueError(
                f"raw views must have shape (batch, channels, {self.n_times})"
            )
        if raw.shape[0] != tangent.shape[0]:
            raise ValueError("raw and tangent batch sizes disagree")

        batch = raw.shape[0]
        # One encoder call per modality makes the shared orbit path explicit.
        raw_pair = self.raw_encoder(torch.cat((raw, raw_reflected), dim=0))
        tangent_pair = self.tangent_encoder(
            torch.cat((tangent, tangent_reflected), dim=0)
        )
        raw_even, raw_odd = self._parity(raw_pair[:batch], raw_pair[batch:])
        tangent_even, tangent_odd = self._parity(
            tangent_pair[:batch], tangent_pair[batch:]
        )
        fused_odd, fusion_weights = self.fusion(
            raw_even, raw_odd, tangent_even, tangent_odd
        )

        # The classifier intercept disappears under odd Reynolds projection:
        # 0.5 * [(w.t + b) - (w.t_M + b)] = w * tangent_odd.
        anchor_logit = nn.functional.linear(
            0.5 * (tangent - tangent_reflected), self.tangent_anchor_weight
        ).squeeze(1)
        residual_logit = self.residual_readout(fused_odd).squeeze(1)
        logit = anchor_logit + residual_logit
        return ParityFuseOutput(
            logit=logit,
            anchor_logit=anchor_logit,
            residual_logit=residual_logit,
            fusion_weights=fusion_weights,
            raw_even=raw_even,
            raw_odd=raw_odd,
            tangent_even=tangent_even,
            tangent_odd=tangent_odd,
            fused_odd=fused_odd,
        )


@dataclass(frozen=True)
class ParityFuseConfig:
    """Training and compact architecture configuration."""

    epochs: int = 240
    batch_size: int = 64
    learning_rate: float = 8e-4
    weight_decay: float = 6e-4
    patience: int = 40
    min_delta: float = 1e-4
    temporal_filters: int = 16
    temporal_kernel: int = 31
    dynamics_channels: int = 24
    dynamics_kernel: int = 15
    raw_dim: int = 40
    tangent_hidden: int = 24
    tangent_band_dim: int = 12
    tangent_dim: int = 32
    fusion_dim: int = 32
    gate_hidden: int = 24
    residual_penalty: float = 0.01
    gradient_clip: float = 5.0
    seed: int = 7
    device: str = "cuda"
    deterministic: bool = True

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0 or self.patience <= 0:
            raise ValueError("epochs, batch size, and patience must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("optimizer settings are invalid")
        if self.min_delta < 0.0 or self.residual_penalty < 0.0:
            raise ValueError("loss settings must be non-negative")
        if self.gradient_clip <= 0.0:
            raise ValueError("gradient_clip must be positive")
        dimensions = (
            self.temporal_filters,
            self.dynamics_channels,
            self.raw_dim,
            self.tangent_hidden,
            self.tangent_band_dim,
            self.tangent_dim,
            self.fusion_dim,
            self.gate_hidden,
        )
        if min(dimensions) <= 0:
            raise ValueError("architecture dimensions must be positive")
        if self.temporal_kernel <= 0 or self.temporal_kernel % 2 == 0:
            raise ValueError("temporal_kernel must be a positive odd integer")
        if self.dynamics_kernel <= 0 or self.dynamics_kernel % 2 == 0:
            raise ValueError("dynamics_kernel must be a positive odd integer")


def _resolve_device(value: str | None) -> torch.device:
    if value in ("", "auto", None):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _binary_loss(logit: Tensor, labels: Tensor) -> Tensor:
    return nn.functional.binary_cross_entropy_with_logits(
        logit, labels.to(logit.dtype)
    )


class ParityFuseClassifier:
    """Sklearn-like estimator with fit-only preprocessing and early stopping."""

    _SERIAL_VERSION = 1

    def __init__(self, config: ParityFuseConfig = ParityFuseConfig()) -> None:
        self.config = config

    def _fit_raw_scaler(
        self, raw: NDArray[np.floating], mirror_index: NDArray[np.int64]
    ) -> None:
        values = np.asarray(raw, dtype=np.float32)
        balanced = np.concatenate((values, _mirror_raw(values, mirror_index)), axis=0)
        self.raw_mean_ = balanced.mean(axis=(0, 2), keepdims=True)
        self.raw_std_ = balanced.std(axis=(0, 2), keepdims=True) + 1e-6

    def _prepare_raw(self, raw: NDArray[np.floating]) -> np.ndarray:
        values = np.asarray(raw, dtype=np.float32)
        if values.ndim != 3:
            raise ValueError("raw epochs must have shape (N, channels, time)")
        return ((values - self.raw_mean_) / self.raw_std_).astype(
            np.float32, copy=False
        )

    def _prepare_views(
        self,
        raw: NDArray[np.floating],
        covariances: NDArray[np.floating],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raw_values = np.asarray(raw)
        cov_values = np.asarray(covariances)
        if len(raw_values) != len(cov_values):
            raise ValueError("raw and covariance arrays have inconsistent lengths")
        return (
            self._prepare_raw(raw_values),
            self._prepare_raw(_mirror_raw(raw_values, self.mirror_index_)),
            self.anchor_.transform(cov_values),
            self.anchor_.transform(
                _mirror_covariances(cov_values, self.mirror_index_)
            ),
        )

    def _model_kwargs(
        self,
        *,
        n_channels: int,
        n_times: int,
        n_bands: int,
        n_tangent_features: int,
        tangent_anchor_weight: NDArray[np.floating] | Tensor,
    ) -> dict[str, Any]:
        config = self.config
        return {
            "n_channels": int(n_channels),
            "n_times": int(n_times),
            "n_bands": int(n_bands),
            "n_tangent_features": int(n_tangent_features),
            "tangent_anchor_weight": tangent_anchor_weight,
            "temporal_filters": config.temporal_filters,
            "temporal_kernel": config.temporal_kernel,
            "dynamics_channels": config.dynamics_channels,
            "dynamics_kernel": config.dynamics_kernel,
            "raw_dim": config.raw_dim,
            "tangent_hidden": config.tangent_hidden,
            "tangent_band_dim": config.tangent_band_dim,
            "tangent_dim": config.tangent_dim,
            "fusion_dim": config.fusion_dim,
            "gate_hidden": config.gate_hidden,
        }

    @staticmethod
    def _validate_fit_arrays(
        raw: NDArray[np.floating],
        covariances: NDArray[np.floating],
        labels: NDArray[np.integer],
        *,
        name: str,
    ) -> None:
        raw_array = np.asarray(raw)
        cov_array = np.asarray(covariances)
        label_array = np.asarray(labels)
        if raw_array.ndim != 3:
            raise ValueError(f"{name} raw array must have shape (N, C, T)")
        if cov_array.ndim != 4 or cov_array.shape[-1] != cov_array.shape[-2]:
            raise ValueError(f"{name} covariances must have shape (N, bands, C, C)")
        if not (len(raw_array) == len(cov_array) == len(label_array)):
            raise ValueError(f"{name} arrays have inconsistent lengths")
        if raw_array.shape[1] != cov_array.shape[-1]:
            raise ValueError(f"{name} raw/covariance channel dimensions disagree")

    def fit(
        self,
        raw_train: NDArray[np.floating],
        cov_train: NDArray[np.floating],
        y_train: NDArray[np.integer],
        raw_validation: NDArray[np.floating],
        cov_validation: NDArray[np.floating],
        y_validation: NDArray[np.integer],
        *,
        channels: Sequence[str] = CHANNELS,
    ) -> "ParityFuseClassifier":
        self._validate_fit_arrays(
            raw_train, cov_train, y_train, name="training"
        )
        self._validate_fit_arrays(
            raw_validation, cov_validation, y_validation, name="validation"
        )
        raw_train = np.asarray(raw_train)
        cov_train = np.asarray(cov_train)
        raw_validation = np.asarray(raw_validation)
        cov_validation = np.asarray(cov_validation)
        y_train = np.asarray(y_train, dtype=np.int64)
        y_validation = np.asarray(y_validation, dtype=np.int64)
        if set(np.unique(y_train).tolist()) != {0, 1}:
            raise ValueError("PARITY-Fuse requires both labels 0 and 1 in training")
        if set(np.unique(y_validation).tolist()) != {0, 1}:
            raise ValueError("PARITY-Fuse requires both labels 0 and 1 in validation")
        if raw_train.shape[1:] != raw_validation.shape[1:]:
            raise ValueError("training and validation raw shapes disagree")
        if cov_train.shape[1:] != cov_validation.shape[1:]:
            raise ValueError("training and validation covariance shapes disagree")
        if len(channels) != raw_train.shape[1]:
            raise ValueError("channel-name count does not match the arrays")

        config = self.config
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        torch.use_deterministic_algorithms(config.deterministic, warn_only=True)
        device = _resolve_device(config.device)
        self.device_ = device
        self.channels_ = tuple(channels)
        self.mirror_index_ = np.asarray(
            left_right_swap_index(self.channels_), dtype=np.int64
        )

        started = time.perf_counter()
        # Both estimators below see training rows (and deterministic reflections
        # of those rows) only.  Validation data affects checkpoint selection only.
        self.anchor_ = FrozenTangentAnchor().fit(
            cov_train, y_train, self.mirror_index_
        )
        self._fit_raw_scaler(raw_train, self.mirror_index_)
        train_views = self._prepare_views(raw_train, cov_train)
        validation_views = self._prepare_views(raw_validation, cov_validation)
        train_tensors = tuple(
            torch.from_numpy(value).to(device) for value in train_views
        )
        validation_tensors = tuple(
            torch.from_numpy(value).to(device) for value in validation_views
        )
        y_tr = torch.from_numpy(y_train).to(device)
        y_va = torch.from_numpy(y_validation).to(device)

        self.model_kwargs_ = self._model_kwargs(
            n_channels=raw_train.shape[1],
            n_times=raw_train.shape[2],
            n_bands=cov_train.shape[1],
            n_tangent_features=train_views[2].shape[1],
            tangent_anchor_weight=self.anchor_.model_.coef_[0],
        )
        self.model_ = ParityFuseNet(**self.model_kwargs_).to(device)
        self.initial_model_state_ = copy.deepcopy(self.model_.state_dict())
        self.param_count_ = sum(p.numel() for p in self.model_.parameters())
        self.trainable_param_count_ = sum(
            p.numel() for p in self.model_.parameters() if p.requires_grad
        )
        if self.trainable_param_count_ >= 30_000:
            raise RuntimeError(
                "PARITY-Fuse exceeded its 30k trainable-parameter contract: "
                f"{self.trainable_param_count_}"
            )
        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs
        )
        generator = torch.Generator(device=device).manual_seed(config.seed + 1)

        # The zero-residual initialization is itself a valid checkpoint.  This
        # lets validation retain the convex anchor if every neural update hurts.
        self.model_.eval()
        with torch.no_grad():
            initial_output = self.model_(*validation_tensors)
            best_loss = float(_binary_loss(initial_output.logit, y_va))
            initial_prediction = (torch.sigmoid(initial_output.logit) >= 0.5).long()
            best_balanced = float(
                balanced_accuracy_score(
                    y_validation, initial_prediction.cpu().numpy()
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
            }
        ]

        for epoch in range(config.epochs):
            self.model_.train()
            order = torch.randperm(len(y_tr), device=device, generator=generator)
            total_loss = 0.0
            for start in range(0, len(order), config.batch_size):
                rows = order[start : start + config.batch_size]
                output = self.model_(*(value[rows] for value in train_tensors))
                loss = _binary_loss(output.logit, y_tr[rows])
                if config.residual_penalty:
                    loss = loss + config.residual_penalty * torch.square(
                        output.residual_logit
                    ).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model_.parameters(), config.gradient_clip
                )
                optimizer.step()
                total_loss += float(loss.detach()) * len(rows)
            scheduler.step()

            self.model_.eval()
            with torch.no_grad():
                validation_output = self.model_(*validation_tensors)
                validation_loss = float(
                    _binary_loss(validation_output.logit, y_va)
                )
                validation_prediction = (
                    torch.sigmoid(validation_output.logit) >= 0.5
                ).long()
                validation_balanced = float(
                    balanced_accuracy_score(
                        y_validation, validation_prediction.cpu().numpy()
                    )
                )
            self.history_.append(
                {
                    "epoch": float(epoch),
                    "train_loss": total_loss / max(1, len(y_tr)),
                    "validation_loss": validation_loss,
                    "validation_balanced_accuracy": validation_balanced,
                }
            )
            if validation_loss < best_loss - config.min_delta:
                best_loss = validation_loss
                best_balanced = validation_balanced
                best_epoch = epoch
                stale = 0
                best_state = copy.deepcopy(self.model_.state_dict())
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
        self.config_ = asdict(config)
        return self

    def fit_fixed_epochs(
        self,
        raw_source: NDArray[np.floating],
        cov_source: NDArray[np.floating],
        y_source: NDArray[np.integer],
        *,
        epochs: int,
        channels: Sequence[str] = CHANNELS,
    ) -> "ParityFuseClassifier":
        """Refit the source anchor and neural residual without validation.

        ``epochs == 0`` is intentional: validation may select the architecture's
        exact convex-anchor checkpoint before any neural optimizer step.
        """

        config = self.config
        epochs = int(epochs)
        if not 0 <= epochs <= config.epochs:
            raise ValueError("fixed refit epochs must be in [0, config.epochs]")
        self._validate_fit_arrays(
            raw_source, cov_source, y_source, name="source"
        )
        raw_source = np.asarray(raw_source)
        cov_source = np.asarray(cov_source)
        y_source = np.asarray(y_source, dtype=np.int64)
        if set(np.unique(y_source).tolist()) != {0, 1}:
            raise ValueError("PARITY-Fuse requires both labels 0 and 1 in source")
        if len(channels) != raw_source.shape[1]:
            raise ValueError("channel-name count does not match source arrays")

        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        torch.use_deterministic_algorithms(config.deterministic, warn_only=True)
        device = _resolve_device(config.device)
        self.device_ = device
        self.channels_ = tuple(channels)
        self.mirror_index_ = np.asarray(
            left_right_swap_index(self.channels_), dtype=np.int64
        )

        started = time.perf_counter()
        self.anchor_ = FrozenTangentAnchor().fit(
            cov_source, y_source, self.mirror_index_
        )
        self._fit_raw_scaler(raw_source, self.mirror_index_)
        source_views = self._prepare_views(raw_source, cov_source)
        source_tensors = tuple(
            torch.from_numpy(value).to(device) for value in source_views
        )
        labels = torch.from_numpy(y_source).to(device)
        self.model_kwargs_ = self._model_kwargs(
            n_channels=raw_source.shape[1],
            n_times=raw_source.shape[2],
            n_bands=cov_source.shape[1],
            n_tangent_features=source_views[2].shape[1],
            tangent_anchor_weight=self.anchor_.model_.coef_[0],
        )
        self.model_ = ParityFuseNet(**self.model_kwargs_).to(device)
        self.initial_model_state_ = copy.deepcopy(self.model_.state_dict())
        self.param_count_ = sum(p.numel() for p in self.model_.parameters())
        self.trainable_param_count_ = sum(
            p.numel() for p in self.model_.parameters() if p.requires_grad
        )
        if self.trainable_param_count_ >= 30_000:
            raise RuntimeError(
                "PARITY-Fuse exceeded its 30k trainable-parameter contract: "
                f"{self.trainable_param_count_}"
            )
        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs
        )
        generator = torch.Generator(device=device).manual_seed(config.seed + 1)
        self.history_: list[dict[str, float]] = []

        for epoch in range(epochs):
            self.model_.train()
            order = torch.randperm(len(labels), device=device, generator=generator)
            total_loss = 0.0
            for start in range(0, len(order), config.batch_size):
                rows = order[start : start + config.batch_size]
                output = self.model_(*(value[rows] for value in source_tensors))
                loss = _binary_loss(output.logit, labels[rows])
                if config.residual_penalty:
                    loss = loss + config.residual_penalty * torch.square(
                        output.residual_logit
                    ).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model_.parameters(), config.gradient_clip
                )
                optimizer.step()
                total_loss += float(loss.detach()) * len(rows)
            scheduler.step()
            self.history_.append(
                {
                    "epoch": float(epoch),
                    "train_loss": total_loss / max(1, len(labels)),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
            )

        self.model_.eval()
        self.fixed_epochs_ = epochs
        self.epochs_run_ = epochs
        self.train_seconds_ = time.perf_counter() - started
        self.config_ = asdict(config)
        return self

    def _predict_output(
        self,
        raw: NDArray[np.floating],
        covariances: NDArray[np.floating],
        *,
        swapped: bool = False,
    ) -> ParityFuseOutput:
        if not hasattr(self, "model_"):
            raise RuntimeError("fit must be called before prediction")
        views = self._prepare_views(raw, covariances)
        if swapped:
            views = (views[1], views[0], views[3], views[2])
        tensors = tuple(torch.from_numpy(value).to(self.device_) for value in views)
        self.model_.eval()
        with torch.no_grad():
            return self.model_(*tensors)

    def decision_function(
        self,
        raw: NDArray[np.floating],
        covariances: NDArray[np.floating],
    ) -> np.ndarray:
        return self._predict_output(raw, covariances).logit.cpu().numpy()

    def predict_proba(
        self,
        raw: NDArray[np.floating],
        covariances: NDArray[np.floating],
    ) -> np.ndarray:
        logit = self.decision_function(raw, covariances)
        positive = 1.0 / (1.0 + np.exp(-np.clip(logit, -40.0, 40.0)))
        return np.column_stack((1.0 - positive, positive))

    def predict(
        self,
        raw: NDArray[np.floating],
        covariances: NDArray[np.floating],
    ) -> np.ndarray:
        return (self.decision_function(raw, covariances) >= 0.0).astype(np.int64)

    def max_equivariance_error(
        self,
        raw: NDArray[np.floating],
        covariances: NDArray[np.floating],
    ) -> float:
        first = self._predict_output(raw, covariances).logit
        reflected = self._predict_output(raw, covariances, swapped=True).logit
        return float(torch.max(torch.abs(first + reflected)).cpu())

    def max_gate_invariance_error(
        self,
        raw: NDArray[np.floating],
        covariances: NDArray[np.floating],
    ) -> float:
        first = self._predict_output(raw, covariances).fusion_weights
        reflected = self._predict_output(
            raw, covariances, swapped=True
        ).fusion_weights
        return float(torch.max(torch.abs(first - reflected)).cpu())

    def save(self, path: str | Path) -> None:
        """Save a fitted estimator; only load files from trusted sources."""

        if not hasattr(self, "model_"):
            raise RuntimeError("fit must be called before save")
        state = {
            key: value.detach().cpu() for key, value in self.model_.state_dict().items()
        }
        payload = {
            "serial_version": self._SERIAL_VERSION,
            "config": asdict(self.config),
            "channels": self.channels_,
            "mirror_index": self.mirror_index_,
            "raw_mean": self.raw_mean_,
            "raw_std": self.raw_std_,
            "anchor": self.anchor_,
            "model_kwargs": self.model_kwargs_,
            "model_state": state,
            "history": self.history_,
            "best_epoch": self.best_epoch_,
            "epochs_run": self.epochs_run_,
            "best_validation_loss": self.best_validation_loss_,
            "best_validation_balanced_accuracy": (
                self.best_validation_balanced_accuracy_
            ),
            "train_seconds": self.train_seconds_,
            "param_count": self.param_count_,
            "trainable_param_count": self.trainable_param_count_,
        }
        torch.save(payload, Path(path))

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str | None = None,
    ) -> "ParityFuseClassifier":
        """Load a trusted PARITY-Fuse checkpoint."""

        try:
            payload = torch.load(
                Path(path), map_location="cpu", weights_only=False
            )
        except TypeError:  # pragma: no cover - support older PyTorch
            payload = torch.load(Path(path), map_location="cpu")
        if payload.get("serial_version") != cls._SERIAL_VERSION:
            raise ValueError("unsupported PARITY-Fuse checkpoint version")
        config_values = dict(payload["config"])
        if device is not None:
            config_values["device"] = device
        estimator = cls(ParityFuseConfig(**config_values))
        estimator.device_ = _resolve_device(estimator.config.device)
        estimator.channels_ = tuple(payload["channels"])
        estimator.mirror_index_ = np.asarray(
            payload["mirror_index"], dtype=np.int64
        )
        estimator.raw_mean_ = np.asarray(payload["raw_mean"])
        estimator.raw_std_ = np.asarray(payload["raw_std"])
        estimator.anchor_ = payload["anchor"]
        estimator.model_kwargs_ = dict(payload["model_kwargs"])
        estimator.model_ = ParityFuseNet(**estimator.model_kwargs_).to(
            estimator.device_
        )
        estimator.model_.load_state_dict(payload["model_state"])
        estimator.model_.eval()
        estimator.history_ = list(payload["history"])
        estimator.best_epoch_ = int(payload["best_epoch"])
        estimator.epochs_run_ = int(payload["epochs_run"])
        estimator.best_validation_loss_ = float(payload["best_validation_loss"])
        estimator.best_validation_balanced_accuracy_ = float(
            payload["best_validation_balanced_accuracy"]
        )
        estimator.train_seconds_ = float(payload["train_seconds"])
        estimator.param_count_ = int(payload["param_count"])
        estimator.trainable_param_count_ = int(payload["trainable_param_count"])
        estimator.config_ = asdict(estimator.config)
        return estimator


__all__ = [
    "HeadlessRawEncoder",
    "InvariantLatentFusion",
    "ParityFuseClassifier",
    "ParityFuseConfig",
    "ParityFuseNet",
    "ParityFuseOutput",
    "SharedBandTangentEncoder",
]
