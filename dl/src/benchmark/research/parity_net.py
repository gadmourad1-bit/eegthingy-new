"""HemiParityNet: an internally Z2-equivariant raw/SPD EEG network.

For binary left/right motor imagery, sagittal channel reflection is the order-two
group ``Z2`` and acts on the signed class logit by multiplication with ``-1``.
HemiParityNet represents this action *inside* the network rather than only using
reflection as augmentation or averaging final predictions.

For every learned raw feature ``h`` and reflected feature ``h_M`` it constructs

``even = (h + h_M) / 2`` and ``odd = (h - h_M) / 2``.

The readout is an odd function whose coefficients are conditioned only on even
quantities (the even feature and squared odd feature).  A second parity readout
operates on train-standardized log-Euclidean tangent features.  Finally, a
sample-wise fusion gate sees only reflection-invariant reliability statistics.
Consequently, in evaluation mode, swapping the paired inputs negates the final
signed logit to floating-point precision while leaving the gate unchanged.

This is distinct from post-hoc mirror averaging: the even/odd representation,
conditional odd readouts, and invariant cross-view gate are trainable internal
layers.  The source-only convex tangent estimator is used only to initialize the
geometric odd readout; that layer remains trainable.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.metrics import balanced_accuracy_score
from torch import Tensor, nn

from ..shared.augment import left_right_swap_index
from .cameo_net import (
    CAMEORawExpert,
    FrozenTangentAnchor,
    _mirror_covariances,
    _mirror_raw,
)
from .config import CHANNELS


@dataclass
class ParityOutput:
    logit: Tensor
    raw_logit: Tensor
    tangent_logit: Tensor
    fusion_weights: Tensor
    raw_even_norm: Tensor
    raw_odd_norm: Tensor
    tangent_even_norm: Tensor
    tangent_odd_norm: Tensor


class ConditionalOddReadout(nn.Module):
    """Odd readout modulated exclusively by reflection-invariant context."""

    def __init__(self, n_features: int, rank: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.n_features = int(n_features)
        self.rank = int(rank)
        self.odd_projection = nn.Linear(n_features, rank, bias=False)
        self.even_context = nn.Linear(n_features, rank, bias=True)
        self.magnitude_context = nn.Linear(n_features, rank, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(rank, 1, bias=False)

    def forward(self, even: Tensor, odd: Tensor) -> Tensor:
        odd_hidden = torch.tanh(self.odd_projection(odd))
        invariant_gate = torch.sigmoid(
            self.even_context(even) + self.magnitude_context(torch.square(odd))
        )
        return self.output(self.dropout(odd_hidden * invariant_gate)).squeeze(1)


class HemiParityNet(nn.Module):
    """Raw-energy/dynamics and tangent parity experts with invariant fusion."""

    def __init__(
        self,
        *,
        n_channels: int,
        n_times: int,
        n_tangent_features: int,
        tangent_initial_weight: NDArray[np.floating],
        temporal_filters: int = 32,
        dynamics_channels: int = 16,
        raw_rank: int = 12,
        tangent_rank: int = 8,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.raw_backbone = CAMEORawExpert(
            n_channels=n_channels,
            n_times=n_times,
            temporal_filters=temporal_filters,
            dynamics_channels=dynamics_channels,
            dropout=0.0,
        )
        # The ordinary CAMEO scalar heads are not part of this architecture.
        for parameter in self.raw_backbone.energy_head.parameters():
            parameter.requires_grad_(False)
        for parameter in self.raw_backbone.dynamics_head.parameters():
            parameter.requires_grad_(False)

        latent_times = n_times - self.raw_backbone.temporal.kernel_size[1] + 1
        pool_kernel = self.raw_backbone.energy_pool.kernel_size
        pool_stride = self.raw_backbone.energy_pool.stride
        if isinstance(pool_kernel, tuple):
            pool_kernel = pool_kernel[0]
        if isinstance(pool_stride, tuple):
            pool_stride = pool_stride[0]
        pooled_times = (latent_times - pool_kernel) // pool_stride + 1
        raw_features = temporal_filters * pooled_times + 2 * dynamics_channels
        self.raw_readout = ConditionalOddReadout(raw_features, raw_rank, dropout)

        self.tangent_base = nn.Linear(n_tangent_features, 1, bias=False)
        initial = np.asarray(tangent_initial_weight, dtype=np.float32).reshape(1, -1)
        if initial.shape != tuple(self.tangent_base.weight.shape):
            raise ValueError("tangent initialization has the wrong feature dimension")
        with torch.no_grad():
            self.tangent_base.weight.copy_(torch.from_numpy(initial))
        self.tangent_interaction = ConditionalOddReadout(
            n_tangent_features, tangent_rank, dropout
        )
        self.tangent_interaction_gate = nn.Parameter(torch.tensor(-3.0))

        # Every input statistic is invariant under swapping original/reflected
        # views: absolute odd logits and even/odd vector norms.
        self.fusion_gate = nn.Sequential(
            nn.Linear(6, 12),
            nn.GELU(),
            nn.Linear(12, 2),
        )

    @staticmethod
    def _parity(first: Tensor, reflected: Tensor) -> tuple[Tensor, Tensor]:
        return 0.5 * (first + reflected), 0.5 * (first - reflected)

    @staticmethod
    def _normalized_norm(value: Tensor) -> Tensor:
        return torch.linalg.vector_norm(value, dim=1) / math.sqrt(value.shape[1])

    def forward(
        self,
        raw: Tensor,
        raw_reflected: Tensor,
        tangent: Tensor,
        tangent_reflected: Tensor,
    ) -> ParityOutput:
        energy, dynamics = self.raw_backbone.encode(raw)
        reflected_energy, reflected_dynamics = self.raw_backbone.encode(raw_reflected)
        raw_features = torch.cat((energy, dynamics), dim=1)
        reflected_raw_features = torch.cat(
            (reflected_energy, reflected_dynamics), dim=1
        )
        raw_even, raw_odd = self._parity(raw_features, reflected_raw_features)
        tangent_even, tangent_odd = self._parity(tangent, tangent_reflected)

        raw_logit = self.raw_readout(raw_even, raw_odd)
        tangent_logit = self.tangent_base(tangent_odd).squeeze(1)
        tangent_logit = tangent_logit + torch.sigmoid(
            self.tangent_interaction_gate
        ) * self.tangent_interaction(tangent_even, tangent_odd)

        raw_even_norm = self._normalized_norm(raw_even)
        raw_odd_norm = self._normalized_norm(raw_odd)
        tangent_even_norm = self._normalized_norm(tangent_even)
        tangent_odd_norm = self._normalized_norm(tangent_odd)
        reliability = torch.stack(
            (
                torch.abs(raw_logit),
                torch.abs(tangent_logit),
                raw_even_norm,
                raw_odd_norm,
                tangent_even_norm,
                tangent_odd_norm,
            ),
            dim=1,
        )
        fusion_weights = torch.softmax(self.fusion_gate(reliability), dim=1)
        logit = (
            fusion_weights[:, 0] * raw_logit
            + fusion_weights[:, 1] * tangent_logit
        )
        return ParityOutput(
            logit=logit,
            raw_logit=raw_logit,
            tangent_logit=tangent_logit,
            fusion_weights=fusion_weights,
            raw_even_norm=raw_even_norm,
            raw_odd_norm=raw_odd_norm,
            tangent_even_norm=tangent_even_norm,
            tangent_odd_norm=tangent_odd_norm,
        )


@dataclass(frozen=True)
class ParityConfig:
    epochs: int = 240
    batch_size: int = 64
    learning_rate: float = 7e-4
    tangent_learning_rate_scale: float = 0.15
    weight_decay: float = 7e-4
    patience: int = 40
    min_delta: float = 1e-4
    temporal_filters: int = 32
    dynamics_channels: int = 16
    raw_rank: int = 12
    tangent_rank: int = 8
    dropout: float = 0.25
    auxiliary_weight: float = 0.2
    gate_balance_weight: float = 0.01
    gradient_clip: float = 5.0
    seed: int = 7
    device: str = "cuda"
    deterministic: bool = True

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0 or self.patience <= 0:
            raise ValueError("epochs, batch size, and patience must be positive")
        if self.learning_rate <= 0.0 or self.tangent_learning_rate_scale <= 0.0:
            raise ValueError("learning rates must be positive")
        if self.auxiliary_weight < 0.0 or self.gate_balance_weight < 0.0:
            raise ValueError("loss weights must be non-negative")


def _resolve_device(value: str) -> torch.device:
    if value in ("", "auto", None):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _binary_loss(logit: Tensor, labels: Tensor) -> Tensor:
    return nn.functional.binary_cross_entropy_with_logits(logit, labels.to(logit.dtype))


class HemiParityClassifier:
    """Train-only preprocessing and early-stopped HemiParityNet estimator."""

    def __init__(self, config: ParityConfig = ParityConfig()) -> None:
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
        return ((values - self.raw_mean_) / self.raw_std_).astype(np.float32, copy=False)

    def _prepare_views(
        self,
        raw: NDArray[np.floating],
        covariances: NDArray[np.floating],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raw_mirror = _mirror_raw(raw, self.mirror_index_)
        cov_mirror = _mirror_covariances(covariances, self.mirror_index_)
        return (
            self._prepare_raw(raw),
            self._prepare_raw(raw_mirror),
            self.anchor_.transform(covariances),
            self.anchor_.transform(cov_mirror),
        )

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
    ) -> "HemiParityClassifier":
        config = self.config
        y_train = np.asarray(y_train, dtype=np.int64)
        y_validation = np.asarray(y_validation, dtype=np.int64)
        if set(np.unique(y_train).tolist()) != {0, 1}:
            raise ValueError("HemiParityNet requires labels 0 and 1")
        if len(raw_train) != len(cov_train) or len(raw_train) != len(y_train):
            raise ValueError("training arrays have inconsistent lengths")

        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        torch.use_deterministic_algorithms(config.deterministic, warn_only=True)
        device = _resolve_device(config.device)
        self.device_ = device
        self.mirror_index_ = np.asarray(left_right_swap_index(channels), dtype=np.int64)

        started = time.perf_counter()
        self.anchor_ = FrozenTangentAnchor().fit(cov_train, y_train, self.mirror_index_)
        self._fit_raw_scaler(raw_train, self.mirror_index_)
        train_views = self._prepare_views(raw_train, cov_train)
        validation_views = self._prepare_views(raw_validation, cov_validation)
        train_tensors = [torch.from_numpy(value).to(device) for value in train_views]
        validation_tensors = [
            torch.from_numpy(value).to(device) for value in validation_views
        ]
        y_tr = torch.from_numpy(y_train).to(device)
        y_va = torch.from_numpy(y_validation).to(device)

        self.model_ = HemiParityNet(
            n_channels=int(train_tensors[0].shape[1]),
            n_times=int(train_tensors[0].shape[2]),
            n_tangent_features=int(train_tensors[2].shape[1]),
            tangent_initial_weight=self.anchor_.model_.coef_[0],
            temporal_filters=config.temporal_filters,
            dynamics_channels=config.dynamics_channels,
            raw_rank=config.raw_rank,
            tangent_rank=config.tangent_rank,
            dropout=config.dropout,
        ).to(device)
        self.initial_model_state_ = copy.deepcopy(self.model_.state_dict())
        tangent_parameters = list(self.model_.tangent_base.parameters())
        tangent_ids = {id(parameter) for parameter in tangent_parameters}
        primary_parameters = [
            parameter
            for parameter in self.model_.parameters()
            if parameter.requires_grad and id(parameter) not in tangent_ids
        ]
        optimizer = torch.optim.AdamW(
            [
                {"params": primary_parameters, "lr": config.learning_rate},
                {
                    "params": tangent_parameters,
                    "lr": config.learning_rate * config.tangent_learning_rate_scale,
                },
            ],
            weight_decay=config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs
        )
        generator = torch.Generator(device=device).manual_seed(config.seed + 1)

        best_state: dict[str, Tensor] | None = None
        best_loss = float("inf")
        best_balanced = float("nan")
        best_epoch = -1
        stale = 0
        self.history_: list[dict[str, float]] = []
        for epoch in range(config.epochs):
            self.model_.train()
            order = torch.randperm(len(y_tr), device=device, generator=generator)
            total_loss = 0.0
            for start in range(0, len(order), config.batch_size):
                rows = order[start : start + config.batch_size]
                output = self.model_(*(value[rows] for value in train_tensors))
                main_loss = _binary_loss(output.logit, y_tr[rows])
                auxiliary = 0.5 * (
                    _binary_loss(output.raw_logit, y_tr[rows])
                    + _binary_loss(output.tangent_logit, y_tr[rows])
                )
                mean_gate = output.fusion_weights.mean(dim=0)
                gate_balance = torch.square(mean_gate - 0.5).sum()
                loss = (
                    main_loss
                    + config.auxiliary_weight * auxiliary
                    + config.gate_balance_weight * gate_balance
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model_.parameters(), config.gradient_clip)
                optimizer.step()
                total_loss += float(loss.detach()) * len(rows)
            scheduler.step()

            self.model_.eval()
            with torch.no_grad():
                validation_output = self.model_(*validation_tensors)
                validation_loss = float(_binary_loss(validation_output.logit, y_va))
                validation_probability = torch.sigmoid(validation_output.logit)
                validation_prediction = (validation_probability >= 0.5).long()
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
        if best_state is None:
            raise RuntimeError("HemiParityNet produced no validation checkpoint")
        self.model_.load_state_dict(best_state)
        self.model_.eval()
        self.best_epoch_ = best_epoch
        self.epochs_run_ = len(self.history_)
        self.best_validation_loss_ = best_loss
        self.best_validation_balanced_accuracy_ = best_balanced
        self.train_seconds_ = time.perf_counter() - started
        self.param_count_ = sum(
            parameter.numel() for parameter in self.model_.parameters()
        )
        self.trainable_param_count_ = sum(
            parameter.numel()
            for parameter in self.model_.parameters()
            if parameter.requires_grad
        )
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
    ) -> "HemiParityClassifier":
        """Refit on all authorized source rows for a selected duration.

        HemiParity's tangent-base warm start is deliberately recomputed from
        the complete source anchor.  The benchmark separately hashes the random
        initialization with that data-dependent tensor excluded.
        """

        config = self.config
        epochs = int(epochs)
        if not 1 <= epochs <= config.epochs:
            raise ValueError("fixed refit epochs must be in [1, config.epochs]")
        raw_source = np.asarray(raw_source)
        cov_source = np.asarray(cov_source)
        y_source = np.asarray(y_source, dtype=np.int64)
        if raw_source.ndim != 3:
            raise ValueError("source raw array must have shape (N, C, T)")
        if cov_source.ndim != 4 or cov_source.shape[-1] != cov_source.shape[-2]:
            raise ValueError("source covariances must have shape (N, bands, C, C)")
        if not (len(raw_source) == len(cov_source) == len(y_source)):
            raise ValueError("source raw/covariance/label arrays disagree")
        if raw_source.shape[1] != cov_source.shape[-1]:
            raise ValueError("source raw/covariance channel dimensions disagree")
        if len(channels) != raw_source.shape[1]:
            raise ValueError("channel-name count does not match source arrays")
        if set(np.unique(y_source).tolist()) != {0, 1}:
            raise ValueError("HemiParityNet requires labels 0 and 1 in source")

        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        torch.use_deterministic_algorithms(config.deterministic, warn_only=True)
        device = _resolve_device(config.device)
        self.device_ = device
        self.mirror_index_ = np.asarray(
            left_right_swap_index(channels), dtype=np.int64
        )

        started = time.perf_counter()
        self.anchor_ = FrozenTangentAnchor().fit(
            cov_source, y_source, self.mirror_index_
        )
        self._fit_raw_scaler(raw_source, self.mirror_index_)
        source_views = self._prepare_views(raw_source, cov_source)
        source_tensors = [
            torch.from_numpy(value).to(device) for value in source_views
        ]
        labels = torch.from_numpy(y_source).to(device)

        self.model_ = HemiParityNet(
            n_channels=int(source_tensors[0].shape[1]),
            n_times=int(source_tensors[0].shape[2]),
            n_tangent_features=int(source_tensors[2].shape[1]),
            tangent_initial_weight=self.anchor_.model_.coef_[0],
            temporal_filters=config.temporal_filters,
            dynamics_channels=config.dynamics_channels,
            raw_rank=config.raw_rank,
            tangent_rank=config.tangent_rank,
            dropout=config.dropout,
        ).to(device)
        self.initial_model_state_ = copy.deepcopy(self.model_.state_dict())
        tangent_parameters = list(self.model_.tangent_base.parameters())
        tangent_ids = {id(parameter) for parameter in tangent_parameters}
        primary_parameters = [
            parameter
            for parameter in self.model_.parameters()
            if parameter.requires_grad and id(parameter) not in tangent_ids
        ]
        optimizer = torch.optim.AdamW(
            [
                {"params": primary_parameters, "lr": config.learning_rate},
                {
                    "params": tangent_parameters,
                    "lr": config.learning_rate
                    * config.tangent_learning_rate_scale,
                },
            ],
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
                main_loss = _binary_loss(output.logit, labels[rows])
                auxiliary = 0.5 * (
                    _binary_loss(output.raw_logit, labels[rows])
                    + _binary_loss(output.tangent_logit, labels[rows])
                )
                mean_gate = output.fusion_weights.mean(dim=0)
                gate_balance = torch.square(mean_gate - 0.5).sum()
                loss = (
                    main_loss
                    + config.auxiliary_weight * auxiliary
                    + config.gate_balance_weight * gate_balance
                )
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
        self.param_count_ = sum(
            parameter.numel() for parameter in self.model_.parameters()
        )
        self.trainable_param_count_ = sum(
            parameter.numel()
            for parameter in self.model_.parameters()
            if parameter.requires_grad
        )
        self.config_ = asdict(config)
        return self

    def _predict_output(
        self,
        raw: NDArray[np.floating],
        covariances: NDArray[np.floating],
        *,
        swapped: bool = False,
    ) -> ParityOutput:
        views = self._prepare_views(raw, covariances)
        if swapped:
            views = (views[1], views[0], views[3], views[2])
        tensors = [torch.from_numpy(value).to(self.device_) for value in views]
        self.model_.eval()
        with torch.no_grad():
            return self.model_(*tensors)

    def predict_proba(
        self,
        raw: NDArray[np.floating],
        covariances: NDArray[np.floating],
    ) -> np.ndarray:
        logit = self._predict_output(raw, covariances).logit.cpu().numpy()
        positive = 1.0 / (1.0 + np.exp(-np.clip(logit, -40.0, 40.0)))
        return np.column_stack((1.0 - positive, positive))

    def max_equivariance_error(
        self,
        raw: NDArray[np.floating],
        covariances: NDArray[np.floating],
    ) -> float:
        first = self._predict_output(raw, covariances).logit
        reflected = self._predict_output(raw, covariances, swapped=True).logit
        return float(torch.max(torch.abs(first + reflected)).cpu())


__all__ = [
    "ConditionalOddReadout",
    "HemiParityClassifier",
    "HemiParityNet",
    "ParityConfig",
    "ParityOutput",
]
