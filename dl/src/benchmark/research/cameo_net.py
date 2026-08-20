"""CAMEO-Net: counterfactual anchor-modulated energy/orbit decoding.

The architecture combines three deliberately low-capacity experts:

* a frozen, convex log-Euclidean tangent classifier fit on source data only;
* a ShallowConvNet-inspired log-energy head; and
* a signed temporal-dynamics head sharing the same learned filter/spatial bank.

The raw heads see an explicit left/right-reflected counterfactual.  Their even
(``nuisance``) logit component is penalized during fitting, while the amount of
counterfactual removal and the expert mixture are selected on the protected
validation group.  The outer test group is never used for fitting or routing.

This module intentionally does not use GeoAdaptNet's gated BiMap residual: the
project's ablations show that a compressed second view of the same covariance is
redundant.  CAMEO instead adds a genuinely complementary temporal/log-power view
while retaining the strong convex geometric floor.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn

from ..shared.augment import left_right_swap_index
from .config import CHANNELS, SFREQ
from ..shared.spd import log_euclidean_recenter, matrix_log, upper_vectorize


def _mirror_raw(values: NDArray[np.floating], index: NDArray[np.int64]) -> np.ndarray:
    return np.asarray(values)[..., index, :]


def _mirror_covariances(
    values: NDArray[np.floating], index: NDArray[np.int64]
) -> np.ndarray:
    covariances = np.asarray(values)
    return covariances[..., index, :][..., :, index]


def _tangent_features(
    covariances: NDArray[np.floating], log_reference: Tensor
) -> NDArray[np.float64]:
    cov = torch.as_tensor(np.asarray(covariances), dtype=torch.float64)
    aligned = log_euclidean_recenter(cov, log_reference)
    return (
        upper_vectorize(matrix_log(aligned))
        .flatten(start_dim=1)
        .detach()
        .cpu()
        .numpy()
    )


class FrozenTangentAnchor:
    """Symmetry-balanced tangent preprocessing and convex logistic anchor.

    All statistics are fit on the source rows.  Mirrored source examples are
    included with flipped labels, making the reference and feature scaler
    consistent with the raw branch's fit-side reflection augmentation.
    """

    def __init__(self, *, regularization: float = 1.0, max_iter: int = 3000) -> None:
        self.regularization = float(regularization)
        self.max_iter = int(max_iter)

    def fit(
        self,
        covariances: NDArray[np.floating],
        labels: NDArray[np.integer],
        mirror_index: NDArray[np.int64],
    ) -> "FrozenTangentAnchor":
        cov = np.asarray(covariances)
        y = np.asarray(labels, dtype=np.int64)
        mirrored = _mirror_covariances(cov, mirror_index)
        balanced_cov = np.concatenate((cov, mirrored), axis=0)
        balanced_y = np.concatenate((y, 1 - y), axis=0)

        cov_tensor = torch.as_tensor(balanced_cov, dtype=torch.float64)
        self.log_reference_ = matrix_log(cov_tensor).mean(dim=0)
        features = _tangent_features(balanced_cov, self.log_reference_)
        self.scaler_ = StandardScaler().fit(features)
        scaled = self.scaler_.transform(features)
        self.model_ = LogisticRegression(
            C=self.regularization,
            max_iter=self.max_iter,
            solver="lbfgs",
        ).fit(scaled, balanced_y)
        self.n_features_in_ = int(scaled.shape[1])
        return self

    def transform(self, covariances: NDArray[np.floating]) -> NDArray[np.float32]:
        features = _tangent_features(covariances, self.log_reference_)
        return self.scaler_.transform(features).astype(np.float32, copy=False)

    def signed_logit_from_features(self, features: NDArray[np.floating]) -> np.ndarray:
        return self.model_.decision_function(np.asarray(features)).astype(np.float32)

    def signed_logit(self, covariances: NDArray[np.floating]) -> np.ndarray:
        return self.signed_logit_from_features(self.transform(covariances))


class CAMEORawExpert(nn.Module):
    """Shared temporal/spatial bank with energy and dynamics readouts."""

    def __init__(
        self,
        *,
        n_channels: int,
        n_times: int,
        temporal_filters: int = 32,
        temporal_kernel: int = 25,
        dynamics_channels: int = 16,
        dynamics_kernel: int = 15,
        pool_kernel: int = 75,
        pool_stride: int = 15,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        if temporal_kernel > n_times or pool_kernel > n_times - temporal_kernel + 1:
            raise ValueError("temporal and pooling kernels do not fit the epoch")
        self.n_channels = int(n_channels)
        self.n_times = int(n_times)
        self.temporal_filters = int(temporal_filters)

        self.temporal = nn.Conv2d(
            1,
            temporal_filters,
            kernel_size=(1, temporal_kernel),
            bias=False,
        )
        self.spatial = nn.Conv2d(
            temporal_filters,
            temporal_filters,
            kernel_size=(n_channels, 1),
            bias=False,
        )
        self.normalization = nn.BatchNorm2d(temporal_filters)

        latent_times = n_times - temporal_kernel + 1
        pooled_times = (latent_times - pool_kernel) // pool_stride + 1
        self.energy_pool = nn.AvgPool1d(pool_kernel, stride=pool_stride)
        self.energy_dropout = nn.Dropout(dropout)
        self.energy_head = nn.Linear(temporal_filters * pooled_times, 1)

        pad = dynamics_kernel // 2
        self.dynamics = nn.Sequential(
            nn.Conv1d(
                temporal_filters,
                temporal_filters,
                kernel_size=dynamics_kernel,
                padding=pad,
                groups=temporal_filters,
                bias=False,
            ),
            nn.Conv1d(temporal_filters, dynamics_channels, kernel_size=1, bias=False),
            nn.GroupNorm(max(1, math.gcd(4, dynamics_channels)), dynamics_channels),
            nn.GELU(),
        )
        self.dynamics_dropout = nn.Dropout(dropout)
        self.dynamics_head = nn.Linear(2 * dynamics_channels, 1)

    def encode(self, epochs: Tensor) -> tuple[Tensor, Tensor]:
        """Return deterministic energy/dynamics features before dropout.

        Keeping feature extraction separate lets parity-aware descendants form
        reflected even/odd representations *before* applying any stochastic
        mask, which is required for exact evaluation-time equivariance.
        """

        if epochs.ndim != 3:
            raise ValueError("epochs must have shape (batch, channels, time)")
        latent = self.normalization(self.spatial(self.temporal(epochs.unsqueeze(1))))
        latent = latent.squeeze(2)

        energy = torch.square(latent)
        energy = torch.log(torch.clamp(self.energy_pool(energy), min=1e-6))
        energy_features = energy.flatten(start_dim=1)

        dynamics = self.dynamics(latent)
        dynamics_features = torch.cat(
            (
                dynamics.mean(dim=-1),
                dynamics.std(dim=-1, unbiased=False),
            ),
            dim=1,
        )
        return energy_features, dynamics_features

    def forward(self, epochs: Tensor) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        energy_features, dynamics_features = self.encode(epochs)
        energy_features = self.energy_dropout(energy_features)
        energy_logit = self.energy_head(energy_features).squeeze(1)

        dynamics_features = self.dynamics_dropout(dynamics_features)
        dynamics_logit = self.dynamics_head(dynamics_features).squeeze(1)
        return energy_logit, dynamics_logit, {
            "energy_features": energy_features,
            "dynamics_features": dynamics_features,
        }


@dataclass(frozen=True)
class CAMEOConfig:
    epochs: int = 220
    batch_size: int = 64
    learning_rate: float = 7e-4
    weight_decay: float = 5e-4
    patience: int = 35
    min_delta: float = 1e-4
    temporal_filters: int = 32
    temporal_kernel: int = 25
    dynamics_channels: int = 16
    dynamics_kernel: int = 15
    pool_kernel: int = 75
    pool_stride: int = 15
    dropout: float = 0.4
    swap_probability: float = 0.5
    mirror_penalty: float = 0.08
    auxiliary_weight: float = 0.25
    gradient_clip: float = 5.0
    seed: int = 7
    device: str = "cuda"
    # Validation-only counterfactual subtraction.  rho=0 is the ordinary
    # prediction and rho=1 is exact orbit projection.
    rho_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    mixture_names: tuple[str, ...] = (
        "geo",
        "energy",
        "dynamics",
        "geo+energy",
        "geo+dynamics",
        "energy+dynamics",
        "geo+energy+dynamics",
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "rho_grid", tuple(float(x) for x in self.rho_grid))
        object.__setattr__(self, "mixture_names", tuple(self.mixture_names))
        if self.epochs <= 0 or self.batch_size <= 0 or self.patience <= 0:
            raise ValueError("epochs, batch size, and patience must be positive")
        if not 0.0 <= self.swap_probability <= 1.0:
            raise ValueError("swap_probability must lie in [0, 1]")
        if self.mirror_penalty < 0.0 or self.auxiliary_weight < 0.0:
            raise ValueError("loss weights must be non-negative")
        if any(not 0.0 <= rho <= 1.0 for rho in self.rho_grid):
            raise ValueError("rho_grid values must lie in [0, 1]")


def _resolve_device(value: str) -> torch.device:
    if value in ("", "auto", None):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _binary_loss(logit: Tensor, labels: Tensor) -> Tensor:
    return nn.functional.binary_cross_entropy_with_logits(logit, labels.to(logit.dtype))


def _probability(logit: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(logit, dtype=np.float64), -40.0, 40.0)
    positive = 1.0 / (1.0 + np.exp(-clipped))
    return np.column_stack((1.0 - positive, positive))


def _component_names(name: str) -> tuple[str, ...]:
    return tuple(part for part in name.split("+") if part)


class CAMEOClassifier:
    """Leakage-controlled estimator wrapper for CAMEO-Net."""

    def __init__(self, config: CAMEOConfig = CAMEOConfig()) -> None:
        self.config = config

    def _fit_raw_scaler(
        self, epochs: NDArray[np.floating], mirror_index: NDArray[np.int64]
    ) -> None:
        raw = np.asarray(epochs, dtype=np.float32)
        mirrored = _mirror_raw(raw, mirror_index)
        balanced = np.concatenate((raw, mirrored), axis=0)
        self.raw_mean_ = balanced.mean(axis=(0, 2), keepdims=True)
        self.raw_std_ = balanced.std(axis=(0, 2), keepdims=True) + 1e-6

    def _prepare_raw(self, epochs: NDArray[np.floating]) -> np.ndarray:
        raw = np.asarray(epochs, dtype=np.float32)
        return ((raw - self.raw_mean_) / self.raw_std_).astype(np.float32, copy=False)

    def _raw_logits_tensor(
        self, raw: Tensor, raw_mirror: Tensor
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        energy, dynamics, _ = self.model_(raw)
        mirror_energy, mirror_dynamics, _ = self.model_(raw_mirror)
        return (
            {"energy": energy, "dynamics": dynamics},
            {"energy": mirror_energy, "dynamics": mirror_dynamics},
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
    ) -> "CAMEOClassifier":
        config = self.config
        y_train = np.asarray(y_train, dtype=np.int64)
        y_validation = np.asarray(y_validation, dtype=np.int64)
        if set(np.unique(y_train).tolist()) != {0, 1}:
            raise ValueError("CAMEO requires both binary labels 0 and 1 in training")
        if len(raw_train) != len(cov_train) or len(raw_train) != len(y_train):
            raise ValueError("training raw/covariance/label arrays disagree")
        if len(raw_validation) != len(cov_validation) or len(raw_validation) != len(y_validation):
            raise ValueError("validation raw/covariance/label arrays disagree")

        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        device = _resolve_device(config.device)
        self.device_ = device
        self.mirror_index_ = np.asarray(left_right_swap_index(channels), dtype=np.int64)

        started = time.perf_counter()
        self.anchor_ = FrozenTangentAnchor().fit(cov_train, y_train, self.mirror_index_)
        self._fit_raw_scaler(raw_train, self.mirror_index_)

        raw_tr_np = self._prepare_raw(raw_train)
        raw_tr_m_np = self._prepare_raw(_mirror_raw(raw_train, self.mirror_index_))
        raw_va_np = self._prepare_raw(raw_validation)
        raw_va_m_np = self._prepare_raw(_mirror_raw(raw_validation, self.mirror_index_))

        raw_tr = torch.from_numpy(raw_tr_np).to(device)
        raw_tr_m = torch.from_numpy(raw_tr_m_np).to(device)
        y_tr = torch.from_numpy(y_train).to(device)
        raw_va = torch.from_numpy(raw_va_np).to(device)
        raw_va_m = torch.from_numpy(raw_va_m_np).to(device)
        y_va = torch.from_numpy(y_validation).to(device)

        n_channels, n_times = raw_tr.shape[1:]
        self.model_ = CAMEORawExpert(
            n_channels=int(n_channels),
            n_times=int(n_times),
            temporal_filters=config.temporal_filters,
            temporal_kernel=config.temporal_kernel,
            dynamics_channels=config.dynamics_channels,
            dynamics_kernel=config.dynamics_kernel,
            pool_kernel=config.pool_kernel,
            pool_stride=config.pool_stride,
            dropout=config.dropout,
        ).to(device)
        # Retain the exact seeded start for protocol audits.  ``deepcopy`` is
        # required because optimizer steps mutate the live tensors in place.
        self.initial_model_state_ = copy.deepcopy(self.model_.state_dict())
        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs
        )
        generator = torch.Generator(device=device).manual_seed(config.seed + 1)

        best_state: dict[str, Tensor] | None = None
        best_validation = float("inf")
        best_epoch = -1
        stale = 0
        self.history_: list[dict[str, float]] = []

        for epoch in range(config.epochs):
            self.model_.train()
            order = torch.randperm(len(raw_tr), device=device, generator=generator)
            train_loss = 0.0
            for start in range(0, len(order), config.batch_size):
                rows = order[start : start + config.batch_size]
                original, reflected = raw_tr[rows], raw_tr_m[rows]
                labels = y_tr[rows]
                choose_mirror = (
                    torch.rand(len(rows), device=device, generator=generator)
                    < config.swap_probability
                )
                current = torch.where(
                    choose_mirror[:, None, None], reflected, original
                )
                counterfactual = torch.where(
                    choose_mirror[:, None, None], original, reflected
                )
                current_labels = torch.where(choose_mirror, 1 - labels, labels)

                current_logits, counter_logits = self._raw_logits_tensor(
                    current, counterfactual
                )
                energy = current_logits["energy"]
                dynamics = current_logits["dynamics"]
                mean_logit = 0.5 * (energy + dynamics)
                task_loss = _binary_loss(mean_logit, current_labels)
                auxiliary = 0.5 * (
                    _binary_loss(energy, current_labels)
                    + _binary_loss(dynamics, current_labels)
                )
                # A perfect label-reflection model has z(x) + z(Mx) = 0.
                mirror_loss = 0.5 * (
                    torch.mean(torch.square(energy + counter_logits["energy"]))
                    + torch.mean(torch.square(dynamics + counter_logits["dynamics"]))
                )
                loss = (
                    task_loss
                    + config.auxiliary_weight * auxiliary
                    + config.mirror_penalty * mirror_loss
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model_.parameters(), config.gradient_clip)
                optimizer.step()
                train_loss += float(loss.detach()) * len(rows)
            scheduler.step()

            self.model_.eval()
            with torch.no_grad():
                validation_logits, _ = self._raw_logits_tensor(raw_va, raw_va_m)
                validation_loss = float(
                    _binary_loss(
                        0.5
                        * (
                            validation_logits["energy"]
                            + validation_logits["dynamics"]
                        ),
                        y_va,
                    )
                )
            self.history_.append(
                {
                    "epoch": float(epoch),
                    "train_loss": train_loss / max(1, len(raw_tr)),
                    "validation_loss": validation_loss,
                }
            )
            if validation_loss < best_validation - config.min_delta:
                best_validation = validation_loss
                best_epoch = epoch
                stale = 0
                best_state = copy.deepcopy(self.model_.state_dict())
            else:
                stale += 1
                if stale >= config.patience:
                    break

        if best_state is None:
            raise RuntimeError("CAMEO training produced no validation checkpoint")
        self.model_.load_state_dict(best_state)
        self.model_.eval()
        self.best_epoch_ = best_epoch
        self.epochs_run_ = len(self.history_)
        self.best_validation_loss_ = best_validation

        validation_components = self._component_logits(
            raw_validation, cov_validation
        )
        self.selection_trace_: list[dict[str, Any]] = []
        best_key: tuple[float, float, float, float] | None = None
        best_choice: tuple[str, float] | None = None
        for mixture_name in config.mixture_names:
            names = _component_names(mixture_name)
            for rho in config.rho_grid:
                probability = self._mixed_probability(
                    validation_components, names=names, rho=rho
                )
                balanced = float(
                    balanced_accuracy_score(y_validation, probability.argmax(axis=1))
                )
                cross_entropy = float(log_loss(y_validation, probability, labels=[0, 1]))
                self.selection_trace_.append(
                    {
                        "mixture": mixture_name,
                        "rho": float(rho),
                        "balanced_accuracy": balanced,
                        "log_loss": cross_entropy,
                    }
                )
                # Accuracy first, calibration second.  Exact ties prefer fewer
                # components and less counterfactual correction.
                key = (balanced, -cross_entropy, -float(len(names)), -rho)
                if best_key is None or key > best_key:
                    best_key = key
                    best_choice = (mixture_name, float(rho))
        if best_choice is None:
            raise RuntimeError("CAMEO validation routing produced no candidate")
        self.selected_mixture_, self.selected_rho_ = best_choice
        self.param_count_ = sum(p.numel() for p in self.model_.parameters()) + (
            self.anchor_.n_features_in_ + 1
        )
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
        selected_mixture: str,
        selected_rho: float,
        channels: Sequence[str] = CHANNELS,
    ) -> "CAMEOClassifier":
        """Refit from the seeded initialization on all source rows.

        The epoch count and routing choice must have been selected by a
        separate :meth:`fit` call that never received prediction-only rows.
        No validation array is accepted here by design.
        """

        config = self.config
        epochs = int(epochs)
        if not 1 <= epochs <= config.epochs:
            raise ValueError("fixed refit epochs must be in [1, config.epochs]")
        component_names = _component_names(selected_mixture)
        if (
            selected_mixture not in config.mixture_names
            or not component_names
            or not set(component_names).issubset({"geo", "energy", "dynamics"})
        ):
            raise ValueError("selected CAMEO mixture is not in the frozen grid")
        selected_rho = float(selected_rho)
        if selected_rho not in config.rho_grid:
            raise ValueError("selected CAMEO rho is not in the frozen grid")

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
            raise ValueError("CAMEO requires both binary labels 0 and 1 in source")

        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
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
        raw_np = self._prepare_raw(raw_source)
        raw_mirror_np = self._prepare_raw(
            _mirror_raw(raw_source, self.mirror_index_)
        )
        raw = torch.from_numpy(raw_np).to(device)
        raw_mirror = torch.from_numpy(raw_mirror_np).to(device)
        labels = torch.from_numpy(y_source).to(device)

        n_channels, n_times = raw.shape[1:]
        self.model_ = CAMEORawExpert(
            n_channels=int(n_channels),
            n_times=int(n_times),
            temporal_filters=config.temporal_filters,
            temporal_kernel=config.temporal_kernel,
            dynamics_channels=config.dynamics_channels,
            dynamics_kernel=config.dynamics_kernel,
            pool_kernel=config.pool_kernel,
            pool_stride=config.pool_stride,
            dropout=config.dropout,
        ).to(device)
        self.initial_model_state_ = copy.deepcopy(self.model_.state_dict())
        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        # Preserve the selection-stage learning-rate trajectory rather than
        # compressing the cosine schedule into the selected duration.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs
        )
        generator = torch.Generator(device=device).manual_seed(config.seed + 1)
        self.history_: list[dict[str, float]] = []

        for epoch in range(epochs):
            self.model_.train()
            order = torch.randperm(len(raw), device=device, generator=generator)
            train_loss = 0.0
            for start in range(0, len(order), config.batch_size):
                rows = order[start : start + config.batch_size]
                original, reflected = raw[rows], raw_mirror[rows]
                current_labels = labels[rows]
                choose_mirror = (
                    torch.rand(len(rows), device=device, generator=generator)
                    < config.swap_probability
                )
                current = torch.where(
                    choose_mirror[:, None, None], reflected, original
                )
                counterfactual = torch.where(
                    choose_mirror[:, None, None], original, reflected
                )
                current_labels = torch.where(
                    choose_mirror, 1 - current_labels, current_labels
                )
                current_logits, counter_logits = self._raw_logits_tensor(
                    current, counterfactual
                )
                energy = current_logits["energy"]
                dynamics = current_logits["dynamics"]
                mean_logit = 0.5 * (energy + dynamics)
                task_loss = _binary_loss(mean_logit, current_labels)
                auxiliary = 0.5 * (
                    _binary_loss(energy, current_labels)
                    + _binary_loss(dynamics, current_labels)
                )
                mirror_loss = 0.5 * (
                    torch.mean(
                        torch.square(energy + counter_logits["energy"])
                    )
                    + torch.mean(
                        torch.square(dynamics + counter_logits["dynamics"])
                    )
                )
                loss = (
                    task_loss
                    + config.auxiliary_weight * auxiliary
                    + config.mirror_penalty * mirror_loss
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model_.parameters(), config.gradient_clip
                )
                optimizer.step()
                train_loss += float(loss.detach()) * len(rows)
            scheduler.step()
            self.history_.append(
                {
                    "epoch": float(epoch),
                    "train_loss": train_loss / max(1, len(raw)),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
            )

        self.model_.eval()
        self.selected_mixture_ = selected_mixture
        self.selected_rho_ = selected_rho
        self.fixed_epochs_ = epochs
        self.epochs_run_ = epochs
        self.param_count_ = sum(p.numel() for p in self.model_.parameters()) + (
            self.anchor_.n_features_in_ + 1
        )
        self.train_seconds_ = time.perf_counter() - started
        self.config_ = asdict(config)
        return self

    def _raw_component_logits(
        self, raw: NDArray[np.floating]
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        prepared = torch.from_numpy(self._prepare_raw(raw)).to(self.device_)
        mirrored_np = _mirror_raw(np.asarray(raw), self.mirror_index_)
        prepared_mirror = torch.from_numpy(self._prepare_raw(mirrored_np)).to(self.device_)
        self.model_.eval()
        original: dict[str, list[np.ndarray]] = {"energy": [], "dynamics": []}
        reflected: dict[str, list[np.ndarray]] = {"energy": [], "dynamics": []}
        with torch.no_grad():
            for start in range(0, len(prepared), 256):
                first, second = self._raw_logits_tensor(
                    prepared[start : start + 256],
                    prepared_mirror[start : start + 256],
                )
                for name in original:
                    original[name].append(first[name].cpu().numpy())
                    reflected[name].append(second[name].cpu().numpy())
        return (
            {name: np.concatenate(values) for name, values in original.items()},
            {name: np.concatenate(values) for name, values in reflected.items()},
        )

    def _component_logits(
        self,
        raw: NDArray[np.floating],
        covariances: NDArray[np.floating],
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        original_raw, mirror_raw = self._raw_component_logits(raw)
        mirror_cov = _mirror_covariances(covariances, self.mirror_index_)
        result: dict[str, tuple[np.ndarray, np.ndarray]] = {
            "geo": (
                self.anchor_.signed_logit(covariances),
                self.anchor_.signed_logit(mirror_cov),
            )
        }
        for name in ("energy", "dynamics"):
            result[name] = (original_raw[name], mirror_raw[name])
        return result

    @staticmethod
    def _mixed_probability(
        components: dict[str, tuple[np.ndarray, np.ndarray]],
        *,
        names: Iterable[str],
        rho: float,
    ) -> np.ndarray:
        probabilities: list[np.ndarray] = []
        for name in names:
            original, reflected = components[name]
            # Partial Reynolds projection: remove rho times the reflection-even
            # nuisance component while retaining rho=0 as a safe fallback.
            corrected = np.asarray(original) - rho * 0.5 * (
                np.asarray(original) + np.asarray(reflected)
            )
            probabilities.append(_probability(corrected))
        if not probabilities:
            raise ValueError("at least one CAMEO component is required")
        mixture = np.mean(np.stack(probabilities, axis=0), axis=0)
        return mixture / mixture.sum(axis=1, keepdims=True)

    def predict_proba(
        self,
        raw: NDArray[np.floating],
        covariances: NDArray[np.floating],
    ) -> np.ndarray:
        components = self._component_logits(raw, covariances)
        return self._mixed_probability(
            components,
            names=_component_names(self.selected_mixture_),
            rho=self.selected_rho_,
        )

    def predict(
        self,
        raw: NDArray[np.floating],
        covariances: NDArray[np.floating],
    ) -> np.ndarray:
        return self.predict_proba(raw, covariances).argmax(axis=1)


__all__ = [
    "CAMEOClassifier",
    "CAMEOConfig",
    "CAMEORawExpert",
    "FrozenTangentAnchor",
]
