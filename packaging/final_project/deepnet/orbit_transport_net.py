"""OrbitTransportNet: exact label-equivariant transport of arbitrary EEG experts.

For binary left/right motor imagery, sagittal reflection exchanges the class
labels.  An ordinary scalar expert need not respect that symmetry: in general
its two view logits ``l(x)`` and ``l(Mx)`` are unrelated.  Orbit transport turns
that arbitrary pair into an exactly odd logit while retaining more information
than fixed mirror subtraction.

Let ``e=(l(x)+l(Mx))/2`` and ``o=(l(x)-l(Mx))/2``.  A learned orientation score
``q(e,o)`` has an odd, bias-free path in ``o`` whose coefficients depend only on
the invariant context ``(e,o**2)``.  Thus reflection negates ``q`` and swaps
``alpha=sigmoid(q)`` with ``1-alpha``.  The transported logit

``z = alpha*l(x) - (1-alpha)*l(Mx)``

therefore satisfies ``z(Mx)=-z(x)`` even when the underlying expert does not.
Separate energy and dynamics experts use this transport.  A source-only
log-Euclidean logistic anchor is projected exactly onto its odd component, and
a per-sample invariant gate convexly fuses all three odd logits.  Every reported
candidate remains exactly label anti-equivariant.

The raw views are evaluated as one concatenated batch through shared weights.
GroupNorm replaces batch-dependent normalization and no stochastic layer is
used, so the symmetry holds in both training and evaluation modes.
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

from .augment import left_right_swap_index
from .cameo_net import (
    CAMEORawExpert,
    FrozenTangentAnchor,
    _mirror_covariances,
    _mirror_raw,
)
from .config import CHANNELS


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
    """Apply the run's determinism contract before CUDA is initialized.

    CUDA matrix multiplication needs one of NVIDIA's documented workspace
    configurations for PyTorch's strict deterministic mode.  An explicitly
    configured valid alternative is preserved; missing/invalid values use the
    larger documented workspace.  Unlike the earlier warning-only setting, an
    unsupported nondeterministic operation now fails the run.
    """

    if enabled:
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(bool(enabled), warn_only=False)


@dataclass
class TransportResult:
    """One arbitrary paired expert after learned equivariant transport."""

    logit: Tensor
    orientation_score: Tensor
    alpha: Tensor
    even: Tensor
    odd: Tensor


class OddOrientationTransport(nn.Module):
    """Transport a paired scalar expert with an exactly odd orientation score.

    The conditioner may contain biases because it sees only reflection-invariant
    quantities.  Both the odd projection and final readout are bias-free, so
    ``q(e, 0)=0`` and ``q(e, -o)=-q(e, o)`` by construction.
    """

    def __init__(self, *, rank: int = 8, hidden: int = 12) -> None:
        super().__init__()
        if rank <= 0 or hidden <= 0:
            raise ValueError("orientation rank and hidden width must be positive")
        self.odd_basis = nn.Linear(1, rank, bias=False)
        self.invariant_conditioner = nn.Sequential(
            nn.Linear(2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, rank),
        )
        self.readout = nn.Linear(rank, 1, bias=False)
        # Begin close to the stable Reynolds projection (alpha=1/2) without
        # blocking gradient flow into the odd basis or its conditioner.
        nn.init.normal_(self.readout.weight, mean=0.0, std=1e-3)

    def forward(self, first: Tensor, reflected: Tensor) -> TransportResult:
        if first.ndim != 1 or reflected.ndim != 1 or first.shape != reflected.shape:
            raise ValueError("paired expert logits must be matching vectors")
        even = 0.5 * (first + reflected)
        odd = 0.5 * (first - reflected)
        odd_hidden = torch.tanh(self.odd_basis(odd.unsqueeze(1)))
        invariant = torch.stack((even, torch.square(odd)), dim=1)
        coefficients = torch.sigmoid(self.invariant_conditioner(invariant))
        orientation_score = self.readout(odd_hidden * coefficients).squeeze(1)
        alpha = torch.sigmoid(orientation_score)
        # This is algebraically alpha*first-(1-alpha)*reflected because
        # 2*sigmoid(q)-1=tanh(q/2).  The odd form avoids asymmetric rounding in
        # separate multiply/subtract paths and gives floating-point parity too.
        logit = odd + torch.tanh(0.5 * orientation_score) * even
        return TransportResult(
            logit=logit,
            orientation_score=orientation_score,
            alpha=alpha,
            even=even,
            odd=odd,
        )


class SharedOrbitRawExpert(nn.Module):
    """CAMEO's compact temporal/spatial bank with view-independent normalization."""

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
        normalization: str = "group",
    ) -> None:
        super().__init__()
        self.backbone = CAMEORawExpert(
            n_channels=n_channels,
            n_times=n_times,
            temporal_filters=temporal_filters,
            temporal_kernel=temporal_kernel,
            dynamics_channels=dynamics_channels,
            dynamics_kernel=dynamics_kernel,
            pool_kernel=pool_kernel,
            pool_stride=pool_stride,
            dropout=0.0,
        )
        if normalization == "group":
            groups = max(1, math.gcd(4, temporal_filters))
            self.backbone.normalization = nn.GroupNorm(groups, temporal_filters)
        elif normalization == "batch":
            # Both orbit members pass through this layer together, making the
            # batch statistics and running-state update view-order symmetric.
            self.backbone.normalization = nn.BatchNorm2d(temporal_filters)
        else:
            raise ValueError("normalization must be 'group' or 'batch'")

    def forward(
        self, first: Tensor, reflected: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if first.ndim != 3 or reflected.shape != first.shape:
            raise ValueError("paired raw views must have shape (batch, channels, time)")
        batch = len(first)
        # A single shared pass makes view treatment explicit.  GroupNorm has no
        # cross-row state, so swapping the two halves simply swaps the outputs.
        energy_features, dynamics_features = self.backbone.encode(
            torch.cat((first, reflected), dim=0)
        )
        energy = self.backbone.energy_head(energy_features).squeeze(1)
        dynamics = self.backbone.dynamics_head(dynamics_features).squeeze(1)
        return (
            energy[:batch],
            energy[batch:],
            dynamics[:batch],
            dynamics[batch:],
        )


@dataclass
class OrbitTransportOutput:
    """All odd expert logits and invariant routing diagnostics."""

    logit: Tensor
    energy_logit: Tensor
    dynamics_logit: Tensor
    anchor_logit: Tensor
    fusion_weights: Tensor
    energy_alpha: Tensor
    dynamics_alpha: Tensor
    energy_orientation: Tensor
    dynamics_orientation: Tensor
    energy_view_logits: Tensor
    dynamics_view_logits: Tensor

    def candidate_logits(self) -> dict[str, Tensor]:
        raw_mean = 0.5 * (self.energy_logit + self.dynamics_logit)
        return {
            "fused": self.logit,
            "anchor": self.anchor_logit,
            "raw_mean": raw_mean,
            "uniform": (raw_mean * 2.0 + self.anchor_logit) / 3.0,
            "energy": self.energy_logit,
            "dynamics": self.dynamics_logit,
        }


class OrbitTransportNet(nn.Module):
    """Energy/dynamics orbit transport with projected geometry and invariant MoE."""

    CANDIDATE_NAMES = (
        "fused",
        "anchor",
        "raw_mean",
        "uniform",
        "energy",
        "dynamics",
    )

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
        normalization: str = "group",
        orientation_rank: int = 8,
        orientation_hidden: int = 12,
        gate_hidden: int = 16,
    ) -> None:
        super().__init__()
        if gate_hidden <= 0:
            raise ValueError("gate hidden width must be positive")
        self.raw_expert = SharedOrbitRawExpert(
            n_channels=n_channels,
            n_times=n_times,
            temporal_filters=temporal_filters,
            temporal_kernel=temporal_kernel,
            dynamics_channels=dynamics_channels,
            dynamics_kernel=dynamics_kernel,
            pool_kernel=pool_kernel,
            pool_stride=pool_stride,
            normalization=normalization,
        )
        self.energy_transport = OddOrientationTransport(
            rank=orientation_rank, hidden=orientation_hidden
        )
        self.dynamics_transport = OddOrientationTransport(
            rank=orientation_rank, hidden=orientation_hidden
        )

        # Every router input is invariant under swapping the paired views.
        self.fusion_gate = nn.Sequential(
            nn.LayerNorm(11),
            nn.Linear(11, gate_hidden),
            nn.SiLU(),
            nn.Linear(gate_hidden, 3),
        )
        final = self.fusion_gate[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        with torch.no_grad():
            # A geometric safety prior still leaves useful gradient for both raw
            # experts.  Validation may select any predeclared odd candidate.
            final.bias.copy_(torch.tensor((-0.5, -0.5, 1.0)))

    def forward(
        self,
        raw: Tensor,
        raw_reflected: Tensor,
        anchor: Tensor,
        anchor_reflected: Tensor,
    ) -> OrbitTransportOutput:
        if anchor.ndim != 1 or anchor_reflected.shape != anchor.shape:
            raise ValueError("paired anchor logits must be matching vectors")
        if len(raw) != len(anchor):
            raise ValueError("raw views and anchor logits must have equal batch size")
        energy, energy_reflected, dynamics, dynamics_reflected = self.raw_expert(
            raw, raw_reflected
        )
        energy_result = self.energy_transport(energy, energy_reflected)
        dynamics_result = self.dynamics_transport(dynamics, dynamics_reflected)

        anchor_even = 0.5 * (anchor + anchor_reflected)
        anchor_logit = 0.5 * (anchor - anchor_reflected)
        invariant_gate_input = torch.stack(
            (
                energy_result.even,
                torch.square(energy_result.odd),
                torch.abs(energy_result.logit),
                dynamics_result.even,
                torch.square(dynamics_result.odd),
                torch.abs(dynamics_result.logit),
                anchor_even,
                torch.square(anchor_logit),
                torch.abs(anchor_logit),
                torch.abs(energy_result.orientation_score),
                torch.abs(dynamics_result.orientation_score),
            ),
            dim=1,
        )
        fusion_weights = torch.softmax(self.fusion_gate(invariant_gate_input), dim=1)
        expert_logits = torch.stack(
            (energy_result.logit, dynamics_result.logit, anchor_logit), dim=1
        )
        logit = torch.sum(fusion_weights * expert_logits, dim=1)
        return OrbitTransportOutput(
            logit=logit,
            energy_logit=energy_result.logit,
            dynamics_logit=dynamics_result.logit,
            anchor_logit=anchor_logit,
            fusion_weights=fusion_weights,
            energy_alpha=energy_result.alpha,
            dynamics_alpha=dynamics_result.alpha,
            energy_orientation=energy_result.orientation_score,
            dynamics_orientation=dynamics_result.orientation_score,
            energy_view_logits=torch.stack((energy, energy_reflected), dim=1),
            dynamics_view_logits=torch.stack((dynamics, dynamics_reflected), dim=1),
        )


@dataclass(frozen=True)
class OrbitTransportConfig:
    epochs: int = 240
    batch_size: int = 64
    learning_rate: float = 7e-4
    weight_decay: float = 5e-4
    patience: int = 40
    min_delta: float = 1e-4
    temporal_filters: int = 32
    temporal_kernel: int = 25
    dynamics_channels: int = 16
    dynamics_kernel: int = 15
    pool_kernel: int = 75
    pool_stride: int = 15
    normalization: str = "group"
    orientation_rank: int = 8
    orientation_hidden: int = 12
    gate_hidden: int = 16
    transport_auxiliary_weight: float = 0.25
    view_auxiliary_weight: float = 0.15
    orientation_penalty: float = 1e-3
    gradient_clip: float = 5.0
    seed: int = 7
    device: str = "cuda"
    deterministic: bool = True
    candidate_names: tuple[str, ...] = OrbitTransportNet.CANDIDATE_NAMES

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_names", tuple(self.candidate_names))
        if self.epochs <= 0 or self.batch_size <= 0 or self.patience <= 0:
            raise ValueError("epochs, batch size, and patience must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError(
                "learning rate must be positive and weight decay non-negative"
            )
        if self.gradient_clip <= 0.0:
            raise ValueError("gradient clip must be positive")
        loss_weights = (
            self.transport_auxiliary_weight,
            self.view_auxiliary_weight,
            self.orientation_penalty,
        )
        if any(weight < 0.0 for weight in loss_weights):
            raise ValueError("loss weights must be non-negative")
        unknown = set(self.candidate_names) - set(OrbitTransportNet.CANDIDATE_NAMES)
        if unknown or not self.candidate_names:
            raise ValueError(f"invalid orbit candidate names: {sorted(unknown)}")
        if len(set(self.candidate_names)) != len(self.candidate_names):
            raise ValueError("candidate names must be unique")
        if self.normalization not in {"group", "batch"}:
            raise ValueError("normalization must be 'group' or 'batch'")


class OrbitTransportClassifier:
    """Source-only estimator with validation-safe checkpoint/candidate selection."""

    def __init__(self, config: OrbitTransportConfig = OrbitTransportConfig()) -> None:
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
        return ((values - self.raw_mean_) / self.raw_std_).astype(
            np.float32, copy=False
        )

    def _prepare_views(
        self,
        raw: NDArray[np.floating],
        covariances: NDArray[np.floating],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raw_values = np.asarray(raw)
        covariance_values = np.asarray(covariances)
        if len(raw_values) != len(covariance_values):
            raise ValueError("raw and covariance arrays have inconsistent lengths")
        raw_reflected = _mirror_raw(raw_values, self.mirror_index_)
        covariance_reflected = _mirror_covariances(
            covariance_values, self.mirror_index_
        )
        return (
            self._prepare_raw(raw_values),
            self._prepare_raw(raw_reflected),
            self.anchor_.signed_logit(covariance_values).astype(np.float32, copy=False),
            self.anchor_.signed_logit(covariance_reflected).astype(
                np.float32, copy=False
            ),
        )

    @staticmethod
    def _validate_split(
        raw: NDArray[np.floating],
        covariances: NDArray[np.floating],
        labels: NDArray[np.integer],
        *,
        name: str,
    ) -> np.ndarray:
        y = np.asarray(labels, dtype=np.int64)
        if len(raw) != len(covariances) or len(raw) != len(y):
            raise ValueError(f"{name} arrays have inconsistent lengths")
        if set(np.unique(y).tolist()) != {0, 1}:
            raise ValueError(f"{name} split must contain labels 0 and 1")
        return y

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
    ) -> "OrbitTransportClassifier":
        config = self.config
        y_train = self._validate_split(raw_train, cov_train, y_train, name="training")
        y_validation = self._validate_split(
            raw_validation, cov_validation, y_validation, name="validation"
        )

        _configure_torch_determinism(config.deterministic)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        device = _resolve_device(config.device)
        self.device_ = device
        self.mirror_index_ = np.asarray(left_right_swap_index(channels), dtype=np.int64)
        if len(self.mirror_index_) != np.asarray(raw_train).shape[1]:
            raise ValueError("channel names do not match the raw channel dimension")

        started = time.perf_counter()
        # Both preprocessing estimators are fit before validation is transformed;
        # neither receives validation or outer-test rows.
        self.anchor_ = FrozenTangentAnchor().fit(cov_train, y_train, self.mirror_index_)
        self._fit_raw_scaler(raw_train, self.mirror_index_)
        train_views = self._prepare_views(raw_train, cov_train)
        validation_views = self._prepare_views(raw_validation, cov_validation)
        train_tensors = [torch.from_numpy(value).to(device) for value in train_views]
        validation_tensors = [
            torch.from_numpy(value).to(device) for value in validation_views
        ]
        y_tr = torch.from_numpy(y_train).to(device)

        self.model_ = OrbitTransportNet(
            n_channels=int(train_tensors[0].shape[1]),
            n_times=int(train_tensors[0].shape[2]),
            temporal_filters=config.temporal_filters,
            temporal_kernel=config.temporal_kernel,
            dynamics_channels=config.dynamics_channels,
            dynamics_kernel=config.dynamics_kernel,
            pool_kernel=config.pool_kernel,
            pool_stride=config.pool_stride,
            normalization=config.normalization,
            orientation_rank=config.orientation_rank,
            orientation_hidden=config.orientation_hidden,
            gate_hidden=config.gate_hidden,
        ).to(device)
        self.initial_model_state_ = copy.deepcopy(self.model_.state_dict())
        trainable_parameters = [
            parameter
            for parameter in self.model_.parameters()
            if parameter.requires_grad
        ]
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs
        )
        generator = torch.Generator(device=device).manual_seed(config.seed + 1)

        best_records: dict[str, dict[str, Any]] = {
            name: {"loss": float("inf"), "epoch": -1} for name in config.candidate_names
        }
        stale = 0
        self.history_: list[dict[str, Any]] = []
        for epoch in range(config.epochs):
            self.model_.train()
            order = torch.randperm(len(y_tr), device=device, generator=generator)
            total_loss = 0.0
            for start in range(0, len(order), config.batch_size):
                rows = order[start : start + config.batch_size]
                labels = y_tr[rows]
                output = self.model_(*(value[rows] for value in train_tensors))
                main_loss = _binary_loss(output.logit, labels)
                transport_auxiliary = 0.5 * (
                    _binary_loss(output.energy_logit, labels)
                    + _binary_loss(output.dynamics_logit, labels)
                )
                reflected_labels = 1 - labels
                view_auxiliary = 0.25 * (
                    _binary_loss(output.energy_view_logits[:, 0], labels)
                    + _binary_loss(output.energy_view_logits[:, 1], reflected_labels)
                    + _binary_loss(output.dynamics_view_logits[:, 0], labels)
                    + _binary_loss(output.dynamics_view_logits[:, 1], reflected_labels)
                )
                orientation_cost = 0.5 * (
                    torch.mean(torch.square(output.energy_orientation))
                    + torch.mean(torch.square(output.dynamics_orientation))
                )
                loss = (
                    main_loss
                    + config.transport_auxiliary_weight * transport_auxiliary
                    + config.view_auxiliary_weight * view_auxiliary
                    + config.orientation_penalty * orientation_cost
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(trainable_parameters, config.gradient_clip)
                optimizer.step()
                total_loss += float(loss.detach()) * len(rows)
            scheduler.step()

            self.model_.eval()
            with torch.no_grad():
                validation_output = self.model_(*validation_tensors)
            candidates = validation_output.candidate_logits()
            validation_metrics: dict[str, dict[str, float]] = {}
            improved = False
            for name in config.candidate_names:
                candidate = candidates[name]
                candidate_loss = float(
                    _binary_loss(
                        candidate,
                        torch.from_numpy(y_validation).to(device),
                    )
                )
                prediction = (candidate >= 0.0).long().cpu().numpy()
                balanced = float(balanced_accuracy_score(y_validation, prediction))
                validation_metrics[name] = {
                    "loss": candidate_loss,
                    "balanced_accuracy": balanced,
                }
                record = best_records[name]
                if candidate_loss < float(record["loss"]) - config.min_delta:
                    record.update(
                        {
                            "loss": candidate_loss,
                            "balanced_accuracy": balanced,
                            "epoch": epoch,
                            "state": copy.deepcopy(self.model_.state_dict()),
                        }
                    )
                    improved = True
            self.history_.append(
                {
                    "epoch": epoch,
                    "train_loss": total_loss / max(1, len(y_tr)),
                    "validation": validation_metrics,
                }
            )
            if improved:
                stale = 0
            else:
                stale += 1
                if stale >= config.patience:
                    break

        if any("state" not in record for record in best_records.values()):
            raise RuntimeError("orbit transport produced no validation checkpoint")
        preference = {name: -index for index, name in enumerate(config.candidate_names)}
        selected_name, selected_record = max(
            best_records.items(),
            key=lambda item: (
                float(item[1]["balanced_accuracy"]),
                -float(item[1]["loss"]),
                preference[item[0]],
            ),
        )
        self.model_.load_state_dict(selected_record["state"])
        self.model_.eval()
        self.selected_candidate_ = selected_name
        self.best_epoch_ = int(selected_record["epoch"])
        self.best_validation_loss_ = float(selected_record["loss"])
        self.best_validation_balanced_accuracy_ = float(
            selected_record["balanced_accuracy"]
        )
        self.selection_trace_ = [
            {
                "candidate": name,
                "best_epoch": int(record["epoch"]),
                "validation_loss": float(record["loss"]),
                "validation_balanced_accuracy": float(record["balanced_accuracy"]),
            }
            for name, record in best_records.items()
        ]
        self.epochs_run_ = len(self.history_)
        self.train_seconds_ = time.perf_counter() - started
        self.param_count_ = sum(
            parameter.numel() for parameter in self.model_.parameters()
        )
        self.trainable_param_count_ = sum(
            parameter.numel()
            for parameter in self.model_.parameters()
            if parameter.requires_grad
        )
        self.convex_anchor_learned_param_count_ = int(
            self.anchor_.model_.coef_.size + self.anchor_.model_.intercept_.size
        )
        self.total_learned_param_count_ = int(
            self.param_count_ + self.convex_anchor_learned_param_count_
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
        selected_candidate: str,
        channels: Sequence[str] = CHANNELS,
    ) -> "OrbitTransportClassifier":
        """Reset and refit on all source rows for a selected duration.

        Candidate identity is frozen from validation and this method has no
        validation or prediction-only argument by construction.
        """

        config = self.config
        epochs = int(epochs)
        if not 1 <= epochs <= config.epochs:
            raise ValueError("fixed refit epochs must be in [1, config.epochs]")
        if selected_candidate not in config.candidate_names:
            raise ValueError("selected ORBIT candidate is not in the frozen grid")
        y_source = self._validate_split(
            raw_source, cov_source, y_source, name="source"
        )
        raw_source = np.asarray(raw_source)
        cov_source = np.asarray(cov_source)
        if raw_source.ndim != 3:
            raise ValueError("source raw array must have shape (N, C, T)")
        if cov_source.ndim != 4 or cov_source.shape[-1] != cov_source.shape[-2]:
            raise ValueError("source covariances must have shape (N, bands, C, C)")
        if raw_source.shape[1] != cov_source.shape[-1]:
            raise ValueError("source raw/covariance channel dimensions disagree")
        if len(channels) != raw_source.shape[1]:
            raise ValueError("channel-name count does not match source arrays")

        _configure_torch_determinism(config.deterministic)
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
        source_views = self._prepare_views(raw_source, cov_source)
        source_tensors = [
            torch.from_numpy(value).to(device) for value in source_views
        ]
        labels = torch.from_numpy(y_source).to(device)
        self.model_ = OrbitTransportNet(
            n_channels=int(source_tensors[0].shape[1]),
            n_times=int(source_tensors[0].shape[2]),
            temporal_filters=config.temporal_filters,
            temporal_kernel=config.temporal_kernel,
            dynamics_channels=config.dynamics_channels,
            dynamics_kernel=config.dynamics_kernel,
            pool_kernel=config.pool_kernel,
            pool_stride=config.pool_stride,
            normalization=config.normalization,
            orientation_rank=config.orientation_rank,
            orientation_hidden=config.orientation_hidden,
            gate_hidden=config.gate_hidden,
        ).to(device)
        self.initial_model_state_ = copy.deepcopy(self.model_.state_dict())
        trainable_parameters = [
            parameter
            for parameter in self.model_.parameters()
            if parameter.requires_grad
        ]
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs
        )
        generator = torch.Generator(device=device).manual_seed(config.seed + 1)
        self.history_: list[dict[str, Any]] = []

        for epoch in range(epochs):
            self.model_.train()
            order = torch.randperm(len(labels), device=device, generator=generator)
            total_loss = 0.0
            for start in range(0, len(order), config.batch_size):
                rows = order[start : start + config.batch_size]
                current_labels = labels[rows]
                output = self.model_(*(value[rows] for value in source_tensors))
                main_loss = _binary_loss(output.logit, current_labels)
                transport_auxiliary = 0.5 * (
                    _binary_loss(output.energy_logit, current_labels)
                    + _binary_loss(output.dynamics_logit, current_labels)
                )
                reflected_labels = 1 - current_labels
                view_auxiliary = 0.25 * (
                    _binary_loss(
                        output.energy_view_logits[:, 0], current_labels
                    )
                    + _binary_loss(
                        output.energy_view_logits[:, 1], reflected_labels
                    )
                    + _binary_loss(
                        output.dynamics_view_logits[:, 0], current_labels
                    )
                    + _binary_loss(
                        output.dynamics_view_logits[:, 1], reflected_labels
                    )
                )
                orientation_cost = 0.5 * (
                    torch.mean(torch.square(output.energy_orientation))
                    + torch.mean(torch.square(output.dynamics_orientation))
                )
                loss = (
                    main_loss
                    + config.transport_auxiliary_weight * transport_auxiliary
                    + config.view_auxiliary_weight * view_auxiliary
                    + config.orientation_penalty * orientation_cost
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    trainable_parameters, config.gradient_clip
                )
                optimizer.step()
                total_loss += float(loss.detach()) * len(rows)
            scheduler.step()
            self.history_.append(
                {
                    "epoch": int(epoch),
                    "train_loss": total_loss / max(1, len(labels)),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
            )

        self.model_.eval()
        self.selected_candidate_ = selected_candidate
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
        self.convex_anchor_learned_param_count_ = int(
            self.anchor_.model_.coef_.size + self.anchor_.model_.intercept_.size
        )
        self.total_learned_param_count_ = int(
            self.param_count_ + self.convex_anchor_learned_param_count_
        )
        self.config_ = asdict(config)
        return self

    def _candidate_logits(
        self,
        raw: NDArray[np.floating],
        covariances: NDArray[np.floating],
        *,
        swapped: bool = False,
    ) -> dict[str, np.ndarray]:
        views = self._prepare_views(raw, covariances)
        if swapped:
            views = (views[1], views[0], views[3], views[2])
        results: dict[str, list[np.ndarray]] = {
            name: [] for name in OrbitTransportNet.CANDIDATE_NAMES
        }
        self.model_.eval()
        with torch.no_grad():
            for start in range(0, len(views[0]), 256):
                tensors = [
                    torch.from_numpy(value[start : start + 256]).to(self.device_)
                    for value in views
                ]
                output = self.model_(*tensors)
                for name, logit in output.candidate_logits().items():
                    results[name].append(logit.cpu().numpy())
        return {
            name: np.concatenate(parts).astype(np.float32, copy=False)
            for name, parts in results.items()
        }

    def predict_all_logits(
        self,
        raw: NDArray[np.floating],
        covariances: NDArray[np.floating],
    ) -> dict[str, np.ndarray]:
        """Return every predeclared exact-odd validation candidate."""

        return self._candidate_logits(raw, covariances)

    def predict_proba(
        self,
        raw: NDArray[np.floating],
        covariances: NDArray[np.floating],
    ) -> np.ndarray:
        logit = self._candidate_logits(raw, covariances)[self.selected_candidate_]
        positive = 1.0 / (1.0 + np.exp(-np.clip(logit, -40.0, 40.0)))
        return np.column_stack((1.0 - positive, positive))

    def predict(
        self,
        raw: NDArray[np.floating],
        covariances: NDArray[np.floating],
    ) -> np.ndarray:
        return self.predict_proba(raw, covariances).argmax(axis=1)

    def max_equivariance_error(
        self,
        raw: NDArray[np.floating],
        covariances: NDArray[np.floating],
    ) -> float:
        first = self._candidate_logits(raw, covariances)[self.selected_candidate_]
        reflected = self._candidate_logits(raw, covariances, swapped=True)[
            self.selected_candidate_
        ]
        return float(np.max(np.abs(first + reflected)))


__all__ = [
    "OddOrientationTransport",
    "OrbitTransportClassifier",
    "OrbitTransportConfig",
    "OrbitTransportNet",
    "OrbitTransportOutput",
    "SharedOrbitRawExpert",
    "TransportResult",
    "_configure_torch_determinism",
]
