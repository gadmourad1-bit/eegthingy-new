"""Small-data training and inference utilities for geometric EEG models."""

from __future__ import annotations

import copy
import os
import random
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from ..shared.augment import augment_batch, left_right_swap_index
from .config import CHANNELS
from .model import GeoAdaptNet


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 250
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-3
    patience: int = 35
    min_delta: float = 1e-4
    label_smoothing: float = 0.05
    intent_loss_weight: float = 0.25
    gradient_clip: float = 5.0
    seed: int = 7
    device: str = "auto"
    num_workers: int = 0
    # Auxiliary loss on the full-strength anchor+residual prediction (gate forced
    # to 1).  With the default gated combination the residual head only receives
    # gradient scaled by the near-zero gate, so the branch never learns and the
    # gate never has a reason to open.  A positive weight here trains the residual
    # head directly, letting the main-task gradient then open the gate.  Zero
    # reproduces the original anchor-collapsing behaviour exactly.
    deep_supervision_weight: float = 0.0
    # AdamW weight decay pulls the gate logit toward sigmoid(0)=0.5 (or shrinks a
    # deliberately larger init); excluding it lets the data, not the regularizer,
    # decide the residual mixing strength.
    exclude_gate_from_weight_decay: bool = False
    # Early stopping monitors "loss" by default; "balanced_accuracy" or the
    # "blend" (loss minus balanced accuracy) can select higher-accuracy epochs.
    select_metric: str = "loss"
    # Fit-side covariance augmentation (training loader only).  0 disables.
    lr_swap_prob: float = 0.0
    mixup_alpha: float = 0.0
    augment_eps: float = 1e-5
    # Determinism is on by default for reproducible research runs.  Disabling it
    # (the "--fast" profile, paired with a larger batch) lets CUDA use faster
    # non-deterministic kernels and TF32 matmuls; seeds still fix init/shuffling,
    # so runs stay close but are no longer bit-exact.
    deterministic: bool = True


@dataclass
class TrainingResult:
    model: GeoAdaptNet
    best_epoch: int
    epochs_ran: int
    best_validation_loss: float
    best_validation_balanced_accuracy: float
    train_seconds: float
    history: list[dict[str, float]]
    config: dict[str, Any]


@dataclass
class FixedTrainingResult:
    """Result of the leakage-safe refit performed after epoch selection.

    ``train_model`` chooses an epoch count using a complete held-out recording.
    This companion result records the subsequent from-scratch fit on all source
    recordings for exactly that many epochs; it never inspects the outer test
    recording.
    """

    model: GeoAdaptNet
    epochs_ran: int
    train_seconds: float
    history: list[dict[str, float]]
    config: dict[str, Any]


class CovarianceDataset(Dataset[tuple[Tensor, Tensor, Tensor, Tensor]]):
    """Tensor dataset with optional per-sample references and rest/intent targets.

    ``labels == -1`` denotes a neutral/rest example.  Such examples contribute to
    the optional intent loss but never to the left/right cross-entropy loss.
    """

    def __init__(
        self,
        covariances: np.ndarray | Tensor,
        labels: np.ndarray | Tensor,
        *,
        log_references: np.ndarray | Tensor | None = None,
        intent_targets: np.ndarray | Tensor | None = None,
    ) -> None:
        cov = torch.as_tensor(covariances, dtype=torch.float32)
        target = torch.as_tensor(labels, dtype=torch.long)
        if cov.ndim != 4 or cov.shape[-1] != cov.shape[-2]:
            raise ValueError(
                "covariances must have shape (N, bands, channels, channels)"
            )
        if len(cov) != len(target):
            raise ValueError("covariances and labels have different lengths")
        if log_references is None:
            refs = torch.empty((len(cov), 0), dtype=torch.float32)
        else:
            refs = torch.as_tensor(log_references, dtype=torch.float32)
            if refs.ndim == 3:
                refs = refs.unsqueeze(0).expand(len(cov), -1, -1, -1)
            if len(refs) != len(cov):
                raise ValueError("references and covariances have different lengths")
        if intent_targets is None:
            intent = (target >= 0).to(torch.float32)
        else:
            intent = torch.as_tensor(intent_targets, dtype=torch.float32)
            if len(intent) != len(cov):
                raise ValueError(
                    "intent targets and covariances have different lengths"
                )
        self.covariances = cov
        self.labels = target
        self.references = refs
        self.intent_targets = intent

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return (
            self.covariances[index],
            self.labels[index],
            self.references[index],
            self.intent_targets[index],
        )


@lru_cache(maxsize=1)
def _mps_supports_spectral_ops() -> bool:
    """Return whether MPS can execute the eigendecomposition this model needs.

    MPS availability alone is insufficient: current Apple builds can expose a
    Metal device while leaving ``torch.linalg.eigh`` unimplemented.  GeoAdaptNet
    uses that operation throughout its SPD path, so automatic selection must
    probe the required kernel and safely retain CPU execution when it is absent.
    Users who enable PyTorch's MPS CPU fallback can still select ``mps`` explicitly.
    """

    if not torch.backends.mps.is_available():
        return False
    try:
        torch.linalg.eigh(torch.eye(2, dtype=torch.float32, device="mps"))
        torch.mps.synchronize()
    except (NotImplementedError, RuntimeError):
        return False
    return True


def resolve_device(preference: str = "auto") -> torch.device:
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_supports_spectral_ops():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(preference)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {preference!r} was requested but CUDA is unavailable"
        )
    if device.type == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                f"MPS device {preference!r} was requested but MPS is unavailable"
            )
        if not _mps_supports_spectral_ops():
            raise RuntimeError(
                "MPS is available but lacks the torch.linalg.eigh kernel required by "
                "GeoAdaptNet; use device='cpu', or set PYTORCH_ENABLE_MPS_FALLBACK=1 "
                "before Python starts to opt into the slower fallback"
            )
    return device


def set_reproducible_seed(seed: int, *, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        if (
            torch.cuda.is_available()
            and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
        ):
            raise RuntimeError(
                "deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8 "
                "before Python starts"
            )
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    else:
        torch.use_deterministic_algorithms(False, warn_only=False)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def _reference_or_none(reference_batch: Tensor) -> Tensor | None:
    return None if reference_batch.numel() == 0 else reference_batch


def _batch_loss(
    model: GeoAdaptNet,
    covariances: Tensor,
    labels: Tensor,
    references: Tensor,
    intent_targets: Tensor,
    *,
    label_smoothing: float,
    intent_weight: float,
    deep_supervision_weight: float = 0.0,
) -> tuple[Tensor, Tensor]:
    output = model(covariances, log_reference=_reference_or_none(references))
    task_mask = labels >= 0
    if torch.any(task_mask):
        task_loss = nn.functional.cross_entropy(
            output.logits[task_mask], labels[task_mask], label_smoothing=label_smoothing
        )
        if deep_supervision_weight > 0.0:
            # Full-strength anchor+residual prediction (as if the gate were open).
            # This gives the residual head an ungated gradient so it learns real
            # structure; the gate is still driven only by the main gated loss.
            full = output.anchor_logits + output.residual_logits
            deep_loss = nn.functional.cross_entropy(
                full[task_mask], labels[task_mask], label_smoothing=label_smoothing
            )
            task_loss = task_loss + deep_supervision_weight * deep_loss
    else:
        task_loss = output.logits.sum() * 0.0
    if output.intent_logit is not None and intent_weight > 0.0:
        intent_loss = nn.functional.binary_cross_entropy_with_logits(
            output.intent_logit.reshape(-1), intent_targets
        )
    else:
        intent_loss = output.logits.sum() * 0.0
    return task_loss + intent_weight * intent_loss, output.logits


def _evaluate_loss(
    model: GeoAdaptNet,
    loader: DataLoader,
    device: torch.device,
    config: TrainConfig,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    task_truth: list[np.ndarray] = []
    task_pred: list[np.ndarray] = []
    with torch.no_grad():
        for cov, labels, references, intent in loader:
            cov, labels = cov.to(device), labels.to(device)
            references, intent = references.to(device), intent.to(device)
            loss, logits = _batch_loss(
                model,
                cov,
                labels,
                references,
                intent,
                label_smoothing=0.0,
                intent_weight=config.intent_loss_weight,
            )
            total_loss += float(loss) * len(cov)
            total_examples += len(cov)
            mask = labels >= 0
            if torch.any(mask):
                task_truth.append(labels[mask].cpu().numpy())
                task_pred.append(logits[mask].argmax(dim=1).cpu().numpy())
    mean_loss = total_loss / max(1, total_examples)
    if task_truth:
        balanced = float(
            balanced_accuracy_score(
                np.concatenate(task_truth), np.concatenate(task_pred)
            )
        )
    else:
        balanced = float("nan")
    return mean_loss, balanced


def _build_optimizer(model: GeoAdaptNet, config: TrainConfig) -> torch.optim.Optimizer:
    """AdamW, optionally holding the residual gate out of weight decay."""

    if not config.exclude_gate_from_weight_decay:
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    decayed: list[Tensor] = []
    undecayed: list[Tensor] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (undecayed if name.endswith("residual_gate_logit") else decayed).append(param)
    return torch.optim.AdamW(
        [
            {"params": decayed, "weight_decay": config.weight_decay},
            {"params": undecayed, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
    )


def _augmentation_enabled(config: TrainConfig) -> bool:
    return config.lr_swap_prob > 0.0 or config.mixup_alpha > 0.0


def _selection_score(loss: float, balanced: float, metric: str) -> float:
    """Return a lower-is-better selection score for the requested metric."""

    if metric == "loss":
        return loss
    finite_balanced = balanced if balanced == balanced else 0.0  # NaN -> 0
    if metric == "balanced_accuracy":
        return -finite_balanced
    if metric == "blend":
        return loss - finite_balanced
    raise ValueError(f"unknown select_metric: {metric!r}")


def train_model(
    model: GeoAdaptNet,
    train_data: CovarianceDataset,
    validation_data: CovarianceDataset,
    config: TrainConfig = TrainConfig(),
) -> TrainingResult:
    """Train with early stopping on a caller-supplied, group-held-out validation set."""

    if len(train_data) == 0 or len(validation_data) == 0:
        raise ValueError("training and validation datasets must be non-empty")
    set_reproducible_seed(config.seed, deterministic=config.deterministic)
    device = resolve_device(config.device)
    model = model.to(device)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=config.num_workers,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=max(config.batch_size, 128),
        shuffle=False,
        num_workers=config.num_workers,
    )
    optimizer = _build_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )
    augment = _augmentation_enabled(config)
    swap_index = torch.as_tensor(
        left_right_swap_index(CHANNELS), dtype=torch.long, device=device
    )
    augment_generator = torch.Generator(device=device).manual_seed(config.seed + 1)

    best_state: dict[str, Tensor] | None = None
    best_score = float("inf")
    best_loss = float("inf")
    best_balanced = float("nan")
    best_epoch = -1
    stale = 0
    history: list[dict[str, float]] = []
    started = time.perf_counter()

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        train_examples = 0
        for cov, labels, references, intent in train_loader:
            cov, labels = cov.to(device), labels.to(device)
            references, intent = references.to(device), intent.to(device)
            if augment:
                with torch.no_grad():
                    cov, labels, augmented_refs = augment_batch(
                        cov,
                        labels,
                        references,
                        swap_index=swap_index,
                        lr_swap_prob=config.lr_swap_prob,
                        mixup_alpha=config.mixup_alpha,
                        eps=config.augment_eps,
                        generator=augment_generator,
                    )
                if augmented_refs is not None:
                    references = augmented_refs
            optimizer.zero_grad(set_to_none=True)
            loss, _ = _batch_loss(
                model,
                cov,
                labels,
                references,
                intent,
                label_smoothing=config.label_smoothing,
                intent_weight=config.intent_loss_weight,
                deep_supervision_weight=config.deep_supervision_weight,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            train_loss += float(loss.detach()) * len(cov)
            train_examples += len(cov)
        scheduler.step()

        validation_loss, validation_balanced = _evaluate_loss(
            model, validation_loader, device, config
        )
        score = _selection_score(
            validation_loss, validation_balanced, config.select_metric
        )
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss / max(1, train_examples),
                "validation_loss": validation_loss,
                "validation_balanced_accuracy": validation_balanced,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if score < best_score - config.min_delta:
            best_score = score
            best_loss = validation_loss
            best_balanced = validation_balanced
            best_epoch = epoch
            stale = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= config.patience:
                break

    if best_state is None:
        raise RuntimeError("training completed without a finite validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    elapsed = time.perf_counter() - started
    return TrainingResult(
        model=model,
        best_epoch=best_epoch,
        epochs_ran=len(history),
        best_validation_loss=best_loss,
        best_validation_balanced_accuracy=best_balanced,
        train_seconds=elapsed,
        history=history,
        config=asdict(config),
    )


def train_fixed_epochs(
    model: GeoAdaptNet,
    train_data: CovarianceDataset,
    *,
    epochs: int,
    config: TrainConfig = TrainConfig(),
) -> FixedTrainingResult:
    """Refit a fresh model on all source sessions for a fixed epoch count.

    The epoch count must have been selected without looking at the outer test
    labels (normally by :func:`train_model` on an earlier, session-held-out
    validation recording).  Keeping this operation separate makes that boundary
    explicit and lets the experiment runner record both stages.
    """

    if len(train_data) == 0:
        raise ValueError("training dataset must be non-empty")
    if epochs <= 0:
        raise ValueError("epochs must be positive")

    set_reproducible_seed(config.seed, deterministic=config.deterministic)
    device = resolve_device(config.device)
    model = model.to(device)
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=config.num_workers,
    )
    optimizer = _build_optimizer(model, config)
    # Match the learning-rate trajectory used during epoch selection.  Compressing
    # the cosine horizon into ``epochs`` would refit with a different optimizer
    # schedule from the one whose stopping point was validated.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )
    history: list[dict[str, float]] = []
    started = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for cov, labels, references, intent in loader:
            cov, labels = cov.to(device), labels.to(device)
            references, intent = references.to(device), intent.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = _batch_loss(
                model,
                cov,
                labels,
                references,
                intent,
                label_smoothing=config.label_smoothing,
                intent_weight=config.intent_loss_weight,
                deep_supervision_weight=config.deep_supervision_weight,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            total_loss += float(loss.detach()) * len(cov)
            total_examples += len(cov)
        scheduler.step()
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": total_loss / max(1, total_examples),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )

    model.eval()
    return FixedTrainingResult(
        model=model,
        epochs_ran=epochs,
        train_seconds=time.perf_counter() - started,
        history=history,
        config=asdict(config),
    )


@torch.no_grad()
def predict_proba(
    model: GeoAdaptNet,
    data: CovarianceDataset,
    *,
    device: str = "auto",
    batch_size: int = 256,
    branch: str = "full",
) -> tuple[np.ndarray, np.ndarray | None]:
    if branch not in {"full", "anchor", "residual"}:
        raise ValueError("branch must be 'full', 'anchor', or 'residual'")
    resolved = resolve_device(device)
    model = model.to(resolved).eval()
    loader = DataLoader(data, batch_size=batch_size, shuffle=False)
    probabilities: list[Tensor] = []
    intent_probabilities: list[Tensor] = []
    has_intent = True
    for cov, _, references, _ in loader:
        output = model(
            cov.to(resolved), log_reference=_reference_or_none(references.to(resolved))
        )
        logits = {
            "full": output.logits,
            "anchor": output.anchor_logits,
            "residual": output.residual_logits,
        }[branch]
        probabilities.append(torch.softmax(logits, dim=1).cpu())
        if output.intent_logit is None:
            has_intent = False
        else:
            intent_probabilities.append(torch.sigmoid(output.intent_logit).cpu())
    task = torch.cat(probabilities).numpy()
    intent = torch.cat(intent_probabilities).numpy() if has_intent else None
    return task, intent


@torch.no_grad()
def batch_one_latency_ms(
    model: GeoAdaptNet,
    covariance: np.ndarray | Tensor,
    *,
    log_reference: np.ndarray | Tensor | None = None,
    device: str = "auto",
    warmup: int = 30,
    repetitions: int = 200,
) -> float:
    """Measure synchronized end-to-end model latency for one covariance window."""

    resolved = resolve_device(device)
    model = model.to(resolved).eval()
    cov = torch.as_tensor(covariance, dtype=torch.float32, device=resolved)
    if cov.ndim == 3:
        cov = cov.unsqueeze(0)
    ref = None
    if log_reference is not None:
        ref = torch.as_tensor(log_reference, dtype=torch.float32, device=resolved)
        if ref.ndim == 3:
            ref = ref.unsqueeze(0)

    def synchronize() -> None:
        if resolved.type == "cuda":
            torch.cuda.synchronize(resolved)
        elif resolved.type == "mps":
            torch.mps.synchronize()

    for _ in range(warmup):
        model(cov, log_reference=ref)
    synchronize()
    started = time.perf_counter()
    for _ in range(repetitions):
        model(cov, log_reference=ref)
    synchronize()
    return (time.perf_counter() - started) * 1000.0 / repetitions


def load_model_checkpoint(
    path: str | Path, *, map_location: str | torch.device = "cpu"
) -> tuple[GeoAdaptNet, dict[str, Any]]:
    """Load a benchmark checkpoint without enabling arbitrary pickle objects."""

    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or payload.get("model") != "GeoAdaptNet":
        raise ValueError("checkpoint is not a GeoAdaptNet benchmark artifact")
    model = GeoAdaptNet(**dict(payload.get("model_config", {})))
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(map_location).eval()
    metadata = {key: value for key, value in payload.items() if key != "state_dict"}
    return model, metadata
