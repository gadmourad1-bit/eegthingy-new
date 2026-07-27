"""Deterministic source-only optimization shared by proposed and baseline nets."""

from __future__ import annotations

import copy
import random
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    log_loss,
    roc_auc_score,
)
from torch import Tensor, nn


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 200
    batch_size: int = 64
    learning_rate: float = 8e-4
    weight_decay: float = 5e-4
    patience: int = 35
    min_delta: float = 1e-4
    label_smoothing: float = 0.05
    segment_probability: float = 0.5
    segment_count: int = 8
    time_shift_samples: int = 8
    noise_std: float = 0.01
    reflection_weight: float = 0.0
    reflection_probability: float = 0.0
    gradient_clip: float = 5.0
    seed: int = 7
    device: str = "cuda"


def configure_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)


def _forward(
    model: nn.Module,
    x: Tensor,
    positions: Tensor,
) -> Tensor:
    if bool(getattr(model, "uses_positions", False)):
        output = model(x, positions)
    else:
        output = model(x)
    if isinstance(output, tuple):
        output = output[0]
    if output.ndim > 2:
        output = output.flatten(start_dim=2).mean(dim=2)
    if output.ndim != 2:
        raise RuntimeError(f"model returned invalid logit shape {tuple(output.shape)}")
    return output


def _segment_reconstruct(
    x: Tensor,
    y: Tensor,
    *,
    segments: int,
    generator: torch.Generator,
) -> Tensor:
    if segments <= 1:
        return x
    result = x.clone()
    boundaries = torch.linspace(0, x.shape[-1], segments + 1, device=x.device).long()
    for label in torch.unique(y):
        rows = torch.nonzero(y == label, as_tuple=False).flatten()
        if len(rows) < 2:
            continue
        for segment in range(segments):
            sampled = rows[
                torch.randint(len(rows), (len(rows),), device=x.device, generator=generator)
            ]
            start, stop = int(boundaries[segment]), int(boundaries[segment + 1])
            result[rows, :, start:stop] = x[sampled, :, start:stop]
    return result


def _augment(
    x: Tensor,
    y: Tensor,
    config: TrainConfig,
    generator: torch.Generator,
) -> Tensor:
    result = x
    if config.segment_probability > 0.0:
        if float(torch.rand((), device=x.device, generator=generator)) < config.segment_probability:
            result = _segment_reconstruct(
                result, y, segments=config.segment_count, generator=generator
            )
    if config.time_shift_samples > 0:
        shifts = torch.randint(
            -config.time_shift_samples,
            config.time_shift_samples + 1,
            (len(result),),
            device=result.device,
            generator=generator,
        )
        shifted = torch.empty_like(result)
        for row, shift_tensor in enumerate(shifts):
            shift = int(shift_tensor)
            if shift == 0:
                shifted[row] = result[row]
            elif shift > 0:
                shifted[row, :, :shift] = 0.0
                shifted[row, :, shift:] = result[row, :, :-shift]
            else:
                shifted[row, :, shift:] = 0.0
                shifted[row, :, :shift] = result[row, :, -shift:]
        result = shifted
    if config.noise_std > 0.0:
        noise = torch.randn(
            result.shape,
            dtype=result.dtype,
            device=result.device,
            generator=generator,
        )
        result = result + config.noise_std * noise
    return result


def _class_permutation(n_outputs: int, device: torch.device) -> Tensor:
    if n_outputs < 2:
        raise ValueError("motor-imagery output requires at least two classes")
    result = torch.arange(n_outputs, device=device)
    result[0], result[1] = result[1].clone(), result[0].clone()
    return result


def _optimization_epoch(
    model: nn.Module,
    x_train: Tensor,
    y_train: Tensor,
    positions: Tensor,
    mirror_index: Tensor | None,
    class_permutation: Tensor,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    generator: torch.Generator,
) -> float:
    """Run one deterministic source-only optimization epoch."""

    model.train()
    order = torch.randperm(len(x_train), device=x_train.device, generator=generator)
    train_loss = 0.0
    seen = 0
    for start in range(0, len(order), config.batch_size):
        rows = order[start : start + config.batch_size]
        batch_x = x_train[rows]
        batch_y = y_train[rows]
        if mirror_index is not None and config.reflection_probability > 0.0:
            reflect = torch.rand(
                len(rows), device=x_train.device, generator=generator
            ) < config.reflection_probability
            if bool(torch.any(reflect)):
                batch_x = batch_x.clone()
                batch_y = batch_y.clone()
                batch_x[reflect] = batch_x[reflect][:, mirror_index]
                batch_y[reflect] = class_permutation[batch_y[reflect]]
        batch_x = _augment(batch_x, batch_y, config, generator)
        optimizer.zero_grad(set_to_none=True)
        logits = _forward(model, batch_x, positions)
        loss = nn.functional.cross_entropy(
            logits, batch_y, label_smoothing=config.label_smoothing
        )
        if mirror_index is not None and config.reflection_weight > 0.0:
            mirrored_logits = _forward(model, batch_x[:, mirror_index], positions)
            direct_centered = logits - logits.mean(dim=1, keepdim=True)
            mirror_centered = mirrored_logits - mirrored_logits.mean(dim=1, keepdim=True)
            reflection = nn.functional.mse_loss(
                mirror_centered, direct_centered[:, class_permutation]
            )
            loss = loss + config.reflection_weight * reflection
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()
        train_loss += float(loss.detach()) * len(rows)
        seen += len(rows)
    return train_loss / max(seen, 1)


def fit_model(
    model: nn.Module,
    x_train: NDArray[np.float32],
    y_train: NDArray[np.int64],
    x_validation: NDArray[np.float32],
    y_validation: NDArray[np.int64],
    positions: NDArray[np.float32],
    *,
    config: TrainConfig = TrainConfig(),
    mirror_index: NDArray[np.int64] | None = None,
) -> dict[str, Any]:
    configure_determinism(config.seed)
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model = model.to(device)
    x_train_t = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    y_train_t = torch.as_tensor(y_train, dtype=torch.long, device=device)
    x_validation_t = torch.as_tensor(x_validation, dtype=torch.float32, device=device)
    y_validation_t = torch.as_tensor(y_validation, dtype=torch.long, device=device)
    positions_t = torch.as_tensor(positions, dtype=torch.float32, device=device)
    mirror_t = (
        torch.as_tensor(mirror_index, dtype=torch.long, device=device)
        if mirror_index is not None
        else None
    )
    n_outputs = int(np.max(y_train)) + 1
    permutation = _class_permutation(n_outputs, device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    generator = torch.Generator(device=device).manual_seed(config.seed + 7919)
    best_state: dict[str, Tensor] | None = None
    best_loss = float("inf")
    best_epoch = -1
    stale = 0
    history: list[dict[str, float]] = []
    started = time.perf_counter()

    for epoch in range(config.epochs):
        train_loss = _optimization_epoch(
            model,
            x_train_t,
            y_train_t,
            positions_t,
            mirror_t,
            permutation,
            optimizer,
            config,
            generator,
        )
        scheduler.step()

        model.eval()
        with torch.no_grad():
            validation_logits = _forward(model, x_validation_t, positions_t)
            validation_loss = float(
                nn.functional.cross_entropy(validation_logits, y_validation_t)
            )
            validation_accuracy = float(
                (validation_logits.argmax(dim=1) == y_validation_t).float().mean()
            )
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if validation_loss < best_loss - config.min_delta:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break

    if best_state is None:
        raise RuntimeError("training produced no finite checkpoint")
    model.load_state_dict(best_state)
    return {
        "model": model,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "best_validation_loss": best_loss,
        "fit_seconds": time.perf_counter() - started,
        "history": history,
        "config": asdict(config),
    }


def refit_model(
    model: nn.Module,
    x_source: NDArray[np.float32],
    y_source: NDArray[np.int64],
    positions: NDArray[np.float32],
    *,
    epochs: int,
    config: TrainConfig = TrainConfig(),
    mirror_index: NDArray[np.int64] | None = None,
) -> dict[str, Any]:
    """Refit from initialization on train+validation for a selected duration.

    ``epochs`` must have been selected without looking at the prediction-only
    test partition.  The cosine schedule retains the original ``T_max`` so the
    learning-rate trajectory through the selected epoch matches model
    selection; only the number of source rows changes.
    """

    if not 1 <= epochs <= config.epochs:
        raise ValueError("refit epochs must be in [1, config.epochs]")
    configure_determinism(config.seed)
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model = model.to(device)
    x_source_t = torch.as_tensor(x_source, dtype=torch.float32, device=device)
    y_source_t = torch.as_tensor(y_source, dtype=torch.long, device=device)
    positions_t = torch.as_tensor(positions, dtype=torch.float32, device=device)
    mirror_t = (
        torch.as_tensor(mirror_index, dtype=torch.long, device=device)
        if mirror_index is not None
        else None
    )
    permutation = _class_permutation(int(np.max(y_source)) + 1, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )
    generator = torch.Generator(device=device).manual_seed(config.seed + 7919)
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    for epoch in range(epochs):
        train_loss = _optimization_epoch(
            model,
            x_source_t,
            y_source_t,
            positions_t,
            mirror_t,
            permutation,
            optimizer,
            config,
            generator,
        )
        scheduler.step()
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
    return {
        "model": model,
        "epochs_run": epochs,
        "fit_seconds": time.perf_counter() - started,
        "history": history,
        "config": asdict(config),
    }


@torch.no_grad()
def predict_probabilities(
    model: nn.Module,
    x: NDArray[np.float32],
    positions: NDArray[np.float32],
    *,
    device: str = "cuda",
    batch_size: int = 256,
) -> NDArray[np.float64]:
    model.eval()
    target = torch.device(device)
    model = model.to(target)
    positions_t = torch.as_tensor(positions, dtype=torch.float32, device=target)
    probabilities: list[np.ndarray] = []
    for start in range(0, len(x), batch_size):
        batch = torch.as_tensor(x[start : start + batch_size], dtype=torch.float32, device=target)
        logits = _forward(model, batch, positions_t)
        probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(probabilities, axis=0).astype(np.float64, copy=False)


def classification_metrics(
    labels: NDArray[np.int64], probabilities: NDArray[np.float64]
) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = np.clip(probabilities, 1e-12, 1.0)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    predictions = probabilities.argmax(axis=1)
    result = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "cohen_kappa": float(cohen_kappa_score(labels, predictions)),
        "negative_log_likelihood": float(log_loss(labels, probabilities)),
    }
    try:
        if probabilities.shape[1] == 2:
            result["roc_auc"] = float(roc_auc_score(labels, probabilities[:, 1]))
        else:
            result["roc_auc"] = float(
                roc_auc_score(labels, probabilities, multi_class="ovr", average="macro")
            )
    except ValueError:
        result["roc_auc"] = float("nan")
    return result
