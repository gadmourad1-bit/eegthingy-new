"""Metrics for decoder quality, calibration, and selective prediction."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class ClassificationMetrics:
    n: int
    accuracy: float
    balanced_accuracy: float
    kappa: float
    roc_auc: float
    brier: float
    ece: float
    coverage: float
    selective_accuracy: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """Return equal-width expected calibration error for binary probabilities."""

    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim == 2:
        if p.shape[1] != 2:
            raise ValueError("binary probabilities must have two columns")
        confidence = p.max(axis=1)
        predicted = p.argmax(axis=1)
    elif p.ndim == 1:
        confidence = np.maximum(p, 1.0 - p)
        predicted = (p >= 0.5).astype(np.int64)
    else:
        raise ValueError("probabilities must be shape (N,) or (N, 2)")
    if len(y) != len(confidence):
        raise ValueError("labels and probabilities have different lengths")
    if len(y) == 0:
        return float("nan")

    correct = (predicted == y).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = float(len(y))
    ece = 0.0
    for index in range(n_bins):
        lower, upper = edges[index], edges[index + 1]
        if index == n_bins - 1:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)
        if np.any(mask):
            ece += mask.sum() / total * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    commit_confidence: float = 0.85,
) -> ClassificationMetrics:
    """Compute binary accuracy, calibration, and confidence-gated coverage metrics."""

    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError("probabilities must have shape (N, 2)")
    if len(y) != len(p):
        raise ValueError("labels and probabilities have different lengths")
    if len(y) == 0:
        raise ValueError("cannot score an empty set")

    pred = p.argmax(axis=1)
    confidence = p.max(axis=1)
    committed = confidence >= commit_confidence
    selective_accuracy = (
        float(accuracy_score(y[committed], pred[committed])) if np.any(committed) else float("nan")
    )
    auc = float(roc_auc_score(y, p[:, 1])) if np.unique(y).size == 2 else float("nan")
    return ClassificationMetrics(
        n=int(len(y)),
        accuracy=float(accuracy_score(y, pred)),
        balanced_accuracy=float(balanced_accuracy_score(y, pred)),
        kappa=float(cohen_kappa_score(y, pred)),
        roc_auc=auc,
        brier=float(np.mean((p[:, 1] - y) ** 2)),
        ece=expected_calibration_error(y, p),
        coverage=float(committed.mean()),
        selective_accuracy=selective_accuracy,
    )
