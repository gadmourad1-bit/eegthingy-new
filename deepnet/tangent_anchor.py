"""Convex-head tangent anchor -- GeoAdaptNet's geometry, fixed estimator.

The diagnosis of the local gap to classical Riemann-TS+LR showed that GeoAdaptNet's
log-Euclidean tangent *features* are sound (they match classical Riemann on identical,
non-transductive footing); the deficit was entirely in how the linear head is fit --
small-batch AdamW with early stopping, entangled with an inert gated residual, instead
of a convex, strongly-regularized objective.

This classifier realizes the measured fix directly.  It computes exactly GeoAdaptNet's
anchor features -- recenter each band covariance by a *frozen train* log-Euclidean
reference (``log(R^{-1/2} C R^{-1/2})``, the same ``SPDMomentumBatchNorm`` congruence,
not the crude ``log C - R`` approximation) and vectorize the tangent -- then fits a
StandardScaler + convex L2 logistic regression instead of the SGD head.  The reference,
scaler, and head are all fit on training data only, so it is fair and non-transductive.
"""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .spd import log_euclidean_recenter, matrix_log, upper_vectorize


def _tangent(covariances: NDArray[np.floating], log_reference: torch.Tensor) -> NDArray[np.float64]:
    """Recenter by the frozen reference, matrix-log, vectorize -> (N, bands*C(C+1)/2)."""

    cov = torch.as_tensor(np.asarray(covariances), dtype=torch.float64)
    aligned = log_euclidean_recenter(cov, log_reference)
    return upper_vectorize(matrix_log(aligned)).flatten(start_dim=1).numpy()


class TangentAnchorClassifier:
    """GeoAdaptNet's frozen-reference tangent features + a convex L2 logistic head."""

    def __init__(self, C: float = 1.0, max_iter: int = 2000) -> None:
        self.C = float(C)
        self.max_iter = int(max_iter)

    def fit(self, covariances: NDArray[np.floating], labels: NDArray[np.int64]) -> "TangentAnchorClassifier":
        cov = torch.as_tensor(np.asarray(covariances), dtype=torch.float64)
        # Frozen per-band log-Euclidean reference from the training covariances only.
        self.log_reference_ = matrix_log(cov).mean(dim=0)
        features = _tangent(covariances, self.log_reference_)
        self.scaler_ = StandardScaler().fit(features)
        self.model_ = LogisticRegression(C=self.C, max_iter=self.max_iter).fit(
            self.scaler_.transform(features), np.asarray(labels)
        )
        self.param_count_ = int(self.model_.coef_.size + self.model_.intercept_.size)
        return self

    def predict_proba(self, covariances: NDArray[np.floating]) -> NDArray[np.float64]:
        features = self.scaler_.transform(_tangent(covariances, self.log_reference_))
        return self.model_.predict_proba(features)


__all__ = ["TangentAnchorClassifier"]
