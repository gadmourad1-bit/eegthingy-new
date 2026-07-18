"""Leakage-aware classical baselines for geometric EEG experiments."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from mne.decoding import CSP
from pyriemann.tangentspace import TangentSpace
try:  # pyRiemann >= 0.12
    from pyriemann.geometry.base import invsqrtm
    from pyriemann.geometry.mean import mean_riemann
except ImportError:  # pyRiemann <= 0.11 (the Mac acquisition environment)
    from pyriemann.utils.base import invsqrtm
    from pyriemann.utils.mean import mean_riemann
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class ProbabilisticBaseline(Protocol):
    classes_: np.ndarray

    def fit(self, features: np.ndarray, labels: np.ndarray, groups: np.ndarray | None = None): ...

    def calibrate(self, unlabeled_features: np.ndarray): ...

    def predict_proba(self, features: np.ndarray) -> np.ndarray: ...


def _check_groups(groups: np.ndarray | None, n_samples: int) -> np.ndarray | None:
    if groups is None:
        return None
    group_array = np.asarray(groups)
    if len(group_array) != n_samples:
        raise ValueError("groups and samples have different lengths")
    return group_array


def _euclidean_whitener(epochs: np.ndarray) -> np.ndarray:
    # epochs: observations x channels x time
    reference = np.einsum("nct,ndt->cd", epochs, epochs, optimize=True)
    reference /= float(len(epochs) * epochs.shape[-1])
    values, vectors = np.linalg.eigh((reference + reference.T) * 0.5)
    values = np.clip(values, 1e-12, None)
    return (vectors * values**-0.5) @ vectors.T


def _align_epochs(epochs: np.ndarray, whiteners: list[np.ndarray]) -> np.ndarray:
    aligned = np.empty_like(epochs, dtype=np.float64)
    for band, whitener in enumerate(whiteners):
        aligned[:, band] = np.einsum(
            "cd,ndt->nct", whitener, epochs[:, band], optimize=True
        )
    return aligned


class EAFilterBankCSP:
    """Euclidean alignment + per-band CSP(2) + shrinkage LDA.

    Source domains are aligned independently when ``groups`` are supplied.  A target
    reference is never inferred in ``predict_proba``; callers must explicitly invoke
    :meth:`calibrate` with the allowed unlabeled calibration prefix.
    """

    def __init__(self, n_components: int = 2) -> None:
        self.n_components = n_components

    def fit(
        self, epochs: np.ndarray, labels: np.ndarray, groups: np.ndarray | None = None
    ) -> "EAFilterBankCSP":
        x = np.asarray(epochs, dtype=np.float64)
        y = np.asarray(labels)
        if x.ndim != 4:
            raise ValueError("filter-bank epochs must have shape (N, bands, channels, time)")
        group_array = _check_groups(groups, len(x))
        self.classes_ = np.unique(y)
        self.source_whiteners_ = [_euclidean_whitener(x[:, band]) for band in range(x.shape[1])]
        if group_array is None:
            aligned = _align_epochs(x, self.source_whiteners_)
        else:
            aligned = np.empty_like(x)
            for group in np.unique(group_array):
                mask = group_array == group
                whiteners = [_euclidean_whitener(x[mask, band]) for band in range(x.shape[1])]
                aligned[mask] = _align_epochs(x[mask], whiteners)

        self.csp_: list[CSP] = []
        features: list[np.ndarray] = []
        for band in range(x.shape[1]):
            csp = CSP(
                n_components=self.n_components,
                reg="ledoit_wolf",
                log=True,
                norm_trace=False,
            )
            features.append(csp.fit_transform(aligned[:, band], y))
            self.csp_.append(csp)
        self.classifier_ = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        self.classifier_.fit(np.hstack(features), y)
        self.target_whiteners_ = self.source_whiteners_
        return self

    def calibrate(self, unlabeled_epochs: np.ndarray) -> "EAFilterBankCSP":
        x = np.asarray(unlabeled_epochs, dtype=np.float64)
        if len(x) < 2:
            raise ValueError("at least two unlabeled epochs are required for calibration")
        self.target_whiteners_ = [_euclidean_whitener(x[:, band]) for band in range(x.shape[1])]
        return self

    def _features(self, epochs: np.ndarray) -> np.ndarray:
        aligned = _align_epochs(np.asarray(epochs, dtype=np.float64), self.target_whiteners_)
        return np.hstack(
            [csp.transform(aligned[:, band]) for band, csp in enumerate(self.csp_)]
        )

    def predict_proba(self, epochs: np.ndarray) -> np.ndarray:
        return self.classifier_.predict_proba(self._features(epochs))

    def predict(self, epochs: np.ndarray) -> np.ndarray:
        return self.classifier_.predict(self._features(epochs))


def _align_covariances(covariances: np.ndarray, whiteners: list[np.ndarray]) -> np.ndarray:
    aligned = np.empty_like(covariances, dtype=np.float64)
    for band, whitener in enumerate(whiteners):
        aligned[:, band] = np.einsum(
            "ij,njk,lk->nil",
            whitener,
            covariances[:, band],
            whitener,
            optimize=True,
        )
    return (aligned + aligned.swapaxes(-1, -2)) * 0.5


def _riemannian_whiteners(covariances: np.ndarray) -> list[np.ndarray]:
    return [invsqrtm(mean_riemann(covariances[:, band])) for band in range(covariances.shape[1])]


class RiemannianTangentLogistic:
    """Per-band Riemannian alignment + tangent space + regularized logistic regression."""

    def __init__(self, c: float = 1.0, max_iter: int = 3000) -> None:
        self.c = c
        self.max_iter = max_iter

    def fit(
        self, covariances: np.ndarray, labels: np.ndarray, groups: np.ndarray | None = None
    ) -> "RiemannianTangentLogistic":
        x = np.asarray(covariances, dtype=np.float64)
        y = np.asarray(labels)
        if x.ndim != 4 or x.shape[-1] != x.shape[-2]:
            raise ValueError("covariances must have shape (N, bands, channels, channels)")
        group_array = _check_groups(groups, len(x))
        self.classes_ = np.unique(y)
        self.source_whiteners_ = _riemannian_whiteners(x)
        if group_array is None:
            aligned = _align_covariances(x, self.source_whiteners_)
        else:
            aligned = np.empty_like(x)
            for group in np.unique(group_array):
                mask = group_array == group
                aligned[mask] = _align_covariances(x[mask], _riemannian_whiteners(x[mask]))
        self.tangent_spaces_: list[TangentSpace] = []
        features: list[np.ndarray] = []
        for band in range(x.shape[1]):
            tangent = TangentSpace(metric="riemann")
            features.append(tangent.fit_transform(aligned[:, band]))
            self.tangent_spaces_.append(tangent)
        self.classifier_ = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=self.c, max_iter=self.max_iter, random_state=0),
        )
        self.classifier_.fit(np.hstack(features), y)
        self.target_whiteners_ = self.source_whiteners_
        return self

    def calibrate(self, unlabeled_covariances: np.ndarray) -> "RiemannianTangentLogistic":
        x = np.asarray(unlabeled_covariances, dtype=np.float64)
        if len(x) < 2:
            raise ValueError("at least two unlabeled covariances are required for calibration")
        self.target_whiteners_ = _riemannian_whiteners(x)
        return self

    def _features(self, covariances: np.ndarray) -> np.ndarray:
        aligned = _align_covariances(
            np.asarray(covariances, dtype=np.float64), self.target_whiteners_
        )
        return np.hstack(
            [
                tangent.transform(aligned[:, band])
                for band, tangent in enumerate(self.tangent_spaces_)
            ]
        )

    def predict_proba(self, covariances: np.ndarray) -> np.ndarray:
        return self.classifier_.predict_proba(self._features(covariances))

    def predict(self, covariances: np.ndarray) -> np.ndarray:
        return self.classifier_.predict(self._features(covariances))
