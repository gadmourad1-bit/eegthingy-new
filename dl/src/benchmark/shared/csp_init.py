"""Supervised CSP warm-start for GeoAdaptNet's per-band BiMap filters.

The residual branch projects each 15x15 band covariance onto a rank-``d`` subspace
through a semi-orthogonal BiMap.  Initialised randomly, that subspace carries no
class information the full-rank tangent anchor does not already see, so the
residual stays inert and its gate never opens (round-1 diagnostics).  Common
spatial patterns give, per band, the subspace whose log-variance separates the two
motor-imagery classes best -- exactly the nonlinear feature classical FB-CSP
exploits and a linear tangent classifier cannot represent.  Seeding each BiMap
with the CSP subspace fitted on the *training* recordings only therefore hands the
residual a genuinely complementary, discriminative starting point.

Everything here is fit-side: the caller passes covariances/labels from the inner
training recordings, never the selection or outer-test recordings.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _safe_eigh(matrix: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    matrix = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(matrix)
    floor = max(float(values.max()) * 1e-12, np.finfo(np.float64).tiny)
    return np.maximum(values, floor), vectors


def _matrix_power(matrix: NDArray[np.float64], power: float) -> NDArray[np.float64]:
    values, vectors = _safe_eigh(matrix)
    return (vectors * (values**power)) @ vectors.T


def _log_euclidean_reference(band_cov: NDArray[np.float64]) -> NDArray[np.float64]:
    """exp(mean_k log(C_k)) -- the log-Euclidean mean the SPD-BN converges to."""

    logs = np.empty_like(band_cov)
    for index, cov in enumerate(band_cov):
        values, vectors = _safe_eigh(cov)
        logs[index] = (vectors * np.log(values)) @ vectors.T
    mean_log = logs.mean(axis=0)
    values, vectors = np.linalg.eigh(0.5 * (mean_log + mean_log.T))
    return (vectors * np.exp(values)) @ vectors.T


def _class_mean(cov: NDArray[np.float64], labels: NDArray[np.int64], cls: int) -> NDArray[np.float64]:
    selected = cov[labels == cls]
    if len(selected) == 0:
        raise ValueError(f"no training covariances for class {cls}")
    return selected.mean(axis=0)


def csp_subspace(
    band_cov: NDArray[np.float64],
    labels: NDArray[np.int64],
    n_components: int,
    *,
    recenter: bool = True,
) -> NDArray[np.float32]:
    """Return a ``(channels, n_components)`` orthonormal basis of the CSP subspace.

    Solves the symmetric generalized problem ``C0 w = lambda (C0 + C1) w`` and keeps
    the ``n_components`` most class-extreme generalized eigenvectors (half from each
    tail), then orthonormalises them into a subspace basis suitable for a BiMap.
    When ``recenter`` is set, covariances are first mapped by the log-Euclidean
    fit-mean congruence so the subspace matches what the recentering layer feeds the
    BiMap at convergence.
    """

    cov = np.asarray(band_cov, dtype=np.float64)
    if recenter:
        reference = _log_euclidean_reference(cov)
        whitener = _matrix_power(reference, -0.5)
        cov = whitener @ cov @ whitener

    c0 = _class_mean(cov, labels, 0)
    c1 = _class_mean(cov, labels, 1)
    composite = c0 + c1
    # Whiten the composite covariance, then diagonalise C0 in that space.  This is
    # the standard, numerically stable route to CSP and avoids scipy's generalized
    # eigh so the module only needs NumPy.
    comp_values, comp_vectors = _safe_eigh(composite)
    whitening = comp_vectors * (1.0 / np.sqrt(comp_values))
    whitened_c0 = whitening.T @ c0 @ whitening
    ratios, rotation = np.linalg.eigh(0.5 * (whitened_c0 + whitened_c0.T))
    filters = whitening @ rotation  # columns: spatial filters, ascending class ratio

    order = np.argsort(ratios)
    half = n_components // 2
    pick = np.concatenate([order[:half], order[len(order) - (n_components - half):]])
    selected = filters[:, pick]
    basis, _ = np.linalg.qr(selected)
    return basis[:, :n_components].astype(np.float32)


def bandwise_csp_bases(
    covariances: NDArray[np.floating],
    labels: NDArray[np.int64],
    n_components: int,
    *,
    recenter: bool = True,
) -> list[NDArray[np.float32]]:
    """Per-band CSP subspace bases from training covariances ``(N, bands, C, C)``."""

    cov = np.asarray(covariances, dtype=np.float64)
    y = np.asarray(labels)
    if cov.ndim != 4:
        raise ValueError("covariances must have shape (N, bands, C, C)")
    task = y >= 0
    cov, y = cov[task], y[task]
    if set(np.unique(y).tolist()) - {0, 1}:
        raise ValueError("CSP warm-start requires binary 0/1 task labels")
    return [csp_subspace(cov[:, band], y, n_components, recenter=recenter) for band in range(cov.shape[1])]


__all__ = ["bandwise_csp_bases", "csp_subspace"]
