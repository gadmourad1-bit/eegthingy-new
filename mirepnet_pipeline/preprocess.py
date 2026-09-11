# MIRepNet expects trials shaped (n_epochs, 45, T) on its fixed channel
# template, at the ~250 Hz it was pretrained at, Euclidean-Aligned. Your
# OpenBCI montage almost certainly isn't 45 channels, so every trial gets
# remapped onto the template below before it ever reaches the model. This
# mirrors what utils/utils.py in the original repo does for its own
# downstream datasets (pad_missing_channels_diff + EA), just generalized to
# take an arbitrary channel list instead of a hardcoded per-dataset one.

import numpy as np
from scipy.spatial.distance import cdist
import mne

from mirepnet_pipeline.channel_list import (
    MIREPNET_TEMPLATE_CHANNELS,
    CHANNEL_POSITIONS,
    normalize_channel_name,
)

MIREPNET_SFREQ = 250.0


def build_template_projection(source_channels):
    """Precompute the (45, n_source) interpolation matrix that maps your
    montage onto MIRepNet's 45-channel template."""
    source_channels = [normalize_channel_name(c) for c in source_channels]
    unknown = [c for c in source_channels if c not in CHANNEL_POSITIONS]
    if unknown:
        raise ValueError(
            f"Channel(s) {unknown} have no known scalp position. Add them to "
            f"CHANNEL_POSITIONS in channel_list.py (approximate 10-20 (x, y) "
            f"in head-radius units is fine) or rename them to their nearest "
            f"standard 10-20 label."
        )
    existing_pos = np.array([CHANNEL_POSITIONS[c] for c in source_channels])
    target_pos = np.array([CHANNEL_POSITIONS[c] for c in MIREPNET_TEMPLATE_CHANNELS])

    W = np.zeros((len(MIREPNET_TEMPLATE_CHANNELS), len(source_channels)))
    for i, (target_ch, pos) in enumerate(zip(MIREPNET_TEMPLATE_CHANNELS, target_pos)):
        if target_ch in source_channels:
            W[i, source_channels.index(target_ch)] = 1.0
        else:
            dist = cdist([pos], existing_pos)[0]
            weights = 1.0 / (dist + 1e-6)
            W[i] = weights / weights.sum()
    return W  # (45, n_source)


def apply_template_projection(X, W):
    """X: (n_epochs, n_source_channels, T) -> (n_epochs, 45, T)."""
    return np.stack([W @ X[i] for i in range(X.shape[0])], axis=0)


def resample_epochs(X, sfreq_in, sfreq_out=MIREPNET_SFREQ):
    """X: (n_epochs, n_channels, T) at sfreq_in -> resampled to sfreq_out."""
    if abs(sfreq_in - sfreq_out) < 1e-6:
        return X
    return mne.filter.resample(X, up=sfreq_out, down=sfreq_in, axis=-1, verbose=False)


def _safe_whitener(cov):
    """Compute R^{-1/2} using eigen-decomposition with eigenvalue clipping."""
    w, V = np.linalg.eigh(cov)
    w = np.clip(w, 1e-12, None)
    return (V * (w ** -0.5)) @ V.T


def euclidean_alignment(X, ref_whitener=None):
    """Whiten each trial by the inverse sqrt of the *mean* trial covariance."""
    if ref_whitener is None:
        cov = np.mean([np.cov(x) for x in X], axis=0)
        cov += np.eye(cov.shape[0]) * 1e-10
        ref_whitener = _safe_whitener(cov)
    return np.stack([ref_whitener @ x for x in X], axis=0), ref_whitener


def fit_ea_reference(X):
    """Compute a whitener from a calibration/training batch."""
    cov = np.mean([np.cov(x) for x in X], axis=0)
    cov += np.eye(cov.shape[0]) * 1e-10
    return _safe_whitener(cov)


def prepare_for_mirepnet(X, source_channels, sfreq_in, W=None, ea_whitener=None,
                         fit_ea=False, project_to_template=False, normalize=True):
    """Full pipeline: raw broadband epochs -> MIRepNet-ready tensor."""
    if project_to_template and W is None:
        W = build_template_projection(source_channels)

    X_rs = resample_epochs(X, sfreq_in, MIREPNET_SFREQ)

    # Add standardization to handle extreme values
    if normalize:
        # Compute global mean and std across all data
        X_mean = X_rs.mean()
        X_std = X_rs.std() + 1e-8
        X_rs = (X_rs - X_mean) / X_std
        print(f"Data after standardization: mean={X_rs.mean():.4f}, std={X_rs.std():.4f}, "
              f"min={X_rs.min():.4f}, max={X_rs.max():.4f}")

    if fit_ea or ea_whitener is None:
        X_ea, ea_whitener = euclidean_alignment(X_rs, ref_whitener=None if fit_ea else ea_whitener)
    else:
        X_ea, ea_whitener = euclidean_alignment(X_rs, ref_whitener=ea_whitener)

    if project_to_template:
        X_out = apply_template_projection(X_ea, W)
    else:
        X_out = X_ea

    if not np.isfinite(X_out).all():
        raise ValueError(
            "Non-finite values detected in preprocessed data. "
            "Check the input EEG for flat channels or extreme artefacts."
        )

    return X_out.astype(np.float32), W, ea_whitener