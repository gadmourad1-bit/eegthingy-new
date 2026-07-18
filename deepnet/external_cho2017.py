"""External validation dataset: Cho et al. 2017 (GigaDB 100295) via MOABB.

52 subjects, binary left/right-hand motor imagery, 64-channel EEG at 512 Hz -- a
completely independent cohort, montage, and amplifier from the local recordings.
To test whether the local data biases the architecture comparison, each subject is
mapped onto the *same* 15-channel sensorimotor layout and 125 Hz sampling as the
local pipeline (10-20 names T5/T6/T3/T4 map to the 10-10 names P7/P8/T7/T8), then
passed through the identical filter-bank + covariance construction.  The exact same
network architectures are then trained and scored, so any ranking difference
reflects the data, not the pipeline.

Cho2017 is a single session per subject, so the protocol is within-subject
stratified cross-validation rather than the local chronological split.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from .config import BANDS, CHANNELS, SFREQ
from .data import make_spd_covariances

# Our 10-20 channel names -> Cho2017's 10-10 names; positions stay in CHANNELS order.
_NAME_MAP = {"T5": "P7", "T6": "P8", "T3": "T7", "T4": "T8"}
CHO_PICKS = [_NAME_MAP.get(name, name) for name in CHANNELS]

_CACHE = Path(__file__).resolve().parent / "cache" / "cho2017"
_LABELS = {"left_hand": 0, "right_hand": 1}


def _cache_path(subject: int, tmin: float, tmax: float) -> Path:
    key = f"cho_s{subject}_{tmin}_{tmax}_{'-'.join(CHO_PICKS)}_{SFREQ}_{BANDS}"
    digest = hashlib.sha256(key.encode()).hexdigest()[:16]
    return _CACHE / f"cho_s{subject:02d}_{digest}.npz"


def load_cho_subject(
    subject: int, *, tmin: float = 0.0, tmax: float = 2.0, use_cache: bool = True
) -> dict[str, np.ndarray]:
    """Return matched covariances, broadband epochs, and labels for one subject."""

    target = _cache_path(subject, tmin, tmax)
    if use_cache and target.is_file():
        with np.load(target) as cached:
            return {k: cached[k] for k in ("covariances", "broadband", "labels")}

    import mne
    from moabb.datasets import Cho2017

    mne.set_log_level("ERROR")
    sessions = Cho2017().get_data(subjects=[subject])[subject]
    raw = next(iter(next(iter(sessions.values())).values()))
    missing = [ch for ch in CHO_PICKS if ch not in raw.ch_names]
    if missing:
        raise ValueError(f"Cho2017 subject {subject} missing channels {missing}")
    raw.pick(CHO_PICKS)  # reorders to our CHANNELS layout
    if not np.isclose(raw.info["sfreq"], SFREQ):
        raw.resample(SFREQ)

    events, event_id = mne.events_from_annotations(raw, event_id=_LABELS)
    # ``events_from_annotations`` may renumber codes; map back through event_id.
    code_to_label = {code: _LABELS[name] for name, code in event_id.items()}

    def epoch(low: float, high: float) -> tuple[np.ndarray, np.ndarray]:
        filtered = raw.copy().filter(
            low, high, method="fir", phase="minimum", fir_design="firwin", verbose=False
        )
        ep = mne.Epochs(
            filtered, events, tmin=tmin, tmax=tmax, baseline=None, preload=True, verbose=False
        )
        labels = np.asarray([code_to_label[c] for c in ep.events[:, -1]], dtype=np.int64)
        return ep.get_data(copy=True).astype(np.float32), labels

    broadband, labels = epoch(8.0, 30.0)
    bands = [epoch(low, high)[0] for low, high in BANDS]
    filter_bank = np.stack(bands, axis=1)  # (N, 4, 15, T)
    covariances = make_spd_covariances(filter_bank, dtype="float32")

    result = {"covariances": covariances, "broadband": broadband, "labels": labels}
    if use_cache:
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, **result)
    return result


__all__ = ["CHO_PICKS", "load_cho_subject"]
