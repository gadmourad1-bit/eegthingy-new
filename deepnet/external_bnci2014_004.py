"""Leakage-controlled BNCI2014-004 confirmation data loader.

This module implements a deliberately separate development/confirmation protocol
for BCI Competition IV dataset 2b (MOABB ``BNCI2014_004``):

* subjects 1--4 are development subjects;
* subjects 5--9 are untouched confirmation subjects and require explicit loader
  authorization;
* sessions ``0train`` and ``1train`` are fit data (240--320 trials total);
* session ``2train`` is selection data (120--160 trials);
* sessions ``3test`` and ``4test`` are held-out test data (240--320 trials
  total; the released competition files contain either 120 or 160 balanced
  trials per feedback session, depending on participant/session).

Only left- and right-hand trials are retained.  The three provided bipolar EEG
signals (C3, Cz, C4) are preserved as supplied by the dataset, resampled from the
nominal 250 Hz source rate to 125 Hz, and epoched from 0.5 through 2.5 seconds
after cue onset (both endpoints included, 251 samples).  The returned tensors are
8--30 Hz broadband epochs plus four fixed-band, shrinkage-regularized SPD
covariances matching the local pipeline's bands.

The loader does not import MOABB or MNE until an authorized, uncached subject is
requested.  Cache files contain derived arrays plus protocol/source identity
metadata and live under ``deepnet/cache/bnci2014_004``.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import numpy as np

from .config import BANDS
from .data import make_spd_covariances


BNCI_CHANNELS: Final[tuple[str, ...]] = ("C3", "Cz", "C4")
REFLECTION_CHANNEL_INDICES: Final[tuple[int, ...]] = (2, 1, 0)
DEVELOPMENT_SUBJECTS: Final[tuple[int, ...]] = (1, 2, 3, 4)
CONFIRMATION_SUBJECTS: Final[tuple[int, ...]] = (5, 6, 7, 8, 9)
ALL_SUBJECTS: Final[tuple[int, ...]] = DEVELOPMENT_SUBJECTS + CONFIRMATION_SUBJECTS

SOURCE_SFREQ: Final[float] = 250.0
BNCI_SFREQ: Final[float] = 125.0
TMIN: Final[float] = 0.5
TMAX: Final[float] = 2.5
EPOCH_SAMPLES: Final[int] = int(round((TMAX - TMIN) * BNCI_SFREQ)) + 1

FIT_SESSIONS: Final[tuple[str, ...]] = ("0train", "1train")
VALIDATION_SESSION: Final[str] = "2train"
TEST_SESSIONS: Final[tuple[str, ...]] = ("3test", "4test")
ALL_SESSIONS: Final[tuple[str, ...]] = (
    FIT_SESSIONS + (VALIDATION_SESSION,) + TEST_SESSIONS
)
PROTOCOL_ID: Final[str] = (
    "bnci2014-004_s1-4-dev_s5-9-confirm_"
    "ses0-1-fit_ses2-select_ses3-4-test"
)

_CACHE_SCHEMA_VERSION = 1
_CACHE = Path(__file__).resolve().parent / "cache" / "bnci2014_004"
_EVENT_CODES = {"left_hand": 1, "right_hand": 2}
_CODE_TO_LABEL = {1: 0, 2: 1}
_ARRAY_NAMES = ("covariances", "broadband", "labels", "session_ids")
_SPLIT_NAMES = ("train", "validation", "test")
_CACHE_METADATA_NAMES = (
    "metadata_cache_schema",
    "metadata_preprocessing_source_sha256",
    "metadata_protocol_id",
    "metadata_protocol_json",
    "metadata_subject",
)
_PREPROCESSING_SOURCE_FILES = ("external_bnci2014_004.py", "data.py")
_EXPECTED_SESSIONS: Final[dict[str, tuple[str, ...]]] = {
    "train": FIT_SESSIONS,
    "validation": (VALIDATION_SESSION,),
    "test": TEST_SESSIONS,
}
_TRIALS_PER_SESSION: Final[dict[str, int]] = {
    "0train": 120,
    "1train": 120,
    "2train": 160,
    "3test": 160,
    "4test": 160,
}
# The nominal screening/feedback designs used 120/160 trials, but some released
# files contain 140 balanced labeled trials (for example development subject 4
# session ``1train``), and some evaluation sessions contain 120 (for example
# subject 2 ``3test``).  These are official-file properties, not preprocessing
# rejection.  Pin the only three observed design-sized counts rather than
# weakening validation to an arbitrary length.
_ALLOWED_TRIALS_PER_SESSION: Final[dict[str, tuple[int, ...]]] = {
    session: (120, 140, 160) for session in ALL_SESSIONS
}
_EXPECTED_COUNTS: Final[dict[str, int]] = {
    split: sum(_TRIALS_PER_SESSION[session] for session in sessions)
    for split, sessions in _EXPECTED_SESSIONS.items()
}
_ALLOWED_SPLIT_COUNTS: Final[dict[str, tuple[int, ...]]] = {
    "train": (240, 260, 280, 300, 320),
    "validation": (120, 140, 160),
    "test": (240, 260, 280, 300, 320),
}
_EPOCH_SAMPLES = EPOCH_SAMPLES


def subject_partition(subject: int) -> str:
    """Return ``development`` or ``confirmation`` for an allowed subject."""

    subject = int(subject)
    if subject in DEVELOPMENT_SUBJECTS:
        return "development"
    if subject in CONFIRMATION_SUBJECTS:
        return "confirmation"
    raise ValueError(f"BNCI2014-004 subject must be in {ALL_SUBJECTS}, got {subject}")


def protocol_metadata(subject: int) -> dict[str, Any]:
    """Return a JSON-safe description of the locked per-subject protocol."""

    subject = int(subject)
    return {
        "protocol_id": PROTOCOL_ID,
        "dataset": "BNCI2014-004",
        "moabb_dataset": "BNCI2014_004",
        "subject": subject,
        "cohort_partition": subject_partition(subject),
        "development_subjects": list(DEVELOPMENT_SUBJECTS),
        "confirmation_subjects": list(CONFIRMATION_SUBJECTS),
        "labels": {"left_hand": 0, "right_hand": 1},
        "channels": list(BNCI_CHANNELS),
        "reflection_channel_indices": list(REFLECTION_CHANNEL_INDICES),
        "reference": "provided_bipolar",
        "source_sfreq": SOURCE_SFREQ,
        "sfreq": BNCI_SFREQ,
        "tmin": TMIN,
        "tmax": TMAX,
        "epoch_samples_inclusive": EPOCH_SAMPLES,
        "broadband_hz": [8.0, 30.0],
        "covariance_bands_hz": [list(band) for band in BANDS],
        "covariance": {
            "estimator": "sample_covariance",
            "shrinkage": "fixed_scaled_identity",
            "shrinkage_weight": 1e-3,
        },
        "nominal_session_trial_counts": dict(_TRIALS_PER_SESSION),
        "allowed_session_trial_counts": {
            session: list(counts)
            for session, counts in _ALLOWED_TRIALS_PER_SESSION.items()
        },
        "fit": {
            "sessions": list(FIT_SESSIONS),
            "allowed_trials": list(_ALLOWED_SPLIT_COUNTS["train"]),
        },
        "select": {
            "sessions": [VALIDATION_SESSION],
            "allowed_trials": list(_ALLOWED_SPLIT_COUNTS["validation"]),
        },
        "test": {
            "sessions": list(TEST_SESSIONS),
            "allowed_trials": list(_ALLOWED_SPLIT_COUNTS["test"]),
        },
    }


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _preprocessing_source_digest() -> str:
    """Hash the in-repository code that creates every cached numeric array."""

    module_dir = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in _PREPROCESSING_SOURCE_FILES:
        payload = (module_dir / name).read_bytes()
        encoded_name = name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(4, "big"))
        digest.update(encoded_name)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _cache_identity(subject: int) -> dict[str, Any]:
    subject = int(subject)
    subject_partition(subject)
    protocol = protocol_metadata(subject)
    return {
        "cache_schema": _CACHE_SCHEMA_VERSION,
        "preprocessing_source_sha256": _preprocessing_source_digest(),
        "protocol_id": PROTOCOL_ID,
        "protocol_json": _canonical_json(protocol),
        "subject": subject,
    }


def _cache_path(subject: int) -> Path:
    settings = _cache_identity(subject)
    digest = hashlib.sha256(_canonical_json(settings).encode("utf-8")).hexdigest()[:16]
    return _CACHE / f"bnci2014_004_s{int(subject):02d}_{digest}.npz"


def _validate_splits(
    splits: Mapping[str, Mapping[str, np.ndarray]],
    *,
    subject: int,
    origin: str,
) -> None:
    """Reject incomplete, malformed, or protocol-inconsistent split payloads."""

    subject = int(subject)
    subject_partition(subject)
    if set(splits) != set(_SPLIT_NAMES):
        raise ValueError(
            f"{origin}: expected split keys {sorted(_SPLIT_NAMES)}, "
            f"found {sorted(splits)} for BNCI2014-004 subject {subject}"
        )

    n_channels = len(BNCI_CHANNELS)
    n_bands = len(BANDS)
    for split in _SPLIT_NAMES:
        arrays = splits[split]
        if not isinstance(arrays, Mapping):
            raise ValueError(
                f"{origin}: BNCI2014-004 subject {subject} {split} is not a mapping"
            )
        if set(arrays) != set(_ARRAY_NAMES):
            raise ValueError(
                f"{origin}: BNCI2014-004 subject {subject} {split} expected array "
                f"keys {sorted(_ARRAY_NAMES)}, found {sorted(arrays)}"
            )
        if any(not isinstance(arrays[name], np.ndarray) for name in _ARRAY_NAMES):
            raise ValueError(
                f"{origin}: BNCI2014-004 subject {subject} {split} arrays must be "
                "numpy arrays"
            )

        covariances = arrays["covariances"]
        broadband = arrays["broadband"]
        labels = arrays["labels"]
        session_ids = arrays["session_ids"]

        count = int(len(labels))
        if count not in _ALLOWED_SPLIT_COUNTS[split]:
            raise ValueError(
                f"{origin}: BNCI2014-004 subject {subject} {split} has {count} "
                f"trials; expected one of {list(_ALLOWED_SPLIT_COUNTS[split])}"
            )

        expected_covariance_shape = (count, n_bands, n_channels, n_channels)
        expected_broadband_shape = (count, n_channels, EPOCH_SAMPLES)
        if covariances.shape != expected_covariance_shape:
            raise ValueError(
                f"{origin}: BNCI2014-004 subject {subject} {split} covariances "
                f"have shape {covariances.shape}; expected {expected_covariance_shape}"
            )
        if broadband.shape != expected_broadband_shape:
            raise ValueError(
                f"{origin}: BNCI2014-004 subject {subject} {split} broadband has "
                f"shape {broadband.shape}; expected {expected_broadband_shape}"
            )
        if labels.shape != (count,) or session_ids.shape != (count,):
            raise ValueError(
                f"{origin}: BNCI2014-004 subject {subject} {split} labels/session_ids "
                f"must each have shape ({count},); found {labels.shape} and "
                f"{session_ids.shape}"
            )

        if covariances.dtype != np.dtype(np.float32):
            raise ValueError(
                f"{origin}: BNCI2014-004 subject {subject} {split} covariances "
                "must be float32"
            )
        if broadband.dtype != np.dtype(np.float32):
            raise ValueError(
                f"{origin}: BNCI2014-004 subject {subject} {split} broadband must "
                "be float32"
            )
        if labels.dtype != np.dtype(np.int64):
            raise ValueError(
                f"{origin}: BNCI2014-004 subject {subject} {split} labels must be int64"
            )
        if session_ids.dtype.kind != "U":
            raise ValueError(
                f"{origin}: BNCI2014-004 subject {subject} {split} session_ids "
                "must be unicode strings"
            )
        if not np.isfinite(covariances).all() or not np.isfinite(broadband).all():
            raise ValueError(
                f"{origin}: BNCI2014-004 subject {subject} {split} contains "
                "non-finite features"
            )
        if not np.allclose(
            covariances,
            np.swapaxes(covariances, -1, -2),
            rtol=1e-5,
            atol=1e-12,
        ):
            raise ValueError(
                f"{origin}: BNCI2014-004 subject {subject} {split} covariances "
                "are not symmetric"
            )
        if np.any(np.linalg.eigvalsh(covariances) <= 0.0):
            raise ValueError(
                f"{origin}: BNCI2014-004 subject {subject} {split} covariances "
                "are not SPD"
            )
        if set(np.unique(labels).tolist()) != {0, 1}:
            raise ValueError(
                f"{origin}: BNCI2014-004 subject {subject} {split} labels must "
                "contain only both binary classes 0 and 1"
            )

        actual_sessions = set(np.unique(session_ids).tolist())
        expected_sessions = set(_EXPECTED_SESSIONS[split])
        if actual_sessions != expected_sessions:
            raise ValueError(
                f"{origin}: BNCI2014-004 subject {subject} {split} session IDs are "
                f"{sorted(actual_sessions)}; expected {sorted(expected_sessions)}"
            )
        for session_id in _EXPECTED_SESSIONS[split]:
            session_labels = labels[session_ids == session_id]
            allowed_trials = _ALLOWED_TRIALS_PER_SESSION[session_id]
            observed_trials = len(session_labels)
            if observed_trials not in allowed_trials:
                raise ValueError(
                    f"{origin}: BNCI2014-004 subject {subject} {split} session "
                    f"{session_id} has {observed_trials} trials; expected one of "
                    f"{list(allowed_trials)}"
                )
            class_counts = np.bincount(session_labels, minlength=2)
            expected_class_counts = np.asarray(
                [observed_trials // 2, observed_trials // 2]
            )
            if not np.array_equal(class_counts, expected_class_counts):
                raise ValueError(
                    f"{origin}: BNCI2014-004 subject {subject} {split} session "
                    f"{session_id} class counts are {class_counts.tolist()}; "
                    f"expected {expected_class_counts.tolist()}"
                )


def _cache_metadata_arrays(subject: int) -> dict[str, np.ndarray]:
    identity = _cache_identity(subject)
    return {
        "metadata_cache_schema": np.asarray(identity["cache_schema"], dtype=np.int64),
        "metadata_preprocessing_source_sha256": np.asarray(
            identity["preprocessing_source_sha256"], dtype=np.str_
        ),
        "metadata_protocol_id": np.asarray(identity["protocol_id"], dtype=np.str_),
        "metadata_protocol_json": np.asarray(identity["protocol_json"], dtype=np.str_),
        "metadata_subject": np.asarray(identity["subject"], dtype=np.int64),
    }


def _validate_cache_metadata(cached: Any, *, subject: int, path: Path) -> None:
    expected = _cache_metadata_arrays(subject)
    for name, expected_array in expected.items():
        actual = np.asarray(cached[name])
        if actual.shape != ():
            raise ValueError(
                f"BNCI2014-004 cache {path} metadata field {name} is not scalar"
            )
        if (
            actual.dtype.kind != expected_array.dtype.kind
            or actual.item() != expected_array.item()
        ):
            raise ValueError(
                f"BNCI2014-004 cache {path} has mismatched {name}; it is not valid "
                f"for subject {subject}, protocol {PROTOCOL_ID}, schema "
                f"{_CACHE_SCHEMA_VERSION}, and the current preprocessing source"
            )


def _unpack_cache(path: Path, *, subject: int) -> dict[str, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as cached:
        expected_keys = set(_CACHE_METADATA_NAMES) | {
            f"{split}_{name}" for split in _SPLIT_NAMES for name in _ARRAY_NAMES
        }
        if set(cached.files) != expected_keys:
            raise ValueError(
                f"BNCI2014-004 cache {path} has unexpected keys: expected "
                f"{sorted(expected_keys)}, found {sorted(cached.files)}"
            )
        _validate_cache_metadata(cached, subject=subject, path=path)
        splits = {
            split: {name: cached[f"{split}_{name}"] for name in _ARRAY_NAMES}
            for split in _SPLIT_NAMES
        }
    _validate_splits(
        splits, subject=subject, origin=f"BNCI2014-004 cache {path}"
    )
    return splits


def _write_cache(
    path: Path, splits: dict[str, dict[str, np.ndarray]], *, subject: int
) -> None:
    _validate_splits(
        splits, subject=subject, origin="generated BNCI2014-004 payload"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        f"{split}_{name}": np.asarray(splits[split][name])
        for split in _SPLIT_NAMES
        for name in _ARRAY_NAMES
    }
    arrays.update(_cache_metadata_arrays(subject))
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.stem}-", suffix=".npz", dir=path.parent, delete=False
        ) as temporary:
            temporary_name = temporary.name
        np.savez_compressed(temporary_name, **arrays)
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _prepare_raw(raw: Any) -> Any:
    """Select/reorder the provided bipolar EEG signals and resample to 125 Hz."""

    prepared = raw.copy()
    missing = [channel for channel in BNCI_CHANNELS if channel not in prepared.ch_names]
    if missing:
        raise ValueError(
            f"BNCI2014-004 recording is missing required channels: {missing}"
        )
    prepared.pick(list(BNCI_CHANNELS))
    if tuple(prepared.ch_names) != BNCI_CHANNELS:
        raise RuntimeError(
            "BNCI2014-004 channel selection did not preserve the locked C3/Cz/C4 order"
        )

    source_sfreq = float(prepared.info["sfreq"])
    if not np.isclose(source_sfreq, SOURCE_SFREQ, rtol=0.0, atol=1e-8):
        raise ValueError(
            f"BNCI2014-004 source sampling frequency is {source_sfreq}; "
            f"expected {SOURCE_SFREQ} Hz"
        )
    prepared.resample(BNCI_SFREQ, npad="auto", verbose=False)
    if not np.isclose(
        float(prepared.info["sfreq"]), BNCI_SFREQ, rtol=0.0, atol=1e-8
    ):
        raise RuntimeError("BNCI2014-004 resampling did not reach 125 Hz")
    return prepared


def _epoch_filtered(raw: Any, events: np.ndarray, low: float, high: float) -> Any:
    import mne

    filtered = raw.copy().filter(
        l_freq=low,
        h_freq=high,
        method="fir",
        phase="minimum",
        fir_design="firwin",
        l_trans_bandwidth=2.0,
        h_trans_bandwidth=2.5,
        verbose=False,
    )
    return mne.Epochs(
        filtered,
        events,
        event_id=None,
        tmin=TMIN,
        tmax=TMAX,
        baseline=None,
        preload=True,
        proj=False,
        reject_by_annotation=True,
        verbose=False,
    )


def _process_session(raw: Any, session_id: str) -> dict[str, np.ndarray]:
    import mne

    raw = _prepare_raw(raw)
    events, event_id = mne.events_from_annotations(
        raw, event_id=_EVENT_CODES, verbose=False
    )
    if event_id != _EVENT_CODES:
        raise RuntimeError(f"unexpected BNCI2014-004 event mapping: {event_id}")
    if len(events) == 0:
        raise ValueError(f"BNCI2014-004 session {session_id} has no left/right trials")

    broadband_epochs = _epoch_filtered(raw, events, 8.0, 30.0)
    retained_events = broadband_epochs.events.copy()
    broadband = broadband_epochs.get_data(copy=True).astype(np.float32, copy=False)
    if broadband.shape[1:] != (len(BNCI_CHANNELS), EPOCH_SAMPLES):
        raise ValueError(
            f"BNCI2014-004 session {session_id} broadband epochs have shape "
            f"{broadband.shape[1:]}; expected "
            f"({len(BNCI_CHANNELS)}, {EPOCH_SAMPLES})"
        )
    labels = np.asarray(
        [_CODE_TO_LABEL[int(code)] for code in retained_events[:, -1]], dtype=np.int64
    )

    band_data: list[np.ndarray] = []
    for low, high in BANDS:
        epochs = _epoch_filtered(raw, retained_events, low, high)
        if not np.array_equal(epochs.events, retained_events):
            raise RuntimeError(
                "BNCI2014-004 filter-band epoch alignment changed unexpectedly"
            )
        values = epochs.get_data(copy=True).astype(np.float32, copy=False)
        if values.shape != broadband.shape:
            raise RuntimeError(
                "BNCI2014-004 filter-band and broadband epoch shapes differ"
            )
        band_data.append(values)
    filter_bank = np.stack(band_data, axis=1)
    covariances = make_spd_covariances(filter_bank, dtype=np.float32)

    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError(
            f"BNCI2014-004 session {session_id} does not contain both task classes"
        )
    return {
        "covariances": covariances,
        "broadband": broadband,
        "labels": labels,
        "session_ids": np.asarray([session_id] * len(labels), dtype=np.str_),
    }


def _concatenate(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not parts:
        raise ValueError("cannot concatenate an empty BNCI2014-004 partition")
    return {
        name: np.concatenate([part[name] for part in parts]) for name in _ARRAY_NAMES
    }


def _single_session_run(session: Any, *, session_id: str) -> Any:
    if not isinstance(session, Mapping):
        raise ValueError(f"BNCI2014-004 session {session_id} is not a run mapping")
    if set(session) != {"0"}:
        raise ValueError(
            f"BNCI2014-004 session {session_id} expected run ['0'], "
            f"found {sorted(str(key) for key in session)}"
        )
    return session["0"]


def load_bnci2014_004_subject(
    subject: int,
    *,
    use_cache: bool = True,
    allow_confirmation: bool = False,
) -> dict[str, dict[str, np.ndarray]]:
    """Load one subject as locked ``train``, ``validation``, and ``test`` splits.

    Subjects 5--9 are denied before cache inspection or MOABB import unless the
    caller explicitly passes ``allow_confirmation=True``.  That keyword is only
    a loader-level tripwire; the benchmark is still responsible for its permanent
    one-shot confirmation receipt.
    """

    subject = int(subject)
    partition = subject_partition(subject)
    if partition == "confirmation" and not allow_confirmation:
        raise PermissionError(
            f"BNCI2014-004 subject {subject} is in the frozen confirmation cohort; "
            "pass allow_confirmation=True only from the locked one-shot benchmark"
        )

    target = _cache_path(subject)
    if use_cache and target.is_file():
        return _unpack_cache(target, subject=subject)

    import mne
    from moabb.datasets import BNCI2014_004

    mne.set_log_level("ERROR")
    sessions = BNCI2014_004().get_data(subjects=[subject])[subject]
    if set(sessions) != set(ALL_SESSIONS):
        raise ValueError(
            f"expected BNCI2014-004 sessions {sorted(ALL_SESSIONS)}, "
            f"found {sorted(sessions)}"
        )

    processed = {
        session_id: _process_session(
            _single_session_run(sessions[session_id], session_id=session_id),
            session_id,
        )
        for session_id in ALL_SESSIONS
    }
    splits = {
        "train": _concatenate([processed[name] for name in FIT_SESSIONS]),
        "validation": _concatenate([processed[VALIDATION_SESSION]]),
        "test": _concatenate([processed[name] for name in TEST_SESSIONS]),
    }
    _validate_splits(
        splits, subject=subject, origin="generated BNCI2014-004 payload"
    )

    if use_cache:
        _write_cache(target, splits, subject=subject)
    return splits


__all__ = [
    "ALL_SESSIONS",
    "ALL_SUBJECTS",
    "BNCI_CHANNELS",
    "BNCI_SFREQ",
    "CONFIRMATION_SUBJECTS",
    "DEVELOPMENT_SUBJECTS",
    "EPOCH_SAMPLES",
    "FIT_SESSIONS",
    "PROTOCOL_ID",
    "REFLECTION_CHANNEL_INDICES",
    "SOURCE_SFREQ",
    "TEST_SESSIONS",
    "TMAX",
    "TMIN",
    "VALIDATION_SESSION",
    "load_bnci2014_004_subject",
    "protocol_metadata",
    "subject_partition",
]
