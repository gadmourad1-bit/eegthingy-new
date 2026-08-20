"""Leakage-controlled BNCI2014-001 confirmation data loader.

This module implements a deliberately separate development/confirmation protocol
for BCI Competition IV dataset 2a (MOABB ``BNCI2014_001``):

* subjects 1--4 are development subjects;
* subjects 5--9 are untouched confirmation subjects;
* the official ``T`` session is the only source of fit/selection data;
* runs 0--4 of ``T`` are training data and run 5 is inner validation data;
* the official ``E`` session is test data and is never used for fitting or
  architecture/checkpoint selection.

Only left- and right-hand trials are retained.  Continuous recordings are first
restricted to a symmetric 15-channel sensorimotor montage and resampled to 125 Hz.
The imagery window is 0.5--2.5 seconds after cue onset.  The returned tensors are
8--30 Hz broadband epochs plus four fixed-band SPD covariances matching the local
pipeline's bands.

The loader does not import MOABB or MNE until uncached data are requested.  Cache
files contain derived arrays plus protocol/source identity metadata and live under
``src/benchmark/research/cache/bnci2014_001``.
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

from .config import BANDS, SFREQ
from .data import make_spd_covariances

BNCI_CHANNELS: Final[tuple[str, ...]] = (
    "Cz",
    "C5",
    "C6",
    "C3",
    "C4",
    "C1",
    "C2",
    "FC3",
    "FC4",
    "FC1",
    "FC2",
    "CP3",
    "CP4",
    "CP1",
    "CP2",
)
DEVELOPMENT_SUBJECTS: Final[tuple[int, ...]] = (1, 2, 3, 4)
CONFIRMATION_SUBJECTS: Final[tuple[int, ...]] = (5, 6, 7, 8, 9)
ALL_SUBJECTS: Final[tuple[int, ...]] = DEVELOPMENT_SUBJECTS + CONFIRMATION_SUBJECTS
TMIN: Final[float] = 0.5
TMAX: Final[float] = 2.5
TRAIN_SESSION: Final[str] = "0train"
TEST_SESSION: Final[str] = "1test"
VALIDATION_RUN: Final[str] = "5"
PROTOCOL_ID: Final[str] = (
    "bnci2014-001_s1-4-dev_s5-9-confirm_T-runs0-4-fit_T-run5-select_E-test"
)

_CACHE_SCHEMA_VERSION = 2
_CACHE = Path(__file__).resolve().parent / "cache" / "bnci2014_001"
_EVENT_CODES = {"left_hand": 1, "right_hand": 2}
_CODE_TO_LABEL = {1: 0, 2: 1}
_ARRAY_NAMES = ("covariances", "broadband", "labels", "run_ids")
_SPLIT_NAMES = ("train", "validation", "test")
_CACHE_METADATA_NAMES = (
    "metadata_cache_schema",
    "metadata_preprocessing_source_sha256",
    "metadata_protocol_id",
    "metadata_protocol_json",
    "metadata_subject",
)
_PREPROCESSING_SOURCE_FILES = ("external_bnci2014.py", "data.py")
_EXPECTED_RUNS: Final[dict[str, tuple[str, ...]]] = {
    "train": ("0", "1", "2", "3", "4"),
    "validation": (VALIDATION_RUN,),
    "test": ("0", "1", "2", "3", "4", "5"),
}
_TRIALS_PER_RUN: Final[int] = 24
_TRIALS_PER_CLASS_PER_RUN: Final[int] = 12
_EXPECTED_COUNTS: Final[dict[str, int]] = {
    split: len(runs) * _TRIALS_PER_RUN for split, runs in _EXPECTED_RUNS.items()
}
_EPOCH_SAMPLES: Final[int] = int(round((TMAX - TMIN) * SFREQ)) + 1


def subject_partition(subject: int) -> str:
    """Return ``development`` or ``confirmation`` for an allowed subject."""

    subject = int(subject)
    if subject in DEVELOPMENT_SUBJECTS:
        return "development"
    if subject in CONFIRMATION_SUBJECTS:
        return "confirmation"
    raise ValueError(f"BNCI2014-001 subject must be in {ALL_SUBJECTS}, got {subject}")


def protocol_metadata(subject: int) -> dict[str, Any]:
    """Return a JSON-safe description of the locked per-subject protocol."""

    subject = int(subject)
    return {
        "protocol_id": PROTOCOL_ID,
        "dataset": "BNCI2014-001",
        "subject": subject,
        "cohort_partition": subject_partition(subject),
        "development_subjects": list(DEVELOPMENT_SUBJECTS),
        "confirmation_subjects": list(CONFIRMATION_SUBJECTS),
        "labels": {"left_hand": 0, "right_hand": 1},
        "channels": list(BNCI_CHANNELS),
        "sfreq": SFREQ,
        "tmin": TMIN,
        "tmax": TMAX,
        "broadband_hz": [8.0, 30.0],
        "covariance_bands_hz": [list(band) for band in BANDS],
        "fit": {"session": TRAIN_SESSION, "runs": ["0", "1", "2", "3", "4"]},
        "select": {"session": TRAIN_SESSION, "runs": [VALIDATION_RUN]},
        "test": {"session": TEST_SESSION, "runs": ["0", "1", "2", "3", "4", "5"]},
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
    return _CACHE / f"bnci2014_001_s{subject:02d}_{digest}.npz"


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
            f"found {sorted(splits)} for BNCI subject {subject}"
        )

    n_channels = len(BNCI_CHANNELS)
    n_bands = len(BANDS)
    for split in _SPLIT_NAMES:
        arrays = splits[split]
        if not isinstance(arrays, Mapping):
            raise ValueError(
                f"{origin}: BNCI subject {subject} {split} is not a mapping"
            )
        if set(arrays) != set(_ARRAY_NAMES):
            raise ValueError(
                f"{origin}: BNCI subject {subject} {split} expected array keys "
                f"{sorted(_ARRAY_NAMES)}, found {sorted(arrays)}"
            )
        if any(not isinstance(arrays[name], np.ndarray) for name in _ARRAY_NAMES):
            raise ValueError(
                f"{origin}: BNCI subject {subject} {split} arrays must be numpy arrays"
            )

        count = _EXPECTED_COUNTS[split]
        covariances = arrays["covariances"]
        broadband = arrays["broadband"]
        labels = arrays["labels"]
        run_ids = arrays["run_ids"]

        expected_covariance_shape = (count, n_bands, n_channels, n_channels)
        expected_broadband_shape = (count, n_channels, _EPOCH_SAMPLES)
        if covariances.shape != expected_covariance_shape:
            raise ValueError(
                f"{origin}: BNCI subject {subject} {split} covariances have shape "
                f"{covariances.shape}; expected {expected_covariance_shape}"
            )
        if broadband.shape != expected_broadband_shape:
            raise ValueError(
                f"{origin}: BNCI subject {subject} {split} broadband has shape "
                f"{broadband.shape}; expected {expected_broadband_shape}"
            )
        if labels.shape != (count,) or run_ids.shape != (count,):
            raise ValueError(
                f"{origin}: BNCI subject {subject} {split} labels/run_ids must each "
                f"have shape ({count},); found {labels.shape} and {run_ids.shape}"
            )

        if covariances.dtype != np.dtype(np.float32):
            raise ValueError(
                f"{origin}: BNCI subject {subject} {split} covariances must be float32"
            )
        if broadband.dtype != np.dtype(np.float32):
            raise ValueError(
                f"{origin}: BNCI subject {subject} {split} broadband must be float32"
            )
        if labels.dtype != np.dtype(np.int64):
            raise ValueError(
                f"{origin}: BNCI subject {subject} {split} labels must be int64"
            )
        if run_ids.dtype.kind != "U":
            raise ValueError(
                f"{origin}: BNCI subject {subject} {split} run_ids must be unicode strings"
            )
        if not np.isfinite(covariances).all() or not np.isfinite(broadband).all():
            raise ValueError(
                f"{origin}: BNCI subject {subject} {split} contains non-finite features"
            )
        if not np.allclose(
            covariances,
            np.swapaxes(covariances, -1, -2),
            rtol=1e-5,
            atol=1e-12,
        ):
            raise ValueError(
                f"{origin}: BNCI subject {subject} {split} covariances are not symmetric"
            )
        if np.any(np.linalg.eigvalsh(covariances) <= 0.0):
            raise ValueError(
                f"{origin}: BNCI subject {subject} {split} covariances are not SPD"
            )
        if set(np.unique(labels).tolist()) != {0, 1}:
            raise ValueError(
                f"{origin}: BNCI subject {subject} {split} labels must contain only "
                "both binary classes 0 and 1"
            )

        actual_runs = set(np.unique(run_ids).tolist())
        expected_runs = set(_EXPECTED_RUNS[split])
        if actual_runs != expected_runs:
            raise ValueError(
                f"{origin}: BNCI subject {subject} {split} run IDs are "
                f"{sorted(actual_runs)}; expected {sorted(expected_runs)}"
            )
        for run_id in _EXPECTED_RUNS[split]:
            run_labels = labels[run_ids == run_id]
            if len(run_labels) != _TRIALS_PER_RUN:
                raise ValueError(
                    f"{origin}: BNCI subject {subject} {split} run {run_id} has "
                    f"{len(run_labels)} trials; expected {_TRIALS_PER_RUN}"
                )
            class_counts = np.bincount(run_labels, minlength=2)
            expected_class_counts = np.asarray(
                [_TRIALS_PER_CLASS_PER_RUN, _TRIALS_PER_CLASS_PER_RUN]
            )
            if not np.array_equal(class_counts, expected_class_counts):
                raise ValueError(
                    f"{origin}: BNCI subject {subject} {split} run {run_id} class "
                    f"counts are {class_counts.tolist()}; expected "
                    f"{expected_class_counts.tolist()}"
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
            raise ValueError(f"BNCI cache {path} metadata field {name} is not scalar")
        if (
            actual.dtype.kind != expected_array.dtype.kind
            or actual.item() != expected_array.item()
        ):
            raise ValueError(
                f"BNCI cache {path} has mismatched {name}; it is not valid for "
                f"subject {subject}, protocol {PROTOCOL_ID}, schema "
                f"{_CACHE_SCHEMA_VERSION}, and the current preprocessing source"
            )


def _unpack_cache(path: Path, *, subject: int) -> dict[str, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as cached:
        expected_keys = set(_CACHE_METADATA_NAMES) | {
            f"{split}_{name}" for split in _SPLIT_NAMES for name in _ARRAY_NAMES
        }
        if set(cached.files) != expected_keys:
            raise ValueError(
                f"BNCI cache {path} has unexpected keys: expected "
                f"{sorted(expected_keys)}, found {sorted(cached.files)}"
            )
        _validate_cache_metadata(cached, subject=subject, path=path)
        splits = {
            split: {name: cached[f"{split}_{name}"] for name in _ARRAY_NAMES}
            for split in _SPLIT_NAMES
        }
    _validate_splits(splits, subject=subject, origin=f"BNCI cache {path}")
    return splits


def _write_cache(
    path: Path, splits: dict[str, dict[str, np.ndarray]], *, subject: int
) -> None:
    _validate_splits(splits, subject=subject, origin="generated BNCI payload")
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


def _process_run(raw: Any, run_id: str) -> dict[str, np.ndarray]:
    import mne

    raw = raw.copy()
    missing = [channel for channel in BNCI_CHANNELS if channel not in raw.ch_names]
    if missing:
        raise ValueError(f"BNCI2014-001 run is missing required channels: {missing}")
    raw.pick(list(BNCI_CHANNELS))
    if not np.isclose(float(raw.info["sfreq"]), SFREQ, rtol=0.0, atol=1e-8):
        raw.resample(SFREQ, npad="auto", verbose=False)

    events, event_id = mne.events_from_annotations(
        raw, event_id=_EVENT_CODES, verbose=False
    )
    if event_id != _EVENT_CODES:
        raise RuntimeError(f"unexpected BNCI event mapping: {event_id}")
    if len(events) == 0:
        raise ValueError(f"BNCI run {run_id} has no left/right trials")

    broadband_epochs = _epoch_filtered(raw, events, 8.0, 30.0)
    retained_events = broadband_epochs.events.copy()
    broadband = broadband_epochs.get_data(copy=True).astype(np.float32, copy=False)
    labels = np.asarray(
        [_CODE_TO_LABEL[int(code)] for code in retained_events[:, -1]], dtype=np.int64
    )

    band_data: list[np.ndarray] = []
    for low, high in BANDS:
        epochs = _epoch_filtered(raw, retained_events, low, high)
        if not np.array_equal(epochs.events, retained_events):
            raise RuntimeError("BNCI filter-band epoch alignment changed unexpectedly")
        band_data.append(epochs.get_data(copy=True).astype(np.float32, copy=False))
    filter_bank = np.stack(band_data, axis=1)
    covariances = make_spd_covariances(filter_bank, dtype=np.float32)

    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError(f"BNCI run {run_id} does not contain both task classes")
    return {
        "covariances": covariances,
        "broadband": broadband,
        "labels": labels,
        "run_ids": np.asarray([str(run_id)] * len(labels), dtype=np.str_),
    }


def _concatenate(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not parts:
        raise ValueError("cannot concatenate an empty BNCI partition")
    return {
        name: np.concatenate([part[name] for part in parts]) for name in _ARRAY_NAMES
    }


def _sorted_runs(
    session: dict[str, Any], expected: tuple[str, ...]
) -> list[tuple[str, Any]]:
    actual = tuple(sorted((str(key) for key in session), key=int))
    if actual != expected:
        raise ValueError(f"expected BNCI runs {expected}, found {actual}")
    return [(run_id, session[run_id]) for run_id in actual]


def load_bnci2014_subject(
    subject: int, *, use_cache: bool = True
) -> dict[str, dict[str, np.ndarray]]:
    """Load one subject as locked ``train``, ``validation``, and ``test`` splits.

    Confirmation subjects are intentionally loadable so a frozen benchmark can
    evaluate them, but callers should use :func:`subject_partition` to enforce the
    project-level freeze before requesting subjects 5--9.
    """

    subject = int(subject)
    subject_partition(subject)
    target = _cache_path(subject)
    if use_cache and target.is_file():
        return _unpack_cache(target, subject=subject)

    import mne
    from moabb.datasets import BNCI2014_001

    mne.set_log_level("ERROR")
    sessions = BNCI2014_001().get_data(subjects=[subject])[subject]
    expected_sessions = {TRAIN_SESSION, TEST_SESSION}
    if set(sessions) != expected_sessions:
        raise ValueError(
            f"expected BNCI sessions {sorted(expected_sessions)}, found {sorted(sessions)}"
        )

    expected_runs = ("0", "1", "2", "3", "4", "5")
    training_parts: list[dict[str, np.ndarray]] = []
    validation_parts: list[dict[str, np.ndarray]] = []
    for run_id, raw in _sorted_runs(sessions[TRAIN_SESSION], expected_runs):
        processed = _process_run(raw, run_id)
        if run_id == VALIDATION_RUN:
            validation_parts.append(processed)
        else:
            training_parts.append(processed)
    test_parts = [
        _process_run(raw, run_id)
        for run_id, raw in _sorted_runs(sessions[TEST_SESSION], expected_runs)
    ]

    splits = {
        "train": _concatenate(training_parts),
        "validation": _concatenate(validation_parts),
        "test": _concatenate(test_parts),
    }
    _validate_splits(splits, subject=subject, origin="generated BNCI payload")

    if use_cache:
        _write_cache(target, splits, subject=subject)
    return splits


__all__ = [
    "ALL_SUBJECTS",
    "BNCI_CHANNELS",
    "CONFIRMATION_SUBJECTS",
    "DEVELOPMENT_SUBJECTS",
    "PROTOCOL_ID",
    "load_bnci2014_subject",
    "protocol_metadata",
    "subject_partition",
]
