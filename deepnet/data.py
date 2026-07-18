"""Offline MNE data loading and stable covariance construction.

This module intentionally has no dependency on the acquisition GUI, BrainFlow, the
live classifier, or robot code.  MNE is imported only when a FIF file is actually
loaded, so manifest/protocol/unit-test code remains lightweight.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from .config import (
    DEFAULT_DATA_CONFIG,
    LABEL_NAMES,
    LEFT_LABEL,
    NEUTRAL_LABEL,
    RIGHT_LABEL,
    SUBJECT_RUNS,
    VALID_SESSION_MANIFEST,
    DataConfig,
    is_valid_subject_run,
)


CACHE_SCHEMA_VERSION = 1
_TRIAL_RE = re.compile(r"(?:^|/)t(\d+)$", flags=re.IGNORECASE)


@dataclass(frozen=True, order=True)
class SessionKey:
    """Stable identifier for one recording file (one subject and one run)."""

    subject: int
    run: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "subject", int(self.subject))
        object.__setattr__(self, "run", int(self.run))
        if self.subject <= 0 or self.run <= 0:
            raise ValueError("subject and run must be positive integers")

    @property
    def session_id(self) -> str:
        return f"S{self.subject:02d}-R{self.run:02d}"

    def __str__(self) -> str:
        return self.session_id


@dataclass(frozen=True)
class SessionData:
    """Epoch tensors plus per-epoch provenance for one or more sessions.

    ``epochs`` contains the continuous-filtered filter-bank representation
    ``(N, bands, channels, time)``.  ``broadband_epochs`` is the continuously
    filtered artifact-band representation ``(N, channels, time)`` used by models
    that expect a single EEG time series.  Rest epochs, when requested, use label
    ``-1``; task examples are always left=0 and right=1.
    """

    epochs: NDArray[np.floating[Any]]
    broadband_epochs: NDArray[np.floating[Any]]
    covariances: NDArray[np.floating[Any]]
    labels: NDArray[np.int64]
    subject_ids: NDArray[np.int64]
    run_ids: NDArray[np.int64]
    session_ids: NDArray[np.str_]
    event_samples: NDArray[np.int64]
    event_onsets: NDArray[np.float64]
    trial_ids: NDArray[np.int64]
    annotations: NDArray[np.str_]
    source_files: NDArray[np.str_]

    def __post_init__(self) -> None:
        arrays = {
            "epochs": np.asarray(self.epochs),
            "broadband_epochs": np.asarray(self.broadband_epochs),
            "covariances": np.asarray(self.covariances),
            "labels": np.asarray(self.labels, dtype=np.int64),
            "subject_ids": np.asarray(self.subject_ids, dtype=np.int64),
            "run_ids": np.asarray(self.run_ids, dtype=np.int64),
            "session_ids": np.asarray(self.session_ids, dtype=np.str_),
            "event_samples": np.asarray(self.event_samples, dtype=np.int64),
            "event_onsets": np.asarray(self.event_onsets, dtype=np.float64),
            "trial_ids": np.asarray(self.trial_ids, dtype=np.int64),
            "annotations": np.asarray(self.annotations, dtype=np.str_),
            "source_files": np.asarray(self.source_files, dtype=np.str_),
        }
        for name, value in arrays.items():
            object.__setattr__(self, name, value)

        n_epochs = len(arrays["labels"])
        for name, value in arrays.items():
            if len(value) != n_epochs:
                raise ValueError(f"{name} has {len(value)} rows, expected {n_epochs}")
        epochs = arrays["epochs"]
        broadband = arrays["broadband_epochs"]
        covariances = arrays["covariances"]
        if epochs.ndim != 4:
            raise ValueError("epochs must have shape (N, bands, channels, time)")
        if broadband.ndim != 3:
            raise ValueError("broadband_epochs must have shape (N, channels, time)")
        if covariances.ndim != 4 or covariances.shape[-1] != covariances.shape[-2]:
            raise ValueError("covariances must have shape (N, bands, channels, channels)")
        if epochs.shape[:3] != (
            n_epochs,
            covariances.shape[1],
            covariances.shape[2],
        ):
            raise ValueError("epoch and covariance band/channel dimensions do not match")
        if broadband.shape != (n_epochs, epochs.shape[2], epochs.shape[3]):
            raise ValueError("broadband and filter-bank epoch dimensions do not match")
        if not np.all(np.isfinite(epochs)) or not np.all(np.isfinite(broadband)):
            raise ValueError("epochs contain NaN or infinity")
        if not np.all(np.isfinite(covariances)):
            raise ValueError("covariances contain NaN or infinity")
        valid_labels = {NEUTRAL_LABEL, LEFT_LABEL, RIGHT_LABEL}
        if not set(arrays["labels"].tolist()).issubset(valid_labels):
            raise ValueError(f"labels must be a subset of {sorted(valid_labels)}")

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def X(self) -> NDArray[np.floating[Any]]:
        """Training-friendly alias for covariance matrices."""

        return self.covariances

    @property
    def y(self) -> NDArray[np.int64]:
        return self.labels

    @property
    def task_mask(self) -> NDArray[np.bool_]:
        return self.labels >= 0

    @property
    def neutral_mask(self) -> NDArray[np.bool_]:
        return self.labels == NEUTRAL_LABEL

    @property
    def key(self) -> SessionKey | None:
        pairs = set(zip(self.subject_ids.tolist(), self.run_ids.tolist(), strict=True))
        if len(pairs) != 1:
            return None
        subject, run = next(iter(pairs))
        return SessionKey(subject, run)

    @property
    def metadata(self) -> Mapping[str, np.ndarray]:
        """Return a read-only-by-convention view of all split/provenance columns."""

        return {
            "subject_ids": self.subject_ids,
            "run_ids": self.run_ids,
            "session_ids": self.session_ids,
            "event_samples": self.event_samples,
            "event_onsets": self.event_onsets,
            "trial_ids": self.trial_ids,
            "annotations": self.annotations,
            "source_files": self.source_files,
        }

    def subset(self, indices: Sequence[int] | NDArray[np.bool_]) -> "SessionData":
        selection = np.asarray(indices)
        return SessionData(
            **{name: np.asarray(getattr(self, name))[selection] for name in _SESSION_FIELDS}
        )

    @classmethod
    def concatenate(cls, sessions: Sequence["SessionData"]) -> "SessionData":
        if not sessions:
            raise ValueError("at least one SessionData object is required")
        return cls(
            **{
                name: np.concatenate([np.asarray(getattr(session, name)) for session in sessions])
                for name in _SESSION_FIELDS
            }
        )


_SESSION_FIELDS = (
    "epochs",
    "broadband_epochs",
    "covariances",
    "labels",
    "subject_ids",
    "run_ids",
    "session_ids",
    "event_samples",
    "event_onsets",
    "trial_ids",
    "annotations",
    "source_files",
)


def valid_sessions(
    config: DataConfig = DEFAULT_DATA_CONFIG, *, existing_only: bool = False
) -> tuple[SessionKey, ...]:
    """Return the fixed 32-session cohort in chronological subject/run order."""

    keys = tuple(SessionKey(subject, run) for subject, run in VALID_SESSION_MANIFEST)
    if existing_only:
        keys = tuple(key for key in keys if session_path(key, config).is_file())
    return keys


def session_path(key: SessionKey, config: DataConfig = DEFAULT_DATA_CONFIG) -> Path:
    """Resolve a manifest session without using a permissive filesystem glob."""

    if not is_valid_subject_run(key.subject, key.run):
        expected = SUBJECT_RUNS.get(key.subject)
        if expected is None:
            raise ValueError(f"subject {key.subject} is excluded from the valid manifest")
        raise ValueError(
            f"run {key.run} is invalid for subject {key.subject}; valid runs are {expected}"
        )
    return config.data_dir / (
        f"exp4_subject{key.subject}_training_{key.run}_mi_raw.fif"
    )


def annotation_label(description: str, *, include_rest: bool = False) -> int | None:
    """Map one hierarchical experiment annotation to a learning target."""

    parts = tuple(part.strip().lower() for part in str(description).split("/"))
    if len(parts) < 2:
        return None
    intent, phase = parts[:2]
    if phase == "task" and intent == "left_hand":
        return LEFT_LABEL
    if phase == "task" and intent == "right_hand":
        return RIGHT_LABEL
    if include_rest and phase == "rest" and intent in {"left_hand", "right_hand"}:
        return NEUTRAL_LABEL
    return None


def make_spd_covariances(
    epochs: NDArray[np.floating[Any]],
    *,
    shrinkage: float = 1e-3,
    demean: bool = True,
    dtype: str | np.dtype[Any] | None = None,
) -> NDArray[np.floating[Any]]:
    """Create shrinkage-regularized spatial covariances in stable float64 math.

    The input may have arbitrary leading dimensions followed by ``(channels,
    time)``.  Computation is always float64; output defaults to the input's float32
    or float64 precision.  A scale-aware final diagonal floor protects degenerate
    synthetic/flat windows without overwhelming EEG covariances measured in volts.
    """

    values = np.asarray(epochs)
    if values.ndim < 3:
        raise ValueError("epochs must end in (channels, time) and include an epoch axis")
    if values.shape[-1] < 2 or values.shape[-2] < 1:
        raise ValueError("each epoch needs at least one channel and two samples")
    if not np.all(np.isfinite(values)):
        raise ValueError("epochs contain NaN or infinity")
    if not 0.0 < shrinkage <= 1.0:
        raise ValueError("shrinkage must be in (0, 1]")

    if dtype is None:
        output_dtype = np.dtype(np.float64 if values.dtype == np.float64 else np.float32)
    else:
        output_dtype = np.dtype(dtype)
    if output_dtype not in {np.dtype(np.float32), np.dtype(np.float64)}:
        raise ValueError("covariance dtype must be float32 or float64")

    work = values.astype(np.float64, copy=False)
    if demean:
        work = work - work.mean(axis=-1, keepdims=True)
    n_times = work.shape[-1]
    covariance = np.einsum("...ct,...dt->...cd", work, work, optimize=True) / float(n_times)
    n_channels = covariance.shape[-1]
    trace_scale = np.trace(covariance, axis1=-2, axis2=-1) / float(n_channels)
    identity = np.eye(n_channels, dtype=np.float64)
    covariance = (1.0 - shrinkage) * covariance + (
        shrinkage * trace_scale[..., None, None] * identity
    )
    covariance = 0.5 * (covariance + np.swapaxes(covariance, -1, -2))

    # A relative floor preserves physical EEG scale (roughly 1e-10 V^2 here).
    dtype_info = np.finfo(output_dtype)
    floor = np.maximum(np.abs(trace_scale) * dtype_info.eps * 8.0, dtype_info.tiny * 8.0)
    minimum = np.linalg.eigvalsh(covariance)[..., 0]
    diagonal_add = np.maximum(0.0, floor - minimum)
    covariance = covariance + diagonal_add[..., None, None] * identity
    return covariance.astype(output_dtype, copy=False)


def preprocessing_settings(config: DataConfig) -> dict[str, Any]:
    """JSON-compatible settings that affect any cached tensor."""

    return {
        "schema": CACHE_SCHEMA_VERSION,
        "sfreq": config.sfreq,
        "channels": list(config.channels),
        "bands": [list(band) for band in config.bands],
        "window_name": config.window_name,
        "epoch_window": list(config.epoch_window),
        "include_rest": config.include_rest,
        "labels": {
            "left": LEFT_LABEL,
            "right": RIGHT_LABEL,
            "neutral": NEUTRAL_LABEL,
        },
        "artifact_threshold": config.artifact_threshold,
        "artifact_band": list(config.artifact_band),
        "covariance_shrinkage": config.covariance_shrinkage,
        "covariance_demean": config.covariance_demean,
        "dtype": config.dtype,
        "filter_method": config.filter_method,
        "filter_phase": config.filter_phase,
        "fir_design": config.fir_design,
        "l_trans_bandwidth": config.l_trans_bandwidth,
        "h_trans_bandwidth": config.h_trans_bandwidth,
        "resample_if_needed": config.resample_if_needed,
    }


def preprocessing_fingerprint(config: DataConfig) -> str:
    payload = json.dumps(preprocessing_settings(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_contract(data: SessionData, config: DataConfig) -> dict[str, Any]:
    """Return a portable, content-addressed preprocessing and source-data contract.

    Result artifacts use this contract to prevent a resumed run from silently
    mixing channel orders, filters, covariance settings, or changed FIF files.
    Paths are recorded relative to ``data_dir`` when possible; file content, not
    the workstation-specific absolute prefix, determines identity.
    """

    sessions: list[dict[str, Any]] = []
    order = np.lexsort((data.run_ids, data.subject_ids))
    ordered_session_ids = tuple(dict.fromkeys(data.session_ids[order].tolist()))
    data_root = config.data_dir.resolve()
    for session_id in ordered_session_ids:
        rows = np.flatnonzero(data.session_ids == session_id)
        subjects = np.unique(data.subject_ids[rows])
        runs = np.unique(data.run_ids[rows])
        sources = np.unique(data.source_files[rows])
        if len(subjects) != 1 or len(runs) != 1 or len(sources) != 1:
            raise ValueError(f"session {session_id} has inconsistent provenance")
        source = Path(str(sources[0])).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"contract source does not exist: {source}")
        try:
            display_path = source.relative_to(data_root).as_posix()
        except ValueError:
            display_path = str(source)
        stat = source.stat()
        sessions.append(
            {
                "session_id": str(session_id),
                "subject": int(subjects[0]),
                "run": int(runs[0]),
                "source_file": display_path,
                "size_bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "sha256": _sha256_file(source),
            }
        )

    core = {
        "preprocessing": preprocessing_settings(config),
        "preprocessing_fingerprint": preprocessing_fingerprint(config),
        "sessions": sessions,
    }
    encoded = json.dumps(core, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {**core, "contract_sha256": hashlib.sha256(encoded).hexdigest()}


def cache_key(
    key: SessionKey,
    config: DataConfig = DEFAULT_DATA_CONFIG,
    *,
    source: Path | None = None,
) -> str:
    """Return a cache filename whose digest covers every preprocessing setting."""

    source_path = source if source is not None else session_path(key, config)
    source_info: dict[str, Any] = {"path": str(source_path.resolve())}
    if source_path.is_file():
        stat = source_path.stat()
        source_info.update({"size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    payload = {
        "session": {"subject": key.subject, "run": key.run},
        "preprocessing": preprocessing_settings(config),
        "source": source_info,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:24]
    rest = "neutral" if config.include_rest else "task"
    return f"{key.session_id.lower()}_win-{config.window_name}_{rest}_{config.dtype}_{digest}.npz"


def _mne_epoch_data(raw: Any, events: np.ndarray, tmin: float, tmax: float) -> Any:
    import mne

    return mne.Epochs(
        raw,
        events,
        event_id=None,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        proj=False,
        on_missing="ignore",
        reject_by_annotation=True,
        verbose=False,
    )


def _continuous_filter(raw: Any, low: float, high: float, config: DataConfig) -> Any:
    return raw.copy().filter(
        l_freq=low,
        h_freq=high,
        method=config.filter_method,
        phase=config.filter_phase,
        fir_design=config.fir_design,
        l_trans_bandwidth=config.l_trans_bandwidth,
        h_trans_bandwidth=config.h_trans_bandwidth,
        verbose=False,
    )


def load_fif_session(path: Path, key: SessionKey, config: DataConfig) -> SessionData:
    """Load one FIF recording through the dedicated offline MNE pipeline."""

    try:
        import mne
    except ImportError as error:  # pragma: no cover - exercised on minimal installations
        raise RuntimeError("MNE is required to load FIF recordings") from error

    raw = mne.io.read_raw_fif(path, preload=True, verbose=False)
    missing = [channel for channel in config.channels if channel not in raw.ch_names]
    if missing:
        raise ValueError(f"{path} is missing required EEG channels: {missing}")
    raw.pick(list(config.channels))
    source_sfreq = float(raw.info["sfreq"])
    if not np.isclose(source_sfreq, config.sfreq, rtol=0.0, atol=1e-8):
        if not config.resample_if_needed:
            raise ValueError(
                f"{path} has {source_sfreq:g} Hz data, expected {config.sfreq:g} Hz"
            )
        raw.resample(config.sfreq, npad="auto", verbose=False)

    descriptions = [str(description) for description in raw.annotations.description]
    selected_descriptions = sorted(
        {
            description
            for description in descriptions
            if annotation_label(description, include_rest=config.include_rest) is not None
        }
    )
    if not selected_descriptions:
        raise ValueError(f"no task annotations were found in {path}")
    event_id = {description: index + 1 for index, description in enumerate(selected_descriptions)}
    events, _ = mne.events_from_annotations(raw, event_id=event_id, verbose=False)
    code_to_description = {code: description for description, code in event_id.items()}

    tmin, tmax = config.epoch_window
    artifact_raw = _continuous_filter(raw, *config.artifact_band, config)
    artifact_epochs = _mne_epoch_data(artifact_raw, events, tmin, tmax)
    broadband = artifact_epochs.get_data(copy=True)
    retained_events = artifact_epochs.events.copy()
    retained_descriptions = np.asarray(
        [code_to_description[int(code)] for code in retained_events[:, -1]], dtype=np.str_
    )
    labels = np.asarray(
        [
            annotation_label(description, include_rest=config.include_rest)
            for description in retained_descriptions
        ],
        dtype=np.int64,
    )

    if config.artifact_threshold is None:
        keep = np.ones(len(broadband), dtype=bool)
    else:
        peak_to_peak = np.ptp(broadband, axis=-1).max(axis=-1)
        keep = peak_to_peak < config.artifact_threshold
    if not np.any(keep):
        raise ValueError(f"artifact rejection removed every epoch from {path}")

    band_data: list[np.ndarray] = []
    for low, high in config.bands:
        filtered = _continuous_filter(raw, low, high, config)
        epochs = _mne_epoch_data(filtered, retained_events, tmin, tmax)
        if len(epochs) != len(retained_events):
            raise RuntimeError("filter-band epoch alignment changed unexpectedly")
        band_data.append(epochs.get_data(copy=True)[keep])
    filter_bank = np.stack(band_data, axis=1)
    broadband = broadband[keep]
    retained_events = retained_events[keep]
    retained_descriptions = retained_descriptions[keep]
    labels = labels[keep]

    output_dtype = np.dtype(config.dtype)
    filter_bank = filter_bank.astype(output_dtype, copy=False)
    broadband = broadband.astype(output_dtype, copy=False)
    covariances = make_spd_covariances(
        filter_bank,
        shrinkage=config.covariance_shrinkage,
        demean=config.covariance_demean,
        dtype=output_dtype,
    )
    n_epochs = len(labels)
    event_samples = retained_events[:, 0].astype(np.int64, copy=False)
    event_onsets = (event_samples - int(raw.first_samp)) / config.sfreq
    trial_ids = np.asarray(
        [
            int(match.group(1)) if (match := _TRIAL_RE.search(description)) else -1
            for description in retained_descriptions
        ],
        dtype=np.int64,
    )
    source = str(path.resolve())
    return SessionData(
        epochs=filter_bank,
        broadband_epochs=broadband,
        covariances=covariances,
        labels=labels,
        subject_ids=np.full(n_epochs, key.subject, dtype=np.int64),
        run_ids=np.full(n_epochs, key.run, dtype=np.int64),
        session_ids=np.asarray([key.session_id] * n_epochs, dtype=np.str_),
        event_samples=event_samples,
        event_onsets=event_onsets,
        trial_ids=trial_ids,
        annotations=retained_descriptions,
        source_files=np.asarray([source] * n_epochs, dtype=np.str_),
    )


def _load_cache(path: Path, expected_fingerprint: str) -> SessionData | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as cached:
            fingerprint = str(cached["preprocessing_fingerprint"].item())
            if fingerprint != expected_fingerprint:
                return None
            values = {name: cached[name] for name in _SESSION_FIELDS}
        return SessionData(**values)
    except (OSError, ValueError, KeyError):
        # An interrupted/stale cache is never trusted; the caller rebuilds it.
        return None


def _write_cache(path: Path, data: SessionData, fingerprint: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", suffix=".npz", prefix=f".{path.stem}-", dir=path.parent, delete=False
        ) as temporary:
            temporary_name = temporary.name
            np.savez_compressed(
                temporary,
                preprocessing_fingerprint=np.asarray(fingerprint),
                **{name: getattr(data, name) for name in _SESSION_FIELDS},
            )
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def load_session(
    key: SessionKey,
    config: DataConfig = DEFAULT_DATA_CONFIG,
    *,
    use_cache: bool | None = None,
) -> SessionData:
    """Load and preprocess one valid manifest session, optionally using an NPZ cache."""

    source = session_path(key, config)
    if not source.is_file():
        raise FileNotFoundError(f"manifest recording does not exist: {source}")
    cache_enabled = config.use_cache if use_cache is None else bool(use_cache)
    fingerprint = preprocessing_fingerprint(config)
    target = config.cache_dir / cache_key(key, config, source=source)
    if cache_enabled:
        cached = _load_cache(target, fingerprint)
        if cached is not None:
            return cached
    loaded = load_fif_session(source, key, config)
    if cache_enabled:
        _write_cache(target, loaded, fingerprint)
    return loaded


def load_sessions(
    keys: Iterable[SessionKey],
    config: DataConfig = DEFAULT_DATA_CONFIG,
    *,
    use_cache: bool | None = None,
) -> SessionData:
    """Load and concatenate sessions while retaining epoch-level provenance."""

    selected = tuple(keys)
    if not selected:
        raise ValueError("at least one session key is required")
    if len(set(selected)) != len(selected):
        raise ValueError("duplicate session keys would duplicate epochs")
    return SessionData.concatenate(
        [load_session(key, config, use_cache=use_cache) for key in selected]
    )


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "SessionData",
    "SessionKey",
    "annotation_label",
    "cache_key",
    "dataset_contract",
    "load_fif_session",
    "load_session",
    "load_sessions",
    "make_spd_covariances",
    "preprocessing_fingerprint",
    "preprocessing_settings",
    "session_path",
    "valid_sessions",
]
