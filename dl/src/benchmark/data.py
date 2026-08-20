"""Coordinate-preserving public EEG cache and leakage-resistant splits."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import stat
import zipfile
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Final, Sequence

import numpy as np
from numpy.typing import NDArray

from .config import (
    BNCI004_3_CHANNELS,
    CANONICAL_21_CHANNELS,
    CHANNEL_SCALING,
    DEFAULT_MONTAGE_PROFILE,
    LOCAL_EXP4_15_CHANNELS,
    LOCAL_EXP4_CHANNEL_ALIASES,
    LOCAL_EXP4_DATA_ROOT,
    LOCAL_EXP4_PREPROCESSING,
    LOCAL_EXP4_SOURCE_CHANNELS,
    LOCAL_EXP4_SUBJECT_RUNS,
    PREPROCESSING,
    REFLECTION_PAIRS,
    MONTAGE_PROFILES,
    ZHOU_8_CHANNELS,
    DatasetSpec,
    channels_for_dataset,
    coordinate_contract_for_dataset,
    dataset_spec,
    preprocessing_for_dataset,
    validate_montage_profile,
)


FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int64]

SUBJECT_CACHE_NPZ_MEMBERS: Final[tuple[str, ...]] = (
    "x",
    "y",
    "positions",
    "channel_names",
    "sessions",
    "runs",
    "identity",
)


# ``PhysionetMI(imagined=True, executed=False)`` loads the original unilateral
# imagery recordings 4, 8, and 12, but MOABB exposes them as run keys 0, 1,
# and 2.  Store the original acquisition IDs in our cache so an audit can
# distinguish the motor-imagery recordings from the executed-movement and
# bilateral hand/feet recordings in the source dataset.
PHYSIONET_MI_RUNS: Final[tuple[str, ...]] = ("4", "8", "12")
_PHYSIONET_MOABB_RUN_TO_SOURCE: Final[dict[str, str]] = {
    "0": "4",
    "1": "8",
    "2": "12",
}

_LOCAL_EXP4_TASK_RE: Final[re.Pattern[str]] = re.compile(
    r"^(left_hand|right_hand)/task/t([1-9][0-9]*)$", re.IGNORECASE
)


def _channel_name_token(name: str) -> str:
    """Normalize harmless acquisition punctuation without erasing references."""

    return re.sub(r"[.\s]+", "", str(name)).upper()


@lru_cache(maxsize=1)
def _standard_1005_name_aliases() -> dict[str, str]:
    """Map case/punctuation-insensitive tokens to exact atlas spelling."""

    import mne

    aliases: dict[str, str] = {}
    ambiguous: set[str] = set()
    for atlas_name in mne.channels.make_standard_montage("standard_1005").ch_names:
        token = _channel_name_token(atlas_name)
        previous = aliases.get(token)
        if previous is not None and previous != atlas_name:
            ambiguous.add(token)
        else:
            aliases[token] = atlas_name
    for token in ambiguous:
        aliases.pop(token, None)
    return aliases


def _canonical_channel_name(name: str) -> str:
    value = re.sub(r"[.\s]+", "", str(name).strip())
    token = value.upper()
    legacy_aliases = {
        source.upper(): target
        for source, target in LOCAL_EXP4_CHANNEL_ALIASES.items()
    }
    if token in legacy_aliases:
        return legacy_aliases[token]
    return _standard_1005_name_aliases().get(token, value)


def canonicalize_standard_1005_channels(
    channel_names: Sequence[str],
) -> tuple[str, ...]:
    """Preflight one ordered point-EEG montage against the fixed atlas."""

    if not channel_names:
        raise ValueError("native channel list must not be empty")
    atlas_tokens = _standard_1005_name_aliases()
    canonical: list[str] = []
    unknown: list[str] = []
    for source_name in channel_names:
        token = _channel_name_token(source_name)
        if token in LOCAL_EXP4_CHANNEL_ALIASES:
            name = LOCAL_EXP4_CHANNEL_ALIASES[token]
        else:
            name = atlas_tokens.get(token, "")
        if not name:
            unknown.append(str(source_name))
        else:
            canonical.append(name)
    if unknown:
        raise ValueError(
            "channels are absent or ambiguous in MNE standard_1005: "
            f"{unknown}"
        )
    if len(set(canonical)) != len(canonical):
        raise ValueError("channel names collide after standard_1005 canonicalization")
    return tuple(canonical)


@lru_cache(maxsize=None)
def _standard_1005_position_rows(
    channel_names: tuple[str, ...],
) -> tuple[tuple[float, float, float], ...]:
    """Resolve names through one fixed MNE atlas and MNE head transform."""

    import mne
    from mne.io.constants import FIFF

    if not channel_names or len(set(channel_names)) != len(channel_names):
        raise ValueError("channel names must be non-empty and unique")
    atlas_info = mne.create_info(
        list(channel_names),
        sfreq=float(PREPROCESSING["sfreq_hz"]),
        ch_types="eeg",
    )
    try:
        atlas_info.set_montage("standard_1005", on_missing="raise")
    except (KeyError, ValueError) as exc:
        raise ValueError(
            f"configured channels are absent from MNE standard_1005: {channel_names}"
        ) from exc
    positions: list[tuple[float, float, float]] = []
    for channel in atlas_info["chs"]:
        if int(channel["coord_frame"]) != int(FIFF.FIFFV_COORD_HEAD):
            raise RuntimeError("standard_1005 coordinate was not transformed to MNE head")
        position = np.asarray(channel["loc"][:3], dtype=np.float64)
        norm = float(np.linalg.norm(position))
        if not np.all(np.isfinite(position)) or norm < 1e-7:
            raise RuntimeError(
                f"standard_1005 channel {channel['ch_name']!r} has no valid coordinate"
            )
        unit = position / norm
        positions.append(tuple(float(value) for value in unit))
    return tuple(positions)


def _atlas_unit_positions(channel_names: Sequence[str]) -> FloatArray:
    """Return float32 unit coordinates from the configured standard_1005 atlas."""

    canonical = canonicalize_standard_1005_channels(channel_names)
    result = np.asarray(_standard_1005_position_rows(canonical), dtype=np.float32)
    if result.shape != (len(canonical), 3) or not np.all(np.isfinite(result)):
        raise RuntimeError("invalid standard_1005 channel-coordinate matrix")
    return result


def _unit_positions(info: Any) -> tuple[tuple[str, ...], FloatArray]:
    """Resolve source channel names through the fixed standard_1005 atlas.

    Source ``loc`` values and coordinate frames are deliberately ignored. Both
    harmonized and native profiles are represented in this one atlas frame.
    """

    names = canonicalize_standard_1005_channels(info["ch_names"])
    return names, _atlas_unit_positions(names)


def _inclusive_api_tmax(contract: dict[str, object]) -> float:
    """Validate the contract and return its inclusive-API boundary.

    MNE/MOABB includes ``tmax``. Requesting the exclusive boundary preserves
    the timestamp needed to verify the half-open cut after library-level
    resampling; requesting ``tmax - 1 / sfreq`` can lose one sample because
    some datasets are cropped before they are resampled.
    """

    tmin = float(contract["tmin_seconds"])
    tmax_exclusive = float(contract["tmax_seconds_exclusive"])
    sfreq = float(contract["sfreq_hz"])
    n_times = int(contract["n_times"])
    exact_samples = (tmax_exclusive - tmin) * sfreq
    if not np.isclose(exact_samples, n_times, rtol=0.0, atol=1e-9):
        raise RuntimeError(
            "epoch contract duration, sample rate, and n_times are inconsistent"
        )
    return tmax_exclusive


def _select_half_open_epoch_samples(
    values: FloatArray,
    times: NDArray[np.float64] | Sequence[float],
    *,
    contract: dict[str, object],
    source: str,
) -> FloatArray:
    """Select the declared timestamp grid rather than blindly truncating."""

    values = np.asarray(values, dtype=np.float32)
    observed_times = np.asarray(times, dtype=np.float64)
    if values.ndim != 3 or observed_times.ndim != 1:
        raise RuntimeError(f"{source} returned invalid epoch values/timestamps")
    if values.shape[-1] != len(observed_times):
        raise RuntimeError(f"{source} epoch values and timestamps disagree")
    tmin = float(contract["tmin_seconds"])
    tmax_exclusive = float(contract["tmax_seconds_exclusive"])
    sfreq = float(contract["sfreq_hz"])
    n_times = int(contract["n_times"])
    _inclusive_api_tmax(contract)
    api_origin = float(contract.get("moabb_interval_start_seconds", 0.0))
    if not np.isfinite(api_origin):
        raise RuntimeError(f"{source} has a non-finite API timestamp origin")
    expected = api_origin + tmin + np.arange(n_times, dtype=np.float64) / sfreq
    if len(observed_times) < n_times or not np.allclose(
        observed_times[:n_times], expected, rtol=0.0, atol=1e-9
    ):
        raise RuntimeError(
            f"{source} timestamps do not realize the declared half-open grid"
        )
    tolerance = 1e-9
    api_tmax_exclusive = api_origin + tmax_exclusive
    if np.any(observed_times[:n_times] >= api_tmax_exclusive - tolerance):
        raise RuntimeError(f"{source} selected a timestamp outside the half-open epoch")
    if (
        len(observed_times) > n_times
        and observed_times[n_times] < api_tmax_exclusive - tolerance
    ):
        raise RuntimeError(f"{source} contains an unselected in-interval timestamp")
    return np.ascontiguousarray(values[..., :n_times], dtype=np.float32)


def _require_exact_epoch_length(
    values: FloatArray,
    *,
    n_times: int,
    source: str,
) -> FloatArray:
    """Require the declared length; never truncate or pad a cache epoch."""

    if values.ndim != 3 or values.shape[-1] != n_times:
        observed = values.shape[-1] if values.ndim else "scalar"
        raise RuntimeError(
            f"{source} has {observed} samples; expected exactly {n_times}"
        )
    return np.ascontiguousarray(values, dtype=np.float32)


def _dataset_instance(spec: DatasetSpec) -> Any:
    from moabb import datasets as moabb_datasets

    constructor = getattr(moabb_datasets, spec.moabb_class, None)
    if constructor is None:
        raise RuntimeError(f"MOABB does not provide {spec.moabb_class}")
    if spec.key == "physionet_mi":
        # Pin the constructor flags instead of relying on MOABB defaults.  The
        # resulting hand runs are source recordings 4, 8, and 12 only.
        return constructor(imagined=True, executed=False)
    return constructor()


def _canonical_run_ids(dataset: str, runs: NDArray[np.str_]) -> NDArray[np.str_]:
    """Translate library-local run keys to auditable source recording IDs."""

    values = np.asarray(runs).astype(str)
    if dataset != "physionet_mi":
        return values.astype("U", copy=False)
    observed = set(values.tolist())
    expected = set(_PHYSIONET_MOABB_RUN_TO_SOURCE)
    if observed != expected:
        raise RuntimeError(
            "PhysionetMI must contain only MOABB hand-run keys 0/1/2 "
            f"(source imagery runs 4/8/12); found {sorted(observed)}"
        )
    return np.asarray(
        [_PHYSIONET_MOABB_RUN_TO_SOURCE[value] for value in values], dtype="U"
    )


def _cache_path(
    root: Path,
    dataset: str,
    subject: int,
    montage_profile: str = DEFAULT_MONTAGE_PROFILE,
) -> Path:
    profile = validate_montage_profile(montage_profile)
    return (
        root
        / PREPROCESSING["schema"]
        / profile
        / dataset
        / f"subject_{subject:03d}.npz"
    )


def _json_scalar(value: Any) -> NDArray[np.str_]:
    return np.asarray(json.dumps(value, sort_keys=True, separators=(",", ":")))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_unique_regular_bytes(path: Path) -> bytes:
    """Read one immutable cache leaf without following filesystem indirection.

    Formal benchmark caches are evidence-bearing inputs.  Resolve neither a
    symlinked ancestor nor a linked/special leaf, and open the leaf
    nonblocking before parsing it.  The descriptor identity and metadata are
    checked on both sides of the read so a pathname replacement or in-place
    mutation cannot silently supply a different archive.
    """

    absolute = Path(os.path.abspath(path))
    if absolute.name in {"", ".", ".."}:
        raise RuntimeError(f"cache path has no regular-file leaf: {absolute}")

    directory_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    directory_flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    parent_descriptor = os.open(absolute.anchor, directory_flags)
    try:
        walked = Path(absolute.anchor)
        for component in absolute.parts[1:-1]:
            walked /= component
            try:
                child_descriptor = os.open(
                    component,
                    directory_flags,
                    dir_fd=parent_descriptor,
                )
            except FileNotFoundError as error:
                raise FileNotFoundError(walked) from error
            except OSError as error:
                raise RuntimeError(
                    f"cache ancestor is not a real directory: {walked}"
                ) from error
            child_stat = os.fstat(child_descriptor)
            if not stat.S_ISDIR(child_stat.st_mode):
                os.close(child_descriptor)
                raise RuntimeError(
                    f"cache ancestor is not a real directory: {walked}"
                )
            os.close(parent_descriptor)
            parent_descriptor = child_descriptor

        try:
            path_stat = os.stat(
                absolute.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError as error:
            raise FileNotFoundError(absolute) from error
        if (
            stat.S_ISLNK(path_stat.st_mode)
            or not stat.S_ISREG(path_stat.st_mode)
            or path_stat.st_nlink != 1
        ):
            raise RuntimeError(
                f"cache leaf must be a regular single-link file: {absolute}"
            )

        file_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        file_flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        descriptor = os.open(
            absolute.name,
            file_flags,
            dir_fd=parent_descriptor,
        )
        before = os.fstat(descriptor)
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or any(
                getattr(before, name) != getattr(path_stat, name)
                for name in stable_fields
            )
        ):
            os.close(descriptor)
            raise RuntimeError(
                f"cache leaf changed while it was being opened: {absolute}"
            )
        try:
            chunks: list[bytes] = []
            while True:
                try:
                    chunk = os.read(descriptor, 1024 * 1024)
                except InterruptedError:
                    continue
                if not chunk:
                    break
                chunks.append(chunk)
            after = os.fstat(descriptor)
            try:
                final_path_stat = os.stat(
                    absolute.name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError as error:
                raise RuntimeError(
                    f"cache leaf changed while it was read: {absolute}"
                ) from error
            if any(
                getattr(before, name) != getattr(observed, name)
                for observed in (after, final_path_stat)
                for name in stable_fields
            ):
                raise RuntimeError(f"cache leaf changed while it was read: {absolute}")
            payload = b"".join(chunks)
            if len(payload) != before.st_size:
                raise RuntimeError(f"cache leaf read was incomplete: {absolute}")
            return payload
        finally:
            os.close(descriptor)
    finally:
        os.close(parent_descriptor)


def _validate_npz_member_contract(
    payload: bytes,
    *,
    expected_members: Sequence[str],
    source: str,
) -> tuple[str, ...]:
    """Validate raw ZIP members before NumPy is allowed to resolve names.

    ``numpy.load`` exposes a de-duplicated logical-name view and can therefore
    hide duplicate ZIP entries. Evidence-bearing NPZ inputs must contain one
    exact ordered ``.npy`` member for every declared logical member.
    """

    expected = tuple(str(value) for value in expected_members)
    if (
        not expected
        or len(expected) != len(set(expected))
        or any(
            not value
            or value in {".", ".."}
            or "/" in value
            or "\\" in value
            for value in expected
        )
    ):
        raise RuntimeError("NPZ member contract itself is invalid")
    try:
        with zipfile.ZipFile(io.BytesIO(payload), mode="r") as archive:
            infos = archive.infolist()
    except (OSError, zipfile.BadZipFile) as error:
        raise RuntimeError(f"{source} is not a valid NPZ archive") from error
    raw_names = tuple(info.filename for info in infos)
    expected_raw_names = tuple(f"{name}.npy" for name in expected)
    if (
        len(infos) != len(expected)
        or len(raw_names) != len(set(raw_names))
        or raw_names != expected_raw_names
        or any(
            info.is_dir()
            or info.flag_bits & 0x1
            or info.filename.startswith(("/", "\\"))
            or "\\" in info.filename
            or any(
                component in {"", ".", ".."}
                for component in Path(info.filename).parts
            )
            for info in infos
        )
    ):
        raise RuntimeError(
            f"{source} raw ZIP members differ from the exact ordered NPZ "
            f"contract: {list(raw_names)}"
        )
    return expected


def _local_exp4_path(root: Path, subject: int, run: int) -> Path:
    """Resolve exactly one manifest recording without filesystem discovery."""

    return root / f"exp4_subject{subject}_training_{run}_mi_raw.fif"


def _local_exp4_task_events(raw: Any, source: Path) -> tuple[np.ndarray, IntArray]:
    """Extract the 60 balanced task cues and reject malformed local metadata."""

    import mne

    selected: list[tuple[str, int, int]] = []
    for duration, description in zip(
        raw.annotations.duration,
        raw.annotations.description,
        strict=True,
    ):
        value = str(description)
        match = _LOCAL_EXP4_TASK_RE.fullmatch(value)
        if match is None:
            continue
        if float(duration) + 1e-6 < float(
            LOCAL_EXP4_PREPROCESSING["tmax_seconds_exclusive"]
        ):
            raise RuntimeError(
                f"{source.name} task annotation {value!r} is shorter than 2 seconds"
            )
        label = 0 if match.group(1).lower() == "left_hand" else 1
        selected.append((value, int(match.group(2)), label))

    trial_ids = [trial_id for _, trial_id, _ in selected]
    labels = [label for _, _, label in selected]
    if len(selected) != 60:
        raise RuntimeError(f"{source.name} has {len(selected)} task cues; expected 60")
    if set(trial_ids) != set(range(1, 61)) or len(set(trial_ids)) != 60:
        raise RuntimeError(f"{source.name} task trial IDs must be exactly 1..60")
    if labels.count(0) != 30 or labels.count(1) != 30:
        raise RuntimeError(f"{source.name} must contain 30 left and 30 right task cues")

    descriptions = sorted(value for value, _, _ in selected)
    if len(set(descriptions)) != 60:
        raise RuntimeError(f"{source.name} contains duplicate task descriptions")
    event_id = {description: index + 1 for index, description in enumerate(descriptions)}
    events, _ = mne.events_from_annotations(
        raw,
        event_id=event_id,
        use_rounding=True,
        verbose="ERROR",
    )
    if len(events) != 60:
        raise RuntimeError(f"{source.name} produced {len(events)} task events; expected 60")
    description_by_code = {code: description for description, code in event_id.items()}
    label_by_description = {description: label for description, _, label in selected}
    ordered_labels = np.asarray(
        [label_by_description[description_by_code[int(code)]] for code in events[:, 2]],
        dtype=np.int64,
    )
    if np.bincount(ordered_labels, minlength=2).tolist() != [30, 30]:
        raise RuntimeError(f"{source.name} event conversion changed class counts")
    return events, ordered_labels


def _load_local_exp4_run(
    root: Path,
    subject: int,
    run: int,
) -> dict[str, Any]:
    """Load one explicit Exp4 run through the frozen harmonized pipeline."""

    import mne

    source = _local_exp4_path(root, subject, run)
    if not source.is_file():
        raise FileNotFoundError(f"manifest recording does not exist: {source}")
    raw = mne.io.read_raw_fif(source, preload=True, verbose="ERROR")
    source_sfreq = float(raw.info["sfreq"])
    expected_source_sfreq = float(LOCAL_EXP4_PREPROCESSING["source_sfreq_hz"])
    if not np.isclose(source_sfreq, expected_source_sfreq, rtol=0.0, atol=1e-8):
        raise RuntimeError(
            f"{source.name} is {source_sfreq:g} Hz; expected {expected_source_sfreq:g} Hz"
        )
    missing = [name for name in LOCAL_EXP4_SOURCE_CHANNELS if name not in raw.ch_names]
    if missing:
        raise RuntimeError(f"{source.name} is missing required EEG channels: {missing}")
    bad_required = sorted(set(raw.info["bads"]) & set(LOCAL_EXP4_SOURCE_CHANNELS))
    if bad_required:
        raise RuntimeError(f"{source.name} marks required channels bad: {bad_required}")

    raw.pick(list(LOCAL_EXP4_SOURCE_CHANNELS))
    if tuple(raw.ch_names) != LOCAL_EXP4_SOURCE_CHANNELS:
        raise RuntimeError(f"{source.name} did not preserve the locked channel order")
    if set(raw.get_channel_types()) != {"eeg"}:
        raise RuntimeError(f"{source.name} contains a required channel not typed as EEG")
    raw.rename_channels(dict(LOCAL_EXP4_CHANNEL_ALIASES))
    if tuple(raw.ch_names) != LOCAL_EXP4_15_CHANNELS:
        raise RuntimeError(f"{source.name} legacy channel normalization failed")

    raw.filter(
        l_freq=float(LOCAL_EXP4_PREPROCESSING["fmin_hz"]),
        h_freq=float(LOCAL_EXP4_PREPROCESSING["fmax_hz"]),
        picks="eeg",
        method=str(LOCAL_EXP4_PREPROCESSING["filter_method"]),
        phase=str(LOCAL_EXP4_PREPROCESSING["filter_phase"]),
        fir_design=str(LOCAL_EXP4_PREPROCESSING["fir_design"]),
        l_trans_bandwidth=float(
            LOCAL_EXP4_PREPROCESSING["l_trans_bandwidth_hz"]
        ),
        h_trans_bandwidth=float(
            LOCAL_EXP4_PREPROCESSING["h_trans_bandwidth_hz"]
        ),
        pad="reflect_limited",
        verbose="ERROR",
    )
    target_sfreq = float(LOCAL_EXP4_PREPROCESSING["sfreq_hz"])
    raw.resample(target_sfreq, npad="auto", verbose="ERROR")
    raw.set_eeg_reference(ref_channels="average", projection=False, verbose="ERROR")

    events, labels = _local_exp4_task_events(raw, source)
    n_times = int(LOCAL_EXP4_PREPROCESSING["n_times"])
    tmin = float(LOCAL_EXP4_PREPROCESSING["tmin_seconds"])
    tmax = _inclusive_api_tmax(LOCAL_EXP4_PREPROCESSING)
    epochs = mne.Epochs(
        raw,
        events,
        event_id=None,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        proj=False,
        reject_by_annotation=False,
        event_repeated="error",
        verbose="ERROR",
    )
    values = epochs.get_data(copy=True).astype(np.float32, copy=False)
    if values.shape[:2] != (60, len(LOCAL_EXP4_15_CHANNELS)):
        raise RuntimeError(
            f"{source.name} produced shape {values.shape}; expected (60, 15, {n_times})"
        )
    values = _select_half_open_epoch_samples(
        values,
        epochs.times,
        contract=LOCAL_EXP4_PREPROCESSING,
        source=source.name,
    )
    if bool(LOCAL_EXP4_PREPROCESSING["epoch_demean"]):
        values = values - values.mean(axis=2, keepdims=True)
    values = np.ascontiguousarray(values, dtype=np.float32)
    if not np.all(np.isfinite(values)):
        raise RuntimeError(f"{source.name} contains non-finite EEG samples")

    channel_names, positions = _unit_positions(epochs.info)
    if channel_names != LOCAL_EXP4_15_CHANNELS:
        raise RuntimeError(
            f"{source.name} channel order {channel_names} does not match the local contract"
        )
    session_id = f"S{subject:02d}-R{run:02d}"
    stat = source.stat()
    return {
        "x": values,
        "y": labels,
        "positions": positions,
        "channel_names": channel_names,
        "sessions": np.asarray([session_id] * 60, dtype="U"),
        "runs": np.asarray([str(run)] * 60, dtype="U"),
        "source": {
            "subject": subject,
            "run": run,
            "filename": source.name,
            "size_bytes": int(stat.st_size),
            "sha256": _sha256_file(source),
        },
    }


def _build_local_exp4_subject(
    subject: int,
    root: Path,
) -> dict[str, Any]:
    """Build all four manifest runs for one local development subject."""

    runs = LOCAL_EXP4_SUBJECT_RUNS[subject]
    loaded = [_load_local_exp4_run(root, subject, run) for run in runs]
    positions = np.asarray(loaded[0]["positions"], dtype=np.float32)
    channel_names = tuple(loaded[0]["channel_names"])
    for item in loaded[1:]:
        if tuple(item["channel_names"]) != channel_names or not np.allclose(
            item["positions"], positions, rtol=0.0, atol=1e-6
        ):
            raise RuntimeError(f"local_exp4 S{subject} montage changes between runs")
    return {
        "x": np.ascontiguousarray(np.concatenate([item["x"] for item in loaded])),
        "y": np.concatenate([item["y"] for item in loaded]).astype(
            np.int64, copy=False
        ),
        "positions": positions,
        "channel_names": channel_names,
        "sessions": np.concatenate([item["sessions"] for item in loaded]),
        "runs": np.concatenate([item["runs"] for item in loaded]),
        "source_manifest": [item["source"] for item in loaded],
    }


def build_subject_cache(
    dataset: str,
    subject: int,
    *,
    cache_root: str | Path,
    stage: str = "development",
    montage_profile: str = DEFAULT_MONTAGE_PROFILE,
    overwrite: bool = False,
    local_data_root: str | Path | None = None,
) -> Path:
    """Materialize one subject without allowing accidental confirmation access.

    The stage guard is meaningful only when the underlying raw files were
    absent at registry creation. Cached BNCI/Cho cohorts are explicitly marked
    development-only in :mod:`benchmark.config`.
    """

    spec = dataset_spec(dataset)
    profile = validate_montage_profile(montage_profile)
    if subject not in spec.subjects:
        raise ValueError(f"subject {subject} is invalid for {dataset}")
    if stage not in {"development", "confirmation"}:
        raise ValueError("stage must be 'development' or 'confirmation'")
    permitted_subjects = (
        spec.development_subjects
        if stage == "development"
        else spec.confirmation_subjects
    )
    if subject not in permitted_subjects:
        cohort = "development" if stage == "confirmation" else "frozen confirmation"
        raise PermissionError(f"S{subject} belongs to the {cohort} cohort for {dataset}")

    # This profile preflight happens before path checks, raw-file access, or
    # MOABB construction. In particular, bipolar BNCI2014-004 can never enter
    # the point-electrode native-montage pretraining pool.
    requested_channels = channels_for_dataset(dataset, profile)

    output = _cache_path(Path(cache_root), dataset, subject, profile)
    if output.exists() and not overwrite:
        # Never silently trust a stale or corrupted file merely because its
        # path matches the requested cache key.
        load_subject_cache(
            dataset,
            subject,
            cache_root=cache_root,
            montage_profile=profile,
        )
        return output

    source_manifest: list[dict[str, Any]] | None = None
    if dataset == "local_exp4":
        local = _build_local_exp4_subject(
            subject,
            Path(local_data_root) if local_data_root is not None else LOCAL_EXP4_DATA_ROOT,
        )
        values = local["x"]
        labels = local["y"]
        positions = local["positions"]
        channel_names = local["channel_names"]
        sessions = local["sessions"]
        runs = local["runs"]
        source_manifest = local["source_manifest"]
    else:
        from moabb.paradigms import MotorImagery

        preprocessing = preprocessing_for_dataset(dataset)
        paradigm = MotorImagery(
            n_classes=spec.n_classes,
            events=list(spec.events),
            fmin=float(preprocessing["fmin_hz"]),
            fmax=float(preprocessing["fmax_hz"]),
            tmin=float(preprocessing["tmin_seconds"]),
            # MotorImagery/MNE interprets tmax inclusively. Request the boundary
            # timestamp, then explicitly select the verified half-open grid.
            tmax=_inclusive_api_tmax(preprocessing),
            resample=float(preprocessing["sfreq_hz"]),
            # None requests every dataset EEG channel from MOABB for the native
            # profile. Returned signals are neither interpolated nor subset.
            channels=(
                list(requested_channels)
                if requested_channels is not None
                else None
            ),
        )
        epochs, string_labels, metadata = paradigm.get_data(
            _dataset_instance(spec), subjects=[subject], return_epochs=True
        )
        values = epochs.get_data(copy=True).astype(np.float32, copy=False)
        values = _select_half_open_epoch_samples(
            values,
            epochs.times,
            contract=preprocessing,
            source=f"{dataset} S{subject}",
        )
        channel_types = tuple(str(value) for value in epochs.get_channel_types())
        if not channel_types or set(channel_types) != {"eeg"}:
            raise RuntimeError(
                f"{dataset} S{subject} returned non-EEG channels: {channel_types}"
            )
        label_index = {label: index for index, label in enumerate(spec.events)}
        try:
            labels = np.asarray(
                [label_index[str(label)] for label in string_labels], dtype=np.int64
            )
        except KeyError as exc:
            raise RuntimeError(f"unexpected label in {dataset} S{subject}: {exc}") from exc
        channel_names, positions = _unit_positions(epochs.info)
        if requested_channels is not None and channel_names != requested_channels:
            raise RuntimeError(
                f"{dataset} channel order {channel_names} does not match "
                f"{requested_channels}"
            )
        if profile == "native" and len(set(channel_names)) != len(channel_names):
            raise RuntimeError(f"{dataset} native channel names are not unique")
        # MNE/MOABB epoch arrays are stored in volts. The common-average operation is
        # symmetric under every configured left/right pair. BNCI2014-004 contains
        # supplied bipolar derivations and must not be re-referenced as point EEG.
        # Its nominal center coordinates must never be used as point-electrode
        # continuity, interpolation, or native cross-montage evidence.
        if dataset != "bnci2014_004":
            values = values - values.mean(axis=1, keepdims=True)
        values = values - values.mean(axis=2, keepdims=True)
        values = np.ascontiguousarray(values, dtype=np.float32)
        sessions = np.asarray(metadata["session"].astype(str).tolist(), dtype="U")
        runs = _canonical_run_ids(
            dataset,
            np.asarray(metadata["run"].astype(str).tolist(), dtype="U"),
        )

    if not (len(values) == len(labels) == len(sessions) == len(runs)):
        raise RuntimeError("trial, label, and metadata row counts differ")
    if set(labels.tolist()) != set(range(spec.n_classes)):
        raise RuntimeError("not every configured class is represented")
    if not np.all(np.isfinite(values)):
        raise RuntimeError("non-finite EEG samples")

    identity = {
        "dataset": asdict(spec),
        "subject": subject,
        "montage_profile": profile,
        "preprocessing": preprocessing_for_dataset(dataset),
        "coordinates": coordinate_contract_for_dataset(
            dataset,
            profile,
            tuple(channel_names),
        ),
        "shape": list(values.shape),
        "channels": list(channel_names),
    }
    if source_manifest is not None:
        identity["source_manifest"] = source_manifest
    digest = hashlib.sha256()
    for array in (values, labels, positions, sessions.astype("U"), runs.astype("U")):
        digest.update(np.ascontiguousarray(array).tobytes())
    identity["array_sha256"] = digest.hexdigest()

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".npz.partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            x=values,
            y=labels,
            positions=positions,
            channel_names=np.asarray(channel_names),
            sessions=sessions,
            runs=runs,
            identity=_json_scalar(identity),
        )
    temporary.replace(output)
    return output


def load_subject_cache(
    dataset: str,
    subject: int,
    *,
    cache_root: str | Path,
    montage_profile: str = DEFAULT_MONTAGE_PROFILE,
) -> dict[str, Any]:
    profile = validate_montage_profile(montage_profile)
    requested_channels = channels_for_dataset(dataset, profile)
    path = _cache_path(Path(cache_root), dataset, subject, profile)
    payload = _read_unique_regular_bytes(path)
    expected_members = _validate_npz_member_contract(
        payload,
        expected_members=SUBJECT_CACHE_NPZ_MEMBERS,
        source=str(path),
    )
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        if tuple(archive.files) != expected_members:
            raise RuntimeError(
                f"cache {path} NumPy members differ from the raw ZIP contract"
            )
        result = {key: archive[key].copy() for key in expected_members}
    result["identity"] = json.loads(str(result["identity"].item()))
    identity = result["identity"]
    if tuple(result) != SUBJECT_CACHE_NPZ_MEMBERS:
        raise RuntimeError(
            f"cache {path} fields differ from the frozen schema: {sorted(result)}"
        )
    if not isinstance(identity, dict):
        raise RuntimeError(f"cache identity is not an object for {dataset} S{subject}")
    expected_identity_keys = {
        "array_sha256",
        "channels",
        "coordinates",
        "dataset",
        "montage_profile",
        "preprocessing",
        "shape",
        "subject",
    }
    if dataset == "local_exp4":
        expected_identity_keys.add("source_manifest")
    if set(identity) != expected_identity_keys:
        raise RuntimeError(
            f"cache identity fields differ from the frozen schema for "
            f"{dataset} S{subject}: {sorted(identity)}"
        )
    if (
        not isinstance(identity["array_sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", identity["array_sha256"]) is None
        or not isinstance(identity["channels"], list)
        or any(not isinstance(value, str) for value in identity["channels"])
        or not isinstance(identity["shape"], list)
        or len(identity["shape"]) != 3
        or any(type(value) is not int or value <= 0 for value in identity["shape"])
        or not isinstance(identity["montage_profile"], str)
    ):
        raise RuntimeError(
            f"cache identity value types are invalid for {dataset} S{subject}"
        )
    expected_dataset_identity = json.loads(
        json.dumps(asdict(dataset_spec(dataset)), sort_keys=True)
    )
    dataset_identity = identity.get("dataset")
    if (
        dataset_identity != expected_dataset_identity
        or type(identity.get("subject")) is not int
        or identity["subject"] != subject
    ):
        raise RuntimeError(f"cache identity does not match {dataset} S{subject}")
    if identity.get("montage_profile") != profile:
        raise RuntimeError(f"cache montage profile is stale for {dataset} S{subject}")
    expected_preprocessing = preprocessing_for_dataset(dataset)
    if identity.get("preprocessing") != expected_preprocessing:
        raise RuntimeError(f"cache preprocessing contract is stale for {dataset} S{subject}")
    channel_names = tuple(str(value) for value in result["channel_names"].tolist())
    if not channel_names or len(set(channel_names)) != len(channel_names):
        raise RuntimeError(f"cache channel contract is stale for {dataset} S{subject}")
    if requested_channels is not None and channel_names != requested_channels:
        raise RuntimeError(f"cache channel contract is stale for {dataset} S{subject}")
    if identity.get("channels") != list(channel_names):
        raise RuntimeError(f"cache ordered channels are stale for {dataset} S{subject}")
    expected_coordinates = coordinate_contract_for_dataset(
        dataset,
        profile,
        channel_names,
    )
    if identity.get("coordinates") != expected_coordinates:
        raise RuntimeError(f"cache coordinate contract is stale for {dataset} S{subject}")
    if dataset == "local_exp4":
        source_manifest = identity["source_manifest"]
        expected_runs = LOCAL_EXP4_SUBJECT_RUNS[subject]
        if (
            not isinstance(source_manifest, list)
            or len(source_manifest) != len(expected_runs)
        ):
            raise RuntimeError(
                f"cache source manifest is invalid for {dataset} S{subject}"
            )
        for source, expected_run in zip(
            source_manifest, expected_runs, strict=True
        ):
            if not isinstance(source, dict) or set(source) != {
                "filename",
                "run",
                "sha256",
                "size_bytes",
                "subject",
            }:
                raise RuntimeError(
                    f"cache source manifest schema is invalid for "
                    f"{dataset} S{subject}"
                )
            if (
                type(source["run"]) is not int
                or source["run"] != expected_run
                or type(source["subject"]) is not int
                or source["subject"] != subject
                or type(source["size_bytes"]) is not int
                or source["size_bytes"] <= 0
                or not isinstance(source["filename"], str)
                or source["filename"]
                != f"exp4_subject{subject}_training_{expected_run}_mi_raw.fif"
                or not isinstance(source["sha256"], str)
                or re.fullmatch(r"[0-9a-f]{64}", source["sha256"]) is None
            ):
                raise RuntimeError(
                    f"cache source manifest values are invalid for "
                    f"{dataset} S{subject}"
                )
    if list(result["x"].shape) != identity.get("shape"):
        raise RuntimeError(f"cache shape identity is stale for {dataset} S{subject}")
    expected_n_times = int(expected_preprocessing["n_times"])
    if result["x"].ndim != 3 or result["x"].shape[1:] != (
        len(channel_names),
        expected_n_times,
    ):
        raise RuntimeError(f"cache EEG shape is invalid for {dataset} S{subject}")
    if result["x"].dtype != np.float32 or result["positions"].dtype != np.float32:
        raise RuntimeError(f"cache floating dtypes are invalid for {dataset} S{subject}")
    if result["y"].dtype != np.int64:
        raise RuntimeError(f"cache label dtype is invalid for {dataset} S{subject}")
    if result["positions"].shape != (len(channel_names), 3):
        raise RuntimeError(f"cache coordinate shape is invalid for {dataset} S{subject}")
    expected_positions = _atlas_unit_positions(channel_names)
    if not np.allclose(
        result["positions"], expected_positions, rtol=0.0, atol=1e-7
    ):
        raise RuntimeError(
            f"cache coordinates do not match standard_1005 for {dataset} S{subject}"
        )
    if not np.allclose(
        np.linalg.norm(result["positions"], axis=1),
        1.0,
        rtol=0.0,
        atol=1e-6,
    ):
        raise RuntimeError(f"cache coordinates are not unit vectors for {dataset} S{subject}")
    trial_count = result["x"].shape[0]
    if not all(
        len(result[key]) == trial_count for key in ("y", "sessions", "runs")
    ):
        raise RuntimeError(f"cache row counts differ for {dataset} S{subject}")
    expected_labels = set(range(dataset_spec(dataset).n_classes))
    if set(result["y"].tolist()) != expected_labels:
        raise RuntimeError(f"cache labels are invalid for {dataset} S{subject}")
    if not np.all(np.isfinite(result["x"])) or not np.all(
        np.isfinite(result["positions"])
    ):
        raise RuntimeError(f"cache contains non-finite values for {dataset} S{subject}")
    digest = hashlib.sha256()
    for key in ("x", "y", "positions", "sessions", "runs"):
        array = result[key]
        if key in {"sessions", "runs"}:
            array = array.astype("U")
        digest.update(np.ascontiguousarray(array).tobytes())
    if digest.hexdigest() != identity.get("array_sha256"):
        raise RuntimeError(f"cache array digest failed for {dataset} S{subject}")
    return result


def split_indices(
    dataset: str,
    labels: IntArray,
    sessions: NDArray[np.str_],
    runs: NDArray[np.str_],
    *,
    fold: int = 0,
    seed: int = 20260719,
    subject: int | None = None,
) -> tuple[IntArray, IntArray, IntArray]:
    """Return source-train, source-validation, and prediction-only test rows."""

    labels = np.asarray(labels, dtype=np.int64)
    sessions = np.asarray(sessions).astype(str)
    runs = np.asarray(runs).astype(str)
    rows = np.arange(len(labels), dtype=np.int64)
    if not (len(labels) == len(sessions) == len(runs)):
        raise ValueError("split arrays have different lengths")

    if dataset == "local_exp4":
        if subject is None:
            raise ValueError("local_exp4 splits require an explicit subject ID")
        subject = int(subject)
        if subject not in LOCAL_EXP4_SUBJECT_RUNS:
            raise ValueError(f"subject {subject} is invalid for local_exp4")
        if fold != 0:
            raise ValueError("local_exp4 has only the fixed chronological fold 0")
        expected_runs = tuple(str(run) for run in LOCAL_EXP4_SUBJECT_RUNS[subject])
        observed_runs = set(runs.tolist())
        if observed_runs != set(expected_runs):
            raise RuntimeError(
                f"local_exp4 S{subject} requires runs {list(expected_runs)}; "
                f"found {sorted(observed_runs)}"
            )
        if len(labels) != 240:
            raise RuntimeError(
                f"local_exp4 S{subject} has {len(labels)} trials; expected 240"
            )
        for run in expected_runs:
            run_rows = rows[runs == run]
            expected_session = f"S{subject:02d}-R{int(run):02d}"
            if len(run_rows) != 60 or set(sessions[run_rows].tolist()) != {
                expected_session
            }:
                raise RuntimeError(
                    f"local_exp4 S{subject} run {run} must be one 60-trial session"
                )
            if np.bincount(labels[run_rows], minlength=2).tolist() != [30, 30]:
                raise RuntimeError(
                    f"local_exp4 S{subject} run {run} must contain 30 trials per class"
                )
        train = rows[np.isin(runs, expected_runs[:2])]
        validation = rows[runs == expected_runs[2]]
        test = rows[runs == expected_runs[3]]
    elif dataset == "bnci2014_001":
        source = rows[sessions == "0train"]
        test = rows[sessions == "1test"]
        source_runs = sorted(set(runs[source].tolist()))
        if len(source_runs) < 2:
            raise RuntimeError("BNCI2014-001 source session has too few runs")
        validation_run = source_runs[-1]
        validation = source[runs[source] == validation_run]
        train = source[runs[source] != validation_run]
    elif dataset == "bnci2014_004":
        train = rows[np.isin(sessions, ("0train", "1train"))]
        validation = rows[sessions == "2train"]
        test = rows[np.isin(sessions, ("3test", "4test"))]
    elif dataset == "cho2017":
        if not 0 <= fold < 5:
            raise ValueError("Cho2017 fold must be in [0, 4]")
        del seed
        # MOABB exposes Cho as one run and orders many trials contiguously by
        # class. Random trial folds can mix temporally adjacent acquisitions.
        # Split each class sequence into five contiguous acquisition blocks,
        # then rotate whole blocks through test and validation roles.
        labels_present = sorted(set(labels.tolist()))
        class_blocks: dict[int, list[IntArray]] = {}
        for label in labels_present:
            class_rows = rows[labels == label]
            if len(class_rows) < 5:
                raise RuntimeError("Cho2017 class has too few trials for five blocks")
            class_blocks[label] = [
                np.asarray(block, dtype=np.int64)
                for block in np.array_split(class_rows, 5)
            ]
        validation_fold = (fold + 1) % 5
        test = np.sort(
            np.concatenate([class_blocks[label][fold] for label in labels_present])
        )
        validation = np.sort(
            np.concatenate(
                [class_blocks[label][validation_fold] for label in labels_present]
            )
        )
        held_out = np.concatenate((test, validation))
        train = rows[~np.isin(rows, held_out)]
    elif dataset == "physionet_mi":
        if subject is None:
            raise ValueError(
                "PhysionetMI development splits require an explicit subject ID"
            )
        spec = dataset_spec(dataset)
        if int(subject) not in spec.development_subjects:
            raise PermissionError(
                f"PhysionetMI S{subject} is outside the development cohort S1-S54"
            )
        if not 0 <= fold < len(PHYSIONET_MI_RUNS):
            raise ValueError("PhysionetMI fold must be in [0, 2]")
        observed_runs = set(runs.tolist())
        expected_runs = set(PHYSIONET_MI_RUNS)
        if observed_runs != expected_runs:
            raise RuntimeError(
                "PhysionetMI split requires exactly source imagery runs 4/8/12; "
                f"found {sorted(observed_runs)}"
            )

        # Rotate each complete acquisition run through every role.  No trial
        # from the held-out run can influence checkpoint selection, and the
        # validation run is also absent from optimization rows.
        test_run = PHYSIONET_MI_RUNS[fold]
        validation_run = PHYSIONET_MI_RUNS[(fold + 1) % len(PHYSIONET_MI_RUNS)]
        train_run = PHYSIONET_MI_RUNS[(fold + 2) % len(PHYSIONET_MI_RUNS)]
        train = rows[runs == train_run]
        validation = rows[runs == validation_run]
        test = rows[runs == test_run]
    else:
        raise ValueError(f"no split protocol for {dataset!r}")

    for first, second in ((train, validation), (train, test), (validation, test)):
        if np.intersect1d(first, second).size:
            raise RuntimeError("train/validation/test overlap")
    if len(train) + len(validation) + len(test) != len(labels):
        raise RuntimeError("split does not cover every trial exactly once")
    for name, indices in (("train", train), ("validation", validation), ("test", test)):
        if set(labels[indices].tolist()) != set(labels.tolist()):
            raise RuntimeError(f"{name} split does not contain every class")
    return train, validation, test


def fit_channel_scaler(
    x_train: FloatArray,
    channel_names: tuple[str, ...] | list[str] | None = None,
) -> tuple[FloatArray, FloatArray]:
    if x_train.ndim != 3:
        raise ValueError("x_train must have shape (trials, channels, time)")
    mean = x_train.mean(axis=(0, 2), keepdims=True, dtype=np.float64).astype(np.float32)
    std = x_train.std(axis=(0, 2), keepdims=True, dtype=np.float64).astype(np.float32)
    if channel_names is not None:
        names = tuple(channel_names)
        if len(names) != x_train.shape[1]:
            raise ValueError("channel_names length does not match x_train")
        indices = {name: index for index, name in enumerate(names)}
        for left, right in REFLECTION_PAIRS:
            if left not in indices or right not in indices:
                continue
            pair = (indices[left], indices[right])
            pooled_values = x_train[:, pair, :].astype(np.float64, copy=False)
            pooled_mean = np.float32(pooled_values.mean())
            pooled_std = np.float32(pooled_values.std())
            mean[:, pair, :] = pooled_mean
            std[:, pair, :] = pooled_std
    # EEG arrays are stored in volts.  An absolute 1e-4 floor is 100 uV and
    # therefore suppresses ordinary BNCI recordings (typically 2--10 uV)
    # instead of standardising them, while leaving higher-amplitude cohorts
    # such as Cho scaled differently.  Ten nanovolts is safely above a
    # numerical zero channel but well below physiological EEG variation.
    std = np.maximum(std, np.float32(CHANNEL_SCALING["epsilon_volts"]))
    return mean, std


def apply_channel_scaler(
    x: FloatArray,
    mean: FloatArray,
    std: FloatArray,
    *,
    clip: float = float(CHANNEL_SCALING["clip_standard_deviations"]),
) -> FloatArray:
    result = (np.asarray(x, dtype=np.float32) - mean) / std
    return np.clip(result, -clip, clip).astype(np.float32, copy=False)


def reflection_index(channel_names: tuple[str, ...] | list[str]) -> IntArray:
    """Return the exact sagittal partner permutation for a symmetric montage."""

    names = tuple(channel_names)
    indices = {name: index for index, name in enumerate(names)}
    partner = {name: name for name in names}
    for left, right in REFLECTION_PAIRS:
        if left in indices and right in indices:
            partner[left] = right
            partner[right] = left
        elif left in indices or right in indices:
            raise ValueError(f"montage contains only one member of reflection pair {left}/{right}")
    result = np.asarray([indices[partner[name]] for name in names], dtype=np.int64)
    if not np.array_equal(result[result], np.arange(len(names))):
        raise RuntimeError("reflection map is not an involution")
    return result


def _parse_subject_list(value: str) -> tuple[int, ...]:
    """Parse one explicit comma-separated subject declaration."""

    tokens = tuple(token.strip() for token in value.split(","))
    if not tokens or any(not token for token in tokens):
        raise ValueError("subjects must be a non-empty comma-separated list")
    try:
        subjects = tuple(int(token) for token in tokens)
    except ValueError as exc:
        raise ValueError("subjects must contain integers only") from exc
    if any(subject <= 0 for subject in subjects):
        raise ValueError("subjects must be positive integers")
    if len(set(subjects)) != len(subjects):
        raise ValueError("subjects must not contain duplicates")
    return subjects


def build_development_caches(
    dataset: str,
    subjects: Sequence[int],
    *,
    cache_root: str | Path,
    montage_profile: str = DEFAULT_MONTAGE_PROFILE,
    overwrite: bool = False,
    local_data_root: str | Path | None = None,
) -> tuple[Path, ...]:
    """Build an explicitly declared development list after all-or-none preflight.

    Every subject is checked before the first cache or raw-data access. This
    development-only entry point has no confirmation mode.
    """

    spec = dataset_spec(dataset)
    profile = validate_montage_profile(montage_profile)
    declared = tuple(int(subject) for subject in subjects)
    if not declared:
        raise ValueError("at least one development subject must be declared")
    if len(set(declared)) != len(declared):
        raise ValueError("development subject list contains duplicates")
    invalid = tuple(subject for subject in declared if subject not in spec.subjects)
    if invalid:
        raise ValueError(f"invalid subjects for {dataset}: {list(invalid)}")
    confirmation = tuple(
        subject for subject in declared if subject in spec.confirmation_subjects
    )
    if confirmation:
        raise PermissionError(
            f"development cache CLI refuses confirmation subjects for {dataset}: "
            f"{list(confirmation)}"
        )
    outside_development = tuple(
        subject for subject in declared if subject not in spec.development_subjects
    )
    if outside_development:
        raise PermissionError(
            f"subjects are outside the development cohort for {dataset}: "
            f"{list(outside_development)}"
        )
    # Validate dataset/profile eligibility once before the first subject build.
    channels_for_dataset(dataset, profile)
    return tuple(
        build_subject_cache(
            dataset,
            subject,
            cache_root=cache_root,
            stage="development",
            montage_profile=profile,
            overwrite=overwrite,
            local_data_root=local_data_root,
        )
        for subject in declared
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Development-only ``python -m benchmark.data`` cache builder."""

    parser = argparse.ArgumentParser(
        description="Build declared development EEG caches; confirmation is forbidden."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--subjects", required=True, help="comma-separated subject IDs")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument(
        "--montage-profile",
        choices=MONTAGE_PROFILES,
        default=DEFAULT_MONTAGE_PROFILE,
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--local-data-root", type=Path)
    args = parser.parse_args(argv)
    try:
        subjects = _parse_subject_list(args.subjects)
        outputs = build_development_caches(
            args.dataset,
            subjects,
            cache_root=args.cache_root,
            montage_profile=args.montage_profile,
            overwrite=bool(args.overwrite),
            local_data_root=args.local_data_root,
        )
    except (PermissionError, ValueError) as exc:
        parser.error(str(exc))
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
