"""Locked dataset definitions for the multi-dataset MI study."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType


@dataclass(frozen=True)
class DatasetSpec:
    """One harmonized public motor-imagery cohort."""

    key: str
    moabb_class: str
    subjects: tuple[int, ...]
    development_subjects: tuple[int, ...]
    confirmation_subjects: tuple[int, ...]
    events: tuple[str, ...]
    n_classes: int
    protocol: str


DATASETS: dict[str, DatasetSpec] = {
    # The in-house Exp4 recordings have already been used throughout decoder
    # development.  They are therefore a development/replication cohort, never
    # confirmation evidence.  Unlike the public cohorts, these files are loaded
    # directly from the repository through an explicit subject/run manifest.
    "local_exp4": DatasetSpec(
        key="local_exp4",
        moabb_class="",
        subjects=(1, 3, 4, 5, 6, 7, 8, 10),
        development_subjects=(1, 3, 4, 5, 6, 7, 8, 10),
        confirmation_subjects=(),
        events=("left_hand", "right_hand"),
        n_classes=2,
        protocol="fixed_chronological_recording_holdout",
    ),
    # Four-class official cross-session benchmark. Earlier repository work has
    # already opened all nine subjects, so this is development evidence only.
    "bnci2014_001": DatasetSpec(
        key="bnci2014_001",
        moabb_class="BNCI2014_001",
        subjects=tuple(range(1, 10)),
        development_subjects=tuple(range(1, 10)),
        confirmation_subjects=(),
        events=("left_hand", "right_hand", "feet", "tongue"),
        n_classes=4,
        protocol="official_session_holdout",
    ),
    # Binary official chronological benchmark. Earlier repository work has
    # already opened every subject, so this is development evidence only.
    "bnci2014_004": DatasetSpec(
        key="bnci2014_004",
        moabb_class="BNCI2014_004",
        subjects=tuple(range(1, 10)),
        development_subjects=tuple(range(1, 10)),
        confirmation_subjects=(),
        events=("left_hand", "right_hand"),
        n_classes=2,
        protocol="official_session_holdout",
    ),
    # All 52 subjects have appeared in earlier CAMEO experiments. They provide
    # a large development benchmark but are not a new confirmation population.
    "cho2017": DatasetSpec(
        key="cho2017",
        moabb_class="Cho2017",
        subjects=tuple(range(1, 53)),
        development_subjects=tuple(range(1, 53)),
        confirmation_subjects=(),
        events=("left_hand", "right_hand"),
        n_classes=2,
        protocol="rotating_within_class_acquisition_order_block_holdout",
    ),
    # This dataset was absent at study creation. Only S1--54 may be used for
    # development; S55--109 remain inaccessible until model/configuration
    # freeze. Subject 88 has a different source sample rate but is valid because
    # every recording is independently resampled by the harmonizer.
    "physionet_mi": DatasetSpec(
        key="physionet_mi",
        moabb_class="PhysionetMI",
        subjects=tuple(range(1, 110)),
        development_subjects=tuple(range(1, 55)),
        confirmation_subjects=tuple(range(55, 110)),
        events=("left_hand", "right_hand"),
        n_classes=2,
        protocol="rotating_run_holdout",
    ),
    # OpenBMI/Lee2019 remains wholly dataset-held-out. Its large raw files must
    # be streamed one subject at a time during confirmation rather than bulk
    # downloaded onto the 123 GB GPU volume.
    "lee2019_mi": DatasetSpec(
        key="lee2019_mi",
        moabb_class="Lee2019_MI",
        subjects=tuple(range(1, 55)),
        development_subjects=(),
        confirmation_subjects=tuple(range(1, 55)),
        events=("left_hand", "right_hand"),
        n_classes=2,
        protocol="official_session_holdout",
    ),
    # Entire unopened datasets are held out as dataset-level confirmations.
    "weibo2014": DatasetSpec(
        key="weibo2014",
        moabb_class="Weibo2014",
        subjects=tuple(range(1, 11)),
        development_subjects=(),
        confirmation_subjects=tuple(range(1, 11)),
        events=("left_hand", "right_hand"),
        n_classes=2,
        protocol="rotating_run_holdout",
    ),
    "zhou2016": DatasetSpec(
        key="zhou2016",
        moabb_class="Zhou2016",
        subjects=tuple(range(1, 5)),
        development_subjects=(),
        confirmation_subjects=tuple(range(1, 5)),
        events=("left_hand", "right_hand"),
        n_classes=2,
        protocol="session_holdout",
    ),
}


PREPROCESSING = {
    "schema": "eeg-mi-cache-v2",
    "fmin_hz": 4.0,
    "fmax_hz": 40.0,
    "tmin_seconds": 0.5,
    "tmax_seconds_exclusive": 3.0,
    "epoch_interval_semantics": "half-open [tmin_seconds, tmax_seconds_exclusive)",
    "inclusive_api_endpoint": (
        "request tmax_seconds_exclusive, then select only timestamps in the "
        "declared half-open interval"
    ),
    "sfreq_hz": 128.0,
    "n_times": 320,
    "labels": "ordered exactly as DatasetSpec.events",
    "reference": "symmetric common average except supplied bipolar BNCI2014-004",
    "epoch_demean": True,
}

# MOABB reports ``Epochs.times`` relative to the beginning of each dataset's
# declared acquisition interval, while ``MotorImagery.tmin``/``tmax`` are
# offsets inside that interval.  Freeze these upstream interval starts instead
# of inferring an origin from returned data.  The values are public dataset
# metadata and do not require opening a subject recording.
MOABB_INTERVAL_START_SECONDS = MappingProxyType(
    {
        "bnci2014_001": 2.0,
        "bnci2014_004": 3.0,
        "cho2017": 0.0,
        "physionet_mi": 0.0,
        "lee2019_mi": 0.0,
        "weibo2014": 3.0,
        "zhou2016": 0.0,
    }
)


# Scaling is fitted independently inside every source-training partition and
# is therefore recorded separately from cache-time preprocessing.
CHANNEL_SCALING = {
    "schema": "eeg-mi-channel-scaling-v1",
    "fit_rows": {
        "selection": "training only",
        "final_refit": "training plus validation; test excluded",
    },
    "location": "per channel mean over the declared fitting rows and time",
    "scale": "per channel standard deviation over the declared fitting rows and time",
    "reflected_pairs_pooled": True,
    "epsilon_volts": 1e-8,
    "clip_standard_deviations": 12.0,
}


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_EXP4_DATA_ROOT = PROJECT_ROOT / "data"

# This is deliberately a literal manifest.  The source directory also contains
# excluded S9 recordings and invalid S10 runs 1--4, so filesystem discovery is
# not an acceptable way to construct the research cohort.
LOCAL_EXP4_SUBJECT_RUNS = MappingProxyType(
    {
        1: (1, 2, 3, 4),
        3: (1, 2, 3, 4),
        4: (1, 2, 3, 4),
        5: (1, 2, 3, 4),
        6: (1, 2, 3, 4),
        7: (1, 2, 3, 4),
        8: (1, 2, 3, 4),
        10: (5, 6, 7, 8),
    }
)

# The public 0.5--3.0 s window is not valid for Exp4: its task phase ends at
# about 2.1 s and the remaining samples are rest.  Keep every other harmonized
# operation, but use the actual deployment task interval as a half-open window.
LOCAL_EXP4_PREPROCESSING = {
    "schema": PREPROCESSING["schema"],
    "source_sfreq_hz": 125.0,
    "fmin_hz": 4.0,
    "fmax_hz": 40.0,
    "filter_method": "fir",
    "filter_phase": "zero",
    "fir_design": "firwin",
    "l_trans_bandwidth_hz": 2.0,
    "h_trans_bandwidth_hz": 5.0,
    "tmin_seconds": 0.0,
    "tmax_seconds_exclusive": 2.0,
    "sfreq_hz": 128.0,
    "n_times": 256,
    "labels": PREPROCESSING["labels"],
    "reference": "common average over all 15 point-EEG channels",
    "epoch_demean": True,
    "artifact_rejection": None,
}


CANONICAL_21_CHANNELS: tuple[str, ...] = (
    "Fz",
    "FC3",
    "FC4",
    "FC1",
    "FC2",
    "C5",
    "C6",
    "C3",
    "C4",
    "C1",
    "C2",
    "Cz",
    "CP3",
    "CP4",
    "CP1",
    "CP2",
    "CPz",
    "P1",
    "P2",
    "Pz",
    "POz",
)

ZHOU_8_CHANNELS: tuple[str, ...] = (
    "FC3",
    "FC4",
    "C3",
    "C4",
    "Cz",
    "CP3",
    "CP4",
    "CPz",
)

# BNCI2014-004 supplies bipolar derivations rather than three independent
# point-electrode signals. Their configured standard_1005 coordinates are only
# nominal derivation centers needed for a common tensor interface. Results on
# this cohort must never be presented as point-electrode continuity,
# interpolation, or native cross-montage evidence.
BNCI004_3_CHANNELS: tuple[str, ...] = ("C3", "Cz", "C4")

# Modern 10-10 names are used in cache metadata.  The acquisition files use
# the equivalent legacy names T5/T6 and T3/T4, which are renamed at load time.
LOCAL_EXP4_SOURCE_CHANNELS: tuple[str, ...] = (
    "Cz",
    "Pz",
    "C3",
    "C4",
    "T5",
    "T6",
    "Fz",
    "F7",
    "F8",
    "F3",
    "F4",
    "T3",
    "T4",
    "P3",
    "P4",
)

LOCAL_EXP4_15_CHANNELS: tuple[str, ...] = (
    "Cz",
    "Pz",
    "C3",
    "C4",
    "P7",
    "P8",
    "Fz",
    "F7",
    "F8",
    "F3",
    "F4",
    "T7",
    "T8",
    "P3",
    "P4",
)

LOCAL_EXP4_CHANNEL_ALIASES = MappingProxyType(
    {"T5": "P7", "T6": "P8", "T3": "T7", "T4": "T8"}
)

MONTAGE_PROFILES: tuple[str, ...] = ("harmonized", "native")
DEFAULT_MONTAGE_PROFILE = "harmonized"

REFLECTION_PAIRS: tuple[tuple[str, str], ...] = (
    ("F7", "F8"),
    ("F3", "F4"),
    ("FC3", "FC4"),
    ("FC1", "FC2"),
    ("C5", "C6"),
    ("C3", "C4"),
    ("C1", "C2"),
    ("T7", "T8"),
    ("CP3", "CP4"),
    ("CP1", "CP2"),
    ("P1", "P2"),
    ("P3", "P4"),
    ("P7", "P8"),
)


def validate_montage_profile(profile: str) -> str:
    value = str(profile).lower().strip()
    if value not in MONTAGE_PROFILES:
        raise ValueError(
            f"unknown montage profile {profile!r}; expected one of {MONTAGE_PROFILES}"
        )
    return value


def channels_for_dataset(
    key: str,
    montage_profile: str = DEFAULT_MONTAGE_PROFILE,
) -> tuple[str, ...] | None:
    """Return requested channels, or ``None`` for every native public EEG channel."""

    dataset_spec(key)
    profile = validate_montage_profile(montage_profile)
    if profile == "native":
        if key == "bnci2014_004":
            raise PermissionError(
                "BNCI2014-004 bipolar derivations are ineligible for native-montage "
                "point-electrode pretraining"
            )
        if key == "local_exp4":
            return LOCAL_EXP4_15_CHANNELS
        # Passing None to MOABB requests its complete dataset EEG channel set.
        # No interpolation or signal-channel subsetting is permitted afterward.
        return None
    if key == "local_exp4":
        return LOCAL_EXP4_15_CHANNELS
    if key == "bnci2014_004":
        return BNCI004_3_CHANNELS
    if key == "zhou2016":
        return ZHOU_8_CHANNELS
    return CANONICAL_21_CHANNELS


def preprocessing_for_dataset(key: str) -> dict[str, object]:
    """Return the frozen cache contract for one registered dataset."""

    dataset_spec(key)
    if key == "local_exp4":
        return dict(LOCAL_EXP4_PREPROCESSING)
    contract = dict(PREPROCESSING)
    contract["moabb_interval_start_seconds"] = float(
        MOABB_INTERVAL_START_SECONDS[key]
    )
    contract["returned_epoch_times_reference"] = (
        "MOABB dataset interval start; tmin/tmax are offsets within that interval"
    )
    return contract


def coordinate_contract_for_dataset(
    key: str,
    montage_profile: str = DEFAULT_MONTAGE_PROFILE,
    channel_names: tuple[str, ...] | None = None,
) -> dict[str, object]:
    """Return the frozen coordinate meaning for one configured cohort.

    Native recording coordinates are intentionally outside the current cache
    schema. Every configured channel name is looked up in one MNE
    ``standard_1005`` montage, transformed to MNE's head frame by
    ``Info.set_montage``, and normalized to a unit vector.
    """

    dataset_spec(key)
    profile = validate_montage_profile(montage_profile)
    requested = channels_for_dataset(key, profile)
    if channel_names is None:
        if requested is None:
            raise ValueError(
                f"native coordinate contract for {key} requires its observed channel order"
            )
        ordered_channels = requested
    else:
        ordered_channels = tuple(channel_names)
        if not ordered_channels or len(set(ordered_channels)) != len(ordered_channels):
            raise ValueError("coordinate contract channels must be non-empty and unique")
        if requested is not None and ordered_channels != requested:
            raise ValueError(
                f"{profile} channel order {ordered_channels} does not match {requested}"
            )
    contract: dict[str, object] = {
        "schema": "eeg-mi-coordinates-v1",
        "atlas": "MNE standard_1005",
        "lookup": "configured canonical channel name",
        "coordinate_frame": "MNE head",
        "normalization": "unit Euclidean vector after Info.set_montage",
        "native_recording_coordinates_used": False,
        "montage_profile": profile,
        "ordered_channels": list(ordered_channels),
        "channel_semantics": "nominal point-electrode locations",
        "point_electrode_continuity_evidence": True,
        "signal_channel_selection": (
            "locked harmonized channel list"
            if profile == "harmonized"
            else "all native dataset EEG channels; no interpolation or subsetting"
        ),
    }
    if key == "bnci2014_004":
        contract.update(
            {
                "channel_semantics": "nominal centers of supplied bipolar derivations",
                "point_electrode_continuity_evidence": False,
                "restriction": (
                    "must not be used as point-electrode continuity, interpolation, "
                    "or native cross-montage evidence"
                ),
            }
        )
    return contract


def dataset_spec(key: str) -> DatasetSpec:
    try:
        return DATASETS[key]
    except KeyError as exc:
        raise ValueError(f"unknown dataset {key!r}; expected one of {sorted(DATASETS)}") from exc
