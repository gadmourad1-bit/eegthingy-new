"""Immutable data and preprocessing configuration for the EEG experiments.

The manifest in this module is deliberately explicit.  In particular, subject 9 is
not a valid research subject and the first four recordings from subject 10 are not
silently rediscovered by a glob.  Keeping the exclusions here makes every training
and evaluation entry point use the same cohort.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Final


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
CACHE_DIR: Final[Path] = PROJECT_ROOT / "deepnet" / "cache"

VALID_SUBJECTS: Final[tuple[int, ...]] = (1, 3, 4, 5, 6, 7, 8, 10)
SUBJECTS = VALID_SUBJECTS  # concise backwards-compatible name for experiment code

_STANDARD_RUNS = (1, 2, 3, 4)
SUBJECT_RUNS = MappingProxyType(
    {
        1: _STANDARD_RUNS,
        3: _STANDARD_RUNS,
        4: _STANDARD_RUNS,
        5: _STANDARD_RUNS,
        6: _STANDARD_RUNS,
        7: _STANDARD_RUNS,
        8: _STANDARD_RUNS,
        10: (5, 6, 7, 8),
    }
)
VALID_SESSION_MANIFEST: Final[tuple[tuple[int, int], ...]] = tuple(
    (subject, run) for subject in VALID_SUBJECTS for run in SUBJECT_RUNS[subject]
)

SFREQ: Final[float] = 125.0
CHANNELS: Final[tuple[str, ...]] = (
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
EEG_CHANNELS = CHANNELS
BANDS: Final[tuple[tuple[float, float], ...]] = (
    (8.0, 12.0),
    (11.0, 15.0),
    (14.0, 20.0),
    (20.0, 30.0),
)
FB_BANDS = BANDS

# The deployment window begins at task-cue onset and matches the online decoder.
# MNE includes both endpoints, so 0.0--2.0 s at 125 Hz contains 251 samples, the
# same tensor length as legacy 0.5--2.5 s and the currently exported live model.
# ``legacy`` is retained only for direct comparison with existing paper numbers.
EPOCH_WINDOWS = MappingProxyType(
    {
        "deployment": (0.0, 2.0),
        "legacy": (0.5, 2.5),
    }
)
DEFAULT_WINDOW: Final[str] = "deployment"

LEFT_LABEL: Final[int] = 0
RIGHT_LABEL: Final[int] = 1
NEUTRAL_LABEL: Final[int] = -1
TASK_LABELS = MappingProxyType({"left_hand": LEFT_LABEL, "right_hand": RIGHT_LABEL})
LABEL_NAMES = MappingProxyType(
    {LEFT_LABEL: "left_hand", RIGHT_LABEL: "right_hand", NEUTRAL_LABEL: "neutral"}
)


@dataclass(frozen=True)
class DataConfig:
    """Settings that fully determine cached epochs and covariance matrices."""

    data_dir: Path = DATA_DIR
    cache_dir: Path = CACHE_DIR
    sfreq: float = SFREQ
    channels: tuple[str, ...] = CHANNELS
    bands: tuple[tuple[float, float], ...] = BANDS
    window_name: str = DEFAULT_WINDOW
    include_rest: bool = False
    artifact_threshold: float | None = 100e-6
    artifact_band: tuple[float, float] = (8.0, 30.0)
    covariance_shrinkage: float = 1e-3
    covariance_demean: bool = True
    dtype: str = "float32"
    filter_method: str = "fir"
    filter_phase: str = "minimum"
    fir_design: str = "firwin"
    l_trans_bandwidth: float = 2.0
    h_trans_bandwidth: float = 2.5
    resample_if_needed: bool = True
    use_cache: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_dir", Path(self.data_dir))
        object.__setattr__(self, "cache_dir", Path(self.cache_dir))
        object.__setattr__(self, "channels", tuple(self.channels))
        object.__setattr__(
            self, "bands", tuple((float(low), float(high)) for low, high in self.bands)
        )
        object.__setattr__(
            self,
            "artifact_band",
            (float(self.artifact_band[0]), float(self.artifact_band[1])),
        )

        if self.window_name not in EPOCH_WINDOWS:
            choices = ", ".join(sorted(EPOCH_WINDOWS))
            raise ValueError(f"unknown epoch window {self.window_name!r}; choose one of {choices}")
        if not self.channels or len(set(self.channels)) != len(self.channels):
            raise ValueError("channels must be a non-empty sequence without duplicates")
        if self.sfreq <= 0:
            raise ValueError("sfreq must be positive")
        nyquist = self.sfreq / 2.0
        if not self.bands:
            raise ValueError("at least one filter band is required")
        if any(not (0.0 < low < high < nyquist) for low, high in self.bands):
            raise ValueError(f"all filter bands must lie strictly inside (0, {nyquist:g}) Hz")
        artifact_low, artifact_high = self.artifact_band
        if not 0.0 < artifact_low < artifact_high < nyquist:
            raise ValueError("artifact_band must lie strictly inside the Nyquist interval")
        if self.artifact_threshold is not None and self.artifact_threshold <= 0.0:
            raise ValueError("artifact_threshold must be positive or None")
        if not 0.0 < self.covariance_shrinkage <= 1.0:
            raise ValueError("covariance_shrinkage must be in (0, 1]")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be 'float32' or 'float64'")
        if self.filter_method != "fir":
            raise ValueError("the validated preprocessing path currently supports FIR filtering")
        if self.l_trans_bandwidth <= 0.0 or self.h_trans_bandwidth <= 0.0:
            raise ValueError("filter transition bandwidths must be positive")

    @property
    def epoch_window(self) -> tuple[float, float]:
        """Return the inclusive MNE epoch bounds for ``window_name``."""

        return EPOCH_WINDOWS[self.window_name]

    @property
    def epoch_samples(self) -> int:
        """Number of samples produced by MNE's inclusive epoch bounds."""

        start, stop = self.epoch_window
        return int(round((stop - start) * self.sfreq)) + 1

    def with_window(self, window_name: str) -> "DataConfig":
        return replace(self, window_name=window_name)


DEFAULT_DATA_CONFIG: Final[DataConfig] = DataConfig()


def is_valid_subject_run(subject: int, run: int) -> bool:
    """Return whether a subject/run pair belongs to the fixed research cohort."""

    return int(subject) in SUBJECT_RUNS and int(run) in SUBJECT_RUNS[int(subject)]


__all__ = [
    "BANDS",
    "CACHE_DIR",
    "CHANNELS",
    "DATA_DIR",
    "DEFAULT_DATA_CONFIG",
    "DEFAULT_WINDOW",
    "DataConfig",
    "EEG_CHANNELS",
    "EPOCH_WINDOWS",
    "FB_BANDS",
    "LABEL_NAMES",
    "LEFT_LABEL",
    "NEUTRAL_LABEL",
    "PROJECT_ROOT",
    "RIGHT_LABEL",
    "SFREQ",
    "SUBJECT_RUNS",
    "SUBJECTS",
    "TASK_LABELS",
    "VALID_SESSION_MANIFEST",
    "VALID_SUBJECTS",
    "is_valid_subject_run",
]
