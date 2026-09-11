"""Configuration and constants for the standalone SSVEP experiment."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = PACKAGE_DIR / "settings.json"
DEFAULT_RECORDINGS_DIR = PACKAGE_DIR / "recordings"

# The board exposes 16 EEG inputs, but Cyton channel 8 is intentionally not
# connected in this laboratory montage.  The saved SSVEP file therefore has 15
# EEG channels, in this exact order.
EEG_CHANNEL_NAMES = (
    "Cz", "Pz", "C3", "C4", "P7", "P8", "Fz",
    "F7", "F8", "F3", "F4", "T7", "T8", "P3", "P4",
)
USABLE_EEG_INDICES = (0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15)

COMMANDS = ("forward", "right", "backward", "left")
COMMAND_IDS = {name: index + 1 for index, name in enumerate(COMMANDS)}
COMMAND_LABELS = {
    "forward": "FORWARD",
    "right": "RIGHT",
    "backward": "BACKWARD",
    "left": "LEFT",
}
COMMAND_POSITIONS = {
    "forward": "top",
    "right": "right",
    "backward": "bottom",
    "left": "left",
}
PHASES = ("preparation", "planning", "action", "rest")
PHASE_IDS = {name: index + 1 for index, name in enumerate(PHASES)}


class ConfigurationError(ValueError):
    """Raised when a collection setting cannot produce a valid experiment."""


@dataclass
class Settings:
    """Everything needed to reproduce one SSVEP collection session."""

    subject: str = "01"
    session: str = "01"
    repetitions_per_command: int = 5
    preparation_s: float = 8.0
    planning_s: float = 2.0
    action_s: float = 5.0
    rest_s: float = 3.0
    frequencies_hz: dict[str, float] = field(
        default_factory=lambda: {
            "forward": 8.0,
            "right": 10.0,
            "backward": 12.0,
            "left": 15.0,
        }
    )
    square_size_cm: float = 4.0
    horizontal_offset_cm: float = 15.0
    vertical_offset_cm: float = 11.0
    bright_level: int = 245
    dark_level: int = 18
    random_seed: int = 7
    serial_port: str = ""
    synthetic: bool = False
    fullscreen: bool = True
    output_dir: str = str(DEFAULT_RECORDINGS_DIR)

    @property
    def trial_count(self) -> int:
        return self.repetitions_per_command * len(COMMANDS)

    @property
    def estimated_duration_s(self) -> float:
        per_trial = self.planning_s + self.action_s + self.rest_s
        return self.preparation_s + self.trial_count * per_trial

    def validate(self) -> "Settings":
        errors: list[str] = []
        self.subject = self.subject.strip()
        self.session = self.session.strip()
        self.serial_port = self.serial_port.strip()
        self.output_dir = self.output_dir.strip()

        if not self.subject:
            errors.append("Subject ID cannot be empty.")
        if not self.session:
            errors.append("Session ID cannot be empty.")
        if self.repetitions_per_command < 1:
            errors.append("Repetitions per command must be at least 1.")
        if self.repetitions_per_command > 250:
            errors.append("Repetitions per command must be 250 or less.")

        durations = {
            "Preparation": self.preparation_s,
            "Planning": self.planning_s,
            "Action": self.action_s,
            "Rest": self.rest_s,
        }
        for name, seconds in durations.items():
            if seconds < 0:
                errors.append(f"{name} time cannot be negative.")
        if self.action_s <= 0:
            errors.append("Action time must be greater than 0 seconds.")

        missing = [name for name in COMMANDS if name not in self.frequencies_hz]
        if missing:
            errors.append("Missing frequencies for: " + ", ".join(missing))
        else:
            values = [float(self.frequencies_hz[name]) for name in COMMANDS]
            if any(value <= 0 or value > 30 for value in values):
                errors.append("Every stimulus frequency must be above 0 and at most 30 Hz.")
            if len({round(value, 6) for value in values}) != len(values):
                errors.append("The four stimulus frequencies must be different.")

        if self.square_size_cm <= 0:
            errors.append("Square size must be greater than 0 cm.")
        if self.horizontal_offset_cm <= 0 or self.vertical_offset_cm <= 0:
            errors.append("Horizontal and vertical offsets must be greater than 0 cm.")
        if not (0 <= self.dark_level < self.bright_level <= 255):
            errors.append("Brightness must satisfy 0 <= dark < bright <= 255.")
        if not self.output_dir:
            errors.append("Output folder cannot be empty.")

        if errors:
            raise ConfigurationError("\n".join(errors))
        return self

    def save(self, path: Path = CONFIG_PATH) -> Path:
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> "Settings":
        if not path.exists():
            return cls()
        try:
            values = json.loads(path.read_text(encoding="utf-8"))
            settings = cls(**values)
            return settings.validate()
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            # A broken local settings file must never prevent the collector from
            # opening.  The GUI will show safe defaults and overwrite it on Start.
            return cls()
