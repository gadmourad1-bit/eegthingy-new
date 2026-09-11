"""Balanced trial generation and self-describing BrainFlow markers."""

from __future__ import annotations

import random
from dataclasses import dataclass

from .config import COMMANDS, COMMAND_IDS, PHASE_IDS


@dataclass(frozen=True)
class Trial:
    number: int
    block: int
    command: str
    frequency_hz: float


def make_trials(repetitions_per_command: int, frequencies_hz: dict[str, float], seed: int) -> list[Trial]:
    """Return balanced randomized trials (one of every command per block)."""
    rng = random.Random(seed)
    trials: list[Trial] = []
    number = 1
    for block in range(1, repetitions_per_command + 1):
        commands = list(COMMANDS)
        rng.shuffle(commands)
        for command in commands:
            trials.append(
                Trial(
                    number=number,
                    block=block,
                    command=command,
                    frequency_hz=float(frequencies_hz[command]),
                )
            )
            number += 1
    return trials


def encode_marker(command: str | None, phase: str, trial_number: int) -> int:
    """Encode command, phase, and trial into one exactly representable integer.

    Layout: ``C P TTTTT`` where C is command ID, P is phase ID, and T is the
    one-based trial.  Command 0 is used only for the session preparation phase.
    """
    command_id = 0 if command is None else COMMAND_IDS[command]
    return command_id * 1_000_000 + PHASE_IDS[phase] * 100_000 + int(trial_number)


def decode_marker(value: float | int) -> dict[str, object]:
    marker = int(round(float(value)))
    command_id = marker // 1_000_000
    phase_id = (marker // 100_000) % 10
    trial_number = marker % 100_000
    id_to_command = {value: key for key, value in COMMAND_IDS.items()}
    id_to_phase = {value: key for key, value in PHASE_IDS.items()}
    return {
        "command": id_to_command.get(command_id),
        "phase": id_to_phase.get(phase_id, "unknown"),
        "trial_number": trial_number,
    }


def marker_description(value: float | int, frequencies_hz: dict[str, float]) -> str:
    decoded = decode_marker(value)
    command = decoded["command"]
    phase = decoded["phase"]
    trial = decoded["trial_number"]
    if command is None:
        return f"ssvep/session/{phase}/t{trial:03d}"
    frequency = frequencies_hz[str(command)]
    return f"ssvep/{command}/{frequency:g}Hz/{phase}/t{trial:03d}"
