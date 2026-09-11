"""Helpers for replaying recorded classifier decisions through the maze."""

from __future__ import annotations

import json
from pathlib import Path

from .standards import get_standard_maze


def load_decision_plan(raw_path: str) -> tuple[dict, Path]:
    path = Path(raw_path[1:] if raw_path.startswith("@") else raw_path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "mirepnet-maze-plan-v1":
        raise ValueError(f"{path} is not a MIRepNet maze plan")
    turns = payload.get("turns")
    if not isinstance(turns, list) or not turns:
        raise ValueError("decision plan must contain at least one turn")
    on_demand = payload.get("inference_mode") == "on-demand"
    for index, row in enumerate(turns, 1):
        true_label = int(row.get("true_label", 0))
        predicted_label = row.get("predicted_label")
        if true_label not in (1, 2):
            raise ValueError(f"turn {index} true label must be 1 (LEFT) or 2 (RIGHT)")
        if not on_demand or predicted_label is not None:
            try:
                predicted_label = int(predicted_label)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"turn {index} prediction must be 1 (LEFT) or 2 (RIGHT)"
                ) from error
            if predicted_label not in (1, 2):
                raise ValueError(
                    f"turn {index} prediction must be 1 (LEFT) or 2 (RIGHT)"
                )
    if "standard_maze" in payload:
        standard = get_standard_maze(payload["standard_maze"])
        if int(payload.get("maze_seed", -1)) != standard.seed:
            raise ValueError(
                f"standard maze {standard.number} must use original seed {standard.seed}"
            )
        labels = tuple(int(row["true_label"]) for row in turns)
        if labels != standard.labels:
            raise ValueError(
                f"decision plan labels do not match standard maze {standard.number}"
            )
    return payload, path


def label_direction(label: int) -> str:
    return "LEFT" if int(label) == 1 else "RIGHT"


def directions_from_plan(turns: list[dict]) -> tuple[str, ...]:
    return tuple(label_direction(row["true_label"]) for row in turns)


def correction_decision(current_heading: int, wanted_heading: int) -> int:
    """Return a 90-degree correction: 1=LEFT, 2=RIGHT, 0=already correct."""
    delta = (int(wanted_heading) - int(current_heading)) % 4
    if delta == 0:
        return 0
    if delta == 3:
        return 2
    return 1  # +90, or first half of a 180-degree recovery
