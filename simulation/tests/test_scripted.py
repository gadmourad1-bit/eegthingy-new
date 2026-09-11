"""MIRepNet decision-plan validation and recovery rules."""

import json

import pytest

from tiago_maze.scripted import (
    correction_decision,
    directions_from_plan,
    label_direction,
    load_decision_plan,
)
from tiago_maze.standards import get_standard_maze


def test_loads_valid_mirepnet_plan(tmp_path):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({
        "format": "mirepnet-maze-plan-v1",
        "turns": [
            {"true_label": 1, "predicted_label": 2},
            {"true_label": 2, "predicted_label": 2},
        ],
    }))
    payload, resolved = load_decision_plan(str(path))
    assert resolved == path.resolve()
    assert directions_from_plan(payload["turns"]) == ("LEFT", "RIGHT")


@pytest.mark.parametrize("label,direction", [(1, "LEFT"), (2, "RIGHT")])
def test_label_direction(label, direction):
    assert label_direction(label) == direction


def test_recovery_turns_toward_open_corridor():
    assert correction_decision(0, 1) == 1
    assert correction_decision(0, 3) == 2
    assert correction_decision(0, 2) == 1
    assert correction_decision(2, 2) == 0


@pytest.mark.parametrize("bad_label", [0, 3, "LEFT"])
def test_rejects_invalid_labels(tmp_path, bad_label):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({
        "format": "mirepnet-maze-plan-v1",
        "turns": [{"true_label": bad_label, "predicted_label": 1}],
    }))
    with pytest.raises(ValueError):
        load_decision_plan(str(path))


def test_accepts_original_fixed_maze_plan(tmp_path):
    standard = get_standard_maze(2)
    path = tmp_path / "standard.json"
    path.write_text(json.dumps({
        "format": "mirepnet-maze-plan-v1",
        "standard_maze": 2,
        "maze_seed": 12,
        "turns": [
            {"true_label": label, "predicted_label": label}
            for label in standard.labels
        ],
    }))
    payload, _ = load_decision_plan(str(path))
    assert payload["standard_maze"] == 2


def test_rejects_plan_that_changes_a_standard_route(tmp_path):
    standard = get_standard_maze(3)
    labels = list(standard.labels)
    labels[0] = 1 if labels[0] == 2 else 2
    path = tmp_path / "changed.json"
    path.write_text(json.dumps({
        "format": "mirepnet-maze-plan-v1",
        "standard_maze": 3,
        "maze_seed": 13,
        "turns": [
            {"true_label": label, "predicted_label": label} for label in labels
        ],
    }))
    with pytest.raises(ValueError, match="do not match standard maze"):
        load_decision_plan(str(path))


def test_on_demand_plan_allows_predictions_to_start_empty(tmp_path):
    standard = get_standard_maze(1)
    path = tmp_path / "pending.json"
    path.write_text(json.dumps({
        "format": "mirepnet-maze-plan-v1",
        "inference_mode": "on-demand",
        "standard_maze": 1,
        "maze_seed": 11,
        "turns": [
            {"true_label": label, "predicted_label": None}
            for label in standard.labels
        ],
    }))
    payload, _ = load_decision_plan(str(path))
    assert all(row["predicted_label"] is None for row in payload["turns"])
