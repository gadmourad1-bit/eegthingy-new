"""Recorded EEG selection for the four fixed-maze replays."""

import json

import numpy as np
import pytest

from scripts.mirepnet_maze_test import OnDemandMIRepNet, select_route_epochs


def _row(index, label, predicted=None):
    predicted = label if predicted is None else predicted
    return {
        "test_epoch": index,
        "true_label": label,
        "predicted_label": predicted,
        "correct": label == predicted,
    }


def test_route_selection_uses_chronological_class_queues():
    rows = [_row(1, 2), _row(2, 1), _row(3, 1), _row(4, 2)]
    selected = select_route_epochs(rows, (1, 2, 1, 2))
    assert [row["test_epoch"] for row in selected] == [2, 1, 3, 4]
    assert [row["maze_turn"] for row in selected] == [1, 2, 3, 4]


def test_route_selection_never_changes_true_labels():
    selected = select_route_epochs([_row(1, 1), _row(2, 2)], (2, 1))
    assert [row["true_label"] for row in selected] == [2, 1]


def test_route_selection_requires_enough_recorded_commands():
    with pytest.raises(SystemExit, match="not enough recorded RIGHT"):
        select_route_epochs([_row(1, 1)], (1, 2))


def test_prediction_is_not_calculated_until_corner_requests_it(tmp_path):
    class FakeModel:
        classes_ = np.asarray([1, 2])

        def __init__(self):
            self.calls = 0

        def predict_proba(self, X):
            self.calls += 1
            assert len(X) == 1
            return np.asarray([[0.2, 0.8]])

    model = FakeModel()
    output = tmp_path / "live.json"
    payload = {
        "turns": [{"true_label": 2, "predicted_label": None}],
        "maze_test": {
            "predictions_completed": 0,
            "accuracy": None,
            "balanced_accuracy": None,
            "confusion_matrix": None,
        },
    }
    provider = OnDemandMIRepNet(
        model=model,
        protocol="zero-shot",
        X_eval=np.zeros((1, 15, 1000), dtype=np.float32),
        selected_indices=[0],
        feature_reference=None,
        payload=payload,
        output=output,
    )
    assert model.calls == 0
    assert payload["turns"][0]["predicted_label"] is None

    result = provider(0, payload["turns"][0])

    assert model.calls == 1
    assert result["predicted_label"] == 2
    saved = json.loads(output.read_text())
    assert saved["maze_test"]["predictions_completed"] == 1
    assert saved["maze_test"]["accuracy"] == 1.0
