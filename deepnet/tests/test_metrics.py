import numpy as np
import pytest

from deepnet.metrics import classification_metrics, expected_calibration_error


def test_perfect_probabilities_have_perfect_scores() -> None:
    y = np.array([0, 1, 0, 1])
    p = np.array([[0.99, 0.01], [0.01, 0.99], [0.9, 0.1], [0.1, 0.9]])
    metrics = classification_metrics(y, p, commit_confidence=0.85)
    assert metrics.accuracy == 1.0
    assert metrics.balanced_accuracy == 1.0
    assert metrics.coverage == 1.0
    assert metrics.selective_accuracy == 1.0
    assert metrics.ece < 0.1


def test_selective_metrics_separate_coverage_from_accuracy() -> None:
    y = np.array([0, 1, 1])
    p = np.array([[0.95, 0.05], [0.45, 0.55], [0.8, 0.2]])
    metrics = classification_metrics(y, p, commit_confidence=0.9)
    assert metrics.coverage == pytest.approx(1 / 3)
    assert metrics.selective_accuracy == 1.0


def test_ece_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError):
        expected_calibration_error(np.array([0]), np.zeros((1, 3)))
