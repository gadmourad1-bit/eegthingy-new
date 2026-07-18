from types import SimpleNamespace

import numpy as np

from deepnet.experiment import (
    _select_boundary_blend,
    _summary,
    calibration_partition,
    relative_matrix_log,
    session_log_references,
)


def _spd(n: int = 5, bands: int = 2, channels: int = 3) -> np.ndarray:
    rng = np.random.default_rng(12)
    factor = rng.normal(size=(n, bands, channels, channels))
    return factor @ np.swapaxes(factor, -1, -2) + 0.2 * np.eye(channels)


def test_relative_matrix_log_preserves_voltage_scale_without_collapsing() -> None:
    covariance = _spd() * 1e-10
    logged = relative_matrix_log(covariance)
    assert np.all(np.isfinite(logged))
    assert np.mean(np.std(logged, axis=0)) > 1e-3
    scale = 7.0
    shifted = relative_matrix_log(covariance * scale)
    eye = np.eye(covariance.shape[-1])
    np.testing.assert_allclose(shifted, logged + np.log(scale) * eye, atol=1e-10)


def test_session_references_do_not_mix_recordings() -> None:
    covariance = _spd(n=6)
    groups = np.asarray(["a", "a", "a", "b", "b", "b"])
    references = session_log_references(covariance, groups)
    np.testing.assert_allclose(references[0], references[2])
    np.testing.assert_allclose(references[3], references[5])
    assert not np.allclose(references[0], references[3])


def test_calibration_partition_is_chronological_and_excludes_prefix() -> None:
    data = SimpleNamespace(
        event_samples=np.asarray([50, 10, 40, 20, 30, 60]),
        labels=np.asarray([1, 0, -1, 1, 0, 1]),
    )
    calibration, scoring = calibration_partition(data, np.arange(6), 2)
    assert calibration.tolist() == [1, 3]
    assert scoring.tolist() == [4, 0, 5]
    assert set(calibration).isdisjoint(scoring)


def test_summary_tolerates_no_committed_predictions() -> None:
    metrics = {
        "accuracy": 0.75,
        "balanced_accuracy": 0.75,
        "kappa": 0.5,
        "roc_auc": 0.8,
        "brier": 0.2,
        "ece": 0.1,
        "coverage": 0.0,
        "selective_accuracy": None,
        "rest_false_commit_rate": None,
    }
    summary = _summary([{"model": "geoadapt", "subject": 1, "metrics": metrics}])
    assert summary["geoadapt"]["selective_accuracy"]["mean"] is None


def test_boundary_blend_is_selected_only_from_fixed_grid() -> None:
    labels = np.asarray([0, 0, 1, 1])
    score = np.asarray([[0.4, 0.6], [0.45, 0.55], [0.2, 0.8], [0.1, 0.9]])
    calibration = np.asarray([[0.2, 0.8], [0.25, 0.75], [0.3, 0.7]])
    blend, balanced = _select_boundary_blend(labels, score, calibration)
    assert blend in {0.0, 0.25, 0.5, 0.75, 1.0}
    assert 0.5 <= balanced <= 1.0
