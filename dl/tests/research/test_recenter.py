"""Contract tests for causal, label-free online recentering."""

from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from benchmark.research.recenter import (
    BoundaryRecenter,
    DualLevelOnlineAdapter,
    LogEuclideanCovarianceRecenter,
)


def _spd_block(
    rng: np.random.Generator,
    windows: int,
    bands: int,
    channels: int,
) -> np.ndarray:
    factors = rng.normal(size=(windows, bands, channels, channels))
    identity = np.eye(channels)[None, None, ...]
    return factors @ np.swapaxes(factors, -1, -2) + 0.25 * identity


class BoundaryRecenterTests(unittest.TestCase):
    def test_first_prediction_uses_center_before_its_own_update(self) -> None:
        head = BoundaryRecenter(
            alpha=1.0,
            clamp=10.0,
            rest_confidence=0.90,
            commit_threshold=0.85,
        ).calibrate([-0.1, 0.0, 0.1])

        result = head.step(0.4)

        self.assertAlmostEqual(result.center_before, 0.0)
        self.assertAlmostEqual(result.margin, 0.4)  # not 0.0 after alpha=1 update
        self.assertEqual(result.prediction, 2)
        self.assertFalse(result.committed)
        self.assertAlmostEqual(result.center_after, 0.4)

    def test_center_is_clamped_around_calibration_seed(self) -> None:
        head = BoundaryRecenter(
            alpha=1.0,
            clamp=0.1,
            rest_confidence=0.99,
        ).calibrate([1.0, 1.0, 1.0])

        result = head.step(1.5)

        self.assertTrue(result.updated)
        self.assertAlmostEqual(head.center, 1.1)
        self.assertLessEqual(head.center, head.seed_center + head.clamp)

    def test_confident_motor_imagery_never_moves_boundary(self) -> None:
        head = BoundaryRecenter(alpha=0.5).calibrate([0.0, 0.0, 0.0])

        result = head.step(8.0, rest_probability=1.0)

        self.assertGreater(result.confidence, 0.99)
        self.assertFalse(result.rest_like)
        self.assertFalse(result.updated)
        self.assertEqual(head.center, 0.0)

    def test_external_rest_probability_adds_a_label_free_gate(self) -> None:
        head = BoundaryRecenter(
            alpha=1.0,
            clamp=2.0,
            rest_confidence=0.9,
            external_rest_threshold=0.7,
        ).calibrate([0.0])

        rejected = head.step(0.2, rest_probability=0.2)
        accepted = head.step(0.2, rest_probability=1.0)

        self.assertFalse(rejected.updated)
        self.assertEqual(rejected.center_after, 0.0)
        self.assertTrue(accepted.updated)
        self.assertAlmostEqual(accepted.center_after, 0.2)

    def test_confidence_is_from_recentered_calibrated_margin(self) -> None:
        head = BoundaryRecenter(temperature=2.0).calibrate([3.0, 3.0, 3.0])
        result = head.predict(5.0)
        expected = 1.0 / (1.0 + math.exp(-1.0))
        self.assertAlmostEqual(result.margin, 2.0)
        self.assertAlmostEqual(result.confidence, expected)


class CovarianceRecenterTests(unittest.TestCase):
    def test_alignment_is_causal_and_does_not_use_current_window(self) -> None:
        identity = np.eye(2)
        calibration = np.repeat(identity[None, None, ...], 2, axis=0)
        adapter = LogEuclideanCovarianceRecenter(
            alpha=1.0, robust_clip=None
        ).calibrate(calibration)
        shifted = 4.0 * identity

        first = adapter.step(shifted)
        second = adapter.step(shifted, update=False)

        np.testing.assert_allclose(first, shifted, atol=1e-10)
        np.testing.assert_allclose(second, identity, atol=1e-10)

    def test_alignment_and_reference_remain_spd(self) -> None:
        rng = np.random.default_rng(7)
        calibration = _spd_block(rng, windows=8, bands=4, channels=5)
        adapter = LogEuclideanCovarianceRecenter(alpha=0.2).calibrate(calibration)
        live = _spd_block(rng, windows=1, bands=4, channels=5)[0]

        aligned = adapter.step(live)

        np.testing.assert_allclose(aligned, np.swapaxes(aligned, -1, -2), atol=1e-10)
        self.assertTrue(np.all(np.linalg.eigvalsh(aligned) > 0.0))
        self.assertTrue(np.all(np.linalg.eigvalsh(adapter.reference_covariances) > 0.0))


class CoordinatorTests(unittest.TestCase):
    def _adapter(self) -> DualLevelOnlineAdapter:
        covariance = LogEuclideanCovarianceRecenter(
            alpha=0.25,
            robust_clip=0.5,
        )
        boundary = BoundaryRecenter(
            alpha=0.5,
            clamp=0.75,
            rest_confidence=0.8,
            commit_threshold=0.85,
        )
        return DualLevelOnlineAdapter(covariance, boundary)

    def test_calibration_and_json_serialization_round_trip(self) -> None:
        rng = np.random.default_rng(11)
        calibration = _spd_block(rng, windows=6, bands=2, channels=3)
        scores = np.array([-0.4, 0.1, 0.2, 0.3, 0.5, 0.9])
        adapter = self._adapter().calibrate(calibration, scores)
        self.assertAlmostEqual(adapter.boundary.seed_center, 0.25)

        live = _spd_block(rng, windows=1, bands=2, channels=3)[0]
        adapter.step(live, score=0.3, rest_probability=1.0)

        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "online-adapter.json"
            adapter.save(state_path)
            restored = DualLevelOnlineAdapter.load(state_path)

        expected = adapter.step(live, score=0.7, update=False)
        actual = restored.step(live, score=0.7, update=False)
        np.testing.assert_allclose(
            actual.aligned_covariances,
            expected.aligned_covariances,
            rtol=1e-12,
            atol=1e-12,
        )
        self.assertEqual(actual.prediction, expected.prediction)
        self.assertEqual(actual.committed, expected.committed)
        self.assertAlmostEqual(actual.margin, expected.margin)
        self.assertAlmostEqual(actual.confidence, expected.confidence)
        np.testing.assert_allclose(
            restored.covariance.reference_log,
            adapter.covariance.reference_log,
            rtol=0.0,
            atol=0.0,
        )

    def test_coordinator_scores_current_alignment_before_updates(self) -> None:
        identity = np.eye(2)
        calibration = np.repeat(identity[None, None, ...], 3, axis=0)
        adapter = DualLevelOnlineAdapter(
            LogEuclideanCovarianceRecenter(alpha=1.0, robust_clip=None),
            BoundaryRecenter(alpha=1.0, rest_confidence=0.95, clamp=10.0),
        ).calibrate(calibration, [0.0, 0.0, 0.0])

        shifted = 4.0 * identity
        result = adapter.step(
            shifted,
            scorer=lambda aligned: math.log(float(aligned[0, 0])),
        )

        np.testing.assert_allclose(result.aligned_covariances, shifted, atol=1e-10)
        self.assertAlmostEqual(result.margin, math.log(4.0))
        self.assertEqual(result.prediction, 2)
        self.assertIn(result.committed, (True, False))


if __name__ == "__main__":
    unittest.main()
