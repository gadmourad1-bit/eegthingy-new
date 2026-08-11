from __future__ import annotations

import numpy as np

from deepnet.deployment import GeoAdaptDecoder
from deepnet.model import GeoAdaptNet


def _covariances(n: int = 8) -> np.ndarray:
    rng = np.random.default_rng(45)
    factor = rng.normal(size=(n, 4, 15, 15))
    covariance = factor @ np.swapaxes(factor, -1, -2)
    return (covariance / 15.0 + 0.1 * np.eye(15)).astype(np.float32)


def test_calibration_and_step_are_frozen_weight_and_causal() -> None:
    model = GeoAdaptNet(auxiliary_intent=True).eval()
    decoder = GeoAdaptDecoder(
        model,
        {
            "temperature": 1.0,
            "target_median_blend": 1.0,
            "benchmark_config": {"commit_confidence": 0.85},
        },
        device="cpu",
    )
    block = _covariances()
    decoder.calibrate(block[:4])
    weights_before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    state_before = decoder.adapter.covariance.reference_log
    decision = decoder.step(block[4])
    assert decision.prediction in {0, 1}
    assert 0.0 <= decision.probability_right <= 1.0
    assert decision.intent_probability is not None
    for key, value in model.state_dict().items():
        np.testing.assert_array_equal(value.detach().numpy(), weights_before[key].numpy())
    # The current decision exposes only the pre-update alignment; state may update
    # afterward, and never changes the frozen model weights.
    state_after = decoder.adapter.covariance.reference_log
    if decision.covariance_updated:
        assert not np.array_equal(state_before, state_after)


def test_stream_shape_is_enforced() -> None:
    decoder = GeoAdaptDecoder(GeoAdaptNet(), {}, device="cpu")
    with np.testing.assert_raises(ValueError):
        decoder.calibrate(_covariances(1))


def test_loso_checkpoint_configuration_controls_commit_threshold() -> None:
    decoder = GeoAdaptDecoder(
        GeoAdaptNet(),
        {"loso_config": {"commit_confidence": 0.91}},
        device="cpu",
    )
    assert decoder.adapter.boundary.commit_threshold == 0.91


def test_live_preprocessing_contract_must_match_checkpoint() -> None:
    metadata = {
        "data_contract": {
            "preprocessing_fingerprint": "expected",
            "preprocessing": {
                "channels": [f"C{index}" for index in range(15)],
                "bands": [[8.0, 12.0]] * 4,
            },
        }
    }
    decoder = GeoAdaptDecoder(GeoAdaptNet(), metadata, device="cpu")
    decoder.validate_preprocessing_contract(
        {"preprocessing_fingerprint": "expected"}
    )
    with np.testing.assert_raises_regex(ValueError, "does not match"):
        decoder.validate_preprocessing_contract(
            {"preprocessing_fingerprint": "different"}
        )
