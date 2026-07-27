from __future__ import annotations

import numpy as np
import torch

from deepnet.cameo_net import (
    CAMEOClassifier,
    CAMEOConfig,
    CAMEORawExpert,
    _mirror_covariances,
    _mirror_raw,
)


def test_raw_expert_has_two_finite_signed_heads() -> None:
    model = CAMEORawExpert(
        n_channels=15,
        n_times=251,
        temporal_filters=4,
        dynamics_channels=4,
    )
    energy, dynamics, features = model(torch.randn(3, 15, 251))
    assert energy.shape == (3,)
    assert dynamics.shape == (3,)
    assert torch.isfinite(energy).all()
    assert torch.isfinite(dynamics).all()
    assert features["energy_features"].shape[0] == 3
    assert features["dynamics_features"].shape == (3, 8)


def test_mirror_is_an_involution_for_raw_and_covariance() -> None:
    index = np.asarray((0, 1, 3, 2, 5, 4, 6, 8, 7, 10, 9, 12, 11, 14, 13))
    raw = np.arange(2 * 15 * 7).reshape(2, 15, 7)
    cov = np.arange(2 * 4 * 15 * 15).reshape(2, 4, 15, 15)
    assert np.array_equal(_mirror_raw(_mirror_raw(raw, index), index), raw)
    assert np.array_equal(
        _mirror_covariances(_mirror_covariances(cov, index), index), cov
    )


def test_rho_one_is_exactly_label_equivariant() -> None:
    components = {
        "geo": (
            np.asarray((2.0, -0.5)),
            np.asarray((-0.2, 1.5)),
        )
    }
    original = CAMEOClassifier._mixed_probability(
        components, names=("geo",), rho=1.0
    )
    reflected_components = {"geo": (components["geo"][1], components["geo"][0])}
    reflected = CAMEOClassifier._mixed_probability(
        reflected_components, names=("geo",), rho=1.0
    )
    assert np.allclose(original[:, 0], reflected[:, 1])
    assert np.allclose(original[:, 1], reflected[:, 0])


def test_small_cpu_fit_returns_normalized_probabilities() -> None:
    generator = np.random.default_rng(4)

    def sample(count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        labels = np.arange(count, dtype=np.int64) % 2
        raw = generator.normal(size=(count, 15, 251)).astype(np.float32)
        factors = generator.normal(size=(count, 4, 15, 15))
        cov = factors @ np.swapaxes(factors, -1, -2)
        cov += 0.2 * np.eye(15)[None, None]
        return raw, cov.astype(np.float32), labels

    train = sample(20)
    validation = sample(10)
    classifier = CAMEOClassifier(
        CAMEOConfig(
            epochs=1,
            patience=1,
            batch_size=10,
            temporal_filters=2,
            dynamics_channels=2,
            device="cpu",
            rho_grid=(0.0, 1.0),
        )
    ).fit(*train, *validation)
    probability = classifier.predict_proba(validation[0], validation[1])
    assert probability.shape == (10, 2)
    assert np.isfinite(probability).all()
    assert np.allclose(probability.sum(axis=1), 1.0)

