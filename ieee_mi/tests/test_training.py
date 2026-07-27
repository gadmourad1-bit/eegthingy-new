from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from ieee_mi.training import (
    TrainConfig,
    classification_metrics,
    predict_probabilities,
    refit_model,
)


class TinyNet(nn.Module):
    uses_positions = False

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(6, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.flatten(1))


def test_prediction_and_metrics_are_well_formed() -> None:
    model = TinyNet()
    x = np.arange(24, dtype=np.float32).reshape(4, 2, 3)
    positions = np.ones((2, 3), dtype=np.float32)
    probabilities = predict_probabilities(
        model, x, positions, device="cpu", batch_size=2
    )
    metrics = classification_metrics(np.asarray((0, 1, 0, 1)), probabilities)
    assert probabilities.shape == (4, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert 0.0 <= metrics["balanced_accuracy"] <= 1.0


def test_refit_restored_initialization_is_deterministic() -> None:
    torch.manual_seed(31)
    initial_model = TinyNet()
    initial_state = copy.deepcopy(initial_model.state_dict())
    x = np.random.default_rng(31).normal(size=(12, 2, 3)).astype(np.float32)
    y = np.asarray([0, 1] * 6, dtype=np.int64)
    positions = np.ones((2, 3), dtype=np.float32)
    config = TrainConfig(
        epochs=5,
        batch_size=4,
        learning_rate=1e-2,
        weight_decay=0.0,
        label_smoothing=0.0,
        segment_probability=0.0,
        time_shift_samples=0,
        noise_std=0.0,
        gradient_clip=10.0,
        seed=17,
        device="cpu",
    )

    first = TinyNet()
    first.load_state_dict(initial_state)
    first_fit = refit_model(
        first, x, y, positions, epochs=3, config=config
    )

    # Consume both RNGs before the second refit. Restoring the captured state
    # and reseeding inside refit_model must still reproduce the exact model.
    _ = torch.randn(100)
    _ = np.random.default_rng().normal(size=100)
    second = TinyNet()
    second.load_state_dict(initial_state)
    second_fit = refit_model(
        second, x, y, positions, epochs=3, config=config
    )

    for key, value in first_fit["model"].state_dict().items():
        assert torch.equal(value, second_fit["model"].state_dict()[key])
    assert first_fit["history"] == second_fit["history"]
    # The scheduler is deliberately not compressed to the three refit epochs.
    expected = config.learning_rate * (1.0 + np.cos(np.pi / config.epochs)) / 2.0
    assert np.isclose(first_fit["history"][0]["learning_rate"], expected)
