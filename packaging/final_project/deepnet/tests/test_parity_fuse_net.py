from __future__ import annotations

import numpy as np
import torch
from torch import nn

from deepnet.augment import left_right_swap_index
from deepnet.cameo_net import _mirror_covariances, _mirror_raw
from deepnet.parity_fuse_net import (
    HeadlessRawEncoder,
    ParityFuseClassifier,
    ParityFuseConfig,
    ParityFuseNet,
)


def _small_network(*, zero_anchor: bool = False) -> ParityFuseNet:
    rng = np.random.default_rng(11)
    anchor = np.zeros(60, dtype=np.float32) if zero_anchor else rng.normal(size=60)
    return ParityFuseNet(
        n_channels=5,
        n_times=65,
        n_bands=2,
        n_tangent_features=60,
        tangent_anchor_weight=anchor,
        temporal_filters=4,
        temporal_kernel=7,
        dynamics_channels=6,
        dynamics_kernel=5,
        raw_dim=8,
        tangent_hidden=6,
        tangent_band_dim=4,
        tangent_dim=8,
        fusion_dim=8,
        gate_hidden=6,
    )


def _inputs(batch: int = 4) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(19)
    return (
        torch.randn(batch, 5, 65, generator=generator),
        torch.randn(batch, 5, 65, generator=generator),
        torch.randn(batch, 60, generator=generator),
        torch.randn(batch, 60, generator=generator),
    )


def _synthetic_data(
    n_rows: int, *, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = np.arange(n_rows, dtype=np.int64) % 2
    raw = rng.normal(scale=0.5, size=(n_rows, 5, 65)).astype(np.float32)
    phase = np.linspace(0.0, 8.0 * np.pi, raw.shape[-1], dtype=np.float32)
    signal = np.sin(phase)
    for row, label in enumerate(labels):
        raw[row, 1 if label == 0 else 2] += signal

    covariances = np.empty((n_rows, 2, 5, 5), dtype=np.float32)
    for row, label in enumerate(labels):
        for band in range(2):
            matrix = rng.normal(size=(5, 5))
            covariance = matrix @ matrix.T / 5.0 + np.eye(5)
            covariance[1 if label == 0 else 2, 1 if label == 0 else 2] += (
                1.5 + 0.2 * band
            )
            covariances[row, band] = covariance.astype(np.float32)
    return raw, covariances, labels


def _small_config() -> ParityFuseConfig:
    return ParityFuseConfig(
        epochs=2,
        batch_size=8,
        learning_rate=1e-3,
        patience=2,
        temporal_filters=4,
        temporal_kernel=7,
        dynamics_channels=6,
        dynamics_kernel=5,
        raw_dim=8,
        tangent_hidden=6,
        tangent_band_dim=4,
        tangent_dim=8,
        fusion_dim=8,
        gate_hidden=6,
        seed=3,
        device="cpu",
    )


def test_raw_encoder_is_headless_sample_local_and_deterministic() -> None:
    encoder = HeadlessRawEncoder(
        n_channels=5,
        temporal_filters=4,
        temporal_kernel=7,
        dynamics_channels=6,
        dynamics_kernel=5,
        output_dim=8,
    ).train()
    inputs = torch.randn(3, 5, 65)
    first = encoder(inputs)
    second = encoder(inputs)
    assert first.shape == (3, 8)
    assert torch.equal(first, second)
    assert any(isinstance(module, nn.GroupNorm) for module in encoder.modules())
    assert not any(
        isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.Dropout))
        for module in encoder.modules()
    )


def test_zero_residual_is_exactly_the_projected_anchor() -> None:
    model = _small_network().train()
    inputs = _inputs()
    output = model(*inputs)
    expected = nn.functional.linear(
        0.5 * (inputs[2] - inputs[3]), model.tangent_anchor_weight
    ).squeeze(1)
    assert torch.equal(output.residual_logit, torch.zeros_like(output.residual_logit))
    assert torch.equal(output.anchor_logit, expected)
    assert torch.equal(output.logit, expected)
    assert "tangent_anchor_weight" not in dict(model.named_parameters())
    assert "tangent_anchor_weight" in dict(model.named_buffers())


def test_network_has_exact_signed_action_and_invariant_latent_gate() -> None:
    model = _small_network().train()
    with torch.no_grad():
        model.residual_readout.weight.normal_(mean=0.0, std=0.2)
    inputs = _inputs()
    first = model(*inputs)
    reflected = model(inputs[1], inputs[0], inputs[3], inputs[2])

    assert torch.allclose(first.logit, -reflected.logit, atol=2e-6, rtol=1e-6)
    assert torch.allclose(
        first.anchor_logit, -reflected.anchor_logit, atol=1e-6, rtol=1e-6
    )
    assert torch.allclose(
        first.residual_logit, -reflected.residual_logit, atol=1e-6, rtol=1e-6
    )
    assert torch.allclose(first.raw_even, reflected.raw_even, atol=1e-7, rtol=1e-6)
    assert torch.allclose(first.raw_odd, -reflected.raw_odd, atol=1e-7, rtol=1e-6)
    assert torch.allclose(
        first.tangent_even, reflected.tangent_even, atol=1e-7, rtol=1e-6
    )
    assert torch.allclose(
        first.tangent_odd, -reflected.tangent_odd, atol=1e-7, rtol=1e-6
    )
    assert torch.allclose(
        first.fusion_weights, reflected.fusion_weights, atol=1e-7, rtol=1e-6
    )
    assert torch.allclose(
        first.fused_odd, -reflected.fused_odd, atol=1e-7, rtol=1e-6
    )


def test_default_network_is_below_parameter_budget() -> None:
    model = ParityFuseNet(
        n_channels=15,
        n_times=251,
        n_bands=4,
        n_tangent_features=480,
        tangent_anchor_weight=np.zeros(480, dtype=np.float32),
    )
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    assert trainable == 19_468
    assert trainable < 30_000
    assert model.residual_readout.bias is None


def test_all_trainable_gradients_are_finite_at_anchor_initialization() -> None:
    model = _small_network(zero_anchor=True).train()
    output = model(*_inputs())
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0])
    loss = nn.functional.binary_cross_entropy_with_logits(output.logit, labels)
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)
    assert torch.linalg.vector_norm(model.residual_readout.weight.grad) > 0


def test_classifier_reflection_save_and_load_round_trip(tmp_path) -> None:
    channels = ("Cz", "C3", "C4", "F3", "F4")
    raw_train, cov_train, y_train = _synthetic_data(16, seed=23)
    raw_validation, cov_validation, y_validation = _synthetic_data(8, seed=29)
    estimator = ParityFuseClassifier(_small_config()).fit(
        raw_train,
        cov_train,
        y_train,
        raw_validation,
        cov_validation,
        y_validation,
        channels=channels,
    )

    mirror_index = np.asarray(left_right_swap_index(channels), dtype=np.int64)
    expected_mean = np.concatenate(
        (raw_train, _mirror_raw(raw_train, mirror_index)), axis=0
    ).mean(axis=(0, 2), keepdims=True)
    assert np.allclose(estimator.raw_mean_, expected_mean)
    assert estimator.trainable_param_count_ < 30_000
    assert estimator.max_equivariance_error(raw_validation, cov_validation) < 2e-5
    assert estimator.max_gate_invariance_error(raw_validation, cov_validation) < 2e-6

    logits = estimator.decision_function(raw_validation, cov_validation)
    reflected_logits = estimator.decision_function(
        _mirror_raw(raw_validation, mirror_index),
        _mirror_covariances(cov_validation, mirror_index),
    )
    assert np.allclose(logits, -reflected_logits, atol=2e-5, rtol=1e-5)

    checkpoint = tmp_path / "parity-fuse.pt"
    estimator.save(checkpoint)
    restored = ParityFuseClassifier.load(checkpoint, device="cpu")
    assert np.allclose(
        estimator.predict_proba(raw_validation, cov_validation),
        restored.predict_proba(raw_validation, cov_validation),
        atol=1e-7,
        rtol=1e-6,
    )
    assert restored.best_epoch_ == estimator.best_epoch_
    assert restored.trainable_param_count_ == estimator.trainable_param_count_
