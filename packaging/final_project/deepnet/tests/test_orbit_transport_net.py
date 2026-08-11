from __future__ import annotations

import copy
import os

import numpy as np
import pytest
import torch

from deepnet.orbit_transport_net import (
    OddOrientationTransport,
    OrbitTransportClassifier,
    OrbitTransportConfig,
    OrbitTransportNet,
)


def _small_model() -> OrbitTransportNet:
    return OrbitTransportNet(
        n_channels=3,
        n_times=101,
        temporal_filters=4,
        temporal_kernel=9,
        dynamics_channels=4,
        dynamics_kernel=5,
        pool_kernel=25,
        pool_stride=10,
        orientation_rank=3,
        orientation_hidden=5,
        gate_hidden=6,
    )


@pytest.mark.parametrize("training", (True, False))
def test_orientation_gate_swaps_and_transport_logit_negates(training: bool) -> None:
    layer = OddOrientationTransport(rank=5, hidden=7)
    layer.train(training)
    first = torch.tensor((-2.0, -0.3, 0.7, 3.0), requires_grad=True)
    reflected = torch.tensor((1.0, -0.8, 1.9, -2.0), requires_grad=True)

    direct = layer(first, reflected)
    swapped = layer(reflected, first)

    assert torch.allclose(
        direct.orientation_score,
        -swapped.orientation_score,
        atol=1e-7,
        rtol=1e-6,
    )
    assert torch.allclose(direct.alpha, 1.0 - swapped.alpha, atol=1e-7, rtol=1e-6)
    assert torch.equal(direct.logit, -swapped.logit)
    explicit_transport = direct.alpha * first - (1.0 - direct.alpha) * reflected
    assert torch.allclose(direct.logit, explicit_transport, atol=2e-7, rtol=1e-6)
    assert torch.equal(direct.even, swapped.even)
    assert torch.equal(direct.odd, -swapped.odd)

    zero_odd = layer(first, first)
    assert torch.equal(zero_odd.orientation_score, torch.zeros_like(first))
    assert torch.equal(zero_odd.alpha, torch.full_like(first, 0.5))


@pytest.mark.parametrize("training", (True, False))
def test_full_network_is_exactly_odd_and_router_is_invariant(training: bool) -> None:
    torch.manual_seed(2)
    model = _small_model()
    model.train(training)
    raw = torch.randn(5, 3, 101)
    reflected = torch.randn(5, 3, 101)
    anchor = torch.randn(5)
    anchor_reflected = torch.randn(5)

    direct = model(raw, reflected, anchor, anchor_reflected)
    swapped = model(reflected, raw, anchor_reflected, anchor)

    for name in OrbitTransportNet.CANDIDATE_NAMES:
        assert torch.allclose(
            direct.candidate_logits()[name],
            -swapped.candidate_logits()[name],
            atol=2e-6,
            rtol=2e-6,
        )
    assert torch.allclose(
        direct.fusion_weights, swapped.fusion_weights, atol=2e-7, rtol=1e-6
    )
    assert torch.allclose(
        direct.energy_alpha, 1.0 - swapped.energy_alpha, atol=1e-7, rtol=1e-6
    )
    assert torch.allclose(
        direct.dynamics_alpha,
        1.0 - swapped.dynamics_alpha,
        atol=1e-7,
        rtol=1e-6,
    )
    assert torch.allclose(
        direct.energy_view_logits,
        swapped.energy_view_logits.flip(1),
        atol=2e-6,
        rtol=2e-6,
    )
    assert torch.allclose(
        direct.dynamics_view_logits,
        swapped.dynamics_view_logits.flip(1),
        atol=2e-6,
        rtol=2e-6,
    )


def test_paired_batch_normalization_preserves_train_mode_orbit_action() -> None:
    torch.manual_seed(23)
    model = OrbitTransportNet(
        n_channels=3,
        n_times=101,
        normalization="batch",
        temporal_filters=4,
        temporal_kernel=9,
        dynamics_channels=4,
        dynamics_kernel=5,
        pool_kernel=25,
        pool_stride=10,
        orientation_rank=3,
        orientation_hidden=5,
        gate_hidden=6,
    ).train()
    swapped_model = copy.deepcopy(model)
    raw = torch.randn(5, 3, 101)
    reflected = torch.randn(5, 3, 101)
    anchor = torch.randn(5)
    anchor_reflected = torch.randn(5)
    direct = model(raw, reflected, anchor, anchor_reflected)
    swapped = swapped_model(reflected, raw, anchor_reflected, anchor)
    assert torch.allclose(direct.logit, -swapped.logit, atol=2e-6, rtol=2e-6)
    assert torch.allclose(
        direct.fusion_weights, swapped.fusion_weights, atol=2e-7, rtol=1e-6
    )
    assert any(isinstance(module, torch.nn.BatchNorm2d) for module in model.modules())
    for direct_buffer, swapped_buffer in zip(
        model.buffers(), swapped_model.buffers(), strict=True
    ):
        assert torch.equal(direct_buffer, swapped_buffer)
    model.eval()
    swapped_model.eval()
    direct_eval = model(raw, reflected, anchor, anchor_reflected)
    swapped_eval = swapped_model(reflected, raw, anchor_reflected, anchor)
    assert torch.equal(direct_eval.logit, -swapped_eval.logit)


def test_gradients_reach_raw_heads_transport_and_fusion() -> None:
    torch.manual_seed(3)
    model = _small_model().train()
    output = model(
        torch.randn(8, 3, 101),
        torch.randn(8, 3, 101),
        torch.randn(8),
        torch.randn(8),
    )
    labels = torch.randint(0, 2, (8,), dtype=torch.float32)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(output.logit, labels)
    loss = loss + 0.2 * torch.nn.functional.binary_cross_entropy_with_logits(
        output.energy_logit, labels
    )
    loss.backward()

    expected = (
        model.raw_expert.backbone.energy_head.weight,
        model.raw_expert.backbone.dynamics_head.weight,
        model.energy_transport.odd_basis.weight,
        model.energy_transport.invariant_conditioner[0].weight,
        model.energy_transport.readout.weight,
        model.fusion_gate[-1].weight,
    )
    for parameter in expected:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0


def test_default_architecture_has_fewer_than_50k_trainable_parameters() -> None:
    model = OrbitTransportNet(n_channels=15, n_times=251)
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    assert trainable < 50_000
    assert trainable > 10_000


def _synthetic_split(
    rng: np.random.Generator,
    n: int,
    *,
    offset: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.tile(np.array((0, 1), dtype=np.int64), n // 2)
    raw = rng.normal(offset, 1.0, size=(n, 3, 101)).astype(np.float32)
    raw[:, 0] += (2 * labels - 1)[:, None] * 0.25
    observations = rng.normal(size=(n, 2, 3, 20))
    covariances = np.einsum(
        "nbct,nbdt->nbcd", observations, observations, optimize=True
    )
    identity = np.eye(3)[None, None]
    covariances = (covariances / 20.0 + 0.2 * identity).astype(np.float32)
    return raw, covariances, labels


def test_classifier_fits_preprocessing_on_source_only_and_keeps_exact_parity() -> None:
    rng = np.random.default_rng(4)
    raw_train, cov_train, y_train = _synthetic_split(rng, 16, offset=0.0)
    raw_validation, cov_validation, y_validation = _synthetic_split(rng, 8, offset=50.0)
    expected_balanced = np.concatenate(
        (raw_train, raw_train[:, (0, 2, 1), :]), axis=0
    ).mean(axis=(0, 2), keepdims=True)
    config = OrbitTransportConfig(
        epochs=3,
        batch_size=8,
        patience=2,
        temporal_filters=4,
        temporal_kernel=9,
        dynamics_channels=4,
        dynamics_kernel=5,
        pool_kernel=25,
        pool_stride=10,
        orientation_rank=3,
        orientation_hidden=5,
        gate_hidden=6,
        device="cpu",
        candidate_names=("fused", "anchor", "raw_mean"),
    )
    classifier = OrbitTransportClassifier(config).fit(
        raw_train,
        cov_train,
        y_train,
        raw_validation,
        cov_validation,
        y_validation,
        channels=("Cz", "C3", "C4"),
    )

    assert np.allclose(classifier.raw_mean_, expected_balanced, atol=1e-7)
    assert np.max(np.abs(classifier.raw_mean_)) < 1.0
    probabilities = classifier.predict_proba(raw_validation, cov_validation)
    assert probabilities.shape == (len(y_validation), 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert classifier.selected_candidate_ in config.candidate_names
    assert classifier.trainable_param_count_ < 50_000
    assert classifier.convex_anchor_learned_param_count_ > 0
    assert classifier.total_learned_param_count_ == (
        classifier.param_count_ + classifier.convex_anchor_learned_param_count_
    )
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.is_deterministic_algorithms_warn_only_enabled()
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] in {":4096:8", ":16:8"}
    assert classifier.max_equivariance_error(raw_validation, cov_validation) < 2e-6
    assert set(classifier.predict_all_logits(raw_validation, cov_validation)) == set(
        OrbitTransportNet.CANDIDATE_NAMES
    )


def test_model_rejects_mismatched_views() -> None:
    model = _small_model()
    with pytest.raises(ValueError, match="paired raw views"):
        model(
            torch.randn(2, 3, 101),
            torch.randn(3, 3, 101),
            torch.randn(2),
            torch.randn(2),
        )
    with pytest.raises(ValueError, match="anchor logits"):
        model(
            torch.randn(2, 3, 101),
            torch.randn(2, 3, 101),
            torch.randn(2, 1),
            torch.randn(2, 1),
        )
