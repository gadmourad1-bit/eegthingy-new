from __future__ import annotations

import torch

from benchmark.models import (
    AdaptiveSincFilterBank,
    CANONICAL_21_POSITIONS,
    EXTENDED_31_POSITIONS,
    CardinalFieldConfig,
    CardinalDynamicsNet,
    CardinalDynamicsConfig,
    CardinalFieldNet,
    CardinalScalpField,
    CardinalMixedTemporalNet,
    JointCardinalScalpField,
    LearnableGaborFIRBank,
    LowRankSpectralTemporalResidual,
    FreeScopeNet,
    OrderedSincFilterBank,
    ScopeConfig,
    ScopeNet,
    parameter_count,
    scalp_quadrature_weights,
    spherical_polynomial_basis,
)


def _positions(channels: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(13 + channels)
    values = torch.randn(channels, 3, generator=generator)
    values[:, 2] = values[:, 2].abs() + 0.2
    return values / torch.linalg.vector_norm(values, dim=1, keepdim=True)


def _small_config() -> ScopeConfig:
    return ScopeConfig(
        n_filters=6,
        n_sources=4,
        sinc_kernel=31,
        width=16,
        mixer_blocks=1,
        dropout=0.0,
        moment_bins=(1, 2, 4),
    )


def test_sinc_filters_are_ordered_bounded_zero_mean_and_normalized() -> None:
    bank = OrderedSincFilterBank(n_filters=8, kernel_size=33)
    frequencies = bank.frequencies_hz()
    kernels = bank.kernels().squeeze(1)
    assert torch.all(frequencies[1:] > frequencies[:-1])
    frequencies = frequencies.detach()
    assert 4.0 < float(frequencies[0]) < float(frequencies[-1]) < 40.0
    assert torch.allclose(kernels.mean(dim=1), torch.zeros(8), atol=2e-7)
    assert torch.allclose(
        torch.linalg.vector_norm(kernels, dim=1), torch.ones(8), atol=2e-6
    )


def test_adaptive_sinc_starts_at_physical_filters_and_learns_fir_residual() -> None:
    torch.manual_seed(11)
    fixed = OrderedSincFilterBank(n_filters=5, kernel_size=33)
    adaptive = AdaptiveSincFilterBank(n_filters=5, kernel_size=33)
    adaptive.load_state_dict(fixed.state_dict(), strict=False)
    assert torch.allclose(fixed.kernels(), adaptive.kernels(), atol=1e-7)
    loss = adaptive(torch.randn(3, 4, 96)).square().mean()
    loss.backward()
    assert adaptive.fir_residual.grad is not None
    assert torch.count_nonzero(adaptive.fir_residual.grad) > 0


def test_short_gabor_firs_are_zero_mean_normalized_and_trainable() -> None:
    bank = LearnableGaborFIRBank(n_filters=8, kernel_size=25, sfreq=128.0)
    kernels = bank.kernels().squeeze(1)
    assert torch.allclose(kernels.mean(dim=-1), torch.zeros(8), atol=2e-7)
    assert torch.allclose(
        torch.linalg.vector_norm(kernels, dim=-1), torch.ones(8), atol=2e-6
    )
    assert torch.allclose(
        bank.initial_frequencies_hz, torch.linspace(6.0, 36.0, 8)
    )
    output = bank(torch.randn(3, 5, 96))
    assert output.shape == (3, 5, 8, 96)
    output.square().mean().backward()
    assert bank.raw_kernels.grad is not None
    assert torch.count_nonzero(bank.raw_kernels.grad) > 0


def test_low_rank_spectral_residual_is_near_identity_and_trainable() -> None:
    torch.manual_seed(12)
    module = LowRankSpectralTemporalResidual(n_filters=8)
    values = torch.randn(2, 5, 8, 96, requires_grad=True)
    output = module(values)
    relative = torch.linalg.vector_norm(output - values) / torch.linalg.vector_norm(values)
    assert float(relative.detach()) < 0.01
    output.square().mean().backward()
    assert module.raw_gates.grad is not None
    assert torch.count_nonzero(module.raw_gates.grad) > 0


def test_low_rank_spectral_residual_rejects_mean_subtracted_unit_kernel() -> None:
    import pytest

    with pytest.raises(ValueError, match="greater than one"):
        LowRankSpectralTemporalResidual(n_filters=8, kernel_sizes=(1, 7))


def test_learned_temporal_windows_start_equal_ordered_and_trainable() -> None:
    torch.manual_seed(16)
    config = CardinalFieldConfig(
        n_filters=4,
        n_sources=3,
        sinc_kernel=31,
        segments=4,
        learned_windows=True,
    )
    model = CardinalFieldNet(n_outputs=2, config=config).train()
    boundaries = model.temporal_boundaries()
    assert torch.allclose(boundaries, torch.linspace(0.0, 1.0, 5), atol=1e-7)
    assert torch.all(boundaries[1:] > boundaries[:-1])

    values = torch.randn(5, 7, 320)
    soft = model._soft_segmented_log_variance(values)
    hard = model._segmented_log_variance(values, 4)
    assert torch.mean(torch.abs(soft - hard)) < 0.015

    output = model(
        torch.randn(6, 21, 128), torch.tensor(CANONICAL_21_POSITIONS)
    )
    assert output.shape == (6, 2)
    output.square().mean().backward()
    assert model.window_logits is not None
    assert model.window_logits.grad is not None
    assert torch.isfinite(model.window_logits.grad).all()
    assert torch.count_nonzero(model.window_logits.grad) > 0


def test_continuous_basis_and_quadrature_are_finite() -> None:
    positions = _positions(21)
    basis = spherical_polynomial_basis(positions)
    weights = scalp_quadrature_weights(positions)
    assert basis.shape == (21, 16)
    assert weights.shape == (21,)
    assert torch.isfinite(basis).all()
    assert torch.all(weights > 0.0)
    assert torch.allclose(weights.sum(), torch.tensor(1.0))


def test_cardinal_basis_is_full_rank_and_nearly_identity_on_atlas() -> None:
    positions = torch.tensor(CANONICAL_21_POSITIONS)
    field = CardinalScalpField(n_filters=4, n_sources=3)
    basis = field.basis(positions)
    assert torch.linalg.matrix_rank(basis) == len(CANONICAL_21_POSITIONS)
    # The 1e-4 cardinal ridge deliberately trades exact interpolation for a
    # well-conditioned solve; the maximum atlas deviation is below one percent.
    assert torch.allclose(basis, torch.eye(len(positions)), atol=1e-2, rtol=1e-2)


def test_extended_cardinal_atlas_is_full_rank() -> None:
    positions = torch.tensor(EXTENDED_31_POSITIONS)
    field = CardinalScalpField(
        n_filters=2,
        n_sources=2,
        anchor_positions=EXTENDED_31_POSITIONS,
    )
    basis = field.basis(positions)
    assert torch.linalg.matrix_rank(basis) == len(EXTENDED_31_POSITIONS)
    # The 31-anchor kernel is more ill-conditioned than the 21-anchor kernel.
    # Ridge regularization retains a well-conditioned near-cardinal basis.
    assert torch.linalg.cond(basis) < 1.2
    assert torch.max(torch.abs(basis - torch.eye(len(positions)))) < 2.1e-2


def test_cardinal_model_is_montage_flexible_and_permutation_invariant() -> None:
    torch.manual_seed(14)
    model = CardinalFieldNet(n_outputs=4).eval()
    positions = torch.tensor(CANONICAL_21_POSITIONS)
    x = torch.randn(2, 21, 96)
    permutation = torch.randperm(21)
    direct = model(x, positions)
    reordered = model(x[:, permutation], positions[permutation])
    assert direct.shape == (2, 4)
    # Parallel and vectorized reductions can differ by a few ulps when their
    # summation order is permuted even though the operator is invariant.  The
    # bound covers both CUDA and Apple-arm64 CPU kernels.
    assert torch.allclose(direct, reordered, atol=3e-5, rtol=3e-5)
    assert model(x[:, :3], positions[:3]).shape == (2, 4)


def test_joint_cardinal_frequency_basis_preserves_inducing_grid() -> None:
    field = JointCardinalScalpField(n_filters=9, n_sources=4)
    basis = field.frequency_basis(field.frequency_anchors)
    assert torch.linalg.matrix_rank(basis) == 9
    assert torch.allclose(basis, torch.eye(9), atol=1e-2, rtol=1e-2)


def test_joint_contrast_model_has_one_output_and_residual_gradients() -> None:
    torch.manual_seed(15)
    config = CardinalFieldConfig(
        n_filters=6,
        n_sources=5,
        sinc_kernel=31,
        frequency_cardinal=True,
        contrast_bins=8,
        contrast_width=4,
    )
    model = CardinalFieldNet(n_outputs=3, config=config).train()
    output = model(torch.randn(4, 21, 128), torch.tensor(CANONICAL_21_POSITIONS))
    assert output.shape == (4, 3)
    output.square().mean().backward()
    assert model.contrast_classifier is not None
    assert model.contrast_classifier.weight.grad is not None
    assert torch.count_nonzero(model.contrast_classifier.weight.grad) > 0


def test_cardinal_dynamics_is_montage_flexible_and_permutation_invariant() -> None:
    torch.manual_seed(21)
    model = CardinalDynamicsNet(n_outputs=4).eval()
    positions = torch.tensor(CANONICAL_21_POSITIONS)
    x = torch.randn(2, 21, 320)
    permutation = torch.randperm(21)
    direct = model(x, positions)
    reordered = model(x[:, permutation], positions[permutation])
    assert direct.shape == (2, 4)
    assert torch.allclose(direct, reordered, atol=3e-5, rtol=3e-5)
    assert model(x[:, :3], positions[:3]).shape == (2, 4)
    assert parameter_count(model) < 100_000


def test_physical_cardinal_dynamics_has_ordered_filters_and_gradients() -> None:
    torch.manual_seed(22)
    config = CardinalDynamicsConfig(
        temporal_filters=8,
        temporal_kernel=33,
        physical_filter_bank=True,
        spectral_residual=True,
        latent_sources=6,
        pool_kernel=32,
        pool_stride=16,
        dynamics_channels=4,
        dynamics_kernel=9,
        dropout=0.0,
    )
    model = CardinalDynamicsNet(n_outputs=2, n_times=128, config=config).train()
    output = model(
        torch.randn(4, 21, 128), torch.tensor(CANONICAL_21_POSITIONS)
    )
    assert output.shape == (4, 2)
    assert isinstance(model.temporal, OrderedSincFilterBank)
    assert torch.all(
        model.temporal.frequencies_hz()[1:] > model.temporal.frequencies_hz()[:-1]
    )
    output.square().mean().backward()
    assert model.temporal.raw_frequency_gaps.grad is not None
    assert torch.count_nonzero(model.temporal.raw_frequency_gaps.grad) > 0


def test_scope_accepts_three_and_twenty_one_channel_montages() -> None:
    model = ScopeNet(n_outputs=2, config=_small_config()).eval()
    for channels in (3, 21):
        output = model(torch.randn(5, channels, 128), _positions(channels))
        assert output.shape == (5, 2)
        assert torch.isfinite(output).all()


def test_scope_is_invariant_to_joint_channel_and_coordinate_permutation() -> None:
    torch.manual_seed(17)
    model = ScopeNet(n_outputs=4, config=_small_config()).eval()
    x = torch.randn(3, 12, 128)
    positions = _positions(12)
    permutation = torch.randperm(12)
    direct = model(x, positions)
    permuted = model(x[:, permutation], positions[permutation])
    assert torch.allclose(direct, permuted, atol=2e-6, rtol=2e-6)


def test_channel_mask_is_supported_and_gradients_reach_physical_filters() -> None:
    torch.manual_seed(19)
    model = ScopeNet(n_outputs=2, config=_small_config()).train()
    x = torch.randn(6, 10, 128)
    mask = torch.ones(6, 10)
    mask[::2, -3:] = 0.0
    loss = model(x, _positions(10), mask).square().mean()
    loss.backward()
    for parameter in (
        model.filter_bank.raw_frequency_gaps,
        model.filter_bank.raw_bandwidths,
        model.spatial_field.coefficients,
        model.floor_classifier.weight,
        model.classifier.weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0


def test_free_control_has_one_head_and_reasonable_budget() -> None:
    model = FreeScopeNet(n_channels=21, n_outputs=4, config=_small_config())
    assert model(torch.randn(2, 21, 128)).shape == (2, 4)
    assert parameter_count(model) < 100_000
