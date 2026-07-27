from __future__ import annotations

import torch
from pathlib import Path

from ieee_mi.baselines import CorrectedCTNet, make_model
from ieee_mi.config import BNCI004_3_CHANNELS, CANONICAL_21_CHANNELS, ZHOU_8_CHANNELS
from ieee_mi.models import (
    CANONICAL_21_POSITIONS,
    CardinalFBMSNet,
    CardinalFBCCompactDynamicsNet,
    CardinalFBCCorrelationNet,
    CardinalFBCMicroDynamicsNet,
    CardinalFBCPhysicalDynamicsNet,
    OrderedSincFilterBank,
)


def test_modern_unified_baselines_have_valid_logits() -> None:
    configurations = (
        ("cardinal_fbc", BNCI004_3_CHANNELS),
        ("cardinal_fbc", CANONICAL_21_CHANNELS),
        ("cardinal_fbc_corr", BNCI004_3_CHANNELS),
        ("cardinal_fbc_corr", CANONICAL_21_CHANNELS),
        ("cardinal_fbc_compactdyn", BNCI004_3_CHANNELS),
        ("cardinal_fbc_compactdyn", CANONICAL_21_CHANNELS),
        ("cardinal_fbc_micro", BNCI004_3_CHANNELS),
        ("cardinal_fbc_micro", CANONICAL_21_CHANNELS),
        ("cardinal_fbc_physical", BNCI004_3_CHANNELS),
        ("cardinal_fbc_physical", CANONICAL_21_CHANNELS),
        ("fbmsnet", CANONICAL_21_CHANNELS),
        ("cardinal_fbms", BNCI004_3_CHANNELS),
        ("cardinal_fbms", CANONICAL_21_CHANNELS),
        ("cardinal_mix_drop", CANONICAL_21_CHANNELS),
        ("ctnet_compact", CANONICAL_21_CHANNELS),
        ("eegsym", BNCI004_3_CHANNELS),
        ("eegsym", ZHOU_8_CHANNELS),
        ("eegsym", CANONICAL_21_CHANNELS),
    )
    for name, channels in configurations:
        model = make_model(
            name,
            n_channels=len(channels),
            n_outputs=4 if len(channels) > 3 else 2,
            n_times=320,
            sfreq=128.0,
            channel_names=tuple(channels),
        ).eval()
        values = torch.randn(2, len(channels), 320)
        if bool(getattr(model, "uses_positions", False)):
            positions = torch.randn(len(channels), 3)
            positions = positions / torch.linalg.vector_norm(
                positions, dim=1, keepdim=True
            )
            output = model(values, positions)
        else:
            output = model(values)
        assert output.shape == (2, 4 if len(channels) > 3 else 2)
        assert torch.isfinite(output).all()


def test_ctnet_wrapper_uses_configured_dropout_path() -> None:
    model = make_model(
        "ctnet_compact",
        n_channels=21,
        n_outputs=4,
        n_times=320,
        sfreq=128.0,
        channel_names=tuple(CANONICAL_21_CHANNELS),
    )
    assert isinstance(model, CorrectedCTNet)
    assert isinstance(model.base.flatten_drop_layer[-1], torch.nn.Dropout)


def test_cardinal_fbc_is_permutation_invariant_with_fbc_sized_budget() -> None:
    torch.manual_seed(23)
    model = make_model(
        "cardinal_fbc",
        n_channels=21,
        n_outputs=4,
        n_times=320,
        sfreq=128.0,
        channel_names=tuple(CANONICAL_21_CHANNELS),
    ).eval()
    values = torch.randn(2, 21, 320)
    positions = torch.tensor(CANONICAL_21_POSITIONS)
    permutation = torch.randperm(21)
    direct = model(values, positions)
    reordered = model(values[:, permutation], positions[permutation])
    assert torch.allclose(direct, reordered, atol=3e-5, rtol=3e-5)
    assert sum(parameter.numel() for parameter in model.parameters()) < 20_000


def test_cardinal_fbc_starts_from_exact_fbc_spatial_floor() -> None:
    positions = torch.tensor(CANONICAL_21_POSITIONS)
    torch.manual_seed(29)
    indexed = make_model(
        "fbcnet",
        n_channels=21,
        n_outputs=4,
        n_times=320,
        sfreq=128.0,
        channel_names=tuple(CANONICAL_21_CHANNELS),
        channel_positions=positions,
    ).train()
    torch.manual_seed(29)
    continuous = make_model(
        "cardinal_fbc",
        n_channels=21,
        n_outputs=4,
        n_times=320,
        sfreq=128.0,
        channel_names=tuple(CANONICAL_21_CHANNELS),
        channel_positions=positions,
    ).train()
    values = torch.randn(8, 21, 320)
    indexed_output = indexed(values)
    continuous_output = continuous(values, positions)
    assert torch.allclose(indexed_output, continuous_output, atol=1e-4, rtol=1e-4)


def test_cardinal_fbc_corr_starts_at_exact_single_head_fbc_floor() -> None:
    positions = torch.tensor(CANONICAL_21_POSITIONS)
    torch.manual_seed(37)
    indexed = make_model(
        "fbcnet",
        n_channels=21,
        n_outputs=4,
        n_times=320,
        sfreq=128.0,
        channel_names=tuple(CANONICAL_21_CHANNELS),
        channel_positions=positions,
    ).train()
    torch.manual_seed(37)
    correlation = make_model(
        "cardinal_fbc_corr",
        n_channels=21,
        n_outputs=4,
        n_times=320,
        sfreq=128.0,
        channel_names=tuple(CANONICAL_21_CHANNELS),
        channel_positions=positions,
    ).train()
    assert isinstance(correlation, CardinalFBCCorrelationNet)
    values = torch.randn(8, 21, 320)
    torch.testing.assert_close(
        correlation(values, positions),
        indexed(values),
        rtol=1e-4,
        atol=1e-4,
    )
    assert correlation.correlation_feature_count == 540
    assert torch.count_nonzero(
        correlation.final_layer.weight[:, correlation.floor_feature_count :]
    ) == 0
    assert correlation.final_layer.in_features == (
        correlation.floor_feature_count + correlation.correlation_feature_count
    )
    assert sum(parameter.numel() for parameter in correlation.parameters()) < 25_000


def test_cardinal_fbc_corr_features_are_finite_and_permutation_invariant() -> None:
    torch.manual_seed(38)
    positions = torch.tensor(CANONICAL_21_POSITIONS)
    model = make_model(
        "cardinal_fbc_corr_extended",
        n_channels=21,
        n_outputs=4,
        n_times=320,
        sfreq=128.0,
        channel_names=tuple(CANONICAL_21_CHANNELS),
    ).eval()
    assert isinstance(model, CardinalFBCCorrelationNet)
    values = torch.randn(3, 21, 320)
    permutation = torch.randperm(21)
    with torch.inference_mode():
        floor, signed = model.encode_floor_with_signed_sources(values, positions)
        correlations = model.encode_correlations(signed)
        reordered_floor, reordered_signed = model.encode_floor_with_signed_sources(
            values[:, permutation], positions[permutation]
        )
        reordered_correlations = model.encode_correlations(reordered_signed)
        constant_correlations = model.encode_correlations(torch.zeros_like(signed))
    assert floor.shape[0] == len(values)
    assert correlations.shape == (len(values), 540)
    assert torch.isfinite(correlations).all()
    assert torch.isfinite(constant_correlations).all()
    assert torch.count_nonzero(constant_correlations) == 0
    torch.testing.assert_close(
        reordered_floor, floor, rtol=1e-4, atol=1e-4
    )
    torch.testing.assert_close(
        reordered_correlations, correlations, rtol=1e-4, atol=1e-4
    )
    projections = model.normalized_correlation_projection()
    row_gram = projections @ projections.transpose(-1, -2)
    identity = torch.eye(model.correlation_rank).expand_as(row_gram)
    torch.testing.assert_close(row_gram, identity, rtol=2e-5, atol=2e-5)


def test_cardinal_fbc_corr_branch_learns_after_head_warm_start() -> None:
    torch.manual_seed(39)
    positions = torch.tensor(CANONICAL_21_POSITIONS)
    model = make_model(
        "cardinal_fbc_corr",
        n_channels=21,
        n_outputs=4,
        n_times=320,
        sfreq=128.0,
        channel_names=tuple(CANONICAL_21_CHANNELS),
        channel_positions=positions,
    ).train()
    assert isinstance(model, CardinalFBCCorrelationNet)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    values = torch.randn(6, 21, 320)
    labels = torch.tensor([0, 1, 2, 3, 0, 1])

    torch.nn.functional.cross_entropy(model(values, positions), labels).backward()
    new_columns = model.final_layer.weight.grad[:, model.floor_feature_count :]
    assert torch.count_nonzero(new_columns) > 0
    optimizer.step()
    assert torch.count_nonzero(
        model.final_layer.weight[:, model.floor_feature_count :]
    ) > 0

    optimizer.zero_grad(set_to_none=True)
    torch.nn.functional.cross_entropy(model(values, positions), labels).backward()
    assert model.correlation_projection.grad is not None
    assert torch.isfinite(model.correlation_projection.grad).all()
    assert torch.count_nonzero(model.correlation_projection.grad) > 0


def test_cardinal_fbc_micro_starts_at_exact_single_head_fbc_floor() -> None:
    positions = torch.tensor(CANONICAL_21_POSITIONS)
    torch.manual_seed(31)
    indexed = make_model(
        "fbcnet",
        n_channels=21,
        n_outputs=4,
        n_times=320,
        sfreq=128.0,
        channel_names=tuple(CANONICAL_21_CHANNELS),
        channel_positions=positions,
    ).train()
    torch.manual_seed(31)
    micro = make_model(
        "cardinal_fbc_micro",
        n_channels=21,
        n_outputs=4,
        n_times=320,
        sfreq=128.0,
        channel_names=tuple(CANONICAL_21_CHANNELS),
        channel_positions=positions,
    ).train()
    assert isinstance(micro, CardinalFBCMicroDynamicsNet)
    values = torch.randn(8, 21, 320)
    assert torch.allclose(
        indexed(values), micro(values, positions), atol=1e-4, rtol=1e-4
    )
    assert torch.count_nonzero(
        micro.final_layer.weight[:, micro.floor_feature_count :]
    ) == 0
    assert micro.final_layer.in_features == (
        micro.floor_feature_count + micro.micro_feature_count
    )
    assert sum(parameter.numel() for parameter in micro.parameters()) < 25_000


def test_cardinal_fbc_micro_is_permutation_invariant() -> None:
    torch.manual_seed(32)
    positions = torch.tensor(CANONICAL_21_POSITIONS)
    model = make_model(
        "cardinal_fbc_micro",
        n_channels=21,
        n_outputs=4,
        n_times=320,
        sfreq=128.0,
        channel_names=tuple(CANONICAL_21_CHANNELS),
    ).eval()
    values = torch.randn(2, 21, 320)
    permutation = torch.randperm(21)
    direct = model(values, positions)
    reordered = model(values[:, permutation], positions[permutation])
    assert torch.allclose(direct, reordered, atol=3e-5, rtol=3e-5)


def test_cardinal_fbc_micro_branch_learns_after_head_warm_start() -> None:
    torch.manual_seed(33)
    positions = torch.tensor(CANONICAL_21_POSITIONS)
    model = make_model(
        "cardinal_fbc_micro",
        n_channels=21,
        n_outputs=4,
        n_times=320,
        sfreq=128.0,
        channel_names=tuple(CANONICAL_21_CHANNELS),
        channel_positions=positions,
    ).train()
    assert isinstance(model, CardinalFBCMicroDynamicsNet)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    values = torch.randn(6, 21, 320)
    labels = torch.tensor([0, 1, 2, 3, 0, 1])

    torch.nn.functional.cross_entropy(model(values, positions), labels).backward()
    new_columns = model.final_layer.weight.grad[:, model.floor_feature_count :]
    assert torch.count_nonzero(new_columns) > 0
    optimizer.step()
    assert torch.count_nonzero(
        model.final_layer.weight[:, model.floor_feature_count :]
    ) > 0

    optimizer.zero_grad(set_to_none=True)
    torch.nn.functional.cross_entropy(model(values, positions), labels).backward()
    branch_parameters = (
        model.micro_filter_bank.raw_kernels,
        model.micro_spatiotemporal_field.coefficients,
        model.micro_dynamics_depthwise.weight,
    )
    assert all(parameter.grad is not None for parameter in branch_parameters)
    assert all(torch.count_nonzero(parameter.grad) > 0 for parameter in branch_parameters)


def test_cardinal_fbc_micro_supports_fixed_conservative_feature_scale() -> None:
    positions = torch.tensor(CANONICAL_21_POSITIONS)

    def build(scale: float) -> CardinalFBCMicroDynamicsNet:
        torch.manual_seed(40)
        base = make_model(
            "fbcnet",
            n_channels=21,
            n_outputs=4,
            n_times=320,
            sfreq=128.0,
            channel_names=tuple(CANONICAL_21_CHANNELS),
            channel_positions=positions,
        )
        return CardinalFBCMicroDynamicsNet(
            spectral_filtering=base.spectral_filtering,
            batch_norm=base.spatial_conv[1],
            activation=base.spatial_conv[2],
            padding_layer=base.padding_layer,
            temporal_layer=base.temporal_layer,
            flatten_layer=base.flatten_layer,
            final_layer=base.final_layer,
            n_times=320,
            sfreq=128.0,
            n_bands=base.n_bands,
            n_sources=base.n_filters_spat,
            stride_factor=base.stride_factor,
            indexed_spatial_weights=base.spatial_conv[0]
            .weight.detach()
            .reshape(base.n_bands, base.n_filters_spat, 21),
            indexed_spatial_bias=base.spatial_conv[0]
            .bias.detach()
            .reshape(base.n_bands, base.n_filters_spat),
            observed_positions=positions,
            continuation_scale=scale,
        ).eval()

    unscaled = build(1.0)
    scaled = build(0.25)
    values = torch.randn(2, 21, 320)
    with torch.inference_mode():
        torch.testing.assert_close(
            scaled.encode_micro(values, positions),
            0.25 * unscaled.encode_micro(values, positions),
            rtol=2e-5,
            atol=2e-5,
        )
        torch.testing.assert_close(
            scaled(values, positions),
            unscaled(values, positions),
            rtol=1e-4,
            atol=1e-4,
        )
    assert scaled.continuation_scale == 0.25
    assert scaled.config["continuation_scale"] == 0.25
    assert sum(p.numel() for p in scaled.parameters()) == sum(
        p.numel() for p in unscaled.parameters()
    )


def test_cardinal_fbc_compactdyn_preserves_reference_and_feature_budget() -> None:
    positions = torch.tensor(CANONICAL_21_POSITIONS)
    torch.manual_seed(41)
    indexed = make_model(
        "fbcnet",
        n_channels=21,
        n_outputs=4,
        n_times=320,
        sfreq=128.0,
        channel_names=tuple(CANONICAL_21_CHANNELS),
        channel_positions=positions,
    ).train()
    torch.manual_seed(41)
    hybrid = make_model(
        "cardinal_fbc_compactdyn",
        n_channels=21,
        n_outputs=4,
        n_times=320,
        sfreq=128.0,
        channel_names=tuple(CANONICAL_21_CHANNELS),
        channel_positions=positions,
    ).train()
    assert isinstance(hybrid, CardinalFBCCompactDynamicsNet)
    values = torch.randn(8, 21, 320)
    torch.testing.assert_close(
        hybrid(values, positions), indexed(values), rtol=1e-4, atol=1e-4
    )
    features = hybrid.encode_compact_dynamics(values, positions)
    assert features.shape == (len(values), 384)
    assert hybrid.compact_dynamics_feature_count == 384
    assert isinstance(hybrid.compact_dynamics.classifier, torch.nn.Identity)
    assert hybrid.compact_dynamics.config.dropout == 0.0
    assert torch.count_nonzero(
        hybrid.final_layer.weight[:, hybrid.floor_feature_count :]
    ) == 0
    assert hybrid.final_layer.in_features == hybrid.floor_feature_count + 384
    assert sum(parameter.numel() for parameter in hybrid.parameters()) < 45_000


def test_cardinal_fbc_compactdyn_fixed_scales_preserve_reference() -> None:
    positions = torch.tensor(CANONICAL_21_POSITIONS)
    values = torch.randn(2, 21, 320)
    variants = {
        "cardinal_fbc_compactdyn_scale010": 0.10,
        "cardinal_fbc_compactdyn_scale010_extended": 0.10,
        "cardinal_fbc_compactdyn_scale025": 0.25,
        "cardinal_fbc_compactdyn_scale025_extended": 0.25,
    }
    for name, scale in variants.items():
        unscaled_name = (
            "cardinal_fbc_compactdyn_extended"
            if name.endswith("_extended")
            else "cardinal_fbc_compactdyn"
        )
        torch.manual_seed(44)
        indexed = make_model(
            "fbcnet",
            n_channels=21,
            n_outputs=4,
            n_times=320,
            sfreq=128.0,
            channel_names=tuple(CANONICAL_21_CHANNELS),
            channel_positions=positions,
        ).eval()
        torch.manual_seed(44)
        unscaled = make_model(
            unscaled_name,
            n_channels=21,
            n_outputs=4,
            n_times=320,
            sfreq=128.0,
            channel_names=tuple(CANONICAL_21_CHANNELS),
            channel_positions=positions,
        ).eval()
        torch.manual_seed(44)
        scaled = make_model(
            name,
            n_channels=21,
            n_outputs=4,
            n_times=320,
            sfreq=128.0,
            channel_names=tuple(CANONICAL_21_CHANNELS),
            channel_positions=positions,
        ).eval()
        assert isinstance(scaled, CardinalFBCCompactDynamicsNet)
        with torch.inference_mode():
            torch.testing.assert_close(
                scaled(values, positions),
                indexed(values),
                rtol=1e-4,
                atol=1e-4,
            )
            torch.testing.assert_close(
                scaled.encode_compact_dynamics(values, positions),
                scale * unscaled.encode_compact_dynamics(values, positions),
                rtol=2e-5,
                atol=2e-5,
            )
        assert scaled.continuation_scale == scale
        assert scaled.config["continuation_scale"] == scale
        assert torch.count_nonzero(
            scaled.final_layer.weight[:, scaled.floor_feature_count :]
        ) == 0
        assert sum(parameter.numel() for parameter in scaled.parameters()) < 45_000


def test_cardinal_fbc_compactdyn_extended_is_permutation_invariant() -> None:
    torch.manual_seed(42)
    positions = torch.tensor(CANONICAL_21_POSITIONS)
    model = make_model(
        "cardinal_fbc_compactdyn_scale025_extended",
        n_channels=21,
        n_outputs=4,
        n_times=320,
        sfreq=128.0,
        channel_names=tuple(CANONICAL_21_CHANNELS),
    ).eval()
    assert isinstance(model, CardinalFBCCompactDynamicsNet)
    values = torch.randn(3, 21, 320)
    permutation = torch.randperm(21)
    with torch.inference_mode():
        direct = model(values, positions)
        reordered = model(values[:, permutation], positions[permutation])
        direct_features = model.encode_compact_dynamics(values, positions)
        reordered_features = model.encode_compact_dynamics(
            values[:, permutation], positions[permutation]
        )
    torch.testing.assert_close(reordered, direct, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(
        reordered_features, direct_features, rtol=1e-4, atol=1e-4
    )
    assert sum(parameter.numel() for parameter in model.parameters()) < 45_000


def test_cardinal_fbc_compactdyn_branch_learns_after_head_warm_start() -> None:
    torch.manual_seed(43)
    positions = torch.tensor(CANONICAL_21_POSITIONS)
    model = make_model(
        "cardinal_fbc_compactdyn_scale010",
        n_channels=21,
        n_outputs=4,
        n_times=320,
        sfreq=128.0,
        channel_names=tuple(CANONICAL_21_CHANNELS),
        channel_positions=positions,
    ).train()
    assert isinstance(model, CardinalFBCCompactDynamicsNet)
    assert model.continuation_scale == 0.10
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    values = torch.randn(6, 21, 320)
    labels = torch.tensor([0, 1, 2, 3, 0, 1])

    torch.nn.functional.cross_entropy(model(values, positions), labels).backward()
    new_columns = model.final_layer.weight.grad[:, model.floor_feature_count :]
    assert torch.count_nonzero(new_columns) > 0
    optimizer.step()
    assert torch.count_nonzero(
        model.final_layer.weight[:, model.floor_feature_count :]
    ) > 0

    optimizer.zero_grad(set_to_none=True)
    torch.nn.functional.cross_entropy(model(values, positions), labels).backward()
    branch_parameters = (
        model.compact_dynamics.temporal.weight,
        model.compact_dynamics.spatiotemporal_field.coefficients,
        model.compact_dynamics.dynamics[0].weight,
    )
    assert all(parameter.grad is not None for parameter in branch_parameters)
    assert all(torch.isfinite(parameter.grad).all() for parameter in branch_parameters)
    assert all(torch.count_nonzero(parameter.grad) > 0 for parameter in branch_parameters)


def test_cardinal_fbc_physical_matches_fbc_reference_and_is_modest() -> None:
    positions = torch.tensor(CANONICAL_21_POSITIONS)
    values = torch.randn(8, 21, 320)
    for name in ("cardinal_fbc_physical", "cardinal_fbc_physical_extended"):
        torch.manual_seed(41)
        indexed = make_model(
            "fbcnet",
            n_channels=21,
            n_outputs=4,
            n_times=320,
            sfreq=128.0,
            channel_names=tuple(CANONICAL_21_CHANNELS),
            channel_positions=positions,
        ).train()
        torch.manual_seed(41)
        physical = make_model(
            name,
            n_channels=21,
            n_outputs=4,
            n_times=320,
            sfreq=128.0,
            channel_names=tuple(CANONICAL_21_CHANNELS),
            channel_positions=positions,
        ).train()
        assert isinstance(physical, CardinalFBCPhysicalDynamicsNet)
        assert isinstance(physical.physical_filter_bank, OrderedSincFilterBank)
        logits = physical(values, positions)
        assert torch.isfinite(logits).all()
        assert torch.allclose(indexed(values), logits, atol=1e-4, rtol=1e-4)
        assert torch.count_nonzero(
            physical.final_layer.weight[:, physical.floor_feature_count :]
        ) == 0
        assert physical.physical_filter_bank.n_filters == 16
        assert physical.physical_filter_bank.kernel_size == 65
        assert physical.physical_filter_bank.stride == 2
        assert physical.physical_sources == 12
        assert physical.energy_windows == 8
        assert physical.dynamics_channels == 8
        assert physical.physical_feature_count == 112
        assert sum(parameter.numel() for parameter in physical.parameters()) < 25_000


def test_cardinal_fbc_physical_is_permutation_invariant() -> None:
    torch.manual_seed(42)
    positions = torch.tensor(CANONICAL_21_POSITIONS)
    model = make_model(
        "cardinal_fbc_physical",
        n_channels=21,
        n_outputs=4,
        n_times=320,
        sfreq=128.0,
        channel_names=tuple(CANONICAL_21_CHANNELS),
    ).eval()
    values = torch.randn(2, 21, 320)
    permutation = torch.randperm(21)
    direct = model(values, positions)
    reordered = model(values[:, permutation], positions[permutation])
    assert torch.isfinite(direct).all()
    assert torch.allclose(direct, reordered, atol=3e-5, rtol=3e-5)


def test_cardinal_fbc_physical_branch_learns_after_head_warm_start() -> None:
    torch.manual_seed(43)
    positions = torch.tensor(CANONICAL_21_POSITIONS)
    model = make_model(
        "cardinal_fbc_physical",
        n_channels=21,
        n_outputs=4,
        n_times=320,
        sfreq=128.0,
        channel_names=tuple(CANONICAL_21_CHANNELS),
        channel_positions=positions,
    ).train()
    assert isinstance(model, CardinalFBCPhysicalDynamicsNet)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    values = torch.randn(6, 21, 320)
    labels = torch.tensor([0, 1, 2, 3, 0, 1])

    loss = torch.nn.functional.cross_entropy(model(values, positions), labels)
    assert torch.isfinite(loss)
    loss.backward()
    new_columns = model.final_layer.weight.grad[:, model.floor_feature_count :]
    assert torch.isfinite(new_columns).all()
    assert torch.count_nonzero(new_columns) > 0
    optimizer.step()

    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.cross_entropy(model(values, positions), labels)
    assert torch.isfinite(loss)
    loss.backward()
    branch_parameters = (
        model.physical_filter_bank.raw_frequency_gaps,
        model.physical_filter_bank.raw_bandwidths,
        model.physical_spatiotemporal_field.coefficients,
        model.physical_dynamics_depthwise.weight,
        model.physical_dynamics_pointwise.weight,
    )
    assert all(parameter.grad is not None for parameter in branch_parameters)
    assert all(torch.isfinite(parameter.grad).all() for parameter in branch_parameters)
    assert all(torch.count_nonzero(parameter.grad) > 0 for parameter in branch_parameters)


def test_cardinal_fbms_extended_is_permutation_invariant() -> None:
    torch.manual_seed(47)
    positions = torch.tensor(CANONICAL_21_POSITIONS)
    model = make_model(
        "cardinal_fbms_extended",
        n_channels=21,
        n_outputs=4,
        n_times=320,
        sfreq=128.0,
        channel_names=tuple(CANONICAL_21_CHANNELS),
    ).eval()
    assert isinstance(model, CardinalFBMSNet)
    assert model.config["reference_architecture"] == "Braindecode FBMSNet"
    assert model.config["extended_atlas"] is True
    assert model.config["n_temporal_views"] == 36
    assert model.config["dilatability"] == 8
    assert model.config["paired_fbms_initialization_fit"] is None
    assert model.config["replaced_component"] == (
        "grouped_channel_indexed_spatial_conv_only"
    )
    values = torch.randn(2, 21, 320)
    permutation = torch.randperm(21)
    with torch.inference_mode():
        direct = model(values, positions)
        reordered = model(values[:, permutation], positions[permutation])
    assert torch.isfinite(direct).all()
    torch.testing.assert_close(reordered, direct, rtol=4e-5, atol=4e-5)


def test_cardinal_fbms_has_finite_nonzero_spatial_gradients() -> None:
    torch.manual_seed(48)
    positions = torch.tensor(CANONICAL_21_POSITIONS)
    model = make_model(
        "cardinal_fbms",
        n_channels=21,
        n_outputs=4,
        n_times=320,
        sfreq=128.0,
        channel_names=tuple(CANONICAL_21_CHANNELS),
        channel_positions=positions,
    ).train()
    assert isinstance(model, CardinalFBMSNet)
    assert model.config["paired_fbms_initialization_fit"] == "exact_minimum_norm"
    assert model.config["paired_fbms_basis_rank"] == 21
    values = torch.randn(6, 21, 320)
    labels = torch.tensor([0, 1, 2, 3, 0, 1])
    loss = torch.nn.functional.cross_entropy(model(values, positions), labels)
    assert torch.isfinite(loss)
    loss.backward()
    parameters = (
        model.spatial_field.coefficients,
        model.spatial_bias,
        model.mix_conv[0].convs[0].weight,
    )
    assert all(parameter.grad is not None for parameter in parameters)
    assert all(torch.isfinite(parameter.grad).all() for parameter in parameters)
    assert all(torch.count_nonzero(parameter.grad) > 0 for parameter in parameters)


def test_eegsym_deterministic_pool_has_backward() -> None:
    torch.use_deterministic_algorithms(True)
    model = make_model(
        "eegsym",
        n_channels=3,
        n_outputs=2,
        n_times=320,
        sfreq=128.0,
        channel_names=tuple(BNCI004_3_CHANNELS),
    ).train()
    loss = model(torch.randn(2, 3, 320)).square().mean()
    loss.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_pinned_official_tcformer_adapter_when_checkout_is_available() -> None:
    checkout = Path.cwd() / "third_party" / "TCFormer" / "models" / "tcformer.py"
    if not checkout.exists():
        return
    model = make_model(
        "tcformer",
        n_channels=21,
        n_outputs=4,
        n_times=320,
        sfreq=128.0,
        channel_names=tuple(CANONICAL_21_CHANNELS),
    ).eval()
    output = model(torch.randn(2, 21, 320))
    assert output.shape == (2, 4)
    assert 50_000 < sum(p.numel() for p in model.parameters()) < 150_000
