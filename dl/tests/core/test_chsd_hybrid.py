from __future__ import annotations

import copy

import torch

from benchmark import baselines
from benchmark.chsd import CHSDConfig, CHSDNet
from benchmark.chsd_hybrid import (
    CHSDResidualNet,
    HYBRID_MODEL_NAME,
    NATIVE_MODEL_NAME,
    make_chsd_hybrid_model,
)


def _positions(channels: int, *, seed: int = 71) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed + channels)
    positions = torch.randn(channels, 3, generator=generator)
    positions[:, 2] = positions[:, 2].abs() + 0.2
    return positions / torch.linalg.vector_norm(
        positions, dim=-1, keepdim=True
    )


def _channel_names(channels: int) -> tuple[str, ...]:
    return tuple(f"EEG{index:02d}" for index in range(channels))


def _make(
    *,
    channels: int,
    n_outputs: int,
    n_times: int,
) -> CHSDResidualNet:
    model = make_chsd_hybrid_model(
        requested_model=HYBRID_MODEL_NAME,
        n_channels=channels,
        n_outputs=n_outputs,
        n_times=n_times,
        sfreq=128.0,
        channel_names=_channel_names(channels),
        channel_positions=_positions(channels),
    )
    assert isinstance(model, CHSDResidualNet)
    return model


def test_hybrid_exactly_preserves_native_logits_at_initialization() -> None:
    torch.manual_seed(72)
    model = _make(channels=15, n_outputs=2, n_times=256).eval()
    x = torch.randn(3, 15, 256)
    positions = _positions(15)
    with torch.no_grad():
        expected = model.native_backbone(x, positions)
        residual = model.chsd_branch(x, positions)
        actual = model(x, positions)
    assert torch.count_nonzero(model.residual_head.weight) > 0
    assert torch.isfinite(residual).all()
    assert torch.equal(model.residual_gate(), torch.tensor(0.0))
    assert torch.equal(actual, expected)


def test_hybrid_supports_locked_binary_multiclass_and_montage_shapes() -> None:
    torch.manual_seed(73)
    cases = (
        (3, 2, 320),
        (15, 2, 256),
        (21, 4, 320),
    )
    with torch.no_grad():
        for channels, n_outputs, n_times in cases:
            model = _make(
                channels=channels,
                n_outputs=n_outputs,
                n_times=n_times,
            ).eval()
            logits = model(
                torch.randn(2, channels, n_times),
                _positions(channels),
            )
            assert logits.shape == (2, n_outputs)
            assert torch.isfinite(logits).all()


def test_wrapper_changes_only_its_zero_gate_not_chsd_branch_state() -> None:
    torch.manual_seed(74)
    native = baselines.make_model(
        NATIVE_MODEL_NAME,
        n_channels=15,
        n_outputs=2,
        n_times=256,
        sfreq=128.0,
        channel_names=_channel_names(15),
        channel_positions=_positions(15),
    )
    branch = CHSDNet(2, config=CHSDConfig())
    before = copy.deepcopy(branch.state_dict())
    model = CHSDResidualNet(native, branch)
    for name, value in model.chsd_branch.state_dict().items():
        assert torch.equal(value, before[name])
    assert torch.equal(model.raw_residual_gate, torch.tensor(0.0))


def test_residual_head_then_hsd_projector_receive_two_step_gradients() -> None:
    torch.manual_seed(75)
    model = _make(channels=3, n_outputs=2, n_times=128).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=2e-3)
    x = torch.randn(4, 3, 128)
    positions = _positions(3)

    first_logits = model(x, positions)
    first_loss = first_logits.square().mean() + 0.2 * first_logits.mean()
    first_loss.backward()
    first_gate_gradient = model.raw_residual_gate.grad
    first_representation_gradient = (
        model.chsd_branch.field_projections[0].weight.grad
    )
    assert first_gate_gradient is not None
    assert torch.isfinite(first_gate_gradient)
    assert torch.count_nonzero(first_gate_gradient) > 0
    assert first_representation_gradient is not None
    assert torch.isfinite(first_representation_gradient).all()
    assert torch.count_nonzero(first_representation_gradient) == 0
    optimizer.step()
    assert torch.count_nonzero(model.raw_residual_gate) > 0

    optimizer.zero_grad(set_to_none=True)
    second_logits = model(x, positions)
    second_loss = second_logits.square().mean() + 0.2 * second_logits.mean()
    second_loss.backward()
    second_representation_gradient = (
        model.chsd_branch.field_projections[0].weight.grad
    )
    assert second_representation_gradient is not None
    assert torch.isfinite(second_representation_gradient).all()
    assert torch.count_nonzero(second_representation_gradient) > 0
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_factory_preserves_native_state_and_records_child_configs() -> None:
    positions = _positions(15)
    names = _channel_names(15)
    torch.manual_seed(76)
    expected_native = baselines.make_model(
        NATIVE_MODEL_NAME,
        n_channels=15,
        n_outputs=2,
        n_times=256,
        sfreq=128.0,
        channel_names=names,
        channel_positions=positions,
    )
    expected_state = copy.deepcopy(expected_native.state_dict())
    expected_config = copy.deepcopy(expected_native.config)

    torch.manual_seed(76)
    hybrid = make_chsd_hybrid_model(
        requested_model=HYBRID_MODEL_NAME,
        n_channels=15,
        n_outputs=2,
        n_times=256,
        sfreq=128.0,
        channel_names=names,
        channel_positions=positions,
    )
    actual_state = hybrid.native_backbone.state_dict()
    assert actual_state.keys() == expected_state.keys()
    for name, expected in expected_state.items():
        assert torch.equal(actual_state[name], expected)
    assert hybrid.native_backbone.config == expected_config
    assert hybrid.config["architecture"] == HYBRID_MODEL_NAME
    assert hybrid.config["native_model"] == NATIVE_MODEL_NAME
    assert hybrid.config["native_config"] == expected_config
    assert hybrid.config["maximum_residual_scale"] == 0.25
    assert hybrid.config["zero_initialized_residual_gate"] is True
    assert hybrid.chsd_branch.config.use_coordinate_projection is False
    assert hybrid.chsd_branch.config.n_virtual_sensors == 15
    assert hybrid.config["residual_config"] == {
        key: value for key, value in vars(hybrid.chsd_branch.config).items()
    }
    assert hybrid.uses_positions is True
