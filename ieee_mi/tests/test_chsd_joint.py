from __future__ import annotations

import copy

import torch

from ieee_mi import baselines
from ieee_mi.chsd_joint import (
    JOINT_MODEL_NAME,
    NATIVE_MODEL_NAME,
    make_chsd_joint_model,
)


def _positions(channels: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(111 + channels)
    value = torch.randn(channels, 3, generator=generator)
    value[:, 2] = value[:, 2].abs() + 0.2
    return value / torch.linalg.vector_norm(value, dim=-1, keepdim=True)


def _names(channels: int) -> tuple[str, ...]:
    return tuple(f"EEG-{index}" for index in range(channels))


def test_joint_head_preserves_native_mapping_exactly_at_initialization() -> None:
    torch.manual_seed(112)
    expected = baselines.make_model(
        NATIVE_MODEL_NAME,
        n_channels=15,
        n_outputs=2,
        n_times=256,
        sfreq=128.0,
        channel_names=_names(15),
        channel_positions=_positions(15),
    ).eval()
    expected_state = copy.deepcopy(expected.state_dict())
    torch.manual_seed(112)
    model = make_chsd_joint_model(
        requested_model=JOINT_MODEL_NAME,
        n_channels=15,
        n_outputs=2,
        n_times=256,
        sfreq=128.0,
        channel_names=_names(15),
        channel_positions=_positions(15),
    ).eval()
    for name, value in expected_state.items():
        if name.startswith("final_layer."):
            continue
        assert torch.equal(model.native_backbone.state_dict()[name], value)
    x = torch.randn(3, 15, 256)
    positions = _positions(15)
    with torch.no_grad():
        expected_logits = expected(x, positions)
        actual_logits = model(x, positions)
    # The zero-column extension is mathematically exact. A wider GEMM can
    # change floating-point accumulation by a few ulps.
    assert torch.allclose(actual_logits, expected_logits, atol=2e-6, rtol=2e-6)
    added = model.native_backbone.final_layer.weight[
        :, model.native_feature_count :
    ]
    assert torch.count_nonzero(added) == 0


def test_joint_supports_locked_shapes_and_chsd_columns_receive_gradient() -> None:
    for channels, outputs, times in ((3, 2, 320), (21, 4, 320)):
        torch.manual_seed(113 + channels)
        model = make_chsd_joint_model(
            requested_model=JOINT_MODEL_NAME,
            n_channels=channels,
            n_outputs=outputs,
            n_times=times,
            sfreq=128.0,
            channel_names=_names(channels),
            channel_positions=_positions(channels),
        ).train()
        logits = model(torch.randn(3, channels, times), _positions(channels))
        assert logits.shape == (3, outputs)
        loss = logits.square().mean()
        loss.backward()
        gradient = model.native_backbone.final_layer.weight.grad[
            :, model.native_feature_count :
        ]
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0
