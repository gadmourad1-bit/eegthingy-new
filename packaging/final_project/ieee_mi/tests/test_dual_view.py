from __future__ import annotations

import torch

from ieee_mi.baselines import make_model
from ieee_mi.config import CANONICAL_21_CHANNELS
from ieee_mi.dual_view import (
    CardinalSplineDualViewNet,
    cardinal_spline_dual_view_loss,
)
from ieee_mi.models import CANONICAL_21_POSITIONS, CardinalFBCNet, parameter_count


def _candidate(*, n_channels: int, positions: torch.Tensor) -> CardinalSplineDualViewNet:
    torch.manual_seed(20260719)
    backbone = make_model(
        "cardinal_fbc",
        n_channels=n_channels,
        n_outputs=2,
        n_times=320,
        sfreq=128.0,
        channel_names=tuple(CANONICAL_21_CHANNELS[:n_channels]),
        channel_positions=positions,
    )
    assert isinstance(backbone, CardinalFBCNet)
    return CardinalSplineDualViewNet(backbone)


def test_identical_scaled_views_have_identical_logits_and_declared_budget() -> None:
    positions = torch.tensor(CANONICAL_21_POSITIONS)
    candidate = _candidate(n_channels=21, positions=positions).eval()
    values = torch.randn(3, 21, 320)
    output = candidate.forward_views(values, values.clone(), positions)

    torch.testing.assert_close(
        output.direct_logits, output.canonical_logits, rtol=1e-6, atol=1e-6
    )
    torch.testing.assert_close(output.canonical_weight, torch.full((3, 1), 0.5))
    torch.testing.assert_close(output.fused_logits, output.direct_logits)
    torch.testing.assert_close(
        output.disagreement_descriptor, torch.zeros(3, 27), atol=1e-6, rtol=0.0
    )
    assert candidate.added_parameter_count == 233
    assert parameter_count(candidate) == parameter_count(candidate.backbone) + 233


def test_zero_initialized_gate_is_exact_equal_logit_fusion_and_learns() -> None:
    native_channels = 15
    positions = torch.tensor(CANONICAL_21_POSITIONS[:native_channels])
    candidate = _candidate(n_channels=native_channels, positions=positions).train()
    values = torch.randn(6, native_channels, 320)
    raw_transport = torch.rand(21, native_channels)
    transport = raw_transport / raw_transport.sum(dim=1, keepdim=True)
    canonical_values = torch.einsum("ac,bct->bat", transport, values)

    output = candidate.forward_views(values, canonical_values, positions)
    torch.testing.assert_close(output.canonical_weight, torch.full((6, 1), 0.5))
    torch.testing.assert_close(
        output.fused_logits,
        0.5 * (output.direct_logits + output.canonical_logits),
    )
    losses = cardinal_spline_dual_view_loss(
        output, torch.tensor([0, 1, 0, 1, 0, 1])
    )
    losses.total.backward()
    final_gate = candidate.preference_gate[-1]
    assert isinstance(final_gate, torch.nn.Linear)
    assert final_gate.weight.grad is not None
    assert torch.count_nonzero(final_gate.weight.grad) > 0


def test_native_permutation_leaves_both_views_unchanged() -> None:
    native_channels = 15
    positions = torch.tensor(CANONICAL_21_POSITIONS[:native_channels])
    candidate = _candidate(n_channels=native_channels, positions=positions).eval()
    values = torch.randn(2, native_channels, 320)
    raw_transport = torch.rand(21, native_channels)
    transport = raw_transport / raw_transport.sum(dim=1, keepdim=True)
    canonical_values = torch.einsum("ac,bct->bat", transport, values)
    permutation = torch.randperm(native_channels)

    first = candidate.forward_views(values, canonical_values, positions)
    second = candidate.forward_views(
        values[:, permutation],
        canonical_values,
        positions[permutation],
    )
    torch.testing.assert_close(first.fused_logits, second.fused_logits, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(first.direct_logits, second.direct_logits, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(first.canonical_logits, second.canonical_logits, atol=3e-5, rtol=3e-5)


def test_canonical_view_contract_fails_closed() -> None:
    positions = torch.tensor(CANONICAL_21_POSITIONS[:15])
    candidate = _candidate(n_channels=15, positions=positions)
    values = torch.randn(2, 15, 320)
    invalid = torch.ones(2, 20, 320)
    try:
        candidate(values, invalid, positions)
    except ValueError as error:
        assert "canonical_x has shape" in str(error)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("invalid canonical view was accepted")
