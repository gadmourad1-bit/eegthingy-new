from __future__ import annotations

import numpy as np
import torch

from benchmark.research.parity_net import ConditionalOddReadout, HemiParityNet


def test_conditional_readout_is_odd_with_invariant_context() -> None:
    model = ConditionalOddReadout(7, rank=3, dropout=0.0).eval()
    even = torch.randn(5, 7)
    odd = torch.randn(5, 7)
    first = model(even, odd)
    reflected = model(even, -odd)
    assert torch.allclose(first, -reflected, atol=1e-7, rtol=1e-6)


def test_full_parity_network_has_exact_signed_label_action() -> None:
    model = HemiParityNet(
        n_channels=15,
        n_times=251,
        n_tangent_features=480,
        tangent_initial_weight=np.zeros(480, dtype=np.float32),
        temporal_filters=4,
        dynamics_channels=4,
        raw_rank=3,
        tangent_rank=2,
        dropout=0.0,
    ).eval()
    raw = torch.randn(3, 15, 251)
    raw_reflected = torch.randn(3, 15, 251)
    tangent = torch.randn(3, 480)
    tangent_reflected = torch.randn(3, 480)

    first = model(raw, raw_reflected, tangent, tangent_reflected)
    reflected = model(raw_reflected, raw, tangent_reflected, tangent)
    assert torch.allclose(first.logit, -reflected.logit, atol=1e-6, rtol=1e-6)
    assert torch.allclose(first.raw_logit, -reflected.raw_logit, atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        first.tangent_logit, -reflected.tangent_logit, atol=1e-6, rtol=1e-6
    )
    assert torch.allclose(
        first.fusion_weights, reflected.fusion_weights, atol=1e-7, rtol=1e-6
    )

