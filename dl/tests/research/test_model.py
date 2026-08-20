from __future__ import annotations

import torch
from torch.nn import functional as F

from benchmark.research.model import GeoAdaptNet, GeoAdaptOutput
from benchmark.shared.spd import matrix_log


def _covariances(batch: int = 5, bands: int = 4, channels: int = 15) -> torch.Tensor:
    generator = torch.Generator().manual_seed(23)
    factor = torch.randn(batch, bands, channels, channels, generator=generator)
    eye = torch.eye(channels)
    return factor @ factor.transpose(-1, -2) / channels + 0.1 * eye


def test_forward_is_structured_compact_and_anchor_preserving() -> None:
    model = GeoAdaptNet(dropout=0.0).eval()
    covariance = _covariances()
    output = model(covariance)
    assert isinstance(output, GeoAdaptOutput)
    assert output.logits.shape == (5, 2)
    assert output.anchor_logits.shape == output.residual_logits.shape == (5, 2)
    assert output.tangent_features.shape == (5, 4 * 15 * 16 // 2)
    assert output.band_features.shape[:2] == (5, 4)
    assert output.band_attention.shape == (5, 4)
    torch.testing.assert_close(output.band_attention.sum(dim=1), torch.ones(5))
    torch.testing.assert_close(output.log_odds, output.logits[:, 1] - output.logits[:, 0])
    torch.testing.assert_close(
        output.logits,
        output.anchor_logits + output.residual_gate * output.residual_logits,
    )
    assert 0.0 < float(output.residual_gate.detach()) < 0.03
    assert model.parameter_count < 50_000
    assert output.intent_logit is None


def test_forward_backward_reaches_anchor_residual_bimap_and_gate() -> None:
    model = GeoAdaptNet(dropout=0.0, auxiliary_intent=True)
    covariance = _covariances(batch=4).requires_grad_()
    labels = torch.tensor([0, 1, 0, 1])
    output = model(covariance)
    assert output.intent_logit is not None and output.intent_logit.shape == (4,)
    loss = F.cross_entropy(output.logits, labels) + F.binary_cross_entropy_with_logits(
        output.intent_logit, torch.ones(4)
    )
    loss.backward()

    assert covariance.grad is not None and torch.isfinite(covariance.grad).all()
    for parameter in (
        model.anchor_head.weight,
        model.residual_head.weight,
        model.bimaps[0].raw_weight,
        model.residual_gate_logit,
        model.intent_head.weight,
    ):
        assert parameter is not None and parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_shared_bandwise_and_per_sample_log_references() -> None:
    model = GeoAdaptNet(dropout=0.0).eval()
    covariance = _covariances(batch=3)
    shared = matrix_log(covariance.mean(dim=(0, 1)))
    bandwise = matrix_log(covariance.mean(dim=0))
    per_sample = matrix_log(covariance.mean(dim=1))
    full = matrix_log(covariance)

    for reference in (shared, bandwise, per_sample, full):
        output = model(covariance, log_reference=reference)
        assert output.logits.shape == (3, 2)
        assert torch.isfinite(output.logits).all()


def test_batch_one_training_forward_preserves_information() -> None:
    model = GeoAdaptNet(dropout=0.0).train()
    first = _covariances(batch=1)
    second = first * 1.7
    output_first = model(first)
    output_second = model(second)
    assert output_first.logits.shape == (1, 2)
    assert not torch.allclose(output_first.tangent_features, output_second.tangent_features)


def test_explicit_reference_preserves_geometry_while_source_norm_updates() -> None:
    model = GeoAdaptNet(dropout=0.0)
    covariance = _covariances(batch=4)
    reference = matrix_log(covariance.mean(dim=0))
    model.train()
    train_output = model(covariance, reference, update_reference=False)
    model.eval()
    eval_output = model(covariance, reference)
    # The explicit SPD reference produces identical geometric coordinates.  The
    # anchor logits legitimately differ because training updates/uses source-only
    # running feature statistics whereas evaluation freezes them.
    torch.testing.assert_close(train_output.tangent_features, eval_output.tangent_features)
    assert model.anchor_norm.num_batches_tracked.item() == 1
    assert torch.isfinite(eval_output.logits).all()


def test_real_eeg_voltage_scale_does_not_collapse_features() -> None:
    # Actual FIF-derived covariance entries are O(1e-10) V^2.  Samples must
    # remain distinguishable with the model's default numerical settings.
    model = GeoAdaptNet(dropout=0.0).eval()
    covariance = _covariances(batch=6) * 1e-10
    output = model(covariance)
    across_sample_std = output.tangent_features.std(dim=0).mean()
    assert float(across_sample_std) > 1e-3
    assert float(output.logits.detach().std(dim=0).mean()) > 1e-4
    assert torch.isfinite(output.logits).all()
