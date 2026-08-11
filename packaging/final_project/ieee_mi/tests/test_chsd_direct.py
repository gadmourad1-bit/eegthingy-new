from __future__ import annotations

import torch

from ieee_mi.chsd_direct import (
    CHSDDirectNet,
    DIRECT_MODEL_NAME,
    make_chsd_direct_model,
)


def _positions(channels: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(91 + channels)
    value = torch.randn(channels, 3, generator=generator)
    value[:, 2] = value[:, 2].abs() + 0.2
    return value / torch.linalg.vector_norm(value, dim=-1, keepdim=True)


def test_direct_factory_and_locked_shapes_are_finite() -> None:
    for channels, outputs, times in ((3, 2, 320), (15, 2, 256), (21, 4, 320)):
        model = make_chsd_direct_model(
            requested_model=DIRECT_MODEL_NAME,
            n_channels=channels,
            n_outputs=outputs,
            n_times=times,
            sfreq=128.0,
            channel_names=tuple(f"EEG-{index}" for index in range(channels)),
            channel_positions=_positions(channels),
        ).eval()
        with torch.no_grad():
            logits = model(torch.randn(2, channels, times), _positions(channels))
        assert logits.shape == (2, outputs)
        assert torch.isfinite(logits).all()


def test_direct_residuals_start_zero_then_receive_finite_gradients() -> None:
    torch.manual_seed(92)
    model = CHSDDirectNet(n_channels=3, n_outputs=2, n_times=256).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    x = torch.randn(4, 3, 256)
    positions = _positions(3)
    model.fit_source_statistics(x, positions, batch_size=2)
    assert bool(model.state_norm.fitted)
    assert all(bool(normalizer.fitted) for normalizer in model.residual_norms)
    assert bool(model.power_norm.fitted)
    assert all(torch.count_nonzero(head.weight) == 0 for head in model.residual_heads)
    assert torch.count_nonzero(model.power_head.weight) == 0
    first_loss = model(x, positions).square().mean()
    first_loss.backward()
    assert all(
        head.weight.grad is not None
        and torch.isfinite(head.weight.grad).all()
        and torch.count_nonzero(head.weight.grad) > 0
        for head in model.residual_heads
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    second_loss = model(x, positions).square().mean()
    second_loss.backward()
    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
