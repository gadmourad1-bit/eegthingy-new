from __future__ import annotations

import copy
import os

import pytest

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from benchmark.chsd_conditioned import (
    CONDITIONED_MODEL_VARIANTS,
    HSD_FIELD_NAMES,
    NATIVE_MODEL_NAME,
    CHSDSurfaceConditionedCardinalFBC,
    make_chsd_conditioned_model,
)


def _positions(channels: int, *, seed: int = 191) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed + channels)
    positions = torch.randn(channels, 3, generator=generator)
    positions[:, 2] = positions[:, 2].abs() + 0.2
    return positions / torch.linalg.vector_norm(
        positions, dim=-1, keepdim=True
    )


def _names(channels: int) -> tuple[str, ...]:
    return tuple(f"EEG-{index:02d}" for index in range(channels))


def _make(
    *,
    requested_model: str = "chsdnet_conditioned_010",
    channels: int = 3,
    outputs: int = 2,
    times: int = 256,
) -> CHSDSurfaceConditionedCardinalFBC:
    return make_chsd_conditioned_model(
        requested_model=requested_model,
        n_channels=channels,
        n_outputs=outputs,
        n_times=times,
        sfreq=128.0,
        channel_names=_names(channels),
        channel_positions=_positions(channels),
    )


@pytest.mark.parametrize("training", (False, True))
def test_zero_conditioner_exactly_preserves_native_function(
    training: bool,
) -> None:
    torch.manual_seed(192)
    model = _make(channels=3, times=256)
    native = copy.deepcopy(model.native_backbone)
    model.train(training)
    native.train(training)
    x = torch.randn(3, 3, 256)
    positions = _positions(3)

    # The native path has no stochastic layer, but resetting the RNG here also
    # protects this invariant if a native regularizer is introduced later.
    torch.manual_seed(193)
    expected = native(x, positions)
    torch.manual_seed(193)
    actual = model(x, positions)
    assert torch.equal(actual, expected)
    assert torch.count_nonzero(model.conditioner_head.weight) == 0
    assert torch.count_nonzero(model.conditioner_head.bias) == 0
    assert torch.count_nonzero(model.modulation(x, positions)) == 0


def test_native_parameter_gradients_match_standalone_at_initialization() -> None:
    torch.manual_seed(194)
    model = _make(channels=3, times=256).train()
    native = copy.deepcopy(model.native_backbone).train()
    x = torch.randn(4, 3, 256)
    positions = _positions(3)
    labels = torch.tensor((0, 1, 1, 0))

    torch.manual_seed(195)
    native_loss = torch.nn.functional.cross_entropy(
        native(x, positions), labels
    )
    native_loss.backward()
    torch.manual_seed(195)
    conditioned_loss = torch.nn.functional.cross_entropy(
        model(x, positions), labels
    )
    conditioned_loss.backward()

    actual_parameters = dict(model.native_backbone.named_parameters())
    expected_parameters = dict(native.named_parameters())
    assert actual_parameters.keys() == expected_parameters.keys()
    matched_nonzero_gradients = 0
    for name, expected_parameter in expected_parameters.items():
        actual_gradient = actual_parameters[name].grad
        expected_gradient = expected_parameter.grad
        assert (actual_gradient is None) == (expected_gradient is None), name
        if expected_gradient is None:
            continue
        assert actual_gradient is not None
        assert torch.equal(actual_gradient, expected_gradient), name
        matched_nonzero_gradients += int(
            torch.count_nonzero(expected_gradient) > 0
        )
    assert matched_nonzero_gradients > 0


def test_first_step_reaches_conditioner_but_not_upstream_chsd() -> None:
    torch.manual_seed(196)
    model = _make(channels=3, times=256).train()
    x = torch.randn(4, 3, 256)
    positions = _positions(3)
    coefficients = torch.tensor(
        ((0.7, -0.2), (-0.4, 0.9), (0.3, 0.5), (-0.8, 0.1))
    )
    loss = (model(x, positions) * coefficients).sum()
    loss.backward()

    head_gradients = (
        model.conditioner_head.weight.grad,
        model.conditioner_head.bias.grad,
    )
    assert all(gradient is not None for gradient in head_gradients)
    assert all(
        torch.isfinite(gradient).all()
        for gradient in head_gradients
        if gradient is not None
    )
    assert any(
        torch.count_nonzero(gradient) > 0
        for gradient in head_gradients
        if gradient is not None
    )

    chsd_gradients = [
        parameter.grad
        for parameter in model.chsd_encoder.parameters()
        if parameter.requires_grad
    ]
    assert chsd_gradients
    assert all(gradient is not None for gradient in chsd_gradients)
    assert all(
        torch.isfinite(gradient).all()
        for gradient in chsd_gradients
        if gradient is not None
    )
    assert all(
        torch.count_nonzero(gradient) == 0
        for gradient in chsd_gradients
        if gradient is not None
    )
    upstream_conditioner_gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if name.startswith(
            (
                "global_",
                "explicit_fields_norm.",
                "conditioner_input_norm.",
                "conditioner_hidden.",
                "conditioner_hidden_norm.",
            )
        )
    ]
    assert upstream_conditioner_gradients
    assert all(
        gradient is not None and torch.isfinite(gradient).all()
        for gradient in upstream_conditioner_gradients
    )
    assert all(
        torch.count_nonzero(gradient) == 0
        for gradient in upstream_conditioner_gradients
        if gradient is not None
    )


@pytest.mark.parametrize(
    ("channels", "times", "outputs"),
    ((3, 320, 2), (15, 256, 2), (21, 320, 4)),
)
def test_locked_shapes_are_finite_and_eval_is_deterministic(
    channels: int,
    times: int,
    outputs: int,
) -> None:
    torch.manual_seed(197 + channels)
    model = _make(
        channels=channels,
        times=times,
        outputs=outputs,
    ).eval()
    x = torch.randn(1, channels, times)
    positions = _positions(channels)
    with torch.no_grad():
        first = model(x, positions)
        second = model(x, positions)
        conditioner_logits = model.conditioner_logits(x, positions)
        modulation = model.modulation(x, positions)
    assert first.shape == (1, outputs)
    assert conditioner_logits.shape == (1, 9, 4)
    assert modulation.shape == (1, 9, 4)
    assert torch.isfinite(first).all()
    assert torch.isfinite(modulation).all()
    assert torch.equal(first, second)
    assert torch.equal(modulation, torch.zeros_like(modulation))


def test_variants_and_metadata_are_prespecified_and_auditable() -> None:
    assert dict(CONDITIONED_MODEL_VARIANTS) == {
        "chsdnet_conditioned_005": 0.05,
        "chsdnet_conditioned_010": 0.10,
        "chsdnet_conditioned_020": 0.20,
    }
    for requested_model, maximum in CONDITIONED_MODEL_VARIANTS.items():
        torch.manual_seed(220)
        model = _make(requested_model=requested_model)
        assert model.maximum_modulation == maximum
        assert model.config["architecture"] == requested_model
        assert model.config["architecture_family"] == (
            "chsd_surface_conditioned_cardinal_fbc"
        )
        assert model.config["native_model"] == NATIVE_MODEL_NAME
        assert model.config["conditioner_fields"] == HSD_FIELD_NAMES
        assert model.config["conditioner_band_interpolation"] == {
            "method": "fixed_piecewise_linear_matrix",
            "endpoint_aligned": True,
            "source_count": 6,
            "target_count": 9,
            "matrix_buffer": "band_interpolation_matrix",
            "source_band_centers_hz": (6.0, 10.0, 14.0, 20.0, 28.0, 36.0),
            "target_source_index_coordinates": (
                0.0,
                0.625,
                1.25,
                1.875,
                2.5,
                3.125,
                3.75,
                4.375,
                5.0,
            ),
        }
        assert model.config["maximum_modulation"] == maximum
        assert model.config["zero_initialized_conditioner_head"] is True
        assert model.config["native_continuation_unchanged"] is True
        assert model.config["native_constrained_head_unchanged"] is True
        assert model.config["raw_chsd_concatenation"] is False
        assert model.config["residual_logit_branch"] is False
        interpolation = model.band_interpolation_matrix
        assert interpolation.shape == (9, 6)
        assert torch.all(interpolation >= 0.0)
        assert torch.equal(
            interpolation.sum(dim=1),
            torch.ones(9),
        )
        assert torch.all(torch.count_nonzero(interpolation, dim=1) <= 2)
        assert torch.equal(
            interpolation[0],
            torch.tensor((1.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
        )
        assert torch.equal(
            interpolation[-1],
            torch.tensor((0.0, 0.0, 0.0, 0.0, 0.0, 1.0)),
        )
        assert "band_interpolation_matrix" in dict(model.named_buffers())
        assert model.chsd_encoder.config.use_coordinate_projection is False
        assert model.chsd_encoder.config.n_virtual_sensors == 3
        assert model.chsd_encoder.config.matrix_log_terms == 64
        assert isinstance(model.chsd_encoder.classifier, torch.nn.Identity)


def test_factory_rejects_unregistered_variant_and_bad_metadata() -> None:
    positions = _positions(3)
    kwargs = {
        "n_channels": 3,
        "n_outputs": 2,
        "n_times": 256,
        "sfreq": 128.0,
        "channel_names": _names(3),
        "channel_positions": positions,
    }
    with pytest.raises(ValueError, match="unsupported requested model"):
        make_chsd_conditioned_model(
            requested_model="chsdnet_conditioned_tuned",
            **kwargs,
        )
    with pytest.raises(ValueError, match="unique"):
        make_chsd_conditioned_model(
            requested_model="chsdnet_conditioned_010",
            **{**kwargs, "channel_names": ("Cz", "Cz", "Pz")},
        )
    with pytest.raises(ValueError, match="shape"):
        make_chsd_conditioned_model(
            requested_model="chsdnet_conditioned_010",
            **{**kwargs, "channel_positions": positions[:, :2]},
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_conditioned_model_strict_cuda_forward_backward() -> None:
    torch.manual_seed(221)
    torch.cuda.manual_seed_all(221)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda")
    try:
        for channels, times, outputs in (
            (3, 320, 2),
            (15, 256, 2),
            (21, 320, 4),
        ):
            positions = _positions(channels).to(device)
            model = make_chsd_conditioned_model(
                requested_model="chsdnet_conditioned_010",
                n_channels=channels,
                n_outputs=outputs,
                n_times=times,
                sfreq=128.0,
                channel_names=_names(channels),
                channel_positions=positions,
            ).to(device).train()
            x = torch.randn(2, channels, times, device=device)
            labels = torch.arange(2, device=device) % outputs
            logits = model(x, positions)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            loss.backward()
            assert logits.shape == (2, outputs)
            assert torch.isfinite(logits).all()
            gradients = [
                gradient
                for parameter in model.parameters()
                if (gradient := parameter.grad) is not None
            ]
            assert gradients
            assert all(torch.isfinite(gradient).all() for gradient in gradients)
    finally:
        torch.use_deterministic_algorithms(False)
