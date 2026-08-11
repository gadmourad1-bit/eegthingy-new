from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path
from types import MappingProxyType

import pytest
import torch
from torch import nn

from ieee_mi import baselines
from ieee_mi.gauge_quotient import (
    BANDS_HZ,
    FIR_UNIT_NORM_TOLERANCE,
    FIR_ZERO_DC_TOLERANCE,
    LAGS_SAMPLES,
    MAXIMUM_PARAMETERS,
    MODEL_NAME,
    NATIVE_MODEL_NAME,
    N_QUOTIENT_SOURCES,
    N_WINDOWS,
    GaugeQuotientCrossMomentNet,
    make_gauge_quotient_model,
)
from ieee_mi.gauge_quotient_screen import (
    DISJOINT115_RULE,
    DISJOINT115_RULE_JSON,
    DISJOINT115_RULE_SHA256,
    DISJOINT115_STAGE,
    EXPECTED_ROBUSTNESS_PLAN_SHA256,
    OPENED17_RULE,
    OPENED17_RULE_JSON,
    OPENED17_RULE_SHA256,
    OPENED17_STAGE,
    SCREEN_SCHEMA,
    candidate_jobs,
    reference_record_paths,
    screen_specification,
)
from ieee_mi.models import CardinalFBCMicroDynamicsNet


def _positions(channels: int, *, seed: int = 2_601) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed + channels)
    positions = torch.randn(channels, 3, generator=generator)
    positions[:, 2] = positions[:, 2].abs() + 0.25
    return positions / torch.linalg.vector_norm(
        positions,
        dim=-1,
        keepdim=True,
    )


def _names(channels: int) -> tuple[str, ...]:
    return tuple(f"EEG-{index:02d}" for index in range(channels))


def _native(
    *,
    channels: int = 3,
    outputs: int = 2,
    times: int = 256,
    positions: torch.Tensor | None = None,
) -> CardinalFBCMicroDynamicsNet:
    if positions is None:
        positions = _positions(channels)
    model = baselines.make_model(
        NATIVE_MODEL_NAME,
        n_channels=channels,
        n_outputs=outputs,
        n_times=times,
        sfreq=128.0,
        channel_names=_names(channels),
        channel_positions=positions,
    )
    assert isinstance(model, CardinalFBCMicroDynamicsNet)
    return model


def _candidate(
    *,
    channels: int = 3,
    outputs: int = 2,
    times: int = 256,
    positions: torch.Tensor | None = None,
) -> GaugeQuotientCrossMomentNet:
    if positions is None:
        positions = _positions(channels)
    return make_gauge_quotient_model(
        n_channels=channels,
        n_outputs=outputs,
        n_times=times,
        sfreq=128.0,
        channel_names=_names(channels),
        channel_positions=positions,
    )


@pytest.mark.parametrize(
    ("channels", "times", "outputs"),
    ((3, 320, 2), (15, 256, 2), (21, 320, 4)),
)
def test_one_frozen_configuration_has_expected_shapes_and_budget(
    channels: int,
    times: int,
    outputs: int,
) -> None:
    torch.manual_seed(2_602 + channels)
    positions = _positions(channels)
    model = _candidate(
        channels=channels,
        times=times,
        outputs=outputs,
        positions=positions,
    ).eval()
    x = torch.randn(2, channels, times)
    with torch.inference_mode():
        logits = model(x, positions)
        features = model.encode_quotient(x, positions)
    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    expected_log_variance = (
        len(BANDS_HZ) * N_WINDOWS * N_QUOTIENT_SOURCES
    )
    expected_cross_moments = (
        len(BANDS_HZ)
        * N_WINDOWS
        * len(LAGS_SAMPLES)
        * N_QUOTIENT_SOURCES
        * (N_QUOTIENT_SOURCES - 1)
        // 2
    )
    assert logits.shape == (2, outputs)
    assert features.shape == (
        2,
        expected_log_variance + expected_cross_moments,
    )
    assert torch.isfinite(logits).all()
    assert torch.isfinite(features).all()
    assert parameter_count <= MAXIMUM_PARAMETERS
    assert model.config["single_configuration"] is True
    assert model.config["complete_predictor_invariant"] is False
    assert model.config["complete_predictor_common_mode_invariant"] is False
    assert (
        model.config["complete_predictor_global_voltage_sign_invariant"]
        is False
    )
    assert (
        model.config["quotient_continuation_global_voltage_sign_invariant"]
        is True
    )
    assert "global_voltage_sign_invariant" not in model.config
    assert model.config["stored_potential_coefficients_sum_zero"] is False
    assert model.config["lag_samples"] == LAGS_SAMPLES
    assert model.config["source_permutation_invariant"] is False


def test_stored_fir_coefficients_have_explicit_numerical_contract() -> None:
    model = _candidate().eval()
    kernels = model.band_bank.kernels[:, 0].to(torch.float64)
    dc_residuals = kernels.sum(dim=-1).abs()
    norms = torch.linalg.vector_norm(kernels, dim=-1)

    assert torch.all(dc_residuals <= FIR_ZERO_DC_TOLERANCE)
    torch.testing.assert_close(
        norms,
        torch.ones_like(norms),
        rtol=0.0,
        atol=FIR_UNIT_NORM_TOLERANCE,
    )


@pytest.mark.parametrize("training", (False, True))
@pytest.mark.parametrize("use_mask", (False, True))
def test_zero_head_columns_preserve_complete_native_function(
    training: bool,
    use_mask: bool,
) -> None:
    torch.manual_seed(2_610)
    positions = _positions(3)
    native = _native(positions=positions)
    expected = copy.deepcopy(native)
    model = GaugeQuotientCrossMomentNet(native)
    expected.train(training)
    model.train(training)
    x = torch.randn(4, 3, 256)
    mask = (
        torch.tensor(
            (
                (True, True, True),
                (True, False, True),
                (False, True, True),
                (True, True, False),
            )
        )
        if use_mask
        else None
    )

    expected_logits = expected(x, positions, mask)
    actual_logits = model(x, positions, mask)
    assert torch.equal(actual_logits, expected_logits)
    torch.testing.assert_close(
        model.final_layer.weight[:, : model.native_feature_count],
        expected.final_layer.weight,
        rtol=0.0,
        atol=0.0,
    )
    assert torch.count_nonzero(
        model.final_layer.weight[:, model.native_feature_count :]
    ) == 0
    assert model.final_layer.bias is not None
    assert expected.final_layer.bias is not None
    torch.testing.assert_close(
        model.final_layer.bias,
        expected.final_layer.bias,
        rtol=0.0,
        atol=0.0,
    )
    assert isinstance(model.native_backbone.final_layer, nn.Identity)
    expected_state = expected.state_dict()
    for name, value in model.native_backbone.state_dict().items():
        torch.testing.assert_close(
            value,
            expected_state[name],
            rtol=0.0,
            atol=0.0,
        )


def _tensor_snapshot(
    values: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().clone()
        for name, value in values.items()
    }


def _assert_tensor_snapshot_unchanged(
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
) -> None:
    assert before.keys() == after.keys()
    for name, value in before.items():
        torch.testing.assert_close(
            after[name],
            value,
            rtol=0.0,
            atol=0.0,
        )


def test_every_rejected_train_call_is_pure_before_native_stateful_modules() -> None:
    torch.manual_seed(2_615)
    positions = _positions(3)
    model = _candidate(positions=positions).train()
    valid_x = torch.randn(2, 3, 256)
    valid_mask = torch.ones(2, 3, dtype=torch.bool)

    nonfinite_x = valid_x.clone()
    nonfinite_x[0, 0, 0] = torch.nan
    nonfinite_positions = positions.clone()
    nonfinite_positions[0, 0] = torch.inf
    zero_positions = positions.clone()
    zero_positions[1].zero_()
    empty_mask = valid_mask.clone()
    empty_mask[1].zero_()
    cases = (
        ("non_tensor_x", [], positions, None),
        ("wrong_x_rank", valid_x[:, 0], positions, None),
        ("empty_batch", valid_x[:0], positions, None),
        (
            "empty_channel_axis",
            torch.empty(2, 0, 256),
            torch.empty(0, 3),
            None,
        ),
        (
            "integer_x",
            torch.ones(2, 3, 256, dtype=torch.int64),
            positions,
            None,
        ),
        ("nonfinite_x", nonfinite_x, positions, None),
        ("non_tensor_positions", valid_x, [], None),
        ("wrong_position_shape", valid_x, positions[:, :2], None),
        (
            "integer_positions",
            valid_x,
            torch.ones(3, 3, dtype=torch.int64),
            None,
        ),
        ("nonfinite_positions", valid_x, nonfinite_positions, None),
        ("zero_position", valid_x, zero_positions, None),
        ("time_not_divisible", torch.randn(2, 3, 258), positions, None),
        ("lag_not_viable", torch.randn(2, 3, 32), positions, None),
        (
            "non_tensor_mask",
            valid_x,
            positions,
            [[True, True, True], [True, True, True]],
        ),
        ("wrong_mask_shape", valid_x, positions, valid_mask[:, :2]),
        ("non_boolean_mask", valid_x, positions, valid_mask.to(torch.float32)),
        ("empty_valid_set", valid_x, positions, empty_mask),
    )

    native_hook_calls = {
        "spectral_filter": 0,
        "floor_batch_norm": 0,
        "continuation_filter": 0,
        "continuation_batch_norm": 0,
    }

    def hook(name: str):
        def count(
            _module: nn.Module,
            _inputs: tuple[torch.Tensor, ...],
        ) -> None:
            native_hook_calls[name] += 1

        return count

    handles = (
        model.native_backbone.spectral_filtering.register_forward_pre_hook(
            hook("spectral_filter")
        ),
        model.native_backbone.batch_norm.register_forward_pre_hook(
            hook("floor_batch_norm")
        ),
        model.native_backbone.continuation_filter_bank.register_forward_pre_hook(
            hook("continuation_filter")
        ),
        model.native_backbone.continuation_batch_norm.register_forward_pre_hook(
            hook("continuation_batch_norm")
        ),
    )
    try:
        for _case_name, x, supplied_positions, mask in cases:
            for entry_point in (model, model.encode_native):
                before_parameters = _tensor_snapshot(
                    dict(model.named_parameters())
                )
                before_buffers = _tensor_snapshot(dict(model.named_buffers()))
                before_cpu_rng = torch.random.get_rng_state().clone()
                before_cuda_rng = (
                    tuple(
                        state.clone()
                        for state in torch.cuda.get_rng_state_all()
                    )
                    if torch.cuda.is_initialized()
                    else None
                )
                before_cuda_initialized = torch.cuda.is_initialized()
                before_hook_calls = dict(native_hook_calls)

                with pytest.raises((TypeError, ValueError)):
                    entry_point(x, supplied_positions, mask)

                _assert_tensor_snapshot_unchanged(
                    before_parameters,
                    _tensor_snapshot(dict(model.named_parameters())),
                )
                _assert_tensor_snapshot_unchanged(
                    before_buffers,
                    _tensor_snapshot(dict(model.named_buffers())),
                )
                torch.testing.assert_close(
                    torch.random.get_rng_state(),
                    before_cpu_rng,
                    rtol=0.0,
                    atol=0.0,
                )
                if before_cuda_rng is not None:
                    after_cuda_rng = torch.cuda.get_rng_state_all()
                    assert len(after_cuda_rng) == len(before_cuda_rng)
                    for expected_rng, actual_rng in zip(
                        before_cuda_rng,
                        after_cuda_rng,
                        strict=True,
                    ):
                        torch.testing.assert_close(
                            actual_rng,
                            expected_rng,
                            rtol=0.0,
                            atol=0.0,
                        )
                assert torch.cuda.is_initialized() is before_cuda_initialized
                assert native_hook_calls == before_hook_calls
    finally:
        for handle in handles:
            handle.remove()


def test_quotient_is_common_mode_invariant_with_exact_mask_projection() -> None:
    torch.manual_seed(2_620)
    positions = _positions(7)
    model = _candidate(
        channels=7,
        positions=positions,
    ).double().eval()
    x = torch.randn(3, 7, 256, dtype=torch.float64)
    mask = torch.tensor(
        (
            (True, True, True, True, True, True, True),
            (True, False, True, True, False, True, True),
            (False, True, True, False, True, False, True),
        )
    )
    common = 4.0 * torch.randn(3, 1, 256, dtype=torch.float64)
    shifted = x + common * mask[:, :, None]

    weights = model.quotient_field.projected_potentials(
        positions.double(),
        batch=3,
        channel_mask=mask,
        dtype=torch.float64,
    )
    sources = model.quotient_sources(x, positions.double(), mask)
    shifted_sources = model.quotient_sources(
        shifted,
        positions.double(),
        mask,
    )
    features = model.encode_quotient(x, positions.double(), mask)
    shifted_features = model.encode_quotient(
        shifted,
        positions.double(),
        mask,
    )

    torch.testing.assert_close(
        weights.sum(dim=-1),
        torch.zeros_like(weights[..., 0]),
        rtol=0.0,
        atol=2e-15,
    )
    torch.testing.assert_close(
        shifted_sources,
        sources,
        rtol=1e-10,
        atol=1e-10,
    )
    torch.testing.assert_close(
        shifted_features,
        features,
        rtol=2e-9,
        atol=2e-9,
    )


def test_permutation_equivariance_includes_positions_and_mask() -> None:
    torch.manual_seed(2_630)
    positions = _positions(9)
    model = _candidate(channels=9, positions=positions).eval()
    x = torch.randn(2, 9, 256)
    mask = torch.tensor(
        (
            (True, True, True, True, True, True, False, True, True),
            (True, False, True, True, True, False, True, True, True),
        )
    )
    permutation = torch.randperm(9)
    with torch.inference_mode():
        direct = model(x, positions, mask)
        reordered = model(
            x[:, permutation],
            positions[permutation],
            mask[:, permutation],
        )
        direct_branch = model.encode_quotient(x, positions, mask)
        reordered_branch = model.encode_quotient(
            x[:, permutation],
            positions[permutation],
            mask[:, permutation],
        )
    torch.testing.assert_close(direct, reordered, rtol=4e-5, atol=4e-5)
    torch.testing.assert_close(
        direct_branch,
        reordered_branch,
        rtol=4e-5,
        atol=4e-5,
    )


def test_masked_channels_cannot_affect_features_and_one_channel_is_trivial() -> None:
    torch.manual_seed(2_640)
    positions = _positions(5)
    model = _candidate(channels=5, positions=positions).eval()
    x = torch.randn(2, 5, 256)
    mask = torch.tensor(
        (
            (True, True, False, True, False),
            (True, False, True, True, False),
        )
    )
    perturbed = x.clone()
    perturbed[~mask] = 1e6 * torch.randn_like(perturbed[~mask])
    with torch.inference_mode():
        expected = model.encode_quotient(x, positions, mask)
        actual = model.encode_quotient(perturbed, positions, mask)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    single = torch.zeros(1, 5, dtype=torch.bool)
    single[:, 2] = True
    sources = model.quotient_sources(x[:1], positions, single)
    features = model.encode_quotient(x[:1], positions, single)
    assert torch.count_nonzero(sources) == 0
    assert torch.isfinite(features).all()
    with pytest.raises(ValueError, match="at least one valid"):
        model.encode_quotient(
            x[:1],
            positions,
            torch.zeros(1, 5, dtype=torch.bool),
        )
    with pytest.raises(TypeError, match="boolean"):
        model.encode_quotient(
            x[:1],
            positions,
            torch.ones(1, 5),
        )


def test_car_inputs_make_unconstrained_and_quotient_linear_responses_equivalent() -> None:
    """Expose, rather than hide, the quotient's redundancy on exact CAR data."""

    torch.manual_seed(2_650)
    positions = _positions(7)
    model = _candidate(channels=7, positions=positions).eval()
    potentials = model.quotient_field.potentials(positions)
    x = torch.randn(2, 7, 256)
    x = x - x.mean(dim=1, keepdim=True)
    centered_potentials = potentials - potentials.mean(
        dim=-1,
        keepdim=True,
    )
    unconstrained = torch.einsum("fsc,bct->bfst", potentials, x)
    quotient = torch.einsum("fsc,bct->bfst", centered_potentials, x)
    torch.testing.assert_close(
        quotient,
        unconstrained,
        rtol=2e-5,
        atol=2e-5,
    )


def test_lag_wedge_is_time_reversal_odd_and_zero_at_lag_zero() -> None:
    torch.manual_seed(2_660)
    model = _candidate().eval()
    sources = torch.randn(
        2,
        len(BANDS_HZ),
        N_QUOTIENT_SOURCES,
        256,
    )
    log_variance, wedges = model.quotient_statistics(sources)
    reversed_log_variance, reversed_wedges = model.quotient_statistics(
        sources.flip(-1)
    )
    torch.testing.assert_close(
        reversed_log_variance,
        log_variance.flip(2),
        rtol=2e-5,
        atol=2e-5,
    )
    torch.testing.assert_close(
        reversed_wedges,
        -wedges.flip(2),
        rtol=3e-5,
        atol=3e-5,
    )

    window_length = sources.shape[-1] // N_WINDOWS
    windows = sources.reshape(
        2,
        len(BANDS_HZ),
        N_QUOTIENT_SOURCES,
        N_WINDOWS,
        window_length,
    ).permute(0, 1, 3, 2, 4)
    centered = windows - windows.mean(dim=-1, keepdim=True)
    same_time = centered @ centered.transpose(-1, -2)
    zero_lag_wedge = 0.5 * (same_time - same_time.transpose(-1, -2))
    assert torch.count_nonzero(zero_lag_wedge) == 0
    assert torch.count_nonzero(wedges) > 0


def test_lag_wedge_is_global_sign_even_for_phase_shifted_quadrature() -> None:
    torch.manual_seed(2_665)
    model = _candidate().eval()
    samples = torch.arange(256, dtype=torch.float32)
    phase = 2.0 * torch.pi * samples / 16.0
    sine = torch.sin(phase)
    cosine = torch.cos(phase)

    same_phase = torch.stack(
        (sine, 0.7 * sine, -1.3 * sine, 2.0 * sine),
        dim=0,
    )[None, None].expand(1, len(BANDS_HZ), -1, -1)
    _, same_phase_wedges = model.quotient_statistics(same_phase)
    torch.testing.assert_close(
        same_phase_wedges,
        torch.zeros_like(same_phase_wedges),
        rtol=0.0,
        atol=2e-6,
    )

    phase_shifted = torch.stack(
        (sine, cosine, 0.5 * sine, -0.5 * cosine),
        dim=0,
    )[None, None].expand(1, len(BANDS_HZ), -1, -1)
    log_variance, wedges = model.quotient_statistics(phase_shifted)
    negative_log_variance, negative_wedges = model.quotient_statistics(
        -phase_shifted
    )
    torch.testing.assert_close(
        negative_log_variance,
        log_variance,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        negative_wedges,
        wedges,
        rtol=0.0,
        atol=0.0,
    )
    assert torch.max(torch.abs(wedges[..., 0])) > 0.05

    _, reversed_wedges = model.quotient_statistics(phase_shifted.flip(-1))
    torch.testing.assert_close(
        reversed_wedges,
        -wedges.flip(2),
        rtol=2e-5,
        atol=2e-5,
    )

    window_length = phase_shifted.shape[-1] // N_WINDOWS
    direct_windows = phase_shifted.reshape(
        1,
        len(BANDS_HZ),
        N_QUOTIENT_SOURCES,
        N_WINDOWS,
        window_length,
    ).permute(0, 1, 3, 2, 4)
    reverse_windows = phase_shifted.flip(-1).reshape(
        1,
        len(BANDS_HZ),
        N_QUOTIENT_SOURCES,
        N_WINDOWS,
        window_length,
    ).permute(0, 1, 3, 2, 4)
    direct_windows = direct_windows - direct_windows.mean(
        dim=-1,
        keepdim=True,
    )
    reverse_windows = reverse_windows - reverse_windows.mean(
        dim=-1,
        keepdim=True,
    )
    direct_covariance = direct_windows @ direct_windows.transpose(-1, -2)
    reverse_covariance = (
        reverse_windows @ reverse_windows.transpose(-1, -2)
    )
    torch.testing.assert_close(
        reverse_covariance,
        direct_covariance.flip(2),
        rtol=2e-5,
        atol=2e-5,
    )


def test_one_model_supports_both_frozen_epoch_lengths_and_runs_floor_once() -> None:
    torch.manual_seed(2_668)
    positions = _positions(15)
    model = _candidate(
        channels=15,
        times=256,
        positions=positions,
    ).eval()
    call_counts = {"batch_norm": 0, "spectral_filter": 0}

    def count_batch_norm(
        _module: nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        _output: torch.Tensor,
    ) -> None:
        call_counts["batch_norm"] += 1

    def count_spectral_filter(
        _module: nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        _output: torch.Tensor,
    ) -> None:
        call_counts["spectral_filter"] += 1

    handles = (
        model.native_backbone.batch_norm.register_forward_hook(
            count_batch_norm
        ),
        model.native_backbone.spectral_filtering.register_forward_hook(
            count_spectral_filter
        ),
    )
    try:
        with torch.inference_mode():
            short = model(torch.randn(2, 15, 256), positions)
            assert call_counts == {"batch_norm": 1, "spectral_filter": 1}
            call_counts.update(batch_norm=0, spectral_filter=0)
            long = model(torch.randn(2, 15, 320), positions)
            assert call_counts == {"batch_norm": 1, "spectral_filter": 1}
    finally:
        for handle in handles:
            handle.remove()
    assert short.shape == long.shape == (2, 2)
    assert torch.isfinite(short).all()
    assert torch.isfinite(long).all()


def test_gradients_are_finite_and_branch_wakes_after_zero_head_step() -> None:
    torch.manual_seed(2_670)
    positions = _positions(3)
    model = _candidate(positions=positions).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    x = torch.randn(4, 3, 256)
    coefficients = torch.tensor(
        ((0.8, -0.2), (-0.4, 0.7), (0.5, 0.3), (-0.6, 0.1))
    )

    (model(x, positions) * coefficients).sum().backward()
    new_head_gradient = model.final_layer.weight.grad[
        :,
        model.native_feature_count :,
    ]
    potential_gradient = (
        model.quotient_field.potential_field.coefficients.grad
    )
    assert torch.isfinite(new_head_gradient).all()
    assert torch.count_nonzero(new_head_gradient) > 0
    assert potential_gradient is not None
    assert torch.isfinite(potential_gradient).all()
    assert torch.count_nonzero(potential_gradient) == 0
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    (model(x, positions) * coefficients).sum().backward()
    potential_gradient = (
        model.quotient_field.potential_field.coefficients.grad
    )
    assert potential_gradient is not None
    assert torch.isfinite(potential_gradient).all()
    assert torch.count_nonzero(potential_gradient) > 0


def test_seeded_construction_and_eval_are_deterministic() -> None:
    positions = _positions(3)
    torch.manual_seed(2_680)
    first = _candidate(positions=positions).eval()
    torch.manual_seed(2_680)
    second = _candidate(positions=positions).eval()
    assert first.state_dict().keys() == second.state_dict().keys()
    for name, value in first.state_dict().items():
        torch.testing.assert_close(
            value,
            second.state_dict()[name],
            rtol=0.0,
            atol=0.0,
        )
    x = torch.randn(2, 3, 256)
    with torch.inference_mode():
        torch.testing.assert_close(
            first(x, positions),
            second(x, positions),
            rtol=0.0,
            atol=0.0,
        )


def test_model_has_no_outcome_interface_and_is_not_formally_registered() -> None:
    model = _candidate().eval()
    assert tuple(inspect.signature(model.forward).parameters) == (
        "x",
        "positions",
        "channel_mask",
    )
    forbidden = ("label", "target", "test", "outcome", "prediction")
    assert all(
        not any(token in name.casefold() for token in forbidden)
        for name in model.state_dict()
    )
    with pytest.raises(ValueError):
        baselines.make_model(
            MODEL_NAME,
            n_channels=3,
            n_outputs=2,
            n_times=256,
            sfreq=128.0,
            channel_names=_names(3),
            channel_positions=_positions(3),
        )


def test_screen_wrapper_is_candidate_only_read_only_and_not_a_plan(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "frozen-reference-does-not-need-to-exist"
    opened = screen_specification(
        OPENED17_STAGE,
        reference_run_root=reference_root,
    )
    disjoint = screen_specification(
        DISJOINT115_STAGE,
        reference_run_root=reference_root,
    )

    assert opened["schema"] == SCREEN_SCHEMA
    assert opened["candidate_job_count"] == 17
    assert opened["reference_record_count"] == 51
    assert disjoint["candidate_job_count"] == 115
    assert disjoint["reference_record_count"] == 345
    assert disjoint["reference_plan_sha256"] == (
        EXPECTED_ROBUSTNESS_PLAN_SHA256
    )
    assert opened["creates_plan"] is False
    assert opened["launches_work"] is False
    assert opened["confirmation_evidence"] is False
    assert isinstance(OPENED17_RULE, MappingProxyType)
    assert isinstance(DISJOINT115_RULE, MappingProxyType)
    assert isinstance(
        DISJOINT115_RULE["paired_subject_bootstrap_ci"],
        MappingProxyType,
    )
    with pytest.raises(TypeError):
        OPENED17_RULE["maximum_dataset_deficit"] = 1.0
    with pytest.raises(TypeError):
        DISJOINT115_RULE["paired_subject_bootstrap_ci"]["seed"] = 0

    canonical_options = {
        "sort_keys": True,
        "separators": (",", ":"),
        "ensure_ascii": True,
        "allow_nan": False,
    }
    assert json.dumps(opened["decision_rule"], **canonical_options) == (
        OPENED17_RULE_JSON
    )
    assert json.dumps(disjoint["decision_rule"], **canonical_options) == (
        DISJOINT115_RULE_JSON
    )
    assert opened["decision_rule_json"] == OPENED17_RULE_JSON
    assert opened["decision_rule_sha256"] == OPENED17_RULE_SHA256
    assert disjoint["decision_rule_json"] == DISJOINT115_RULE_JSON
    assert disjoint["decision_rule_sha256"] == DISJOINT115_RULE_SHA256
    assert hashlib.sha256(
        disjoint["decision_rule_json"].encode("ascii")
    ).hexdigest() == DISJOINT115_RULE_SHA256

    disjoint["decision_rule"]["paired_subject_bootstrap_ci"]["seed"] = -1
    disjoint["train_config"]["epochs"] = -1
    fresh_disjoint = screen_specification(
        DISJOINT115_STAGE,
        reference_run_root=reference_root,
    )
    assert (
        fresh_disjoint["decision_rule"]["paired_subject_bootstrap_ci"]["seed"]
        == 2_026_073_001
    )
    assert fresh_disjoint["train_config"]["epochs"] == 200
    assert fresh_disjoint["decision_rule_json"] == DISJOINT115_RULE_JSON
    assert fresh_disjoint["decision_rule_sha256"] == (
        DISJOINT115_RULE_SHA256
    )
    assert not reference_root.exists()
    assert len(candidate_jobs(OPENED17_STAGE)) == 17
    assert len(candidate_jobs(DISJOINT115_STAGE)) == 115
    assert len(
        reference_record_paths(OPENED17_STAGE, reference_root)
    ) == 51
    assert len(
        reference_record_paths(DISJOINT115_STAGE, reference_root)
    ) == 345
