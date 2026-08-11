from __future__ import annotations

import os

import pytest

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from ieee_mi.chsd import (
    CHSDConfig,
    CHSDNet,
    CoordinateProjector,
    HSDLayer,
    MultitaperCSD,
    dpss_tapers,
    hermitian_log_series_error_bound,
    hermitian_matrix_log,
    hermitian_matrix_log_series,
    hermitian_unvech,
    hermitian_vech,
    log_domain_difference_fields,
    make_chsd_model,
    shrink_complex_coherency,
)


def _positions(channels: int, *, seed: int = 31) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed + channels)
    positions = torch.randn(channels, 3, generator=generator)
    positions[:, 2] = positions[:, 2].abs() + 0.15
    return positions / torch.linalg.vector_norm(
        positions, dim=-1, keepdim=True
    )


def _small_config() -> CHSDConfig:
    return CHSDConfig(
        n_virtual_sensors=5,
        window_length=64,
        hop_length=32,
        time_bandwidth=2.5,
        n_tapers=3,
        hsd_width=12,
        power_width=8,
        decoder_blocks=1,
        dropout=0.0,
    )


def test_dpss_tapers_are_deterministic_orthonormal_and_zero_ordered() -> None:
    first = dpss_tapers(64, time_bandwidth=2.5, n_tapers=3)
    second = dpss_tapers(64, time_bandwidth=2.5, n_tapers=3)
    assert first.shape == (3, 64)
    assert torch.equal(first, second)
    assert torch.allclose(first @ first.transpose(0, 1), torch.eye(3), atol=2e-6)
    peak_indices = first.abs().argmax(dim=1)
    assert torch.all(first.gather(1, peak_indices[:, None]) > 0.0)


def test_coordinate_projection_is_smooth_and_channel_permutation_invariant() -> None:
    torch.manual_seed(32)
    projector = CoordinateProjector(6)
    positions = _positions(9)
    x = torch.randn(2, 9, 80)
    permutation = torch.randperm(9)
    direct = projector(x, positions)
    reordered = projector(x[:, permutation], positions[permutation])
    weights = projector.interpolation_weights(positions)
    assert direct.shape == (2, 6, 80)
    assert torch.allclose(direct, reordered, atol=2e-6, rtol=2e-6)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(6), atol=1e-7)
    assert torch.all(weights > 0.0)


def test_multitaper_csd_is_hermitian_psd_and_reliable() -> None:
    torch.manual_seed(33)
    estimator = MultitaperCSD(
        window_length=64,
        hop_length=32,
        time_bandwidth=2.5,
        n_tapers=3,
    )
    x = torch.randn(2, 5, 150)
    csd, power, reliability = estimator(x)
    assert csd.shape == (2, 4, 6, 5, 5)
    assert power.shape == (2, 4, 6, 5)
    assert reliability.shape == (2, 4, 6)
    assert torch.allclose(csd, csd.mH, atol=2e-6, rtol=2e-6)
    assert torch.linalg.eigvalsh(csd).amin() > -2e-6
    assert torch.all(power > 0.0)
    assert torch.all((0.0 <= reliability) & (reliability <= 1.0))
    assert torch.isfinite(power).all()
    assert torch.isfinite(reliability).all()


def test_shrinkage_log_and_isometric_vech_are_correct() -> None:
    torch.manual_seed(34)
    real = torch.randn(2, 4, 4)
    imaginary = torch.randn(2, 4, 4)
    factors = torch.complex(real, imaginary)
    covariance = factors @ factors.mH
    covariance = covariance + 0.2 * torch.eye(4, dtype=covariance.dtype)

    coherency, power = shrink_complex_coherency(covariance, 0.12)
    eigenvalues = torch.linalg.eigvalsh(coherency)
    assert eigenvalues.amin() > 0.11
    assert torch.allclose(
        coherency.diagonal(dim1=-2, dim2=-1).real,
        torch.ones(2, 4),
        atol=2e-6,
    )
    assert torch.all(power > 0.0)

    reference_log = hermitian_matrix_log(coherency)
    series_log = hermitian_matrix_log_series(coherency, n_terms=32)
    assert torch.isfinite(series_log.real).all()
    assert torch.isfinite(series_log.imag).all()
    assert torch.allclose(series_log, series_log.mH, atol=2e-6, rtol=2e-6)
    assert torch.allclose(series_log, reference_log, atol=2e-5, rtol=2e-5)
    assert torch.allclose(
        torch.matrix_exp(series_log), coherency, atol=2e-4, rtol=2e-4
    )

    vector = hermitian_vech(series_log)
    reconstructed = hermitian_unvech(vector, 4)
    assert vector.shape == (2, 16)
    assert torch.allclose(reconstructed, series_log, atol=2e-6, rtol=2e-6)
    matrix_norm = series_log.abs().square().sum(dim=(-2, -1))
    vector_norm = vector.square().sum(dim=-1)
    assert torch.allclose(vector_norm, matrix_norm, atol=2e-5, rtol=2e-5)


def test_matrix_log_series_bound_is_validated_for_hsd_configuration() -> None:
    bound = hermitian_log_series_error_bound(8, 0.10, 32)
    assert bound < 1e-5
    assert hermitian_log_series_error_bound(21, 0.10, 32) > 1e-5
    assert hermitian_log_series_error_bound(21, 0.10, 64) < 1e-5
    layer = HSDLayer(
        8,
        fixed_shrinkage=0.10,
        matrix_log_terms=32,
        maximum_series_error=1e-5,
    )
    assert layer.series_error_bound == bound
    assert torch.equal(layer.fixed_shrinkage, torch.tensor(0.10))
    assert dict(layer.named_parameters()) == {}


def test_hsd_fields_and_reliability_have_exact_finite_differences() -> None:
    state = torch.arange(1 * 3 * 4 * 2, dtype=torch.float32).reshape(1, 3, 4, 2)
    reliability = torch.linspace(0.4, 0.9, 12).reshape(1, 3, 4)
    time_step_seconds = 0.25
    band_centers_hz = torch.tensor([6.0, 10.0, 15.0, 25.0])
    fields, weights = log_domain_difference_fields(
        state,
        reliability,
        time_step_seconds=time_step_seconds,
        band_centers_hz=band_centers_hz,
    )
    assert fields.shape == (1, 3, 4, 4, 2)
    assert weights.shape == (1, 3, 4, 4)
    assert torch.equal(fields[..., 0, :], state)
    assert torch.equal(fields[:, 0, :, 1], torch.zeros(1, 4, 2))
    assert torch.equal(fields[:, :, 0, 2], torch.zeros(1, 3, 2))
    assert torch.equal(fields[:, 0, :, 3], torch.zeros(1, 4, 2))
    assert torch.equal(fields[:, :, 0, 3], torch.zeros(1, 3, 2))
    assert torch.equal(
        fields[:, 1:, :, 1],
        (state[:, 1:] - state[:, :-1]) / time_step_seconds,
    )
    frequency_steps = band_centers_hz[1:] - band_centers_hz[:-1]
    assert torch.allclose(
        fields[:, :, 1:, 2],
        (state[:, :, 1:] - state[:, :, :-1])
        / frequency_steps[None, None, :, None],
    )
    expected_mixed = (
        state[:, 1:, 1:]
        - state[:, :-1, 1:]
        - state[:, 1:, :-1]
        + state[:, :-1, :-1]
    ) / (time_step_seconds * frequency_steps[None, None, :, None])
    assert torch.allclose(fields[:, 1:, 1:, 3], expected_mixed)
    assert torch.equal(weights[:, 0, :, 1], torch.zeros(1, 4))
    assert torch.equal(weights[:, :, 0, 2], torch.zeros(1, 3))
    assert torch.equal(weights[:, 0, :, 3], torch.zeros(1, 4))
    assert torch.equal(weights[:, :, 0, 3], torch.zeros(1, 3))
    assert torch.isfinite(fields).all()
    assert torch.all((0.0 <= weights) & (weights <= 1.0))


def test_chsdnet_supports_two_and_four_classes_and_variable_c_t() -> None:
    torch.manual_seed(35)
    two_class = CHSDNet(2, config=_small_config()).eval()
    four_class = CHSDNet(4, config=_small_config()).eval()
    cases = ((3, 40), (7, 97), (11, 161))
    with torch.no_grad():
        for channels, n_times in cases:
            x = torch.randn(2, channels, n_times)
            positions = _positions(channels)
            logits = two_class(x, positions)
            assert logits.shape == (2, 2)
            assert torch.isfinite(logits).all()
        four_logits = four_class(torch.randn(3, 9, 129), _positions(9))
        assert four_logits.shape == (3, 4)
        assert torch.isfinite(four_logits).all()


def test_chsdnet_has_finite_input_coordinate_and_parameter_gradients() -> None:
    torch.manual_seed(36)
    config = CHSDConfig(
        n_virtual_sensors=8,
        window_length=64,
        hop_length=32,
        hsd_width=12,
        power_width=8,
        decoder_blocks=1,
        dropout=0.0,
    )
    model = CHSDNet(4, config=config).train()
    # Three physical channels make every 8x8 unshrunk CSD rank <= 3.  The
    # fixed shrinkage consequently creates a repeated eigenvalue with
    # multiplicity at least five, reproducing the BNCI004 gradient edge case.
    x = torch.randn(2, 3, 144, requires_grad=True)
    positions = _positions(3).requires_grad_()
    logits = model(x, positions)
    loss = logits.square().mean() + logits.mean()
    loss.backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert positions.grad is not None and torch.isfinite(positions.grad).all()
    assert torch.count_nonzero(x.grad) > 0
    assert torch.count_nonzero(positions.grad) > 0
    assert model.projector.raw_concentration.grad is not None
    assert torch.isfinite(model.projector.raw_concentration.grad)
    assert "fixed_shrinkage" in dict(model.hsd.named_buffers())
    assert "raw_shrinkage" not in dict(model.hsd.named_parameters())
    parameter_gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert parameter_gradients
    assert all(torch.isfinite(gradient).all() for gradient in parameter_gradients)


def test_chsdnet_zero_input_is_finite_and_repeated_cpu_calls_are_identical() -> None:
    torch.manual_seed(37)
    model = CHSDNet(2, config=_small_config()).eval()
    x = torch.zeros(2, 6, 23)
    positions = _positions(6)
    with torch.no_grad():
        first = model(x, positions)
        second = model(x, positions)
        features = model.forward_features(x, positions)
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()
    for value in features.values():
        assert torch.isfinite(value.real).all()
        if value.is_complex():
            assert torch.isfinite(value.imag).all()


def test_full_grid_factory_validates_metadata_and_constructs_locked_model() -> None:
    positions = _positions(15)
    model = make_chsd_model(
        requested_model="chsdnet",
        n_channels=15,
        n_outputs=2,
        n_times=256,
        sfreq=128.0,
        channel_names=tuple(f"EEG-{index:02d}" for index in range(15)),
        channel_positions=positions,
    )
    assert isinstance(model, CHSDNet)
    assert model.n_outputs == 2
    assert model.config.sfreq == 128.0
    assert model.config.n_virtual_sensors == 15
    assert model.config.use_coordinate_projection is False
    assert model.config.matrix_log_terms == 64
    with pytest.raises(ValueError, match="unsupported requested model"):
        make_chsd_model(
            requested_model="not_chsdnet",
            n_channels=15,
            n_outputs=2,
            n_times=256,
            sfreq=128.0,
            channel_names=tuple(f"EEG-{index:02d}" for index in range(15)),
            channel_positions=positions,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_chsdnet_cuda_forward_backward_on_locked_dataset_shapes() -> None:
    torch.manual_seed(38)
    torch.cuda.manual_seed_all(38)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda")
    try:
        for n_outputs, channels, n_times in (
            (2, 15, 256),
            (4, 21, 320),
            (2, 3, 320),
        ):
            x = torch.randn(4, channels, n_times, device=device)
            positions = _positions(channels).to(device)
            # Exercise the actual native-montage factory configuration rather
            # than the reduced CPU-test configuration.
            model = make_chsd_model(
                requested_model="chsdnet",
                n_channels=channels,
                n_outputs=n_outputs,
                n_times=n_times,
                sfreq=128.0,
                channel_names=tuple(
                    f"EEG-{index:02d}" for index in range(channels)
                ),
                channel_positions=positions,
            ).to(device).train()
            labels = torch.arange(4, device=device) % n_outputs
            logits = model(x, positions)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            loss.backward()
            assert logits.shape == (4, n_outputs)
            assert torch.isfinite(logits).all()
            gradients = [
                parameter.grad
                for parameter in model.parameters()
                if parameter.requires_grad and parameter.grad is not None
            ]
            assert gradients
            nonfinite = [
                name
                for name, parameter in model.named_parameters()
                if parameter.grad is not None
                and not bool(torch.isfinite(parameter.grad).all())
            ]
            assert not nonfinite, (
                f"non-finite gradients for C={channels}, T={n_times}: {nonfinite}"
            )
    finally:
        torch.use_deterministic_algorithms(False)
