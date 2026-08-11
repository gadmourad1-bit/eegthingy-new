from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

import deepnet.hemi_q_field_net as hemi_q_module
from deepnet.hemi_q_field_net import (
    ConstrainedQuadratureGaborBank,
    HemiQFieldClassifier,
    HemiQFieldConfig,
    HemiQFieldNet,
)


def _small_model() -> HemiQFieldNet:
    return HemiQFieldNet(
        n_times=65,
        n_filters=4,
        kernel_size=15,
        temporal_stride=2,
        local_window=7,
        local_stride=4,
        width=8,
    )


@pytest.mark.parametrize("training", (True, False))
def test_exact_reflection_action_through_every_parity_stage(training: bool) -> None:
    torch.manual_seed(13)
    model = _small_model()
    model.train(training)
    raw = torch.randn(5, 3, 65)

    direct = model.extract_features(raw)
    reflected = model.extract_features(model.reflect(raw))

    assert torch.equal(direct.invariant_context, reflected.invariant_context)
    assert torch.equal(direct.odd_field, -reflected.odd_field)
    assert torch.equal(direct.hidden_odd, -reflected.hidden_odd)
    assert torch.equal(direct.attention, reflected.attention)
    assert torch.equal(direct.logit, -reflected.logit)
    assert torch.equal(model(raw), direct.logit)
    assert direct.logit.shape == (5,)


def test_multiscale_token_shapes_and_global_scale_coordinate() -> None:
    model = _small_model()
    features = model.extract_features(torch.randn(3, 3, 65))

    assert model.local_token_count == 7
    assert model.token_count_per_frequency == 8
    assert model.invariant_dim == 9
    assert model.odd_field_dim == 8
    assert features.invariant_context.shape == (3, 4, 8, 9)
    assert features.odd_field.shape == (3, 4, 8, 8)
    assert features.hidden_odd.shape == (3, 4, 8, 8)
    assert features.attention.shape == (3, 4, 8)
    assert torch.all(features.invariant_context[:, :, :-1, -1] < 0.0)
    assert torch.equal(
        features.invariant_context[:, :, -1, -1], torch.zeros(3, 4)
    )


def test_robust_asymmetry_moments_are_bounded_and_signed() -> None:
    torch.manual_seed(14)
    model = _small_model().eval()
    raw = torch.randn(2, 3, 65)
    field = model.extract_features(raw).odd_field
    reflected_field = model.extract_features(model.reflect(raw)).odd_field
    asymmetry_moments = field[..., 4:]

    assert torch.equal(asymmetry_moments, -reflected_field[..., 4:])
    assert torch.all(torch.isfinite(asymmetry_moments))
    assert float(asymmetry_moments.abs().max()) <= 1.0


def test_single_logit_network_has_no_stochastic_or_batch_state() -> None:
    model = _small_model().train()
    assert not any(
        isinstance(
            module,
            (
                nn.Dropout,
                nn.Dropout1d,
                nn.Dropout2d,
                nn.BatchNorm1d,
                nn.BatchNorm2d,
                nn.BatchNorm3d,
            ),
        )
        for module in model.modules()
    )
    assert not hasattr(model, "candidate_logits")
    assert not any("router" in name.lower() for name, _ in model.named_modules())
    assert model(torch.randn(2, 3, 65)).ndim == 1


def test_gradients_reach_gabors_conditioned_odd_blocks_attention_and_readout() -> None:
    torch.manual_seed(17)
    model = _small_model().train()
    raw = torch.randn(10, 3, 65)
    labels = torch.arange(10, dtype=torch.float32) % 2
    loss = nn.functional.binary_cross_entropy_with_logits(model(raw), labels)
    loss.backward()

    expected = (
        model.filter_bank.raw_frequency_gaps,
        model.filter_bank.raw_bandwidths,
        model.context_encoder[0].weight,
        model.odd_block_1.primary.weight,
        model.odd_block_1.conditioner[0].weight,
        model.odd_block_2.modulated.weight,
        model.attention_score[0].weight,
        model.odd_readout.weight,
    )
    for parameter in expected:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0


def test_gabor_centres_bandwidths_quadrature_order_and_normalization() -> None:
    bank = ConstrainedQuadratureGaborBank(
        n_filters=6,
        kernel_size=31,
        sfreq=125.0,
        frequency_low=8.0,
        frequency_high=30.0,
    )
    with torch.no_grad():
        bank.raw_frequency_gaps.copy_(
            torch.tensor((-8.0, -2.0, 0.0, 1.0, 4.0, -5.0, 3.0))
        )
        bank.raw_bandwidths.copy_(torch.linspace(-8.0, 8.0, 6))

    frequencies = bank.frequencies_hz()
    bandwidths = bank.bandwidths_hz()
    assert torch.all(frequencies[1:] > frequencies[:-1])
    assert 8.0 < float(frequencies[0]) < float(frequencies[-1]) < 30.0
    assert torch.all(bandwidths > 1.5)
    assert torch.all(bandwidths < 8.0)

    kernels = bank.kernels().reshape(6, 2, 31)
    cosine = kernels[:, 0]
    sine = kernels[:, 1]
    assert torch.allclose(cosine.mean(dim=1), torch.zeros(6), atol=2e-7)
    assert torch.allclose(sine.mean(dim=1), torch.zeros(6), atol=2e-7)
    assert torch.allclose(
        torch.linalg.vector_norm(cosine, dim=1), torch.ones(6), atol=2e-6
    )
    assert torch.allclose(
        torch.linalg.vector_norm(sine, dim=1), torch.ones(6), atol=2e-6
    )
    assert torch.allclose(cosine, cosine.flip(1), atol=2e-6)
    assert torch.allclose(sine, -sine.flip(1), atol=2e-6)
    assert torch.allclose(
        torch.sum(cosine * sine, dim=1), torch.zeros(6), atol=2e-6
    )


def test_default_locked_window_and_parameter_budget() -> None:
    model = HemiQFieldNet()
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    assert model.n_times == 251
    assert trainable == 11_354
    assert trainable < 50_000
    assert model.odd_block_1.primary.bias is None
    assert model.odd_block_1.modulated.bias is None
    assert model.odd_block_1.output.bias is None
    assert model.odd_block_2.primary.bias is None
    assert model.odd_block_2.modulated.bias is None
    assert model.odd_block_2.output.bias is None
    assert model.odd_readout.bias is None


def test_source_scaler_is_hemispherically_symmetric_and_commutes_with_reflection() -> None:
    rng = np.random.default_rng(19)
    source = rng.normal(size=(9, 3, 49)).astype(np.float32)
    source[:, 0] += 7.0
    source[:, 2] -= 4.0
    classifier = HemiQFieldClassifier(
        HemiQFieldConfig(
            n_times=49,
            n_filters=3,
            kernel_size=15,
            local_window=5,
            local_stride=3,
            width=8,
            teacher_weight=0.0,
            device="cpu",
        )
    )
    classifier.channels_ = ("C3", "Cz", "C4")
    classifier.channel_selection_ = np.asarray((0, 1, 2), dtype=np.int64)
    classifier._fit_raw_scaler(source)

    assert classifier.raw_mean_[0, 0, 0] == classifier.raw_mean_[0, 2, 0]
    assert classifier.raw_std_[0, 0, 0] == classifier.raw_std_[0, 2, 0]
    expected_hemi_mean = np.concatenate(
        (source[:, 0].ravel(), source[:, 2].ravel())
    ).mean()
    assert np.isclose(classifier.raw_mean_[0, 0, 0], expected_hemi_mean)
    prepared = classifier._prepare_raw(source)
    prepared_reflected = classifier._prepare_raw(source[:, (2, 1, 0), :])
    assert np.array_equal(prepared_reflected, prepared[:, (2, 1, 0), :])


def _toy_raw(
    *,
    n_rows: int,
    n_times: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = np.arange(n_rows, dtype=np.int64) % 2
    time = np.arange(n_times, dtype=np.float32) / 125.0
    carrier = np.sin(2.0 * np.pi * 14.0 * time).astype(np.float32)
    raw = rng.normal(scale=0.04, size=(n_rows, 3, n_times)).astype(np.float32)
    for row, label in enumerate(labels):
        raw[row, 1] += carrier
        raw[row, 0 if label == 1 else 2] += carrier
    return raw, labels


def test_teacher_uses_source_covariances_only_and_is_absent_at_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    class FakeTeacher:
        def __init__(self, *, regularization: float) -> None:
            calls["regularization"] = regularization

        def fit(
            self,
            covariances: np.ndarray,
            labels: np.ndarray,
            mirror_index: np.ndarray,
        ) -> "FakeTeacher":
            calls["fit_covariances"] = covariances.copy()
            calls["fit_labels"] = labels.copy()
            calls["mirror_index"] = mirror_index.copy()
            self.model_ = SimpleNamespace(
                coef_=np.zeros((1, 6)), intercept_=np.zeros(1)
            )
            return self

        def signed_logit(self, covariances: np.ndarray) -> np.ndarray:
            calls.setdefault("logit_rows", []).append(len(covariances))  # type: ignore[union-attr]
            return (
                covariances[:, 0, 0, 0] - covariances[:, 0, 2, 2]
            ).astype(np.float32)

    monkeypatch.setattr(hemi_q_module, "FrozenTangentAnchor", FakeTeacher)
    raw_train, y_train = _toy_raw(n_rows=8, n_times=49, seed=23)
    raw_validation, y_validation = _toy_raw(n_rows=4, n_times=49, seed=29)
    rng = np.random.default_rng(31)
    observations = rng.normal(size=(8, 1, 3, 12))
    cov_train = np.einsum(
        "nbct,nbdt->nbcd", observations, observations, optimize=True
    ).astype(np.float32)
    validation_covariance_sentinel = object()
    classifier = HemiQFieldClassifier(
        HemiQFieldConfig(
            n_times=49,
            epochs=1,
            batch_size=4,
            patience=1,
            n_filters=3,
            kernel_size=15,
            temporal_stride=2,
            local_window=5,
            local_stride=3,
            width=8,
            teacher_weight=0.25,
            device="cpu",
            seed=3,
        )
    ).fit(
        raw_train,
        cov_train,
        y_train,
        raw_validation,
        validation_covariance_sentinel,  # type: ignore[arg-type]
        y_validation,
        channels=("C3", "Cz", "C4"),
    )

    assert np.array_equal(calls["fit_covariances"], cov_train)
    assert np.array_equal(calls["fit_labels"], y_train)
    assert np.array_equal(calls["mirror_index"], np.asarray((2, 1, 0)))
    assert calls["logit_rows"] == [8, 8]
    assert classifier.teacher_was_used_
    assert classifier.teacher_param_count_ == 7
    assert not hasattr(classifier, "teacher_")
    assert classifier.history_[1]["teacher_coefficient"] == 0.25
    assert classifier.decision_threshold_ == 0.0
    assert classifier.decision_function(raw_validation).shape == (4,)
    assert classifier.predict_proba(raw_validation).shape == (4, 2)
    assert classifier.max_equivariance_error(raw_validation) == 0.0


def test_lightweight_toy_problem_overfits_with_fixed_zero_threshold() -> None:
    raw_train, y_train = _toy_raw(n_rows=16, n_times=49, seed=37)
    raw_validation, y_validation = _toy_raw(n_rows=8, n_times=49, seed=41)
    classifier = HemiQFieldClassifier(
        HemiQFieldConfig(
            n_times=49,
            epochs=30,
            batch_size=8,
            learning_rate=4e-3,
            weight_decay=0.0,
            patience=30,
            min_delta=0.0,
            n_filters=4,
            kernel_size=15,
            temporal_stride=2,
            local_window=5,
            local_stride=3,
            width=8,
            teacher_weight=0.0,
            device="cpu",
            seed=5,
        )
    ).fit(
        raw_train,
        None,
        y_train,
        raw_validation,
        None,
        y_validation,
        channels=("C3", "Cz", "C4"),
    )

    assert np.mean(classifier.predict(raw_train) == y_train) >= 0.95
    assert np.mean(classifier.predict(raw_validation) == y_validation) >= 0.95
    assert classifier.best_validation_loss_ < math_log_two()
    diagnostics = classifier.filter_diagnostics()
    assert len(diagnostics["frequencies_hz"]) == 4
    assert np.all(np.diff(diagnostics["frequencies_hz"]) > 0.0)


def math_log_two() -> float:
    # A local helper avoids coupling this behavioural assertion to torch dtype.
    return float(np.log(2.0))
