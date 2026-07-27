from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch
from scipy import signal
from torch import nn

from ieee_mi.reference_training import (
    BRAIDECODE_VERSION,
    FBCNET_BANDS,
    FBCNET_SOURCE_COMMIT,
    FBCNetReferenceConfig,
    PrefilteredFBCNetAdapter,
    RelativeNoDecrease,
    TCFormerReferenceConfig,
    apply_fbcnet_cheby2_filterbank,
    design_fbcnet_cheby2_filterbank,
    fbcnet_stage1_transition_threshold,
    fbcnet_filterbank_frequency_response,
    fit_fbcnet_reference,
    fit_tcformer_reference,
    model_state_sha256,
    tcformer_augmented_batch,
    tcformer_lr_multiplier,
    tcformer_reference_config,
    tcformer_segment_reconstruct,
)


class TinyNet(nn.Module):
    uses_positions = False

    def __init__(self, channels: int = 2, times: int = 8) -> None:
        super().__init__()
        self.linear = nn.Linear(channels * times, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.flatten(1))


class TinyFBC(nn.Module):
    def __init__(
        self,
        *,
        n_bands: int = 2,
        channels: int = 3,
        times: int = 8,
        stride_factor: int = 2,
    ) -> None:
        super().__init__()
        if times % stride_factor:
            raise ValueError("times must divide by stride_factor")
        self.n_bands = n_bands
        self.stride_factor = stride_factor
        self.n_times_padded = times
        spatial_features = 2 * n_bands
        self.spatial_conv = nn.Conv2d(
            n_bands,
            spatial_features,
            kernel_size=(channels, 1),
            groups=n_bands,
            bias=False,
        )
        self.padding_layer = nn.Identity()
        self.temporal_layer = nn.Identity()
        self.flatten_layer = nn.Flatten()
        self.final_layer = nn.Linear(spatial_features * times, 2)


def test_tcformer_dataset_horizons_and_schedule_are_exact() -> None:
    iv_2a = tcformer_reference_config("bnci2014_001")
    iv_2b = tcformer_reference_config("bnci2014_004")
    assert (iv_2a.epochs, iv_2a.horizon_provenance) == (
        1000,
        "released_bcic_iv_2a",
    )
    assert (iv_2b.epochs, iv_2b.horizon_provenance) == (
        500,
        "released_bcic_iv_2b",
    )
    for dataset in ("local_exp4", "cho2017", "physionet_mi"):
        adapted = tcformer_reference_config(dataset)
        assert adapted.epochs == 1000
        assert adapted.horizon_provenance == (
            "adapted_from_released_bcic_iv_2a_not_released"
        )
    with pytest.raises(ValueError, match="no declared"):
        tcformer_reference_config("sealed_unknown")
    with pytest.raises(ValueError, match="owns epochs"):
        tcformer_reference_config("cho2017", epochs=12)

    assert tcformer_lr_multiplier(0, total_epochs=100, warmup_epochs=20) == 0.0
    assert tcformer_lr_multiplier(19, total_epochs=100, warmup_epochs=20) == 0.95
    assert tcformer_lr_multiplier(20, total_epochs=100, warmup_epochs=20) == 1.0
    expected = 0.5 * (1.0 + np.cos(np.pi * 0.5))
    assert np.isclose(
        tcformer_lr_multiplier(60, total_epochs=100, warmup_epochs=20), expected
    )
    assert tcformer_lr_multiplier(100, total_epochs=100, warmup_epochs=20) == 0.0
    with pytest.raises(ValueError, match=r"\[0, total_epochs\]"):
        tcformer_lr_multiplier(101, total_epochs=100, warmup_epochs=20)


def test_tcformer_sr_uses_only_same_class_donors_and_doubles_batch() -> None:
    values = torch.arange(6, dtype=torch.float32).reshape(6, 1, 1).expand(6, 1, 16)
    labels = torch.tensor((0, 0, 0, 1, 1, 1))
    generator = torch.Generator().manual_seed(41)
    synthetic = tcformer_segment_reconstruct(
        values, labels, segments=8, generator=generator
    )
    for row, label in enumerate(labels.tolist()):
        allowed = {0.0, 1.0, 2.0} if label == 0 else {3.0, 4.0, 5.0}
        for segment in range(8):
            fragment = synthetic[row, 0, 2 * segment : 2 * segment + 2]
            assert len(torch.unique(fragment)) == 1
            assert float(fragment[0]) in allowed

    combined_x, combined_y = tcformer_augmented_batch(
        values,
        labels,
        segments=8,
        generator=torch.Generator().manual_seed(43),
    )
    assert combined_x.shape == (12, 1, 16)
    assert torch.bincount(combined_y).tolist() == [6, 6]
    with pytest.raises(ValueError, match="divide exactly"):
        tcformer_segment_reconstruct(
            values[..., :-1],
            labels,
            segments=8,
            generator=torch.Generator().manual_seed(1),
        )


def test_tcformer_reference_fit_has_official_optimizer_and_no_test_api() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(8, 2, 8)).astype(np.float32)
    y = np.asarray((0, 1, 0, 1, 0, 1, 0, 1), dtype=np.int64)
    positions = np.eye(3, dtype=np.float32)[:2]
    config = TCFormerReferenceConfig(
        epochs=3,
        warmup_epochs=1,
        batch_size=4,
        segment_count=8,
        seed=5,
        device="cpu",
    )
    fit = fit_tcformer_reference(lambda: TinyNet(), x, y, positions, config=config)
    repeat = fit_tcformer_reference(lambda: TinyNet(), x, y, positions, config=config)
    assert fit["epochs_run"] == 3
    assert [row["learning_rate"] for row in fit["history"]] == pytest.approx(
        (0.0, 9e-4, 4.5e-4)
    )
    assert fit["optimizer"].defaults["betas"] == (0.5, 0.999)
    assert fit["optimizer"].defaults["weight_decay"] == 1e-3
    assert fit["optimizer"].param_groups[0]["lr"] == 0.0
    assert fit["terminal_learning_rate"] == 0.0
    assert fit["initial_state_sha256"] == repeat["initial_state_sha256"]
    assert fit["final_state_sha256"] == repeat["final_state_sha256"]
    assert fit["final_state_sha256"] == model_state_sha256(fit["model"])
    assert fit["recipe"]["test_access_during_fit"] is False
    assert "test" not in " ".join(inspect.signature(fit_tcformer_reference).parameters)
    with pytest.raises(TypeError, match="model factory"):
        fit_tcformer_reference(TinyNet(), x, y, positions, config=config)


def test_original_fbcnet_cheby2_design_and_causal_application() -> None:
    sfreq = 128.0
    designs = design_fbcnet_cheby2_filterbank(sfreq)
    assert len(designs) == 9
    assert tuple(design.passband_hz for design in designs) == FBCNET_BANDS
    first = designs[0]
    order, _natural = signal.cheb2ord(
        np.asarray((4.0, 8.0)) / 64.0,
        np.asarray((2.0, 10.0)) / 64.0,
        3.0,
        30.0,
    )
    normalized_stop = np.asarray((2.0, 10.0)) / 64.0
    numerator, denominator = signal.cheby2(
        order, 30.0, normalized_stop, btype="bandpass"
    )
    assert first.order == order
    assert np.allclose(first.numerator, numerator)
    assert np.allclose(first.denominator, denominator)

    hz, response = fbcnet_filterbank_frequency_response(
        sfreq, designs=designs, frequencies=32768
    )
    for row, design in zip(response, designs, strict=True):
        magnitude_db = 20.0 * np.log10(np.maximum(np.abs(row), 1e-15))
        center = sum(design.passband_hz) / 2.0
        assert magnitude_db[np.argmin(np.abs(hz - center))] >= -3.05
        assert np.max(magnitude_db[hz <= design.stopband_hz[0]]) <= -29.8
        assert np.max(magnitude_db[hz >= design.stopband_hz[1]]) <= -29.8

    impulse = np.zeros((2, 3, 64), dtype=np.float32)
    impulse[..., 0] = 1.0
    filtered = apply_fbcnet_cheby2_filterbank(
        impulse, sfreq, designs=designs
    )
    assert filtered.shape == (2, 9, 3, 64)
    expected = signal.lfilter(first.numerator, first.denominator, impulse, axis=-1)
    assert np.allclose(filtered[:, 0], expected.astype(np.float32))
    with pytest.raises(ValueError, match="at least one"):
        apply_fbcnet_cheby2_filterbank(impulse, sfreq, designs=())


def test_prefiltered_fbc_adapter_bypasses_only_spectral_layer() -> None:
    adapter = PrefilteredFBCNetAdapter(TinyFBC())
    output = adapter(torch.randn(5, 2, 3, 8))
    assert output.shape == (5, 2)
    with pytest.raises(ValueError, match="prefiltered"):
        adapter(torch.randn(5, 3, 8))


def test_fbcnet_patience_zero_and_transition_threshold_match_release() -> None:
    patience = RelativeNoDecrease(patience=2, min_relative_change=1e-6)
    assert patience.update(0.0) is False
    assert patience.update(0.0) is False
    assert patience.update(0.0) is False
    assert patience.minimum == 0.0
    assert patience.stale_epochs == 0

    nonzero = RelativeNoDecrease(patience=2, min_relative_change=1e-6)
    assert nonzero.update(0.5) is False
    assert nonzero.update(0.5) is False
    assert nonzero.update(0.5) is True

    history = ({"train_loss": 0.2}, {"train_loss": 0.7})
    assert fbcnet_stage1_transition_threshold(history) == 0.7
    with pytest.raises(ValueError, match="at least one"):
        fbcnet_stage1_transition_threshold(())


def test_fbcnet_full_causal_filter_adapter_fit_and_state_transition() -> None:
    rng = np.random.default_rng(7)
    raw_train = rng.normal(size=(8, 2, 64)).astype(np.float32)
    y_train = np.asarray((0, 1, 0, 1, 0, 1, 0, 1), dtype=np.int64)
    raw_validation = rng.normal(size=(4, 2, 64)).astype(np.float32)
    y_validation = np.asarray((0, 1, 0, 1), dtype=np.int64)
    positions = np.eye(3, dtype=np.float32)[:2]
    x_train = apply_fbcnet_cheby2_filterbank(raw_train, 128.0)
    x_validation = apply_fbcnet_cheby2_filterbank(raw_validation, 128.0)
    assert x_train.shape == (8, 9, 2, 64)

    config = FBCNetReferenceConfig(
        stage1_max_epochs=2,
        stage1_patience=1,
        stage2_max_epochs=2,
        batch_size=4,
        learning_rate=1e-2,
        seed=11,
        device="cpu",
    )

    def factory() -> PrefilteredFBCNetAdapter:
        return PrefilteredFBCNetAdapter(
            TinyFBC(n_bands=9, channels=2, times=64, stride_factor=4)
        )

    fit = fit_fbcnet_reference(
        factory,
        x_train,
        y_train,
        x_validation,
        y_validation,
        positions,
        config=config,
    )
    repeat = fit_fbcnet_reference(
        factory,
        x_train,
        y_train,
        x_validation,
        y_validation,
        positions,
        config=config,
    )
    assert 1 <= fit["stage1_epochs_run"] <= 2
    assert 1 <= fit["stage2_epochs_run"] <= 2
    assert fit["initial_state_sha256"] == repeat["initial_state_sha256"]
    assert fit["final_state_sha256"] == repeat["final_state_sha256"]
    assert fit["final_state_sha256"] == model_state_sha256(fit["model"])
    assert fit["stage1_train_loss_threshold"] == pytest.approx(
        fit["stage1_history"][-1]["train_loss"]
    )
    assert fit["stage1_threshold_origin"] == (
        "last_stage1_epoch_before_best_restore"
    )
    assert fit["stage1_threshold_origin_epoch"] == fit["stage1_epochs_run"] - 1
    assert fit["stage1_best_model_state_sha256"] == (
        fit["stage2_start_model_state_sha256"]
    )
    assert fit["stage1_best_optimizer_state_sha256"] == (
        fit["stage2_start_optimizer_state_sha256"]
    )
    assert fit["stage2_source_count"] == len(x_train) + len(x_validation)
    assert fit["stage2_original_validation_count"] == len(x_validation)
    assert fit["stage2_history"][0]["source_rows"] == 12.0
    assert fit["final_state_sha256"] != fit["stage2_start_model_state_sha256"]
    assert fit["recipe"]["test_access_during_fit"] is False
    assert fit["recipe"]["scheduler"] is None
    assert fit["recipe"]["source"]["commit"] == FBCNET_SOURCE_COMMIT
    assert fit["recipe"]["implementation"]["version"] == BRAIDECODE_VERSION
    epoch_counts = fit["recipe"]["epoch_count_semantics"]
    assert epoch_counts["implemented_default_stage1_epoch_count"] == 1500
    assert (
        epoch_counts["released_stage1_effective_epoch_count_without_patience_stop"]
        == 1501
    )
    assert epoch_counts["released_stage2_effective_epoch_count"] == 600
    assert fit["optimizer"].defaults["betas"] == (0.9, 0.999)
    assert fit["optimizer"].defaults["weight_decay"] == 0.0
    assert "test" not in " ".join(inspect.signature(fit_fbcnet_reference).parameters)

    with pytest.raises(ValueError, match="apply_fbcnet_cheby2_filterbank"):
        fit_fbcnet_reference(
            factory,
            raw_train,
            y_train,
            raw_validation,
            y_validation,
            positions,
            config=config,
        )
    with pytest.raises(TypeError, match="PrefilteredFBCNetAdapter"):
        fit_fbcnet_reference(
            lambda: TinyNet(channels=18, times=64),
            x_train,
            y_train,
            x_validation,
            y_validation,
            positions,
            config=config,
        )
    with pytest.raises(TypeError, match="model factory"):
        fit_fbcnet_reference(
            factory(),
            x_train,
            y_train,
            x_validation,
            y_validation,
            positions,
            config=config,
        )
