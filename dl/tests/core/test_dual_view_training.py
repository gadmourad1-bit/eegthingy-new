from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from torch import nn

import benchmark.dual_view_training as training
from benchmark.config import CANONICAL_21_CHANNELS
from benchmark.data import fit_channel_scaler
from benchmark.dual_view import CardinalSplineDualViewOutput
from benchmark.models import CANONICAL_21_POSITIONS
from benchmark.native_pretraining import (
    NativeMontageSubject,
    NativePretrainConfig,
    hashed_subject_partition,
)
from benchmark.training import TrainConfig, configure_determinism


class TinyDualViewNet(nn.Module):
    """Shape-dynamic shared-head model for protocol tests."""

    uses_positions = True
    uses_canonical_view = True

    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(2)
        self.classifier = nn.Linear(1, 2)
        self.preference_gate = nn.Linear(1, 1)
        nn.init.zeros_(self.preference_gate.weight)
        nn.init.zeros_(self.preference_gate.bias)

    def forward_views(
        self,
        native_x: torch.Tensor,
        canonical_x: torch.Tensor,
        positions: torch.Tensor,
    ) -> CardinalSplineDualViewOutput:
        assert positions.shape == (native_x.shape[1], 3)
        direct_scalar = native_x.mean(dim=(1, 2))
        canonical_scalar = canonical_x.mean(dim=(1, 2))
        paired = self.bn(torch.stack((direct_scalar, canonical_scalar), dim=1))
        direct_logits = self.classifier(paired[:, :1])
        canonical_logits = self.classifier(paired[:, 1:2])
        descriptor = (paired[:, 1:2] - paired[:, :1]).abs()
        canonical_weight = torch.sigmoid(
            self.preference_gate(descriptor.detach())
        )
        fused = direct_logits + canonical_weight * (
            canonical_logits - direct_logits
        )
        return CardinalSplineDualViewOutput(
            fused_logits=fused,
            direct_logits=direct_logits,
            canonical_logits=canonical_logits,
            canonical_weight=canonical_weight,
            disagreement_descriptor=descriptor,
        )

    def forward(
        self,
        native_x: torch.Tensor,
        canonical_x: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_views(native_x, canonical_x, positions).fused_logits


def _subject(
    dataset: str,
    subject: int,
    *,
    channels: int,
    n_times: int,
    offset: float,
) -> NativeMontageSubject:
    rng = np.random.default_rng(10_000 + subject + 100 * len(dataset))
    x = rng.normal(
        loc=offset,
        scale=0.5,
        size=(6, channels, n_times),
    ).astype(np.float32)
    return NativeMontageSubject(
        dataset=dataset,
        subject=subject,
        x=x,
        y=np.asarray((0, 1, 0, 1, 0, 1), dtype=np.int64),
        positions=np.asarray(CANONICAL_21_POSITIONS[:channels], dtype=np.float32),
        channel_names=tuple(CANONICAL_21_CHANNELS[:channels]),
    )


def _corpus() -> tuple[NativeMontageSubject, ...]:
    # Different C/T grids prove that one shared model is not tied to a montage
    # or epoch duration. Large subject offsets make role leakage detectable.
    return tuple(
        [
            _subject(
                "alpha",
                subject,
                channels=4,
                n_times=13,
                offset=50.0 * subject,
            )
            for subject in (1, 2, 3)
        ]
        + [
            _subject(
                "beta",
                subject,
                channels=6,
                n_times=17,
                offset=-30.0 * subject,
            )
            for subject in (1, 2, 3)
        ]
    )


def test_source_transforms_fit_only_explicit_partition_subjects() -> None:
    records = _corpus()
    partition = hashed_subject_partition(
        records, validation_fraction=1 / 3, salt="paired-test"
    )
    transforms = training.fit_paired_dataset_transforms(
        records,
        set(partition.train),
        fit_rows="unit-test train subjects only",
    )

    for dataset, transform in transforms.items():
        selected = sorted(
            (record for record in records if record.key in set(partition.train) and record.key[0] == dataset),
            key=lambda record: record.key,
        )
        excluded = {
            record.key
            for record in records
            if record.key in set(partition.validation) and record.key[0] == dataset
        }
        raw = np.concatenate([record.x for record in selected])
        expected_native_mean, expected_native_std = fit_channel_scaler(
            raw, selected[0].channel_names
        )
        np.testing.assert_array_equal(
            transform.native_scaler.mean, expected_native_mean
        )
        np.testing.assert_array_equal(
            transform.native_scaler.std, expected_native_std
        )
        canonical_raw = transform.interpolator.transform(raw)
        expected_canonical_mean, expected_canonical_std = fit_channel_scaler(
            canonical_raw, CANONICAL_21_CHANNELS
        )
        np.testing.assert_array_equal(
            transform.canonical_scaler.mean, expected_canonical_mean
        )
        np.testing.assert_array_equal(
            transform.canonical_scaler.std, expected_canonical_std
        )
        assert transform.row_count == len(raw)
        assert set(transform.fit_subjects) == {record.key for record in selected}
        assert not excluded & set(transform.fit_subjects)
        assert transform.native_scaler.sha256 != transform.canonical_scaler.sha256


def test_paired_augmentation_is_identical_and_reproducible() -> None:
    values = torch.arange(8 * 5 * 19, dtype=torch.float32).reshape(8, 5, 19)
    labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    config = replace(
        NativePretrainConfig(),
        segment_probability=1.0,
        segment_count=4,
        time_shift_samples=3,
        noise_std=0.01,
        device="cpu",
    )
    first = training.paired_augment(
        values,
        values.clone(),
        labels,
        config,
        torch.Generator(device="cpu").manual_seed(77),
    )
    second = training.paired_augment(
        values,
        values.clone(),
        labels,
        config,
        torch.Generator(device="cpu").manual_seed(77),
    )
    torch.testing.assert_close(first[0], first[1], rtol=0.0, atol=0.0)
    torch.testing.assert_close(first[0], second[0], rtol=0.0, atol=0.0)
    assert first[2] == second[2]
    assert first[2].segment_applied
    assert len(first[2].segment_sources) > 0
    assert len(first[2].shifts) == len(values)


def test_pretraining_is_order_independent_and_state_deterministic() -> None:
    records = _corpus()
    config = replace(
        NativePretrainConfig(),
        macro_epochs=2,
        batch_size_per_dataset=4,
        steps_per_macro_epoch=1,
        patience=2,
        validation_fraction=1 / 3,
        partition_salt="paired-pretrain-test",
        segment_probability=0.5,
        segment_count=2,
        time_shift_samples=1,
        noise_std=0.01,
        validation_batch_size=8,
        seed=19,
        device="cpu",
    )
    first = training.fit_dual_view_native_pretraining(
        TinyDualViewNet, records, config=config
    )
    second = training.fit_dual_view_native_pretraining(
        TinyDualViewNet, tuple(reversed(records)), config=config
    )

    assert first.partition == second.partition
    assert first.initial_state_sha256 == second.initial_state_sha256
    assert first.selection_state_sha256 == second.selection_state_sha256
    assert first.checkpoint_sha256 == second.checkpoint_sha256
    assert first.selection_history == second.selection_history
    assert first.refit_history == second.refit_history
    assert {
        dataset: transform.sha256
        for dataset, transform in first.selection_transforms.items()
    } == {
        dataset: transform.sha256
        for dataset, transform in second.selection_transforms.items()
    }
    for row in first.selection_history:
        dataset_train = row["training_dataset_loss"]
        dataset_validation = row["validation_dataset_fused_ce"]
        assert isinstance(dataset_train, dict)
        assert isinstance(dataset_validation, dict)
        assert row["training_equal_dataset_loss"] == pytest.approx(
            np.mean(list(dataset_train.values()))
        )
        assert row["validation_equal_dataset_fused_ce"] == pytest.approx(
            np.mean(list(dataset_validation.values()))
        )
    assert first.selection_transforms["alpha"].native_scaler.mean.shape == (1, 4, 1)
    assert first.selection_transforms["beta"].native_scaler.mean.shape == (1, 6, 1)


def test_target_scalers_exclude_validation_and_test_and_prediction_is_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    channels = 5
    n_times = 19
    rng = np.random.default_rng(44)
    train_x = rng.normal(0.0, 0.2, size=(8, channels, n_times)).astype(np.float32)
    validation_x = rng.normal(40.0, 0.2, size=(4, channels, n_times)).astype(np.float32)
    test_x = rng.normal(1000.0, 0.2, size=(3, channels, n_times)).astype(np.float32)
    train_y = np.asarray((0, 1, 0, 1, 0, 1, 0, 1), dtype=np.int64)
    validation_y = np.asarray((0, 1, 0, 1), dtype=np.int64)
    positions = np.asarray(CANONICAL_21_POSITIONS[:channels], dtype=np.float32)
    names = tuple(CANONICAL_21_CHANNELS[:channels])

    spline_fit_rows: list[np.ndarray] = []
    scaler_fit_rows: list[np.ndarray] = []
    real_spline_fit = training.fit_spherical_spline_interpolator
    real_scaler_fit = training.fit_channel_scaler

    def recording_spline_fit(values: np.ndarray, *args: object, **kwargs: object):
        spline_fit_rows.append(np.asarray(values).copy())
        return real_spline_fit(values, *args, **kwargs)

    def recording_scaler_fit(values: np.ndarray, *args: object, **kwargs: object):
        scaler_fit_rows.append(np.asarray(values).copy())
        return real_scaler_fit(values, *args, **kwargs)

    monkeypatch.setattr(training, "fit_spherical_spline_interpolator", recording_spline_fit)
    monkeypatch.setattr(training, "fit_channel_scaler", recording_scaler_fit)

    configure_determinism(31)
    pretrained = TinyDualViewNet().state_dict()
    config = replace(
        TrainConfig(),
        epochs=2,
        batch_size=4,
        patience=2,
        segment_probability=0.0,
        time_shift_samples=0,
        noise_std=0.0,
        seed=31,
        device="cpu",
    )
    result = training.fit_dual_view_target(
        TinyDualViewNet,
        pretrained,
        train_x,
        train_y,
        validation_x,
        validation_y,
        positions,
        channel_names=names,
        config=config,
    )

    assert [len(values) for values in spline_fit_rows] == [8, 12]
    assert [len(values) for values in scaler_fit_rows] == [8, 8, 12, 12]
    assert float(np.max(spline_fit_rows[0])) < 2.0
    assert float(np.max(spline_fit_rows[1])) < 50.0
    assert all(float(np.max(values)) < 100.0 for values in scaler_fit_rows)
    expected_selection_mean, _ = fit_channel_scaler(train_x, names)
    expected_refit_mean, _ = fit_channel_scaler(
        np.concatenate((train_x, validation_x)), names
    )
    np.testing.assert_array_equal(
        result.selection_transform.native_scaler.mean,
        expected_selection_mean,
    )
    np.testing.assert_array_equal(
        result.refit_transform.native_scaler.mean,
        expected_refit_mean,
    )
    assert result.selection_transform.row_count == 8
    assert result.refit_transform.row_count == 12
    assert result.selection_start_sha256 == result.refit_start_sha256
    assert result.reset_batch_norm_modules == ("bn",)

    fit_call_count = (len(spline_fit_rows), len(scaler_fit_rows))
    probabilities = result.predict_test_once(test_x, batch_size=2)
    assert probabilities.shape == (3, 2)
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert result.test_consumed
    assert fit_call_count == (len(spline_fit_rows), len(scaler_fit_rows))
    with pytest.raises(RuntimeError, match="already been consumed"):
        result.predict_test_once(test_x)


def test_target_arbitrary_native_shape_has_deterministic_refit_hash() -> None:
    channels = 7
    n_times = 23
    rng = np.random.default_rng(91)
    train_x = rng.normal(size=(6, channels, n_times)).astype(np.float32)
    validation_x = rng.normal(size=(4, channels, n_times)).astype(np.float32)
    train_y = np.asarray((0, 1, 0, 1, 0, 1), dtype=np.int64)
    validation_y = np.asarray((0, 1, 0, 1), dtype=np.int64)
    positions = np.asarray(CANONICAL_21_POSITIONS[:channels], dtype=np.float32)
    names = tuple(CANONICAL_21_CHANNELS[:channels])
    configure_determinism(7)
    checkpoint = TinyDualViewNet().state_dict()
    config = replace(
        TrainConfig(),
        epochs=1,
        batch_size=4,
        patience=1,
        segment_probability=0.0,
        time_shift_samples=0,
        noise_std=0.0,
        seed=7,
        device="cpu",
    )
    first = training.fit_dual_view_target(
        TinyDualViewNet,
        checkpoint,
        train_x,
        train_y,
        validation_x,
        validation_y,
        positions,
        channel_names=names,
        config=config,
    )
    second = training.fit_dual_view_target(
        TinyDualViewNet,
        checkpoint,
        train_x,
        train_y,
        validation_x,
        validation_y,
        positions,
        channel_names=names,
        config=config,
    )
    assert first.selection_state_sha256 == second.selection_state_sha256
    assert first.refit_state_sha256 == second.refit_state_sha256
    assert first.selection_history == second.selection_history
    assert first.refit_history == second.refit_history
    assert first.native_channel_names == names
    assert first.n_times == n_times
