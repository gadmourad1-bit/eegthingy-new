from __future__ import annotations

import copy
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from benchmark import hemiq_v2_model
from benchmark.config import PROJECT_ROOT


def _toy_raw(n_rows: int, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = np.arange(n_rows, dtype=np.int64) % 2
    time = np.arange(320, dtype=np.float32) / 128.0
    carrier = np.sin(2.0 * np.pi * 14.0 * time)
    raw = rng.normal(scale=0.03, size=(n_rows, 3, 320)).astype(np.float32)
    for row, label in enumerate(labels):
        raw[row, 1] += carrier
        raw[row, 0 if label else 2] += carrier
    return raw, labels


def _small_config(*, seed: int = 7):
    config, _ = hemiq_v2_model.load_frozen_config(
        PROJECT_ROOT, seed=seed, device="cpu"
    )
    return type(config)(
        **{
            **config.__dict__,
            "epochs": 3,
            "patience": 2,
            "batch_size": 4,
            "n_filters": 3,
            "kernel_size": 15,
            "local_window": 5,
            "local_stride": 3,
            "width": 8,
        }
    )


def test_frozen_config_changes_only_declared_parent_fields() -> None:
    config, identity = hemiq_v2_model.load_frozen_config(
        PROJECT_ROOT, seed=17, device="cpu"
    )
    assert config.n_times == 320
    assert config.sfreq == 128.0
    assert config.seed == 17
    assert config.device == "cpu"
    assert identity["runtime_overrides"] == {"device": "cpu", "seed": 17}
    assert identity["parent_frozen_manifest"]["size_bytes"] == 6688


def test_frozen_config_accepts_parent_equal_formal_seed_and_device() -> None:
    config, identity = hemiq_v2_model.load_frozen_config(
        PROJECT_ROOT, seed=7, device="cuda"
    )
    assert config.seed == 7
    assert config.device == "cuda"
    assert config.n_times == 320
    assert config.sfreq == 128.0
    assert identity["runtime_overrides"] == {"device": "cuda", "seed": 7}


def test_frozen_config_rejects_symlink_and_hardlink(tmp_path: Path) -> None:
    root = tmp_path / "project"
    (root / "configs/hemiq").mkdir(parents=True)
    source_config = PROJECT_ROOT / hemiq_v2_model.CONFIG_RELATIVE_PATH
    target_config = root / hemiq_v2_model.CONFIG_RELATIVE_PATH
    target_config.symlink_to(source_config)
    with pytest.raises(hemiq_v2_model.HemiQV2Error):
        hemiq_v2_model.load_frozen_config(root, seed=7, device="cpu")
    target_config.unlink()
    target_config.hardlink_to(source_config)
    with pytest.raises(hemiq_v2_model.HemiQV2Error):
        hemiq_v2_model.load_frozen_config(root, seed=7, device="cpu")
    target_config.unlink()


def test_frozen_config_does_not_open_historical_parent_manifest(
    tmp_path: Path,
) -> None:
    root = tmp_path / "isolated-project"
    target_config = root / hemiq_v2_model.CONFIG_RELATIVE_PATH
    target_config.parent.mkdir(parents=True)
    target_config.write_bytes(
        (PROJECT_ROOT / hemiq_v2_model.CONFIG_RELATIVE_PATH).read_bytes()
    )
    assert not (root / hemiq_v2_model.PARENT_MANIFEST_RELATIVE_PATH).exists()
    config, identity = hemiq_v2_model.load_frozen_config(
        root,
        seed=11,
        device="cpu",
    )
    assert config.seed == 11
    assert identity["parent_frozen_manifest"]["sha256"] == (
        hemiq_v2_model.PARENT_MANIFEST_SHA256
    )


def test_config_reader_pins_ancestor_during_path_swap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_root = tmp_path / "config-root"
    config_root.mkdir()
    config_path = config_root / "frozen.json"
    original = b'{"frozen":"original"}'
    replacement = b'{"frozen":"replaced"}'
    config_path.write_bytes(original)
    moved_root = tmp_path / "moved-config-root"
    real_read = os.read
    swapped = False

    def swap_ancestor_then_read(descriptor: int, size: int) -> bytes:
        nonlocal swapped
        if not swapped:
            swapped = True
            config_root.rename(moved_root)
            config_root.mkdir()
            (config_root / config_path.name).write_bytes(replacement)
        return real_read(descriptor, size)

    monkeypatch.setattr(hemiq_v2_model.os, "read", swap_ancestor_then_read)
    assert hemiq_v2_model._read_unique_regular_bytes(config_path) == original
    assert swapped
    assert config_path.read_bytes() == replacement


@pytest.mark.parametrize(
    "raw",
    (
        np.zeros((2, 3, 320), dtype=np.float32),
        np.ones((2, 3, 320), dtype=np.float32),
        np.broadcast_to(
            np.linspace(-1e-20, 1e-20, 320, dtype=np.float32),
            (2, 3, 320),
        ).copy(),
    ),
)
def test_views_are_finite_float32_spd_on_degenerate_inputs(raw: np.ndarray) -> None:
    views = hemiq_v2_model.derive_hemiq_views(
        raw, channel_names=("C3", "Cz", "C4")
    )
    assert views["raw"].shape == raw.shape
    assert views["raw"].dtype == np.float32
    assert views["covariance"].shape == (2, 4, 3, 3)
    assert views["covariance"].dtype == np.float32
    assert np.all(np.linalg.eigvalsh(views["covariance"].astype(np.float64)) > 0)


def test_view_transform_is_trial_local_and_requires_supplied_bipolar_order() -> None:
    raw, _ = _toy_raw(4, seed=11)
    together = hemiq_v2_model.derive_hemiq_views(
        raw, channel_names=("C3", "Cz", "C4")
    )
    separately = [
        hemiq_v2_model.derive_hemiq_views(
            raw[index : index + 1], channel_names=("C3", "Cz", "C4")
        )
        for index in range(len(raw))
    ]
    assert np.array_equal(
        together["raw"], np.concatenate([value["raw"] for value in separately])
    )
    assert np.array_equal(
        together["covariance"],
        np.concatenate([value["covariance"] for value in separately]),
    )
    with pytest.raises(ValueError, match="exact supplied-bipolar"):
        hemiq_v2_model.derive_hemiq_views(
            raw[:, (2, 1, 0)], channel_names=("C4", "Cz", "C3")
        )


def test_selection_and_refit_reconstruct_exact_initial_state_and_zero_epoch() -> None:
    raw_train, y_train = _toy_raw(8, seed=13)
    raw_validation, y_validation = _toy_raw(4, seed=17)
    train_views = hemiq_v2_model.derive_hemiq_views(
        raw_train, channel_names=("C3", "Cz", "C4")
    )
    validation_views = hemiq_v2_model.derive_hemiq_views(
        raw_validation, channel_names=("C3", "Cz", "C4")
    )
    config = _small_config(seed=19)
    selection = hemiq_v2_model.HemiQHarmonizedV2Classifier(
        config
    ).fit_selection(
        train_views["raw"],
        train_views["covariance"],
        y_train,
        validation_views["raw"],
        y_validation,
    )
    source_raw = np.concatenate((train_views["raw"], validation_views["raw"]))
    source_cov = np.concatenate(
        (train_views["covariance"], validation_views["covariance"])
    )
    source_y = np.concatenate((y_train, y_validation))
    refit = hemiq_v2_model.HemiQHarmonizedV2Classifier(
        config
    ).fit_fixed_epochs(
        source_raw,
        source_cov,
        source_y,
        epochs=selection.selected_epoch_count_,
    )
    assert selection.initial_model_state_sha256_ == refit.initial_model_state_sha256_
    assert refit.epochs_run_ == selection.selected_epoch_count_
    zero = hemiq_v2_model.HemiQHarmonizedV2Classifier(
        config
    ).fit_fixed_epochs(source_raw, source_cov, source_y, epochs=0)
    assert zero.epochs_run_ == 0
    assert zero.history_ == []
    assert zero.initial_model_state_sha256_ == zero.model_state_sha256_


def test_validation_covariance_is_not_an_input_to_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_train, y_train = _toy_raw(8, seed=23)
    raw_validation, y_validation = _toy_raw(4, seed=29)
    train_views = hemiq_v2_model.derive_hemiq_views(
        raw_train, channel_names=("C3", "Cz", "C4")
    )
    validation_views = hemiq_v2_model.derive_hemiq_views(
        raw_validation, channel_names=("C3", "Cz", "C4")
    )
    seen_rows: list[int] = []
    original = hemiq_v2_model.FrozenTangentAnchor

    class TrackingAnchor(original):
        def fit(self, covariances, labels, mirror_index):
            seen_rows.append(len(covariances))
            return super().fit(covariances, labels, mirror_index)

    monkeypatch.setattr(hemiq_v2_model, "FrozenTangentAnchor", TrackingAnchor)
    hemiq_v2_model.HemiQHarmonizedV2Classifier(
        _small_config(seed=31)
    ).fit_selection(
        train_views["raw"],
        train_views["covariance"],
        y_train,
        validation_views["raw"],
        y_validation,
    )
    assert seen_rows == [len(raw_train)]


def test_reflection_is_exact_after_source_symmetric_scaling() -> None:
    raw, labels = _toy_raw(8, seed=37)
    views = hemiq_v2_model.derive_hemiq_views(
        raw, channel_names=("C3", "Cz", "C4")
    )
    classifier = hemiq_v2_model.HemiQHarmonizedV2Classifier(
        _small_config(seed=41)
    ).fit_fixed_epochs(
        views["raw"],
        views["covariance"],
        labels,
        epochs=0,
    )
    prepared = torch.from_numpy(classifier._prepare_raw(views["raw"]))
    reflected = prepared[:, (2, 1, 0), :]
    classifier.model_.eval()
    with torch.no_grad():
        direct = classifier.model_(prepared)
        mirrored = classifier.model_(reflected)
    assert torch.equal(direct, -mirrored)


def test_fixed_refit_repeats_forward_backward_and_optimizer_state() -> None:
    raw, labels = _toy_raw(8, seed=43)
    views = hemiq_v2_model.derive_hemiq_views(
        raw, channel_names=("C3", "Cz", "C4")
    )
    config = _small_config(seed=47)
    first = hemiq_v2_model.HemiQHarmonizedV2Classifier(
        config
    ).fit_fixed_epochs(
        views["raw"],
        views["covariance"],
        labels,
        epochs=1,
    )
    first_probability = first.predict_proba(views["raw"])
    second = hemiq_v2_model.HemiQHarmonizedV2Classifier(
        config
    ).fit_fixed_epochs(
        views["raw"],
        views["covariance"],
        labels,
        epochs=1,
    )
    second_probability = second.predict_proba(views["raw"])
    assert first.initial_model_state_sha256_ == second.initial_model_state_sha256_
    assert first.model_state_sha256_ == second.model_state_sha256_
    assert first.history_ == second.history_
    assert np.array_equal(first_probability, second_probability)


def test_view_contract_is_outcome_free_and_stable() -> None:
    contract = hemiq_v2_model.hemiq_view_contract()
    assert contract["channels"] == ["C3", "Cz", "C4"]
    assert contract["broadband_hz"] == [8.0, 30.0]
    assert len(contract["filter"]["teacher_sos_sha256"]) == 4
    lowered = repr(copy.deepcopy(contract)).lower()
    for forbidden in ("accuracy", "label", "score", "prediction"):
        assert forbidden not in lowered
