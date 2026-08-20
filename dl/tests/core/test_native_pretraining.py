from __future__ import annotations

import copy
import hashlib
import inspect
import json
from dataclasses import asdict, replace
from types import MappingProxyType

import numpy as np
import pytest
import torch
from torch import nn

from benchmark.baselines import make_model
from benchmark.config import CANONICAL_21_CHANNELS
from benchmark.config import DATASETS, PREPROCESSING
from benchmark.data import fit_channel_scaler
from benchmark.models import CANONICAL_21_POSITIONS, CardinalFBMSNet, CardinalFBCNet
from benchmark import native_pretraining
from benchmark.native_pretraining import (
    PRIMARY_NATIVE_DEVELOPMENT_SOURCES,
    NativeMontageSubject,
    NativePretrainConfig,
    _balanced_batch,
    _make_dataset_pools,
    fine_tune_pretrained_target,
    fit_native_montage_pretraining,
    hashed_subject_partition,
    load_primary_native_pretraining_corpus,
    model_state_sha256,
    reset_batch_norm_running_stats,
    state_dict_sha256,
    validate_primary_pretraining_corpus,
)
from benchmark.training import TrainConfig


class TinyCoordinateBinaryNet(nn.Module):
    """Small shared model that accepts variable channel/time dimensions."""

    uses_positions = True

    def __init__(self) -> None:
        super().__init__()
        self.coordinate_projection = nn.Linear(3, 2, bias=False)
        self.batch_norm = nn.BatchNorm1d(2)
        self.classifier = nn.Linear(2, 2)

    def forward(
        self, x: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        temporal = torch.stack(
            (x.mean(dim=-1), x.square().mean(dim=-1)), dim=-1
        )
        coordinate_weights = torch.tanh(self.coordinate_projection(positions))
        features = (temporal * coordinate_weights[None]).mean(dim=1)
        return self.classifier(self.batch_norm(features))


def _normalized_positions(channels: int, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    positions = generator.normal(size=(channels, 3)).astype(np.float32)
    positions[:, 2] = np.abs(positions[:, 2]) + 0.3
    positions /= np.linalg.norm(positions, axis=1, keepdims=True)
    return positions


def _synthetic_corpus() -> tuple[NativeMontageSubject, ...]:
    records: list[NativeMontageSubject] = []
    definitions = (
        ("bnci2014_001", 3, 8, _normalized_positions(3, 101)),
        ("local_exp4", 5, 12, _normalized_positions(5, 102)),
    )
    for dataset_index, (dataset, channels, n_times, positions) in enumerate(
        definitions
    ):
        for subject in range(1, 6):
            generator = np.random.default_rng(1000 * (dataset_index + 1) + subject)
            y = np.asarray([0, 1] * 4, dtype=np.int64)
            x = generator.normal(size=(len(y), channels, n_times)).astype(np.float32)
            x[y == 1] += np.float32(0.35)
            records.append(
                NativeMontageSubject(
                    dataset=dataset,
                    subject=subject,
                    x=x,
                    y=y,
                    positions=positions.copy(),
                    channel_names=tuple(f"E{index}" for index in range(channels)),
                )
            )
    return tuple(records)


def _pretrain_config() -> NativePretrainConfig:
    return NativePretrainConfig(
        macro_epochs=2,
        batch_size_per_dataset=4,
        steps_per_macro_epoch=2,
        learning_rate=5e-3,
        weight_decay=0.0,
        patience=2,
        min_delta=0.0,
        label_smoothing=0.0,
        segment_probability=0.0,
        time_shift_samples=0,
        noise_std=0.0,
        gradient_clip=10.0,
        validation_batch_size=8,
        seed=23,
        device="cpu",
    )


def _target_config() -> TrainConfig:
    return TrainConfig(
        epochs=3,
        batch_size=4,
        learning_rate=5e-3,
        weight_decay=0.0,
        patience=3,
        min_delta=0.0,
        label_smoothing=0.0,
        segment_probability=0.0,
        time_shift_samples=0,
        noise_std=0.0,
        gradient_clip=10.0,
        seed=29,
        device="cpu",
    )


def test_cardinal_fbc_same_instance_accepts_256_and_320_samples() -> None:
    positions = torch.tensor(CANONICAL_21_POSITIONS)
    torch.manual_seed(91)
    model = make_model(
        "cardinal_fbc",
        n_channels=21,
        n_outputs=2,
        n_times=320,
        sfreq=128.0,
        channel_names=CANONICAL_21_CHANNELS,
        channel_positions=positions,
    ).eval()
    assert isinstance(model, CardinalFBCNet)
    configured_channels = model.spectral_filtering.n_chans
    with torch.inference_mode():
        # One shared model sees both a 15-channel/256-sample montage and its
        # configured 21-channel/320-sample montage.
        features_256 = model.encode_floor(
            torch.randn(2, 15, 256), positions[:15]
        )
        features_320 = model.encode_floor(torch.randn(2, 21, 320), positions)
    assert features_256.shape == features_320.shape == (2, 1152)
    assert model.spectral_filtering.n_chans == configured_channels


def test_cardinal_fbc_dynamic_length_preserves_paired_fbc_mapping() -> None:
    positions = torch.tensor(CANONICAL_21_POSITIONS)
    torch.manual_seed(92)
    indexed = make_model(
        "fbcnet",
        n_channels=21,
        n_outputs=2,
        n_times=256,
        sfreq=128.0,
        channel_names=CANONICAL_21_CHANNELS,
        channel_positions=positions,
    ).eval()
    # Construct the Cardinal model for a different configured duration. Its
    # seeded reference modules are otherwise identical to the 256-sample FBC.
    torch.manual_seed(92)
    cardinal = make_model(
        "cardinal_fbc",
        n_channels=21,
        n_outputs=2,
        n_times=320,
        sfreq=128.0,
        channel_names=CANONICAL_21_CHANNELS,
        channel_positions=positions,
    ).eval()
    values = torch.randn(3, 21, 256)
    with torch.inference_mode():
        indexed_logits = indexed(values)
        cardinal_logits = cardinal(values, positions)
    torch.testing.assert_close(
        cardinal_logits, indexed_logits, rtol=1e-4, atol=1e-4
    )


def test_cardinal_fbms_same_instance_accepts_native_channel_and_time_grids() -> None:
    positions = torch.tensor(CANONICAL_21_POSITIONS)
    torch.manual_seed(93)
    model = make_model(
        "cardinal_fbms",
        n_channels=21,
        n_outputs=2,
        n_times=320,
        sfreq=128.0,
        channel_names=CANONICAL_21_CHANNELS,
        channel_positions=positions,
    ).eval()
    assert isinstance(model, CardinalFBMSNet)
    configured_channels = model.spectral_filtering.n_chans
    generator = torch.Generator().manual_seed(9301)
    positions_64 = torch.randn(64, 3, generator=generator)
    positions_64 /= torch.linalg.vector_norm(positions_64, dim=1, keepdim=True)
    with torch.inference_mode():
        features_256 = model.encode(torch.randn(2, 15, 256), positions[:15])
        features_320 = model.encode(torch.randn(2, 21, 320), positions)
        features_64 = model.encode(
            torch.randn(2, 64, 320, generator=generator), positions_64
        )
    assert (
        features_256.shape
        == features_320.shape
        == features_64.shape
        == (2, 1152)
    )
    assert model.spectral_filtering.n_chans == configured_channels


def test_cardinal_fbms_dynamic_length_preserves_paired_fbms_mapping() -> None:
    positions = torch.tensor(CANONICAL_21_POSITIONS)
    torch.manual_seed(94)
    indexed = make_model(
        "fbmsnet",
        n_channels=21,
        n_outputs=2,
        n_times=256,
        sfreq=128.0,
        channel_names=CANONICAL_21_CHANNELS,
        channel_positions=positions,
    ).eval()
    torch.manual_seed(94)
    cardinal = make_model(
        "cardinal_fbms",
        n_channels=21,
        n_outputs=2,
        n_times=320,
        sfreq=128.0,
        channel_names=CANONICAL_21_CHANNELS,
        channel_positions=positions,
    ).eval()
    values = torch.randn(3, 21, 256)
    with torch.inference_mode():
        indexed_logits = indexed(values)
        cardinal_logits = cardinal(values, positions)
    torch.testing.assert_close(
        cardinal_logits, indexed_logits, rtol=1e-4, atol=1e-4
    )


def test_primary_corpus_explicitly_rejects_bnci004_bipolar_data() -> None:
    record = _synthetic_corpus()[0]
    bipolar = NativeMontageSubject(
        dataset="BNCI2014-004",
        subject=1,
        x=record.x,
        y=record.y,
        positions=record.positions,
        channel_names=record.channel_names,
    )
    with pytest.raises(PermissionError, match="bipolar"):
        validate_primary_pretraining_corpus((bipolar, bipolar))


def test_hashed_partition_is_order_independent_and_subject_disjoint() -> None:
    corpus = _synthetic_corpus()
    direct = hashed_subject_partition(corpus)
    reversed_order = hashed_subject_partition(tuple(reversed(corpus)))
    assert direct == reversed_order
    assert set(direct.train).isdisjoint(direct.validation)
    assert set(direct.train) | set(direct.validation) == {
        record.key for record in corpus
    }
    for dataset in {record.key[0] for record in corpus}:
        assert sum(key[0] == dataset for key in direct.validation) == 1
        assert sum(key[0] == dataset for key in direct.train) == 4


def test_default_partition_preserves_established_subject_membership() -> None:
    partition = hashed_subject_partition(_synthetic_corpus())
    assert partition.validation == (
        ("bnci2014_001", "5"),
        ("local_exp4", "3"),
    )
    assert partition.salt == native_pretraining.DEFAULT_PARTITION_SALT


def test_balanced_batch_is_dataset_homogeneous_and_exactly_balanced() -> None:
    corpus = _synthetic_corpus()
    dataset = "bnci2014_001"
    keys = {record.key for record in corpus if record.key[0] == dataset}
    pool = _make_dataset_pools(corpus, keys)[dataset]
    x, y, positions = _balanced_batch(
        pool,
        batch_size=6,
        config=_pretrain_config(),
        generator=torch.Generator().manual_seed(7),
        device=torch.device("cpu"),
    )
    assert x.shape[1:] == (3, 8)
    assert positions.shape == (3, 3)
    assert torch.bincount(y, minlength=2).tolist() == [3, 3]


def test_balanced_batch_common_augmentation_is_seeded_and_deterministic() -> None:
    corpus = _synthetic_corpus()
    dataset = "bnci2014_001"
    keys = {record.key for record in corpus if record.key[0] == dataset}
    pool = _make_dataset_pools(corpus, keys)[dataset]
    plain_config = _pretrain_config()
    augmented_config = replace(
        plain_config,
        segment_probability=1.0,
        segment_count=4,
        time_shift_samples=2,
        noise_std=0.05,
    )
    seed = 812
    first_x, first_y, _ = _balanced_batch(
        pool,
        batch_size=6,
        config=augmented_config,
        generator=torch.Generator().manual_seed(seed),
        device=torch.device("cpu"),
    )
    second_x, second_y, _ = _balanced_batch(
        pool,
        batch_size=6,
        config=augmented_config,
        generator=torch.Generator().manual_seed(seed),
        device=torch.device("cpu"),
    )
    plain_x, plain_y, _ = _balanced_batch(
        pool,
        batch_size=6,
        config=plain_config,
        generator=torch.Generator().manual_seed(seed),
        device=torch.device("cpu"),
    )
    torch.testing.assert_close(first_x, second_x, rtol=0.0, atol=0.0)
    assert torch.equal(first_y, second_y)
    assert torch.equal(first_y, plain_y)
    assert torch.bincount(first_y, minlength=2).tolist() == [3, 3]
    assert not torch.equal(first_x, plain_x)


def test_pretraining_averages_dataset_losses_equally_and_refits_deterministically() -> None:
    corpus = _synthetic_corpus()
    first = fit_native_montage_pretraining(
        TinyCoordinateBinaryNet, corpus, config=_pretrain_config()
    )
    second = fit_native_montage_pretraining(
        TinyCoordinateBinaryNet, tuple(reversed(corpus)), config=_pretrain_config()
    )
    assert first.partition == second.partition
    assert first.initial_state_sha256 == second.initial_state_sha256
    assert first.selection_state_sha256 == second.selection_state_sha256
    assert first.checkpoint_sha256 == second.checkpoint_sha256
    assert first.selection_history == second.selection_history
    assert first.refit_history == second.refit_history
    assert first.selection_scalers == second.selection_scalers
    assert first.refit_scalers == second.refit_scalers
    assert state_dict_sha256(first.checkpoint_state) == first.checkpoint_sha256
    assert model_state_sha256(first.model) == first.checkpoint_sha256
    for epoch in first.selection_history:
        per_dataset = epoch["training_dataset_loss"]
        assert isinstance(per_dataset, dict)
        assert np.isclose(
            epoch["training_equal_dataset_loss"],
            np.mean(list(per_dataset.values())),
        )
    for dataset, scaler in first.selection_scalers.items():
        fit_keys = {
            tuple(value) for value in scaler["fit_subjects"]
        }
        assert fit_keys == {
            key for key in first.partition.train if key[0] == dataset
        }
        assert fit_keys.isdisjoint(first.partition.validation)
    for dataset, scaler in first.refit_scalers.items():
        assert {tuple(value) for value in scaler["fit_subjects"]} == {
            record.key for record in corpus if record.key[0] == dataset
        }


def test_batch_norm_reset_retains_affine_and_clears_running_state() -> None:
    model = TinyCoordinateBinaryNet()
    with torch.no_grad():
        model.batch_norm.weight.copy_(torch.tensor([2.0, 3.0]))
        model.batch_norm.bias.copy_(torch.tensor([-1.0, 1.0]))
        model.batch_norm.running_mean.copy_(torch.tensor([4.0, 5.0]))
        model.batch_norm.running_var.copy_(torch.tensor([6.0, 7.0]))
        model.batch_norm.num_batches_tracked.fill_(9)
    weight = model.batch_norm.weight.detach().clone()
    bias = model.batch_norm.bias.detach().clone()
    assert reset_batch_norm_running_stats(model) == ("batch_norm",)
    torch.testing.assert_close(model.batch_norm.weight, weight)
    torch.testing.assert_close(model.batch_norm.bias, bias)
    torch.testing.assert_close(model.batch_norm.running_mean, torch.zeros(2))
    torch.testing.assert_close(model.batch_norm.running_var, torch.ones(2))
    assert int(model.batch_norm.num_batches_tracked) == 0


def test_target_fine_tune_restarts_selection_and_refit_without_state_carryover() -> None:
    assert "test" not in " ".join(
        inspect.signature(fine_tune_pretrained_target).parameters
    )
    corpus = _synthetic_corpus()
    pretrained = fit_native_montage_pretraining(
        TinyCoordinateBinaryNet, corpus, config=_pretrain_config()
    )
    checkpoint_before = copy.deepcopy(pretrained.checkpoint_state)
    generator = np.random.default_rng(333)
    positions = _normalized_positions(4, 334)
    y_train = np.asarray([0, 1] * 4, dtype=np.int64)
    y_validation = np.asarray([0, 1] * 2, dtype=np.int64)
    x_train = generator.normal(size=(8, 4, 10)).astype(np.float32)
    x_validation = generator.normal(size=(4, 4, 10)).astype(np.float32)
    x_train[y_train == 1] += np.float32(0.3)
    x_validation[y_validation == 1] += np.float32(0.3)
    # A large raw-domain validation offset makes a refit scaler accidentally
    # computed from selection-scaled arrays straightforward to detect.
    x_validation += np.float32(4.0)
    channel_names = ("C3", "C4", "Cz", "Pz")
    x_train_before = x_train.copy()
    x_validation_before = x_validation.copy()
    expected_selection_mean, expected_selection_std = fit_channel_scaler(
        x_train, channel_names
    )
    expected_refit_mean, expected_refit_std = fit_channel_scaler(
        np.concatenate((x_train, x_validation), axis=0), channel_names
    )

    first = fine_tune_pretrained_target(
        TinyCoordinateBinaryNet,
        pretrained.checkpoint_state,
        x_train,
        y_train,
        x_validation,
        y_validation,
        positions,
        channel_names=channel_names,
        config=_target_config(),
    )
    second = fine_tune_pretrained_target(
        TinyCoordinateBinaryNet,
        pretrained.checkpoint_state,
        x_train,
        y_train,
        x_validation,
        y_validation,
        positions,
        channel_names=channel_names,
        config=_target_config(),
    )
    assert first.selection_start_sha256 == first.refit_start_sha256
    assert first.selection_start_sha256 == second.selection_start_sha256
    assert first.selection_state_sha256 == second.selection_state_sha256
    assert first.refit_state_sha256 == second.refit_state_sha256
    assert first.selection_history == second.selection_history
    assert first.refit_history == second.refit_history
    assert first.reset_batch_norm_modules == ("batch_norm",)
    assert first.selection_scaler["fit_rows"] == "training only"
    assert first.refit_scaler["fit_rows"] == (
        "training plus validation; test excluded"
    )
    assert first.selection_scaler["row_count"] == len(x_train)
    assert first.refit_scaler["row_count"] == len(x_train) + len(x_validation)
    assert first.selection_scaler["channel_names"] == list(channel_names)
    np.testing.assert_allclose(
        first.selection_scaler["mean"], expected_selection_mean.reshape(-1)
    )
    np.testing.assert_allclose(
        first.selection_scaler["std"], expected_selection_std.reshape(-1)
    )
    np.testing.assert_allclose(
        first.refit_scaler["mean"], expected_refit_mean.reshape(-1)
    )
    np.testing.assert_allclose(
        first.refit_scaler["std"], expected_refit_std.reshape(-1)
    )
    assert not np.allclose(
        first.selection_scaler["mean"], first.refit_scaler["mean"]
    )
    np.testing.assert_array_equal(x_train, x_train_before)
    np.testing.assert_array_equal(x_validation, x_validation_before)
    # Loading/fine-tuning must never mutate the reusable CPU checkpoint.
    assert state_dict_sha256(pretrained.checkpoint_state) == state_dict_sha256(
        checkpoint_before
    )


def _install_locked_cache_stub(tmp_path, monkeypatch):
    payloads: dict[tuple[str, int], dict[str, object]] = {}
    calls: list[tuple[str, int, str]] = []
    definitions = {
        "bnci2014_001": (3, 8),
        "cho2017": (4, 10),
        "local_exp4": (5, 12),
    }
    for dataset_index, (dataset, subjects) in enumerate(
        PRIMARY_NATIVE_DEVELOPMENT_SOURCES.items()
    ):
        channels, n_times = definitions[dataset]
        positions = _normalized_positions(channels, 900 + dataset_index)
        channel_names = np.asarray(
            [f"{dataset[:2].upper()}{index}" for index in range(channels)]
        )
        for subject in subjects:
            generator = np.random.default_rng(100_000 * dataset_index + subject)
            labels = (
                np.asarray([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64)
                if dataset == "bnci2014_001"
                else np.asarray([0, 1] * 4, dtype=np.int64)
            )
            values = generator.normal(
                loc=subject / 10.0,
                scale=1.0 + dataset_index / 10.0,
                size=(len(labels), channels, n_times),
            ).astype(np.float32)
            array_hash = hashlib.sha256(
                np.ascontiguousarray(values).tobytes()
            ).hexdigest()
            identity = {
                # Reproduce the actual npz identity path: JSON turns registry
                # tuples into arrays before load_subject_cache returns it.
                "dataset": json.loads(json.dumps(asdict(DATASETS[dataset]))),
                "subject": subject,
                "montage_profile": "native",
                "coordinates": {
                    "point_electrode_continuity_evidence": True,
                },
                "array_sha256": array_hash,
            }
            payloads[(dataset, subject)] = {
                "x": values,
                "y": labels,
                "positions": positions,
                "channel_names": channel_names,
                "identity": identity,
            }
            path = (
                tmp_path
                / PREPROCESSING["schema"]
                / "native"
                / dataset
                / f"subject_{subject:03d}.npz"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"locked-cache:{dataset}:S{subject}".encode())

    def fake_load_subject_cache(
        dataset: str,
        subject: int,
        *,
        cache_root,
        montage_profile: str,
    ):
        del cache_root
        calls.append((dataset, subject, montage_profile))
        payload = payloads[(dataset, subject)]
        return {
            key: copy.deepcopy(value)
            for key, value in payload.items()
        }

    monkeypatch.setattr(
        native_pretraining, "load_subject_cache", fake_load_subject_cache
    )
    return payloads, calls


def test_locked_native_cache_loader_filters_scales_and_records_provenance(
    tmp_path, monkeypatch
) -> None:
    payloads, calls = _install_locked_cache_stub(tmp_path, monkeypatch)
    corpus = load_primary_native_pretraining_corpus(tmp_path)
    expected_keys = [
        (dataset, subject)
        for dataset, subjects in PRIMARY_NATIVE_DEVELOPMENT_SOURCES.items()
        for subject in subjects
    ]
    assert len(corpus.subjects) == len(corpus.sources) == 32
    assert [(dataset, subject, "native") for dataset, subject in expected_keys] == calls
    assert [record.key for record in corpus.subjects] == [
        (dataset, str(subject)) for dataset, subject in expected_keys
    ]
    assert corpus.cache_schema == "eeg-mi-cache-v2"
    assert corpus.montage_profile == "native"
    assert len(corpus.sha256) == 64

    by_key = {
        (source["dataset"], source["subject"]): (record, source)
        for record, source in zip(corpus.subjects, corpus.sources, strict=True)
    }
    bnci_record, bnci_source = by_key[("bnci2014_001", 1)]
    assert bnci_record.y.tolist() == [0, 1, 0, 1]
    assert bnci_source["authorized_rows"] == [0, 1, 4, 5]
    assert bnci_source["authorized_shape"][0] == 4
    raw = payloads[("bnci2014_001", 1)]["x"][
        bnci_source["authorized_rows"]
    ]
    names = tuple(payloads[("bnci2014_001", 1)]["channel_names"].tolist())
    np.testing.assert_allclose(bnci_record.x, raw, rtol=0.0, atol=0.0)
    assert bnci_source["scaling"] == "deferred to partition-aware pretraining"
    cache_path = tmp_path / PREPROCESSING["schema"] / "native" / "bnci2014_001" / "subject_001.npz"
    assert bnci_source["cache_file_sha256"] == hashlib.sha256(
        cache_path.read_bytes()
    ).hexdigest()
    assert bnci_source["cache_identity"] == payloads[("bnci2014_001", 1)][
        "identity"
    ]
    assert bnci_source["cache_array_sha256"] == payloads[("bnci2014_001", 1)][
        "identity"
    ]["array_sha256"]
    assert len(bnci_source["cache_identity_sha256"]) == 64
    assert len(bnci_source["authorized_rows_sha256"]) == 64


def test_locked_native_loader_guards_profile_forbidden_and_sealed_before_io(
    tmp_path, monkeypatch
) -> None:
    calls: list[object] = []

    def forbidden_loader(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("guard must run before cache I/O")

    monkeypatch.setattr(
        native_pretraining, "load_subject_cache", forbidden_loader
    )
    with pytest.raises(PermissionError, match="only montage_profile='native'"):
        load_primary_native_pretraining_corpus(
            tmp_path, montage_profile="harmonized"
        )
    assert not calls

    monkeypatch.setattr(
        native_pretraining,
        "PRIMARY_NATIVE_DEVELOPMENT_SOURCES",
        MappingProxyType({"bnci2014_004": (1,)}),
    )
    with pytest.raises(PermissionError, match="forbidden BNCI2014-004"):
        load_primary_native_pretraining_corpus(tmp_path)
    assert not calls

    monkeypatch.setattr(
        native_pretraining,
        "PRIMARY_NATIVE_DEVELOPMENT_SOURCES",
        MappingProxyType({"physionet_mi": (55,)}),
    )
    with pytest.raises(PermissionError, match="sealed subjects"):
        load_primary_native_pretraining_corpus(tmp_path)
    assert not calls


def test_locked_native_loader_requires_v2_schema_before_io(
    tmp_path, monkeypatch
) -> None:
    calls: list[object] = []

    def forbidden_loader(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("schema guard must run before cache I/O")

    monkeypatch.setattr(
        native_pretraining, "load_subject_cache", forbidden_loader
    )
    monkeypatch.setitem(PREPROCESSING, "schema", "eeg-mi-cache-v1")
    with pytest.raises(RuntimeError, match="locked v2 cache schema"):
        load_primary_native_pretraining_corpus(tmp_path)
    assert not calls
