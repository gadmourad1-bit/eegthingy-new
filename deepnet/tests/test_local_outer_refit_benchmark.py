from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import deepnet.local_outer_refit_benchmark as outer
from deepnet.cameo_net import CAMEOClassifier, CAMEOConfig
from deepnet.orbit_transport_net import OrbitTransportClassifier, OrbitTransportConfig
from deepnet.parity_fuse_net import ParityFuseClassifier, ParityFuseConfig
from deepnet.parity_net import HemiParityClassifier, ParityConfig


CHANNELS_3 = ("Cz", "C3", "C4")


def _synthetic_source(
    rows: int = 16, *, seed: int = 11
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = np.arange(rows, dtype=np.int64) % 2
    raw = rng.normal(size=(rows, 3, 101)).astype(np.float32)
    raw[np.arange(rows), 1 + labels] += np.sin(
        np.linspace(0.0, 8.0 * np.pi, 101, dtype=np.float32)
    )
    factors = rng.normal(size=(rows, 2, 3, 12))
    covariance = np.einsum(
        "nbct,nbdt->nbcd", factors, factors, optimize=True
    ) / 12.0
    covariance += 0.2 * np.eye(3)[None, None]
    return raw, covariance.astype(np.float32), labels


def _local_split_data(subject: int = 1) -> dict[str, np.ndarray]:
    runs = np.repeat(np.asarray((1, 2, 3, 4)).astype(str), 60)
    sessions = np.repeat(
        np.asarray([f"S{subject:02d}-R{run:02d}" for run in (1, 2, 3, 4)]),
        60,
    )
    labels = np.tile(np.arange(60, dtype=np.int64) % 2, 4)
    return {"y": labels, "runs": runs, "sessions": sessions}


def test_covariance_view_is_deterministic_spd_and_fully_contracted() -> None:
    rng = np.random.default_rng(5)
    raw = rng.normal(size=(3, 15, 256)).astype(np.float32)
    first = outer.derive_covariance_view(raw)
    second = outer.derive_covariance_view(raw.copy())
    assert first.shape == (3, 4, 15, 15)
    assert first.dtype == np.float32
    assert np.array_equal(first, second)
    assert np.linalg.eigvalsh(first).min() > 0.0
    contract = outer.covariance_view_contract()
    assert contract["bands_hz"] == [[8.0, 12.0], [11.0, 15.0], [14.0, 20.0], [20.0, 30.0]]
    assert len(contract["filter"]["sos_sha256"]) == 4
    assert contract["fit_scope"].startswith("independent deterministic")


def test_exact_local_split_contract_is_120_60_180_60() -> None:
    data = _local_split_data()
    train, validation, test, contract = outer._split_contract(data, subject=1)
    assert (len(train), len(validation), len(test)) == (120, 60, 60)
    assert contract["phase_a_train"]["runs"] == ["1", "2"]
    assert contract["phase_a_validation"]["runs"] == ["3"]
    assert contract["phase_b_source"]["runs"] == ["1", "2", "3"]
    assert contract["prediction_only_test"]["runs"] == ["4"]


def test_frozen_configuration_artifacts_still_match_the_registry() -> None:
    expected_types = {
        "cameo": CAMEOConfig,
        "hemiparity": ParityConfig,
        "parity_fuse": ParityFuseConfig,
        "orbit_v3": OrbitTransportConfig,
    }
    for model_name, expected_type in expected_types.items():
        config, identity = outer._load_frozen_config(
            model_name, seed=37, device="cpu"
        )
        assert isinstance(config, expected_type)
        assert config.seed == 37
        assert config.device == "cpu"
        assert len(identity["artifact_sha256"]) == 64


def test_cameo_fixed_refit_freezes_validation_route() -> None:
    raw, covariance, labels = _synthetic_source()
    config = CAMEOConfig(
        epochs=2,
        patience=1,
        batch_size=8,
        temporal_filters=2,
        temporal_kernel=9,
        dynamics_channels=2,
        dynamics_kernel=5,
        pool_kernel=25,
        pool_stride=10,
        dropout=0.0,
        mixture_names=("geo", "energy"),
        rho_grid=(0.0, 1.0),
        device="cpu",
    )
    classifier = CAMEOClassifier(config).fit_fixed_epochs(
        raw,
        covariance,
        labels,
        epochs=1,
        selected_mixture="energy",
        selected_rho=1.0,
        channels=CHANNELS_3,
    )
    assert classifier.selected_mixture_ == "energy"
    assert classifier.selected_rho_ == 1.0
    assert classifier.epochs_run_ == 1
    assert len(classifier.history_) == 1
    assert np.allclose(
        classifier.predict_proba(raw, covariance).sum(axis=1), 1.0
    )


def test_hemiparity_fixed_refit_has_no_validation_surface() -> None:
    raw, covariance, labels = _synthetic_source(seed=13)
    classifier = HemiParityClassifier(
        ParityConfig(
            epochs=2,
            patience=1,
            batch_size=8,
            temporal_filters=2,
            dynamics_channels=2,
            raw_rank=2,
            tangent_rank=2,
            dropout=0.0,
            device="cpu",
        )
    ).fit_fixed_epochs(
        raw,
        covariance,
        labels,
        epochs=1,
        channels=CHANNELS_3,
    )
    assert classifier.epochs_run_ == 1
    assert len(classifier.history_) == 1
    assert classifier.max_equivariance_error(raw, covariance) < 2e-5


def test_parity_fuse_zero_epoch_refit_is_the_source_anchor_checkpoint() -> None:
    raw, covariance, labels = _synthetic_source(seed=17)
    config = ParityFuseConfig(
        epochs=2,
        patience=1,
        batch_size=8,
        temporal_filters=4,
        temporal_kernel=7,
        dynamics_channels=4,
        dynamics_kernel=5,
        raw_dim=8,
        tangent_hidden=6,
        tangent_band_dim=4,
        tangent_dim=8,
        fusion_dim=8,
        gate_hidden=6,
        device="cpu",
    )
    classifier = ParityFuseClassifier(config).fit_fixed_epochs(
        raw,
        covariance,
        labels,
        epochs=0,
        channels=CHANNELS_3,
    )
    assert classifier.epochs_run_ == 0
    assert classifier.history_ == []
    for name, value in classifier.model_.state_dict().items():
        assert torch.equal(value, classifier.initial_model_state_[name])
    assert classifier.max_equivariance_error(raw, covariance) < 2e-5


def test_orbit_fixed_refit_preserves_validation_candidate() -> None:
    raw, covariance, labels = _synthetic_source(seed=19)
    config = OrbitTransportConfig(
        epochs=2,
        patience=1,
        batch_size=8,
        temporal_filters=2,
        temporal_kernel=9,
        dynamics_channels=2,
        dynamics_kernel=5,
        pool_kernel=25,
        pool_stride=10,
        normalization="group",
        orientation_rank=2,
        orientation_hidden=4,
        gate_hidden=4,
        candidate_names=("fused", "anchor"),
        device="cpu",
    )
    classifier = OrbitTransportClassifier(config).fit_fixed_epochs(
        raw,
        covariance,
        labels,
        epochs=1,
        selected_candidate="anchor",
        channels=CHANNELS_3,
    )
    assert classifier.selected_candidate_ == "anchor"
    assert classifier.epochs_run_ == 1
    assert len(classifier.history_) == 1
    assert classifier.max_equivariance_error(raw, covariance) < 2e-5


@pytest.mark.parametrize("model_name", outer.MODEL_NAMES)
def test_selection_and_refit_random_starts_match_under_declared_exception(
    model_name: str,
) -> None:
    raw_train, covariance_train, labels_train = _synthetic_source(seed=31)
    raw_validation, covariance_validation, labels_validation = _synthetic_source(
        rows=8, seed=37
    )
    if model_name == "cameo":
        config = CAMEOConfig(
            epochs=1,
            patience=1,
            batch_size=8,
            temporal_filters=2,
            temporal_kernel=9,
            dynamics_channels=2,
            dynamics_kernel=5,
            pool_kernel=25,
            pool_stride=10,
            dropout=0.0,
            mixture_names=("geo", "energy"),
            rho_grid=(0.0, 1.0),
            device="cpu",
        )
    elif model_name == "hemiparity":
        config = ParityConfig(
            epochs=1,
            patience=1,
            batch_size=8,
            temporal_filters=2,
            dynamics_channels=2,
            raw_rank=2,
            tangent_rank=2,
            dropout=0.0,
            device="cpu",
        )
    elif model_name == "parity_fuse":
        config = ParityFuseConfig(
            epochs=1,
            patience=1,
            batch_size=8,
            temporal_filters=4,
            temporal_kernel=7,
            dynamics_channels=4,
            dynamics_kernel=5,
            raw_dim=8,
            tangent_hidden=6,
            tangent_band_dim=4,
            tangent_dim=8,
            fusion_dim=8,
            gate_hidden=6,
            device="cpu",
        )
    else:
        config = OrbitTransportConfig(
            epochs=1,
            patience=1,
            batch_size=8,
            temporal_filters=2,
            temporal_kernel=9,
            dynamics_channels=2,
            dynamics_kernel=5,
            pool_kernel=25,
            pool_stride=10,
            normalization="group",
            orientation_rank=2,
            orientation_hidden=4,
            gate_hidden=4,
            candidate_names=("fused", "anchor"),
            device="cpu",
        )

    selection, selection_detail = outer._selection_fit(
        model_name,
        config,
        raw_train=raw_train,
        covariance_train=covariance_train,
        labels_train=labels_train,
        raw_validation=raw_validation,
        covariance_validation=covariance_validation,
        labels_validation=labels_validation,
        channels=CHANNELS_3,
    )
    raw_source = np.concatenate((raw_train, raw_validation))
    covariance_source = np.concatenate(
        (covariance_train, covariance_validation)
    )
    labels_source = np.concatenate((labels_train, labels_validation))
    refitted, detail = outer._refit(
        model_name,
        config,
        selection,
        selection_detail,
        raw_source=raw_source,
        covariance_source=covariance_source,
        labels_source=labels_source,
        channels=CHANNELS_3,
    )
    assert detail["reset_verified"] is True
    assert detail["selection_start_reset_sha256"] == detail[
        "refit_start_reset_sha256"
    ]
    assert refitted.epochs_run_ == selection_detail["selected_epoch_count"]


def test_run_one_never_passes_prediction_only_rows_to_selection_or_refit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = np.arange(120, dtype=np.int64)
    validation = np.arange(120, 180, dtype=np.int64)
    test = np.arange(180, 240, dtype=np.int64)
    raw = np.zeros((240, 15, 256), dtype=np.float32)
    raw[train] = 1.0
    raw[validation] = 3.0
    raw[test] = 99.0
    covariance = np.broadcast_to(
        np.eye(15, dtype=np.float32), (240, 4, 15, 15)
    ).copy()
    base = _local_split_data()
    data = {
        **base,
        "x": raw,
        "channel_names": np.asarray(
            (
                "Cz",
                "Pz",
                "C3",
                "C4",
                "P7",
                "P8",
                "Fz",
                "F7",
                "F8",
                "F3",
                "F4",
                "T7",
                "T8",
                "P3",
                "P4",
            )
        ),
    }
    calls: list[str] = []

    @dataclass(frozen=True)
    class FakeConfig:
        epochs: int = 1
        seed: int = 7
        device: str = "cpu"

    def fake_load(*args, **kwargs):
        return FakeConfig(), {"artifact_sha256": "a" * 64}

    def fake_selection(*args, **kwargs):
        calls.append("selection")
        assert kwargs["raw_train"].max() == 1.0
        assert kwargs["raw_validation"].max() == 3.0
        assert 99.0 not in kwargs["raw_train"]
        assert 99.0 not in kwargs["raw_validation"]
        return SimpleNamespace(), {
            "selected_epoch_count": 1,
            "selection_decision": {},
        }

    class FakeRefit:
        def predict_proba(self, raw_test, covariance_test):
            calls.append("prediction")
            assert np.all(raw_test == 99.0)
            return np.tile(np.asarray((0.4, 0.6)), (len(raw_test), 1))

    def fake_refit(*args, **kwargs):
        calls.append("refit")
        assert len(kwargs["raw_source"]) == 180
        assert kwargs["raw_source"].max() == 3.0
        assert 99.0 not in kwargs["raw_source"]
        return FakeRefit(), {"epoch_count": 1}

    monkeypatch.setattr(outer, "_load_frozen_config", fake_load)
    monkeypatch.setattr(outer, "_selection_fit", fake_selection)
    monkeypatch.setattr(outer, "_refit", fake_refit)
    record = outer.run_one(
        model_name="cameo",
        subject=1,
        seed=7,
        device="cpu",
        data=data,
        covariance=covariance,
        split=(train, validation, test),
        split_contract={},
        config_identity={"artifact_sha256": "a" * 64},
    )
    assert calls == ["selection", "refit", "prediction"]
    assert record["prediction_only_test"]["metrics"]["balanced_accuracy"] == 0.5


def test_atomic_per_record_resume_and_contract_rejection(tmp_path) -> None:
    output = tmp_path / "result.json"
    contract = {
        "subjects": [1],
        "seeds": [7],
        "model": "cameo",
        "nonce": "same-contract",
    }
    payload = outer._load_or_initialize(output, contract=contract, resume=True)
    labels = [index % 2 for index in range(60)]
    probabilities = [[0.75, 0.25] if label == 0 else [0.25, 0.75] for label in labels]
    metrics = outer._metrics(np.asarray(labels), np.asarray(probabilities))
    payload["records"].append(
        {
            "model": "cameo",
            "subject": 1,
            "seed": 7,
            "split": {
                "phase_a_train": {"count": 120},
                "phase_a_validation": {"count": 60},
                "phase_b_source": {"count": 180},
                "prediction_only_test": {"count": 60},
            },
            "phase_a": {"selected_epoch_count": 1},
            "phase_b": {"epoch_count": 1, "reset_verified": True},
            "cache_identity_sha256": "a" * 64,
            "prediction_only_test": {
                "metrics": metrics,
                "predictions": [
                    {
                        "row": index,
                        "label": label,
                        "probability_left": probability[0],
                        "probability_right": probability[1],
                    }
                    for index, (label, probability) in enumerate(
                        zip(labels, probabilities, strict=True)
                    )
                ],
            },
        }
    )
    outer._save_payload(output, payload, complete=True)
    resumed = outer._load_or_initialize(output, contract=contract, resume=True)
    assert resumed["completion"]["complete"] is True
    assert resumed["completion"]["actual_records"] == 1
    assert not list(tmp_path.glob(".result-*.json"))
    with pytest.raises(ValueError, match="different experiment contract"):
        outer._load_or_initialize(
            output,
            contract={**contract, "nonce": "changed"},
            resume=True,
        )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        outer._load_or_initialize(output, contract=contract, resume=False)


def test_resume_rejects_duplicate_json_keys(tmp_path) -> None:
    output = tmp_path / "result.json"
    output.write_text('{"contract_sha256": "x", "records": [], "records": []}\n')
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        outer._load_or_initialize(
            output,
            contract={"subjects": [1], "seeds": [7], "model": "cameo"},
            resume=True,
        )
