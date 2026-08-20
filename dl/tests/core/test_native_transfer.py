from __future__ import annotations

import copy
import hashlib
import inspect
import json
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest
import torch
from mne.channels.interpolation import _make_interpolation_matrix
from torch import nn

from benchmark import native_transfer as transfer
from benchmark.baselines import make_model
from benchmark.config import CANONICAL_21_CHANNELS, PREPROCESSING
from benchmark.models import CANONICAL_21_POSITIONS, CardinalFBCNet
from benchmark.native_pretraining import (
    PRIMARY_NATIVE_DEVELOPMENT_SOURCES,
    TargetFineTuneResult,
    state_dict_sha256,
)
from benchmark.training import TrainConfig


def _synthetic_target(*, n_channels: int = 8) -> transfer.NativeTargetSubject:
    generator = np.random.default_rng(20260719)
    labels = np.asarray([0, 1] * 9, dtype=np.int64)
    values = generator.normal(size=(len(labels), n_channels, 32)).astype(np.float32)
    rows = np.arange(len(labels), dtype=np.int64)
    positions = np.asarray(CANONICAL_21_POSITIONS[:n_channels], dtype=np.float32)
    return transfer.NativeTargetSubject(
        dataset="cho2017",
        subject=16,
        fold=0,
        x=values,
        y=labels,
        positions=positions,
        channel_names=CANONICAL_21_CHANNELS[:n_channels],
        sessions=np.asarray(["session"] * len(labels)),
        runs=np.asarray(["run"] * len(labels)),
        train_rows=rows[:10],
        validation_rows=rows[10:14],
        test_rows=rows[14:],
        cache_path="/synthetic/native/cho2017/subject_016.npz",
        cache_file_sha256=hashlib.sha256(b"synthetic-cache").hexdigest(),
        cache_identity=MappingProxyType(
            {
                "montage_profile": "native",
                "array_sha256": hashlib.sha256(b"synthetic-arrays").hexdigest(),
            }
        ),
    )


class _TinyModel(nn.Module):
    uses_positions = True

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.25))
        self.config = {"identity": "synthetic"}

    def forward(self, values: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        del positions
        score = values.mean(dim=(1, 2)) * self.scale
        return torch.stack((-score, score), dim=1)


def _synthetic_checkpoint() -> transfer.ImmutableNativeCheckpoint:
    state = {name: value.detach().cpu().clone() for name, value in _TinyModel().state_dict().items()}
    return transfer.ImmutableNativeCheckpoint(
        path="/synthetic/checkpoint.pt",
        file_sha256=hashlib.sha256(b"synthetic-checkpoint-file").hexdigest(),
        size_bytes=123,
        state_sha256=state_dict_sha256(state),
        state=MappingProxyType(state),
        corpus=MappingProxyType({"sha256": hashlib.sha256(b"corpus").hexdigest()}),
        model=MappingProxyType({"identity": "cardinal_fbc"}),
        pretraining=MappingProxyType(
            {
                "training_config": {"seed": 7},
                "partition": {"sha256": hashlib.sha256(b"partition").hexdigest()},
            }
        ),
    )


def _synthetic_fine_tune_result(
    model: nn.Module,
    state: dict[str, torch.Tensor],
    *,
    n_channels: int,
) -> TargetFineTuneResult:
    digest = state_dict_sha256(state)
    scaler = {
        "schema": "eeg-mi-channel-scaling-v1",
        "fit_rows": "synthetic",
        "row_count": 1,
        "channel_names": [f"C{index}" for index in range(n_channels)],
        "mean": [0.0] * n_channels,
        "std": [1.0] * n_channels,
        "clip_standard_deviations": 12.0,
    }
    return TargetFineTuneResult(
        model=model,
        best_epoch=0,
        pretrained_checkpoint_sha256=digest,
        selection_start_sha256=digest,
        refit_start_sha256=digest,
        selection_state_sha256=digest,
        refit_state_sha256=state_dict_sha256(model.state_dict()),
        reset_batch_norm_modules=(),
        selection_scaler=copy.deepcopy(scaler),
        refit_scaler=copy.deepcopy(scaler),
        selection_history=[],
        refit_history=[],
    )


def _checkpoint_payload() -> dict[str, object]:
    state = {"synthetic_weight": torch.arange(5, dtype=torch.float32)}
    digest = hashlib.sha256(b"source-entry").hexdigest()
    identities: list[dict[str, object]] = []
    for dataset, subjects in PRIMARY_NATIVE_DEVELOPMENT_SOURCES.items():
        for subject in subjects:
            identities.append(
                {
                    "dataset": dataset,
                    "subject": subject,
                    "cache_file_sha256": digest,
                    "cache_identity_sha256": digest,
                    "cache_array_sha256": digest,
                    "authorized_rows_sha256": digest,
                }
            )
    return {
        "schema": transfer.CHECKPOINT_SCHEMA,
        "mode": "development",
        "confirmation_access": False,
        "model": {
            "identity": "cardinal_fbc",
            "construction": {
                "n_channels": 21,
                "n_outputs": 2,
                "n_times": 320,
                "sfreq_hz": 128.0,
                "channel_names": list(CANONICAL_21_CHANNELS),
                "channel_positions": [list(row) for row in CANONICAL_21_POSITIONS],
            },
        },
        "model_state_dict": state,
        "training_config": {"seed": 7, "device": "cpu"},
        "partition": {
            "train": [["cho2017", "1"]],
            "validation": [["cho2017", "2"]],
            "sha256": hashlib.sha256(b"partition").hexdigest(),
        },
        "state_hashes": {"checkpoint": state_dict_sha256(state)},
        "best_epoch_zero_based": 0,
        "selected_epoch_count": 1,
        "corpus": {
            "sha256": hashlib.sha256(b"locked-corpus").hexdigest(),
            "cache_schema": PREPROCESSING["schema"],
            "montage_profile": "native",
            "source_record_count": len(identities),
            "source_identities": identities,
        },
    }


def test_transfer_surface_is_exactly_the_locked_development_design() -> None:
    assert dict(transfer.NATIVE_TARGET_DEVELOPMENT_COHORTS) == {
        "cho2017": tuple(range(16, 53)),
        "physionet_mi": tuple(range(1, 55)),
    }
    assert dict(transfer.NATIVE_TARGET_FOLDS) == {
        "cho2017": tuple(range(5)),
        "physionet_mi": tuple(range(3)),
    }
    assert transfer.TRANSFER_CONDITIONS == (
        "pretrained_cardinal_fbc",
        "scratch_cardinal_fbc",
        "scratch_fbmsnet",
        "pretrained_indexed_fbcnet_spherical_spline",
    )
    parser_destinations = {action.dest for action in transfer.build_parser()._actions}
    assert parser_destinations.isdisjoint(
        {"stage", "montage_profile", "build_cache", "subjects", "artifact_mode"}
    )
    source = inspect.getsource(transfer)
    assert "build_subject_cache" not in source
    assert "classification_metrics" not in source


@pytest.mark.parametrize(
    ("dataset", "subject", "fold", "message"),
    (
        ("cho2017", 15, 0, "outside"),
        ("physionet_mi", 55, 0, "outside"),
        ("lee2019_mi", 1, 0, "restricted"),
        ("cho2017", 16, 5, "fold"),
    ),
)
def test_target_guard_fails_before_cache_path_or_loader_io(
    dataset: str,
    subject: int,
    fold: int,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        transfer,
        "_sha256_file",
        lambda *args, **kwargs: pytest.fail("cache path was probed"),
    )
    monkeypatch.setattr(
        transfer,
        "load_subject_cache",
        lambda *args, **kwargs: pytest.fail("cache loader was called"),
    )
    with pytest.raises((PermissionError, ValueError), match=message):
        transfer.load_native_target_subject(
            "/must/not/be/probed",
            dataset=dataset,
            subject=subject,
            fold=fold,
        )


def test_existing_output_refuses_before_checkpoint_or_cache_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "existing-record"
    output.mkdir()
    monkeypatch.setattr(
        transfer,
        "load_immutable_native_checkpoint",
        lambda *args, **kwargs: pytest.fail("checkpoint was opened"),
    )
    monkeypatch.setattr(
        transfer,
        "load_native_target_subject",
        lambda *args, **kwargs: pytest.fail("cache was opened"),
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        transfer.run_record(
            dataset="cho2017",
            subject=16,
            fold=0,
            condition=transfer.PRETRAINED_CARDINAL_FBC,
            seed=7,
            cache_root=tmp_path / "cache",
            checkpoint_path=tmp_path / "checkpoint.pt",
            expected_checkpoint_file_sha256="0" * 64,
            output=output,
            device="cpu",
        )


def test_checkpoint_is_read_once_and_pinned_to_expected_bytes(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(_checkpoint_payload(), checkpoint_path)
    file_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    loaded = transfer.load_immutable_native_checkpoint(
        checkpoint_path,
        expected_file_sha256=file_hash,
    )
    assert loaded.file_sha256 == file_hash
    assert loaded.state_sha256 == state_dict_sha256(loaded.state)
    assert loaded.corpus["montage_profile"] == "native"
    with pytest.raises(RuntimeError, match="caller-pinned"):
        transfer.load_immutable_native_checkpoint(
            checkpoint_path,
            expected_file_sha256="0" * 64,
        )


def test_spherical_spline_is_deterministic_amplitude_free_and_preserves_constants() -> None:
    generator = np.random.default_rng(51)
    train = generator.normal(size=(6, 8, 40)).astype(np.float32)
    positions = np.asarray(CANONICAL_21_POSITIONS[:8], dtype=np.float32)
    first = transfer.fit_spherical_spline_interpolator(
        train,
        positions,
        CANONICAL_21_CHANNELS[:8],
    )
    second = transfer.fit_spherical_spline_interpolator(
        train * np.float32(37.0) + np.float32(11.0),
        positions,
        CANONICAL_21_CHANNELS[:8],
    )
    np.testing.assert_array_equal(first.matrix, second.matrix)
    assert first.matrix_sha256 == second.matrix_sha256
    np.testing.assert_allclose(
        first.matrix,
        _make_interpolation_matrix(
            positions.astype(np.float64),
            np.asarray(CANONICAL_21_POSITIONS, dtype=np.float64),
        ),
        rtol=0.0,
        atol=2e-9,
    )
    constant = np.full((2, 8, 17), 3.25, dtype=np.float32)
    transformed = first.transform(constant)
    assert transformed.shape == (2, 21, 17)
    np.testing.assert_allclose(transformed, 3.25, rtol=0.0, atol=1e-5)


def test_cardinal_checkpoint_conversion_matches_indexed_fbc_logits() -> None:
    positions = torch.as_tensor(CANONICAL_21_POSITIONS, dtype=torch.float32)
    torch.manual_seed(73)
    cardinal = make_model(
        "cardinal_fbc",
        n_channels=21,
        n_outputs=2,
        n_times=320,
        sfreq=128.0,
        channel_names=CANONICAL_21_CHANNELS,
        channel_positions=positions,
    ).eval()
    assert isinstance(cardinal, CardinalFBCNet)
    # Make this a nontrivial checkpoint, not merely the factory's paired floor.
    with torch.no_grad():
        cardinal.spatial_field.coefficients.mul_(0.83).add_(0.017)
        cardinal.spatial_bias.add_(0.013)
    source_state = {
        name: value.detach().cpu().clone()
        for name, value in cardinal.state_dict().items()
    }
    source_hash = state_dict_sha256(source_state)
    indexed_state = transfer.cardinal_checkpoint_to_indexed_fbcnet(source_state)
    torch.manual_seed(991)
    indexed_state_repeat = transfer.cardinal_checkpoint_to_indexed_fbcnet(source_state)
    assert state_dict_sha256(indexed_state_repeat) == state_dict_sha256(indexed_state)
    indexed = make_model(
        "fbcnet",
        n_channels=21,
        n_outputs=2,
        n_times=320,
        sfreq=128.0,
        channel_names=CANONICAL_21_CHANNELS,
        channel_positions=positions,
    ).eval()
    indexed.load_state_dict(indexed_state, strict=True)
    values = torch.randn(3, 21, 320)
    with torch.inference_mode():
        cardinal_logits = cardinal(values, positions)
        indexed_logits = indexed(values)
    torch.testing.assert_close(indexed_logits, cardinal_logits, rtol=2e-5, atol=2e-5)
    assert state_dict_sha256(source_state) == source_hash


@pytest.mark.parametrize("condition", transfer.TRANSFER_CONDITIONS)
def test_each_condition_uses_two_fresh_factory_models_and_never_passes_test_to_fit(
    condition: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _synthetic_target()
    checkpoint = _synthetic_checkpoint()
    factory_models: list[nn.Module] = []
    fit_arguments: list[tuple[np.ndarray, np.ndarray]] = []

    def fake_factory(*args, **kwargs):
        del args, kwargs

        def construct() -> nn.Module:
            model = _TinyModel()
            factory_models.append(model)
            return model

        return construct

    def fake_fine_tune(
        factory,
        state,
        x_train,
        y_train,
        x_validation,
        y_validation,
        positions,
        *,
        channel_names,
        config,
    ):
        del y_train, y_validation, positions, channel_names, config
        selection = factory()
        refit = factory()
        assert selection is not refit
        selection.load_state_dict(state, strict=True)
        refit.load_state_dict(state, strict=True)
        with torch.no_grad():
            selection.scale.add_(9.0)
        assert not torch.equal(selection.scale, refit.scale)
        fit_arguments.append((x_train.copy(), x_validation.copy()))
        state_copy = {
            name: value.detach().cpu().clone()
            for name, value in state.items()
        }
        return _synthetic_fine_tune_result(
            refit,
            state_copy,
            n_channels=x_train.shape[1],
        )

    monkeypatch.setattr(transfer, "_make_factory", fake_factory)
    monkeypatch.setattr(transfer, "fine_tune_pretrained_target", fake_fine_tune)
    monkeypatch.setattr(
        transfer,
        "cardinal_checkpoint_to_indexed_fbcnet",
        lambda state: {name: value.detach().clone() for name, value in state.items()},
    )
    monkeypatch.setattr(
        transfer,
        "predict_probabilities",
        lambda model, values, positions, **kwargs: np.tile(
            np.asarray([[0.4, 0.6]], dtype=np.float64), (len(values), 1)
        ),
    )
    before = state_dict_sha256(checkpoint.state)
    evaluation = transfer.evaluate_transfer_condition(
        target,
        checkpoint,
        condition=condition,
        train_config=replace(TrainConfig(), epochs=1, patience=1, seed=7, device="cpu"),
    )
    assert evaluation.probabilities.shape == (len(target.test_rows), 2)
    assert len(fit_arguments) == 1
    assert len(factory_models) >= 2
    assert state_dict_sha256(checkpoint.state) == before
    # The fitting API receives train and validation separately and has no test
    # argument; held-out prediction happens only after the returned refit.
    assert "test" not in inspect.signature(transfer.fine_tune_pretrained_target).parameters


def test_record_publication_is_atomic_prediction_only_and_refuses_second_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _synthetic_target()
    checkpoint = _synthetic_checkpoint()
    model = _TinyModel()
    state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    fine = _synthetic_fine_tune_result(model, state, n_channels=8)
    probabilities = np.tile(
        np.asarray([[0.45, 0.55]], dtype=np.float64),
        (len(target.test_rows), 1),
    )
    evaluation = transfer.TransferEvaluation(
        probabilities=probabilities,
        predicted_labels=np.ones(len(target.test_rows), dtype=np.int64),
        test_labels=target.y[target.test_rows].copy(),
        fine_tune=fine,
        requested_condition=transfer.PRETRAINED_CARDINAL_FBC,
        effective_model="cardinal_fbc",
        initialization="immutable_native_pretraining_checkpoint",
        initialization_state_sha256=state_dict_sha256(state),
        derived_state_sha256=None,
        model_class="synthetic.TinyModel",
        model_config={"identity": "synthetic"},
        trainable_parameter_count=1,
        channels=target.channel_names,
        positions=target.positions,
        interpolation=None,
    )
    calls = {"checkpoint": 0, "target": 0, "evaluation": 0}

    def checkpoint_loader(*args, **kwargs):
        calls["checkpoint"] += 1
        return checkpoint

    def target_loader(*args, **kwargs):
        calls["target"] += 1
        return target

    def evaluator(*args, **kwargs):
        calls["evaluation"] += 1
        return evaluation

    monkeypatch.setattr(transfer, "load_immutable_native_checkpoint", checkpoint_loader)
    monkeypatch.setattr(transfer, "load_native_target_subject", target_loader)
    monkeypatch.setattr(transfer, "evaluate_transfer_condition", evaluator)
    monkeypatch.setattr(
        transfer,
        "_source_file_hashes",
        lambda: {"src/benchmark/native_transfer.py": "1" * 64},
    )
    monkeypatch.setattr(
        transfer,
        "_environment_record",
        lambda device: {"requested_device": device, "synthetic": True},
    )
    output = tmp_path / "record"
    arguments = {
        "dataset": "cho2017",
        "subject": 16,
        "fold": 0,
        "condition": transfer.PRETRAINED_CARDINAL_FBC,
        "seed": 7,
        "cache_root": tmp_path / "cache",
        "checkpoint_path": tmp_path / "checkpoint.pt",
        "expected_checkpoint_file_sha256": "0" * 64,
        "output": output,
        "device": "cpu",
    }
    assert transfer.run_record(**arguments) == output.resolve()
    assert calls == {"checkpoint": 1, "target": 1, "evaluation": 1}
    assert {path.name for path in output.iterdir()} == {
        transfer.PREDICTIONS_FILENAME,
        transfer.PROVENANCE_FILENAME,
    }
    provenance = json.loads((output / transfer.PROVENANCE_FILENAME).read_text())
    with np.load(output / transfer.PREDICTIONS_FILENAME, allow_pickle=False) as archive:
        assert archive["probabilities"].shape == (len(target.test_rows), 2)
        np.testing.assert_array_equal(archive["test_rows"], target.test_rows)
        np.testing.assert_array_equal(archive["test_labels"], target.y[target.test_rows])
    assert provenance["mode"] == "development"
    assert provenance["confirmation_access"] is False
    assert provenance["analysis_policy"] == transfer.ANALYSIS_POLICY
    assert provenance["protocol"]["test"]["aggregate_metrics_emitted"] is False
    assert provenance["protocol"]["test"]["inferential_statistics_emitted"] is False
    assert "test_metrics" not in provenance
    assert not list(tmp_path.glob(f".{output.name}.staging-*"))
    assert not (tmp_path / f".{output.name}.native-transfer.lock").exists()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        transfer.run_record(**arguments)
    assert calls == {"checkpoint": 1, "target": 1, "evaluation": 1}
