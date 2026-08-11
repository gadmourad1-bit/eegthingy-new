from __future__ import annotations

import copy
import hashlib
import inspect
import json
from dataclasses import asdict, replace
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest
import torch
from torch import nn

from ieee_mi import native_fbms_pretraining_cli as pretrain_cli
from ieee_mi import native_fbms_transfer as transfer
from ieee_mi import native_pretraining_cli as pretrain_core
from ieee_mi.baselines import make_model
from ieee_mi.config import CANONICAL_21_CHANNELS, PREPROCESSING
from ieee_mi.models import (
    CANONICAL_21_POSITIONS,
    CardinalFBMSNet,
    parameter_count,
)
from ieee_mi.native_pretraining import (
    PRIMARY_NATIVE_DEVELOPMENT_SOURCES,
    TargetFineTuneResult,
    state_dict_sha256,
)
from ieee_mi.training import TrainConfig


def _synthetic_target(*, n_channels: int = 8) -> transfer.NativeTargetSubject:
    generator = np.random.default_rng(20260719)
    labels = np.asarray([0, 1] * 9, dtype=np.int64)
    values = generator.normal(
        size=(len(labels), n_channels, 32)
    ).astype(np.float32)
    rows = np.arange(len(labels), dtype=np.int64)
    return transfer.NativeTargetSubject(
        dataset="cho2017",
        subject=16,
        fold=0,
        x=values,
        y=labels,
        positions=np.asarray(
            CANONICAL_21_POSITIONS[:n_channels], dtype=np.float32
        ),
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
                "array_sha256": hashlib.sha256(
                    b"synthetic-arrays"
                ).hexdigest(),
            }
        ),
    )


class _TinyModel(nn.Module):
    uses_positions = True

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.25))
        self.config = {"identity": "synthetic"}

    def forward(
        self, values: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        del positions
        score = values.mean(dim=(1, 2)) * self.scale
        return torch.stack((-score, score), dim=1)


def _synthetic_checkpoint() -> transfer.ImmutableNativeCheckpoint:
    state = {
        name: value.detach().cpu().clone()
        for name, value in _TinyModel().state_dict().items()
    }
    return transfer.ImmutableNativeCheckpoint(
        path="/synthetic/cardinal-fbms-checkpoint.pt",
        file_sha256=hashlib.sha256(b"synthetic-checkpoint-file").hexdigest(),
        size_bytes=123,
        state_sha256=state_dict_sha256(state),
        state=MappingProxyType(state),
        corpus=MappingProxyType(
            {"sha256": hashlib.sha256(b"corpus").hexdigest()}
        ),
        model=MappingProxyType({"identity": "cardinal_fbms"}),
        pretraining=MappingProxyType(
            {
                "training_config": {"seed": 7},
                "partition": {
                    "sha256": hashlib.sha256(b"partition").hexdigest()
                },
            }
        ),
        source_code=MappingProxyType(
            {"hash_algorithm": "sha256", "files": {}}
        ),
        checkpoint_schema=pretrain_cli.CHECKPOINT_SCHEMA,
    )


def _synthetic_fine_tune_result(
    model: nn.Module,
    state: dict[str, torch.Tensor],
    *,
    n_channels: int,
) -> TargetFineTuneResult:
    digest = state_dict_sha256(state)
    scaler = {
        "schema": "ieee-mi-channel-scaling-v1",
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


def _strict_checkpoint_payload(
    *, source_hashes: dict[str, str] | None = None
) -> dict[str, object]:
    transfer.configure_determinism(7)
    model = transfer._canonical_cardinal_factory()().cpu()
    assert isinstance(model, CardinalFBMSNet)
    state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    digest = hashlib.sha256(b"locked-source-entry").hexdigest()
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
    if source_hashes is None:
        source_hashes = pretrain_core._source_file_hashes(
            pretrain_cli.ENTRY_POINT.extra_source_files
        )
    corpus_hash = hashlib.sha256(
        json.dumps(
            identities, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    state_hash = state_dict_sha256(state)
    training_config = asdict(transfer.FROZEN_SOURCE_PRETRAIN_CONFIG)
    training_config["device"] = "cpu"
    return {
        "schema": pretrain_cli.CHECKPOINT_SCHEMA,
        "mode": "development",
        "confirmation_access": False,
        "model": {
            "identity": "cardinal_fbms",
            "class": f"{type(model).__module__}.{type(model).__qualname__}",
            "construction": {
                "n_channels": 21,
                "n_outputs": 2,
                "n_times": 320,
                "sfreq_hz": 128.0,
                "channel_names": list(CANONICAL_21_CHANNELS),
                "channel_positions": [
                    list(row) for row in CANONICAL_21_POSITIONS
                ],
            },
            "config": copy.deepcopy(model.config),
            "trainable_parameter_count": parameter_count(model),
            "uses_positions": True,
        },
        "model_state_dict": state,
        "training_config": training_config,
        "partition": transfer._frozen_partition_record(),
        "state_hashes": {
            "initial": state_hash,
            "selection": state_hash,
            "checkpoint": state_hash,
        },
        "best_epoch_zero_based": 0,
        "selected_epoch_count": 1,
        "corpus": {
            "sha256": corpus_hash,
            "cache_schema": PREPROCESSING["schema"],
            "montage_profile": "native",
            "source_record_count": len(identities),
            "source_identities": identities,
        },
        "source_code": {
            "hash_algorithm": "sha256",
            "files": source_hashes,
        },
    }


def test_transfer_surface_is_the_exact_independent_five_condition_family() -> None:
    assert transfer.TRANSFER_CONDITIONS == (
        "pretrained_cardinal_fbms",
        "scratch_cardinal_fbms_canonical_seeded",
        "scratch_cardinal_fbms_native_projected",
        "scratch_fbmsnet_native",
        "pretrained_indexed_fbmsnet_spherical_spline",
    )
    assert transfer.ARTIFACT_SCHEMA.startswith(
        "ieee-mi-native-cardinal-fbms-transfer-"
    )
    destinations = {action.dest for action in transfer.build_parser()._actions}
    assert destinations.isdisjoint(
        {"stage", "montage_profile", "build_cache", "subjects", "model"}
    )
    source = inspect.getsource(transfer)
    assert "build_subject_cache" not in source
    assert "classification_metrics" not in source


def test_family_and_condition_guards_fail_before_target_cache_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        transfer._core,
        "_sha256_file",
        lambda *args, **kwargs: pytest.fail("cache path was probed"),
    )
    with pytest.raises(ValueError, match="condition"):
        transfer.load_native_target_subject(
            "/must/not/be-probed",
            dataset="cho2017",
            subject=16,
            fold=0,
            condition="legacy_fbc_condition",
            seed=7,
        )

    monkeypatch.setattr(
        transfer,
        "load_immutable_native_checkpoint",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("checkpoint family mismatch")
        ),
    )
    monkeypatch.setattr(
        transfer,
        "_source_file_hashes",
        lambda: {"ieee_mi/native_fbms_transfer.py": "1" * 64},
    )
    monkeypatch.setattr(
        transfer,
        "load_native_target_subject",
        lambda *args, **kwargs: pytest.fail("target cache was opened"),
    )
    with pytest.raises(RuntimeError, match="family mismatch"):
        transfer.run_record(
            dataset="cho2017",
            subject=16,
            fold=0,
            condition=transfer.PRETRAINED_CARDINAL_FBMS,
            seed=7,
            cache_root=tmp_path / "cache",
            checkpoint_path=tmp_path / "wrong-family.pt",
            expected_checkpoint_file_sha256="0" * 64,
            output=tmp_path / "record",
            device="cpu",
        )


def test_strict_checkpoint_loader_pins_family_state_corpus_and_sources(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    payload = _strict_checkpoint_payload()
    torch.save(payload, checkpoint_path)
    file_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    loaded = transfer.load_immutable_native_checkpoint(
        checkpoint_path, expected_file_sha256=file_hash
    )
    assert loaded.checkpoint_schema == pretrain_cli.CHECKPOINT_SCHEMA
    assert loaded.model["identity"] == "cardinal_fbms"
    assert loaded.model["construction"]["channel_names"] == list(
        CANONICAL_21_CHANNELS
    )
    assert loaded.state_sha256 == state_dict_sha256(loaded.state)
    assert loaded.corpus["montage_profile"] == "native"
    assert loaded.source_code is not None

    with pytest.raises(RuntimeError, match="caller-pinned"):
        transfer.load_immutable_native_checkpoint(
            checkpoint_path, expected_file_sha256="0" * 64
        )

    source_hashes = pretrain_core._source_file_hashes(
        pretrain_cli.ENTRY_POINT.extra_source_files
    )
    source_hashes["ieee_mi/models.py"] = "0" * 64
    stale_path = tmp_path / "stale-source.pt"
    torch.save(
        _strict_checkpoint_payload(source_hashes=source_hashes), stale_path
    )
    with pytest.raises(RuntimeError, match="current source file differs"):
        transfer.load_immutable_native_checkpoint(
            stale_path,
            expected_file_sha256=hashlib.sha256(
                stale_path.read_bytes()
            ).hexdigest(),
        )

    changed_config = copy.deepcopy(payload)
    changed_config["training_config"]["learning_rate"] = 0.123
    config_path = tmp_path / "changed-config.pt"
    torch.save(changed_config, config_path)
    with pytest.raises(RuntimeError, match="configuration changed learning_rate"):
        transfer.load_immutable_native_checkpoint(
            config_path,
            expected_file_sha256=hashlib.sha256(
                config_path.read_bytes()
            ).hexdigest(),
        )

    changed_initial = copy.deepcopy(payload)
    changed_initial["state_hashes"]["initial"] = "0" * 64
    initial_path = tmp_path / "changed-initial.pt"
    torch.save(changed_initial, initial_path)
    with pytest.raises(RuntimeError, match="canonical scratch state differs"):
        transfer.load_immutable_native_checkpoint(
            initial_path,
            expected_file_sha256=hashlib.sha256(
                initial_path.read_bytes()
            ).hexdigest(),
        )

    changed_corpus = copy.deepcopy(payload)
    changed_corpus["corpus"]["sha256"] = "0" * 64
    corpus_path = tmp_path / "changed-corpus.pt"
    torch.save(changed_corpus, corpus_path)
    with pytest.raises(RuntimeError, match="corpus hash is not reproducible"):
        transfer.load_immutable_native_checkpoint(
            corpus_path,
            expected_file_sha256=hashlib.sha256(
                corpus_path.read_bytes()
            ).hexdigest(),
        )


@pytest.mark.parametrize(
    ("n_channels", "n_times"), ((15, 256), (21, 320), (64, 320))
)
def test_one_cardinal_checkpoint_runs_multiple_native_montages(
    n_channels: int, n_times: int
) -> None:
    canonical = transfer._canonical_cardinal_factory()().cpu().eval()
    state = {
        name: value.detach().cpu().clone()
        for name, value in canonical.state_dict().items()
    }
    repeats = int(np.ceil(n_channels / 21))
    positions = np.tile(
        np.asarray(CANONICAL_21_POSITIONS, dtype=np.float32), (repeats, 1)
    )[:n_channels]
    positions = positions + np.linspace(
        0.0, 1e-3, n_channels, dtype=np.float32
    )[:, None]
    channels = tuple(f"E{index}" for index in range(n_channels))
    target = transfer._make_factory(
        "cardinal_fbms",
        channel_names=channels,
        positions=positions,
        n_times=n_times,
        paired_geometry=False,
    )().cpu().eval()
    target.load_state_dict(state, strict=True)
    with torch.inference_mode():
        logits = target(
            torch.randn(2, n_channels, n_times),
            torch.as_tensor(positions),
        )
    assert logits.shape == (2, 2)
    assert torch.isfinite(logits).all()


def test_checkpoint_conversion_is_exact_on_canonical21_and_nonmutating() -> None:
    positions = torch.as_tensor(CANONICAL_21_POSITIONS, dtype=torch.float32)
    torch.manual_seed(73)
    cardinal = transfer._canonical_cardinal_factory()().cpu().eval()
    with torch.no_grad():
        cardinal.spatial_field.coefficients.mul_(0.83).add_(0.017)
        cardinal.spatial_bias.add_(0.013)
    source_state = {
        name: value.detach().cpu().clone()
        for name, value in cardinal.state_dict().items()
    }
    source_hash = state_dict_sha256(source_state)
    indexed_state = transfer.cardinal_fbms_checkpoint_to_indexed_fbmsnet(
        source_state
    )
    torch.manual_seed(991)
    repeated = transfer.cardinal_fbms_checkpoint_to_indexed_fbmsnet(source_state)
    assert state_dict_sha256(repeated) == state_dict_sha256(indexed_state)
    indexed = make_model(
        "fbmsnet",
        n_channels=21,
        n_outputs=2,
        n_times=320,
        sfreq=128.0,
        channel_names=CANONICAL_21_CHANNELS,
        channel_positions=positions,
    ).cpu().eval()
    indexed.load_state_dict(indexed_state, strict=True)
    values = torch.randn(3, 21, 320)
    with torch.inference_mode():
        cardinal_logits = cardinal(values, positions)
        indexed_logits = indexed(values)
    torch.testing.assert_close(
        indexed_logits, cardinal_logits, rtol=2e-5, atol=2e-5
    )
    assert state_dict_sha256(source_state) == source_hash


def test_interpolation_provenance_hashes_boundary_coordinate_arrays() -> None:
    """Raw cache/model geometry must not be confused with normalized copies."""

    target = _synthetic_target(n_channels=8)
    interpolator = transfer._core.fit_spherical_spline_interpolator(
        target.x[target.train_rows],
        target.positions,
        target.channel_names,
    )
    record = transfer._interpolation_record(interpolator, target)
    canonical = np.asarray(CANONICAL_21_POSITIONS, dtype=np.float32)

    assert record["source_positions_sha256"] == transfer._core._array_sha256(
        target.positions
    )
    assert record["target_positions_sha256"] == transfer._core._array_sha256(
        canonical
    )
    # The interpolator deliberately stores normalized float64 copies. Their
    # byte hashes differ even when they describe the same electrode geometry.
    assert record["source_positions_sha256"] != transfer._core._array_sha256(
        interpolator.source_positions
    )
    assert record["target_positions_sha256"] != transfer._core._array_sha256(
        interpolator.target_positions
    )


@pytest.mark.parametrize("condition", transfer.TRANSFER_CONDITIONS)
def test_all_five_conditions_use_independent_phase_models_and_test_once(
    condition: str, monkeypatch: pytest.MonkeyPatch
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
            refit, state_copy, n_channels=x_train.shape[1]
        )

    prediction_calls: list[int] = []

    def fake_predict(model, values, positions, **kwargs):
        del model, positions, kwargs
        prediction_calls.append(len(values))
        return np.tile(
            np.asarray([[0.4, 0.6]], dtype=np.float64), (len(values), 1)
        )

    monkeypatch.setattr(transfer, "_make_factory", fake_factory)
    monkeypatch.setattr(
        transfer, "fine_tune_pretrained_target", fake_fine_tune
    )
    monkeypatch.setattr(
        transfer,
        "cardinal_fbms_checkpoint_to_indexed_fbmsnet",
        lambda state: {
            name: value.detach().clone() for name, value in state.items()
        },
    )
    monkeypatch.setattr(transfer, "predict_probabilities", fake_predict)
    immutable_before = state_dict_sha256(checkpoint.state)
    evaluation = transfer.evaluate_transfer_condition(
        target,
        checkpoint,
        condition=condition,
        train_config=replace(
            TrainConfig(), epochs=1, patience=1, seed=7, device="cpu"
        ),
    )
    assert evaluation.requested_condition == condition
    assert evaluation.probabilities.shape == (len(target.test_rows), 2)
    assert fit_arguments and len(fit_arguments) == 1
    assert len(factory_models) >= 2
    assert prediction_calls == [len(target.test_rows)]
    assert state_dict_sha256(checkpoint.state) == immutable_before
    assert "test" not in inspect.signature(
        transfer.fine_tune_pretrained_target
    ).parameters


def test_record_publication_is_atomic_family_typed_and_write_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
        requested_condition=transfer.PRETRAINED_CARDINAL_FBMS,
        effective_model="cardinal_fbms",
        initialization="immutable_cardinal_fbms_pretraining_checkpoint",
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

    monkeypatch.setattr(
        transfer, "load_immutable_native_checkpoint", checkpoint_loader
    )
    monkeypatch.setattr(
        transfer, "load_native_target_subject", target_loader
    )
    monkeypatch.setattr(
        transfer, "evaluate_transfer_condition", evaluator
    )
    monkeypatch.setattr(
        transfer,
        "_source_file_hashes",
        lambda: {
            "ieee_mi/native_fbms_pretraining_cli.py": "1" * 64,
            "ieee_mi/native_fbms_transfer.py": "2" * 64,
        },
    )
    monkeypatch.setattr(
        transfer._core,
        "_environment_record",
        lambda device: {"requested_device": device, "synthetic": True},
    )
    output = tmp_path / "record"
    arguments = {
        "dataset": "cho2017",
        "subject": 16,
        "fold": 0,
        "condition": transfer.PRETRAINED_CARDINAL_FBMS,
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
    provenance = json.loads(
        (output / transfer.PROVENANCE_FILENAME).read_text(encoding="utf-8")
    )
    with np.load(
        output / transfer.PREDICTIONS_FILENAME, allow_pickle=False
    ) as archive:
        assert str(archive["schema"].item()) == transfer.PREDICTION_SCHEMA
        np.testing.assert_array_equal(archive["test_rows"], target.test_rows)
    assert provenance["schema"] == transfer.ARTIFACT_SCHEMA
    assert provenance["source_checkpoint"]["schema"] == (
        pretrain_cli.CHECKPOINT_SCHEMA
    )
    assert provenance["source_checkpoint"]["model"]["identity"] == (
        "cardinal_fbms"
    )
    assert provenance["condition"]["requested"] == (
        transfer.PRETRAINED_CARDINAL_FBMS
    )
    assert provenance["protocol"]["test"]["aggregate_metrics_emitted"] is False
    assert not list(tmp_path.glob(f".{output.name}.staging-*"))

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        transfer.run_record(**arguments)
    assert calls == {"checkpoint": 1, "target": 1, "evaluation": 1}
