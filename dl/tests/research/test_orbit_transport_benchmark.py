from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

import benchmark.research.orbit_transport_benchmark as benchmark
from benchmark.research.orbit_transport_net import OrbitTransportConfig, OrbitTransportOutput


def _config() -> OrbitTransportConfig:
    return OrbitTransportConfig(epochs=2, patience=2, device="cpu", seed=7)


def _array_manifest(shape: tuple[int, ...], dtype: str) -> dict[str, object]:
    return {"sha256": "a" * 64, "dtype": dtype, "shape": list(shape)}


def _valid_development_payload(
    config: OrbitTransportConfig, source: dict[str, str]
) -> dict[str, object]:
    candidates = tuple(benchmark.OrbitTransportNet.CANDIDATE_NAMES)
    records: list[dict[str, object]] = []
    for subject in benchmark.DEVELOPMENT_SUBJECTS:
        predictions: list[dict[str, object]] = []
        labels: list[int] = []
        probabilities: list[list[float]] = []
        for index in range(144):
            run_id = str(index // 24)
            ordinal = index % 24
            label = ordinal % 2
            right = 0.8 if label else 0.2
            logit = math.log(right / (1.0 - right))
            labels.append(label)
            probabilities.append([1.0 - right, right])
            predictions.append(
                {
                    "window_index": index,
                    "provenance": {
                        "dataset": "BNCI2014-001",
                        "session": "1test",
                        "run_id": run_id,
                        "trial_index_within_run": ordinal,
                        "source_trial_index": index,
                    },
                    "label": label,
                    "selected_candidate": "energy",
                    "probability_left": 1.0 - right,
                    "probability_right": right,
                    "candidate_probability_right": {name: right for name in candidates},
                    "candidate_logits": {name: logit for name in candidates},
                }
            )
        labels_array = np.asarray(labels, dtype=np.int64)
        probability_array = np.asarray(probabilities, dtype=np.float64)
        metrics = benchmark._metrics(labels_array, probability_array)
        candidate_metrics = {name: dict(metrics) for name in candidates}
        data_manifest: dict[str, object] = {}
        for split, count in (("train", 120), ("validation", 24), ("test", 144)):
            data_manifest[split] = {
                "broadband": _array_manifest((count, 15, 251), "<f4"),
                "covariances": _array_manifest((count, 4, 15, 15), "<f4"),
                "labels": _array_manifest((count,), "<i8"),
                "run_ids": _array_manifest((count,), "<U1"),
            }
        fit = {
            "selected_candidate": "energy",
            "selection_source": "validation_only",
            "selection_trace": [
                {
                    "candidate": name,
                    "best_epoch": 1 if name == "energy" else 0,
                    "validation_loss": 0.4 if name == "energy" else 0.5,
                    "validation_balanced_accuracy": 0.8 if name == "energy" else 0.7,
                }
                for name in config.candidate_names
            ],
            "best_epoch": 1,
            "epochs_run": 2,
            "best_validation_loss": 0.4,
            "best_validation_balanced_accuracy": 0.8,
            "candidate_test_metrics_descriptive_only": candidate_metrics,
            "parameter_count": 18_211,
            "trainable_parameter_count": 18_211,
            "neural_parameter_count": 18_211,
            "neural_trainable_parameter_count": 18_211,
            "convex_anchor_learned_parameter_count": 481,
            "total_learned_parameter_count": 18_692,
            "effective_seed": 7,
            "n_train": 120,
            "n_validation": 24,
            "n_test": 144,
            "train_run_ids": ["0", "1", "2", "3", "4"],
            "validation_run_ids": ["5"],
            "test_run_ids": ["0", "1", "2", "3", "4", "5"],
            "equivariance": {
                "max_abs_selected_logit_sum": 0.0,
                "max_abs_candidate_logit_sum": {name: 0.0 for name in candidates},
                "max_abs_gate_difference": 0.0,
                "max_abs_energy_alpha_complement_error": 0.0,
                "max_abs_dynamics_alpha_complement_error": 0.0,
                "max_abs_energy_view_swap_error": 0.0,
                "max_abs_dynamics_view_swap_error": 0.0,
            },
            "routing": {
                "energy": {"mean": 0.3},
                "dynamics": {"mean": 0.2},
                "anchor": {"mean": 0.5},
            },
            "source_only_preprocessing": True,
            "unlabeled_test_calibration": False,
        }
        records.append(
            {
                "subject": subject,
                "seed": 7,
                "protocol": benchmark.protocol_metadata(subject),
                "data_sha256": data_manifest,
                "metrics": dict(metrics),
                "anchor_metrics": dict(metrics),
                "candidate_test_metrics_descriptive_only": candidate_metrics,
                "delta_balanced_accuracy_vs_anchor": 0.0,
                "fits": [fit],
                "predictions": predictions,
            }
        )
    contract = benchmark._experiment_contract(
        mode="bnci-dev",
        subjects=benchmark.DEVELOPMENT_SUBJECTS,
        seeds=(7,),
        folds=5,
        config=config,
        frozen_manifest_sha256=None,
    )
    expected_keys = benchmark._expected_record_keys(
        benchmark.DEVELOPMENT_SUBJECTS, (7,)
    )
    payload: dict[str, object] = {
        "schema_version": benchmark.SCHEMA_VERSION,
        "architecture": {"name": benchmark.ARCHITECTURE},
        **contract,
        "contract_sha256": benchmark._json_sha256(contract),
        "confirmation_access": False,
        "candidate_selection_source": "validation_only",
        "candidate_test_metrics_are_descriptive_only": True,
        "source_sha256": source,
        "records": records,
        "summary": {},
        "completion": {
            "status": "complete",
            "complete": True,
            "expected_record_keys": expected_keys,
            "actual_record_keys": expected_keys,
            "expected_records": 4,
            "actual_records": 4,
        },
    }
    payload["summary"] = benchmark._summarize(payload)
    return payload


def _write_valid_development(
    path: Path, config: OrbitTransportConfig, source: dict[str, str]
) -> None:
    path.write_text(
        json.dumps(_valid_development_payload(config, source), sort_keys=True),
        encoding="utf-8",
    )


def _forbid_confirmation_loader(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    loaded: list[int] = []

    def forbidden(subject: int):
        loaded.append(subject)
        raise AssertionError("confirmation loader must not be reached")

    monkeypatch.setattr(benchmark, "load_bnci2014_subject", forbidden)
    return loaded


def test_source_manifest_covers_all_loaded_first_party_dependencies() -> None:
    assert {
        "src/benchmark/research/__init__.py",
        "src/benchmark/research/orbit_transport_benchmark.py",
        "src/benchmark/research/orbit_transport_net.py",
        "src/benchmark/research/cameo_net.py",
        "src/benchmark/research/cameo_benchmark.py",
        "src/benchmark/research/parity_fuse_benchmark.py",
        "src/benchmark/research/parity_fuse_net.py",
        "src/benchmark/shared/augment.py",
        "src/benchmark/research/config.py",
        "src/benchmark/research/data.py",
        "src/benchmark/research/external_cho2017.py",
        "src/benchmark/research/external_bnci2014.py",
        "src/benchmark/shared/spd.py",
    } == set(benchmark._SOURCE_FILES)


@pytest.mark.parametrize(
    ("subjects", "seeds", "token", "message"),
    [
        ([5, 6, 7, 8], [7], benchmark.CONFIRMATION_TOKEN, "exact ordered subjects"),
        ([5, 6, 7, 8, 9], [7, 17], benchmark.CONFIRMATION_TOKEN, "sole frozen seed"),
        ([5, 6, 7, 8, 9], [7], None, "explicit confirmation token"),
    ],
)
def test_confirmation_guards_precede_loader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    subjects: list[int],
    seeds: list[int],
    token: str | None,
    message: str,
) -> None:
    loaded = _forbid_confirmation_loader(monkeypatch)
    with pytest.raises(ValueError, match=message):
        benchmark.run(
            mode="bnci-confirm",
            subjects=subjects,
            seeds=seeds,
            config=_config(),
            output=tmp_path / "must-not-exist.json",
            confirmation_token=token,
        )
    assert loaded == []
    assert not (tmp_path / "must-not-exist.json").exists()


def test_confirmation_requires_frozen_manifest_before_loader(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loaded = _forbid_confirmation_loader(monkeypatch)
    with pytest.raises(ValueError, match="frozen-manifest"):
        benchmark.run(
            mode="bnci-confirm",
            subjects=[5, 6, 7, 8, 9],
            seeds=[7],
            config=_config(),
            output=tmp_path / "must-not-exist.json",
            confirmation_token=benchmark.CONFIRMATION_TOKEN,
        )
    assert loaded == []


def test_local_runner_physically_excludes_recording_four(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    observed: list[tuple[int, int]] = []

    def inspect_then_stop(keys, config):
        observed.extend((int(key.subject), int(key.run)) for key in keys)
        raise RuntimeError("intentional stop")

    monkeypatch.setattr(benchmark, "load_sessions", inspect_then_stop)
    monkeypatch.setattr(benchmark, "_environment", lambda: {"test": True})
    output = tmp_path / "local.json"
    with pytest.raises(RuntimeError, match="intentional stop"):
        benchmark.run(
            mode="local-dev",
            subjects=[1],
            seeds=[7],
            config=_config(),
            output=output,
        )
    assert observed == [(1, int(run)) for run in benchmark.SUBJECT_RUNS[1][:3]]
    assert all(run != int(benchmark.SUBJECT_RUNS[1][3]) for _, run in observed)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["completion"]["status"] == "interrupted"


def test_frozen_manifest_pins_orbit_source_dev_file_and_summary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = {
        "src/benchmark/research/orbit_transport_net.py": "1" * 64,
        "src/benchmark/research/orbit_transport_benchmark.py": "2" * 64,
    }
    monkeypatch.setattr(benchmark, "_source_manifest", lambda: source)
    receipt = tmp_path / "study-receipt.json"
    monkeypatch.setattr(benchmark, "CONFIRMATION_RECEIPT_PATH", receipt)
    config = _config()
    development = tmp_path / "dev.json"
    _write_valid_development(development, config, source)
    frozen = tmp_path / "frozen.json"
    confirmation = tmp_path / "confirmation.json"
    manifest = benchmark.write_frozen_manifest(
        development,
        frozen,
        config=config,
        confirmation_output=confirmation,
    )
    assert manifest["source_sha256"] == source
    assert manifest["confirmation_contract"] == {
        "study_id": benchmark.CONFIRMATION_STUDY_ID,
        "dataset": "BNCI2014-001",
        "subjects": [5, 6, 7, 8, 9],
        "seeds": [7],
        "output_path": str(confirmation.resolve()),
        "receipt_path": str(receipt.resolve()),
    }
    assert manifest["development_artifact"]["sha256"] == benchmark._file_sha256(
        development
    )
    assert benchmark._validate_frozen_manifest(frozen, config) == manifest
    development.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact hash"):
        benchmark._validate_frozen_manifest(frozen, config)


def test_empty_development_records_cannot_be_frozen(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = {"src/benchmark/research/orbit_transport_net.py": "1" * 64}
    monkeypatch.setattr(benchmark, "_source_manifest", lambda: source)
    monkeypatch.setattr(
        benchmark, "CONFIRMATION_RECEIPT_PATH", tmp_path / "receipt.json"
    )
    config = _config()
    payload = _valid_development_payload(config, source)
    payload["records"] = [{}, {}, {}, {}]
    development = tmp_path / "malformed.json"
    development.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="record identities"):
        benchmark.write_frozen_manifest(
            development,
            tmp_path / "must-not-freeze.json",
            config=config,
            confirmation_output=tmp_path / "confirmation.json",
        )


def test_permanent_study_receipt_blocks_alternate_output_and_duplicate_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = {"src/benchmark/research/orbit_transport_net.py": "1" * 64}
    monkeypatch.setattr(benchmark, "_source_manifest", lambda: source)
    monkeypatch.setattr(benchmark, "_environment", lambda: {"test": True})
    receipt_path = tmp_path / "permanent-receipt.json"
    monkeypatch.setattr(benchmark, "CONFIRMATION_RECEIPT_PATH", receipt_path)
    config = _config()
    development = tmp_path / "dev.json"
    _write_valid_development(development, config, source)
    first_output = tmp_path / "confirmation-first.json"
    first_manifest = tmp_path / "frozen-first.json"
    benchmark.write_frozen_manifest(
        development,
        first_manifest,
        config=config,
        confirmation_output=first_output,
    )

    loaded = _forbid_confirmation_loader(monkeypatch)
    with pytest.raises(AssertionError, match="confirmation loader"):
        benchmark.run(
            mode="bnci-confirm",
            subjects=[5, 6, 7, 8, 9],
            seeds=[7],
            config=config,
            output=first_output,
            frozen_manifest=first_manifest,
            confirmation_token=benchmark.CONFIRMATION_TOKEN,
        )
    assert loaded == [5]
    assert receipt_path.is_file()
    assert json.loads(receipt_path.read_text())["state"] == "interrupted"

    duplicate_manifest = tmp_path / "frozen-copy.json"
    duplicate_manifest.write_bytes(first_manifest.read_bytes())
    with pytest.raises(FileExistsError):
        benchmark.run(
            mode="bnci-confirm",
            subjects=[5, 6, 7, 8, 9],
            seeds=[7],
            config=config,
            output=first_output,
            frozen_manifest=duplicate_manifest,
            confirmation_token=benchmark.CONFIRMATION_TOKEN,
        )
    assert loaded == [5]

    alternate_output = tmp_path / "confirmation-alternate.json"
    alternate_manifest = tmp_path / "frozen-alternate.json"
    benchmark.write_frozen_manifest(
        development,
        alternate_manifest,
        config=config,
        confirmation_output=alternate_output,
    )
    with pytest.raises(FileExistsError, match="permanent study"):
        benchmark.run(
            mode="bnci-confirm",
            subjects=[5, 6, 7, 8, 9],
            seeds=[7],
            config=config,
            output=alternate_output,
            frozen_manifest=alternate_manifest,
            confirmation_token=benchmark.CONFIRMATION_TOKEN,
        )
    assert loaded == [5]


def test_resume_rejects_environment_change_before_confirmation_loader(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = {"src/benchmark/research/orbit_transport_net.py": "1" * 64}
    monkeypatch.setattr(benchmark, "_source_manifest", lambda: source)
    monkeypatch.setattr(benchmark, "_environment", lambda: {"runtime": "first"})
    monkeypatch.setattr(
        benchmark, "CONFIRMATION_RECEIPT_PATH", tmp_path / "receipt.json"
    )
    config = _config()
    development = tmp_path / "dev.json"
    _write_valid_development(development, config, source)
    output = tmp_path / "confirmation.json"
    frozen = tmp_path / "frozen.json"
    benchmark.write_frozen_manifest(
        development, frozen, config=config, confirmation_output=output
    )
    loaded = _forbid_confirmation_loader(monkeypatch)
    with pytest.raises(AssertionError):
        benchmark.run(
            mode="bnci-confirm",
            subjects=[5, 6, 7, 8, 9],
            seeds=[7],
            config=config,
            output=output,
            frozen_manifest=frozen,
            confirmation_token=benchmark.CONFIRMATION_TOKEN,
        )
    assert loaded == [5]
    monkeypatch.setattr(benchmark, "_environment", lambda: {"runtime": "changed"})
    with pytest.raises(ValueError, match="different execution environment"):
        benchmark.run(
            mode="bnci-confirm",
            subjects=[5, 6, 7, 8, 9],
            seeds=[7],
            config=config,
            output=output,
            resume=True,
            frozen_manifest=frozen,
            confirmation_token=benchmark.CONFIRMATION_TOKEN,
        )
    assert loaded == [5]


def _output() -> OrbitTransportOutput:
    return OrbitTransportOutput(
        logit=torch.tensor([1.0, -1.0]),
        energy_logit=torch.tensor([0.7, -0.6]),
        dynamics_logit=torch.tensor([0.5, -0.4]),
        anchor_logit=torch.tensor([0.8, -0.7]),
        fusion_weights=torch.tensor([[0.2, 0.3, 0.5], [0.4, 0.1, 0.5]]),
        energy_alpha=torch.tensor([0.6, 0.4]),
        dynamics_alpha=torch.tensor([0.55, 0.45]),
        energy_orientation=torch.tensor([0.4, -0.4]),
        dynamics_orientation=torch.tensor([0.2, -0.2]),
        energy_view_logits=torch.tensor([[0.9, -0.5], [-0.6, 0.8]]),
        dynamics_view_logits=torch.tensor([[0.7, -0.3], [-0.4, 0.6]]),
    )


def test_prediction_trace_records_candidates_orientation_and_routing() -> None:
    output = _output()
    candidates = {
        name: benchmark._probabilities_from_logit(value.detach().numpy())
        for name, value in output.candidate_logits().items()
    }
    rows = benchmark._prediction_trace(
        np.asarray([1, 0]),
        candidates["fused"],
        candidates,
        output,
        "fused",
        [{"fold": 0}, {"fold": 1}],
    )
    assert rows[0]["selected_candidate"] == "fused"
    assert set(rows[0]["candidate_logits"]) == set(output.candidate_logits())
    assert rows[0]["fusion_weights"] == {
        "energy": pytest.approx(0.2),
        "dynamics": pytest.approx(0.3),
        "anchor": pytest.approx(0.5),
    }
    assert rows[0]["orientation"]["energy_alpha"] == pytest.approx(0.6)


def test_summary_keeps_selected_candidates_and_descriptive_candidate_metrics() -> None:
    metrics = {"accuracy": 0.8, "balanced_accuracy": 0.8, "roc_auc": 0.9}
    anchor = {"accuracy": 0.7, "balanced_accuracy": 0.7, "roc_auc": 0.8}
    payload = {
        "records": [
            {
                "subject": 1,
                "metrics": metrics,
                "anchor_metrics": anchor,
                "candidate_test_metrics_descriptive_only": {
                    "fused": metrics,
                    "anchor": anchor,
                },
                "fits": [
                    {
                        "selected_candidate": "fused",
                        "best_epoch": 2,
                        "parameter_count": 18_211,
                        "trainable_parameter_count": 18_211,
                        "neural_parameter_count": 18_211,
                        "neural_trainable_parameter_count": 18_211,
                        "convex_anchor_learned_parameter_count": 481,
                        "total_learned_parameter_count": 18_692,
                        "equivariance": {
                            "max_abs_selected_logit_sum": 0.0,
                            "max_abs_gate_difference": 0.0,
                        },
                        "routing": {
                            "energy": {"mean": 0.2},
                            "dynamics": {"mean": 0.3},
                            "anchor": {"mean": 0.5},
                        },
                    }
                ],
            }
        ]
    }
    summary = benchmark._summarize(payload)
    assert summary["selected_candidate_counts"] == {"fused": 1}
    assert summary["mean_delta_vs_anchor"] == pytest.approx(0.1)
    assert summary["candidate_balanced_accuracy_means_descriptive_only"] == {
        "anchor": pytest.approx(0.7),
        "fused": pytest.approx(0.8),
    }
    assert summary["neural_parameter_counts"] == [18_211]
    assert summary["convex_anchor_learned_parameter_counts"] == [481]
    assert summary["total_learned_parameter_counts"] == [18_692]
