from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import deepnet.hemi_q_benchmark as benchmark


DEFAULT_SESSION_COUNTS = {
    "0train": 120,
    "1train": 140,
    "2train": 120,
    "3test": 120,
    "4test": 140,
}


def _split(
    sessions: tuple[str, ...],
    session_counts: dict[str, int],
    *,
    offset: float,
) -> dict[str, np.ndarray]:
    labels: list[int] = []
    session_ids: list[str] = []
    for session in sessions:
        count = session_counts[session]
        labels.extend([0, 1] * (count // 2))
        session_ids.extend([session] * count)
    label_array = np.asarray(labels, dtype=np.int64)
    n_trials = len(label_array)
    broadband = np.zeros(
        (n_trials, len(benchmark.BNCI_CHANNELS), benchmark.EPOCH_SAMPLES),
        dtype=np.float32,
    )
    time = np.linspace(0.0, 2.0 * np.pi, benchmark.EPOCH_SAMPLES, dtype=np.float32)
    broadband[:, 0] = np.sin(time)[None] + offset + 0.2 * label_array[:, None]
    broadband[:, 1] = np.cos(time)[None] + offset
    broadband[:, 2] = (
        np.sin(2.0 * time)[None] + offset + 0.2 * (1 - label_array[:, None])
    )
    covariances = np.empty((n_trials, 4, 3, 3), dtype=np.float32)
    for trial, label in enumerate(label_array):
        for band in range(4):
            diagonal = np.asarray(
                [1.8 if label == 1 else 0.8, 1.0, 1.8 if label == 0 else 0.8],
                dtype=np.float32,
            )
            diagonal += 0.01 * ((trial % 7) - 3) + 0.02 * band
            covariances[trial, band] = np.diag(diagonal) + 0.01 * np.ones((3, 3))
    return {
        "broadband": broadband,
        "covariances": covariances,
        "labels": label_array,
        "session_ids": np.asarray(session_ids, dtype=np.str_),
    }


def _fake_data(
    session_counts: dict[str, int] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    counts = dict(DEFAULT_SESSION_COUNTS if session_counts is None else session_counts)
    return {
        "train": _split(tuple(benchmark.FIT_SESSIONS), counts, offset=0.0),
        "validation": _split((benchmark.VALIDATION_SESSION,), counts, offset=0.1),
        "test": _split(tuple(benchmark.TEST_SESSIONS), counts, offset=0.2),
    }


def _perfect_probabilities(labels: np.ndarray) -> np.ndarray:
    return np.where(
        np.asarray(labels)[:, None] == 0,
        np.asarray([0.8, 0.2]),
        np.asarray([0.2, 0.8]),
    )


def _fit_detail(data: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
    counts = {
        session: int(
            np.sum(
                np.asarray(data[split]["session_ids"]).astype(str) == session
            )
        )
        for split in ("train", "validation", "test")
        for session in benchmark.EXPECTED_SPLITS[split]["sessions"]
    }
    return {
        "output": "sole_hemi_q_logit",
        "candidate_selection": False,
        "checkpoint_selection": {
            "metric": "validation_binary_cross_entropy",
            "direction": "minimize",
            "best_epoch": 3,
            "best_value": 0.42,
        },
        "epochs_run": 5,
        "parameter_count": 1234,
        "trainable_parameter_count": 1200,
        "teacher_parameter_count": 17,
        "teacher_was_used": True,
        "train_seconds": 0.01,
        "wall_seconds": 0.02,
        "n_train": len(data["train"]["labels"]),
        "n_validation": len(data["validation"]["labels"]),
        "n_test": len(data["test"]["labels"]),
        "train_session_ids": list(benchmark.FIT_SESSIONS),
        "validation_session_ids": [benchmark.VALIDATION_SESSION],
        "test_session_ids": list(benchmark.TEST_SESSIONS),
        "channels": list(benchmark.BNCI_CHANNELS),
        "reference": "provided_bipolar",
        "source_only_preprocessing": True,
        "unlabeled_test_calibration": False,
        "session_trial_counts": counts,
    }


def _parity_detail() -> dict[str, Any]:
    return {
        "definition": "max_abs_logit_x_plus_logit_reflection_x",
        "validation_max_abs_logit_error": 1e-7,
        "test_max_abs_logit_error": 2e-7,
    }


def _filter_detail() -> dict[str, Any]:
    return {
        "center_hz": [9.0, 12.0, 18.0],
        "bandwidth_hz": [3.0, 4.0, 5.0],
        "finite": True,
    }


def test_protocol_allows_only_official_variable_balanced_counts() -> None:
    assert benchmark.EXPECTED_SPLITS == {
        "train": {
            "counts": [240, 260, 280, 300, 320],
            "sessions": ["0train", "1train"],
        },
        "validation": {"counts": [120, 140, 160], "sessions": ["2train"]},
        "test": {
            "counts": [240, 260, 280, 300, 320],
            "sessions": ["3test", "4test"],
        },
    }
    benchmark._validate_loaded_data(_fake_data())
    assert benchmark.MODEL_CONTRACT["candidate_selection"] is False
    assert benchmark.MODEL_CONTRACT["checkpoint_selection_metric"] == (
        "validation_binary_cross_entropy"
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("121_trials", "expected one of"),
        ("unbalanced", "not class balanced"),
        ("wrong_session", "session ids"),
        ("nonfinite", "non-finite"),
    ],
)
def test_loaded_data_guard_rejects_protocol_corruption(
    mutation: str, message: str
) -> None:
    data = _fake_data()
    if mutation == "121_trials":
        for name in data["validation"]:
            data["validation"][name] = data["validation"][name][:-1]
    elif mutation == "unbalanced":
        data["validation"]["labels"][0] = 1
    elif mutation == "wrong_session":
        data["test"]["session_ids"][0] = "2train"
    elif mutation == "nonfinite":
        data["train"]["broadband"][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match=message):
        benchmark._validate_loaded_data(data)


def test_development_rejects_confirmation_subject_before_loading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loaded: list[int] = []
    monkeypatch.setattr(
        benchmark,
        "load_bnci2014_004_subject",
        lambda subject, **kwargs: loaded.append(subject),
    )
    with pytest.raises(ValueError, match="locked partition"):
        benchmark.run(
            mode="bnci004-dev",
            subjects=[5],
            seeds=[7],
            config=benchmark.HemiQFieldConfig(),
            device="cpu",
            output=tmp_path / "forbidden.json",
        )
    assert loaded == []


def test_fit_uses_covariance_teacher_only_on_train_and_validation_only_for_bce(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _fake_data()

    class SpyClassifier:
        def __init__(self, config: benchmark.HemiQFieldConfig) -> None:
            assert config.seed == 17
            assert config.device == "cpu"
            self.best_epoch_ = 3
            self.best_validation_loss_ = 0.42
            self.epochs_run_ = 5
            self.param_count_ = 1234
            self.trainable_param_count_ = 1200
            self.teacher_param_count_ = 17
            self.teacher_was_used_ = True
            self.train_seconds_ = 0.01

        def fit(self, raw_train, cov_train, y_train, raw_val, cov_val, y_val, *, channels):
            assert raw_train is data["train"]["broadband"]
            assert cov_train is data["train"]["covariances"]
            assert y_train is data["train"]["labels"]
            assert raw_val is data["validation"]["broadband"]
            assert cov_val is None
            assert y_val is data["validation"]["labels"]
            assert tuple(channels) == benchmark.BNCI_CHANNELS
            return self

        def predict_proba(self, raw):
            assert raw is data["test"]["broadband"]
            return _perfect_probabilities(data["test"]["labels"])

        def max_equivariance_error(self, raw):
            assert raw is not data["train"]["broadband"]
            return 1e-7

        def filter_diagnostics(self):
            return _filter_detail()

    monkeypatch.setattr(benchmark, "HemiQFieldClassifier", SpyClassifier)
    probabilities, fit, parity, filters = benchmark._fit_predict(
        data,
        config=benchmark.HemiQFieldConfig(),
        seed=17,
        device="cpu",
    )
    assert probabilities.shape == (260, 2)
    assert fit["candidate_selection"] is False
    assert fit["checkpoint_selection"] == {
        "metric": "validation_binary_cross_entropy",
        "direction": "minimize",
        "best_epoch": 3,
        "best_value": 0.42,
    }
    assert fit["session_trial_counts"] == DEFAULT_SESSION_COUNTS
    assert parity["test_max_abs_logit_error"] == pytest.approx(1e-7)
    assert filters == _filter_detail()


def test_development_run_loads_subject_once_for_multiple_declared_seeds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[int, bool]] = []

    def loader(subject: int, *, use_cache: bool, allow_confirmation: bool):
        assert use_cache is True
        calls.append((subject, allow_confirmation))
        return _fake_data()

    def fit(data, *, config, seed: int, device: str):
        return (
            _perfect_probabilities(data["test"]["labels"]),
            _fit_detail(data),
            _parity_detail(),
            _filter_detail(),
        )

    monkeypatch.setattr(benchmark, "load_bnci2014_004_subject", loader)
    monkeypatch.setattr(benchmark, "_fit_predict", fit)
    monkeypatch.setattr(benchmark, "_source_manifest", lambda: {"synthetic.py": "a" * 64})
    monkeypatch.setattr(benchmark, "_git_state", lambda: {"synthetic": True})
    monkeypatch.setattr(benchmark, "_run_environment", lambda: {"synthetic": "stable"})
    payload = benchmark.run(
        mode="bnci004-dev",
        subjects=[1, 2],
        seeds=[7, 17],
        config=benchmark.HemiQFieldConfig(),
        device="cpu",
        output=tmp_path / "development.json",
    )
    assert calls == [(1, False), (2, False)]
    assert payload["completion"]["complete"] is True
    assert payload["completion"]["actual_records"] == 4


def _counts_for_subject(subject: int) -> dict[str, int]:
    return {
        1: {"0train": 120, "1train": 120, "2train": 120, "3test": 120, "4test": 120},
        2: {"0train": 120, "1train": 140, "2train": 140, "3test": 120, "4test": 140},
        3: {"0train": 140, "1train": 140, "2train": 160, "3test": 140, "4test": 160},
        4: {"0train": 140, "1train": 160, "2train": 120, "3test": 160, "4test": 160},
    }[subject]


def _synthetic_development_payload(
    *, source: dict[str, str] | None = None
) -> dict[str, Any]:
    config = replace(benchmark.HemiQFieldConfig(), device="cpu", seed=7)
    device = "cpu"
    contract = benchmark._experiment_contract(
        mode="bnci004-dev",
        subjects=benchmark.DEVELOPMENT_SUBJECTS,
        seeds=[7],
        config=config,
        device=device,
        frozen_manifest_sha256=None,
    )
    source = {"synthetic.py": "a" * 64} if source is None else source
    environment = {"synthetic_environment": "stable"}
    payload: dict[str, Any] = {
        "schema_version": benchmark.SCHEMA_VERSION,
        **contract,
        "contract_sha256": benchmark._json_sha256(contract),
        "confirmation_access": False,
        "source_sha256": source,
        "source_manifest_sha256": benchmark._json_sha256(source),
        "environment": environment,
        "environment_sha256": benchmark._json_sha256(environment),
        "records": [],
        "summary": {},
    }
    for subject in benchmark.DEVELOPMENT_SUBJECTS:
        data = _fake_data(_counts_for_subject(subject))
        labels = data["test"]["labels"]
        probabilities = _perfect_probabilities(labels)
        digest = f"{subject:064x}"
        data_sha256 = {
            split: {
                name: digest
                for name in ("broadband", "covariances", "labels", "session_ids")
            }
            for split in ("train", "validation", "test")
        }
        payload["records"].append(
            {
                "model": benchmark.MODEL_NAME,
                "subject": subject,
                "seed": 7,
                "deterministic": True,
                "protocol": benchmark.protocol_metadata(subject),
                "data_sha256": data_sha256,
                "metrics": benchmark._metrics(labels, probabilities),
                "per_session_metrics": benchmark._session_metrics(
                    labels, probabilities, data["test"]["session_ids"]
                ),
                "fit": _fit_detail(data),
                "parity_diagnostics": _parity_detail(),
                "filter_diagnostics": _filter_detail(),
                "predictions": benchmark._prediction_trace(
                    labels, probabilities, data["test"]["session_ids"]
                ),
            }
        )
    payload["summary"] = benchmark._summarize(payload)
    benchmark._update_completion(payload, status="complete")
    return payload


def _freeze_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[Path, Path, Path]:
    source = {"deepnet/synthetic.py": "b" * 64}
    receipt = tmp_path / "permanent-hemiq-receipt.json"
    monkeypatch.setattr(benchmark, "_source_manifest", lambda: source)
    monkeypatch.setattr(benchmark, "CONFIRMATION_RECEIPT_PATH", receipt)
    development = tmp_path / "development.json"
    development.write_text(
        json.dumps(_synthetic_development_payload(source=source), allow_nan=False),
        encoding="utf-8",
    )
    output = tmp_path / "confirmation.json"
    frozen = tmp_path / "frozen.json"
    benchmark.write_frozen_manifest(
        development, frozen, confirmation_output=output, device="cpu"
    )
    return frozen, output, receipt


@pytest.mark.parametrize(
    ("subjects", "seeds", "token", "message"),
    [
        ([5, 6, 7, 8], [7], benchmark.CONFIRMATION_TOKEN, "exact ordered"),
        ([5, 6, 7, 8, 9], [7, 17], benchmark.CONFIRMATION_TOKEN, "sole frozen seed"),
        ([5, 6, 7, 8, 9], [7], None, "explicit confirmation token"),
    ],
)
def test_confirmation_contract_guards_precede_loader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    subjects: list[int],
    seeds: list[int],
    token: str | None,
    message: str,
) -> None:
    loaded: list[int] = []
    monkeypatch.setattr(
        benchmark,
        "load_bnci2014_004_subject",
        lambda subject, **kwargs: loaded.append(subject),
    )
    with pytest.raises(ValueError, match=message):
        benchmark.run(
            mode="bnci004-confirm",
            subjects=subjects,
            seeds=seeds,
            config=benchmark.HemiQFieldConfig(),
            device="cpu",
            output=tmp_path / "forbidden.json",
            confirmation_token=token,
        )
    assert loaded == []


def test_frozen_manifest_pins_output_receipt_config_source_and_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    frozen, output, receipt = _freeze_contract(monkeypatch, tmp_path)
    manifest, digest = benchmark._read_and_validate_frozen_manifest(frozen)
    assert len(digest) == 64
    contract = manifest["confirmation_contract"]
    assert contract["subjects"] == [5, 6, 7, 8, 9]
    assert contract["seeds"] == [7]
    assert contract["config"] == benchmark._json_config(
        replace(benchmark.HemiQFieldConfig(), device="cpu", seed=7)
    )
    assert contract["environment"] == {"synthetic_environment": "stable"}
    assert contract["output_path"] == str(output.resolve())
    assert contract["receipt_path"] == str(receipt.resolve())
    development = Path(manifest["development_artifact"]["path"])
    payload = json.loads(development.read_text(encoding="utf-8"))
    payload["records"][0]["predictions"][0]["label"] = 1
    development.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hash no longer matches"):
        benchmark._read_and_validate_frozen_manifest(frozen)


@pytest.mark.parametrize(
    ("corrupt", "message"),
    [
        ("records", "wrong record count"),
        ("predictions", "predictions are incomplete"),
        ("stored_prediction", "stored prediction is inconsistent"),
        ("session_counts", "test count is inconsistent"),
        ("candidate", "fit contract is invalid"),
        ("metric", "does not recompute"),
        ("environment", "environment digest is invalid"),
        ("completion", "not complete"),
    ],
)
def test_development_artifact_validator_rejects_corruption(
    corrupt: str, message: str
) -> None:
    payload = _synthetic_development_payload()
    if corrupt == "records":
        payload["records"].pop()
    elif corrupt == "predictions":
        payload["records"][0]["predictions"].pop()
    elif corrupt == "stored_prediction":
        payload["records"][0]["predictions"][0]["prediction"] = 1
    elif corrupt == "session_counts":
        payload["records"][0]["fit"]["n_test"] = 260
    elif corrupt == "candidate":
        payload["records"][0]["fit"]["candidate_selection"] = True
    elif corrupt == "metric":
        payload["records"][0]["metrics"]["balanced_accuracy"] = 0.1
    elif corrupt == "environment":
        payload["environment_sha256"] = "0" * 64
    elif corrupt == "completion":
        payload["completion"]["complete"] = False
    with pytest.raises(ValueError, match=message):
        benchmark._validate_development_payload(payload)


def test_receipt_is_claimed_before_loader_and_survives_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    frozen, output, receipt = _freeze_contract(monkeypatch, tmp_path)
    environment = {"synthetic_environment": "stable"}
    monkeypatch.setattr(benchmark, "_configure_torch_determinism", lambda: None)
    monkeypatch.setattr(benchmark, "_run_environment", lambda: dict(environment))
    loaded: list[int] = []

    def interrupted_loader(subject: int, *, use_cache: bool, allow_confirmation: bool):
        assert use_cache is True
        assert allow_confirmation is True
        loaded.append(subject)
        observed = json.loads(receipt.read_text(encoding="utf-8"))
        assert observed["state"] == "running"
        assert observed["output_path"] == str(output.resolve())
        raise RuntimeError("synthetic interruption")

    monkeypatch.setattr(benchmark, "load_bnci2014_004_subject", interrupted_loader)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        benchmark.run(
            mode="bnci004-confirm",
            subjects=benchmark.CONFIRMATION_SUBJECTS,
            seeds=[7],
            config=benchmark.HemiQFieldConfig(),
            device="cpu",
            output=output,
            frozen_manifest=frozen,
            confirmation_token=benchmark.CONFIRMATION_TOKEN,
        )
    assert loaded == [5]
    assert json.loads(receipt.read_text(encoding="utf-8"))["state"] == "interrupted"
    with pytest.raises(FileExistsError, match="permanent HemiQ confirmation receipt"):
        benchmark.run(
            mode="bnci004-confirm",
            subjects=benchmark.CONFIRMATION_SUBJECTS,
            seeds=[7],
            config=benchmark.HemiQFieldConfig(),
            device="cpu",
            output=output,
            frozen_manifest=frozen,
            confirmation_token=benchmark.CONFIRMATION_TOKEN,
        )
    environment["synthetic_environment"] = "changed"
    with pytest.raises(ValueError, match="environment differs"):
        benchmark.run(
            mode="bnci004-confirm",
            subjects=benchmark.CONFIRMATION_SUBJECTS,
            seeds=[7],
            config=benchmark.HemiQFieldConfig(),
            device="cpu",
            output=output,
            resume=True,
            frozen_manifest=frozen,
            confirmation_token=benchmark.CONFIRMATION_TOKEN,
        )
    assert loaded == [5]


def test_completed_receipt_blocks_all_reevaluation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    frozen, output, receipt = _freeze_contract(monkeypatch, tmp_path)
    monkeypatch.setattr(benchmark, "_configure_torch_determinism", lambda: None)
    monkeypatch.setattr(
        benchmark,
        "_run_environment",
        lambda: {"synthetic_environment": "stable"},
    )
    loaded: list[int] = []

    def loader(subject: int, *, use_cache: bool, allow_confirmation: bool):
        assert receipt.is_file()
        assert allow_confirmation is True
        loaded.append(subject)
        return _fake_data()

    def fit(data, *, config, seed: int, device: str):
        return (
            _perfect_probabilities(data["test"]["labels"]),
            _fit_detail(data),
            _parity_detail(),
            _filter_detail(),
        )

    monkeypatch.setattr(benchmark, "load_bnci2014_004_subject", loader)
    monkeypatch.setattr(benchmark, "_fit_predict", fit)
    payload = benchmark.run(
        mode="bnci004-confirm",
        subjects=benchmark.CONFIRMATION_SUBJECTS,
        seeds=[7],
        config=benchmark.HemiQFieldConfig(),
        device="cpu",
        output=output,
        frozen_manifest=frozen,
        confirmation_token=benchmark.CONFIRMATION_TOKEN,
    )
    assert loaded == [5, 6, 7, 8, 9]
    assert payload["completion"]["actual_records"] == 5
    completed = json.loads(receipt.read_text(encoding="utf-8"))
    assert completed["state"] == "complete"
    assert completed["output_sha256"] == benchmark._file_sha256(output)
    with pytest.raises(FileExistsError, match="already complete"):
        benchmark.run(
            mode="bnci004-confirm",
            subjects=benchmark.CONFIRMATION_SUBJECTS,
            seeds=[7],
            config=benchmark.HemiQFieldConfig(),
            device="cpu",
            output=output,
            resume=True,
            frozen_manifest=frozen,
            confirmation_token=benchmark.CONFIRMATION_TOKEN,
        )
    assert loaded == [5, 6, 7, 8, 9]


def test_manifest_rejects_a_different_configuration_before_loader(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    frozen, output, _ = _freeze_contract(monkeypatch, tmp_path)
    loaded: list[int] = []
    monkeypatch.setattr(
        benchmark,
        "load_bnci2014_004_subject",
        lambda subject, **kwargs: loaded.append(subject),
    )
    changed = replace(
        benchmark.HemiQFieldConfig(),
        teacher_weight=benchmark.HemiQFieldConfig().teacher_weight + 0.01,
    )
    with pytest.raises(ValueError, match="configuration differs"):
        benchmark.run(
            mode="bnci004-confirm",
            subjects=benchmark.CONFIRMATION_SUBJECTS,
            seeds=[7],
            config=changed,
            device="cpu",
            output=output,
            frozen_manifest=frozen,
            confirmation_token=benchmark.CONFIRMATION_TOKEN,
        )
    assert loaded == []
