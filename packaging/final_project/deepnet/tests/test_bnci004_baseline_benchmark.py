from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import deepnet.bnci004_baseline_benchmark as benchmark


def _split(
    sessions: tuple[str, ...], trials_per_session: int, *, offset: float
) -> dict[str, np.ndarray]:
    labels: list[int] = []
    session_ids: list[str] = []
    for session in sessions:
        labels.extend(([0, 1] * (trials_per_session // 2)))
        session_ids.extend([session] * trials_per_session)
    label_array = np.asarray(labels, dtype=np.int64)
    n_trials = len(label_array)
    broadband = np.zeros(
        (n_trials, len(benchmark.BNCI_CHANNELS), benchmark.EPOCH_SAMPLES),
        dtype=np.float32,
    )
    time = np.linspace(0.0, 2.0 * np.pi, benchmark.EPOCH_SAMPLES, dtype=np.float32)
    broadband[:, 0] = np.sin(time)[None] + offset + 0.2 * label_array[:, None]
    broadband[:, 1] = np.cos(time)[None] + offset
    broadband[:, 2] = np.sin(2.0 * time)[None] + offset + 0.2 * (1 - label_array[:, None])
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


def _fake_data() -> dict[str, dict[str, np.ndarray]]:
    return {
        "train": _split(tuple(benchmark.FIT_SESSIONS), 120, offset=0.0),
        "validation": _split((benchmark.VALIDATION_SESSION,), 160, offset=0.1),
        "test": _split(tuple(benchmark.TEST_SESSIONS), 160, offset=0.2),
    }


def _concatenate_splits(*parts: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        name: np.concatenate([part[name] for part in parts], axis=0)
        for name in parts[0]
    }


def _fake_variable_count_data() -> dict[str, dict[str, np.ndarray]]:
    return {
        "train": _concatenate_splits(
            _split(("0train",), 120, offset=0.0),
            _split(("1train",), 140, offset=0.0),
        ),
        "validation": _split(("2train",), 140, offset=0.1),
        "test": _concatenate_splits(
            _split(("3test",), 120, offset=0.2),
            _split(("4test",), 160, offset=0.2),
        ),
    }


def _perfect_probabilities(labels: np.ndarray) -> np.ndarray:
    return np.where(
        np.asarray(labels)[:, None] == 0,
        np.asarray([0.8, 0.2]),
        np.asarray([0.2, 0.8]),
    )


def _fit_detail(baseline: str) -> dict[str, Any]:
    return {
        "parameter_count": 11,
        "fit_seconds": 0.01,
        "wall_seconds": 0.02,
        "n_train": 240,
        "n_validation": 160,
        "n_test": 320,
        "train_session_ids": list(benchmark.FIT_SESSIONS),
        "validation_session_ids": [benchmark.VALIDATION_SESSION],
        "test_session_ids": list(benchmark.TEST_SESSIONS),
        "channels": list(benchmark.BNCI_CHANNELS),
        "reference": "provided_bipolar",
        "source_only_preprocessing": True,
        "unlabeled_test_calibration": False,
        "validation_role": benchmark.BASELINE_CONTRACTS[baseline]["validation_role"],
    }


def test_protocol_contract_is_the_fixed_bipolar_five_session_study() -> None:
    assert benchmark.BNCI_CHANNELS == ("C3", "Cz", "C4")
    assert benchmark.BNCI_SFREQ == 125.0
    assert benchmark.EXPECTED_SPLITS == {
        "train": {
            "sessions": ["0train", "1train"],
            "allowed_totals": [240, 260, 280, 300, 320],
        },
        "validation": {
            "sessions": ["2train"],
            "allowed_totals": [120, 140, 160],
        },
        "test": {
            "sessions": ["3test", "4test"],
            "allowed_totals": [240, 260, 280, 300, 320],
        },
    }
    for contract in benchmark.BASELINE_CONTRACTS.values():
        assert contract["channels"] == ["C3", "Cz", "C4"]
        assert contract["reference"] == "provided_bipolar"
        assert contract["unlabeled_test_calibration"] is False


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
            baselines=["riemann"],
            device="cpu",
            output=tmp_path / "forbidden.json",
        )
    assert loaded == []


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("wrong_count", "broadband shape"),
        ("wrong_session", "session ids"),
        ("unbalanced", "not class balanced"),
        ("overlap", "session ids"),
    ],
)
def test_loaded_data_guard_rejects_protocol_corruption(
    mutation: str, message: str
) -> None:
    data = _fake_data()
    if mutation == "wrong_count":
        data["test"]["broadband"] = data["test"]["broadband"][:-1]
    elif mutation == "wrong_session":
        data["test"]["session_ids"][0] = "2train"
    elif mutation == "unbalanced":
        data["validation"]["labels"][1] = 0
    elif mutation == "overlap":
        data["validation"]["session_ids"][:] = "0train"
    with pytest.raises(ValueError, match=message):
        benchmark._validate_loaded_data(data)


def test_loaded_data_guard_accepts_official_variable_balanced_session_counts() -> None:
    data = _fake_variable_count_data()
    benchmark._validate_loaded_data(data)
    shapes = benchmark._data_shape_manifest(data)
    assert shapes["train"]["labels"] == [260]
    assert shapes["validation"]["labels"] == [140]
    assert shapes["test"]["labels"] == [280]


def test_shallow_fits_train_and_uses_validation_only_for_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from deepnet import dnn_baselines

    data = _fake_data()

    class SpyClassifier:
        def __init__(self, architecture: str, **kwargs: Any) -> None:
            assert architecture == "shallow"
            assert kwargs["n_times"] == 251
            assert kwargs["sfreq"] == 125.0
            assert kwargs["lr_swap_prob"] == 0.5
            assert tuple(kwargs["channels"]) == ("C3", "Cz", "C4")
            self.param_count_ = 123
            self.epochs_run_ = 4
            self.fit_seconds_ = 0.01

        def fit(self, x_train, y_train, x_validation, y_validation):
            assert x_train is data["train"]["broadband"]
            assert y_train is data["train"]["labels"]
            assert x_validation is data["validation"]["broadband"]
            assert y_validation is data["validation"]["labels"]
            return self

        def predict_proba(self, x_test):
            assert x_test is data["test"]["broadband"]
            return _perfect_probabilities(data["test"]["labels"])

    monkeypatch.setattr(dnn_baselines, "TorchEEGClassifier", SpyClassifier)
    probabilities, detail = benchmark._fit_predict(
        "shallow_swap", data, seed=7, device="cpu"
    )
    assert probabilities.shape == (320, 2)
    assert detail["validation_role"] == "early_stopping_only"
    assert detail["reference"] == "provided_bipolar"


def test_riemann_never_calibrates_or_fits_on_outer_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from deepnet import baselines

    data = _fake_data()

    class SpyRiemann:
        def __init__(self, *, c: float, max_iter: int) -> None:
            assert (c, max_iter) == (1.0, 3000)

        def fit(self, covariances, labels):
            assert covariances is data["train"]["covariances"]
            assert labels is data["train"]["labels"]
            return self

        def calibrate(self, covariances):  # pragma: no cover - forbidden
            raise AssertionError("outer-test calibration is forbidden")

        def predict_proba(self, covariances):
            assert covariances is data["test"]["covariances"]
            return _perfect_probabilities(data["test"]["labels"])

    monkeypatch.setattr(baselines, "RiemannianTangentLogistic", SpyRiemann)
    _, detail = benchmark._fit_predict("riemann", data, seed=7, device="cpu")
    assert detail["validation_role"] == "unused_fixed_estimator"
    assert detail["unlabeled_test_calibration"] is False


def test_tangent_anchor_learns_reference_and_scaler_from_train_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from deepnet import tangent_anchor

    data = _fake_data()

    class SpyAnchor:
        def __init__(self, *, C: float, max_iter: int) -> None:
            assert (C, max_iter) == (1.0, 2000)
            self.param_count_ = 25

        def fit(self, covariances, labels):
            assert covariances is data["train"]["covariances"]
            assert labels is data["train"]["labels"]
            return self

        def predict_proba(self, covariances):
            assert covariances is data["test"]["covariances"]
            return _perfect_probabilities(data["test"]["labels"])

    monkeypatch.setattr(tangent_anchor, "TangentAnchorClassifier", SpyAnchor)
    _, detail = benchmark._fit_predict(
        "tangent_anchor", data, seed=7, device="cpu"
    )
    assert detail["validation_role"] == "unused_fixed_estimator"
    assert detail["source_only_preprocessing"] is True


def test_fbcsp_uses_only_train_to_fit_and_test_to_predict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _fake_data()

    class SpyFBCSP:
        def __init__(self, *, covariance_shrinkage: float) -> None:
            assert covariance_shrinkage == 0.1
            self.param_count_ = 17

        def fit(self, covariances, labels):
            assert covariances is data["train"]["covariances"]
            assert labels is data["train"]["labels"]
            return self

        def predict_proba(self, covariances):
            assert covariances is data["test"]["covariances"]
            return _perfect_probabilities(data["test"]["labels"])

    monkeypatch.setattr(benchmark, "ShrinkageFilterBankCSPLDA", SpyFBCSP)
    _, detail = benchmark._fit_predict("fb_csp_lda", data, seed=999, device="cpu")
    assert detail["validation_role"] == "unused_fixed_estimator"


def test_fbcsp_is_deterministic_and_well_formed() -> None:
    data = _fake_data()
    first = benchmark.ShrinkageFilterBankCSPLDA().fit(
        data["train"]["covariances"], data["train"]["labels"]
    )
    second = benchmark.ShrinkageFilterBankCSPLDA().fit(
        data["train"]["covariances"], data["train"]["labels"]
    )
    first_probabilities = first.predict_proba(data["test"]["covariances"])
    second_probabilities = second.predict_proba(data["test"]["covariances"])
    np.testing.assert_array_equal(first_probabilities, second_probabilities)
    np.testing.assert_allclose(first_probabilities.sum(axis=1), 1.0, atol=1e-12)
    assert first_probabilities.shape == (320, 2)
    assert first.param_count_ == 9


def test_deterministic_baselines_run_only_the_first_seed() -> None:
    assert list(benchmark._seeds_for_baseline("shallow_swap", [7, 17])) == [7, 17]
    for baseline in ("riemann", "tangent_anchor", "fb_csp_lda"):
        assert list(benchmark._seeds_for_baseline(baseline, [7, 17])) == [7]


def test_development_run_loads_each_subject_once_and_counts_records(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[int, bool]] = []

    def loader(subject: int, *, use_cache: bool, allow_confirmation: bool):
        assert use_cache is True
        calls.append((subject, allow_confirmation))
        return _fake_data()

    def fit(baseline: str, data, *, seed: int, device: str):
        return _perfect_probabilities(data["test"]["labels"]), _fit_detail(baseline)

    monkeypatch.setattr(benchmark, "load_bnci2014_004_subject", loader)
    monkeypatch.setattr(benchmark, "_fit_predict", fit)
    monkeypatch.setattr(benchmark, "_source_manifest", lambda: {"synthetic.py": "a" * 64})
    monkeypatch.setattr(benchmark, "_git_state", lambda: {"synthetic": True})
    monkeypatch.setattr(benchmark, "_run_environment", lambda: {"synthetic": "stable"})
    payload = benchmark.run(
        mode="bnci004-dev",
        subjects=[1, 2],
        seeds=[7, 17],
        baselines=benchmark.BASELINES,
        device="cpu",
        output=tmp_path / "development.json",
    )
    assert calls == [(1, False), (2, False)]
    assert payload["completion"]["complete"] is True
    assert payload["completion"]["actual_records"] == 10


def _synthetic_development_payload() -> dict[str, Any]:
    device = "cpu"
    environment = {"synthetic_environment": "stable"}
    contract = benchmark._experiment_contract(
        mode="bnci004-dev",
        subjects=benchmark.DEVELOPMENT_SUBJECTS,
        seeds=[7],
        baselines=benchmark.BASELINES,
        device=device,
        frozen_manifest_sha256=None,
    )
    payload: dict[str, Any] = {
        "schema_version": benchmark.SCHEMA_VERSION,
        **contract,
        "contract_sha256": benchmark._json_sha256(contract),
        "confirmation_access": False,
        "source_sha256": {"synthetic.py": "a" * 64},
        "environment": environment,
        "environment_sha256": benchmark._json_sha256(environment),
        "records": [],
        "summary": {},
    }
    labels = np.tile(np.asarray([0, 1], dtype=np.int64), 160)
    probabilities = _perfect_probabilities(labels)
    session_ids = np.repeat(np.asarray(benchmark.TEST_SESSIONS), 160)
    for subject in benchmark.DEVELOPMENT_SUBJECTS:
        digest = f"{subject:064x}"
        data_sha256 = {
            split: {
                name: digest
                for name in ("broadband", "covariances", "labels", "session_ids")
            }
            for split in ("train", "validation", "test")
        }
        for baseline in benchmark.BASELINES:
            payload["records"].append(
                {
                    "baseline": baseline,
                    "subject": subject,
                    "seed": 7,
                    "deterministic": baseline in benchmark.DETERMINISTIC_BASELINES,
                    "protocol": benchmark.protocol_metadata(subject),
                    "data_sha256": data_sha256,
                    "data_shapes": {
                        "train": {
                            "broadband": [240, 3, 251],
                            "covariances": [240, 4, 3, 3],
                            "labels": [240],
                            "session_ids": [240],
                        },
                        "validation": {
                            "broadband": [160, 3, 251],
                            "covariances": [160, 4, 3, 3],
                            "labels": [160],
                            "session_ids": [160],
                        },
                        "test": {
                            "broadband": [320, 3, 251],
                            "covariances": [320, 4, 3, 3],
                            "labels": [320],
                            "session_ids": [320],
                        },
                    },
                    "metrics": benchmark._metrics(labels, probabilities),
                    "fit": _fit_detail(baseline),
                    "predictions": benchmark._prediction_trace(
                        labels, probabilities, session_ids
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
    receipt = tmp_path / "permanent-bnci004-receipt.json"
    monkeypatch.setattr(benchmark, "_source_manifest", lambda: source)
    monkeypatch.setattr(benchmark, "CONFIRMATION_RECEIPT_PATH", receipt)
    development = tmp_path / "development.json"
    development.write_text(
        json.dumps(_synthetic_development_payload(), allow_nan=False), encoding="utf-8"
    )
    output = tmp_path / "confirmation.json"
    frozen = tmp_path / "frozen.json"
    benchmark.write_frozen_manifest(
        development, frozen, confirmation_output=output, device="cpu"
    )
    return frozen, output, receipt


@pytest.mark.parametrize(
    ("subjects", "seeds", "baselines", "token", "message"),
    [
        ([5, 6, 7, 8], [7], list(benchmark.BASELINES), benchmark.CONFIRMATION_TOKEN, "exact ordered"),
        ([5, 6, 7, 8, 9], [7, 17], list(benchmark.BASELINES), benchmark.CONFIRMATION_TOKEN, "sole frozen seed"),
        ([5, 6, 7, 8, 9], [7], ["riemann"], benchmark.CONFIRMATION_TOKEN, "all fixed baselines"),
        ([5, 6, 7, 8, 9], [7], list(benchmark.BASELINES), None, "explicit confirmation token"),
    ],
)
def test_confirmation_contract_guards_precede_loader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    subjects: list[int],
    seeds: list[int],
    baselines: list[str],
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
            baselines=baselines,
            device="cpu",
            output=tmp_path / "forbidden.json",
            confirmation_token=token,
        )
    assert loaded == []


def test_frozen_manifest_pins_exact_output_receipt_and_development_hash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    frozen, output, receipt = _freeze_contract(monkeypatch, tmp_path)
    environment = {"synthetic_environment": "stable"}
    manifest, digest = benchmark._read_and_validate_frozen_manifest(
        frozen, current_environment=environment
    )
    assert len(digest) == 64
    assert manifest["confirmation_contract"] == {
        "study_id": benchmark.CONFIRMATION_STUDY_ID,
        "dataset": benchmark.DATASET,
        "protocol_id": benchmark.PROTOCOL_ID,
        "subjects": list(benchmark.CONFIRMATION_SUBJECTS),
        "seeds": [7],
        "baselines": list(benchmark.BASELINES),
        "device": "cpu",
        "environment": environment,
        "environment_sha256": benchmark._json_sha256(environment),
        "output_path": str(output.resolve()),
        "receipt_path": str(receipt.resolve()),
    }
    assert manifest["development_artifact"]["environment_sha256"] == (
        benchmark._json_sha256(environment)
    )
    development = Path(manifest["development_artifact"]["path"])
    payload = json.loads(development.read_text(encoding="utf-8"))
    payload["records"][0]["predictions"][0]["label"] = 1
    development.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hash no longer matches"):
        benchmark._read_and_validate_frozen_manifest(
            frozen, current_environment=environment
        )


def test_frozen_environment_digest_and_runtime_match_are_mandatory_before_access(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    frozen, output, receipt = _freeze_contract(monkeypatch, tmp_path)
    changed_environment = {"synthetic_environment": "changed"}
    with pytest.raises(ValueError, match="current execution environment differs"):
        benchmark._read_and_validate_frozen_manifest(
            frozen, current_environment=changed_environment
        )

    loaded: list[int] = []
    monkeypatch.setattr(benchmark, "_configure_torch_determinism", lambda: None)
    monkeypatch.setattr(
        benchmark, "_run_environment", lambda: dict(changed_environment)
    )
    monkeypatch.setattr(
        benchmark,
        "load_bnci2014_004_subject",
        lambda subject, **kwargs: loaded.append(subject),
    )
    with pytest.raises(ValueError, match="current execution environment differs"):
        benchmark.run(
            mode="bnci004-confirm",
            subjects=benchmark.CONFIRMATION_SUBJECTS,
            seeds=[7],
            baselines=benchmark.BASELINES,
            device="cpu",
            output=output,
            frozen_manifest=frozen,
            confirmation_token=benchmark.CONFIRMATION_TOKEN,
        )
    assert loaded == []
    assert not receipt.exists()
    assert not output.exists()


def test_frozen_manifest_rejects_noncanonical_environment_digest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    frozen, _, _ = _freeze_contract(monkeypatch, tmp_path)
    payload = json.loads(frozen.read_text(encoding="utf-8"))
    payload["confirmation_contract"]["environment_sha256"] = "0" * 64
    frozen.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="environment contract is invalid"):
        benchmark._read_and_validate_frozen_manifest(
            frozen,
            current_environment={"synthetic_environment": "stable"},
        )


@pytest.mark.parametrize(
    ("corrupt", "message"),
    [
        ("records", "wrong record count"),
        ("predictions", "predictions are incomplete"),
        ("session", "test sessions are invalid"),
        ("environment", "environment digest is invalid"),
        ("completion", "not complete"),
    ],
)
def test_development_artifact_validator_rejects_incomplete_or_corrupt_payload(
    corrupt: str, message: str
) -> None:
    payload = _synthetic_development_payload()
    if corrupt == "records":
        payload["records"].pop()
    elif corrupt == "predictions":
        payload["records"][0]["predictions"].pop()
    elif corrupt == "session":
        payload["records"][0]["predictions"][0]["session_id"] = "2train"
    elif corrupt == "environment":
        payload["environment_sha256"] = "0" * 64
    elif corrupt == "completion":
        payload["completion"]["complete"] = False
    with pytest.raises(ValueError, match=message):
        benchmark._validate_development_payload(payload)


def test_receipt_is_claimed_before_confirmation_loader_and_survives_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    frozen, output, receipt = _freeze_contract(monkeypatch, tmp_path)
    environment = {"synthetic_environment": "stable"}
    monkeypatch.setattr(benchmark, "_configure_torch_determinism", lambda: None)
    monkeypatch.setattr(benchmark, "_run_environment", lambda: environment)
    loaded: list[int] = []

    def interrupted_loader(subject: int, *, use_cache: bool, allow_confirmation: bool):
        assert use_cache is True
        assert allow_confirmation is True
        loaded.append(subject)
        observed = json.loads(receipt.read_text(encoding="utf-8"))
        assert observed["state"] == "running"
        assert observed["output_path"] == str(output.resolve())
        assert observed["environment"] == environment
        assert observed["environment_sha256"] == benchmark._json_sha256(environment)
        raise RuntimeError("synthetic interruption")

    monkeypatch.setattr(benchmark, "load_bnci2014_004_subject", interrupted_loader)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        benchmark.run(
            mode="bnci004-confirm",
            subjects=benchmark.CONFIRMATION_SUBJECTS,
            seeds=[7],
            baselines=benchmark.BASELINES,
            device="cpu",
            output=output,
            frozen_manifest=frozen,
            confirmation_token=benchmark.CONFIRMATION_TOKEN,
        )
    assert loaded == [5]
    assert json.loads(receipt.read_text(encoding="utf-8"))["state"] == "interrupted"
    with pytest.raises(FileExistsError, match="permanent BNCI004 baseline receipt"):
        benchmark.run(
            mode="bnci004-confirm",
            subjects=benchmark.CONFIRMATION_SUBJECTS,
            seeds=[7],
            baselines=benchmark.BASELINES,
            device="cpu",
            output=output,
            frozen_manifest=frozen,
            confirmation_token=benchmark.CONFIRMATION_TOKEN,
        )
    environment["synthetic_environment"] = "changed"
    with pytest.raises(ValueError, match="current execution environment differs"):
        benchmark.run(
            mode="bnci004-confirm",
            subjects=benchmark.CONFIRMATION_SUBJECTS,
            seeds=[7],
            baselines=benchmark.BASELINES,
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
        benchmark, "_run_environment", lambda: {"synthetic_environment": "stable"}
    )
    loaded: list[int] = []

    def loader(subject: int, *, use_cache: bool, allow_confirmation: bool):
        assert receipt.is_file()
        assert allow_confirmation is True
        loaded.append(subject)
        return _fake_data()

    def fit(baseline: str, data, *, seed: int, device: str):
        return _perfect_probabilities(data["test"]["labels"]), _fit_detail(baseline)

    monkeypatch.setattr(benchmark, "load_bnci2014_004_subject", loader)
    monkeypatch.setattr(benchmark, "_fit_predict", fit)
    payload = benchmark.run(
        mode="bnci004-confirm",
        subjects=benchmark.CONFIRMATION_SUBJECTS,
        seeds=[7],
        baselines=benchmark.BASELINES,
        device="cpu",
        output=output,
        frozen_manifest=frozen,
        confirmation_token=benchmark.CONFIRMATION_TOKEN,
    )
    assert loaded == [5, 6, 7, 8, 9]
    assert payload["completion"]["actual_records"] == 20
    completed_receipt = json.loads(receipt.read_text(encoding="utf-8"))
    assert completed_receipt["state"] == "complete"
    assert completed_receipt["output_sha256"] == benchmark._file_sha256(output)
    with pytest.raises(FileExistsError, match="already complete"):
        benchmark.run(
            mode="bnci004-confirm",
            subjects=benchmark.CONFIRMATION_SUBJECTS,
            seeds=[7],
            baselines=benchmark.BASELINES,
            device="cpu",
            output=output,
            resume=True,
            frozen_manifest=frozen,
            confirmation_token=benchmark.CONFIRMATION_TOKEN,
        )
    assert loaded == [5, 6, 7, 8, 9]


def test_prediction_trace_preserves_session_and_window_order() -> None:
    trace = benchmark._prediction_trace(
        np.asarray([0, 1]),
        np.asarray([[0.8, 0.2], [0.1, 0.9]]),
        np.asarray(["3test", "4test"]),
    )
    assert trace == [
        {
            "window_index": 0,
            "session_id": "3test",
            "label": 0,
            "probability_left": 0.8,
            "probability_right": 0.2,
        },
        {
            "window_index": 1,
            "session_id": "4test",
            "label": 1,
            "probability_left": 0.1,
            "probability_right": 0.9,
        },
    ]
