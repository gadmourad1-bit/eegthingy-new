from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import deepnet.bnci_baseline_benchmark as benchmark


def _fake_data() -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {}
    for split, n, run in (("train", 6, "0"), ("validation", 2, "5"), ("test", 4, "0")):
        labels = np.arange(n, dtype=np.int64) % 2
        split_value = {"train": 1, "validation": 2, "test": 3}[split]
        result[split] = {
            "broadband": np.full(
                (n, 15, 11), split_value, dtype=np.float32
            ),
            "covariances": np.repeat(
                np.eye(15, dtype=np.float32)[None, None], n * 4, axis=0
            ).reshape(n, 4, 15, 15),
            "labels": labels,
            "run_ids": np.asarray([run] * n, dtype=np.str_),
        }
    return result


def test_development_rejects_confirmation_subject_before_loading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loaded = False

    def forbidden_loader(subject: int):
        nonlocal loaded
        loaded = True
        raise AssertionError(subject)

    monkeypatch.setattr(benchmark, "load_bnci2014_subject", forbidden_loader)
    with pytest.raises(ValueError, match="locked partition"):
        benchmark.run(
            mode="bnci-dev",
            subjects=[5],
            seeds=[7],
            baselines=["riemann"],
            device="cpu",
            output=tmp_path / "forbidden.json",
        )
    assert not loaded


def test_riemannian_baseline_never_fits_or_calibrates_on_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from deepnet import baselines

    data = _fake_data()

    class SpyRiemann:
        def __init__(self, *, c: float, max_iter: int) -> None:
            assert c == 1.0
            assert max_iter == 3000

        def fit(self, covariances: np.ndarray, labels: np.ndarray):
            assert covariances is data["train"]["covariances"]
            assert labels is data["train"]["labels"]
            return self

        def calibrate(self, covariances: np.ndarray):  # pragma: no cover - must stay unused
            raise AssertionError("test calibration is forbidden")

        def predict_proba(self, covariances: np.ndarray) -> np.ndarray:
            assert covariances is data["test"]["covariances"]
            return np.tile([0.4, 0.6], (len(covariances), 1))

    monkeypatch.setattr(baselines, "RiemannianTangentLogistic", SpyRiemann)
    probabilities, detail = benchmark._fit_predict(
        "riemann", data, seed=7, device="cpu"
    )
    assert probabilities.shape == (4, 2)
    assert detail["unlabeled_test_calibration"] is False
    assert detail["validation_role"] == "unused_fixed_estimator"


def test_shallow_uses_train_scaling_and_validation_only_for_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from deepnet import dnn_baselines

    data = _fake_data()

    class SpyTorchEEG:
        def __init__(self, arch: str, **kwargs) -> None:
            assert arch == "shallow"
            assert kwargs["lr_swap_prob"] == 0.5
            assert tuple(kwargs["channels"]) == benchmark.BNCI_CHANNELS
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
            return np.tile([0.55, 0.45], (len(x_test), 1))

    monkeypatch.setattr(dnn_baselines, "TorchEEGClassifier", SpyTorchEEG)
    probabilities, detail = benchmark._fit_predict(
        "shallow_swap", data, seed=7, device="cpu"
    )
    assert probabilities.shape == (4, 2)
    assert detail["validation_role"] == "early_stopping_only"
    assert detail["source_only_preprocessing"] is True


def test_prediction_trace_retains_window_order_and_probabilities() -> None:
    trace = benchmark._prediction_trace(
        np.asarray([0, 1]),
        np.asarray([[0.8, 0.2], [0.1, 0.9]]),
        np.asarray(["0", "3"]),
    )
    assert trace == [
        {
            "window_index": 0,
            "run_id": "0",
            "label": 0,
            "probability_left": 0.8,
            "probability_right": 0.2,
        },
        {
            "window_index": 1,
            "run_id": "3",
            "label": 1,
            "probability_left": 0.1,
            "probability_right": 0.9,
        },
    ]


def test_deterministic_baselines_use_one_seed() -> None:
    assert list(benchmark._seeds_for_baseline("riemann", [7, 17, 27])) == [7]
    assert list(benchmark._seeds_for_baseline("tangent_anchor", [7, 17, 27])) == [7]
    assert list(benchmark._seeds_for_baseline("shallow_swap", [7, 17, 27])) == [7, 17, 27]


def _synthetic_development_payload() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": benchmark.SCHEMA_VERSION,
        "dataset": "BNCI2014-001",
        "protocol_id": benchmark.PROTOCOL_ID,
        "mode": "bnci-dev",
        "confirmation_access": False,
        "subjects": list(benchmark.DEVELOPMENT_SUBJECTS),
        "seeds": [7],
        "baselines": list(benchmark.BASELINES),
        "device": "cpu",
        "baseline_contracts": benchmark.BASELINE_CONTRACTS,
        "source_sha256": {"synthetic.py": "a" * 64},
        "records": [],
        "summary": {},
    }
    labels = np.tile(np.asarray([0, 1], dtype=np.int64), 72)
    probabilities = np.where(
        labels[:, None] == 0,
        np.asarray([0.8, 0.2]),
        np.asarray([0.2, 0.8]),
    )
    run_ids = np.repeat(np.arange(6).astype(str), 24)
    for subject in benchmark.DEVELOPMENT_SUBJECTS:
        digest = f"{subject:064x}"
        data_sha256 = {
            split: {name: digest for name in ("broadband", "covariances", "labels", "run_ids")}
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
                    "metrics": benchmark._metrics(labels, probabilities),
                    "fit": {
                        "parameter_count": {
                            "shallow_swap": 26_002,
                            "riemann": 0,
                            "tangent_anchor": 481,
                        }[baseline],
                        "fit_seconds": 0.1,
                        "wall_seconds": 0.2,
                        "n_train": 120,
                        "n_validation": 24,
                        "n_test": 144,
                        "train_run_ids": ["0", "1", "2", "3", "4"],
                        "validation_run_ids": ["5"],
                        "test_run_ids": ["0", "1", "2", "3", "4", "5"],
                        "source_only_preprocessing": True,
                        "unlabeled_test_calibration": False,
                        "validation_role": benchmark.BASELINE_CONTRACTS[baseline][
                            "validation_role"
                        ],
                    },
                    "predictions": benchmark._prediction_trace(
                        labels, probabilities, run_ids
                    ),
                }
            )
    payload["summary"] = benchmark._summarize(payload)
    return payload


def _freeze_confirmation_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[Path, Path, Path]:
    source = {"deepnet/synthetic.py": "b" * 64}
    receipt = tmp_path / "permanent-study-receipt.json"
    monkeypatch.setattr(benchmark, "_source_manifest", lambda: source)
    monkeypatch.setattr(benchmark, "CONFIRMATION_RECEIPT_PATH", receipt)
    development = tmp_path / "development.json"
    development.write_text(
        json.dumps(_synthetic_development_payload(), allow_nan=False),
        encoding="utf-8",
    )
    output = tmp_path / "confirmation.json"
    frozen = tmp_path / "frozen.json"
    benchmark.write_frozen_manifest(
        development,
        frozen,
        confirmation_output=output,
        device="cpu",
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

    def forbidden_loader(subject: int):
        loaded.append(subject)
        raise AssertionError("confirmation loader must not be reached")

    monkeypatch.setattr(benchmark, "load_bnci2014_subject", forbidden_loader)
    with pytest.raises(ValueError, match=message):
        benchmark.run(
            mode="bnci-confirm",
            subjects=subjects,
            seeds=seeds,
            baselines=baselines,
            device="cpu",
            output=tmp_path / "forbidden.json",
            confirmation_token=token,
        )
    assert loaded == []


def test_confirmation_requires_frozen_manifest_before_loader(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loaded: list[int] = []
    monkeypatch.setattr(
        benchmark,
        "load_bnci2014_subject",
        lambda subject: loaded.append(subject),
    )
    with pytest.raises(ValueError, match="frozen-manifest"):
        benchmark.run(
            mode="bnci-confirm",
            subjects=benchmark.CONFIRMATION_SUBJECTS,
            seeds=[7],
            baselines=benchmark.BASELINES,
            device="cpu",
            output=tmp_path / "forbidden.json",
            confirmation_token=benchmark.CONFIRMATION_TOKEN,
        )
    assert loaded == []


def test_frozen_manifest_pins_development_contract_and_exact_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    frozen, output, receipt = _freeze_confirmation_contract(monkeypatch, tmp_path)
    manifest, manifest_sha256 = benchmark._read_and_validate_frozen_manifest(frozen)
    assert len(manifest_sha256) == 64
    assert manifest["confirmation_contract"] == {
        "study_id": benchmark.CONFIRMATION_STUDY_ID,
        "dataset": "BNCI2014-001",
        "protocol_id": benchmark.PROTOCOL_ID,
        "subjects": list(benchmark.CONFIRMATION_SUBJECTS),
        "seeds": [7],
        "baselines": list(benchmark.BASELINES),
        "device": "cpu",
        "output_path": str(output.resolve()),
        "receipt_path": str(receipt.resolve()),
    }
    development = Path(manifest["development_artifact"]["path"])
    payload = json.loads(development.read_text(encoding="utf-8"))
    payload["records"][0]["predictions"][0]["label"] = 1
    development.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hash no longer matches"):
        benchmark._read_and_validate_frozen_manifest(frozen)


def test_receipt_is_claimed_before_loader_and_blocks_changed_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    frozen, output, receipt = _freeze_confirmation_contract(monkeypatch, tmp_path)
    environment = {"synthetic_environment": "one"}
    monkeypatch.setattr(benchmark, "_configure_torch_determinism", lambda: None)
    monkeypatch.setattr(benchmark, "_run_environment", lambda: environment)
    loaded: list[int] = []

    def interrupted_loader(subject: int):
        loaded.append(subject)
        observed = json.loads(receipt.read_text(encoding="utf-8"))
        assert observed["state"] == "running"
        assert observed["output_path"] == str(output.resolve())
        raise RuntimeError("synthetic interruption")

    monkeypatch.setattr(benchmark, "load_bnci2014_subject", interrupted_loader)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        benchmark.run(
            mode="bnci-confirm",
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

    with pytest.raises(FileExistsError, match="permanent baseline confirmation receipt"):
        benchmark.run(
            mode="bnci-confirm",
            subjects=benchmark.CONFIRMATION_SUBJECTS,
            seeds=[7],
            baselines=benchmark.BASELINES,
            device="cpu",
            output=output,
            frozen_manifest=frozen,
            confirmation_token=benchmark.CONFIRMATION_TOKEN,
        )
    assert loaded == [5]

    environment["synthetic_environment"] = "changed"
    with pytest.raises(ValueError, match="contract/environment"):
        benchmark.run(
            mode="bnci-confirm",
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


def test_completed_receipt_prevents_any_reevaluation_with_synthetic_data(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    frozen, output, receipt = _freeze_confirmation_contract(monkeypatch, tmp_path)
    monkeypatch.setattr(benchmark, "_configure_torch_determinism", lambda: None)
    monkeypatch.setattr(
        benchmark, "_run_environment", lambda: {"synthetic_environment": "stable"}
    )
    loaded: list[int] = []

    def synthetic_loader(subject: int):
        assert receipt.is_file(), "receipt must exist before the mocked loader"
        loaded.append(subject)
        return _fake_data()

    def synthetic_fit(baseline: str, data, *, seed: int, device: str):
        labels = data["test"]["labels"]
        probabilities = np.where(
            labels[:, None] == 0,
            np.asarray([0.75, 0.25]),
            np.asarray([0.25, 0.75]),
        )
        return probabilities, {
            "parameter_count": 0,
            "fit_seconds": 0.01,
            "wall_seconds": 0.01,
            "n_train": len(data["train"]["labels"]),
            "n_validation": len(data["validation"]["labels"]),
            "n_test": len(data["test"]["labels"]),
            "train_run_ids": ["0"],
            "validation_run_ids": ["5"],
            "test_run_ids": ["0"],
            "source_only_preprocessing": True,
            "unlabeled_test_calibration": False,
            "validation_role": benchmark.BASELINE_CONTRACTS[baseline][
                "validation_role"
            ],
        }

    monkeypatch.setattr(benchmark, "load_bnci2014_subject", synthetic_loader)
    monkeypatch.setattr(benchmark, "_fit_predict", synthetic_fit)
    payload = benchmark.run(
        mode="bnci-confirm",
        subjects=benchmark.CONFIRMATION_SUBJECTS,
        seeds=[7],
        baselines=benchmark.BASELINES,
        device="cpu",
        output=output,
        frozen_manifest=frozen,
        confirmation_token=benchmark.CONFIRMATION_TOKEN,
    )
    assert loaded == [5, 6, 7, 8, 9]
    assert payload["completion"]["complete"] is True
    assert payload["completion"]["actual_records"] == 15
    completed_receipt = json.loads(receipt.read_text(encoding="utf-8"))
    assert completed_receipt["state"] == "complete"
    assert completed_receipt["output_sha256"] == benchmark._file_sha256(output)

    with pytest.raises(FileExistsError, match="already complete"):
        benchmark.run(
            mode="bnci-confirm",
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
