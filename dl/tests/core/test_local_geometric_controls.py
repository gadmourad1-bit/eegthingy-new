from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from benchmark import local_geometric_controls as controls


def _partition(count: int, *, seed: int) -> controls.FeaturePartition:
    rng = np.random.default_rng(seed)
    epochs = rng.normal(size=(count, 4, 3, 96)).astype(np.float32)
    covariance = np.tile(np.eye(3), (count, 4, 1, 1)).astype(np.float64)
    labels = np.arange(count, dtype=np.int64) % 2
    return controls.FeaturePartition(epochs, covariance, labels)


def _prepared(*, test_count: int = 6) -> controls.PreparedSubject:
    train = _partition(8, seed=1)
    validation = _partition(6, seed=2)
    source = _partition(14, seed=3)
    test = _partition(test_count, seed=4)
    return controls.PreparedSubject(
        train=train,
        validation=validation,
        source=source,
        test=test,
        train_rows=np.arange(8, dtype=np.int64),
        validation_rows=np.arange(8, 14, dtype=np.int64),
        source_rows=np.arange(14, dtype=np.int64),
        test_rows=np.arange(14, 14 + test_count, dtype=np.int64),
        selection_mean=np.zeros((1, 3, 1), dtype=np.float32),
        selection_std=np.ones((1, 3, 1), dtype=np.float32),
        refit_mean=np.zeros((1, 3, 1), dtype=np.float32),
        refit_std=np.ones((1, 3, 1), dtype=np.float32),
    )


def test_fixed_filter_bank_and_covariances_are_deterministic_and_spd() -> None:
    values = np.random.default_rng(7).normal(size=(5, 15, 256)).astype(np.float32)
    first = controls.fixed_filter_bank(values)
    second = controls.fixed_filter_bank(values)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (5, 4, 15, 256)
    covariance = controls.spd_covariances(first)
    assert covariance.shape == (5, 4, 15, 15)
    assert covariance.dtype == np.float64
    assert np.min(np.linalg.eigvalsh(covariance)) > 0.0


@pytest.mark.parametrize("model", controls.MODELS)
def test_selection_refit_never_fits_or_calibrates_on_test(
    model: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepared()
    fit_inputs: list[np.ndarray] = []
    predict_inputs: list[np.ndarray] = []

    class SpyEstimator:
        def fit(self, values: np.ndarray, labels: np.ndarray) -> "SpyEstimator":
            assert values is not controls._model_features(model, prepared.test)
            assert labels is not prepared.test.labels
            fit_inputs.append(values)
            return self

        def calibrate(self, values: np.ndarray) -> None:  # pragma: no cover
            raise AssertionError("R4 calibration is forbidden")

        def predict_proba(self, values: np.ndarray) -> np.ndarray:
            predict_inputs.append(values)
            positive = np.full(len(values), 0.55, dtype=np.float64)
            return np.column_stack((1.0 - positive, positive))

    monkeypatch.setattr(
        controls, "_new_estimator", lambda requested, parameters: SpyEstimator()
    )
    monkeypatch.setattr(
        controls, "_estimator_state_sha256", lambda estimator: "a" * 64
    )
    result = controls.select_and_refit(model, prepared)

    train_features = controls._model_features(model, prepared.train)
    source_features = controls._model_features(model, prepared.source)
    validation_features = controls._model_features(model, prepared.validation)
    test_features = controls._model_features(model, prepared.test)
    assert len(fit_inputs[:-1]) == len(controls.HYPERPARAMETER_GRID[model])
    assert all(values is train_features for values in fit_inputs[:-1])
    assert fit_inputs[-1] is source_features
    assert len(predict_inputs[:-1]) == len(controls.HYPERPARAMETER_GRID[model])
    assert all(values is validation_features for values in predict_inputs[:-1])
    assert predict_inputs[-1] is test_features
    assert result["test_probability"].shape == (len(prepared.test.labels), 2)


def test_seed_contract_replicates_all_sixty_r4_predictions_exactly() -> None:
    prepared = _prepared(test_count=60)
    prepared = controls.PreparedSubject(
        **{
            **prepared.__dict__,
            "train_rows": np.arange(120, dtype=np.int64),
            "validation_rows": np.arange(120, 180, dtype=np.int64),
            "source_rows": np.arange(180, dtype=np.int64),
            "test_rows": np.arange(180, 240, dtype=np.int64),
        }
    )
    probability = np.column_stack(
        (
            np.where(prepared.test.labels == 0, 0.8, 0.2),
            np.where(prepared.test.labels == 1, 0.8, 0.2),
        )
    )
    probability = probability[:, ::-1]
    result = {
        "selected_parameters": {"c": 1.0},
        "selection_trace": [{"parameters": {"c": 1.0}, "validation_nll": 0.5}],
        "selection_validation_nll": 0.5,
        "selection_validation_metrics": controls.classification_metrics(
            prepared.validation.labels,
            np.tile([0.5, 0.5], (len(prepared.validation.labels), 1)),
        ),
        "selection_state_sha256": "a" * 64,
        "refit_state_sha256": "b" * 64,
        "selection_fit_seconds": 1.0,
        "refit_fit_seconds": 2.0,
        "test_probability": probability,
    }
    data = {
        "sessions": np.asarray(["source"] * 180 + ["test"] * 60),
        "runs": np.asarray(["1"] * 60 + ["2"] * 60 + ["3"] * 60 + ["4"] * 60),
        "identity": {"array_sha256": "c" * 64},
    }
    records = controls._records_for_result(
        model="riemann",
        subject=1,
        result=result,
        prepared=prepared,
        data=data,
    )
    assert [record["seed"] for record in records] == list(controls.SEEDS)
    assert records[0]["seed_role"] == "canonical_deterministic_fit"
    assert all(len(record["test"]["probabilities"]) == 60 for record in records)
    for record in records[1:]:
        assert record["seed_role"] == "exact_deterministic_replication"
        assert record["fit"] == records[0]["fit"]
        assert record["test"] == records[0]["test"]


def test_strict_load_rejects_duplicate_and_nonfinite_json(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a": 1, "a": 2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON"):
        controls.strict_load(duplicate)
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"a": NaN}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite JSON"):
        controls.strict_load(nonfinite)


def test_final_validation_can_require_complete_cartesian_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {"records": [], "summary": controls._summary([])}
    monkeypatch.setattr(controls, "load_subject_cache", lambda *args, **kwargs: {})
    with pytest.raises(ValueError, match="artifact is incomplete"):
        controls.validate_payload(
            payload,
            contract={},
            cache_root=Path("unused"),
            require_complete=True,
        )
