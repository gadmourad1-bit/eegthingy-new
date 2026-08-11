from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

import deepnet.loso_baselines as loso_baselines
from deepnet.config import VALID_SUBJECTS
from deepnet.loso import nested_loso_fold
from deepnet.loso_baselines import (
    BaselineLOSOConfig,
    run_loso_baselines,
    score_baseline_fold,
)
from deepnet.report import build_report, metrics_from_predictions


def _cohort_data() -> SimpleNamespace:
    labels: list[int] = []
    subjects: list[int] = []
    runs: list[int] = []
    sessions: list[str] = []
    event_samples: list[int] = []
    values: list[float] = []
    for subject in VALID_SUBJECTS:
        for run in range(1, 5):
            session = f"S{subject:02d}-R{run:02d}"
            for position in range(23):
                task_number = position // 2
                label = task_number % 2 if position % 2 == 0 else -1
                # Each calibration prefix contains ten negative/ten positive
                # values. The later task pair is confidently and correctly
                # separated, while its intervening rest is confidently forced
                # into a left/right class (there is deliberately no intent gate).
                value = -4.0 if task_number % 2 == 0 else 4.0
                labels.append(label)
                subjects.append(subject)
                runs.append(run)
                sessions.append(session)
                event_samples.append(100_000 * subject + 1_000 * run + position)
                values.append(value)
    feature = np.asarray(values, dtype=np.float64).reshape(-1, 1, 1, 1)
    n_rows = len(labels)
    return SimpleNamespace(
        labels=np.asarray(labels, dtype=np.int64),
        subject_ids=np.asarray(subjects, dtype=np.int64),
        run_ids=np.asarray(runs, dtype=np.int64),
        session_ids=np.asarray(sessions, dtype=np.str_),
        event_samples=np.asarray(event_samples, dtype=np.int64),
        event_onsets=np.asarray(event_samples, dtype=np.float64) / 250.0,
        trial_ids=np.arange(n_rows, dtype=np.int64),
        annotations=np.asarray(["synthetic"] * n_rows, dtype=np.str_),
        covariances=feature,
        epochs=feature,
    )


class _RecordingEstimator:
    def __init__(self) -> None:
        self.classes_ = np.asarray([0, 1])
        self.fit_groups: np.ndarray | None = None
        self.fit_labels: np.ndarray | None = None
        self.calibration_sizes: list[int] = []

    def fit(
        self, features: np.ndarray, labels: np.ndarray, groups: np.ndarray | None = None
    ) -> _RecordingEstimator:
        del features
        self.fit_groups = np.asarray(groups).copy()
        self.fit_labels = np.asarray(labels).copy()
        return self

    def calibrate(self, features: np.ndarray) -> _RecordingEstimator:
        self.calibration_sizes.append(len(features))
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        positive = np.where(np.asarray(features).reshape(len(features), -1)[:, 0] > 0, 0.95, 0.05)
        return np.column_stack((1.0 - positive, positive))


def _one_candidate_config() -> BaselineLOSOConfig:
    return BaselineLOSOConfig(
        temperature_grid=(1.0,),
        target_median_blends=(0.0,),
    )


def test_strict_baseline_fold_protects_inner_and_outer_subjects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _cohort_data()
    fold = nested_loso_fold(data, 4)
    estimator = _RecordingEstimator()
    monkeypatch.setattr(loso_baselines, "_new_estimator", lambda name, config: estimator)

    row = score_baseline_fold("riemann", data, fold, _one_candidate_config())

    assert estimator.fit_groups is not None
    fit_sessions = set(estimator.fit_groups.tolist())
    assert len(fit_sessions) == 6 * 4
    assert not any(session.startswith("S04-") for session in fit_sessions)
    assert not any(
        session.startswith(f"S{fold.validation_subject:02d}-")
        for session in fit_sessions
    )
    assert set(row["source_subjects"]) == set(fold.selection_subjects)
    assert row["selection"]["inner_validation_used_for_estimator_fit"] is False
    assert row["selection"]["temperature"] == 1.0
    assert row["selection"]["target_median_blend"] == 0.0
    assert len(row["selection"]["candidate_trace"]) == 1

    # Four validation sessions select the fixed candidate, then four untouched
    # outer sessions are calibrated and scored. Every prefix contains exactly
    # ten task plus ten rest windows and target state is reset each time.
    assert estimator.calibration_sizes == [20] * 8
    assert len(row["sessions"]) == 4
    assert all(session["calibration_task_events"] == 10 for session in row["sessions"])
    assert all(session["calibration_windows"] == 20 for session in row["sessions"])
    assert row["deployment"]["scored_task_windows"] == 8
    assert row["deployment"]["target_alignment_reset_per_session"] is True
    assert len(row["predictions"]) == 12
    assert all(prediction["intent_gate_applied"] is False for prediction in row["predictions"])


def test_no_intent_gate_counts_confident_rest_as_false_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _cohort_data()
    fold = nested_loso_fold(data, VALID_SUBJECTS[0])
    estimator = _RecordingEstimator()
    monkeypatch.setattr(loso_baselines, "_new_estimator", lambda name, config: estimator)

    row = score_baseline_fold("riemann", data, fold, _one_candidate_config())

    assert row["metrics"]["balanced_accuracy"] == 1.0
    assert row["metrics"]["coverage"] == 1.0
    assert row["metrics"]["rest_false_commit_rate"] == 1.0
    assert row["deployment"]["intent_gate"] is False
    assert row["deployment"]["commit_rule"] == "max_left_right_probability_at_threshold"
    assert all(session["intent_gate"] is False for session in row["sessions"])


def test_baseline_artifact_is_atomic_resumable_and_report_compatible(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _cohort_data()
    calls: list[tuple[str, int]] = []

    def fake_score(name, loaded, fold, config):
        del loaded, config
        calls.append((name, fold.outer_subject))
        predictions = [
            {
                "label": 0,
                "probability_left": 0.9,
                "probability_right": 0.1,
                "committed": True,
            },
            {
                "label": 1,
                "probability_left": 0.2,
                "probability_right": 0.8,
                "committed": False,
            },
            {
                "label": -1,
                "probability_left": 0.9,
                "probability_right": 0.1,
                "committed": True,
            },
        ]
        metrics = metrics_from_predictions(predictions, commit_confidence=0.85)
        return {
            "model": name,
            "subject": fold.outer_subject,
            "outer_subject": fold.outer_subject,
            "seed": 0,
            "metrics": metrics,
            "predictions": predictions,
            "sessions": [],
            "selection": None,
            "deployment": {"intent_gate": False},
        }

    monkeypatch.setattr(loso_baselines, "load_sessions", lambda keys, config: data)
    monkeypatch.setattr(
        loso_baselines,
        "dataset_contract",
        lambda loaded, config: {"contract_sha256": "same-data"},
    )
    monkeypatch.setattr(loso_baselines, "score_baseline_fold", fake_score)
    monkeypatch.setattr(loso_baselines, "_environment", lambda: {"test": True})
    output = tmp_path / "baselines.json"
    config = _one_candidate_config()

    first = run_loso_baselines(
        subjects=(1,),
        models=("riemann",),
        config=config,
        output=output,
    )
    second = run_loso_baselines(
        subjects=(1,),
        models=("riemann",),
        config=config,
        output=output,
    )

    assert calls == [("riemann", 1)]
    assert len(first["folds"]) == len(second["folds"]) == 1
    on_disk = json.loads(output.read_text(encoding="utf-8"))
    assert on_disk["folds"] == second["folds"]
    assert on_disk["schema_version"] == 2
    assert on_disk["data_contract"] == {"contract_sha256": "same-data"}
    assert not list(tmp_path.glob(".baselines-*.json"))
    enriched, markdown = build_report(on_disk)
    assert enriched["summary"]["riemann"]["participants"] == 1
    assert markdown.startswith("# Initial nested leave-one-subject-out benchmark")

    monkeypatch.setattr(
        loso_baselines,
        "dataset_contract",
        lambda loaded, config: {"contract_sha256": "changed-data"},
    )
    with pytest.raises(ValueError, match="preprocessing/data contract"):
        run_loso_baselines(
            subjects=(1,),
            models=("riemann",),
            config=config,
            output=output,
        )


def test_strict_config_rejects_a_different_prefix_length() -> None:
    with pytest.raises(ValueError, match="exactly 10"):
        BaselineLOSOConfig(calibration_task_events=9)
