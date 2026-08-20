from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from benchmark import combined_local_report as combined


def _write(path: Path, value: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _session_run(subject: int) -> tuple[str, str]:
    run = 8 if subject == 10 else 4
    return f"S{subject:02d}-R{run:02d}", str(run)


def _prediction(subject: int, *, correct_per_class: int) -> dict[str, Any]:
    if not 0 <= correct_per_class <= 30:
        raise ValueError(correct_per_class)
    labels = np.repeat(np.asarray((0, 1), dtype=np.int64), 30)
    predicted = labels.copy()
    for label in (0, 1):
        indices = np.flatnonzero(labels == label)
        predicted[indices[correct_per_class:]] = 1 - label
    probability = np.full((60, 2), 0.1, dtype=np.float64)
    probability[np.arange(60), predicted] = 0.9
    session, run = _session_run(subject)
    score = correct_per_class / 30.0
    return {
        "rows": list(range(180, 240)),
        "labels": labels.tolist(),
        "probabilities": probability.tolist(),
        "sessions": [session] * 60,
        "runs": [run] * 60,
        "score": score,
    }


def _matrix(base: float) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    subject_offsets = np.linspace(-0.007, 0.007, len(combined.SUBJECTS))
    seed_offsets = np.linspace(-0.004, 0.004, len(combined.SEEDS))
    for subject, subject_offset in zip(
        combined.SUBJECTS, subject_offsets, strict=True
    ):
        result[str(subject)] = {
            str(seed): float(base + subject_offset + seed_offset)
            for seed, seed_offset in zip(combined.SEEDS, seed_offsets, strict=True)
        }
    return result


def _summary_values(matrix: dict[str, dict[str, float]]) -> tuple[dict[str, float], np.ndarray]:
    subjects = {
        str(subject): float(np.mean(list(matrix[str(subject)].values())))
        for subject in combined.SUBJECTS
    }
    return subjects, np.asarray(list(subjects.values()), dtype=np.float64)


def _neural_payload() -> dict[str, Any]:
    model_rows: dict[str, dict[str, Any]] = {}
    for index, model in enumerate(combined.NEURAL_CONFIGURATIONS):
        matrix = _matrix(0.55 + 0.006 * index)
        subjects, values = _summary_values(matrix)
        mean = float(values.mean())
        model_rows[model] = {
            "model": model,
            "balanced_accuracy_mean": mean,
            "balanced_accuracy_standard_deviation": float(values.std(ddof=1)),
            "bootstrap_95_ci": [max(0.0, mean - 0.02), min(1.0, mean + 0.02)],
            "minimum_subject_balanced_accuracy": float(values.min()),
            "median_subject_balanced_accuracy": float(np.median(values)),
            "subject_balanced_accuracy": subjects,
            "subject_seed_balanced_accuracy": matrix,
            "parameter_count": 1_000 + index,
            "total_fit_seconds": 10.0,
            "artifact": f"/unavailable/artifacts/{model}.json",
            "artifact_sha256": _sha(f"artifact:{model}"),
        }
    order = sorted(
        model_rows,
        key=lambda model: (-model_rows[model]["balanced_accuracy_mean"], model),
    )
    ranking = []
    for rank, model in enumerate(order, start=1):
        row = dict(model_rows[model])
        row["rank"] = rank
        ranking.append(row)
    return {
        "schema": combined.NEURAL_SCHEMA,
        "evidence_scope": "development_only_not_confirmation",
        "confirmation_evidence": False,
        "primary_metric": "mean_subject_balanced_accuracy_after_averaging_seeds_within_subject",
        "winner": order[0],
        "co_winners": [order[0]],
        "runner_up": order[1],
        "winner_balanced_accuracy": model_rows[order[0]]["balanced_accuracy_mean"],
        "ranking": ranking,
        "reference_analysis": {
            "evidence_scope": "development_only_not_confirmation",
            "confirmation_evidence": False,
        },
        "plan_sha256": _sha("plan"),
    }


def _geometric_payload() -> dict[str, Any]:
    correct = {"riemann": 27, "tangent_anchor": 28, "ea_fbcsp": 26}
    records: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for model in combined.GEOMETRIC_MODELS:
        subject_scores: dict[str, float] = {}
        for subject in combined.SUBJECTS:
            prediction = _prediction(subject, correct_per_class=correct[model])
            subject_scores[str(subject)] = prediction["score"]
            test = {
                "metrics": {"balanced_accuracy": prediction["score"]},
                "rows": prediction["rows"],
                "labels": prediction["labels"],
                "probabilities": prediction["probabilities"],
                "sessions": prediction["sessions"],
                "runs": prediction["runs"],
            }
            for seed in combined.SEEDS:
                records.append(
                    {
                        "dataset": combined.DATASET,
                        "model": model,
                        "subject": subject,
                        "fold": 0,
                        "seed": seed,
                        "deterministic_estimator": True,
                        "seed_role": (
                            "canonical_deterministic_fit"
                            if seed == combined.SEEDS[0]
                            else "exact_deterministic_replication"
                        ),
                        "split": {
                            "train_count": 120,
                            "validation_count": 60,
                            "refit_source_count": 180,
                            "test_count": 60,
                        },
                        "test": copy.deepcopy(test),
                    }
                )
        values = np.asarray(list(subject_scores.values()))
        summary[model] = {
            "balanced_accuracy_mean": float(values.mean()),
            "balanced_accuracy_std": float(values.std(ddof=1)),
            "n_subjects": 8,
            "n_records": 40,
            "subject_balanced_accuracy": subject_scores,
        }
    return {
        "schema": combined.GEOMETRIC_SCHEMA,
        "dataset": combined.DATASET,
        "evidence_scope": "formal_development_not_confirmation",
        "confirmation_evidence": False,
        "subjects": list(combined.SUBJECTS),
        "folds": [0],
        "models": list(combined.GEOMETRIC_MODELS),
        "seeds": list(combined.SEEDS),
        "canonical_fit_seed": combined.SEEDS[0],
        "split_protocol": "R1-2 selection fit; R3 selection; R1-3 refit; R4 prediction-only",
        "test_calibration": False,
        "records": records,
        "summary": summary,
    }


def _deepnet_payload(model: str, *, correct_per_class: int) -> dict[str, Any]:
    contract = {
        "schema_version": combined.DEEPNET_SCHEMA,
        "dataset": combined.DATASET,
        "model": model,
        "subjects": list(combined.SUBJECTS),
        "seeds": list(combined.SEEDS),
        "formal_complete_dimensions": True,
        "evidence_scope": "post_selection_local_development_all_four_recordings_previously_opened",
        "outer_protocol": {
            "phase_a": "recordings 1-2 fit; recording 3 selects duration and route only",
            "phase_b": "fresh reset; source preprocessing refit on recordings 1-3; fixed selected duration",
            "test": "recording 4 prediction only after phase B",
            "fold": 0,
        },
        "preprocessing": {"schema": "synthetic-v2"},
        "covariance_view": {"bands_hz": [[8, 12], [11, 15], [14, 20], [20, 30]]},
    }
    records: list[dict[str, Any]] = []
    subject_scores: dict[str, list[float]] = {
        str(subject): [] for subject in combined.SUBJECTS
    }
    for subject in combined.SUBJECTS:
        prediction = _prediction(subject, correct_per_class=correct_per_class)
        trace = [
            {
                "row": row,
                "label": label,
                "probability_left": probability[0],
                "probability_right": probability[1],
                "session": session,
                "run": run,
            }
            for row, label, probability, session, run in zip(
                prediction["rows"],
                prediction["labels"],
                prediction["probabilities"],
                prediction["sessions"],
                prediction["runs"],
                strict=True,
            )
        ]
        for seed in combined.SEEDS:
            subject_scores[str(subject)].append(prediction["score"])
            records.append(
                {
                    "model": model,
                    "subject": subject,
                    "seed": seed,
                    "split": {
                        "phase_a_train": {"count": 120},
                        "phase_a_validation": {"count": 60},
                        "phase_b_source": {"count": 180},
                        "prediction_only_test": {"count": 60},
                    },
                    "phase_a": {"selected_epoch_count": 2},
                    "phase_b": {
                        "reset_verified": True,
                        "epoch_count": 2,
                        "parameter_count": 2_000 + combined.DEEPNET_MODELS.index(model),
                    },
                    "prediction_only_test": {
                        "metrics": {"balanced_accuracy": prediction["score"]},
                        "predictions": copy.deepcopy(trace),
                    },
                    "cache_identity_sha256": _sha(f"cache:S{subject}"),
                }
            )
    averaged = {
        subject: float(np.mean(scores)) for subject, scores in subject_scores.items()
    }
    values = np.asarray(list(averaged.values()))
    return {
        "schema_version": combined.DEEPNET_SCHEMA,
        "status": "complete",
        "contract": contract,
        "contract_sha256": combined._canonical_sha256(contract),
        "records": records,
        "summary": {
            "balanced_accuracy_mean": float(values.mean()),
            "balanced_accuracy_std": float(values.std(ddof=1)),
            "subject_balanced_accuracy": averaged,
            "n_participants": 8,
            "n_records": 40,
            "aggregation": "mean seeds within participant, then equal-weight participant mean",
        },
        "completion": {
            "expected_records": 40,
            "actual_records": 40,
            "complete": True,
        },
    }


def _inputs(tmp_path: Path) -> tuple[Path, Path, dict[str, Path]]:
    neural = _write(tmp_path / "neural.json", _neural_payload())
    geometric = _write(tmp_path / "geometric.json", _geometric_payload())
    correct = {"cameo": 29, "hemiparity": 25, "parity_fuse": 24, "orbit_v3": 23}
    deepnet = {
        model: _write(
            tmp_path / f"{model}.json",
            _deepnet_payload(model, correct_per_class=correct[model]),
        )
        for model in combined.DEEPNET_MODELS
    }
    return neural, geometric, deepnet


def test_combines_all_fifty_conditions_and_recomputes_raw_winner(
    tmp_path: Path,
) -> None:
    neural, geometric, deepnet = _inputs(tmp_path)
    report = combined.build_report(
        neural_ranking=neural,
        geometric_controls=geometric,
        deepnet_paths=deepnet,
        bootstrap_repetitions=500,
        random_seed=20260721,
    )
    assert report["condition_counts"]["total"] == 50
    assert len(report["ranking"]) == 50
    assert len(report["tables"][combined.CATEGORY_NEURAL]) == 43
    assert len(report["tables"][combined.CATEGORY_GEOMETRIC]) == 3
    assert len(report["tables"][combined.CATEGORY_PROCEDURE]) == 4
    assert report["winner"] == "cameo"
    assert report["runner_up"] == "tangent_anchor"
    assert report["winner_balanced_accuracy"] == pytest.approx(29 / 30)
    assert report["winner_vs_runner_up"]["wins_ties_losses"] == [8, 0, 0]
    assert report["winner_vs_runner_up"][
        "exact_two_sided_sign_flip_p_descriptive"
    ] == pytest.approx(2 / 256)
    # 120 deterministic control records + 4*40 procedure records. The synthetic
    # neural final ranking intentionally has no raw artifact directory.
    assert report["audit"]["raw_prediction_records_validated"] == 280
    assert report["audit"]["neural_trial_level_limit"] is not None
    markdown = combined.render_markdown(report)
    assert "Numerical winner" in markdown
    assert "not 43 independent architecture families" in markdown
    assert "disabled or paralyzed users" in markdown

    output_json = tmp_path / "combined.json"
    output_markdown = tmp_path / "combined.md"
    combined.write_report(
        report, output_json=output_json, output_markdown=output_markdown
    )
    assert json.loads(output_json.read_text(encoding="utf-8"))["winner"] == "cameo"
    assert output_markdown.read_text(encoding="utf-8").startswith(
        "# Combined local Exp4 benchmark audit"
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        combined.write_report(
            report, output_json=output_json, output_markdown=tmp_path / "other.md"
        )


def test_rejects_incomplete_geometric_cartesian_grid(tmp_path: Path) -> None:
    neural, geometric, deepnet = _inputs(tmp_path)
    payload = json.loads(geometric.read_text(encoding="utf-8"))
    payload["records"].pop()
    _write(geometric, payload)
    with pytest.raises(ValueError, match="exactly 120"):
        combined.build_report(
            neural_ranking=neural,
            geometric_controls=geometric,
            deepnet_paths=deepnet,
            bootstrap_repetitions=10,
        )


def test_rejects_stale_deepnet_prediction_metric(tmp_path: Path) -> None:
    neural, geometric, deepnet = _inputs(tmp_path)
    payload = json.loads(deepnet["cameo"].read_text(encoding="utf-8"))
    payload["records"][0]["prediction_only_test"]["metrics"][
        "balanced_accuracy"
    ] = 0.5
    _write(deepnet["cameo"], payload)
    with pytest.raises(ValueError, match="stale balanced accuracy"):
        combined.build_report(
            neural_ranking=neural,
            geometric_controls=geometric,
            deepnet_paths=deepnet,
            bootstrap_repetitions=10,
        )


def test_rejects_cross_source_r4_row_identity_mismatch(tmp_path: Path) -> None:
    neural, geometric, deepnet = _inputs(tmp_path)
    payload = json.loads(deepnet["orbit_v3"].read_text(encoding="utf-8"))
    for prediction in payload["records"][0]["prediction_only_test"]["predictions"]:
        prediction["row"] += 1_000
    _write(deepnet["orbit_v3"], payload)
    with pytest.raises(ValueError, match="raw R4 row/label/session identity differs"):
        combined.build_report(
            neural_ranking=neural,
            geometric_controls=geometric,
            deepnet_paths=deepnet,
            bootstrap_repetitions=10,
        )


def test_rejects_stale_neural_subject_seed_aggregation(tmp_path: Path) -> None:
    neural, geometric, deepnet = _inputs(tmp_path)
    payload = json.loads(neural.read_text(encoding="utf-8"))
    payload["ranking"][0]["subject_balanced_accuracy"]["1"] -= 0.1
    _write(neural, payload)
    with pytest.raises(ValueError, match="stale subject score"):
        combined.build_report(
            neural_ranking=neural,
            geometric_controls=geometric,
            deepnet_paths=deepnet,
            bootstrap_repetitions=10,
        )


def test_neural_raw_artifact_recomputes_all_forty_prediction_records(
    tmp_path: Path,
) -> None:
    model = combined.NEURAL_CONFIGURATIONS[0]
    records: list[dict[str, Any]] = []
    expected = {str(subject): {} for subject in combined.SUBJECTS}
    for subject in combined.SUBJECTS:
        prediction = _prediction(subject, correct_per_class=27)
        for seed in combined.SEEDS:
            expected[str(subject)][str(seed)] = prediction["score"]
            records.append(
                {
                    "dataset": combined.DATASET,
                    "model": model,
                    "subject": subject,
                    "fold": 0,
                    "seed": seed,
                    "split": {
                        "train_count": 120,
                        "validation_count": 60,
                        "refit_source_count": 180,
                        "test_count": 60,
                    },
                    "test": {
                        "metrics": {"balanced_accuracy": prediction["score"]},
                        "rows": prediction["rows"],
                        "labels": prediction["labels"],
                        "probabilities": prediction["probabilities"],
                        "sessions": prediction["sessions"],
                        "runs": prediction["runs"],
                    },
                }
            )
    artifact = {
        "schema": combined.NEURAL_ARTIFACT_SCHEMA,
        "mode": "development",
        "artifact_mode": "formal",
        "evidence_scope": "prespecified_formal_development_not_confirmation",
        "confirmation_evidence": False,
        "dataset": combined.DATASET,
        "subjects": list(combined.SUBJECTS),
        "models": [model],
        "folds": [0],
        "seeds": list(combined.SEEDS),
        "records": records,
    }
    path = _write(tmp_path / "artifact.json", artifact)
    matrix = combined._validate_neural_artifact(
        path,
        model=model,
        expected_matrix=expected,
        identities=combined.TestIdentityRegistry(),
    )
    assert matrix == expected

    corrupt = copy.deepcopy(artifact)
    corrupt["records"][0]["test"]["probabilities"].pop()
    _write(path, corrupt)
    with pytest.raises(ValueError, match=r"shape \(60, 2\)"):
        combined._validate_neural_artifact(
            path,
            model=model,
            expected_matrix=expected,
            identities=combined.TestIdentityRegistry(),
        )


def test_strict_json_rejects_duplicates_and_nonfinite_values(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a": 1, "a": 2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON"):
        combined.strict_load(duplicate)
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"a": NaN}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite JSON"):
        combined.strict_load(nonfinite)
