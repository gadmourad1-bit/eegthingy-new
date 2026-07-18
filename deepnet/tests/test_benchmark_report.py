import pytest

from deepnet.report import (
    build_report,
    metrics_from_predictions,
    participant_values,
    validate_prediction_metrics,
)


def _metrics(value: float) -> dict[str, float]:
    return {
        "accuracy": value,
        "balanced_accuracy": value,
        "kappa": 2.0 * value - 1.0,
        "roc_auc": value,
        "brier": 1.0 - value,
        "ece": 0.1,
        "coverage": 0.5,
        "selective_accuracy": value,
        "rest_false_commit_rate": 0.2,
    }


def test_seed_rows_are_averaged_within_participant() -> None:
    rows = [
        {"model": "geoadapt", "subject": 1, "seed": 7, "metrics": _metrics(0.8)},
        {"model": "geoadapt", "subject": 1, "seed": 17, "metrics": _metrics(1.0)},
        {"model": "geoadapt", "subject": 3, "seed": 7, "metrics": _metrics(0.6)},
    ]
    assert participant_values(rows, "geoadapt", "balanced_accuracy") == {1: 0.9, 3: 0.6}
    enriched, markdown = build_report({"folds": rows})
    assert enriched["summary"]["geoadapt"]["participants"] == 2
    assert enriched["summary"]["geoadapt"]["runs"] == 3
    assert "Participant-level means" in markdown


def test_nested_loso_report_uses_protocol_specific_title() -> None:
    rows = [
        {"model": "geoadapt", "subject": 1, "seed": 7, "metrics": _metrics(0.8)}
    ]
    _, markdown = build_report(
        {"protocol": "nested_loso_single_fixed_inner_subject", "folds": rows}
    )
    assert markdown.startswith("# Initial nested leave-one-subject-out benchmark")


def test_schema_v2_metrics_are_auditable_from_prediction_trace() -> None:
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
            "probability_left": 0.5,
            "probability_right": 0.5,
            "committed": False,
        },
    ]
    metrics = metrics_from_predictions(predictions, commit_confidence=0.85)
    payload = {
        "schema_version": 2,
        "benchmark_config": {"commit_confidence": 0.85},
        "folds": [
            {
                "model": "geoadapt",
                "subject": 1,
                "seed": 7,
                "metrics": metrics,
                "predictions": predictions,
            }
        ],
    }
    validate_prediction_metrics(payload)
    payload["folds"][0]["metrics"]["balanced_accuracy"] = 0.0
    with pytest.raises(ValueError, match="balanced_accuracy"):
        validate_prediction_metrics(payload)
