import pytest

from benchmark.research.merge_results import merge_payloads


def _payload(model: str, *, contract: str = "same") -> dict:
    return {
        "schema_version": 2,
        "protocol": "nested_loso_train-6_validate_calibrate-1_test-1",
        "data_contract": {"contract_sha256": contract},
        "outer_subjects": [1],
        "loso_config": {"commit_confidence": 0.85},
        "folds": [
            {
                "model": model,
                "subject": 1,
                "seed": 0,
                "metrics": {
                    "accuracy": 0.5,
                    "balanced_accuracy": 0.5,
                    "kappa": 0.0,
                    "roc_auc": 0.5,
                    "brier": 0.25,
                    "ece": 0.0,
                    "coverage": 0.0,
                    "selective_accuracy": None,
                    "rest_false_commit_rate": None,
                },
            }
        ],
    }


def test_merge_requires_matching_protocol_data_and_unique_rows() -> None:
    merged = merge_payloads((_payload("geoadapt"), _payload("riemann")))
    assert merged["models"] == ["geoadapt", "riemann"]
    assert len(merged["folds"]) == 2

    with pytest.raises(ValueError, match="data contracts"):
        merge_payloads((_payload("geoadapt"), _payload("riemann", contract="changed")))
    with pytest.raises(ValueError, match="duplicate"):
        merge_payloads((_payload("geoadapt"), _payload("geoadapt")))
