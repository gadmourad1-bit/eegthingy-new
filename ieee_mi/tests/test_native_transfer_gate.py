from __future__ import annotations

import copy
import json

import pytest

from ieee_mi import native_transfer_gate as gate


def _actual_fold0_seed7_means() -> dict[str, dict[str, float]]:
    return {
        gate.CHO2017: {
            gate.PRETRAINED_CARDINAL_FBC: 0.6644144144,
            gate.PRETRAINED_INDEXED_FBCNET_SPLINE: 0.6469594595,
            gate.SCRATCH_CARDINAL_FBC: 0.6292792793,
            gate.SCRATCH_FBMSNET: 0.6708333333,
        },
        gate.PHYSIONET_MI: {
            gate.PRETRAINED_CARDINAL_FBC: 0.5911044974,
            gate.PRETRAINED_INDEXED_FBCNET_SPLINE: 0.5843253968,
            gate.SCRATCH_CARDINAL_FBC: 0.5292658730,
            gate.SCRATCH_FBMSNET: 0.5530753968,
        },
    }


def _bootstrap(*, lower_bound: float = 0.0362103) -> dict[str, object]:
    return {
        "schema": gate.BOOTSTRAP_SCHEMA,
        "dataset": gate.PHYSIONET_MI,
        "candidate_condition": gate.PRETRAINED_CARDINAL_FBC,
        "comparator_condition": gate.SCRATCH_CARDINAL_FBC,
        "resampling_unit": "subject",
        "paired": True,
        "confidence_level": 0.95,
        "bound": "one_sided_lower",
        "lower_bound_balanced_accuracy_delta": lower_bound,
        "repetitions": 200_000,
        "seed": 20_260_719,
    }


def _passing_means() -> dict[str, dict[str, float]]:
    return {
        dataset: {
            gate.PRETRAINED_CARDINAL_FBC: 0.61,
            gate.SCRATCH_CARDINAL_FBC: 0.60,
            gate.PRETRAINED_INDEXED_FBCNET_SPLINE: 0.605,
            gate.SCRATCH_FBMSNET: 0.59,
        }
        for dataset in gate.LOCKED_DATASETS
    }


def test_actual_fold0_seed7_numbers_fail_only_reference_envelope_macro() -> None:
    result = gate.evaluate_native_transfer_gate(
        _actual_fold0_seed7_means(),
        physionet_bootstrap=_bootstrap(),
    )

    assert result["mode"] == "development"
    assert result["confirmation_access"] is False
    assert result["confirmatory_evidence"] is False
    assert result["overall_pass"] is False
    failed = [
        name for name, check in result["gate_checks"].items() if not check["pass"]
    ]
    assert failed == [
        "candidate_minus_reference_envelope_equal_dataset_at_least_0_5pp"
    ]

    datasets = result["dataset_results"]
    assert datasets[gate.CHO2017]["reference_envelope_conditions"] == [
        gate.SCRATCH_FBMSNET
    ]
    assert datasets[gate.PHYSIONET_MI]["reference_envelope_conditions"] == [
        gate.PRETRAINED_INDEXED_FBCNET_SPLINE
    ]
    assert datasets[gate.CHO2017]["deltas"][
        "candidate_minus_reference_envelope"
    ]["percentage_points"] == pytest.approx(-0.64189189)
    assert datasets[gate.PHYSIONET_MI]["deltas"][
        "candidate_minus_reference_envelope"
    ]["percentage_points"] == pytest.approx(0.67791006)

    macro = result["equal_dataset_macro"]
    assert macro["condition_mean_balanced_accuracy"][
        gate.PRETRAINED_CARDINAL_FBC
    ] == pytest.approx(0.6277594559)
    assert macro["reference_envelope_balanced_accuracy"] == pytest.approx(
        0.62757936505
    )
    assert macro["deltas"]["candidate_minus_scratch_cardinal_fbc"][
        "percentage_points"
    ] == pytest.approx(4.848687975)
    assert macro["deltas"][
        "candidate_minus_pretrained_indexed_fbcnet_spherical_spline"
    ]["percentage_points"] == pytest.approx(1.211702775)
    assert macro["deltas"]["candidate_minus_scratch_fbmsnet"][
        "percentage_points"
    ] == pytest.approx(1.580509085)
    assert macro["deltas"]["candidate_minus_reference_envelope"][
        "percentage_points"
    ] == pytest.approx(0.018009085)

    # The result is a finite JSON artifact, not a Python-only report object.
    json.dumps(result, allow_nan=False)


def test_inclusive_macro_and_worst_thresholds_and_strict_bootstrap_pass() -> None:
    result = gate.evaluate_native_transfer_gate(
        _passing_means(),
        physionet_bootstrap=_bootstrap(lower_bound=1e-12),
    )
    assert result["overall_pass"] is True
    assert all(check["pass"] for check in result["gate_checks"].values())


def test_bootstrap_lower_bound_is_strictly_greater_than_zero() -> None:
    result = gate.evaluate_native_transfer_gate(
        _passing_means(),
        physionet_bootstrap=_bootstrap(lower_bound=0.0),
    )
    check = result["gate_checks"][
        "physionet_paired_subject_bootstrap_one_sided_95_lower_above_zero"
    ]
    assert check["pass"] is False
    assert result["overall_pass"] is False


def test_scratch_delta_must_be_strictly_positive_on_each_dataset() -> None:
    means = _passing_means()
    means[gate.CHO2017][gate.SCRATCH_CARDINAL_FBC] = means[gate.CHO2017][
        gate.PRETRAINED_CARDINAL_FBC
    ]
    means[gate.PHYSIONET_MI][gate.PRETRAINED_CARDINAL_FBC] = 0.63
    result = gate.evaluate_native_transfer_gate(
        means,
        physionet_bootstrap=_bootstrap(),
    )
    check = result["gate_checks"][
        "candidate_minus_scratch_positive_on_both_datasets"
    ]
    assert check["pass_by_dataset"] == {
        gate.CHO2017: False,
        gate.PHYSIONET_MI: True,
    }
    assert check["pass"] is False


@pytest.mark.parametrize("extra", [False, True])
def test_rejects_missing_or_extra_dataset(extra: bool) -> None:
    means = _passing_means()
    if extra:
        means["unlocked_dataset"] = copy.deepcopy(means[gate.CHO2017])
    else:
        means.pop(gate.CHO2017)
    with pytest.raises(ValueError, match="dataset_condition_means fields differ"):
        gate.evaluate_native_transfer_gate(
            means,
            physionet_bootstrap=_bootstrap(),
        )


@pytest.mark.parametrize("extra", [False, True])
def test_rejects_missing_or_extra_condition(extra: bool) -> None:
    means = _passing_means()
    if extra:
        means[gate.CHO2017]["unlocked_condition"] = 0.5
    else:
        means[gate.CHO2017].pop(gate.SCRATCH_FBMSNET)
    with pytest.raises(ValueError, match="conditions|fields differ"):
        gate.evaluate_native_transfer_gate(
            means,
            physionet_bootstrap=_bootstrap(),
        )


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -0.01, 1.01, True])
def test_rejects_nonfinite_out_of_range_or_boolean_score(score: float) -> None:
    means = _passing_means()
    means[gate.CHO2017][gate.PRETRAINED_CARDINAL_FBC] = score
    with pytest.raises(ValueError, match=r"finite|\[0, 1\]"):
        gate.evaluate_native_transfer_gate(
            means,
            physionet_bootstrap=_bootstrap(),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema", "wrong", "schema"),
        ("dataset", gate.CHO2017, "dataset"),
        ("candidate_condition", gate.SCRATCH_CARDINAL_FBC, "candidate_condition"),
        ("comparator_condition", gate.SCRATCH_FBMSNET, "comparator_condition"),
        ("resampling_unit", "trial", "resampling_unit"),
        ("paired", False, "paired"),
        ("paired", 1, "paired"),
        ("confidence_level", 0.90, "confidence_level"),
        ("bound", "two_sided_lower", "bound"),
        ("lower_bound_balanced_accuracy_delta", float("nan"), "finite"),
        ("lower_bound_balanced_accuracy_delta", 1.01, r"\[-1, 1\]"),
        ("repetitions", 0, "positive integer"),
        ("repetitions", True, "positive integer"),
        ("seed", -1, "nonnegative integer"),
        ("seed", True, "nonnegative integer"),
    ],
)
def test_rejects_invalid_bootstrap_provenance(
    field: str,
    value: object,
    message: str,
) -> None:
    bootstrap = _bootstrap()
    bootstrap[field] = value
    with pytest.raises(ValueError, match=message):
        gate.evaluate_native_transfer_gate(
            _passing_means(),
            physionet_bootstrap=bootstrap,
        )


def test_rejects_missing_or_extra_bootstrap_metadata() -> None:
    missing = _bootstrap()
    missing.pop("seed")
    with pytest.raises(ValueError, match="missing=.*seed"):
        gate.evaluate_native_transfer_gate(
            _passing_means(),
            physionet_bootstrap=missing,
        )

    extra = _bootstrap()
    extra["notes"] = "not part of the locked schema"
    with pytest.raises(ValueError, match="extra=.*notes"):
        gate.evaluate_native_transfer_gate(
            _passing_means(),
            physionet_bootstrap=extra,
        )
