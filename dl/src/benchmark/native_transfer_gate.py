"""Locked development-only decision gate for native-montage transfer.

This module evaluates already-computed aggregate balanced accuracies.  It has
no data-loading, prediction, training, filesystem, or confirmation surface.
The two development datasets, four conditions, comparison directions, and
numeric thresholds are constants so a caller cannot redefine the screen while
evaluating it.

Balanced accuracies and deltas use the unit interval.  Every emitted delta also
has an explicitly named percentage-point representation to avoid unit
ambiguity in reports.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real


GATE_SCHEMA = "eeg-mi-native-transfer-development-gate-v1"
BOOTSTRAP_SCHEMA = "eeg-mi-native-transfer-physionet-paired-bootstrap-v1"
MODE = "development"
ANALYSIS_POLICY = "locked_development_screen_not_confirmation_evidence"

CHO2017 = "cho2017"
PHYSIONET_MI = "physionet_mi"
LOCKED_DATASETS: tuple[str, ...] = (CHO2017, PHYSIONET_MI)

PRETRAINED_CARDINAL_FBC = "pretrained_cardinal_fbc"
SCRATCH_CARDINAL_FBC = "scratch_cardinal_fbc"
PRETRAINED_INDEXED_FBCNET_SPLINE = (
    "pretrained_indexed_fbcnet_spherical_spline"
)
SCRATCH_FBMSNET = "scratch_fbmsnet"
LOCKED_CONDITIONS: tuple[str, ...] = (
    PRETRAINED_CARDINAL_FBC,
    SCRATCH_CARDINAL_FBC,
    PRETRAINED_INDEXED_FBCNET_SPLINE,
    SCRATCH_FBMSNET,
)
REFERENCE_CONDITIONS: tuple[str, ...] = (
    PRETRAINED_INDEXED_FBCNET_SPLINE,
    SCRATCH_FBMSNET,
)

# Thresholds are balanced-accuracy fractions.  For example, 0.01 is one
# percentage point, not one percent relative improvement.
MIN_SCRATCH_EQUAL_DATASET_DELTA = 0.01
MIN_REFERENCE_ENVELOPE_EQUAL_DATASET_DELTA = 0.005
MIN_WORST_DATASET_REFERENCE_ENVELOPE_DELTA = -0.01
BOOTSTRAP_CONFIDENCE_LEVEL = 0.95

_BOOTSTRAP_FIELDS = frozenset(
    {
        "schema",
        "dataset",
        "candidate_condition",
        "comparator_condition",
        "resampling_unit",
        "paired",
        "confidence_level",
        "bound",
        "lower_bound_balanced_accuracy_delta",
        "repetitions",
        "seed",
    }
)


def _require_exact_keys(
    value: Mapping[object, object],
    expected: frozenset[str],
    *,
    name: str,
) -> None:
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected, key=repr)
        raise ValueError(f"{name} fields differ; missing={missing}, extra={extra}")


def _finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _balanced_accuracy(value: object, *, name: str) -> float:
    result = _finite_number(value, name=name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


def _balanced_accuracy_delta(value: object, *, name: str) -> float:
    result = _finite_number(value, name=name)
    if not -1.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [-1, 1]")
    return result


def _validated_condition_means(
    dataset_condition_means: Mapping[str, Mapping[str, Real]],
) -> dict[str, dict[str, float]]:
    if not isinstance(dataset_condition_means, Mapping):
        raise ValueError("dataset_condition_means must be a mapping")
    _require_exact_keys(
        dataset_condition_means,
        frozenset(LOCKED_DATASETS),
        name="dataset_condition_means",
    )

    result: dict[str, dict[str, float]] = {}
    for dataset in LOCKED_DATASETS:
        conditions = dataset_condition_means[dataset]
        if not isinstance(conditions, Mapping):
            raise ValueError(f"dataset_condition_means[{dataset!r}] must be a mapping")
        _require_exact_keys(
            conditions,
            frozenset(LOCKED_CONDITIONS),
            name=f"dataset_condition_means[{dataset!r}]",
        )
        result[dataset] = {
            condition: _balanced_accuracy(
                conditions[condition],
                name=(
                    f"dataset_condition_means[{dataset!r}]"
                    f"[{condition!r}]"
                ),
            )
            for condition in LOCKED_CONDITIONS
        }
    return result


def _validated_bootstrap(
    bootstrap: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(bootstrap, Mapping):
        raise ValueError("physionet_bootstrap must be a mapping")
    _require_exact_keys(
        bootstrap,
        _BOOTSTRAP_FIELDS,
        name="physionet_bootstrap",
    )

    exact_values = {
        "schema": BOOTSTRAP_SCHEMA,
        "dataset": PHYSIONET_MI,
        "candidate_condition": PRETRAINED_CARDINAL_FBC,
        "comparator_condition": SCRATCH_CARDINAL_FBC,
        "resampling_unit": "subject",
        "paired": True,
        "bound": "one_sided_lower",
    }
    for field, expected in exact_values.items():
        if bootstrap[field] != expected or (
            field == "paired" and not isinstance(bootstrap[field], bool)
        ):
            raise ValueError(
                f"physionet_bootstrap.{field} must equal {expected!r}"
            )

    confidence_level = _finite_number(
        bootstrap["confidence_level"],
        name="physionet_bootstrap.confidence_level",
    )
    if not math.isclose(
        confidence_level,
        BOOTSTRAP_CONFIDENCE_LEVEL,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "physionet_bootstrap.confidence_level must equal the locked 0.95"
        )

    lower_bound = _balanced_accuracy_delta(
        bootstrap["lower_bound_balanced_accuracy_delta"],
        name="physionet_bootstrap.lower_bound_balanced_accuracy_delta",
    )
    repetitions = bootstrap["repetitions"]
    if (
        isinstance(repetitions, bool)
        or not isinstance(repetitions, int)
        or repetitions <= 0
    ):
        raise ValueError("physionet_bootstrap.repetitions must be a positive integer")
    seed = bootstrap["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("physionet_bootstrap.seed must be a nonnegative integer")

    return {
        **exact_values,
        "confidence_level": confidence_level,
        "lower_bound_balanced_accuracy_delta": lower_bound,
        "repetitions": repetitions,
        "seed": seed,
    }


def _delta(candidate: float, comparator: float) -> dict[str, float]:
    value = candidate - comparator
    return {
        "balanced_accuracy": value,
        "percentage_points": 100.0 * value,
    }


def _check(
    *,
    observed: float,
    threshold: float,
    operator: str,
    passed: bool,
) -> dict[str, object]:
    return {
        "observed_balanced_accuracy_delta": observed,
        "observed_percentage_points": 100.0 * observed,
        "operator": operator,
        "threshold_balanced_accuracy_delta": threshold,
        "threshold_percentage_points": 100.0 * threshold,
        "pass": bool(passed),
    }


def evaluate_native_transfer_gate(
    dataset_condition_means: Mapping[str, Mapping[str, Real]],
    *,
    physionet_bootstrap: Mapping[str, object],
) -> dict[str, object]:
    """Evaluate the immutable native-transfer development screen.

    ``dataset_condition_means`` must contain exactly the two locked datasets
    and exactly the four locked conditions for each dataset.  The reference
    envelope is formed independently within each dataset before equal-dataset
    averaging; it is never a per-subject oracle and never the maximum of
    already-averaged condition macros.

    The PhysioNet bootstrap result is deliberately supplied rather than
    recomputed here.  Its schema pins the paired comparison and records the
    repetitions and seed used by the separate subject-level bootstrap.
    """

    means = _validated_condition_means(dataset_condition_means)
    bootstrap = _validated_bootstrap(physionet_bootstrap)

    dataset_results: dict[str, dict[str, object]] = {}
    scratch_deltas: list[float] = []
    indexed_deltas: list[float] = []
    fbms_deltas: list[float] = []
    envelope_deltas: list[float] = []

    for dataset in LOCKED_DATASETS:
        scores = means[dataset]
        candidate = scores[PRETRAINED_CARDINAL_FBC]
        scratch = scores[SCRATCH_CARDINAL_FBC]
        indexed = scores[PRETRAINED_INDEXED_FBCNET_SPLINE]
        fbms = scores[SCRATCH_FBMSNET]
        envelope = max(indexed, fbms)
        envelope_conditions = [
            condition
            for condition in REFERENCE_CONDITIONS
            if scores[condition] == envelope
        ]

        scratch_delta = candidate - scratch
        indexed_delta = candidate - indexed
        fbms_delta = candidate - fbms
        envelope_delta = candidate - envelope
        scratch_deltas.append(scratch_delta)
        indexed_deltas.append(indexed_delta)
        fbms_deltas.append(fbms_delta)
        envelope_deltas.append(envelope_delta)

        dataset_results[dataset] = {
            "condition_mean_balanced_accuracy": dict(scores),
            "reference_envelope_conditions": envelope_conditions,
            "reference_envelope_balanced_accuracy": envelope,
            "deltas": {
                "candidate_minus_scratch_cardinal_fbc": _delta(candidate, scratch),
                "candidate_minus_pretrained_indexed_fbcnet_spherical_spline": (
                    _delta(candidate, indexed)
                ),
                "candidate_minus_scratch_fbmsnet": _delta(candidate, fbms),
                "candidate_minus_reference_envelope": _delta(candidate, envelope),
            },
        }

    def equal_dataset_mean(condition: str) -> float:
        return sum(means[dataset][condition] for dataset in LOCKED_DATASETS) / len(
            LOCKED_DATASETS
        )

    candidate_macro = equal_dataset_mean(PRETRAINED_CARDINAL_FBC)
    scratch_macro = equal_dataset_mean(SCRATCH_CARDINAL_FBC)
    indexed_macro = equal_dataset_mean(PRETRAINED_INDEXED_FBCNET_SPLINE)
    fbms_macro = equal_dataset_mean(SCRATCH_FBMSNET)
    envelope_macro = sum(
        float(dataset_results[dataset]["reference_envelope_balanced_accuracy"])
        for dataset in LOCKED_DATASETS
    ) / len(LOCKED_DATASETS)

    scratch_macro_delta = sum(scratch_deltas) / len(scratch_deltas)
    indexed_macro_delta = sum(indexed_deltas) / len(indexed_deltas)
    fbms_macro_delta = sum(fbms_deltas) / len(fbms_deltas)
    envelope_macro_delta = sum(envelope_deltas) / len(envelope_deltas)
    worst_envelope_delta = min(envelope_deltas)
    positive_scratch_by_dataset = {
        dataset: scratch_deltas[index] > 0.0
        for index, dataset in enumerate(LOCKED_DATASETS)
    }

    gate_checks = {
        "candidate_minus_scratch_equal_dataset_at_least_1pp": _check(
            observed=scratch_macro_delta,
            threshold=MIN_SCRATCH_EQUAL_DATASET_DELTA,
            operator=">=",
            passed=scratch_macro_delta >= MIN_SCRATCH_EQUAL_DATASET_DELTA,
        ),
        "candidate_minus_scratch_positive_on_both_datasets": {
            "observed_by_dataset": {
                dataset: _delta(
                    means[dataset][PRETRAINED_CARDINAL_FBC],
                    means[dataset][SCRATCH_CARDINAL_FBC],
                )
                for dataset in LOCKED_DATASETS
            },
            "pass_by_dataset": positive_scratch_by_dataset,
            "operator": ">",
            "threshold_balanced_accuracy_delta": 0.0,
            "threshold_percentage_points": 0.0,
            "pass": all(positive_scratch_by_dataset.values()),
        },
        "physionet_paired_subject_bootstrap_one_sided_95_lower_above_zero": {
            **_check(
                observed=float(
                    bootstrap["lower_bound_balanced_accuracy_delta"]
                ),
                threshold=0.0,
                operator=">",
                passed=(
                    float(bootstrap["lower_bound_balanced_accuracy_delta"])
                    > 0.0
                ),
            ),
            "repetitions": bootstrap["repetitions"],
            "seed": bootstrap["seed"],
        },
        "candidate_minus_reference_envelope_equal_dataset_at_least_0_5pp": (
            _check(
                observed=envelope_macro_delta,
                threshold=MIN_REFERENCE_ENVELOPE_EQUAL_DATASET_DELTA,
                operator=">=",
                passed=(
                    envelope_macro_delta
                    >= MIN_REFERENCE_ENVELOPE_EQUAL_DATASET_DELTA
                ),
            )
        ),
        "worst_dataset_reference_envelope_delta_at_least_minus_1pp": _check(
            observed=worst_envelope_delta,
            threshold=MIN_WORST_DATASET_REFERENCE_ENVELOPE_DELTA,
            operator=">=",
            passed=(
                worst_envelope_delta
                >= MIN_WORST_DATASET_REFERENCE_ENVELOPE_DELTA
            ),
        ),
    }

    return {
        "schema": GATE_SCHEMA,
        "mode": MODE,
        "confirmation_access": False,
        "confirmatory_evidence": False,
        "analysis_policy": ANALYSIS_POLICY,
        "locked_contract": {
            "datasets": list(LOCKED_DATASETS),
            "candidate_condition": PRETRAINED_CARDINAL_FBC,
            "scratch_transfer_comparator": SCRATCH_CARDINAL_FBC,
            "reference_envelope_conditions": list(REFERENCE_CONDITIONS),
            "reference_envelope_aggregation": (
                "maximum condition mean within dataset, then equal-dataset mean"
            ),
            "minimum_candidate_minus_scratch_equal_dataset_percentage_points": 1.0,
            "candidate_minus_scratch_must_be_positive_on_every_dataset": True,
            "physionet_bootstrap_confidence_level": BOOTSTRAP_CONFIDENCE_LEVEL,
            "physionet_bootstrap_lower_bound_operator": ">",
            (
                "minimum_candidate_minus_reference_envelope_equal_dataset_"
                "percentage_points"
            ): 0.5,
            "minimum_worst_dataset_reference_envelope_delta_percentage_points": -1.0,
        },
        "inputs": {
            "dataset_condition_mean_balanced_accuracy": means,
            "physionet_paired_subject_bootstrap": bootstrap,
        },
        "dataset_results": dataset_results,
        "equal_dataset_macro": {
            "condition_mean_balanced_accuracy": {
                PRETRAINED_CARDINAL_FBC: candidate_macro,
                SCRATCH_CARDINAL_FBC: scratch_macro,
                PRETRAINED_INDEXED_FBCNET_SPLINE: indexed_macro,
                SCRATCH_FBMSNET: fbms_macro,
            },
            "reference_envelope_balanced_accuracy": envelope_macro,
            "deltas": {
                "candidate_minus_scratch_cardinal_fbc": _delta(
                    candidate_macro, scratch_macro
                ),
                "candidate_minus_pretrained_indexed_fbcnet_spherical_spline": (
                    _delta(candidate_macro, indexed_macro)
                ),
                "candidate_minus_scratch_fbmsnet": _delta(
                    candidate_macro, fbms_macro
                ),
                "candidate_minus_reference_envelope": {
                    "balanced_accuracy": envelope_macro_delta,
                    "percentage_points": 100.0 * envelope_macro_delta,
                },
            },
        },
        "gate_checks": gate_checks,
        "overall_pass": all(
            bool(check["pass"]) for check in gate_checks.values()
        ),
    }


__all__ = [
    "ANALYSIS_POLICY",
    "BOOTSTRAP_CONFIDENCE_LEVEL",
    "BOOTSTRAP_SCHEMA",
    "CHO2017",
    "GATE_SCHEMA",
    "LOCKED_CONDITIONS",
    "LOCKED_DATASETS",
    "MIN_REFERENCE_ENVELOPE_EQUAL_DATASET_DELTA",
    "MIN_SCRATCH_EQUAL_DATASET_DELTA",
    "MIN_WORST_DATASET_REFERENCE_ENVELOPE_DELTA",
    "PHYSIONET_MI",
    "PRETRAINED_CARDINAL_FBC",
    "PRETRAINED_INDEXED_FBCNET_SPLINE",
    "REFERENCE_CONDITIONS",
    "SCRATCH_CARDINAL_FBC",
    "SCRATCH_FBMSNET",
    "evaluate_native_transfer_gate",
]
