"""Immutable decision gate over freshly revalidated CardinalFBMS records.

The public gate accepts no caller-authored means, confidence interval, or
threshold inputs.  It reopens the two roots carried by a complete locked audit,
validates every one of the 455 records again, recomputes equal-subject means,
and performs the prespecified paired Physionet bootstrap internally.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from . import native_fbms_transfer as transfer
from . import native_fbms_transfer_audit as audit
from . import native_transfer_audit as artifact_audit
from . import native_transfer_gate as _core


GATE_SCHEMA = "ieee-mi-native-cardinal-fbms-transfer-gate-v1"
BOOTSTRAP_SCHEMA = "ieee-mi-native-cardinal-fbms-physionet-bootstrap-v1"
MODE = "development"
ANALYSIS_POLICY = "locked_fbms_development_screen_not_confirmation_evidence"

CHO2017 = audit.CHO2017
PHYSIONET_MI = audit.PHYSIONET_MI
LOCKED_DATASETS = audit.LOCKED_DATASETS
LOCKED_CONDITIONS = transfer.TRANSFER_CONDITIONS
CANDIDATE_CONDITION = transfer.PRETRAINED_CARDINAL_FBMS
TRANSFER_COMPARATOR = transfer.SCRATCH_CARDINAL_FBMS_CANONICAL
REFERENCE_CONDITIONS: tuple[str, ...] = (
    transfer.SCRATCH_CARDINAL_FBMS_NATIVE,
    transfer.SCRATCH_FBMSNET_NATIVE,
    transfer.PRETRAINED_INDEXED_FBMSNET_SPLINE,
)
HISTORICAL_REFERENCE_NAME = "historical_opened_screen_best"
# Frozen from the fully audited legacy fold0/seed7 opened screen.  Using its
# best condition within each dataset prevents a new family from advancing by
# merely underperforming a previously observed development result.
HISTORICAL_OPENED_SCREEN_FLOOR = {
    CHO2017: 0.6708333333,
    PHYSIONET_MI: 0.5911044974,
}

MIN_SCRATCH_EQUAL_DATASET_DELTA = 0.01
MIN_REFERENCE_ENVELOPE_EQUAL_DATASET_DELTA = 0.005
MIN_WORST_DATASET_REFERENCE_ENVELOPE_DELTA = -0.01
BOOTSTRAP_CONFIDENCE_LEVEL = 0.95
BOOTSTRAP_REPETITIONS = 200_000
BOOTSTRAP_SEED = 20_260_719
BOOTSTRAP_BATCH_SIZE = 4096

_SUMMARY_FIELDS = frozenset(
    {
        "schema",
        "mode",
        "confirmation_access",
        "confirmatory_evidence",
        "analysis_policy",
        "inferential_statistics",
        "complete_grid",
        "locked_grid",
        "expected_record_count",
        "validated_record_count",
        "dataset_condition_mean_balanced_accuracy",
        "datasets",
    }
)
def _validated_screen_summary(
    summary: Mapping[str, object],
) -> dict[str, dict[str, float]]:
    if not isinstance(summary, Mapping):
        raise ValueError("screen_summary must be a mapping")
    _core._require_exact_keys(summary, _SUMMARY_FIELDS, name="screen_summary")
    exact = {
        "schema": audit.SUMMARY_SCHEMA,
        "mode": MODE,
        "confirmation_access": False,
        "confirmatory_evidence": False,
        "analysis_policy": audit.SUMMARY_ANALYSIS_POLICY,
        "inferential_statistics": False,
        "complete_grid": True,
        "expected_record_count": audit.EXPECTED_RECORD_COUNT,
        "validated_record_count": audit.EXPECTED_RECORD_COUNT,
    }
    for field, expected in exact.items():
        value = summary[field]
        if value != expected or (
            isinstance(expected, bool) and not isinstance(value, bool)
        ):
            raise ValueError(f"screen_summary.{field} must equal {expected!r}")
    if summary["locked_grid"] != audit.locked_screen_grid():
        raise ValueError("screen_summary does not represent the immutable full grid")

    means = summary["dataset_condition_mean_balanced_accuracy"]
    if not isinstance(means, Mapping):
        raise ValueError("screen summary condition means must be a mapping")
    _core._require_exact_keys(means, frozenset(LOCKED_DATASETS), name="means")
    result: dict[str, dict[str, float]] = {}
    for dataset in LOCKED_DATASETS:
        conditions = means[dataset]
        if not isinstance(conditions, Mapping):
            raise ValueError(f"condition means for {dataset} must be a mapping")
        _core._require_exact_keys(
            conditions,
            frozenset(LOCKED_CONDITIONS),
            name=f"condition means for {dataset}",
        )
        result[dataset] = {
            condition: _core._balanced_accuracy(
                conditions[condition],
                name=f"{dataset}.{condition}",
            )
            for condition in LOCKED_CONDITIONS
        }
    return result


def _expected_directory_keys(
    dataset: str,
) -> tuple[artifact_audit.TransferRecordKey, ...]:
    return artifact_audit.development_transfer_grid(
        dataset=dataset,
        subjects=audit.LOCKED_SUBJECTS[dataset],
        folds=audit.LOCKED_FOLDS,
        seeds=audit.LOCKED_SEEDS,
        conditions=audit.LOCKED_CONDITIONS,
        contract=audit.AUDIT_CONTRACT,
    )


def _revalidate_complete_screen(
    screen: audit.LockedFBMSScreenAudit,
) -> audit.LockedFBMSScreenAudit:
    """Reject fabricated/stale aggregates and reopen every record artifact."""

    if not isinstance(screen, audit.LockedFBMSScreenAudit):
        raise TypeError("screen must be a LockedFBMSScreenAudit")
    directories = {
        CHO2017: screen.cho2017,
        PHYSIONET_MI: screen.physionet_mi,
    }
    for dataset, directory in directories.items():
        expected = _expected_directory_keys(dataset)
        if directory.contract != audit.AUDIT_CONTRACT:
            raise ValueError(f"{dataset} audit uses a different family contract")
        if directory.name_template != artifact_audit.DEFAULT_RECORD_NAME_TEMPLATE:
            raise ValueError(f"{dataset} audit uses a nonlocked record name template")
        if directory.expected_keys != expected:
            raise ValueError(f"{dataset} audit does not represent the full locked grid")
        if not directory.complete or directory.missing_keys:
            raise ValueError(f"{dataset} audit is incomplete")
        if tuple(record.key for record in directory.records) != expected:
            raise ValueError(f"{dataset} audit records differ from the locked grid")
    if len(screen.records) != audit.EXPECTED_RECORD_COUNT:
        raise ValueError("screen does not contain exactly 455 validated records")

    # A previously valid dataclass is not a durable authorization token: files
    # can change after construction.  Reopening both roots binds gate inputs to
    # current prediction/provenance bytes and reruns every cross-record check.
    refreshed = audit.audit_locked_fbms_screen(
        cho_root=screen.cho2017.root,
        physionet_root=screen.physionet_mi.root,
    )
    if not refreshed.complete or len(refreshed.records) != audit.EXPECTED_RECORD_COUNT:
        raise ValueError("freshly revalidated screen is incomplete")
    return refreshed


def _physionet_subject_scores(
    screen_summary: Mapping[str, object],
    condition: str,
) -> tuple[np.ndarray, np.ndarray]:
    datasets = screen_summary.get("datasets")
    if not isinstance(datasets, Mapping):
        raise RuntimeError("auditor summary datasets are missing")
    dataset = datasets.get(PHYSIONET_MI)
    if not isinstance(dataset, Mapping):
        raise RuntimeError("auditor Physionet summary is missing")
    conditions = dataset.get("conditions")
    if not isinstance(conditions, Mapping):
        raise RuntimeError("auditor Physionet condition summaries are missing")
    condition_summary = conditions.get(condition)
    if not isinstance(condition_summary, Mapping):
        raise RuntimeError(f"auditor summary misses condition {condition}")
    scores = condition_summary.get("subject_balanced_accuracy")
    if not isinstance(scores, Mapping):
        raise RuntimeError("auditor subject balanced accuracies are missing")
    subjects = np.asarray(audit.LOCKED_SUBJECTS[PHYSIONET_MI], dtype=np.int64)
    expected_keys = {str(subject) for subject in subjects.tolist()}
    if set(scores) != expected_keys:
        raise RuntimeError("auditor subject scores differ from locked Physionet cohort")
    values = np.asarray(
        [
            _core._balanced_accuracy(
                scores[str(subject)],
                name=f"{PHYSIONET_MI}.{condition}.S{subject}",
            )
            for subject in subjects.tolist()
        ],
        dtype=np.float64,
    )
    return subjects, values


def _paired_physionet_bootstrap(
    screen_summary: Mapping[str, object],
) -> dict[str, object]:
    candidate_subjects, candidate = _physionet_subject_scores(
        screen_summary, CANDIDATE_CONDITION
    )
    comparator_subjects, comparator = _physionet_subject_scores(
        screen_summary, TRANSFER_COMPARATOR
    )
    if not np.array_equal(candidate_subjects, comparator_subjects):
        raise RuntimeError("paired Physionet subject identities differ")
    differences = candidate - comparator
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap_means = np.empty(BOOTSTRAP_REPETITIONS, dtype=np.float64)
    offset = 0
    while offset < BOOTSTRAP_REPETITIONS:
        count = min(BOOTSTRAP_BATCH_SIZE, BOOTSTRAP_REPETITIONS - offset)
        indices = generator.integers(
            0,
            len(differences),
            size=(count, len(differences)),
            dtype=np.int64,
        )
        bootstrap_means[offset : offset + count] = differences[indices].mean(
            axis=1
        )
        offset += count
    lower = float(
        np.quantile(
            bootstrap_means,
            1.0 - BOOTSTRAP_CONFIDENCE_LEVEL,
            method="linear",
        )
    )
    return {
        "schema": BOOTSTRAP_SCHEMA,
        "dataset": PHYSIONET_MI,
        "candidate_condition": CANDIDATE_CONDITION,
        "comparator_condition": TRANSFER_COMPARATOR,
        "resampling_unit": "subject",
        "paired": True,
        "method": "nonparametric_paired_percentile_bootstrap",
        "bound": "one_sided_lower",
        "confidence_level": BOOTSTRAP_CONFIDENCE_LEVEL,
        "quantile_probability": 1.0 - BOOTSTRAP_CONFIDENCE_LEVEL,
        "quantile_method": "linear",
        "lower_bound_balanced_accuracy_delta": lower,
        "repetitions": BOOTSTRAP_REPETITIONS,
        "seed": BOOTSTRAP_SEED,
        "subject_count": len(candidate_subjects),
        "subjects": candidate_subjects.tolist(),
        "candidate_subject_scores_sha256": artifact_audit._array_sha256(
            candidate
        ),
        "comparator_subject_scores_sha256": artifact_audit._array_sha256(
            comparator
        ),
        "paired_subject_differences_sha256": artifact_audit._array_sha256(
            differences
        ),
    }


def evaluate_fbms_transfer_gate(
    screen: audit.LockedFBMSScreenAudit,
) -> dict[str, object]:
    """Evaluate only freshly revalidated files from the exact locked screen."""

    refreshed = _revalidate_complete_screen(screen)
    screen_summary = audit.descriptive_screen_summary(refreshed)
    means = _validated_screen_summary(screen_summary)
    bootstrap = _paired_physionet_bootstrap(screen_summary)
    manifest = audit.audit_manifest(refreshed)
    manifest_sha256 = hashlib.sha256(
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    dataset_results: dict[str, dict[str, object]] = {}
    scratch_deltas: list[float] = []
    envelope_deltas: list[float] = []

    for dataset in LOCKED_DATASETS:
        scores = means[dataset]
        candidate = scores[CANDIDATE_CONDITION]
        scratch = scores[TRANSFER_COMPARATOR]
        reference_scores = {
            condition: scores[condition] for condition in REFERENCE_CONDITIONS
        }
        reference_scores[HISTORICAL_REFERENCE_NAME] = (
            HISTORICAL_OPENED_SCREEN_FLOOR[dataset]
        )
        envelope = max(reference_scores.values())
        envelope_sources = [
            name for name, score in reference_scores.items() if score == envelope
        ]
        scratch_delta = candidate - scratch
        envelope_delta = candidate - envelope
        scratch_deltas.append(scratch_delta)
        envelope_deltas.append(envelope_delta)
        dataset_results[dataset] = {
            "condition_mean_balanced_accuracy": dict(scores),
            "historical_opened_screen_floor_balanced_accuracy": (
                HISTORICAL_OPENED_SCREEN_FLOOR[dataset]
            ),
            "reference_envelope_sources": envelope_sources,
            "reference_envelope_balanced_accuracy": envelope,
            "deltas": {
                "candidate_minus_canonical_seeded_scratch": _core._delta(
                    candidate, scratch
                ),
                "candidate_minus_reference_envelope": _core._delta(
                    candidate, envelope
                ),
            },
        }

    scratch_macro_delta = sum(scratch_deltas) / len(scratch_deltas)
    envelope_macro_delta = sum(envelope_deltas) / len(envelope_deltas)
    worst_envelope_delta = min(envelope_deltas)
    positive_by_dataset = {
        dataset: scratch_deltas[index] > 0.0
        for index, dataset in enumerate(LOCKED_DATASETS)
    }
    lower = float(bootstrap["lower_bound_balanced_accuracy_delta"])
    checks = {
        "candidate_minus_scratch_equal_dataset_at_least_1pp": _core._check(
            observed=scratch_macro_delta,
            threshold=MIN_SCRATCH_EQUAL_DATASET_DELTA,
            operator=">=",
            passed=scratch_macro_delta >= MIN_SCRATCH_EQUAL_DATASET_DELTA,
        ),
        "candidate_minus_scratch_positive_on_both_datasets": {
            "observed_by_dataset": {
                dataset: _core._delta(
                    means[dataset][CANDIDATE_CONDITION],
                    means[dataset][TRANSFER_COMPARATOR],
                )
                for dataset in LOCKED_DATASETS
            },
            "pass_by_dataset": positive_by_dataset,
            "operator": ">",
            "threshold_balanced_accuracy_delta": 0.0,
            "threshold_percentage_points": 0.0,
            "pass": all(positive_by_dataset.values()),
        },
        "physionet_paired_subject_bootstrap_one_sided_95_lower_above_zero": {
            **_core._check(
                observed=lower,
                threshold=0.0,
                operator=">",
                passed=lower > 0.0,
            ),
            "repetitions": bootstrap["repetitions"],
            "seed": bootstrap["seed"],
        },
        "candidate_minus_reference_envelope_equal_dataset_at_least_0_5pp": (
            _core._check(
                observed=envelope_macro_delta,
                threshold=MIN_REFERENCE_ENVELOPE_EQUAL_DATASET_DELTA,
                operator=">=",
                passed=(
                    envelope_macro_delta
                    >= MIN_REFERENCE_ENVELOPE_EQUAL_DATASET_DELTA
                ),
            )
        ),
        "worst_dataset_reference_envelope_delta_at_least_minus_1pp": (
            _core._check(
                observed=worst_envelope_delta,
                threshold=MIN_WORST_DATASET_REFERENCE_ENVELOPE_DELTA,
                operator=">=",
                passed=(
                    worst_envelope_delta
                    >= MIN_WORST_DATASET_REFERENCE_ENVELOPE_DELTA
                ),
            )
        ),
    }
    condition_macros = {
        condition: sum(means[d][condition] for d in LOCKED_DATASETS)
        / len(LOCKED_DATASETS)
        for condition in LOCKED_CONDITIONS
    }
    envelope_macro = sum(
        float(dataset_results[d]["reference_envelope_balanced_accuracy"])
        for d in LOCKED_DATASETS
    ) / len(LOCKED_DATASETS)
    return {
        "schema": GATE_SCHEMA,
        "mode": MODE,
        "confirmation_access": False,
        "confirmatory_evidence": False,
        "analysis_policy": ANALYSIS_POLICY,
        "locked_contract": {
            "screening_grid": audit.locked_screen_grid(),
            "candidate_condition": CANDIDATE_CONDITION,
            "scratch_transfer_comparator": TRANSFER_COMPARATOR,
            "reference_envelope_conditions": list(REFERENCE_CONDITIONS),
            "historical_reference_floor": dict(HISTORICAL_OPENED_SCREEN_FLOOR),
            "reference_envelope_aggregation": (
                "maximum new reference or historical floor within dataset, "
                "then equal-dataset mean"
            ),
            "minimum_candidate_minus_scratch_equal_dataset_percentage_points": 1.0,
            "candidate_minus_scratch_must_be_positive_on_every_dataset": True,
            "minimum_candidate_minus_reference_envelope_equal_dataset_percentage_points": 0.5,
            "minimum_worst_dataset_reference_envelope_delta_percentage_points": -1.0,
        },
        "inputs": {
            "freshly_revalidated_screen_audit": manifest,
            "freshly_revalidated_screen_audit_sha256": manifest_sha256,
            "recomputed_screen_summary": screen_summary,
            "internally_computed_physionet_paired_subject_bootstrap": bootstrap,
        },
        "dataset_results": dataset_results,
        "equal_dataset_macro": {
            "condition_mean_balanced_accuracy": condition_macros,
            "reference_envelope_balanced_accuracy": envelope_macro,
            "deltas": {
                "candidate_minus_canonical_seeded_scratch": _core._delta(
                    condition_macros[CANDIDATE_CONDITION],
                    condition_macros[TRANSFER_COMPARATOR],
                ),
                "candidate_minus_reference_envelope": {
                    "balanced_accuracy": envelope_macro_delta,
                    "percentage_points": 100.0 * envelope_macro_delta,
                },
            },
        },
        "gate_checks": checks,
        "overall_pass": all(bool(check["pass"]) for check in checks.values()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cho-root", required=True, type=Path)
    parser.add_argument("--physionet-root", required=True, type=Path)
    return parser


def run(argv: Sequence[str] | None = None) -> dict[str, object]:
    arguments = build_parser().parse_args(argv)
    screen = audit.audit_locked_fbms_screen(
        cho_root=arguments.cho_root,
        physionet_root=arguments.physionet_root,
    )
    return evaluate_fbms_transfer_gate(screen)


def main() -> None:
    print(json.dumps(run(), indent=2, sort_keys=True, allow_nan=False))


__all__ = [
    "BOOTSTRAP_SCHEMA",
    "BOOTSTRAP_REPETITIONS",
    "BOOTSTRAP_SEED",
    "CANDIDATE_CONDITION",
    "GATE_SCHEMA",
    "HISTORICAL_OPENED_SCREEN_FLOOR",
    "LOCKED_CONDITIONS",
    "LOCKED_DATASETS",
    "REFERENCE_CONDITIONS",
    "TRANSFER_COMPARATOR",
    "evaluate_fbms_transfer_gate",
]


if __name__ == "__main__":  # pragma: no cover
    main()
