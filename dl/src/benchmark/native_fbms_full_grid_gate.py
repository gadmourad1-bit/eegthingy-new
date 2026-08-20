"""Prespecified GO/no-GO analysis for the complete CardinalFBMS full grid.

The gate accepts only a typed audit, reopens all 8,675 artifacts, recomputes
subject-unit scores, and computes its own paired bootstrap.  It accepts no
caller-supplied score, confidence bound, threshold, seed, or grid dimension.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from . import native_fbms_full_grid_audit as audit
from . import native_fbms_transfer as transfer
from . import native_fbms_transfer_gate as screen_gate
from . import native_transfer_audit as artifact_audit
from . import native_transfer_gate as _core


GATE_SCHEMA = "eeg-mi-native-cardinal-fbms-full-grid-gate-v1"
BOOTSTRAP_SCHEMA = "eeg-mi-native-cardinal-fbms-full-grid-paired-bootstrap-v1"
PRIMARY_COMPARATOR_BOOTSTRAP_SCHEMA = (
    "eeg-mi-native-cardinal-fbms-full-grid-checkpoint-matched-bootstrap-v1"
)
MODE = "development"
ANALYSIS_POLICY = (
    "locked_fbms_full_development_subject_unit_analysis_not_confirmation_evidence"
)

LOCKED_DATASETS = audit.LOCKED_DATASETS
LOCKED_CONDITIONS = audit.LOCKED_CONDITIONS
CANDIDATE_CONDITION = transfer.PRETRAINED_CARDINAL_FBMS
TRANSFER_COMPARATOR = transfer.SCRATCH_CARDINAL_FBMS_CANONICAL
PRIMARY_CHECKPOINT_MATCHED_COMPARATOR = (
    transfer.PRETRAINED_INDEXED_FBMSNET_SPLINE
)
PRIMARY_COMPARATOR_SCOPE = (
    "development-only checkpoint-matched indexed projection control; "
    "not an author-faithful FBMSNet reproduction"
)
REFERENCE_CONDITIONS: tuple[str, ...] = (
    transfer.SCRATCH_CARDINAL_FBMS_NATIVE,
    transfer.SCRATCH_FBMSNET_NATIVE,
    transfer.PRETRAINED_INDEXED_FBMSNET_SPLINE,
)
HISTORICAL_REFERENCE_NAME = screen_gate.HISTORICAL_REFERENCE_NAME
HISTORICAL_OPENED_SCREEN_FLOOR = dict(
    screen_gate.HISTORICAL_OPENED_SCREEN_FLOOR
)

MIN_SCRATCH_EQUAL_DATASET_DELTA = 0.01
MIN_REFERENCE_ENVELOPE_EQUAL_DATASET_DELTA = 0.005
MIN_WORST_DATASET_REFERENCE_ENVELOPE_DELTA = -0.01
BOOTSTRAP_REPETITIONS = 200_000
BOOTSTRAP_SEED = 20_260_720
PRIMARY_COMPARATOR_BOOTSTRAP_SEED = 20_260_721
BOOTSTRAP_BATCH_SIZE = 4096
BOOTSTRAP_CONFIDENCE_LEVEL = 0.95


def _expected_keys(dataset: str) -> tuple[artifact_audit.TransferRecordKey, ...]:
    return artifact_audit.development_transfer_grid(
        dataset=dataset,
        subjects=audit.LOCKED_SUBJECTS[dataset],
        folds=audit.LOCKED_FOLDS[dataset],
        seeds=audit.LOCKED_SEEDS,
        conditions=audit.LOCKED_CONDITIONS,
        contract=audit.AUDIT_CONTRACT,
    )


def _revalidate_complete_grid(
    full_grid: audit.LockedFBMSFullGridAudit,
) -> audit.LockedFBMSFullGridAudit:
    if not isinstance(full_grid, audit.LockedFBMSFullGridAudit):
        raise TypeError("full_grid must be a LockedFBMSFullGridAudit")
    directories = {
        audit.CHO2017: full_grid.cho2017,
        audit.PHYSIONET_MI: full_grid.physionet_mi,
    }
    for dataset, directory in directories.items():
        expected = _expected_keys(dataset)
        if directory.contract != audit.AUDIT_CONTRACT:
            raise ValueError(f"{dataset} audit uses a different full-grid contract")
        if directory.name_template != audit.FULL_RECORD_NAME_TEMPLATE:
            raise ValueError(f"{dataset} audit uses a nonlocked record name template")
        if directory.expected_keys != expected:
            raise ValueError(f"{dataset} audit differs from the locked full grid")
        if not directory.complete or directory.missing_keys:
            raise ValueError(f"{dataset} full-grid audit is incomplete")
        if tuple(record.key for record in directory.records) != expected:
            raise ValueError(f"{dataset} records differ from the locked full grid")
    if len(full_grid.records) != audit.EXPECTED_RECORD_COUNT:
        raise ValueError("full grid does not contain exactly 8,675 records")

    refreshed = audit.audit_locked_fbms_full_grid(
        cho_root=full_grid.cho2017.root,
        physionet_root=full_grid.physionet_mi.root,
    )
    if not refreshed.complete or len(refreshed.records) != audit.EXPECTED_RECORD_COUNT:
        raise ValueError("freshly revalidated FBMS full grid is incomplete")
    return refreshed


def _validated_means(
    summary: Mapping[str, object],
) -> dict[str, dict[str, float]]:
    expected_fields = {
        "schema",
        "mode",
        "confirmation_access",
        "confirmatory_evidence",
        "analysis_policy",
        "inferential_statistics",
        "scientific_unit",
        "aggregation",
        "complete_grid",
        "locked_grid",
        "expected_record_count",
        "validated_record_count",
        "dataset_condition_mean_balanced_accuracy",
        "equal_dataset_condition_mean_balanced_accuracy",
        "datasets",
    }
    _core._require_exact_keys(summary, frozenset(expected_fields), name="full_summary")
    exact = {
        "schema": audit.SUMMARY_SCHEMA,
        "mode": MODE,
        "confirmation_access": False,
        "confirmatory_evidence": False,
        "analysis_policy": audit.SUMMARY_ANALYSIS_POLICY,
        "inferential_statistics": False,
        "scientific_unit": "subject",
        "complete_grid": True,
        "expected_record_count": audit.EXPECTED_RECORD_COUNT,
        "validated_record_count": audit.EXPECTED_RECORD_COUNT,
    }
    for name, expected in exact.items():
        if summary[name] != expected or (
            isinstance(expected, bool) and not isinstance(summary[name], bool)
        ):
            raise ValueError(f"full_summary.{name} must equal {expected!r}")
    if summary["locked_grid"] != audit.locked_full_grid():
        raise ValueError("full summary does not represent the locked grid")
    expected_aggregation = {
        "folds": (
            "concatenate disjoint outer-test predictions within "
            "dataset/subject/seed/condition before scoring"
        ),
        "fold_coverage": "every cached target row exactly once",
        "seeds": "arithmetic mean of OOF seed scores within subject/condition",
        "subjects": "equal-subject arithmetic mean within dataset/condition",
        "datasets": "equal-dataset arithmetic mean",
        "metric": "binary balanced accuracy",
        "pseudoreplicates": "folds and seeds are never inferential units",
    }
    if summary["aggregation"] != expected_aggregation:
        raise ValueError("full summary aggregation differs from subject-unit policy")

    means = summary["dataset_condition_mean_balanced_accuracy"]
    if not isinstance(means, Mapping):
        raise ValueError("full summary means must be a mapping")
    _core._require_exact_keys(means, frozenset(LOCKED_DATASETS), name="means")
    result: dict[str, dict[str, float]] = {}
    for dataset in LOCKED_DATASETS:
        values = means[dataset]
        if not isinstance(values, Mapping):
            raise ValueError(f"means for {dataset} must be a mapping")
        _core._require_exact_keys(
            values, frozenset(LOCKED_CONDITIONS), name=f"means for {dataset}"
        )
        result[dataset] = {
            condition: _core._balanced_accuracy(
                values[condition], name=f"{dataset}.{condition}"
            )
            for condition in LOCKED_CONDITIONS
        }
    reported_macro = summary["equal_dataset_condition_mean_balanced_accuracy"]
    if not isinstance(reported_macro, Mapping):
        raise ValueError("reported equal-dataset means must be a mapping")
    _core._require_exact_keys(
        reported_macro,
        frozenset(LOCKED_CONDITIONS),
        name="equal-dataset means",
    )
    for condition in LOCKED_CONDITIONS:
        recomputed = float(np.mean([result[d][condition] for d in LOCKED_DATASETS]))
        reported = _core._balanced_accuracy(
            reported_macro[condition], name=f"equal_dataset.{condition}"
        )
        if reported != recomputed:
            raise ValueError("reported equal-dataset mean differs from subject means")
    return result


def _subject_scores(
    summary: Mapping[str, object],
    dataset: str,
    condition: str,
) -> tuple[np.ndarray, np.ndarray]:
    datasets = summary.get("datasets")
    if not isinstance(datasets, Mapping):
        raise RuntimeError("full summary datasets are missing")
    dataset_summary = datasets.get(dataset)
    if not isinstance(dataset_summary, Mapping):
        raise RuntimeError(f"full summary dataset {dataset} is missing")
    conditions = dataset_summary.get("conditions")
    if not isinstance(conditions, Mapping):
        raise RuntimeError(f"full summary conditions for {dataset} are missing")
    condition_summary = conditions.get(condition)
    if not isinstance(condition_summary, Mapping):
        raise RuntimeError(f"full summary condition {condition} is missing")
    scores = condition_summary.get("subject_balanced_accuracy")
    if not isinstance(scores, Mapping):
        raise RuntimeError("full summary subject scores are missing")
    subjects = np.asarray(audit.LOCKED_SUBJECTS[dataset], dtype=np.int64)
    if set(scores) != {str(subject) for subject in subjects.tolist()}:
        raise RuntimeError(f"subject scores differ from locked {dataset} cohort")
    values = np.asarray(
        [
            _core._balanced_accuracy(
                scores[str(subject)], name=f"{dataset}.{condition}.S{subject}"
            )
            for subject in subjects.tolist()
        ],
        dtype=np.float64,
    )
    return subjects, values


def _paired_subject_bootstrap(
    summary: Mapping[str, object],
    *,
    comparator_condition: str = TRANSFER_COMPARATOR,
    bootstrap_schema: str = BOOTSTRAP_SCHEMA,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, object]:
    differences: dict[str, np.ndarray] = {}
    subjects: dict[str, np.ndarray] = {}
    score_hashes: dict[str, dict[str, str]] = {}
    for dataset in LOCKED_DATASETS:
        candidate_subjects, candidate = _subject_scores(
            summary, dataset, CANDIDATE_CONDITION
        )
        comparator_subjects, comparator = _subject_scores(
            summary, dataset, comparator_condition
        )
        if not np.array_equal(candidate_subjects, comparator_subjects):
            raise RuntimeError(f"paired subject identities differ for {dataset}")
        subjects[dataset] = candidate_subjects
        differences[dataset] = candidate - comparator
        score_hashes[dataset] = {
            "candidate_subject_scores_sha256": artifact_audit._array_sha256(candidate),
            "comparator_subject_scores_sha256": artifact_audit._array_sha256(comparator),
            "paired_subject_differences_sha256": artifact_audit._array_sha256(
                differences[dataset]
            ),
        }

    distributions = {
        dataset: np.empty(BOOTSTRAP_REPETITIONS, dtype=np.float64)
        for dataset in LOCKED_DATASETS
    }
    distributions["equal_dataset_macro"] = np.empty(
        BOOTSTRAP_REPETITIONS, dtype=np.float64
    )
    generator = np.random.default_rng(bootstrap_seed)
    offset = 0
    while offset < BOOTSTRAP_REPETITIONS:
        count = min(BOOTSTRAP_BATCH_SIZE, BOOTSTRAP_REPETITIONS - offset)
        batch_means: list[np.ndarray] = []
        for dataset in LOCKED_DATASETS:
            diff = differences[dataset]
            indices = generator.integers(
                0, len(diff), size=(count, len(diff)), dtype=np.int64
            )
            values = diff[indices].mean(axis=1)
            distributions[dataset][offset : offset + count] = values
            batch_means.append(values)
        distributions["equal_dataset_macro"][offset : offset + count] = np.mean(
            np.stack(batch_means, axis=0), axis=0
        )
        offset += count

    def interval(values: np.ndarray) -> dict[str, float]:
        alpha = 1.0 - BOOTSTRAP_CONFIDENCE_LEVEL
        return {
            "one_sided_95_lower": float(
                np.quantile(values, alpha, method="linear")
            ),
            "two_sided_95_lower": float(
                np.quantile(values, alpha / 2.0, method="linear")
            ),
            "two_sided_95_upper": float(
                np.quantile(values, 1.0 - alpha / 2.0, method="linear")
            ),
        }

    dataset_results = {
        dataset: {
            "subject_count": len(subjects[dataset]),
            "subjects": subjects[dataset].tolist(),
            "observed_mean_balanced_accuracy_delta": float(
                differences[dataset].mean()
            ),
            **interval(distributions[dataset]),
            **score_hashes[dataset],
        }
        for dataset in LOCKED_DATASETS
    }
    macro_observed = float(
        np.mean([differences[dataset].mean() for dataset in LOCKED_DATASETS])
    )
    return {
        "schema": bootstrap_schema,
        "mode": MODE,
        "candidate_condition": CANDIDATE_CONDITION,
        "comparator_condition": comparator_condition,
        "resampling_unit": "subject",
        "paired": True,
        "fold_handling": "OOF predictions concatenated before each seed score",
        "seed_handling": "seed scores averaged within subject before resampling",
        "dataset_handling": "subjects resampled within dataset; datasets weighted equally",
        "method": "stratified_nonparametric_paired_percentile_bootstrap",
        "confidence_level": BOOTSTRAP_CONFIDENCE_LEVEL,
        "quantile_method": "linear",
        "repetitions": BOOTSTRAP_REPETITIONS,
        "seed": bootstrap_seed,
        "datasets": dataset_results,
        "equal_dataset_macro": {
            "observed_mean_balanced_accuracy_delta": macro_observed,
            **interval(distributions["equal_dataset_macro"]),
        },
    }


def evaluate_fbms_full_grid_gate(
    full_grid: audit.LockedFBMSFullGridAudit,
) -> dict[str, object]:
    refreshed = _revalidate_complete_grid(full_grid)
    summary = audit.descriptive_full_grid_summary(refreshed)
    means = _validated_means(summary)
    bootstrap = _paired_subject_bootstrap(summary)
    primary_comparator_bootstrap = _paired_subject_bootstrap(
        summary,
        comparator_condition=PRIMARY_CHECKPOINT_MATCHED_COMPARATOR,
        bootstrap_schema=PRIMARY_COMPARATOR_BOOTSTRAP_SCHEMA,
        bootstrap_seed=PRIMARY_COMPARATOR_BOOTSTRAP_SEED,
    )
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
        references = {
            condition: scores[condition] for condition in REFERENCE_CONDITIONS
        }
        references[HISTORICAL_REFERENCE_NAME] = HISTORICAL_OPENED_SCREEN_FLOOR[
            dataset
        ]
        envelope = max(references.values())
        scratch_delta = candidate - scratch
        envelope_delta = candidate - envelope
        scratch_deltas.append(scratch_delta)
        envelope_deltas.append(envelope_delta)
        dataset_results[dataset] = {
            "condition_mean_balanced_accuracy": dict(scores),
            "historical_opened_screen_floor_balanced_accuracy": (
                HISTORICAL_OPENED_SCREEN_FLOOR[dataset]
            ),
            "reference_envelope_sources": [
                name for name, score in references.items() if score == envelope
            ],
            "reference_envelope_balanced_accuracy": envelope,
            "deltas": {
                "candidate_minus_canonical_seeded_scratch": _core._delta(
                    candidate, scratch
                ),
                "candidate_minus_reference_envelope": _core._delta(
                    candidate, envelope
                ),
                "candidate_minus_primary_checkpoint_matched_comparator": (
                    _core._delta(
                        candidate,
                        scores[PRIMARY_CHECKPOINT_MATCHED_COMPARATOR],
                    )
                ),
            },
            "primary_checkpoint_matched_comparison": {
                "comparator_condition": PRIMARY_CHECKPOINT_MATCHED_COMPARATOR,
                "scope": PRIMARY_COMPARATOR_SCOPE,
                **dict(primary_comparator_bootstrap["datasets"][dataset]),
            },
        }

    scratch_macro_delta = float(np.mean(scratch_deltas))
    envelope_macro_delta = float(np.mean(envelope_deltas))
    worst_envelope_delta = min(envelope_deltas)
    bootstrap_units = {
        **{
            dataset: float(
                bootstrap["datasets"][dataset]["one_sided_95_lower"]
            )
            for dataset in LOCKED_DATASETS
        },
        "equal_dataset_macro": float(
            bootstrap["equal_dataset_macro"]["one_sided_95_lower"]
        ),
    }
    checks = {
        "candidate_minus_scratch_equal_dataset_at_least_1pp": _core._check(
            observed=scratch_macro_delta,
            threshold=MIN_SCRATCH_EQUAL_DATASET_DELTA,
            operator=">=",
            passed=scratch_macro_delta >= MIN_SCRATCH_EQUAL_DATASET_DELTA,
        ),
        "candidate_minus_scratch_positive_on_every_dataset": {
            "observed_by_dataset": {
                dataset: _core._delta(
                    means[dataset][CANDIDATE_CONDITION],
                    means[dataset][TRANSFER_COMPARATOR],
                )
                for dataset in LOCKED_DATASETS
            },
            "operator": ">",
            "threshold_balanced_accuracy_delta": 0.0,
            "pass_by_dataset": {
                dataset: delta > 0.0
                for dataset, delta in zip(LOCKED_DATASETS, scratch_deltas)
            },
            "pass": all(delta > 0.0 for delta in scratch_deltas),
        },
        "paired_subject_bootstrap_one_sided_95_lower_above_zero_each_dataset_and_macro": {
            "observed_lower_bounds": bootstrap_units,
            "operator": ">",
            "threshold_balanced_accuracy_delta": 0.0,
            "pass_by_unit": {
                name: value > 0.0 for name, value in bootstrap_units.items()
            },
            "pass": all(value > 0.0 for value in bootstrap_units.values()),
        },
        "candidate_minus_primary_checkpoint_matched_comparator_equal_dataset_macro_one_sided_95_lower_above_zero": {
            "observed_equal_dataset_macro_one_sided_95_lower": float(
                primary_comparator_bootstrap["equal_dataset_macro"][
                    "one_sided_95_lower"
                ]
            ),
            "reported_dataset_intervals_are_gate_criteria": False,
            "operator": ">",
            "threshold_balanced_accuracy_delta": 0.0,
            "pass": float(
                primary_comparator_bootstrap["equal_dataset_macro"][
                    "one_sided_95_lower"
                ]
            )
            > 0.0,
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
        condition: float(np.mean([means[d][condition] for d in LOCKED_DATASETS]))
        for condition in LOCKED_CONDITIONS
    }
    envelope_macro = float(
        np.mean(
            [
                dataset_results[d]["reference_envelope_balanced_accuracy"]
                for d in LOCKED_DATASETS
            ]
        )
    )
    return {
        "schema": GATE_SCHEMA,
        "mode": MODE,
        "confirmation_access": False,
        "confirmatory_evidence": False,
        "analysis_policy": ANALYSIS_POLICY,
        "scientific_unit": "subject",
        "locked_contract": {
            "full_grid": audit.locked_full_grid(),
            "scientific_pins": audit.locked_scientific_pins(),
            "candidate_condition": CANDIDATE_CONDITION,
            "scratch_transfer_comparator": TRANSFER_COMPARATOR,
            "primary_checkpoint_matched_comparator": (
                PRIMARY_CHECKPOINT_MATCHED_COMPARATOR
            ),
            "primary_checkpoint_matched_comparator_scope": PRIMARY_COMPARATOR_SCOPE,
            "primary_comparator_bootstrap_seed": PRIMARY_COMPARATOR_BOOTSTRAP_SEED,
            "primary_comparator_equal_dataset_macro_one_sided_95_lower_must_be_positive": True,
            "reference_envelope_conditions": list(REFERENCE_CONDITIONS),
            "reference_envelope_role": "secondary_oracle_point_checks",
            "historical_reference_floor": dict(HISTORICAL_OPENED_SCREEN_FLOOR),
            "minimum_candidate_minus_scratch_equal_dataset_percentage_points": 1.0,
            "candidate_minus_scratch_must_be_positive_on_every_dataset": True,
            "paired_subject_lower_bound_must_be_positive_for_each_dataset_and_macro": True,
            "minimum_candidate_minus_reference_envelope_equal_dataset_percentage_points": 0.5,
            "minimum_worst_dataset_reference_envelope_delta_percentage_points": -1.0,
        },
        "inputs": {
            "freshly_revalidated_full_grid_audit": manifest,
            "freshly_revalidated_full_grid_audit_sha256": manifest_sha256,
            "recomputed_full_grid_summary": summary,
            "internally_computed_paired_subject_bootstrap": bootstrap,
            "internally_computed_primary_checkpoint_matched_paired_subject_bootstrap": (
                primary_comparator_bootstrap
            ),
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
                "candidate_minus_reference_envelope": _core._delta(
                    condition_macros[CANDIDATE_CONDITION], envelope_macro
                ),
                "candidate_minus_primary_checkpoint_matched_comparator": (
                    _core._delta(
                        condition_macros[CANDIDATE_CONDITION],
                        condition_macros[PRIMARY_CHECKPOINT_MATCHED_COMPARATOR],
                    )
                ),
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
    full_grid = audit.audit_locked_fbms_full_grid(
        cho_root=arguments.cho_root,
        physionet_root=arguments.physionet_root,
    )
    return evaluate_fbms_full_grid_gate(full_grid)


def main() -> None:
    print(json.dumps(run(), indent=2, sort_keys=True, allow_nan=False))


__all__ = [
    "BOOTSTRAP_REPETITIONS",
    "BOOTSTRAP_SCHEMA",
    "BOOTSTRAP_SEED",
    "CANDIDATE_CONDITION",
    "GATE_SCHEMA",
    "HISTORICAL_OPENED_SCREEN_FLOOR",
    "PRIMARY_CHECKPOINT_MATCHED_COMPARATOR",
    "PRIMARY_COMPARATOR_BOOTSTRAP_SCHEMA",
    "PRIMARY_COMPARATOR_BOOTSTRAP_SEED",
    "PRIMARY_COMPARATOR_SCOPE",
    "REFERENCE_CONDITIONS",
    "TRANSFER_COMPARATOR",
    "evaluate_fbms_full_grid_gate",
]


if __name__ == "__main__":  # pragma: no cover
    main()
