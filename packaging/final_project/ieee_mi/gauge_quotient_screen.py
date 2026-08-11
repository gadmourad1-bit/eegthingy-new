"""Read-only scratch screen specification for the gauge-quotient candidate.

This is deliberately not a formal runner.  It defines the two candidate-only
opened-development gates and maps every comparator cell to an already frozen
CHSD-era reference record.  It has no function that writes a plan, creates a
run directory, trains a model, or alters a reference record.  The command-line
entry point only prints the in-memory specification as JSON.

A later reviewed runner may consume this specification after the model and
gate are approved.  Until then, importing or executing this module cannot
create a plan or launch a score run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from .conditioned_screen import EXPECTED_REFERENCE_PLAN_SHA256
from .development_screen import CURATED_SPLITS, ScreenJob
from .gauge_quotient import MODEL_NAME
from .robustness_screen import (
    REFERENCE_MODELS,
    ROBUSTNESS_COHORTS,
    TRAIN_CONFIG,
)


SCREEN_SCHEMA: Final = "ieee-mi-gauge-quotient-read-only-screen-spec-v1"
OPENED17_STAGE: Final = "opened_17_subject_gate"
DISJOINT115_STAGE: Final = "disjoint_115_subject_gate"
STAGES: Final = (OPENED17_STAGE, DISJOINT115_STAGE)
EXPECTED_ROBUSTNESS_PLAN_SHA256: Final = (
    "1e4009ae67e7b11ae60c76c5d110ec49a27246422d2617c9c818199ca799b9ae"
)
MODEL_FACTORY: Final = (
    "ieee_mi.gauge_quotient:make_gauge_quotient_model"
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _freeze_json(value: Any) -> Any:
    """Recursively freeze a JSON value without changing its JSON meaning."""

    if isinstance(value, dict):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"value of type {type(value).__name__} is not JSON-safe")


def _bind_rule(
    value: Mapping[str, Any],
) -> tuple[Mapping[str, Any], str, str]:
    """Return an immutable rule plus its exact canonical JSON identity."""

    canonical = _canonical_json(value)
    reparsed = json.loads(canonical)
    digest = hashlib.sha256(canonical.encode("ascii")).hexdigest()
    return _freeze_json(reparsed), canonical, digest


STAGE_COHORTS: Mapping[
    str, tuple[tuple[str, tuple[int, ...]], ...]
] = MappingProxyType(
    {
        OPENED17_STAGE: tuple(
            (dataset, tuple(subjects))
            for dataset, subjects in CURATED_SPLITS
        ),
        DISJOINT115_STAGE: tuple(
            (dataset, tuple(subjects))
            for dataset, subjects in ROBUSTNESS_COHORTS
        ),
    }
)
STAGE_REFERENCE_PLAN_SHA256: Mapping[str, str] = MappingProxyType(
    {
        OPENED17_STAGE: EXPECTED_REFERENCE_PLAN_SHA256,
        DISJOINT115_STAGE: EXPECTED_ROBUSTNESS_PLAN_SHA256,
    }
)

(
    OPENED17_RULE,
    OPENED17_RULE_JSON,
    OPENED17_RULE_SHA256,
) = _bind_rule(
    {
        "primary_metric": "equal_dataset_balanced_accuracy",
        "primary_reference": "cardinal_fbc_micro_extended",
        "candidate_must_strictly_exceed_primary_reference": True,
        "minimum_nonnegative_dataset_deltas": 3,
        "maximum_dataset_deficit": 0.03,
        "minimum_strict_subject_win_rate": 9 / 17,
        "failure_action": "kill_candidate_before_disjoint_gate",
    }
)
(
    DISJOINT115_RULE,
    DISJOINT115_RULE_JSON,
    DISJOINT115_RULE_SHA256,
) = _bind_rule(
    {
        "primary_metric": "equal_dataset_balanced_accuracy",
        "primary_reference": "cardinal_fbc_micro_extended",
        "minimum_primary_reference_margin": 0.0025,
        "minimum_nonnegative_dataset_deltas": 3,
        "maximum_dataset_deficit": 0.02,
        "minimum_strict_subject_win_rate": 0.50,
        "paired_subject_bootstrap_ci": {
            "replicates": 10_000,
            "confidence": 0.95,
            "seed": 2_026_073_001,
            "lower_bound_must_strictly_exceed": 0.0,
        },
        "failure_action": "stop_candidate_promotion",
    }
)
STAGE_RULES: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {
        OPENED17_STAGE: OPENED17_RULE,
        DISJOINT115_STAGE: DISJOINT115_RULE,
    }
)
STAGE_RULE_JSON: Mapping[str, str] = MappingProxyType(
    {
        OPENED17_STAGE: OPENED17_RULE_JSON,
        DISJOINT115_STAGE: DISJOINT115_RULE_JSON,
    }
)
STAGE_RULE_SHA256: Mapping[str, str] = MappingProxyType(
    {
        OPENED17_STAGE: OPENED17_RULE_SHA256,
        DISJOINT115_STAGE: DISJOINT115_RULE_SHA256,
    }
)
TRAIN_CONFIG_JSON: Final = _canonical_json(TRAIN_CONFIG)
TRAIN_CONFIG_SHA256: Final = hashlib.sha256(
    TRAIN_CONFIG_JSON.encode("ascii")
).hexdigest()


def _cohorts(stage: str) -> tuple[tuple[str, tuple[int, ...]], ...]:
    try:
        return STAGE_COHORTS[stage]
    except KeyError as error:
        raise ValueError(f"unknown screen stage {stage!r}") from error


def candidate_jobs(stage: str) -> tuple[ScreenJob, ...]:
    """Return the candidate-only jobs for one prespecified gate."""

    jobs = tuple(
        ScreenJob(
            dataset=dataset,
            model=MODEL_NAME,
            subject=subject,
            fold=0,
            seed=7,
        )
        for dataset, subjects in _cohorts(stage)
        for subject in subjects
    )
    expected = 17 if stage == OPENED17_STAGE else 115
    if len(jobs) != expected:
        raise AssertionError(
            f"{stage} contains {len(jobs)} subjects rather than {expected}"
        )
    return jobs


def _record_relative_path(job: ScreenJob) -> Path:
    return (
        Path("records")
        / job.dataset
        / job.model
        / f"subject_{job.subject:03d}"
        / f"fold_{job.fold:02d}"
        / f"seed_{job.seed:03d}.json"
    )


def reference_record_paths(
    stage: str,
    reference_run_root: Path,
) -> tuple[Path, ...]:
    """Map comparator cells to frozen records without opening or changing them."""

    root = Path(reference_run_root)
    paths = tuple(
        root
        / _record_relative_path(
            ScreenJob(
                dataset=dataset,
                model=model,
                subject=subject,
                fold=0,
                seed=7,
            )
        )
        for model in REFERENCE_MODELS
        for dataset, subjects in _cohorts(stage)
        for subject in subjects
    )
    expected = (17 if stage == OPENED17_STAGE else 115) * len(
        REFERENCE_MODELS
    )
    if len(paths) != expected or len(paths) != len(set(paths)):
        raise AssertionError("reference record mapping is incomplete or duplicated")
    return paths


def screen_specification(
    stage: str,
    *,
    reference_run_root: Path,
) -> dict[str, Any]:
    """Return a JSON-safe, read-only candidate screen description."""

    root = Path(reference_run_root)
    cohorts = _cohorts(stage)
    candidates = candidate_jobs(stage)
    references = reference_record_paths(stage, root)
    return {
        "schema": SCREEN_SCHEMA,
        "stage": stage,
        "status": "specification_only_not_planned_not_run",
        "candidate": MODEL_NAME,
        "candidate_factory": MODEL_FACTORY,
        "candidate_job_count": len(candidates),
        "candidate_jobs": [job.identity() for job in candidates],
        "reference_models": list(REFERENCE_MODELS),
        "reference_run_root": str(root),
        "reference_plan_sha256": STAGE_REFERENCE_PLAN_SHA256[stage],
        "reference_record_count": len(references),
        "reference_records": [str(path) for path in references],
        "reference_access": "read_only_reuse_no_rerun_no_mutation",
        "cohorts": [
            {"dataset": dataset, "subjects": list(subjects)}
            for dataset, subjects in cohorts
        ],
        "fold": 0,
        "seed": 7,
        # Parse the bound canonical strings so callers receive independent
        # nested objects.  The module-level rules remain recursively immutable.
        "train_config": json.loads(TRAIN_CONFIG_JSON),
        "train_config_json": TRAIN_CONFIG_JSON,
        "train_config_sha256": TRAIN_CONFIG_SHA256,
        "decision_rule": json.loads(STAGE_RULE_JSON[stage]),
        "decision_rule_json": STAGE_RULE_JSON[stage],
        "decision_rule_sha256": STAGE_RULE_SHA256[stage],
        "opened_development_only": True,
        "confirmation_evidence": False,
        "creates_plan": False,
        "launches_work": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print the read-only gauge-quotient screen specification"
    )
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument(
        "--reference-run-root",
        required=True,
        type=Path,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    print(
        json.dumps(
            screen_specification(
                arguments.stage,
                reference_run_root=arguments.reference_run_root,
            ),
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DISJOINT115_RULE",
    "DISJOINT115_RULE_JSON",
    "DISJOINT115_RULE_SHA256",
    "DISJOINT115_STAGE",
    "EXPECTED_ROBUSTNESS_PLAN_SHA256",
    "MODEL_FACTORY",
    "OPENED17_RULE",
    "OPENED17_RULE_JSON",
    "OPENED17_RULE_SHA256",
    "OPENED17_STAGE",
    "SCREEN_SCHEMA",
    "STAGES",
    "STAGE_RULE_JSON",
    "STAGE_RULE_SHA256",
    "TRAIN_CONFIG_JSON",
    "TRAIN_CONFIG_SHA256",
    "candidate_jobs",
    "main",
    "reference_record_paths",
    "screen_specification",
]
