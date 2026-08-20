from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import pytest

from benchmark.model_registry import (
    BENCHMARK_DATASETS,
    COMMON_RECORDS,
    COMMON_ROSTER_NAMES,
    IN_HOUSE_COMMON_RECORDS,
    MODEL_REGISTRY,
    PUBLIC_COMMON_RECORDS,
    DatasetEligibility,
    record_by_id,
    records_for_track,
    validate_registry,
)


def _frozen_tournament_roster() -> tuple[str, ...]:
    """Read the frozen roster without importing the Torch experiment runner."""

    runner_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "benchmark"
        / "local_model_tournament.py"
    )
    module = ast.parse(runner_path.read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            if any(
                isinstance(target, ast.Name) and target.id == "ARCHITECTURES"
                for target in targets
            ):
                assert node.value is not None
                roster = ast.literal_eval(node.value)
                assert isinstance(roster, tuple)
                assert all(isinstance(name, str) for name in roster)
                return roster
    raise AssertionError("ARCHITECTURES was not found in local_model_tournament.py")


def test_common_registry_exactly_covers_frozen_43_model_roster() -> None:
    architectures = _frozen_tournament_roster()
    assert len(COMMON_RECORDS) == 43
    assert COMMON_ROSTER_NAMES == architectures
    assert set(COMMON_ROSTER_NAMES) == set(architectures)


def test_common_registry_has_declared_ownership_split() -> None:
    assert len(IN_HOUSE_COMMON_RECORDS) == 28
    assert len(PUBLIC_COMMON_RECORDS) == 15
    assert all(record.ownership == "in_house" for record in IN_HOUSE_COMMON_RECORDS)
    assert all(record.ownership.startswith("public") for record in PUBLIC_COMMON_RECORDS)


def test_registry_ids_and_dataset_decisions_are_complete() -> None:
    stable_ids = [record.stable_id for record in MODEL_REGISTRY]
    assert len(stable_ids) == len(set(stable_ids))
    for record in MODEL_REGISTRY:
        assert tuple(
            decision.dataset for decision in record.dataset_eligibility
        ) == BENCHMARK_DATASETS
        assert record.documentation_link.startswith("docs/MODEL_REGISTRY.md#")
        for decision in record.dataset_eligibility:
            assert (decision.reason is None) == decision.eligible


def test_public_common_records_have_traceable_provenance() -> None:
    for record in PUBLIC_COMMON_RECORDS:
        assert record.citation
        assert record.citation_url
        assert record.source_url
        assert record.source_pin
        assert record.license != "unknown/not recorded"


def test_lookup_and_track_helpers_do_not_silently_fall_back() -> None:
    assert record_by_id("common.tcformer").common_roster_name == "tcformer"
    assert all(record.track == "control" for record in records_for_track("control"))
    with pytest.raises(KeyError):
        record_by_id("not-a-model")
    with pytest.raises(ValueError, match="unknown track"):
        records_for_track("not-a-track")


def test_validate_registry_rejects_duplicate_ids() -> None:
    validate_registry(MODEL_REGISTRY)
    with pytest.raises(ValueError, match="duplicate stable model IDs"):
        validate_registry((MODEL_REGISTRY[0], MODEL_REGISTRY[0]))


def test_validate_registry_requires_na_reason() -> None:
    record = MODEL_REGISTRY[0]
    broken_decisions = tuple(
        DatasetEligibility(decision.dataset, False, None)
        if index == 0
        else decision
        for index, decision in enumerate(record.dataset_eligibility)
    )
    broken = replace(record, dataset_eligibility=broken_decisions)
    with pytest.raises(ValueError, match="must carry an N/A reason"):
        validate_registry((broken,))
