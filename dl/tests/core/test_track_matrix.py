from __future__ import annotations

from collections import Counter

import pytest

from benchmark.model_registry import BENCHMARK_DATASETS, MODEL_REGISTRY
from benchmark.track_matrix import (
    COMMON_FULL_GRID,
    DATASET_GRIDS,
    DESCRIPTOR_BY_ID,
    EXECUTION_BACKLOG,
    EXECUTION_DESCRIPTORS,
    MATRIX_SCHEMA,
    MISSING_ADAPTER,
    NATIVE_TARGET_GRIDS,
    REGISTRY_NA,
    SEPARATE_ELIGIBLE,
    TRACK_MATRIX,
    duplicate_callable_groups,
    matrix_manifest,
    proposed_cardinality,
    validate_track_matrix,
)


def _cell(stable_id: str, dataset: str):
    matches = [
        cell
        for cell in TRACK_MATRIX
        if cell.stable_id == stable_id and cell.dataset == dataset
    ]
    assert len(matches) == 1
    return matches[0]


def test_matrix_is_exact_registry_cartesian_product() -> None:
    validate_track_matrix()
    assert len(MODEL_REGISTRY) == 70
    assert len(EXECUTION_DESCRIPTORS) == 27
    assert len(DESCRIPTOR_BY_ID) == 27
    assert len(TRACK_MATRIX) == 350
    assert len({(cell.stable_id, cell.dataset) for cell in TRACK_MATRIX}) == 350
    assert [
        (cell.stable_id, cell.dataset) for cell in TRACK_MATRIX
    ] == [
        (record.stable_id, dataset)
        for record in MODEL_REGISTRY
        for dataset in BENCHMARK_DATASETS
    ]


def test_status_partition_and_registry_na_reasons_are_exact() -> None:
    assert Counter(cell.status for cell in TRACK_MATRIX) == {
        COMMON_FULL_GRID: 215,
        MISSING_ADAPTER: 60,
        REGISTRY_NA: 47,
        SEPARATE_ELIGIBLE: 28,
    }
    records = {record.stable_id: record for record in MODEL_REGISTRY}
    for cell in TRACK_MATRIX:
        eligibility = records[cell.stable_id].eligibility_for(cell.dataset)
        assert cell.registry_eligible is eligibility.eligible
        if cell.status == REGISTRY_NA:
            assert cell.reason == eligibility.reason
            assert cell.atomic_jobs == 0
            assert cell.result_records == 0
        if cell.status == COMMON_FULL_GRID:
            assert cell.track == "common"
        if cell.status == SEPARATE_ELIGIBLE:
            assert cell.track != "common"
            assert cell.adapter_callable


def test_common_grid_cardinality_is_2240_per_model() -> None:
    expected_by_dataset = {
        "local_exp4": 40,
        "bnci2014_001": 45,
        "bnci2014_004": 45,
        "cho2017": 1300,
        "physionet_mi": 810,
    }
    assert {
        dataset: grid.stochastic_jobs for dataset, grid in DATASET_GRIDS.items()
    } == expected_by_dataset
    for stable_id in (
        record.stable_id for record in MODEL_REGISTRY if record.track == "common"
    ):
        cells = [cell for cell in TRACK_MATRIX if cell.stable_id == stable_id]
        assert sum(cell.atomic_jobs for cell in cells) == 2240
        assert sum(cell.result_records for cell in cells) == 2240


def test_binary_procedure_and_native_target_cardinalities() -> None:
    standard = {
        dataset: proposed_cardinality("architecture.cameo", dataset).atomic_jobs
        for dataset in ("local_exp4", "bnci2014_004", "cho2017", "physionet_mi")
    }
    assert standard == {
        "local_exp4": 40,
        "bnci2014_004": 45,
        "cho2017": 1300,
        "physionet_mi": 810,
    }
    assert NATIVE_TARGET_GRIDS["cho2017"].subjects == tuple(range(16, 53))
    assert proposed_cardinality(
        "procedure.pretrained_cardinal_fbms", "cho2017"
    ).atomic_jobs == 925
    assert proposed_cardinality(
        "procedure.pretrained_cardinal_fbms", "physionet_mi"
    ).atomic_jobs == 810
    assert proposed_cardinality(
        "architecture.cardinal_spline_dual_view", "cho2017"
    ).atomic_jobs == 925
    assert proposed_cardinality(
        "architecture.hemi_q_field", "bnci2014_004"
    ).atomic_jobs == 45


def test_deterministic_control_cardinality_has_no_fake_seed_dimension() -> None:
    expected = {
        "local_exp4": 8,
        "bnci2014_001": 9,
        "bnci2014_004": 9,
        "cho2017": 260,
        "physionet_mi": 162,
    }
    for dataset, count in expected.items():
        cardinality = proposed_cardinality("control.riemann", dataset)
        assert (cardinality.atomic_jobs, cardinality.result_records) == (
            count,
            count,
        )
        assert cardinality.seeds == ()
        assert "no synthetic seed-identity records" in cardinality.seed_policy


def test_proposed_cardinality_rejects_na_unknown_and_common_pairs() -> None:
    with pytest.raises(ValueError, match="registry-ineligible"):
        proposed_cardinality("architecture.cameo", "bnci2014_001")
    with pytest.raises(ValueError, match="registry-ineligible"):
        proposed_cardinality("architecture.hemi_q_field", "local_exp4")
    with pytest.raises(KeyError, match="unknown registry stable_id"):
        proposed_cardinality("architecture.does_not_exist", "local_exp4")
    with pytest.raises(KeyError, match="unknown benchmark dataset"):
        proposed_cardinality("architecture.cameo", "not_a_dataset")
    with pytest.raises(ValueError, match="common-grid entry"):
        proposed_cardinality("common.eegnet", "local_exp4")


def test_only_proven_current_adapters_are_marked_ready() -> None:
    ready = {
        (cell.stable_id, cell.dataset)
        for cell in TRACK_MATRIX
        if cell.status == SEPARATE_ELIGIBLE
    }
    expected_local = {
        ("architecture.cameo", "local_exp4"),
        ("architecture.hemiparity", "local_exp4"),
        ("architecture.parity_fuse", "local_exp4"),
        ("architecture.orbit_v3", "local_exp4"),
        ("architecture.hemi_q_field", "bnci2014_004"),
    }
    expected_controls = {
        (stable_id, dataset)
        for stable_id in (
            "control.riemann",
            "control.tangent_anchor",
            "control.ea_fbcsp",
        )
        for dataset in BENCHMARK_DATASETS
    }
    transfer_ids = {
        descriptor.stable_id
        for descriptor in EXECUTION_DESCRIPTORS
        if descriptor.stable_id.startswith("procedure.")
        and descriptor.component_callable == "benchmark.native_transfer.run_record"
    }
    expected_transfer = {
        (stable_id, dataset)
        for stable_id in transfer_ids
        for dataset in ("cho2017", "physionet_mi")
    }
    assert ready == expected_local | expected_transfer | expected_controls
    assert _cell("reference.tcformer", "local_exp4").status == MISSING_ADAPTER
    assert (
        _cell("architecture.hemi_q_field", "bnci2014_004").status
        == SEPARATE_ELIGIBLE
    )
    assert (
        _cell("architecture.hemi_q_field", "bnci2014_004").adapter_callable
        == "benchmark.hemiq_v2_grid.execute_job"
    )
    assert (
        _cell("architecture.cardinal_spline_dual_view", "cho2017").status
        == MISSING_ADAPTER
    )
    retired_fbms = _cell("procedure.pretrained_cardinal_fbms", "cho2017")
    assert retired_fbms.status == MISSING_ADAPTER
    assert retired_fbms.adapter_callable is None
    assert "writer was retired" in retired_fbms.reason


def test_ready_paths_are_outer_cache_bound_callables() -> None:
    for stable_id in (
        "architecture.cameo",
        "architecture.hemiparity",
        "architecture.parity_fuse",
        "architecture.orbit_v3",
    ):
        cell = _cell(stable_id, "local_exp4")
        assert (
            cell.adapter_callable
            == "benchmark.local_outer_refit_benchmark.run_benchmark"
        )
        assert cell.adapter_callable != "benchmark.local_outer_refit_benchmark.run_one"
    for stable_id in (
        "control.riemann",
        "control.tangent_anchor",
        "control.ea_fbcsp",
    ):
        for dataset in BENCHMARK_DATASETS:
            cell = _cell(stable_id, dataset)
            assert cell.adapter_callable == "benchmark.control_grid.main"
            assert cell.seeds == ()
            assert cell.atomic_jobs == cell.result_records
    hemiq = _cell("architecture.hemi_q_field", "bnci2014_004")
    assert hemiq.adapter_callable == "benchmark.hemiq_v2_grid.execute_job"
    assert hemiq.atomic_jobs == hemiq.result_records == 45


def test_registry_identity_component_and_adapter_are_not_conflated() -> None:
    records = {record.stable_id: record for record in MODEL_REGISTRY}
    common = _cell("common.eegnet", "local_exp4")
    assert common.registry_implementation == records["common.eegnet"].implementation
    assert common.component_callable is None
    assert common.adapter_callable == "benchmark.full_grid.execute_benchmark_job"

    local_procedure = _cell("architecture.cameo", "local_exp4")
    assert (
        local_procedure.registry_implementation
        == records["architecture.cameo"].implementation
    )
    assert local_procedure.component_callable == "benchmark.research.cameo_net.CAMEOClassifier.fit"
    assert (
        local_procedure.adapter_callable
        == "benchmark.local_outer_refit_benchmark.run_benchmark"
    )

    control = _cell("control.riemann", "bnci2014_001")
    assert (
        control.component_callable
        == "benchmark.research.baselines.RiemannianTangentLogistic.fit"
    )
    assert control.adapter_callable == "benchmark.control_grid.main"


def test_harmonized_profile_is_not_substituted_for_native_transfer() -> None:
    for stable_id in (
        "procedure.pretrained_cardinal_fbc",
        "procedure.pretrained_cardinal_fbms",
        "architecture.cardinal_spline_dual_view",
    ):
        cell = _cell(stable_id, "cho2017")
        assert cell.harmonized_v2_safe is False
        assert cell.cache_compatibility == "native_v2_required"
    assert _cell("architecture.cameo", "local_exp4").harmonized_v2_safe is True
    assert _cell("control.riemann", "bnci2014_001").harmonized_v2_safe is True


def test_duplicate_callables_are_explicitly_visible() -> None:
    groups = duplicate_callable_groups()
    assert groups["benchmark.research.filterbank_net.FilterBankSPDClassifier.fit"] == (
        "architecture.geoadaptnet_fb",
        "architecture.geoadaptnet_fbsp",
    )
    assert groups["benchmark.research.orbit_transport_net.OrbitTransportClassifier.fit"] == (
        "architecture.orbit_v1",
        "architecture.orbit_v2",
        "architecture.orbit_v3",
        "architecture.orbit_v4",
        "architecture.orbit_v5",
    )
    assert len(groups["benchmark.native_transfer.run_record"]) == 4
    assert len(groups["benchmark.native_fbms_transfer.run_record"]) == 5


def test_manifest_is_json_ready_and_backlog_is_ordered() -> None:
    manifest = matrix_manifest()
    assert manifest["schema"] == MATRIX_SCHEMA
    assert MATRIX_SCHEMA.endswith("-v2")
    assert manifest["shape"] == [70, 5]
    assert manifest["cell_count"] == 350
    assert len(manifest["cells"]) == 350
    assert len(manifest["non_common_descriptors"]) == 27
    assert manifest["status_counts"] == {
        COMMON_FULL_GRID: 215,
        MISSING_ADAPTER: 60,
        REGISTRY_NA: 47,
        SEPARATE_ELIGIBLE: 28,
    }
    assert manifest["proposed_cardinality_totals"] == {
        "common": {
            "atomic_execution_units": 96_320,
            "result_records": 96_320,
        },
        "non_common": {
            "atomic_execution_units": 47_274,
            "result_records": 47_274,
        },
        "all": {
            "atomic_execution_units": 143_594,
            "result_records": 143_594,
        },
        "by_status": {
            COMMON_FULL_GRID: {
                "atomic_execution_units": 96_320,
                "result_records": 96_320,
            },
                MISSING_ADAPTER: {
                    "atomic_execution_units": 38_785,
                    "result_records": 38_785,
            },
            REGISTRY_NA: {
                "atomic_execution_units": 0,
                "result_records": 0,
            },
                SEPARATE_ELIGIBLE: {
                    "atomic_execution_units": 8_489,
                    "result_records": 8_489,
            },
        },
    }
    for descriptor in manifest["non_common_descriptors"]:
        assert "registry_implementation" in descriptor
        assert "component_callable" in descriptor
        assert "implementation_path" not in descriptor
        assert "callable_path" not in descriptor
    for cell in manifest["cells"]:
        assert "registry_implementation" in cell
        assert "component_callable" in cell
        assert "adapter_callable" in cell
        assert "callable_path" not in cell
        assert "adapter_path" not in cell
    assert [item.priority for item in EXECUTION_BACKLOG] == list(
        range(1, len(EXECUTION_BACKLOG) + 1)
    )
