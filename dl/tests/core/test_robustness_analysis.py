from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmark import robustness_analysis
from benchmark.robustness_analysis import (
    ANALYSIS_SCHEMA,
    RobustnessAnalysisError,
    _decision,
    _paired_comparisons,
    analyze_and_write,
    analyze_run,
    validate_analysis_payload,
)
from benchmark.robustness_screen import (
    CANDIDATE_MODEL,
    EXPECTED_JOB_COUNT,
    EXPECTED_SUBJECT_COUNT,
    HISTORICAL_V3_DECISION,
    MODELS,
    REFERENCE_MODELS,
    SOURCE_FILES,
    _record_path,
    robustness_jobs,
)
from tests.benchmark.test_robustness_screen import (
    _digest,
    environment_identity,
    write_complete_run,
)


@pytest.fixture(autouse=True)
def _synthetic_analysis_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Match the synthetic plan without weakening production runtime checks."""

    monkeypatch.setattr(
        robustness_analysis,
        "source_identity",
        lambda: {relative: _digest(relative) for relative in SOURCE_FILES},
    )
    monkeypatch.setattr(
        robustness_analysis,
        "robustness_environment_identity",
        environment_identity,
    )


def _overall_scores(
    *,
    candidate: float = 0.750,
    cardinal: float = 0.759,
    fbcnet: float = 0.751,
    tcformer: float = 0.749,
) -> list[dict[str, object]]:
    scores = {
        CANDIDATE_MODEL: candidate,
        "cardinal_fbc_micro_extended": cardinal,
        "fbcnet": fbcnet,
        "tcformer": tcformer,
    }
    return [
        {
            "model": model,
            "equal_dataset_balanced_accuracy": scores[model],
        }
        for model in MODELS
    ]


def _paired_summaries(
    *,
    nonnegative_datasets: int = 3,
    minimum_dataset_delta: float = -0.05,
    paired_subject_win_rate: float = 0.45,
) -> list[dict[str, object]]:
    return [
        {
            "scope": "all_subjects_and_equal_dataset",
            "candidate": CANDIDATE_MODEL,
            "reference": reference,
            "nonnegative_datasets": nonnegative_datasets,
            "minimum_dataset_delta": minimum_dataset_delta,
            "paired_subject_win_rate": paired_subject_win_rate,
            "equal_dataset_balanced_accuracy_delta": 0.0,
        }
        for reference in REFERENCE_MODELS
    ]


def test_gate_passes_only_when_every_prespecified_condition_passes() -> None:
    decision = _decision(_overall_scores(), _paired_summaries())
    assert decision["decision"] == "pass_robustness_gate"
    assert decision["passed"] is True
    assert decision["within_one_percentage_point_gate_passed"] is True
    assert decision["strictly_above_tcformer_gate_passed"] is True
    assert all(item["passed"] for item in decision["reference_diagnostics"].values())


def test_gate_exact_boundaries_pass_and_just_outside_fail() -> None:
    at_boundary = _decision(
        _overall_scores(cardinal=0.760),
        _paired_summaries(
            minimum_dataset_delta=-0.05,
            paired_subject_win_rate=0.45,
        ),
    )
    assert at_boundary["passed"] is True
    outside_gap = _decision(
        _overall_scores(cardinal=0.760000000002),
        _paired_summaries(),
    )
    assert outside_gap["within_one_percentage_point_gate_passed"] is False
    outside_deficit = _decision(
        _overall_scores(),
        _paired_summaries(minimum_dataset_delta=-0.050000000002),
    )
    assert all(
        not row["dataset_deficit_gate_passed"]
        for row in outside_deficit["reference_diagnostics"].values()
    )
    outside_win_rate = _decision(
        _overall_scores(),
        _paired_summaries(paired_subject_win_rate=0.449999999998),
    )
    assert all(
        not row["paired_subject_win_rate_gate_passed"]
        for row in outside_win_rate["reference_diagnostics"].values()
    )


def test_dataset_nonnegative_count_uses_frozen_tolerance() -> None:
    rows = []
    for job in robustness_jobs():
        score = 0.70
        if job.model != CANDIDATE_MODEL:
            score += 5e-13
        rows.append(
            {
                **job.identity(),
                "balanced_accuracy": score,
                "accuracy": score,
            }
        )
    summaries = _paired_comparisons(rows)
    overall = [
        row
        for row in summaries
        if row["scope"] == "all_subjects_and_equal_dataset"
    ]
    assert {int(row["nonnegative_datasets"]) for row in overall} == {5}


@pytest.mark.parametrize(
    ("overall", "pairs", "failed_field"),
    (
        (
            _overall_scores(cardinal=0.761),
            _paired_summaries(),
            "within_one_percentage_point_gate_passed",
        ),
        (
            _overall_scores(tcformer=0.750),
            _paired_summaries(),
            "strictly_above_tcformer_gate_passed",
        ),
        (
            _overall_scores(),
            _paired_summaries(nonnegative_datasets=2),
            "nonnegative_dataset_gate_passed",
        ),
        (
            _overall_scores(),
            _paired_summaries(minimum_dataset_delta=-0.050001),
            "dataset_deficit_gate_passed",
        ),
        (
            _overall_scores(),
            _paired_summaries(paired_subject_win_rate=0.449999),
            "paired_subject_win_rate_gate_passed",
        ),
    ),
)
def test_each_gate_failure_stops_promotion(
    overall: list[dict[str, object]],
    pairs: list[dict[str, object]],
    failed_field: str,
) -> None:
    decision = _decision(overall, pairs)
    assert decision["decision"] == "stop_chsd_promotion"
    assert decision["passed"] is False
    if failed_field in {
        "within_one_percentage_point_gate_passed",
        "strictly_above_tcformer_gate_passed",
    }:
        assert decision[failed_field] is False
    else:
        assert all(
            diagnostic[failed_field] is False
            for diagnostic in decision["reference_diagnostics"].values()
        )


def test_analyzer_validates_all_460_without_opening_eeg_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "run"
    write_complete_run(run_root)

    def forbidden_np_load(*args: object, **kwargs: object) -> None:
        raise AssertionError("analyzer attempted to open an EEG cache")

    monkeypatch.setattr(np, "load", forbidden_np_load)
    result = analyze_run(run_root)
    validate_analysis_payload(result)
    assert result["schema"] == ANALYSIS_SCHEMA
    assert result["audit"] == {
        "expected_records": EXPECTED_JOB_COUNT,
        "valid_records": EXPECTED_JOB_COUNT,
        "expected_subjects": EXPECTED_SUBJECT_COUNT,
        "models": 4,
        "complete": True,
        "decision_count": 1,
    }
    assert len(result["jobs"]) == 460
    assert len(result["dataset_scores"]) == 20
    assert len(result["overall_scores"]) == 4
    assert len(result["paired_comparisons"]) == 18
    assert result["analysis_policy"]["cache_files_opened"] is False
    assert result["analysis_policy"]["trial_arrays_opened"] is False
    assert result["robustness_decision"]["decision"] == ("stop_chsd_promotion")
    assert result["lineage"]["conditioned_v3_analysis"]["decision"] == (
        HISTORICAL_V3_DECISION
    )
    assert result["lineage"]["conditioned_v3_analysis"]["selected_model"] is None


def test_missing_corrupt_or_unexpected_record_prevents_any_decision(
    tmp_path: Path,
) -> None:
    missing_root = tmp_path / "missing"
    write_complete_run(missing_root)
    first = robustness_jobs()[0]
    _record_path(missing_root, first).unlink()
    with pytest.raises(RobustnessAnalysisError, match="all 460"):
        analyze_run(missing_root)

    corrupt_root = tmp_path / "corrupt"
    write_complete_run(corrupt_root)
    corrupt_path = _record_path(corrupt_root, first)
    payload = json.loads(corrupt_path.read_text(encoding="utf-8"))
    payload["cache_identity"]["file_sha256"] = "0" * 64
    corrupt_path.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RobustnessAnalysisError, match="all 460"):
        analyze_run(corrupt_root)

    extra_root = tmp_path / "extra"
    write_complete_run(extra_root)
    extra = extra_root / "records" / "unexpected.json"
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RobustnessAnalysisError, match="unexpected record"):
        analyze_run(extra_root)

    hidden_root = tmp_path / "hidden"
    write_complete_run(hidden_root)
    hidden = hidden_root / "records" / "predictions.npy"
    hidden.write_bytes(b"forbidden")
    with pytest.raises(RobustnessAnalysisError, match="unexpected record"):
        analyze_run(hidden_root)

    symlink_root = tmp_path / "symlink"
    write_complete_run(symlink_root)
    linked_path = _record_path(symlink_root, first)
    target = tmp_path / "outside-record.json"
    linked_path.replace(target)
    linked_path.symlink_to(target)
    with pytest.raises(RobustnessAnalysisError, match="symlink"):
        analyze_run(symlink_root)


def test_analysis_outputs_are_deterministic_and_refuse_overwrite(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    output_root = tmp_path / "analysis"
    write_complete_run(run_root)
    first = analyze_and_write(run_root, output_root)
    second = analyze_and_write(run_root, output_root)
    assert first == second
    assert {path.name for path in output_root.iterdir() if path.is_file()} == {
        "analysis.json",
        "dataset_scores.csv",
        "overall_scores.csv",
        "paired_comparisons.csv",
        "REPORT.md",
    }
    report = output_root / "REPORT.md"
    report.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RobustnessAnalysisError, match="differs"):
        analyze_and_write(run_root, output_root)


def test_analysis_output_cannot_pollute_immutable_run(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    write_complete_run(run_root)
    with pytest.raises(RobustnessAnalysisError, match="sibling"):
        analyze_and_write(run_root, run_root)
    with pytest.raises(RobustnessAnalysisError, match="sibling"):
        analyze_and_write(run_root, run_root / "analysis")
    with pytest.raises(RobustnessAnalysisError, match="sibling"):
        analyze_and_write(run_root, tmp_path / "nested" / "analysis")

    polluted = tmp_path / "polluted"
    polluted.mkdir()
    (polluted / "unexpected.txt").write_text("unexpected\n", encoding="utf-8")
    with pytest.raises(RobustnessAnalysisError, match="unexpected"):
        analyze_and_write(run_root, polluted)


def test_analysis_rejects_source_or_uv_runtime_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "run"
    write_complete_run(run_root)
    monkeypatch.setattr(robustness_analysis, "source_identity", dict)
    with pytest.raises(RobustnessAnalysisError, match="source differs"):
        analyze_run(run_root)

    monkeypatch.setattr(
        robustness_analysis,
        "source_identity",
        lambda: {relative: _digest(relative) for relative in SOURCE_FILES},
    )
    monkeypatch.setattr(
        robustness_analysis,
        "robustness_environment_identity",
        lambda: {"changed": True},
    )
    with pytest.raises(RobustnessAnalysisError, match="UV/runtime"):
        analyze_run(run_root)
