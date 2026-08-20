from __future__ import annotations

import csv
from functools import lru_cache
import hashlib
import importlib.util
import io
from pathlib import Path
import shlex
import shutil
import stat
import subprocess
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "generate_reviewer_score_cells.py"
SPEC = importlib.util.spec_from_file_location(
    "generate_reviewer_score_cells", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
score_cells = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(score_cells)


@lru_cache(maxsize=1)
def _rows() -> list[dict[str, str]]:
    return score_cells.build_rows(PROJECT_ROOT)


def _csv_index(path: Path, keys: tuple[str, ...]) -> dict[tuple[str, ...], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {tuple(row[key] for key in keys): row for row in rows}


def test_exact_model_dataset_plus_overall_roster_and_order() -> None:
    rows = _rows()
    assert len(rows) == 43 * 6 == 258
    assert len({row["score_cell_id"] for row in rows}) == 258
    assert len({row["row_id"] for row in rows}) == 258
    assert tuple(rows[0][key] for key in ("model", "dataset", "score_level")) == (
        "cardinal_fbc_compactdyn_scale025_extended",
        "local_exp4",
        "dataset",
    )
    assert [row["dataset"] for row in rows[:6]] == [
        "local_exp4",
        "bnci2014_001",
        "bnci2014_004",
        "cho2017",
        "physionet_mi",
        "overall",
    ]
    assert {row["model"] for row in rows} == {
        row["model"]
        for row in rows
        if row["dataset"] == "overall"
    }
    for model_index in range(43):
        block = rows[model_index * 6 : (model_index + 1) * 6]
        assert len({row["model"] for row in block}) == 1
        assert [row["score_level"] for row in block] == ["dataset"] * 5 + [
            "overall"
        ]

    recommended = [
        row for row in rows if row["recommended_for_complete_table_run"] == "true"
    ]
    by_row_id = {row["row_id"]: row for row in rows}
    assert len(recommended) == 43
    assert {row["replay_scope"] for row in recommended} == {"model"}
    assert sum(int(row["job_count"]) for row in recommended) == 96_320
    for row in rows:
        expected_model_row = f"model::{row['model']}"
        assert row["covered_by_model_run_row_id"] == expected_model_row
        covering = by_row_id[row["covered_by_model_run_row_id"]]
        assert covering["model"] == row["model"]
        assert covering["replay_scope"] == "model"
        assert covering["recommended_for_complete_table_run"] == "true"
        if row["score_level"] == "overall":
            assert row["row_id"] == expected_model_row
            assert row["recommended_for_complete_table_run"] == "true"
        else:
            assert row["row_id"].startswith(f"dataset::{row['model']}::")
            assert row["recommended_for_complete_table_run"] == "false"
    assert [int(rows[index * 6]["primary_ba_rank"]) for index in range(43)] == list(
        range(1, 44)
    )


def test_score_strings_are_copied_from_only_the_sealed_authority_tables() -> None:
    rows = _rows()
    dataset_authority = _csv_index(
        PROJECT_ROOT
        / "results/common_grid_v6/analysis/dataset_summary.csv",
        ("model", "dataset"),
    )
    overall_authority = _csv_index(
        PROJECT_ROOT
        / "results/common_grid_v6/analysis/overall_summary.csv",
        ("model",),
    )
    for row in rows:
        if row["score_level"] == "dataset":
            source = dataset_authority[(row["model"], row["dataset"])]
            assert row["score_source"].endswith("analysis/dataset_summary.csv")
        else:
            source = overall_authority[(row["model"],)]
            assert row["score_source"].endswith("analysis/overall_summary.csv")
        assert row["accuracy"] == source["accuracy"]
        assert row["balanced_accuracy"] == source["balanced_accuracy"]
        assert row["aggregation"] == source["aggregation"]

    leader = next(
        row
        for row in rows
        if row["score_cell_id"]
        == "cardinal_fbc_compactdyn_scale025_extended::overall"
    )
    assert leader["accuracy"] == "0.7402086182336182"
    assert leader["balanced_accuracy"] == "0.7398999919431077"
    assert leader["accuracy_percent_2dp"] == "74.02%"
    assert leader["accuracy_percent_3dp"] == "74.021%"
    assert leader["balanced_accuracy_percent_2dp"] == "73.99%"
    assert leader["balanced_accuracy_percent_3dp"] == "73.990%"
    assert leader["primary_ba_rank"] == "1"
    assert leader["primary_ba_rank_metric"] == (
        "equal_dataset_macro_balanced_accuracy"
    )
    assert leader["primary_ba_rank_status"] == (
        "descriptive_opened_development_suite_only"
    )
    assert leader["descriptive_accuracy_rank"] == "1"
    assert "frozen plan architecture order" in (
        leader["descriptive_accuracy_rank_derivation"]
    )
    assert leader["reference_plan_sha256"] == score_cells.EXPECTED_PLAN_SHA256
    assert leader["raw_plan_file_sha256"] == score_cells.EXPECTED_RAW_SHA256[
        score_cells.PLAN_RELATIVE.as_posix()
    ]
    assert leader["sealed_analysis_manifest_sha256"] == (
        score_cells.EXPECTED_RAW_SHA256[score_cells.MANIFEST_RELATIVE.as_posix()]
    )
    assert leader["rank_source_sha256"] == score_cells.EXPECTED_RAW_SHA256[
        score_cells.RANKING_TABLE_RELATIVE.as_posix()
    ]
    assert leader["timing_source_sha256"] == score_cells.EXPECTED_RAW_SHA256[
        score_cells.JOB_METRICS_RELATIVE.as_posix()
    ]


def test_job_counts_and_command_templates_are_exact() -> None:
    rows = _rows()
    expected_jobs = {
        "local_exp4": "40",
        "bnci2014_001": "45",
        "bnci2014_004": "45",
        "cho2017": "1300",
        "physionet_mi": "810",
        "overall": "2240",
    }
    assert {
        dataset: {row["job_count"] for row in rows if row["dataset"] == dataset}
        for dataset in expected_jobs
    } == {dataset: {count} for dataset, count in expected_jobs.items()}

    dimensions = {
        dataset: {
            (
                row["subject_count"],
                row["folds_per_subject"],
                row["subject_fold_count"],
                row["seed_count"],
            )
            for row in rows
            if row["dataset"] == dataset
        }
        for dataset in expected_jobs
    }
    assert dimensions == {
        "local_exp4": {("8", "1", "8", "5")},
        "bnci2014_001": {("9", "1", "9", "5")},
        "bnci2014_004": {("9", "1", "9", "5")},
        "cho2017": {("52", "5", "260", "5")},
        "physionet_mi": {("54", "3", "162", "5")},
        "overall": {("132", "by_dataset:1|1|1|5|3", "448", "5")},
    }

    dataset = next(
        row
        for row in rows
        if row["score_cell_id"] == "eegnet::bnci2014_004"
    )
    assert dataset["estimate_command"] == (
        "scripts/reproduce.sh reviewer estimate --scope dataset --model eegnet "
        "--dataset bnci2014_004"
    )
    assert dataset["run_command"] == (
        "scripts/reproduce.sh reviewer run --scope dataset --model eegnet "
        "--dataset bnci2014_004 --cache-root \"${CACHE_ROOT:?set CACHE_ROOT}\" "
        "--run-root \"${REVIEWER_ROOT:?set REVIEWER_ROOT}/"
        "eegnet__bnci2014_004\" --gpu \"${GPU:?set GPU}\""
    )
    assert shlex.split(dataset["run_command"]) == [
        "scripts/reproduce.sh",
        "reviewer",
        "run",
        "--scope",
        "dataset",
        "--model",
        "eegnet",
        "--dataset",
        "bnci2014_004",
        "--cache-root",
        "${CACHE_ROOT:?set CACHE_ROOT}",
        "--run-root",
        "${REVIEWER_ROOT:?set REVIEWER_ROOT}/eegnet__bnci2014_004",
        "--gpu",
        "${GPU:?set GPU}",
    ]
    assert dataset["status_command"] == (
        "scripts/reproduce.sh reviewer status --run-root "
        "\"${REVIEWER_ROOT:?set REVIEWER_ROOT}/eegnet__bnci2014_004\""
    )
    assert dataset["compare_command"] == (
        "scripts/reproduce.sh reviewer compare --run-root "
        "\"${REVIEWER_ROOT:?set REVIEWER_ROOT}/eegnet__bnci2014_004\" "
        "--cache-root \"${CACHE_ROOT:?set CACHE_ROOT}\""
    )
    assert dataset["cache_root_placeholder"] == "${CACHE_ROOT:?set CACHE_ROOT}"
    assert dataset["reviewer_root_placeholder"] == (
        "${REVIEWER_ROOT:?set REVIEWER_ROOT}"
    )
    assert dataset["gpu_placeholder"] == "${GPU:?set GPU}"
    assert dataset["resolved_run_root_template"].endswith(
        "/eegnet__bnci2014_004"
    )

    overall = next(row for row in rows if row["score_cell_id"] == "eegnet::overall")
    assert overall["replay_scope"] == "model"
    assert overall["estimate_command"] == (
        "scripts/reproduce.sh reviewer estimate --scope model --model eegnet"
    )
    assert "--dataset" not in shlex.split(overall["run_command"])
    assert overall["run_command"] == (
        "scripts/reproduce.sh reviewer run --scope model --model eegnet "
        "--cache-root \"${CACHE_ROOT:?set CACHE_ROOT}\" --run-root "
        "\"${REVIEWER_ROOT:?set REVIEWER_ROOT}/eegnet__overall\" "
        "--gpu \"${GPU:?set GPU}\""
    )
    assert overall["historical_a5000_summed_hours"] == "2.119633702609"


def test_access_timing_and_caveat_context_is_explicit() -> None:
    rows = _rows()
    leader_local = next(
        row
        for row in rows
        if row["score_cell_id"]
        == "cardinal_fbc_compactdyn_scale025_extended::local_exp4"
    )
    leader_overall = next(
        row
        for row in rows
        if row["score_cell_id"]
        == "cardinal_fbc_compactdyn_scale025_extended::overall"
    )
    public = next(
        row for row in rows if row["score_cell_id"] == "eegnet::physionet_mi"
    )
    assert leader_local["historical_a5000_summed_hours"] == "0.090329976381"
    assert leader_overall["historical_a5000_summed_hours"] == "3.145145336892"
    assert "job_total_seconds" in leader_overall["historical_timing_caveat"]
    assert "A5000" in leader_overall["historical_timing_caveat"]
    assert leader_local["data_access"] == (
        "private_institutional_authorization_required"
    )
    assert leader_local["general_access"] == "false"
    assert public["general_access"] == "true"
    assert {row["cache_redistributable"] for row in rows} == {"false"}
    assert "opened-development" in leader_overall["evidence_caveat"]
    assert "does not imply bit-exact" in leader_overall["strict_score_caveat"]
    assert "all 43 complete model rows" in leader_overall["rank_caveat"]
    assert "physical GPU" in leader_overall["execution_identity_caveat"]


def test_rendering_and_cli_stdout_are_byte_deterministic() -> None:
    rows = _rows()
    first = score_cells.render_csv(rows)
    second = score_cells.render_csv(_rows())
    assert first == second
    assert b"\r" not in first
    assert first.endswith(b"\n")
    assert hashlib.sha256(first).hexdigest() == (
        "05a1e581f69f8f87a989c98c130eb7e434cae109e993708229038888cc8a619a"
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--project-root",
            str(PROJECT_ROOT),
            "--output",
            "-",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8")
    assert completed.stderr == b""
    assert completed.stdout == first
    parsed = list(csv.DictReader(io.StringIO(first.decode("utf-8"), newline="")))
    assert len(parsed) == 258
    assert tuple(parsed[0]) == score_cells.OUTPUT_FIELDS


def _copy_authority_inputs(destination: Path) -> None:
    for relative in (
        score_cells.PLAN_RELATIVE,
        score_cells.PLAN_SIDECAR_RELATIVE,
        score_cells.AUDIT_RELATIVE,
        score_cells.MANIFEST_RELATIVE,
        score_cells.DATASET_TABLE_RELATIVE,
        score_cells.OVERALL_TABLE_RELATIVE,
        score_cells.RANKING_TABLE_RELATIVE,
        score_cells.JOB_METRICS_RELATIVE,
    ):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT_ROOT / relative, target)


def test_tampered_authority_fails_before_any_csv_is_generated(tmp_path: Path) -> None:
    _copy_authority_inputs(tmp_path)
    for relative in (
        score_cells.DATASET_TABLE_RELATIVE,
        score_cells.RANKING_TABLE_RELATIVE,
        score_cells.JOB_METRICS_RELATIVE,
    ):
        target = tmp_path / relative
        original = target.read_bytes()
        target.write_bytes(original + b"\n")
        with pytest.raises(score_cells.ScoreCellError, match="checksum mismatch"):
            score_cells.build_rows(tmp_path)
        target.write_bytes(original)


def test_output_writer_refuses_to_replace_any_authoritative_input(
    tmp_path: Path,
) -> None:
    protected = tmp_path / score_cells.OVERALL_TABLE_RELATIVE
    protected.parent.mkdir(parents=True)
    protected.write_bytes(b"authority")
    original = protected.read_bytes()
    with pytest.raises(score_cells.ScoreCellError, match="refusing to overwrite"):
        score_cells._write_atomic(protected, b"forbidden", project_root=tmp_path)
    assert protected.read_bytes() == original


def test_atomic_file_output_matches_stdout_bytes(tmp_path: Path) -> None:
    output = tmp_path / "reviewer_score_cells.csv"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--project-root",
            str(PROJECT_ROOT),
            "--output",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8")
    assert completed.stdout == b""
    assert output.read_bytes() == score_cells.render_csv(_rows())
    assert stat.S_IMODE(output.stat().st_mode) == 0o644
    assert not list(tmp_path.glob(".reviewer_score_cells.csv.*.tmp"))

    checked = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--project-root",
            str(PROJECT_ROOT),
            "--output",
            str(output),
            "--check",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert checked.returncode == 0, checked.stderr
    assert checked.stdout == "PASS: 258 deterministic reviewer score cells\n"
    assert checked.stderr == ""


def test_check_mode_detects_a_stale_catalog_without_rewriting(tmp_path: Path) -> None:
    output = tmp_path / "reviewer_score_cells.csv"
    output.write_text("stale\n", encoding="utf-8")
    before = output.read_bytes()
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--project-root",
            str(PROJECT_ROOT),
            "--output",
            str(output),
            "--check",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "stale or noncanonical" in completed.stderr
    assert output.read_bytes() == before
