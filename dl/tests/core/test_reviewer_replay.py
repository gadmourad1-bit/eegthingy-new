from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib

import numpy as np
import pytest

from benchmark import reviewer_replay

full_grid = reviewer_replay.full_grid


def test_reviewer_import_keeps_replay_archive_private() -> None:
    script = """
from pathlib import Path
import sys
import benchmark
import benchmark.reviewer_replay as replay
active_root = Path(benchmark.__file__).resolve().parent
assert active_root.name == 'benchmark'
assert active_root.parent.name == 'src'
for name, module in tuple(sys.modules.items()):
    if name == 'benchmark' or name.startswith('benchmark.'):
        source = getattr(module, '__file__', None)
        if source is not None:
            assert 'historical/common_grid_v6/source' not in Path(source).as_posix()
private = sys.modules[replay.HISTORICAL_PRIVATE_PACKAGE]
assert Path(private.__file__).resolve().parent.name == 'eeg_mi_v6'
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(reviewer_replay.PROJECT_ROOT / "src")
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=reviewer_replay.PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


def _selection(scope: str) -> reviewer_replay.Selection:
    if scope == "job":
        return reviewer_replay.Selection(
            scope="job",
            model="eegnet",
            dataset="local_exp4",
            subject=1,
            fold=0,
            seed=7,
        )
    if scope == "subject":
        return reviewer_replay.Selection(
            scope="subject",
            model="eegnet",
            dataset="cho2017",
            subject=1,
        )
    if scope == "dataset":
        return reviewer_replay.Selection(
            scope="dataset",
            model="eegnet",
            dataset="bnci2014_001",
        )
    return reviewer_replay.Selection(scope="model", model="eegnet")


@pytest.mark.parametrize(
    ("scope", "expected"),
    (("job", 1), ("subject", 25), ("dataset", 45), ("model", 2_240)),
)
def test_exact_scope_cardinalities(scope: str, expected: int) -> None:
    plan = reviewer_replay.load_reference_plan()
    jobs = reviewer_replay.select_jobs(plan, _selection(scope))
    assert len(jobs) == expected
    assert len({job.job_id for job in jobs}) == expected
    assert {job.model for job in jobs} == {"eegnet"}
    assert {job.seed for job in jobs}.issubset(set(full_grid.FORMAL_SEEDS))


def test_dataset_cardinalities_are_the_published_cells() -> None:
    plan = reviewer_replay.load_reference_plan()
    expected = {
        "local_exp4": 40,
        "bnci2014_001": 45,
        "bnci2014_004": 45,
        "cho2017": 1_300,
        "physionet_mi": 810,
    }
    for dataset, count in expected.items():
        selection = reviewer_replay.Selection("dataset", "eegnet", dataset)
        assert len(reviewer_replay.select_jobs(plan, selection)) == count


@pytest.mark.parametrize(
    "selection",
    (
        reviewer_replay.Selection("model", "not_a_model"),
        reviewer_replay.Selection("model", "eegnet", "cho2017"),
        reviewer_replay.Selection("dataset", "eegnet"),
        reviewer_replay.Selection("subject", "eegnet", "cho2017"),
        reviewer_replay.Selection("subject", "eegnet", "cho2017", 53),
        reviewer_replay.Selection("job", "eegnet", "cho2017", 1, 5, 7),
        reviewer_replay.Selection("job", "eegnet", "cho2017", 1, 0, 8),
    ),
)
def test_invalid_or_estimand_changing_selectors_fail(selection: reviewer_replay.Selection) -> None:
    with pytest.raises(reviewer_replay.ReviewerReplayError):
        reviewer_replay.select_jobs(reviewer_replay.load_reference_plan(), selection)


def test_reference_reader_accepts_normal_checkout_modes() -> None:
    plan_path = reviewer_replay.REFERENCE_ROOT / "plan.json"
    assert os.stat(plan_path).st_mode & 0o200
    plan = reviewer_replay.load_reference_plan()
    assert plan["plan_sha256"] == "65b93b7e5d09cfc30fb6ce28156368e66aec1503f4efa37faa3189e25ebbc3b1"


def test_historical_tcformer_environment_is_scoped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = str(reviewer_replay.PROJECT_ROOT / "third_party" / "TCFormer")
    monkeypatch.setenv("EEG_MI_TCFORMER_ROOT", root)
    monkeypatch.delenv("EEG_MI_TCFORMER_ROOT", raising=False)
    with reviewer_replay._historical_tcformer_environment():
        assert os.environ["EEG_MI_TCFORMER_ROOT"] == root
    assert "EEG_MI_TCFORMER_ROOT" not in os.environ


def test_source_compatibility_preserves_exact_sanitized_v6(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = reviewer_replay.load_reference_plan()
    compatible = reviewer_replay.source_compatibility(plan)
    repaired = {
        row["path"]
        for row in compatible["formal_sources"]
        if row["status"] != "exact"
    }
    assert repaired == set()
    current = reviewer_replay._historical_source_identity()
    current["eeg_mi/training.py"] = "0" * 64
    monkeypatch.setattr(
        reviewer_replay, "_historical_source_identity", lambda: current
    )
    with pytest.raises(reviewer_replay.ReviewerReplayError, match="unapproved source"):
        reviewer_replay.source_compatibility(plan)


def test_v1_source_snapshot_compatibility_is_retired() -> None:
    with pytest.raises(reviewer_replay.ReviewerReplayError, match="retired"):
        reviewer_replay.source_compatibility(
            reviewer_replay.load_reference_plan(), legacy_v1_snapshot=True
        )


def test_release_stack_match_covers_transitive_locked_packages() -> None:
    lock = tomllib.loads((reviewer_replay.PROJECT_ROOT / "uv.lock").read_text())
    packages = [
        [row["name"], row["version"]]
        for row in lock["package"]
        if row["name"] != "benchmark"
    ]
    reference = reviewer_replay.load_reference_plan()["environment_identity"]
    gpu_uuid = reference["nvidia_gpu_inventory"][0]["uuid"]
    environment = {
        **reference,
        "packages": packages,
        "python_version": "3.12.13 (locked test runtime)",
        "platform": "Linux-6.8.0-test-x86_64-with-glibc2.35",
        "uv_version": "uv 0.12.0 (x86_64-unknown-linux-gnu)",
        "torch": {
            **reference["torch"],
            "version": "2.6.0+cu124",
            "cuda": "12.4",
        },
    }
    compatible = reviewer_replay._validate_runtime_environment(environment, gpu_uuid)
    assert compatible["release_locked_stack_match"] is True

    drifted = dict(environment)
    drifted["packages"] = [
        [name, "999.0.0" if name == "filelock" else version]
        for name, version in packages
    ]
    incompatible = reviewer_replay._validate_runtime_environment(drifted, gpu_uuid)
    assert incompatible["release_locked_stack_match"] is False
    assert incompatible["installed_version_drift"]["filelock"] == {
        "locked": "3.32.2",
        "installed": "999.0.0",
    }


def test_exact_execution_identity_requires_release_locked_stack() -> None:
    """Equal sealed source/environment/GPU identities cannot hide lock drift."""

    assert reviewer_replay._exact_execution_identity_match(
        exact_sources=True,
        exact_environment=True,
        exact_gpu=True,
        compatibility={"release_locked_stack_match": False},
    ) is False
    assert reviewer_replay._exact_execution_identity_match(
        exact_sources=True,
        exact_environment=True,
        exact_gpu=True,
        compatibility={"release_locked_stack_match": True},
    ) is True


def test_replay_plan_is_immutable_and_selection_bound(tmp_path: Path) -> None:
    reference = reviewer_replay.load_reference_plan()
    first = reviewer_replay.build_replay_plan(reference, _selection("job"))
    root = tmp_path / "run"
    reviewer_replay.write_or_validate_replay_plan(root, first)
    assert reviewer_replay.load_replay_plan(root) == first

    second_selection = reviewer_replay.Selection(
        "job", "eegnet", "local_exp4", 1, 0, 17
    )
    second = reviewer_replay.build_replay_plan(reference, second_selection)
    with pytest.raises(reviewer_replay.ReviewerReplayError, match="differs"):
        reviewer_replay.write_or_validate_replay_plan(root, second)


def test_v1_plan_creation_is_retired() -> None:
    reference = reviewer_replay.load_reference_plan()
    with pytest.raises(reviewer_replay.ReviewerReplayError, match="retired"):
        reviewer_replay.build_replay_plan(
            reference,
            _selection("job"),
            schema=reviewer_replay.LEGACY_REPLAY_SCHEMA,
        )


def test_artifact_schemas_dispatch_at_namespace_boundary() -> None:
    active = {"schema": reviewer_replay.REPLAY_SCHEMA}
    assert reviewer_replay._artifact_schema(active, "record").startswith("eeg-mi-")
    assert reviewer_replay._artifact_schema(active, "environment").endswith("-v2")
    assert reviewer_replay._artifact_schema(active, "comparison").endswith("-v2")
    assert reviewer_replay._artifact_schema(active, "status").endswith("-v2")
    assert reviewer_replay.TRACK_SCOPE == "reviewer_replay_v2"
    assert reviewer_replay.ESTIMATE_SCHEMA == "eeg-mi-reviewer-replay-estimate-v2"


def test_sanitized_dataset_replay_plan_is_source_bound() -> None:
    reference = reviewer_replay.load_reference_plan()
    selection = reviewer_replay.Selection(
        "dataset",
        "cardinal_fbc_compactdyn_scale025_extended",
        "bnci2014_004",
    )
    replay = reviewer_replay.build_replay_plan(reference, selection)
    assert replay["source_compatibility"]["formal_sources"]
    assert {
        row["status"] for row in replay["source_compatibility"]["formal_sources"]
    } == {"exact"}


def _matching_job_record(
    replay_plan: dict[str, object], environment_sha: str
) -> tuple[full_grid.Job, dict[str, object]]:
    job = full_grid.Job(
        dataset="local_exp4", model="eegnet", subject=1, fold=0, seed=7
    )
    prediction_sha = reviewer_replay._prediction_ledger()[(job.job_id,)][
        "predictions_sha256"
    ]
    return job, {
        "schema": reviewer_replay.RECORD_SCHEMA,
        "replay_plan_sha256": replay_plan["replay_plan_sha256"],
        "reference_plan_sha256": replay_plan["reference_plan_sha256"],
        "job": job.identity(),
        "job_id": job.job_id,
        "cache_array_sha256": replay_plan["cache_identity"][
            "local_exp4:s001"
        ]["array_sha256"],
        "split_identity_sha256": reviewer_replay._sha256_bytes(
            reviewer_replay._canonical_bytes(
                replay_plan["split_identity"]["local_exp4:s001:f00"]
            )
        ),
        "test_count": 60,
        "n_classes": 2,
        "confusion_matrix": [[30, 0], [30, 0]],
        "accuracy": 0.5,
        "balanced_accuracy": 0.5,
        "replay_predictions_sha256": prediction_sha,
        "published_predictions_sha256": prediction_sha,
        "prediction_sha256_exact": True,
        "fit": {
            "initial_state_sha256": "2" * 64,
            "selection_state_sha256": "3" * 64,
            "refit_state_sha256": "4" * 64,
            "selected_epoch_index": 19,
            "selection_epochs_run": 55,
            "refit_epochs_run": 20,
            "parameter_count": 1602,
            "architecture": {"requested_name": "eegnet"},
        },
        "timing_seconds": {
            "selection_fit": 1.0,
            "refit_fit": 1.0,
            "test_inference": 0.1,
            "job_total": 2.1,
        },
        "environment_sha256": environment_sha,
        "retained_trial_rows": False,
        "retained_probabilities": False,
    }


def test_immutable_record_retains_no_trial_predictions_and_resumes(tmp_path: Path) -> None:
    reference = reviewer_replay.load_reference_plan()
    replay_plan = reviewer_replay.build_replay_plan(reference, _selection("job"))
    root = tmp_path / "run"
    reviewer_replay.write_or_validate_replay_plan(root, replay_plan)
    environment_sha = reviewer_replay._write_or_validate_environment(
        root, {"test_environment": True}
    )
    job, record = _matching_job_record(replay_plan, environment_sha)
    reviewer_replay._publish_record(root, job, record)
    reviewer_replay._publish_record(root, job, record)
    observed = reviewer_replay._validate_record(root, replay_plan, job)
    assert "probabilities" not in observed
    assert "rows" not in observed
    assert observed["retained_trial_rows"] is False
    assert observed["retained_probabilities"] is False
    assert reviewer_replay.status(root)["exact_selected_complete"] is True


def test_record_published_digest_must_match_sealed_ledger(tmp_path: Path) -> None:
    reference = reviewer_replay.load_reference_plan()
    replay_plan = reviewer_replay.build_replay_plan(reference, _selection("job"))
    root = tmp_path / "run"
    reviewer_replay.write_or_validate_replay_plan(root, replay_plan)
    environment_sha = reviewer_replay._write_or_validate_environment(
        root, {"test_environment": True}
    )
    job, record = _matching_job_record(replay_plan, environment_sha)
    record["published_predictions_sha256"] = "0" * 64
    record["prediction_sha256_exact"] = False
    reviewer_replay._publish_record(root, job, record)
    with pytest.raises(reviewer_replay.ReviewerReplayError, match="sealed prediction ledger"):
        reviewer_replay._validate_record(root, replay_plan, job)


def test_tampered_record_checksum_fails_closed(tmp_path: Path) -> None:
    reference = reviewer_replay.load_reference_plan()
    replay_plan = reviewer_replay.build_replay_plan(reference, _selection("job"))
    root = tmp_path / "run"
    reviewer_replay.write_or_validate_replay_plan(root, replay_plan)
    environment_sha = reviewer_replay._write_or_validate_environment(root, {"test": 1})
    job, record = _matching_job_record(replay_plan, environment_sha)
    reviewer_replay._publish_record(root, job, record)
    record_path = reviewer_replay._record_directory(root, job) / "record.json"
    os.chmod(record_path, 0o644)
    record_path.write_bytes(record_path.read_bytes() + b" ")
    os.chmod(record_path, 0o444)
    with pytest.raises(reviewer_replay.ReviewerReplayError, match="checksum"):
        reviewer_replay._validate_record(root, replay_plan, job)


def test_confusion_aggregation_concatenates_folds_then_averages_seeds() -> None:
    reference = reviewer_replay.load_reference_plan()
    selection = reviewer_replay.Selection(
        "subject", "eegnet", "local_exp4", 1
    )
    replay_plan = reviewer_replay.build_replay_plan(reference, selection)
    records = []
    for index, job in enumerate(reviewer_replay.select_jobs(reference, selection)):
        matrix = [[30 - index, index], [index, 30 - index]]
        records.append({"job": job.identity(), "confusion_matrix": matrix})
    result = reviewer_replay._aggregate_records(replay_plan, records)
    expected = np.mean([(60 - 2 * index) / 60 for index in range(5)])
    assert result["subject_rows"][0]["accuracy"] == pytest.approx(expected)
    assert result["subject_rows"][0]["balanced_accuracy"] == pytest.approx(expected)


def test_comparison_distinguishes_score_display_and_prediction_identity() -> None:
    published = {"accuracy": "0.5", "balanced_accuracy": "0.5"}
    exact = reviewer_replay._comparison_row(
        "job", {"job_id": "x"}, {"accuracy": 0.5, "balanced_accuracy": 0.5}, published
    )
    assert exact["strict_match"] is True
    assert exact["display_match"] is True

    display_only = reviewer_replay._comparison_row(
        "job",
        {"job_id": "x"},
        {"accuracy": 0.500004, "balanced_accuracy": 0.500004},
        published,
    )
    assert display_only["strict_match"] is False
    assert display_only["display_match"] is True


def test_parser_exposes_only_bounded_reviewer_scopes() -> None:
    parser = reviewer_replay.build_parser()
    parsed = parser.parse_args(
        [
            "run",
            "--scope",
            "dataset",
            "--model",
            "eegnet",
            "--dataset",
            "cho2017",
            "--cache-root",
            "/cache",
            "--run-root",
            "/run",
            "--gpu",
            "0",
        ]
    )
    assert parsed.scope == "dataset"
    assert parsed.model == "eegnet"
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["estimate", "--scope", "all", "--model", "eegnet"]
        )
