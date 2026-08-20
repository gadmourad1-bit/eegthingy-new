from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict
from itertools import product
from pathlib import Path

import numpy as np
import pytest

from benchmark.analysis import _bootstrap_mean, _holm_adjust, analyze
from benchmark.runner import (
    DEVELOPMENT_ARTIFACT_SCHEMA,
    FORMAL_DEVELOPMENT_SEEDS,
    LEGACY_SCREEN_ARTIFACT_SCHEMA,
    formal_development_contract,
)
from benchmark.config import (
    CHANNEL_SCALING,
    channels_for_dataset,
    dataset_spec,
    preprocessing_for_dataset,
)
from benchmark.training import TrainConfig


def test_holm_adjustment_is_monotone_in_raw_p_order() -> None:
    raw = {"a": 0.01, "b": 0.03, "c": 0.20}
    adjusted = _holm_adjust(raw)
    assert adjusted == {"a": 0.03, "b": 0.06, "c": 0.20}


def test_subject_bootstrap_is_deterministic_and_bounded() -> None:
    values = np.asarray((0.5, 0.75, 1.0), dtype=np.float64)
    first = _bootstrap_mean(
        values,
        generator=np.random.default_rng(9),
        repetitions=100,
    )
    second = _bootstrap_mean(
        values,
        generator=np.random.default_rng(9),
        repetitions=100,
    )
    assert np.array_equal(first, second)
    assert np.all((first >= 0.5) & (first <= 1.0))


def _record(
    dataset: str,
    *,
    subject: int,
    model: str,
    fold: int,
    seed: int,
) -> dict[str, object]:
    channels = list(channels_for_dataset(dataset))
    preprocessing = preprocessing_for_dataset(dataset)
    rows = [0, 1]
    labels = [0, 1]
    if model == "proposed":
        probabilities = [[0.9, 0.1], [0.1, 0.9]]
    else:
        probabilities = [[0.9, 0.1], [0.9, 0.1]]
    train_config = asdict(TrainConfig(device="cpu", seed=seed))
    zeros = [0.0] * len(channels)
    ones = [1.0] * len(channels)
    return {
        "dataset": dataset,
        "subject": subject,
        "fold": fold,
        "model": model,
        "seed": seed,
        "n_classes": 2,
        "channels": channels,
        "split": {
            "train_count": 2,
            "validation_count": 2,
            "refit_source_count": 4,
            "test_count": 2,
            "train_runs": ["0"],
            "validation_runs": ["1"],
            "test_runs": ["2"],
            "train_rows_sha256": "1" * 64,
            "validation_rows_sha256": "2" * 64,
            "test_rows_sha256": hashlib.sha256(
                np.asarray(rows, dtype=np.int64).tobytes()
            ).hexdigest(),
        },
        "scaler": {
            "selection_train": {"mean": zeros, "std": ones},
            "refit_source": {"mean": zeros, "std": ones},
        },
        "fit": {"train_config": train_config},
        "test": {
            "metrics": {"balanced_accuracy": 1.0 if model == "proposed" else 0.5},
            "rows": rows,
            "labels": labels,
            "probabilities": probabilities,
            "sessions": ["test", "test"],
            "runs": ["2", "2"],
        },
        "cache_identity": {
            "dataset": asdict(dataset_spec(dataset)),
            "subject": subject,
            "preprocessing": preprocessing,
            "shape": [2, len(channels), int(preprocessing["n_times"])],
            "channels": channels,
            "array_sha256": hashlib.sha256(
                f"{dataset}:{subject}".encode("utf-8")
            ).hexdigest(),
        },
    }


def _artifact(
    *,
    dataset: str = "bnci2014_004",
    artifact_mode: str = "screen",
    subjects: tuple[int, ...] = (1, 2),
    models: tuple[str, ...] = ("reference", "proposed"),
    folds: tuple[int, ...] = (0,),
    seeds: tuple[int, ...] = (7,),
) -> dict[str, object]:
    records = [
        _record(
            dataset,
            subject=subject,
            model=model,
            fold=fold,
            seed=seed,
        )
        for subject, fold, model, seed in product(subjects, folds, models, seeds)
    ]
    return {
        "schema": DEVELOPMENT_ARTIFACT_SCHEMA,
        "created_at": "2026-07-19T00:00:00+00:00",
        "mode": "development",
        "artifact_mode": artifact_mode,
        "evidence_scope": (
            "exploratory_development_screen_descriptive_only"
            if artifact_mode == "screen"
            else "prespecified_formal_development_not_confirmation"
        ),
        "confirmation_evidence": False,
        "analysis_policy": (
            "descriptive_only"
            if artifact_mode == "screen"
            else "prespecified_development_inference_not_confirmation"
        ),
        "formal_protocol": (
            formal_development_contract(dataset) if artifact_mode == "formal" else None
        ),
        "dataset": dataset,
        "dataset_spec": asdict(dataset_spec(dataset)),
        "subjects": list(subjects),
        "models": list(models),
        "seeds": list(seeds),
        "folds": list(folds),
        "preprocessing": preprocessing_for_dataset(dataset),
        "channel_scaling": CHANNEL_SCALING,
        "train_config": asdict(TrainConfig(device="cpu")),
        "environment": {"runtime": "test"},
        "source_sha256": {"benchmark.py": "a" * 64},
        "records": records,
        "summary": {},
    }


def _write(path: Path, artifact: dict[str, object]) -> Path:
    path.write_text(json.dumps(artifact), encoding="utf-8")
    return path


def test_screen_analysis_is_explicitly_descriptive_and_has_no_inference(
    tmp_path: Path,
) -> None:
    path = _write(tmp_path / "screen.json", _artifact())
    result = analyze([path], reference="reference", bootstrap_repetitions=100)

    assert result["artifact_mode"] == "screen"
    assert result["confirmation_evidence"] is False
    assert result["inferential_statistics"] is False
    assert result["bootstrap_repetitions"] == 0
    assert "bootstrap_95_ci" not in result["datasets"]["bnci2014_004"]["proposed"]
    comparison = result["paired_comparisons"][
        "bnci2014_004:proposed-vs-reference"
    ]
    assert comparison["analysis"] == "descriptive_only"
    assert "wilcoxon_p_raw" not in comparison
    assert "wilcoxon_p_holm" not in comparison


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("evidence_scope", "confirmation", "invalid evidence scope"),
        ("formal_protocol", {"claimed": True}, "must not claim a formal protocol"),
    ),
)
def test_screen_artifact_rejects_semantics_that_overstate_its_status(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    artifact = _artifact()
    artifact[field] = value
    path = _write(tmp_path / f"invalid-{field}.json", artifact)
    with pytest.raises(ValueError, match=message):
        analyze([path], reference="reference")


def test_legacy_v3_artifact_is_forced_to_noninferential_screen_semantics(
    tmp_path: Path,
) -> None:
    artifact = _artifact()
    artifact["schema"] = LEGACY_SCREEN_ARTIFACT_SCHEMA
    for key in (
        "artifact_mode",
        "evidence_scope",
        "confirmation_evidence",
        "analysis_policy",
        "formal_protocol",
    ):
        artifact.pop(key)
    path = _write(tmp_path / "legacy.json", artifact)

    result = analyze([path], reference="reference")
    assert result["artifact_mode"] == "screen"
    assert result["contains_legacy_screen_artifact"] is True
    assert result["inferential_statistics"] is False


def test_formal_analysis_requires_full_contract_and_permits_development_inference(
    tmp_path: Path,
) -> None:
    subjects = dataset_spec("bnci2014_004").development_subjects
    artifact = _artifact(
        artifact_mode="formal",
        subjects=subjects,
        seeds=FORMAL_DEVELOPMENT_SEEDS,
    )
    path = _write(tmp_path / "formal.json", artifact)
    result = analyze([path], reference="reference", bootstrap_repetitions=100)

    assert result["artifact_mode"] == "formal"
    assert result["evidence_scope"] == "development_only_not_confirmation"
    assert result["confirmation_evidence"] is False
    assert result["inferential_statistics"] is True
    assert "bootstrap_95_ci" in result["datasets"]["bnci2014_004"]["proposed"]
    comparison = result["paired_comparisons"][
        "bnci2014_004:proposed-vs-reference"
    ]
    assert "wilcoxon_p_raw" in comparison
    assert "wilcoxon_p_holm" in comparison


def test_formal_artifact_rejects_partial_frozen_seed_set(tmp_path: Path) -> None:
    artifact = _artifact(
        artifact_mode="formal",
        subjects=dataset_spec("bnci2014_004").development_subjects,
        seeds=FORMAL_DEVELOPMENT_SEEDS[:-1],
    )
    path = _write(tmp_path / "partial.json", artifact)
    with pytest.raises(ValueError, match="seeds differ"):
        analyze([path], reference="reference")


def test_artifact_validation_rejects_noncartesian_records(tmp_path: Path) -> None:
    artifact = _artifact()
    artifact["records"].pop()  # type: ignore[union-attr]
    path = _write(tmp_path / "missing.json", artifact)
    with pytest.raises(ValueError, match="declared Cartesian product"):
        analyze([path], reference="reference")


@pytest.mark.parametrize("field", ("cache", "rows", "labels", "scaler"))
def test_artifact_validation_rejects_unpaired_data_contracts(
    tmp_path: Path,
    field: str,
) -> None:
    artifact = _artifact(subjects=(1,))
    records = artifact["records"]
    assert isinstance(records, list)
    proposed = next(record for record in records if record["model"] == "proposed")
    if field == "cache":
        proposed["cache_identity"]["array_sha256"] = "f" * 64
    elif field == "rows":
        proposed["test"]["rows"] = [2, 3]
        proposed["split"]["test_rows_sha256"] = hashlib.sha256(
            np.asarray([2, 3], dtype=np.int64).tobytes()
        ).hexdigest()
    elif field == "labels":
        proposed["test"]["labels"] = [1, 0]
    else:
        proposed["scaler"]["refit_source"]["std"][0] = 2.0
    path = _write(tmp_path / f"mismatch-{field}.json", artifact)
    with pytest.raises(ValueError, match="paired data contract differs"):
        analyze([path], reference="reference")


def test_artifact_validation_rejects_effective_train_config_mismatch(
    tmp_path: Path,
) -> None:
    artifact = _artifact(subjects=(1, 2))
    records = artifact["records"]
    assert isinstance(records, list)
    target = next(
        record
        for record in records
        if record["model"] == "proposed" and record["subject"] == 2
    )
    target["fit"]["train_config"]["learning_rate"] = 0.123
    path = _write(tmp_path / "config-mismatch.json", artifact)
    with pytest.raises(ValueError, match="effective train configurations"):
        analyze([path], reference="reference")


def test_artifact_validation_rejects_environment_mismatch_across_shards(
    tmp_path: Path,
) -> None:
    first = _artifact(subjects=(1,))
    second = _artifact(subjects=(2,))
    second["environment"] = {"runtime": "different"}
    paths = [
        _write(tmp_path / "first.json", first),
        _write(tmp_path / "second.json", second),
    ]
    with pytest.raises(ValueError, match="different environments"):
        analyze(paths, reference="reference")


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("dataset_spec", "dataset specification is stale"),
        ("preprocessing", "preprocessing contract is stale"),
        ("channel_scaling", "channel-scaling contract is stale"),
    ),
)
def test_artifact_validation_rejects_stale_data_contract_metadata(
    tmp_path: Path,
    field: str,
    message: str,
) -> None:
    artifact = _artifact()
    metadata = copy.deepcopy(artifact[field])
    assert isinstance(metadata, dict)
    metadata["tampered"] = True
    artifact[field] = metadata
    path = _write(tmp_path / f"stale-{field}.json", artifact)
    with pytest.raises(ValueError, match=message):
        analyze([path], reference="reference")


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("train_config", "different base train configurations"),
        ("source_sha256", "different source revisions"),
    ),
)
def test_artifact_validation_rejects_provenance_mismatch_across_shards(
    tmp_path: Path,
    field: str,
    message: str,
) -> None:
    first = _artifact(subjects=(1,))
    second = _artifact(subjects=(2,))
    if field == "train_config":
        second["train_config"] = copy.deepcopy(second["train_config"])
        second["train_config"]["learning_rate"] = 0.123
    else:
        second["source_sha256"] = {"benchmark.py": "b" * 64}
    paths = [
        _write(tmp_path / "first.json", first),
        _write(tmp_path / "second.json", second),
    ]
    with pytest.raises(ValueError, match=message):
        analyze(paths, reference="reference")


def test_artifact_validation_rejects_cache_changes_between_outer_folds(
    tmp_path: Path,
) -> None:
    artifact = _artifact(subjects=(1,), folds=(0, 1))
    records = artifact["records"]
    assert isinstance(records, list)
    target = next(record for record in records if record["fold"] == 1)
    target["cache_identity"]["array_sha256"] = "f" * 64
    path = _write(tmp_path / "cache-across-folds.json", artifact)
    with pytest.raises(ValueError, match="cache identity differs"):
        analyze([path], reference="reference")


def test_artifact_validation_rejects_malformed_source_hash(tmp_path: Path) -> None:
    artifact = _artifact()
    artifact["source_sha256"] = {"benchmark.py": "not-a-sha256"}
    path = _write(tmp_path / "bad-source.json", artifact)
    with pytest.raises(ValueError, match="invalid source-code identity"):
        analyze([path], reference="reference")
