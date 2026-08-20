from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import benchmark.runner as benchmark
from benchmark.runner import _require_unique, _summary
from benchmark.training import TrainConfig


def _record(
    *,
    model: str,
    subject: int,
    fold: int,
    seed: int,
    row: int,
    label: int,
    probability: tuple[float, float],
) -> dict[str, object]:
    return {
        "model": model,
        "subject": subject,
        "fold": fold,
        "seed": seed,
        "test": {
            "rows": [row],
            "labels": [label],
            "probabilities": [list(probability)],
            "metrics": {"balanced_accuracy": 0.0},
        },
    }


def test_summary_concatenates_folds_then_averages_seeds_within_subject() -> None:
    records: list[dict[str, object]] = []
    for model in ("proposed", "baseline"):
        for subject in (1, 2):
            for seed in (7, 19):
                records.append(
                    _record(
                        model=model,
                        subject=subject,
                        fold=0,
                        seed=seed,
                        row=0,
                        label=0,
                        probability=(0.9, 0.1),
                    )
                )
                second = (0.1, 0.9) if subject == 1 else (0.9, 0.1)
                records.append(
                    _record(
                        model=model,
                        subject=subject,
                        fold=1,
                        seed=seed,
                        row=1,
                        label=1,
                        probability=second,
                    )
                )

    summary = _summary(records)
    assert summary["proposed"]["n_subjects"] == 2
    assert summary["proposed"]["n_records"] == 8
    assert summary["proposed"]["subject_balanced_accuracy"] == {
        "1": 1.0,
        "2": 0.5,
    }
    assert summary["proposed"]["balanced_accuracy_mean"] == 0.75


def test_summary_rejects_duplicate_or_incomplete_model_keys() -> None:
    record = _record(
        model="a",
        subject=1,
        fold=0,
        seed=7,
        row=0,
        label=0,
        probability=(0.9, 0.1),
    )
    with pytest.raises(RuntimeError, match="duplicate benchmark key"):
        _summary([record, record])

    other = _record(
        model="b",
        subject=2,
        fold=0,
        seed=7,
        row=0,
        label=0,
        probability=(0.9, 0.1),
    )
    with pytest.raises(RuntimeError, match="incomplete Cartesian"):
        _summary([record, other])


def test_cli_dimensions_reject_duplicates() -> None:
    _require_unique("seeds", (7, 19))
    with pytest.raises(ValueError, match="duplicates"):
        _require_unique("seeds", (7, 7))


def test_screen_design_permits_partial_exploratory_dimensions() -> None:
    benchmark.validate_development_design(
        "cho2017",
        artifact_mode="screen",
        subjects=(1, 2),
        folds=(0,),
        seeds=(7,),
    )


def test_formal_design_requires_the_exact_frozen_dimensions() -> None:
    contract = benchmark.formal_development_contract("bnci2014_004")
    subjects = tuple(contract["subjects"])
    folds = tuple(contract["folds"])
    seeds = tuple(contract["seeds"])
    assert len(seeds) >= 5
    benchmark.validate_development_design(
        "bnci2014_004",
        artifact_mode="formal",
        subjects=subjects,
        folds=folds,
        seeds=seeds,
    )

    with pytest.raises(ValueError, match="subjects differ"):
        benchmark.validate_development_design(
            "bnci2014_004",
            artifact_mode="formal",
            subjects=subjects[:-1],
            folds=folds,
            seeds=seeds,
        )
    with pytest.raises(ValueError, match="seeds differ"):
        benchmark.validate_development_design(
            "bnci2014_004",
            artifact_mode="formal",
            subjects=subjects,
            folds=folds,
            seeds=seeds[:-1],
        )


def test_formal_cho_contract_requires_every_outer_fold() -> None:
    contract = benchmark.formal_development_contract("cho2017")
    assert contract["folds"] == [0, 1, 2, 3, 4]
    with pytest.raises(ValueError, match="folds differ"):
        benchmark.validate_development_design(
            "cho2017",
            artifact_mode="formal",
            subjects=tuple(contract["subjects"]),
            folds=(0,),
            seeds=tuple(contract["seeds"]),
        )


def test_micro_dynamics_name_routes_to_the_factory() -> None:
    assert benchmark._model_definition("cardinal_fbc_micro") == (
        "cardinal_fbc_micro",
        "baseline",
    )
    assert benchmark._model_definition("cardinal_fbc_corr") == (
        "cardinal_fbc_corr",
        "baseline",
    )
    assert benchmark._model_definition("cardinal_fbc_corr_extended") == (
        "cardinal_fbc_corr_extended",
        "baseline",
    )
    assert benchmark._model_definition("cardinal_fbc_compactdyn") == (
        "cardinal_fbc_compactdyn",
        "baseline",
    )
    assert benchmark._model_definition("cardinal_fbc_compactdyn_extended") == (
        "cardinal_fbc_compactdyn_extended",
        "baseline",
    )
    for scale_name in (
        "cardinal_fbc_compactdyn_scale010",
        "cardinal_fbc_compactdyn_scale010_extended",
        "cardinal_fbc_compactdyn_scale025",
        "cardinal_fbc_compactdyn_scale025_extended",
    ):
        assert benchmark._model_definition(scale_name) == (scale_name, "baseline")
    assert benchmark._model_definition("cardinal_fbc_physical") == (
        "cardinal_fbc_physical",
        "baseline",
    )
    assert benchmark._model_definition("cardinal_fbc_physical_extended") == (
        "cardinal_fbc_physical_extended",
        "baseline",
    )
    assert benchmark._model_definition("cardinal_fbms") == (
        "cardinal_fbms",
        "baseline",
    )
    assert benchmark._model_definition("cardinal_fbms_extended") == (
        "cardinal_fbms_extended",
        "baseline",
    )


class _TinyBenchmarkNet(nn.Module):
    uses_positions = False

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.full((1,), 2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        score = self.weight * x.mean(dim=(1, 2))
        return torch.stack((-score, score), dim=1)


def test_run_one_restores_initialization_and_uses_refit_for_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(41)
    x = rng.normal(size=(12, 3, 16)).astype(np.float32)
    # Make validation/source statistics visibly different from train-only.
    x[4:8] += 3.0
    labels = np.asarray([0, 1] * 6, dtype=np.int64)
    data = {
        "x": x,
        "y": labels,
        "positions": np.eye(3, dtype=np.float32),
        "channel_names": np.asarray(("C3", "Cz", "C4")),
        "sessions": np.asarray(["s"] * 12),
        "runs": np.asarray([str(index // 4) for index in range(12)]),
        "identity": {"array_sha256": "synthetic"},
    }
    monkeypatch.setattr(benchmark, "load_subject_cache", lambda *args, **kwargs: data)
    monkeypatch.setattr(
        benchmark,
        "split_indices",
        lambda *args, **kwargs: (
            np.arange(0, 4),
            np.arange(4, 8),
            np.arange(8, 12),
        ),
    )
    monkeypatch.setattr(benchmark, "make_model", lambda *args, **kwargs: _TinyBenchmarkNet())

    def fake_fit(model: nn.Module, *args: object, **kwargs: object) -> dict[str, object]:
        model.weight.data.fill_(9.0)  # type: ignore[attr-defined]
        return {
            "model": model,
            "best_epoch": 0,
            "epochs_run": 1,
            "best_validation_loss": 0.5,
            "fit_seconds": 0.1,
            "history": [{"epoch": 0.0}],
            "config": {},
        }

    refit_seen = {"called": False}

    def fake_refit(
        model: nn.Module, *args: object, **kwargs: object
    ) -> dict[str, object]:
        assert torch.equal(model.weight.detach().cpu(), torch.full((1,), 2.0))  # type: ignore[attr-defined]
        assert kwargs["epochs"] == 1
        refit_seen["called"] = True
        model.weight.data.fill_(1.0)  # type: ignore[attr-defined]
        return {
            "model": model,
            "epochs_run": 1,
            "fit_seconds": 0.2,
            "history": [{"epoch": 0.0}],
        }

    monkeypatch.setattr(benchmark, "fit_model", fake_fit)
    monkeypatch.setattr(benchmark, "refit_model", fake_refit)

    def fake_predict(model: nn.Module, values: np.ndarray, *args: object, **kwargs: object) -> np.ndarray:
        weight = float(model.weight.detach())  # type: ignore[attr-defined]
        positive = 0.9 if weight == 9.0 else 0.7
        return np.tile((1.0 - positive, positive), (len(values), 1))

    monkeypatch.setattr(benchmark, "predict_probabilities", fake_predict)
    record = benchmark.run_one(
        dataset="bnci2014_004",
        subject=1,
        fold=0,
        requested_model="fbcnet",
        seed=7,
        cache_root=Path("unused"),
        train_config=TrainConfig(epochs=2, device="cpu"),
    )
    assert refit_seen["called"]
    assert record["test"]["probabilities"][0] == [0.30000000000000004, 0.7]
    assert record["scaler"]["selection_train"] != record["scaler"]["refit_source"]
