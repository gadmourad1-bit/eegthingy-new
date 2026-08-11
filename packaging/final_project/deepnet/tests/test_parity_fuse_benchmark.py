from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

import deepnet.parity_fuse_benchmark as benchmark
from deepnet.parity_fuse_net import ParityFuseConfig, ParityFuseOutput


def _config() -> ParityFuseConfig:
    return ParityFuseConfig(epochs=2, patience=2, device="cpu", seed=7)


def _forbid_confirmation_loader(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    loaded: list[int] = []

    def forbidden(subject: int):
        loaded.append(subject)
        raise AssertionError("confirmation loader must not be reached")

    monkeypatch.setattr(benchmark, "load_bnci2014_subject", forbidden)
    return loaded


@pytest.mark.parametrize(
    ("subjects", "seeds", "token", "message"),
    [
        ([5, 6, 7, 8], [7], benchmark.CONFIRMATION_TOKEN, "exact ordered subjects"),
        ([5, 6, 7, 8, 9], [7, 17], benchmark.CONFIRMATION_TOKEN, "sole frozen seed"),
        ([5, 6, 7, 8, 9], [7], None, "explicit confirmation token"),
    ],
)
def test_confirmation_contract_guards_run_before_loader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    subjects: list[int],
    seeds: list[int],
    token: str | None,
    message: str,
) -> None:
    loaded = _forbid_confirmation_loader(monkeypatch)
    with pytest.raises(ValueError, match=message):
        benchmark.run(
            mode="bnci-confirm",
            subjects=subjects,
            seeds=seeds,
            config=_config(),
            output=tmp_path / "must-not-exist.json",
            confirmation_token=token,
        )
    assert loaded == []
    assert not (tmp_path / "must-not-exist.json").exists()


def test_confirmation_requires_manifest_before_loader(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loaded = _forbid_confirmation_loader(monkeypatch)
    with pytest.raises(ValueError, match="frozen-manifest"):
        benchmark.run(
            mode="bnci-confirm",
            subjects=[5, 6, 7, 8, 9],
            seeds=[7],
            config=_config(),
            output=tmp_path / "must-not-exist.json",
            confirmation_token=benchmark.CONFIRMATION_TOKEN,
        )
    assert loaded == []


def test_manifest_config_mismatch_precedes_confirmation_loader(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loaded = _forbid_confirmation_loader(monkeypatch)
    manifest = tmp_path / "frozen.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": benchmark.FROZEN_MANIFEST_SCHEMA_VERSION,
                "architecture": benchmark.ARCHITECTURE,
                "config": benchmark._json_config(
                    ParityFuseConfig(epochs=3, patience=2, device="cpu", seed=7)
                ),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="config differs"):
        benchmark.run(
            mode="bnci-confirm",
            subjects=[5, 6, 7, 8, 9],
            seeds=[7],
            config=_config(),
            output=tmp_path / "must-not-exist.json",
            frozen_manifest=manifest,
            confirmation_token=benchmark.CONFIRMATION_TOKEN,
        )
    assert loaded == []


def test_existing_output_is_never_overwritten_before_loader(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loaded = _forbid_confirmation_loader(monkeypatch)
    output = tmp_path / "existing.json"
    output.write_text('{"belongs":"to someone else"}\n', encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        benchmark.run(
            mode="bnci-dev",
            subjects=[1],
            seeds=[7],
            config=_config(),
            output=output,
        )
    assert loaded == []
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "belongs": "to someone else"
    }
    assert not output.with_name(output.name + ".lock").exists()


def test_exclusive_result_lock_rejects_parallel_invocation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loaded = _forbid_confirmation_loader(monkeypatch)
    output = tmp_path / "result.json"
    lock = output.with_name(output.name + ".lock")
    lock.write_text("occupied\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="concurrent access"):
        benchmark.run(
            mode="bnci-dev",
            subjects=[1],
            seeds=[7],
            config=_config(),
            output=output,
        )
    assert loaded == []
    assert not output.exists()
    assert lock.read_text(encoding="utf-8") == "occupied\n"


def test_local_development_passes_only_first_three_recordings_to_loader(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    observed: list[tuple[int, int]] = []

    def inspect_then_stop(keys, config):
        observed.extend((int(key.subject), int(key.run)) for key in keys)
        raise RuntimeError("intentional stop after loader-contract inspection")

    monkeypatch.setattr(benchmark, "load_sessions", inspect_then_stop)
    monkeypatch.setattr(benchmark, "_environment", lambda: {"test": True})
    output = tmp_path / "local.json"
    with pytest.raises(RuntimeError, match="intentional stop"):
        benchmark.run(
            mode="local-dev",
            subjects=[1],
            seeds=[7],
            config=_config(),
            output=output,
        )
    assert observed == [
        (1, int(run)) for run in benchmark.SUBJECT_RUNS[1][:3]
    ]
    assert all(run != int(benchmark.SUBJECT_RUNS[1][3]) for _, run in observed)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["completion"]["status"] == "interrupted"
    assert payload["completion"]["actual_records"] == 0
    assert not output.with_name(output.name + ".lock").exists()


def test_write_and_validate_frozen_manifest_pins_dev_file_and_summary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = {"deepnet/parity_fuse_net.py": "1" * 64}
    monkeypatch.setattr(benchmark, "_source_manifest", lambda: source)
    config = _config()
    development = tmp_path / "dev.json"
    summary = {"balanced_accuracy_mean": 0.75}
    development.write_text(
        json.dumps(
            {
                "architecture": {"name": benchmark.ARCHITECTURE},
                "mode": "bnci-dev",
                "subjects": [1, 2, 3, 4],
                "seeds": [7],
                "config": benchmark._json_config(config),
                "source_sha256": source,
                "summary": summary,
                "records": [{}, {}, {}, {}],
                "completion": {
                    "status": "complete",
                    "complete": True,
                    "expected_records": 4,
                    "actual_records": 4,
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    frozen = tmp_path / "frozen.json"
    manifest = benchmark.write_frozen_manifest(
        development, frozen, config=config
    )
    assert frozen.is_file()
    assert manifest["development_artifact"]["sha256"] == benchmark._file_sha256(
        development
    )
    assert benchmark._validate_frozen_manifest(frozen, config) == manifest

    development.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact hash"):
        benchmark._validate_frozen_manifest(frozen, config)


def test_prediction_trace_keeps_provenance_anchor_residual_and_gate() -> None:
    output = ParityFuseOutput(
        logit=torch.tensor([1.0, -1.0]),
        anchor_logit=torch.tensor([0.8, -0.7]),
        residual_logit=torch.tensor([0.2, -0.3]),
        fusion_weights=torch.tensor(
            [
                [[0.75, 0.25], [0.25, 0.75]],
                [[0.40, 0.60], [0.60, 0.40]],
            ]
        ),
        raw_even=torch.empty(2, 0),
        raw_odd=torch.empty(2, 0),
        tangent_even=torch.empty(2, 0),
        tangent_odd=torch.empty(2, 0),
        fused_odd=torch.empty(2, 0),
    )
    probabilities = np.asarray([[0.2, 0.8], [0.7, 0.3]])
    anchor_probabilities = np.asarray([[0.3, 0.7], [0.65, 0.35]])
    rows = benchmark._prediction_trace(
        np.asarray([1, 0]),
        probabilities,
        anchor_probabilities,
        output,
        [
            {"fold": 0, "source_trial_index": 11},
            {"fold": 1, "source_trial_index": 4},
        ],
    )
    assert rows[0]["window_index"] == 0
    assert rows[0]["provenance"] == {"fold": 0, "source_trial_index": 11}
    assert rows[0]["anchor_logit"] == pytest.approx(0.8)
    assert rows[0]["residual_logit"] == pytest.approx(0.2)
    assert rows[0]["mean_raw_gate"] == pytest.approx(0.5)
    assert rows[0]["mean_tangent_gate"] == pytest.approx(0.5)


def test_completion_records_anchor_checkpoint_epoch_minus_one() -> None:
    payload = {
        "records": [
            {
                "subject": 1,
                "seed": 7,
                "metrics": {"balanced_accuracy": 0.8},
                "anchor_metrics": {"balanced_accuracy": 0.75},
                "fits": [
                    {
                        "best_epoch": -1,
                        "parameter_count": 19_468,
                        "trainable_parameter_count": 19_468,
                        "equivariance": {
                            "max_abs_end_to_end_logit_sum": 0.0,
                            "max_abs_gate_difference": 0.0,
                        },
                        "residual": {"abs_mean": 0.0},
                        "gate": {"raw": {"mean": 0.5}},
                    }
                ],
            }
        ]
    }
    summary = benchmark._summarize(payload)
    assert summary["best_epoch_counts"] == {"-1": 1}
    assert summary["anchor_checkpoint_count"] == 1
    assert summary["mean_delta_vs_anchor"] == pytest.approx(0.05)
