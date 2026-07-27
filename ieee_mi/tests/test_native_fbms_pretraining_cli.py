from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from ieee_mi import native_fbms_pretraining_cli as cli
from ieee_mi import native_fbms_transfer as transfer
from ieee_mi import native_pretraining_cli as core_cli
from ieee_mi.models import CANONICAL_21_POSITIONS
from ieee_mi.native_pretraining import (
    LoadedNativePretrainingCorpus,
    NativeMontageSubject,
    PRIMARY_NATIVE_DEVELOPMENT_SOURCES,
    state_dict_sha256,
)


def _synthetic_native_corpus() -> LoadedNativePretrainingCorpus:
    canonical = np.asarray(CANONICAL_21_POSITIONS, dtype=np.float32)
    subjects: list[NativeMontageSubject] = []
    sources: list[dict[str, object]] = []
    settings = {
        "bnci2014_001": (21, 320, 100),
        "cho2017": (21, 320, 200),
        "local_exp4": (15, 256, 300),
    }
    for dataset, subject_ids in PRIMARY_NATIVE_DEVELOPMENT_SOURCES.items():
        n_channels, n_times, seed = settings[dataset]
        for subject in subject_ids:
            generator = np.random.default_rng(seed + subject)
            labels = np.asarray([0, 1, 0, 1], dtype=np.int64)
            subjects.append(
                NativeMontageSubject(
                    dataset=dataset,
                    subject=subject,
                    x=generator.normal(
                        size=(len(labels), n_channels, n_times)
                    ).astype(np.float32),
                    y=labels,
                    positions=canonical[:n_channels].copy(),
                    channel_names=tuple(
                        f"E{index}" for index in range(n_channels)
                    ),
                )
            )
            digest = hashlib.sha256(
                f"{dataset}:S{subject}".encode()
            ).hexdigest()
            sources.append(
                {
                    "dataset": dataset,
                    "subject": subject,
                    "cache_file_sha256": digest,
                    "cache_identity_sha256": digest,
                    "cache_array_sha256": digest,
                    "authorized_rows_sha256": digest,
                }
            )
    corpus_hash = hashlib.sha256(
        json.dumps(
            sources, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return LoadedNativePretrainingCorpus(
        subjects=tuple(subjects),
        sources=tuple(sources),
        sha256=corpus_hash,
        cache_schema="ieee-mi-cache-v2",
        montage_profile="native",
    )


def test_fbms_cli_is_a_distinct_locked_no_model_selection_surface() -> None:
    destinations = {action.dest for action in cli.build_parser()._actions}
    assert destinations == {"help", "cache_root", "output", "device"}
    assert cli.ENTRY_POINT.model_key == "cardinal_fbms"
    assert cli.ARTIFACT_SCHEMA != core_cli.ARTIFACT_SCHEMA
    assert cli.CHECKPOINT_SCHEMA != core_cli.CHECKPOINT_SCHEMA
    assert "build_subject_cache" not in inspect.getsource(cli)


def test_fbms_cli_publishes_family_pinned_checkpoint_and_refuses_overwrite(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = _synthetic_native_corpus()
    calls: list[tuple[object, str]] = []

    def fake_loader(cache_root, *, montage_profile: str):
        calls.append((cache_root, montage_profile))
        assert montage_profile == "native"
        return corpus

    monkeypatch.setattr(
        core_cli, "load_primary_native_pretraining_corpus", fake_loader
    )
    real_fitter = core_cli.fit_native_montage_pretraining

    def fast_fitter(factory, subjects, *, config):
        # Exercise the real fitter/publisher while keeping the unit test short.
        # The published configuration remains the frozen candidate contract.
        return real_fitter(
            factory,
            subjects,
            config=replace(
                config,
                macro_epochs=1,
                patience=1,
                batch_size_per_dataset=4,
                steps_per_macro_epoch=1,
                validation_batch_size=4,
            ),
        )

    monkeypatch.setattr(
        core_cli, "fit_native_montage_pretraining", fast_fitter
    )
    output = tmp_path / "cardinal-fbms-pretraining"
    argv = [
        "--cache-root",
        str(tmp_path / "unused-cache"),
        "--output",
        str(output),
        "--device",
        "cpu",
    ]
    assert cli.run(argv) == output.resolve()
    assert len(calls) == 1

    checkpoint_path = output / cli.CHECKPOINT_FILENAME
    provenance = json.loads(
        (output / cli.PROVENANCE_FILENAME).read_text(encoding="utf-8")
    )
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    file_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    assert provenance["schema"] == cli.ARTIFACT_SCHEMA
    assert provenance["model"]["identity"] == "cardinal_fbms"
    assert provenance["model"]["uses_positions"] is True
    assert provenance["checkpoint"]["schema"] == cli.CHECKPOINT_SCHEMA
    assert provenance["checkpoint"]["file_sha256"] == file_hash
    assert checkpoint["schema"] == cli.CHECKPOINT_SCHEMA
    assert checkpoint["model"]["identity"] == "cardinal_fbms"
    expected_config = asdict(cli.FROZEN_CONFIG)
    expected_config["device"] = "cpu"
    assert checkpoint["training_config"] == expected_config
    assert checkpoint["training_config"]["seed"] == 7
    assert checkpoint["corpus"]["sha256"] == corpus.sha256
    assert checkpoint["state_hashes"]["checkpoint"] == state_dict_sha256(
        checkpoint["model_state_dict"]
    )
    assert checkpoint["source_code"] == provenance["source_code"]
    assert "ieee_mi/native_fbms_pretraining_cli.py" in checkpoint[
        "source_code"
    ]["files"]
    assert all(
        tensor.device.type == "cpu"
        for tensor in checkpoint["model_state_dict"].values()
    )
    assert not list(tmp_path.glob(f".{output.name}.staging-*"))
    assert not (tmp_path / f".{output.name}.native-pretraining.lock").exists()

    # Publisher and transfer loader share the exact family/config/partition/
    # corpus contract, including string-valued SubjectKey identities.
    loaded = transfer.load_immutable_native_checkpoint(
        checkpoint_path, expected_file_sha256=file_hash
    )
    assert loaded.file_sha256 == file_hash
    assert loaded.pretraining["partition"]["sha256"] == checkpoint[
        "partition"
    ]["sha256"]

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        cli.run(argv)
    assert len(calls) == 1
