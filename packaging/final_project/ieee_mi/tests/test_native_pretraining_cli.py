from __future__ import annotations

import hashlib
import inspect
import json

import numpy as np
import pytest
import torch

from ieee_mi import native_pretraining_cli as cli
from ieee_mi.models import CANONICAL_21_POSITIONS
from ieee_mi.native_pretraining import (
    LoadedNativePretrainingCorpus,
    NativeMontageSubject,
    state_dict_sha256,
)


def _synthetic_native_corpus() -> LoadedNativePretrainingCorpus:
    canonical = np.asarray(CANONICAL_21_POSITIONS, dtype=np.float32)
    subjects: list[NativeMontageSubject] = []
    sources: list[dict[str, object]] = []
    for dataset, n_channels, n_times, seed in (
        ("bnci2014_001", 21, 320, 100),
        ("local_exp4", 15, 256, 200),
    ):
        positions = canonical[:n_channels].copy()
        for subject in (1, 2):
            generator = np.random.default_rng(seed + subject)
            labels = np.asarray([0, 1, 0, 1], dtype=np.int64)
            values = generator.normal(
                size=(len(labels), n_channels, n_times)
            ).astype(np.float32)
            subjects.append(
                NativeMontageSubject(
                    dataset=dataset,
                    subject=subject,
                    x=values,
                    y=labels,
                    positions=positions.copy(),
                    channel_names=tuple(
                        f"E{index}" for index in range(n_channels)
                    ),
                )
            )
            source_key = f"{dataset}:S{subject}"
            digest = hashlib.sha256(source_key.encode()).hexdigest()
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
    return LoadedNativePretrainingCorpus(
        subjects=tuple(subjects),
        sources=tuple(sources),
        sha256=hashlib.sha256(b"synthetic-native-corpus").hexdigest(),
        cache_schema="ieee-mi-cache-v2",
        montage_profile="native",
    )


def test_cli_surface_has_no_cache_build_or_confirmation_switches() -> None:
    destinations = {
        action.dest for action in cli.build_parser()._actions
    }
    assert destinations.isdisjoint(
        {"stage", "montage_profile", "dataset", "subjects", "model"}
    )
    assert "build_subject_cache" not in inspect.getsource(cli)


def test_cli_atomically_publishes_checkpoint_and_refuses_overwrite_before_load(
    tmp_path, monkeypatch
) -> None:
    corpus = _synthetic_native_corpus()
    loader_calls: list[tuple[object, str]] = []

    def fake_loader(cache_root, *, montage_profile: str):
        loader_calls.append((cache_root, montage_profile))
        assert montage_profile == "native"
        return corpus

    monkeypatch.setattr(
        cli, "load_primary_native_pretraining_corpus", fake_loader
    )
    output = tmp_path / "native-cardinal-artifact"
    argv = [
        "--cache-root",
        str(tmp_path / "unused-cache-root"),
        "--output",
        str(output),
        "--macro-epochs",
        "1",
        "--patience",
        "1",
        "--batch-size-per-dataset",
        "4",
        "--steps-per-macro-epoch",
        "1",
        "--validation-batch-size",
        "4",
        "--seed",
        "43",
        "--device",
        "cpu",
    ]
    assert cli.run(argv) == output.resolve()
    assert len(loader_calls) == 1
    checkpoint_path = output / cli.CHECKPOINT_FILENAME
    provenance_path = output / cli.PROVENANCE_FILENAME
    assert {path.name for path in output.iterdir()} == {
        cli.CHECKPOINT_FILENAME,
        cli.PROVENANCE_FILENAME,
    }

    provenance = json.loads(provenance_path.read_text())
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    checkpoint_file_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    assert provenance["schema"] == cli.ARTIFACT_SCHEMA
    assert provenance["mode"] == "development"
    assert provenance["confirmation_access"] is False
    assert provenance["model"]["identity"] == "cardinal_fbc"
    assert provenance["model"]["construction"]["n_channels"] == 21
    assert provenance["model"]["construction"]["n_times"] == 320
    assert provenance["corpus"]["sha256"] == corpus.sha256
    assert provenance["partition"]["sha256"]
    assert provenance["selection"]["history"]
    assert provenance["refit"]["history"]
    assert provenance["checkpoint"]["file_sha256"] == checkpoint_file_hash
    assert checkpoint["schema"] == cli.CHECKPOINT_SCHEMA
    assert checkpoint["corpus"]["sha256"] == corpus.sha256
    assert checkpoint["state_hashes"]["checkpoint"] == state_dict_sha256(
        checkpoint["model_state_dict"]
    )
    assert all(
        tensor.device.type == "cpu"
        for tensor in checkpoint["model_state_dict"].values()
    )
    assert not list(tmp_path.glob(f".{output.name}.staging-*"))
    assert not (tmp_path / f".{output.name}.native-pretraining.lock").exists()

    # The second invocation must fail before the cache loader is called.
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        cli.run(argv)
    assert len(loader_calls) == 1
