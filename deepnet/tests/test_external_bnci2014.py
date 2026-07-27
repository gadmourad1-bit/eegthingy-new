from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

import deepnet.external_bnci2014 as bnci


def _valid_splits() -> dict[str, dict[str, np.ndarray]]:
    splits: dict[str, dict[str, np.ndarray]] = {}
    for split, runs in bnci._EXPECTED_RUNS.items():
        count = len(runs) * bnci._TRIALS_PER_RUN
        run_ids = np.repeat(np.asarray(runs, dtype=np.str_), bnci._TRIALS_PER_RUN)
        labels = np.concatenate(
            [
                np.asarray(
                    [0] * bnci._TRIALS_PER_CLASS_PER_RUN
                    + [1] * bnci._TRIALS_PER_CLASS_PER_RUN,
                    dtype=np.int64,
                )
                for _ in runs
            ]
        )
        covariance = np.eye(len(bnci.BNCI_CHANNELS), dtype=np.float32)[None, None]
        splits[split] = {
            "covariances": np.broadcast_to(
                covariance,
                (
                    count,
                    len(bnci.BANDS),
                    len(bnci.BNCI_CHANNELS),
                    len(bnci.BNCI_CHANNELS),
                ),
            ).copy(),
            "broadband": np.zeros(
                (count, len(bnci.BNCI_CHANNELS), bnci._EPOCH_SAMPLES),
                dtype=np.float32,
            ),
            "labels": labels,
            "run_ids": run_ids,
        }
    return splits


def test_cache_identity_pins_schema_protocol_subject_and_preprocessing_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bnci, "_preprocessing_source_digest", lambda: "a" * 64)
    first_identity = bnci._cache_identity(1)
    first_path = bnci._cache_path(1)

    assert bnci._CACHE_SCHEMA_VERSION == 2
    assert first_identity == {
        "cache_schema": 2,
        "preprocessing_source_sha256": "a" * 64,
        "protocol_id": bnci.PROTOCOL_ID,
        "protocol_json": bnci._canonical_json(bnci.protocol_metadata(1)),
        "subject": 1,
    }

    monkeypatch.setattr(bnci, "_preprocessing_source_digest", lambda: "b" * 64)
    assert bnci._cache_path(1) != first_path
    assert bnci._cache_path(2) != bnci._cache_path(1)


def test_preprocessing_source_digest_covers_loader_and_covariance_code() -> None:
    digest = bnci._preprocessing_source_digest()
    assert len(digest) == 64
    assert set(digest) <= set("0123456789abcdef")
    assert bnci._PREPROCESSING_SOURCE_FILES == ("external_bnci2014.py", "data.py")


def test_cache_round_trip_validates_metadata_and_payload(tmp_path: Path) -> None:
    expected = _valid_splits()
    target = tmp_path / "subject_01.npz"
    bnci._write_cache(target, expected, subject=1)

    actual = bnci._unpack_cache(target, subject=1)
    for split in bnci._SPLIT_NAMES:
        for name in bnci._ARRAY_NAMES:
            np.testing.assert_array_equal(actual[split][name], expected[split][name])

    with np.load(target, allow_pickle=False) as cached:
        assert int(cached["metadata_cache_schema"].item()) == 2
        assert int(cached["metadata_subject"].item()) == 1
        assert str(cached["metadata_protocol_id"].item()) == bnci.PROTOCOL_ID
        assert str(cached["metadata_preprocessing_source_sha256"].item()) == (
            bnci._preprocessing_source_digest()
        )


def _remove_array(splits: dict[str, dict[str, np.ndarray]]) -> None:
    splits["train"].pop("labels")


def _break_broadband_shape(splits: dict[str, dict[str, np.ndarray]]) -> None:
    splits["validation"]["broadband"] = splits["validation"]["broadband"][..., :-1]


def _break_label_count(splits: dict[str, dict[str, np.ndarray]]) -> None:
    splits["test"]["labels"] = splits["test"]["labels"][:-1]


def _break_classes(splits: dict[str, dict[str, np.ndarray]]) -> None:
    splits["validation"]["labels"].fill(0)


def _break_run_identity(splits: dict[str, dict[str, np.ndarray]]) -> None:
    splits["validation"]["run_ids"].fill("4")


def _break_run_count(splits: dict[str, dict[str, np.ndarray]]) -> None:
    splits["train"]["run_ids"][0] = "1"


def _insert_nonfinite_value(splits: dict[str, dict[str, np.ndarray]]) -> None:
    splits["test"]["covariances"][0, 0, 0, 0] = np.inf


def _break_positive_definiteness(splits: dict[str, dict[str, np.ndarray]]) -> None:
    splits["test"]["covariances"][0, 0, 0, 0] = -1.0


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (_remove_array, "array keys"),
        (_break_broadband_shape, "broadband has shape"),
        (_break_label_count, "labels/run_ids"),
        (_break_classes, "both binary classes"),
        (_break_run_identity, "run IDs"),
        (_break_run_count, "has 23 trials"),
        (_insert_nonfinite_value, "non-finite features"),
        (_break_positive_definiteness, "not SPD"),
    ),
)
def test_split_validation_rejects_corrupt_generated_payloads(
    tmp_path: Path,
    mutate: Callable[[dict[str, dict[str, np.ndarray]]], None],
    message: str,
) -> None:
    splits = _valid_splits()
    mutate(splits)
    with pytest.raises(ValueError, match=message):
        bnci._write_cache(tmp_path / "invalid.npz", splits, subject=1)
    assert not (tmp_path / "invalid.npz").exists()


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("metadata_cache_schema", np.asarray(1, dtype=np.int64)),
        ("metadata_subject", np.asarray(2, dtype=np.int64)),
        ("metadata_protocol_id", np.asarray("wrong-protocol", dtype=np.str_)),
        ("metadata_protocol_json", np.asarray("{}", dtype=np.str_)),
        (
            "metadata_preprocessing_source_sha256",
            np.asarray("0" * 64, dtype=np.str_),
        ),
    ),
)
def test_cache_load_rejects_wrong_subject_protocol_or_source_identity(
    tmp_path: Path, field: str, replacement: np.ndarray
) -> None:
    target = tmp_path / "corrupt_metadata.npz"
    bnci._write_cache(target, _valid_splits(), subject=1)
    with np.load(target, allow_pickle=False) as cached:
        arrays = {name: cached[name] for name in cached.files}
    arrays[field] = replacement
    np.savez_compressed(target, **arrays)

    with pytest.raises(ValueError, match=f"mismatched {field}"):
        bnci._unpack_cache(target, subject=1)


def test_cache_load_rejects_missing_or_extra_keys(tmp_path: Path) -> None:
    target = tmp_path / "corrupt_keys.npz"
    bnci._write_cache(target, _valid_splits(), subject=1)
    with np.load(target, allow_pickle=False) as cached:
        arrays = {name: cached[name] for name in cached.files}
    arrays.pop("test_labels")
    arrays["untrusted_extra"] = np.asarray(1)
    np.savez_compressed(target, **arrays)

    with pytest.raises(ValueError, match="unexpected keys"):
        bnci._unpack_cache(target, subject=1)


def test_cache_load_rejects_corrupt_numeric_payload(tmp_path: Path) -> None:
    target = tmp_path / "corrupt_features.npz"
    bnci._write_cache(target, _valid_splits(), subject=1)
    with np.load(target, allow_pickle=False) as cached:
        arrays = {name: cached[name] for name in cached.files}
    arrays["test_broadband"] = arrays["test_broadband"].copy()
    arrays["test_broadband"][0, 0, 0] = np.nan
    np.savez_compressed(target, **arrays)

    with pytest.raises(ValueError, match="non-finite features"):
        bnci._unpack_cache(target, subject=1)


def test_public_loader_validates_a_cache_hit_without_importing_moabb(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(bnci, "_CACHE", tmp_path)
    target = bnci._cache_path(1)
    bnci._write_cache(target, _valid_splits(), subject=1)

    loaded = bnci.load_bnci2014_subject(1)
    assert set(loaded) == {"train", "validation", "test"}
    assert loaded["train"]["labels"].shape == (120,)
