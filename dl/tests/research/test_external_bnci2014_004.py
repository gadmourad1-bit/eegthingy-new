from __future__ import annotations

import builtins
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import benchmark.research.external_bnci2014_004 as bnci


def _valid_splits(
    *, evaluation_trials: tuple[int, int] = (160, 160)
) -> dict[str, dict[str, np.ndarray]]:
    splits: dict[str, dict[str, np.ndarray]] = {}
    counts = dict(bnci._TRIALS_PER_SESSION)
    counts.update(dict(zip(bnci.TEST_SESSIONS, evaluation_trials, strict=True)))
    for split, sessions in bnci._EXPECTED_SESSIONS.items():
        session_ids = np.concatenate(
            [
                np.asarray(
                    [session] * counts[session], dtype=np.str_
                )
                for session in sessions
            ]
        )
        labels = np.concatenate(
            [
                np.asarray(
                    [0] * (counts[session] // 2)
                    + [1] * (counts[session] // 2),
                    dtype=np.int64,
                )
                for session in sessions
            ]
        )
        count = len(labels)
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
                (count, len(bnci.BNCI_CHANNELS), bnci.EPOCH_SAMPLES),
                dtype=np.float32,
            ),
            "labels": labels,
            "session_ids": session_ids,
        }
    return splits


def test_locked_protocol_metadata_describes_bipolar_three_channel_data() -> None:
    metadata = bnci.protocol_metadata(1)

    assert bnci.BNCI_CHANNELS == ("C3", "Cz", "C4")
    assert bnci.REFLECTION_CHANNEL_INDICES == (2, 1, 0)
    assert bnci.SOURCE_SFREQ == 250.0
    assert bnci.BNCI_SFREQ == 125.0
    assert bnci.EPOCH_SAMPLES == 251
    assert metadata["dataset"] == "BNCI2014-004"
    assert metadata["reference"] == "provided_bipolar"
    assert metadata["reflection_channel_indices"] == [2, 1, 0]
    assert metadata["fit"] == {
        "sessions": ["0train", "1train"],
        "allowed_trials": [240, 260, 280, 300, 320],
    }
    assert metadata["select"] == {
        "sessions": ["2train"],
        "allowed_trials": [120, 140, 160],
    }
    assert metadata["test"] == {
        "sessions": ["3test", "4test"],
        "allowed_trials": [240, 260, 280, 300, 320],
    }
    assert metadata["nominal_session_trial_counts"] == {
        "0train": 120,
        "1train": 120,
        "2train": 160,
        "3test": 160,
        "4test": 160,
    }
    assert metadata["allowed_session_trial_counts"] == {
        "0train": [120, 140, 160],
        "1train": [120, 140, 160],
        "2train": [120, 140, 160],
        "3test": [120, 140, 160],
        "4test": [120, 140, 160],
    }


def test_subject_partition_and_confirmation_tripwire() -> None:
    assert [bnci.subject_partition(subject) for subject in range(1, 5)] == [
        "development"
    ] * 4
    assert [bnci.subject_partition(subject) for subject in range(5, 10)] == [
        "confirmation"
    ] * 5
    with pytest.raises(ValueError, match="must be in"):
        bnci.subject_partition(0)
    with pytest.raises(PermissionError, match="frozen confirmation cohort"):
        bnci.load_bnci2014_004_subject(5)


def test_cache_identity_pins_schema_protocol_subject_and_preprocessing_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bnci, "_preprocessing_source_digest", lambda: "a" * 64)
    first_identity = bnci._cache_identity(1)
    first_path = bnci._cache_path(1)

    assert bnci._CACHE_SCHEMA_VERSION == 1
    assert first_identity == {
        "cache_schema": 1,
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
    assert bnci._PREPROCESSING_SOURCE_FILES == (
        "external_bnci2014_004.py",
        "data.py",
    )


def test_cache_round_trip_validates_metadata_and_payload(tmp_path: Path) -> None:
    expected = _valid_splits()
    target = tmp_path / "subject_01.npz"
    bnci._write_cache(target, expected, subject=1)

    actual = bnci._unpack_cache(target, subject=1)
    for split in bnci._SPLIT_NAMES:
        for name in bnci._ARRAY_NAMES:
            np.testing.assert_array_equal(actual[split][name], expected[split][name])

    with np.load(target, allow_pickle=False) as cached:
        assert int(cached["metadata_cache_schema"].item()) == 1
        assert int(cached["metadata_subject"].item()) == 1
        assert str(cached["metadata_protocol_id"].item()) == bnci.PROTOCOL_ID
        assert str(cached["metadata_preprocessing_source_sha256"].item()) == (
            bnci._preprocessing_source_digest()
        )


@pytest.mark.parametrize("evaluation_trials", ((120, 120), (120, 160), (160, 120)))
def test_official_variable_evaluation_session_counts_are_valid(
    tmp_path: Path, evaluation_trials: tuple[int, int]
) -> None:
    expected = _valid_splits(evaluation_trials=evaluation_trials)
    target = tmp_path / f"subject_variable_{sum(evaluation_trials)}.npz"
    bnci._write_cache(target, expected, subject=1)
    actual = bnci._unpack_cache(target, subject=1)
    assert len(actual["test"]["labels"]) == sum(evaluation_trials)
    for session, count in zip(bnci.TEST_SESSIONS, evaluation_trials, strict=True):
        assert int(np.sum(actual["test"]["session_ids"] == session)) == count


def _remove_array(splits: dict[str, dict[str, np.ndarray]]) -> None:
    splits["train"].pop("labels")


def _break_broadband_shape(splits: dict[str, dict[str, np.ndarray]]) -> None:
    splits["validation"]["broadband"] = splits["validation"]["broadband"][..., :-1]


def _break_label_count(splits: dict[str, dict[str, np.ndarray]]) -> None:
    splits["test"]["labels"] = splits["test"]["labels"][:-1]


def _break_classes(splits: dict[str, dict[str, np.ndarray]]) -> None:
    splits["validation"]["labels"].fill(0)


def _break_session_identity(splits: dict[str, dict[str, np.ndarray]]) -> None:
    splits["validation"]["session_ids"].fill("1train")


def _break_session_count(splits: dict[str, dict[str, np.ndarray]]) -> None:
    splits["train"]["session_ids"][0] = "1train"


def _insert_nonfinite_value(splits: dict[str, dict[str, np.ndarray]]) -> None:
    splits["test"]["covariances"][0, 0, 0, 0] = np.inf


def _break_positive_definiteness(splits: dict[str, dict[str, np.ndarray]]) -> None:
    splits["test"]["covariances"][0, 0, 0, 0] = -1.0


def _break_symmetry(splits: dict[str, dict[str, np.ndarray]]) -> None:
    splits["test"]["covariances"][0, 0, 0, 1] = 0.25


def _break_dtype(splits: dict[str, dict[str, np.ndarray]]) -> None:
    splits["train"]["broadband"] = splits["train"]["broadband"].astype(np.float64)


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (_remove_array, "array keys"),
        (_break_broadband_shape, "broadband has shape"),
        (_break_label_count, "test has 319 trials"),
        (_break_classes, "both binary classes"),
        (_break_session_identity, "session IDs"),
        (_break_session_count, "has 119 trials"),
        (_insert_nonfinite_value, "non-finite features"),
        (_break_positive_definiteness, "not SPD"),
        (_break_symmetry, "not symmetric"),
        (_break_dtype, "broadband must be float32"),
    ),
)
def test_split_validation_rejects_corrupt_generated_payloads(
    tmp_path: Path,
    mutate: Callable[[dict[str, dict[str, np.ndarray]]], None],
    message: str,
) -> None:
    splits = _valid_splits()
    mutate(splits)
    target = tmp_path / "invalid.npz"
    with pytest.raises(ValueError, match=message):
        bnci._write_cache(target, splits, subject=1)
    assert not target.exists()


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("metadata_cache_schema", np.asarray(0, dtype=np.int64)),
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


def test_failed_atomic_cache_write_preserves_existing_target(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = tmp_path / "existing.npz"
    target.write_bytes(b"keep-me")

    def fail_write(*args: Any, **kwargs: Any) -> None:
        raise OSError("synthetic write failure")

    monkeypatch.setattr(bnci.np, "savez_compressed", fail_write)
    with pytest.raises(OSError, match="synthetic write failure"):
        bnci._write_cache(target, _valid_splits(), subject=1)

    assert target.read_bytes() == b"keep-me"
    assert list(tmp_path.iterdir()) == [target]


def test_public_loader_validates_cache_hit_without_importing_mne_or_moabb(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(bnci, "_CACHE", tmp_path)
    target = bnci._cache_path(1)
    bnci._write_cache(target, _valid_splits(), subject=1)

    imported: list[str] = []
    original_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "mne" or name.startswith("moabb"):
            imported.append(name)
            raise AssertionError(f"cache hit attempted to import {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    loaded = bnci.load_bnci2014_004_subject(1)

    assert set(loaded) == {"train", "validation", "test"}
    assert loaded["train"]["labels"].shape == (240,)
    assert imported == []


def test_confirmation_cache_requires_explicit_authorization(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(bnci, "_CACHE", tmp_path)
    target = bnci._cache_path(5)
    bnci._write_cache(target, _valid_splits(), subject=5)

    with pytest.raises(PermissionError, match="frozen confirmation cohort"):
        bnci.load_bnci2014_004_subject(5)
    loaded = bnci.load_bnci2014_004_subject(5, allow_confirmation=True)
    assert loaded["test"]["labels"].shape == (320,)


class _FakeRaw:
    def __init__(
        self,
        *,
        channels: tuple[str, ...] = ("EOG:ch01", "C4", "C3", "Cz"),
        sfreq: float = 250.0,
        history: list[tuple[str, Any]] | None = None,
    ) -> None:
        self.ch_names = list(channels)
        self.info = {"sfreq": sfreq}
        self.history = [] if history is None else history

    def copy(self) -> "_FakeRaw":
        self.history.append(("copy", None))
        return _FakeRaw(
            channels=tuple(self.ch_names),
            sfreq=float(self.info["sfreq"]),
            history=self.history,
        )

    def pick(self, channels: list[str]) -> "_FakeRaw":
        self.history.append(("pick", tuple(channels)))
        self.ch_names = list(channels)
        return self

    def resample(self, sfreq: float, **kwargs: Any) -> "_FakeRaw":
        self.history.append(("resample", sfreq))
        self.info["sfreq"] = sfreq
        return self

    def set_eeg_reference(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("the provided bipolar recordings must not be rereferenced")


def test_prepare_raw_preserves_provided_bipolar_reference_and_resamples() -> None:
    raw = _FakeRaw()
    prepared = bnci._prepare_raw(raw)

    assert raw.ch_names == ["EOG:ch01", "C4", "C3", "Cz"]
    assert prepared.ch_names == ["C3", "Cz", "C4"]
    assert prepared.info["sfreq"] == 125.0
    assert raw.history == [
        ("copy", None),
        ("pick", ("C3", "Cz", "C4")),
        ("resample", 125.0),
    ]


def test_prepare_raw_rejects_missing_channel_or_wrong_source_rate() -> None:
    with pytest.raises(ValueError, match="missing required channels"):
        bnci._prepare_raw(_FakeRaw(channels=("C3", "Cz")))
    with pytest.raises(ValueError, match="expected 250.0 Hz"):
        bnci._prepare_raw(_FakeRaw(sfreq=125.0))


def test_single_session_run_requires_exact_moabb_run_identity() -> None:
    marker = object()
    assert bnci._single_session_run({"0": marker}, session_id="0train") is marker
    with pytest.raises(ValueError, match="expected run"):
        bnci._single_session_run({"1": marker}, session_id="0train")
    with pytest.raises(ValueError, match="not a run mapping"):
        bnci._single_session_run(marker, session_id="0train")


def test_shrinkage_makes_rank_deficient_three_channel_windows_spd() -> None:
    time = np.linspace(0.0, 1.0, bnci.EPOCH_SAMPLES, dtype=np.float32)
    rank_one = np.stack((time, 2.0 * time, -time), axis=0)
    filter_bank = np.broadcast_to(rank_one, (2, len(bnci.BANDS), *rank_one.shape))

    covariance = bnci.make_spd_covariances(filter_bank, dtype=np.float32)

    assert covariance.shape == (2, len(bnci.BANDS), 3, 3)
    assert np.all(np.linalg.eigvalsh(covariance) > 0.0)
