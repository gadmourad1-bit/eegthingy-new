from dataclasses import replace

import numpy as np
import pytest

from benchmark.research.config import CHANNELS, EPOCH_WINDOWS, SUBJECT_RUNS, DataConfig
from benchmark.research.data import (
    SessionData,
    SessionKey,
    annotation_label,
    cache_key,
    dataset_contract,
    make_spd_covariances,
    preprocessing_fingerprint,
    session_path,
    valid_sessions,
)


def _synthetic_session(
    key: SessionKey, n_epochs: int = 4, source_file: str | None = None
) -> SessionData:
    rng = np.random.default_rng(key.subject * 100 + key.run)
    epochs = rng.normal(size=(n_epochs, 4, 15, 21)).astype(np.float32)
    broadband = rng.normal(size=(n_epochs, 15, 21)).astype(np.float32)
    covariance = make_spd_covariances(epochs)
    return SessionData(
        epochs=epochs,
        broadband_epochs=broadband,
        covariances=covariance,
        labels=np.asarray([0, 1, 0, -1], dtype=np.int64),
        subject_ids=np.full(n_epochs, key.subject),
        run_ids=np.full(n_epochs, key.run),
        session_ids=np.full(n_epochs, key.session_id),
        event_samples=np.arange(n_epochs) * 250,
        event_onsets=np.arange(n_epochs, dtype=np.float64) * 2.0,
        trial_ids=np.arange(1, n_epochs + 1),
        annotations=np.asarray(
            ["left_hand/task/t1", "right_hand/task/t2", "left_hand/task/t3", "left_hand/rest/t4"]
        ),
        source_files=np.full(
            n_epochs, source_file or f"/synthetic/{key.session_id}.fif"
        ),
    )


def test_manifest_is_explicit_and_excludes_bad_recordings() -> None:
    sessions = valid_sessions()
    assert len(sessions) == 32
    assert tuple(sorted({key.subject for key in sessions})) == (1, 3, 4, 5, 6, 7, 8, 10)
    assert all(key.subject != 9 for key in sessions)
    assert tuple(key.run for key in sessions if key.subject == 10) == (5, 6, 7, 8)
    assert SUBJECT_RUNS[1] == (1, 2, 3, 4)


def test_manifest_path_hard_rejects_excluded_subject_and_bad_s10_run(tmp_path) -> None:
    config = DataConfig(data_dir=tmp_path, cache_dir=tmp_path / "cache")
    assert session_path(SessionKey(10, 5), config).name == (
        "exp4_subject10_training_5_mi_raw.fif"
    )
    with pytest.raises(ValueError, match="excluded"):
        session_path(SessionKey(9, 1), config)
    with pytest.raises(ValueError, match="invalid for subject 10"):
        session_path(SessionKey(10, 1), config)


def test_named_windows_include_deployment_and_legacy() -> None:
    deployment = DataConfig(window_name="deployment")
    legacy = DataConfig(window_name="legacy")
    assert EPOCH_WINDOWS["deployment"] == (0.0, 2.0)
    assert EPOCH_WINDOWS["legacy"] == (0.5, 2.5)
    assert deployment.epoch_samples == legacy.epoch_samples == 251
    assert len(CHANNELS) == 15


def test_cache_fingerprint_changes_for_preprocessing_and_window_settings(tmp_path) -> None:
    base = DataConfig(data_dir=tmp_path, cache_dir=tmp_path / "cache")
    variants = (
        replace(base, window_name="legacy"),
        replace(base, include_rest=True),
        replace(base, covariance_shrinkage=0.01),
        replace(base, filter_phase="zero"),
        replace(base, dtype="float64"),
    )
    fingerprints = {preprocessing_fingerprint(base)} | {
        preprocessing_fingerprint(config) for config in variants
    }
    keys = {cache_key(SessionKey(1, 1), config) for config in (base, *variants)}
    assert len(fingerprints) == len(variants) + 1
    assert len(keys) == len(variants) + 1
    assert "win-deployment" in cache_key(SessionKey(1, 1), base)
    assert "win-legacy" in cache_key(SessionKey(1, 1), variants[0])


def test_dataset_contract_locks_preprocessing_and_file_content(tmp_path) -> None:
    source = tmp_path / "recording.fif"
    source.write_bytes(b"first recording bytes")
    config = DataConfig(data_dir=tmp_path, cache_dir=tmp_path / "cache")
    data = _synthetic_session(SessionKey(1, 1), source_file=str(source))
    first = dataset_contract(data, config)
    assert first["sessions"][0]["source_file"] == "recording.fif"
    assert first["preprocessing"]["channels"] == list(CHANNELS)

    source.write_bytes(b"different recording bytes")
    second = dataset_contract(data, config)
    assert first["sessions"][0]["sha256"] != second["sessions"][0]["sha256"]
    assert first["contract_sha256"] != second["contract_sha256"]


def test_hierarchical_annotation_mapping_and_optional_neutral() -> None:
    assert annotation_label(" Left_Hand/Task/t1 ") == 0
    assert annotation_label("right_hand/task/t2") == 1
    assert annotation_label("left_hand/rest/t1") is None
    assert annotation_label("right_hand/rest/t2", include_rest=True) == -1
    assert annotation_label("left_foot/task/t3", include_rest=True) is None


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_covariances_are_symmetric_positive_definite_and_keep_precision(dtype) -> None:
    rng = np.random.default_rng(5)
    epochs = rng.normal(size=(6, 4, 15, 31)).astype(dtype)
    epochs[..., -1, :] = epochs[..., 0, :]  # deliberately rank deficient
    covariance = make_spd_covariances(epochs, shrinkage=1e-3)
    assert covariance.dtype == dtype
    np.testing.assert_allclose(covariance, covariance.swapaxes(-1, -2), rtol=1e-6, atol=0.0)
    eigenvalues = np.linalg.eigvalsh(covariance.astype(np.float64))
    assert np.all(eigenvalues > 0.0)


def test_degenerate_zero_epochs_receive_a_scale_aware_spd_floor() -> None:
    covariance = make_spd_covariances(np.zeros((2, 4, 15, 20), dtype=np.float32))
    assert np.all(np.linalg.eigvalsh(covariance.astype(np.float64)) > 0.0)


def test_session_data_concatenation_subset_and_metadata() -> None:
    first = _synthetic_session(SessionKey(1, 1))
    second = _synthetic_session(SessionKey(3, 1))
    combined = SessionData.concatenate((first, second))
    assert len(combined) == 8
    assert combined.key is None
    assert set(combined.metadata) >= {"subject_ids", "run_ids", "session_ids"}
    assert combined.task_mask.sum() == 6
    assert combined.neutral_mask.sum() == 2
    subset = combined.subset(np.asarray([0, 4]))
    assert subset.subject_ids.tolist() == [1, 3]
    np.testing.assert_array_equal(subset.X, combined.covariances[[0, 4]])
    np.testing.assert_array_equal(subset.y, combined.labels[[0, 4]])
