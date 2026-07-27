from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

import ieee_mi.data as data_module
from ieee_mi.config import (
    BNCI004_3_CHANNELS,
    DATASETS,
    DEFAULT_MONTAGE_PROFILE,
    LOCAL_EXP4_15_CHANNELS,
    LOCAL_EXP4_PREPROCESSING,
    LOCAL_EXP4_SOURCE_CHANNELS,
    LOCAL_EXP4_SUBJECT_RUNS,
    MONTAGE_PROFILES,
    PREPROCESSING,
    channels_for_dataset,
    coordinate_contract_for_dataset,
    preprocessing_for_dataset,
)
from ieee_mi.data import (
    PHYSIONET_MI_RUNS,
    _atlas_unit_positions,
    _cache_path,
    _canonical_run_ids,
    _inclusive_api_tmax,
    _require_exact_epoch_length,
    _select_half_open_epoch_samples,
    _unit_positions,
    apply_channel_scaler,
    build_development_caches,
    build_subject_cache,
    canonicalize_standard_1005_channels,
    fit_channel_scaler,
    load_subject_cache,
    reflection_index,
    split_indices,
)


def _local_split_metadata(subject: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    runs = LOCAL_EXP4_SUBJECT_RUNS[subject]
    labels = np.tile(np.asarray((0, 1), dtype=np.int64), 30 * len(runs))
    run_rows = np.repeat(np.asarray([str(run) for run in runs]), 60)
    sessions = np.concatenate(
        [np.asarray([f"S{subject:02d}-R{run:02d}"] * 60) for run in runs]
    )
    return labels, sessions, run_rows


def _write_synthetic_local_run(root: Path, subject: int, run: int) -> Path:
    import mne

    sfreq = 125.0
    onsets = 1.0 + np.arange(60, dtype=np.float64) * 2.25
    n_samples = int(np.ceil((onsets[-1] + 2.25) * sfreq)) + 1
    rng = np.random.default_rng(1000 + run)
    values = rng.normal(
        scale=2e-6,
        size=(len(LOCAL_EXP4_SOURCE_CHANNELS), n_samples),
    )
    # This deliberately exceeds the legacy 100-uV rejection threshold.  The
    # harmonized cache must retain it because artifact rejection is disabled.
    values[0, int(round(onsets[0] * sfreq)) + 20] += 5e-3
    info = mne.create_info(
        list(LOCAL_EXP4_SOURCE_CHANNELS), sfreq=sfreq, ch_types="eeg"
    )
    raw = mne.io.RawArray(values, info, verbose="ERROR")
    raw.set_montage("standard_1020", verbose="ERROR")
    descriptions = [
        f"{'left_hand' if trial % 2 else 'right_hand'}/task/t{trial}"
        for trial in range(1, 61)
    ]
    raw.set_annotations(
        mne.Annotations(
            onset=onsets,
            duration=np.full(60, 2.1),
            description=descriptions,
        )
    )
    output = root / f"exp4_subject{subject}_training_{run}_mi_raw.fif"
    raw.save(output, overwrite=True, verbose="ERROR")
    return output


def _write_synthetic_native_public_cache(
    root: Path,
    *,
    tamper_coordinates: bool = False,
) -> tuple[Path, tuple[str, ...]]:
    dataset = "bnci2014_001"
    subject = 1
    channel_names = ("Fp1", "F3", "C3", "Cz", "C4", "Pz", "O1")
    rng = np.random.default_rng(20260721)
    values = rng.normal(size=(8, len(channel_names), 320)).astype(np.float32)
    labels = np.tile(np.arange(4, dtype=np.int64), 2)
    positions = _atlas_unit_positions(channel_names)
    if tamper_coordinates:
        positions = positions.copy()
        positions[0] = positions[1]
    sessions = np.asarray(["0train"] * 8, dtype="U")
    runs = np.asarray(["0"] * 8, dtype="U")
    identity = {
        "dataset": asdict(DATASETS[dataset]),
        "subject": subject,
        "montage_profile": "native",
        "preprocessing": preprocessing_for_dataset(dataset),
        "coordinates": coordinate_contract_for_dataset(
            dataset,
            "native",
            channel_names,
        ),
        "shape": list(values.shape),
        "channels": list(channel_names),
    }
    digest = hashlib.sha256()
    for array in (values, labels, positions, sessions.astype("U"), runs.astype("U")):
        digest.update(np.ascontiguousarray(array).tobytes())
    identity["array_sha256"] = digest.hexdigest()
    output = _cache_path(root, dataset, subject, "native")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        x=values,
        y=labels,
        positions=positions,
        channel_names=np.asarray(channel_names),
        sessions=sessions,
        runs=runs,
        identity=np.asarray(
            json.dumps(identity, sort_keys=True, separators=(",", ":"))
        ),
    )
    return output, channel_names


def test_development_and_confirmation_cohorts_are_disjoint_and_complete() -> None:
    for spec in DATASETS.values():
        assert not set(spec.development_subjects) & set(spec.confirmation_subjects)
        assert set(spec.development_subjects) | set(spec.confirmation_subjects) == set(
            spec.subjects
        )
    assert PREPROCESSING["n_times"] == 320
    assert PREPROCESSING["schema"] == "ieee-mi-cache-v2"


def test_public_epoch_contract_selects_the_verified_half_open_grid() -> None:
    assert PREPROCESSING["tmin_seconds"] == 0.5
    assert PREPROCESSING["tmax_seconds_exclusive"] == 3.0
    assert PREPROCESSING["epoch_interval_semantics"] == (
        "half-open [tmin_seconds, tmax_seconds_exclusive)"
    )
    assert _inclusive_api_tmax(PREPROCESSING) == pytest.approx(3.0)
    inclusive_times = 0.5 + np.arange(321, dtype=np.float64) / 128.0
    inclusive = np.zeros((2, 3, 321), dtype=np.float32)
    selected = _select_half_open_epoch_samples(
        inclusive,
        inclusive_times,
        contract=PREPROCESSING,
        source="synthetic",
    )
    assert selected.shape == (2, 3, 320)
    bnci_contract = preprocessing_for_dataset("bnci2014_001")
    assert bnci_contract["moabb_interval_start_seconds"] == 2.0
    bnci_times = 2.5 + np.arange(321, dtype=np.float64) / 128.0
    assert _select_half_open_epoch_samples(
        inclusive,
        bnci_times,
        contract=bnci_contract,
        source="synthetic BNCI2014-001",
    ).shape == (2, 3, 320)
    with pytest.raises(RuntimeError, match="declared half-open grid"):
        _select_half_open_epoch_samples(
            inclusive,
            inclusive_times,
            contract=bnci_contract,
            source="wrong-origin BNCI2014-001",
        )
    exact = np.zeros((2, 3, 320), dtype=np.float32)
    assert _require_exact_epoch_length(
        exact, n_times=320, source="synthetic"
    ).shape == (2, 3, 320)
    with pytest.raises(RuntimeError, match="321 samples; expected exactly 320"):
        _require_exact_epoch_length(
            np.zeros((2, 3, 321), dtype=np.float32),
            n_times=320,
            source="synthetic",
        )


def test_all_configured_coordinates_come_from_one_standard_1005_atlas() -> None:
    import mne

    for dataset in DATASETS:
        names = channels_for_dataset(dataset)
        source_info = mne.create_info(list(names), sfreq=128.0, ch_types="eeg")
        # Poison otherwise valid source-frame locations. The cache must ignore
        # these and deterministically resolve channel names through its atlas.
        for index, channel in enumerate(source_info["chs"], start=1):
            channel["loc"][:3] = np.asarray((index, index + 1, index + 2))
        resolved_names, resolved = _unit_positions(source_info)
        expected = _atlas_unit_positions(names)
        assert resolved_names == names
        np.testing.assert_allclose(resolved, expected, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            np.linalg.norm(resolved, axis=1), 1.0, rtol=0.0, atol=1e-6
        )


def test_native_channel_preflight_normalizes_common_eegbci_names() -> None:
    assert canonicalize_standard_1005_channels(
        ("fc5.", " T3 ", "t5.", "fp1")
    ) == ("FC5", "T7", "P7", "Fp1")
    with pytest.raises(ValueError, match="absent or ambiguous"):
        canonicalize_standard_1005_channels(("C3", "not-an-electrode"))
    with pytest.raises(ValueError, match="collide"):
        canonicalize_standard_1005_channels(("T3", "T7"))


def test_bnci004_coordinates_are_nominal_bipolar_centers_not_point_evidence() -> None:
    assert channels_for_dataset("bnci2014_004") == BNCI004_3_CHANNELS
    contract = coordinate_contract_for_dataset("bnci2014_004")
    assert contract["atlas"] == "MNE standard_1005"
    assert contract["native_recording_coordinates_used"] is False
    assert contract["channel_semantics"] == (
        "nominal centers of supplied bipolar derivations"
    )
    assert contract["point_electrode_continuity_evidence"] is False
    assert "must not be used" in str(contract["restriction"])
    assert coordinate_contract_for_dataset("bnci2014_001")[
        "point_electrode_continuity_evidence"
    ] is True


def test_montage_profiles_separate_locked_and_dynamic_channel_contracts() -> None:
    assert MONTAGE_PROFILES == ("harmonized", "native")
    assert DEFAULT_MONTAGE_PROFILE == "harmonized"
    assert channels_for_dataset("cho2017") == channels_for_dataset(
        "cho2017", "harmonized"
    )
    assert channels_for_dataset("cho2017", "native") is None
    assert channels_for_dataset("local_exp4", "native") == LOCAL_EXP4_15_CHANNELS
    with pytest.raises(PermissionError, match="bipolar derivations are ineligible"):
        channels_for_dataset("bnci2014_004", "native")

    native_channels = ("F3", "C3", "Cz", "C4", "Pz")
    contract = coordinate_contract_for_dataset(
        "cho2017",
        "native",
        native_channels,
    )
    assert contract["montage_profile"] == "native"
    assert contract["ordered_channels"] == list(native_channels)
    assert contract["signal_channel_selection"] == (
        "all native dataset EEG channels; no interpolation or subsetting"
    )
    with pytest.raises(ValueError, match="requires its observed channel order"):
        coordinate_contract_for_dataset("cho2017", "native")


def test_cache_paths_are_separated_by_montage_profile(tmp_path: Path) -> None:
    harmonized = _cache_path(tmp_path, "cho2017", 1, "harmonized")
    native = _cache_path(tmp_path, "cho2017", 1, "native")
    assert harmonized != native
    assert harmonized.parts[-3:-1] == ("harmonized", "cho2017")
    assert native.parts[-3:-1] == ("native", "cho2017")


def test_native_loader_validates_dynamic_ordered_channels_and_atlas(
    tmp_path: Path,
) -> None:
    output, channel_names = _write_synthetic_native_public_cache(tmp_path)
    assert output.is_file()
    cached = load_subject_cache(
        "bnci2014_001",
        1,
        cache_root=tmp_path,
        montage_profile="native",
    )
    assert tuple(cached["channel_names"].tolist()) == channel_names
    assert cached["x"].shape == (8, len(channel_names), 320)
    assert cached["identity"]["coordinates"]["ordered_channels"] == list(
        channel_names
    )
    assert cached["identity"]["montage_profile"] == "native"
    with pytest.raises(FileNotFoundError):
        load_subject_cache("bnci2014_001", 1, cache_root=tmp_path)

    _write_synthetic_native_public_cache(tmp_path, tamper_coordinates=True)
    with pytest.raises(RuntimeError, match="do not match standard_1005"):
        load_subject_cache(
            "bnci2014_001",
            1,
            cache_root=tmp_path,
            montage_profile="native",
        )


def test_cho_protocol_names_within_class_acquisition_order_blocks() -> None:
    assert DATASETS["cho2017"].protocol == (
        "rotating_within_class_acquisition_order_block_holdout"
    )


def test_local_exp4_manifest_and_preprocessing_are_frozen() -> None:
    assert DATASETS["local_exp4"].development_subjects == (1, 3, 4, 5, 6, 7, 8, 10)
    assert DATASETS["local_exp4"].confirmation_subjects == ()
    assert dict(LOCAL_EXP4_SUBJECT_RUNS) == {
        1: (1, 2, 3, 4),
        3: (1, 2, 3, 4),
        4: (1, 2, 3, 4),
        5: (1, 2, 3, 4),
        6: (1, 2, 3, 4),
        7: (1, 2, 3, 4),
        8: (1, 2, 3, 4),
        10: (5, 6, 7, 8),
    }
    assert LOCAL_EXP4_15_CHANNELS == (
        "Cz",
        "Pz",
        "C3",
        "C4",
        "P7",
        "P8",
        "Fz",
        "F7",
        "F8",
        "F3",
        "F4",
        "T7",
        "T8",
        "P3",
        "P4",
    )
    contract = preprocessing_for_dataset("local_exp4")
    assert contract == LOCAL_EXP4_PREPROCESSING
    assert contract["fmin_hz"] == 4.0
    assert contract["fmax_hz"] == 40.0
    assert contract["sfreq_hz"] == 128.0
    assert contract["tmin_seconds"] == 0.0
    assert contract["tmax_seconds_exclusive"] == 2.0
    assert contract["n_times"] == 256
    assert contract["artifact_rejection"] is None


@pytest.mark.parametrize(
    ("subject", "expected_train", "expected_validation", "expected_test"),
    (
        (1, {"1", "2"}, {"3"}, {"4"}),
        (10, {"5", "6"}, {"7"}, {"8"}),
    ),
)
def test_local_exp4_split_uses_fixed_chronological_manifest_runs(
    subject: int,
    expected_train: set[str],
    expected_validation: set[str],
    expected_test: set[str],
) -> None:
    labels, sessions, runs = _local_split_metadata(subject)
    train, validation, test = split_indices(
        "local_exp4", labels, sessions, runs, fold=0, subject=subject
    )
    assert (len(train), len(validation), len(test)) == (120, 60, 60)
    assert set(runs[train]) == expected_train
    assert set(runs[validation]) == expected_validation
    assert set(runs[test]) == expected_test
    assert set(train).isdisjoint(validation)
    assert set(train).isdisjoint(test)
    assert set(validation).isdisjoint(test)
    for indices in (train, validation, test):
        assert np.bincount(labels[indices], minlength=2).tolist() == [len(indices) // 2] * 2


def test_local_exp4_split_rejects_any_alternate_fold_or_run_mapping() -> None:
    labels, sessions, runs = _local_split_metadata(10)
    with pytest.raises(ValueError, match="only the fixed chronological fold 0"):
        split_indices("local_exp4", labels, sessions, runs, fold=1, subject=10)
    changed = runs.copy()
    changed[changed == "8"] = "4"
    with pytest.raises(RuntimeError, match="requires runs"):
        split_indices("local_exp4", labels, sessions, changed, fold=0, subject=10)


def test_local_exp4_cache_harmonizes_all_four_runs_without_rejection(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    for run in LOCAL_EXP4_SUBJECT_RUNS[1]:
        _write_synthetic_local_run(raw_root, 1, run)

    cache_root = tmp_path / "cache"
    output = build_subject_cache(
        "local_exp4",
        1,
        cache_root=cache_root,
        stage="development",
        local_data_root=raw_root,
    )
    assert output.is_file()
    cached = load_subject_cache("local_exp4", 1, cache_root=cache_root)
    assert cached["x"].shape == (240, 15, 256)
    assert cached["x"].dtype == np.float32
    assert cached["y"].tolist().count(0) == 120
    assert cached["y"].tolist().count(1) == 120
    assert tuple(cached["channel_names"].tolist()) == LOCAL_EXP4_15_CHANNELS
    assert cached["runs"].tolist() == [str(run) for run in (1, 2, 3, 4) for _ in range(60)]
    assert cached["sessions"].tolist() == [
        f"S01-R{run:02d}" for run in (1, 2, 3, 4) for _ in range(60)
    ]
    assert np.all(np.isfinite(cached["x"]))
    assert np.allclose(cached["x"].mean(axis=2), 0.0, atol=1e-10)
    assert np.allclose(cached["x"].mean(axis=1), 0.0, atol=1e-10)
    assert np.allclose(np.linalg.norm(cached["positions"], axis=1), 1.0, atol=1e-6)
    assert cached["identity"]["preprocessing"] == LOCAL_EXP4_PREPROCESSING
    assert cached["identity"]["montage_profile"] == "harmonized"
    assert cached["identity"]["coordinates"] == coordinate_contract_for_dataset(
        "local_exp4"
    )
    assert [row["run"] for row in cached["identity"]["source_manifest"]] == [1, 2, 3, 4]
    assert all(len(row["sha256"]) == 64 for row in cached["identity"]["source_manifest"])


def test_local_exp4_is_development_only_before_any_file_access(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    with pytest.raises(PermissionError, match="development cohort"):
        build_subject_cache(
            "local_exp4",
            1,
            cache_root=cache_root,
            stage="confirmation",
            local_data_root=tmp_path / "does-not-exist",
        )
    assert not cache_root.exists()


def test_bnci001_split_is_chronological_and_run_grouped() -> None:
    labels = np.tile(np.arange(4), 24)
    sessions = np.asarray(["0train"] * 48 + ["1test"] * 48)
    runs = np.asarray(
        ["0"] * 8 + ["1"] * 8 + ["2"] * 8 + ["3"] * 8 + ["4"] * 8 + ["5"] * 8
        + ["0"] * 48
    )
    train, validation, test = split_indices(
        "bnci2014_001", labels, sessions, runs
    )
    assert set(sessions[train]) == {"0train"}
    assert set(runs[validation]) == {"5"}
    assert set(sessions[test]) == {"1test"}


def test_bnci004_split_uses_only_predeclared_sessions() -> None:
    labels = np.tile(np.arange(2), 50)
    sessions = np.repeat(
        np.asarray(("0train", "1train", "2train", "3test", "4test")), 20
    )
    runs = np.asarray(["0"] * len(labels))
    train, validation, test = split_indices(
        "bnci2014_004", labels, sessions, runs
    )
    assert set(sessions[train]) == {"0train", "1train"}
    assert set(sessions[validation]) == {"2train"}
    assert set(sessions[test]) == {"3test", "4test"}


@pytest.mark.parametrize("fold", range(5))
def test_cho_split_uses_within_class_acquisition_order_blocks(fold: int) -> None:
    labels = np.repeat(np.arange(2), 100).astype(np.int64)
    sessions = np.asarray(["0"] * len(labels))
    runs = np.asarray(["0"] * len(labels))
    train, validation, test = split_indices(
        "cho2017", labels, sessions, runs, fold=fold
    )
    assert len(set(train) | set(validation) | set(test)) == len(labels)
    for indices in (train, validation, test):
        assert set(labels[indices]) == {0, 1}
    for label in (0, 1):
        class_rows = np.flatnonzero(labels == label)
        blocks = [np.asarray(block) for block in np.array_split(class_rows, 5)]
        assert set(test[labels[test] == label]) == set(blocks[fold])
        assert set(validation[labels[validation] == label]) == set(
            blocks[(fold + 1) % 5]
        )


@pytest.mark.parametrize(
    ("fold", "expected_train", "expected_validation", "expected_test"),
    (
        (0, "12", "8", "4"),
        (1, "4", "12", "8"),
        (2, "8", "4", "12"),
    ),
)
def test_physionet_development_split_rotates_whole_imagery_runs(
    fold: int,
    expected_train: str,
    expected_validation: str,
    expected_test: str,
) -> None:
    # Synthetic metadata only: three balanced acquisition runs, deliberately
    # interleaved in row order so the test catches trial-level slicing.
    runs = np.tile(np.asarray(PHYSIONET_MI_RUNS), 10)
    labels = np.repeat(np.arange(10) % 2, len(PHYSIONET_MI_RUNS)).astype(np.int64)
    sessions = np.asarray(["0"] * len(labels))

    train, validation, test = split_indices(
        "physionet_mi",
        labels,
        sessions,
        runs,
        fold=fold,
        subject=17,
    )

    assert set(runs[train]) == {expected_train}
    assert set(runs[validation]) == {expected_validation}
    assert set(runs[test]) == {expected_test}
    assert set(train).isdisjoint(validation)
    assert set(train).isdisjoint(test)
    assert set(validation).isdisjoint(test)
    assert set(train) | set(validation) | set(test) == set(range(len(labels)))
    for indices in (train, validation, test):
        assert set(labels[indices]) == {0, 1}


def test_physionet_moabb_run_keys_are_recorded_as_source_imagery_runs() -> None:
    library_runs = np.asarray(("0", "1", "2", "0", "2"))
    source_runs = _canonical_run_ids("physionet_mi", library_runs)
    assert source_runs.tolist() == ["4", "8", "12", "4", "12"]


def test_physionet_development_split_requires_subject_identity() -> None:
    labels = np.tile(np.asarray((0, 1)), 6).astype(np.int64)
    sessions = np.asarray(["0"] * len(labels))
    runs = np.repeat(np.asarray(PHYSIONET_MI_RUNS), 4)
    with pytest.raises(ValueError, match="explicit subject ID"):
        split_indices("physionet_mi", labels, sessions, runs, fold=0)


@pytest.mark.parametrize("fold", (-1, 3))
def test_physionet_development_split_rejects_invalid_fold(fold: int) -> None:
    labels = np.tile(np.asarray((0, 1)), 6).astype(np.int64)
    sessions = np.asarray(["0"] * len(labels))
    runs = np.repeat(np.asarray(PHYSIONET_MI_RUNS), 4)
    with pytest.raises(ValueError, match="fold must be in"):
        split_indices(
            "physionet_mi",
            labels,
            sessions,
            runs,
            fold=fold,
            subject=1,
        )


def test_physionet_split_rejects_non_imagery_or_missing_run_groups() -> None:
    labels = np.tile(np.asarray((0, 1)), 6).astype(np.int64)
    sessions = np.asarray(["0"] * len(labels))
    runs = np.repeat(np.asarray(("4", "8", "10")), 4)
    with pytest.raises(RuntimeError, match="exactly source imagery runs 4/8/12"):
        split_indices(
            "physionet_mi",
            labels,
            sessions,
            runs,
            fold=0,
            subject=1,
        )


def test_physionet_split_and_cache_guard_reject_confirmation_subject(
    tmp_path: Path,
) -> None:
    labels = np.tile(np.asarray((0, 1)), 6).astype(np.int64)
    sessions = np.asarray(["0"] * len(labels))
    runs = np.repeat(np.asarray(PHYSIONET_MI_RUNS), 4)
    with pytest.raises(PermissionError, match="outside the development cohort"):
        split_indices(
            "physionet_mi",
            labels,
            sessions,
            runs,
            fold=0,
            subject=55,
        )
    with pytest.raises(PermissionError, match="frozen confirmation"):
        build_subject_cache(
            "physionet_mi",
            55,
            cache_root=tmp_path,
            stage="development",
        )
    assert not any(tmp_path.iterdir())


def test_development_cache_list_preflights_every_subject_before_building(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int, str]] = []

    def fake_build(
        dataset: str,
        subject: int,
        *,
        cache_root: str | Path,
        stage: str,
        montage_profile: str,
        overwrite: bool,
        local_data_root: str | Path | None,
    ) -> Path:
        del overwrite, local_data_root
        assert montage_profile == "harmonized"
        calls.append((dataset, subject, stage))
        return Path(cache_root) / f"subject_{subject:03d}.npz"

    monkeypatch.setattr(data_module, "build_subject_cache", fake_build)
    outputs = build_development_caches(
        "physionet_mi",
        (2, 1),
        cache_root=tmp_path,
    )
    assert calls == [
        ("physionet_mi", 2, "development"),
        ("physionet_mi", 1, "development"),
    ]
    assert outputs == (
        tmp_path / "subject_002.npz",
        tmp_path / "subject_001.npz",
    )

    calls.clear()
    with pytest.raises(PermissionError, match="refuses confirmation subjects"):
        build_development_caches(
            "physionet_mi",
            (1, 55),
            cache_root=tmp_path / "must-not-exist",
        )
    assert calls == []
    assert not (tmp_path / "must-not-exist").exists()

    with pytest.raises(PermissionError, match="ineligible for native-montage"):
        build_development_caches(
            "bnci2014_004",
            (1,),
            cache_root=tmp_path / "bipolar-must-not-exist",
            montage_profile="native",
        )
    assert calls == []
    assert not (tmp_path / "bipolar-must-not-exist").exists()


def test_development_cache_cli_parses_declared_list_and_refuses_confirmation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, object] = {}

    def fake_build_list(
        dataset: str,
        subjects: tuple[int, ...],
        *,
        cache_root: str | Path,
        montage_profile: str,
        overwrite: bool,
        local_data_root: str | Path | None,
    ) -> tuple[Path, ...]:
        captured.update(
            dataset=dataset,
            subjects=subjects,
            cache_root=Path(cache_root),
            montage_profile=montage_profile,
            overwrite=overwrite,
            local_data_root=local_data_root,
        )
        return (Path(cache_root) / "subject_003.npz",)

    monkeypatch.setattr(data_module, "build_development_caches", fake_build_list)
    assert data_module.main(
        [
            "--dataset",
            "cho2017",
            "--subjects",
            "3,1",
            "--cache-root",
            str(tmp_path),
            "--montage-profile",
            "native",
            "--overwrite",
        ]
    ) == 0
    assert captured == {
        "dataset": "cho2017",
        "subjects": (3, 1),
        "cache_root": tmp_path,
        "montage_profile": "native",
        "overwrite": True,
        "local_data_root": None,
    }
    assert "subject_003.npz" in capsys.readouterr().out

    monkeypatch.setattr(
        data_module,
        "build_development_caches",
        build_development_caches,
    )
    with pytest.raises(SystemExit) as error:
        data_module.main(
            [
                "--dataset",
                "physionet_mi",
                "--subjects",
                "55",
                "--cache-root",
                str(tmp_path / "sealed"),
            ]
        )
    assert error.value.code == 2
    assert "refuses confirmation subjects" in capsys.readouterr().err
    assert not (tmp_path / "sealed").exists()


def test_scaler_is_source_only_and_channelwise() -> None:
    rng = np.random.default_rng(7)
    train = rng.normal(size=(8, 3, 20)).astype(np.float32)
    test = rng.normal(loc=100.0, size=(4, 3, 20)).astype(np.float32)
    mean, std = fit_channel_scaler(train)
    scaled_train = apply_channel_scaler(train, mean, std)
    scaled_test = apply_channel_scaler(test, mean, std)
    assert np.allclose(scaled_train.mean(axis=(0, 2)), 0.0, atol=1e-6)
    assert np.allclose(scaled_train.std(axis=(0, 2)), 1.0, atol=1e-6)
    assert scaled_test.mean() > 5.0


def test_scaler_standardises_microvolt_eeg() -> None:
    rng = np.random.default_rng(11)
    train = (rng.normal(size=(24, 3, 128)) * 3e-6).astype(np.float32)
    mean, std = fit_channel_scaler(train)
    scaled = apply_channel_scaler(train, mean, std)

    assert np.all(std < 1e-5)
    assert np.allclose(scaled.mean(axis=(0, 2)), 0.0, atol=1e-6)
    assert np.allclose(scaled.std(axis=(0, 2)), 1.0, atol=1e-6)


def test_scaler_ties_reflected_channel_pairs() -> None:
    rng = np.random.default_rng(11)
    train = rng.normal(size=(10, 3, 30)).astype(np.float32)
    train[:, 0] += 4.0
    train[:, 2] -= 2.0
    mean, std = fit_channel_scaler(train, ("C3", "Cz", "C4"))
    assert mean[0, 0, 0] == mean[0, 2, 0]
    assert std[0, 0, 0] == std[0, 2, 0]


def test_reflection_index_is_an_exact_involution() -> None:
    names = ("Fz", "FC3", "FC4", "C3", "Cz", "C4")
    index = reflection_index(names)
    assert index.tolist() == [0, 2, 1, 5, 4, 3]
    assert np.array_equal(index[index], np.arange(len(names)))


def test_local_exp4_reflection_swaps_every_lateral_pair() -> None:
    index = reflection_index(LOCAL_EXP4_15_CHANNELS)
    reflected = {
        name: LOCAL_EXP4_15_CHANNELS[int(index[position])]
        for position, name in enumerate(LOCAL_EXP4_15_CHANNELS)
    }
    assert reflected == {
        "Cz": "Cz",
        "Pz": "Pz",
        "C3": "C4",
        "C4": "C3",
        "P7": "P8",
        "P8": "P7",
        "Fz": "Fz",
        "F7": "F8",
        "F8": "F7",
        "F3": "F4",
        "F4": "F3",
        "T7": "T8",
        "T8": "T7",
        "P3": "P4",
        "P4": "P3",
    }
    assert np.array_equal(index[index], np.arange(len(index)))
