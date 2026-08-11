import numpy as np
import pytest

from deepnet.protocols import (
    LeakageError,
    ProtocolSplit,
    chronological_cross_session_splits,
    loso_splits,
    validate_split,
    within_subject_splits,
)


def _metadata() -> dict[str, np.ndarray]:
    # Two samples per run; S10's valid chronological run numbers begin at five.
    subject_ids: list[int] = []
    run_ids: list[int] = []
    session_ids: list[str] = []
    for subject, runs in ((1, (1, 2, 3, 4)), (10, (5, 6, 7, 8))):
        for run in runs:
            subject_ids.extend((subject, subject))
            run_ids.extend((run, run))
            session_ids.extend((f"S{subject:02d}-R{run:02d}",) * 2)
    return {
        "subject_ids": np.asarray(subject_ids),
        "run_ids": np.asarray(run_ids),
        "session_ids": np.asarray(session_ids),
    }


def test_within_subject_folds_hold_out_whole_sessions() -> None:
    metadata = _metadata()
    splits = within_subject_splits(metadata)
    assert len(splits) == 8
    for split in splits:
        train_subjects = set(metadata["subject_ids"][split.train].tolist())
        test_subjects = set(metadata["subject_ids"][split.test].tolist())
        train_sessions = set(metadata["session_ids"][split.train].tolist())
        test_sessions = set(metadata["session_ids"][split.test].tolist())
        assert train_subjects == test_subjects == {split.held_out_subject}
        assert train_sessions.isdisjoint(test_sessions)
        assert len(test_sessions) == 1
        validate_split(split, metadata, require_session_disjoint=True)


def test_chronological_cross_session_uses_first_three_valid_runs_then_last() -> None:
    metadata = _metadata()
    splits = chronological_cross_session_splits(metadata)
    assert len(splits) == 2
    by_subject = {split.held_out_subject: split for split in splits}
    assert set(metadata["run_ids"][by_subject[1].train].tolist()) == {1, 2, 3}
    assert set(metadata["run_ids"][by_subject[1].test].tolist()) == {4}
    assert set(metadata["run_ids"][by_subject[10].train].tolist()) == {5, 6, 7}
    assert set(metadata["run_ids"][by_subject[10].test].tolist()) == {8}


def test_chronological_cross_session_rejects_incomplete_subject() -> None:
    metadata = _metadata()
    keep = metadata["run_ids"] != 4
    incomplete = {name: values[keep] for name, values in metadata.items()}
    with pytest.raises(ValueError, match="exactly four valid runs"):
        chronological_cross_session_splits(incomplete)


def test_loso_never_shares_a_subject() -> None:
    metadata = _metadata()
    splits = loso_splits(metadata)
    assert len(splits) == 2
    for split in splits:
        train_subjects = set(metadata["subject_ids"][split.train].tolist())
        test_subjects = set(metadata["subject_ids"][split.test].tolist())
        assert train_subjects.isdisjoint(test_subjects)
        assert test_subjects == {split.held_out_subject}
        validate_split(split, metadata, require_subject_disjoint=True)


def test_protocol_split_hard_rejects_index_overlap_and_duplicates() -> None:
    with pytest.raises(LeakageError, match="row overlap"):
        ProtocolSplit("bad", "overlap", np.asarray([0, 1]), np.asarray([1, 2]))
    with pytest.raises(LeakageError, match="duplicate"):
        ProtocolSplit("bad", "duplicate", np.asarray([0, 0]), np.asarray([1]))


def test_validation_detects_group_overlap_even_when_rows_are_disjoint() -> None:
    metadata = {
        "subject_ids": np.asarray([1, 1, 1, 1]),
        "run_ids": np.asarray([1, 1, 2, 2]),
        "session_ids": np.asarray(["shared", "shared", "other", "other"]),
    }
    split = ProtocolSplit("bad", "session", np.asarray([0, 2]), np.asarray([1, 3]))
    with pytest.raises(LeakageError, match="session overlap"):
        validate_split(split, metadata, require_session_disjoint=True)
    with pytest.raises(LeakageError, match="subject overlap"):
        validate_split(split, metadata, require_subject_disjoint=True)
