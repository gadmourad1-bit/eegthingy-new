"""Leakage-resistant evaluation protocols based on subject and recording groups."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


class LeakageError(ValueError):
    """Raised when a proposed evaluation split shares a protected group."""


@dataclass(frozen=True)
class ProtocolSplit:
    """Integer row indices and the group deliberately held out for one fold."""

    protocol: str
    fold: str
    train_indices: NDArray[np.int64]
    test_indices: NDArray[np.int64]
    held_out_subject: int | None = None
    held_out_sessions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        train = np.asarray(self.train_indices, dtype=np.int64).reshape(-1)
        test = np.asarray(self.test_indices, dtype=np.int64).reshape(-1)
        object.__setattr__(self, "train_indices", train)
        object.__setattr__(self, "test_indices", test)
        object.__setattr__(
            self, "held_out_sessions", tuple(str(value) for value in self.held_out_sessions)
        )
        if not self.protocol or not self.fold:
            raise ValueError("protocol and fold names must be non-empty")
        if len(train) == 0 or len(test) == 0:
            raise ValueError("both training and test partitions must be non-empty")
        if np.any(train < 0) or np.any(test < 0):
            raise ValueError("split indices must be non-negative")
        if len(np.unique(train)) != len(train) or len(np.unique(test)) != len(test):
            raise LeakageError("duplicate row indices are not allowed inside a partition")
        overlap = np.intersect1d(train, test, assume_unique=True)
        if len(overlap):
            raise LeakageError(f"train/test row overlap detected: {overlap[:10].tolist()}")

    @property
    def train(self) -> NDArray[np.int64]:
        return self.train_indices

    @property
    def test(self) -> NDArray[np.int64]:
        return self.test_indices


@dataclass(frozen=True)
class _Metadata:
    subjects: NDArray[np.int64]
    runs: NDArray[np.int64] | None
    sessions: NDArray[np.str_] | None

    @property
    def n_samples(self) -> int:
        return len(self.subjects)


def _read_column(source: Any, *names: str) -> Any | None:
    if isinstance(source, Mapping):
        for name in names:
            if name in source:
                return source[name]
        return None
    for name in names:
        if hasattr(source, name):
            return getattr(source, name)
    return None


def _coerce_metadata(
    data_or_subjects: Any,
    run_ids: Sequence[int] | None = None,
    session_ids: Sequence[str] | None = None,
    *,
    require_runs: bool = False,
    require_sessions: bool = False,
) -> _Metadata:
    """Accept SessionData, a metadata mapping, or explicit subject arrays."""

    embedded_subjects = _read_column(data_or_subjects, "subject_ids", "subjects", "subject")
    if embedded_subjects is None:
        embedded_subjects = data_or_subjects
    if run_ids is None:
        run_ids = _read_column(data_or_subjects, "run_ids", "runs", "run")
    if session_ids is None:
        session_ids = _read_column(data_or_subjects, "session_ids", "sessions", "session")

    subjects = np.asarray(embedded_subjects, dtype=np.int64).reshape(-1)
    if len(subjects) == 0:
        raise ValueError("metadata is empty")
    if np.any(subjects <= 0):
        raise ValueError("subject IDs must be positive")
    runs = None if run_ids is None else np.asarray(run_ids, dtype=np.int64).reshape(-1)
    sessions = (
        None if session_ids is None else np.asarray(session_ids, dtype=np.str_).reshape(-1)
    )
    if runs is not None and len(runs) != len(subjects):
        raise ValueError("run metadata length does not match subjects")
    if sessions is not None and len(sessions) != len(subjects):
        raise ValueError("session metadata length does not match subjects")
    if runs is not None and np.any(runs <= 0):
        raise ValueError("run IDs must be positive")
    if sessions is not None and np.any(np.char.str_len(sessions) == 0):
        raise ValueError("session IDs must be non-empty")
    if require_runs and runs is None:
        raise ValueError("run IDs are required for chronological cross-session evaluation")
    if require_sessions and sessions is None:
        raise ValueError("session IDs are required for session-held-out evaluation")
    return _Metadata(subjects=subjects, runs=runs, sessions=sessions)


def _compound_sessions(metadata: _Metadata, indices: NDArray[np.int64]) -> set[tuple[int, str]]:
    if metadata.sessions is None:
        if metadata.runs is None:
            raise ValueError("session or run metadata is required for leakage validation")
        values = metadata.runs.astype(str)
    else:
        values = metadata.sessions
    return {
        (int(metadata.subjects[index]), str(values[index])) for index in indices.tolist()
    }


def validate_split(
    split: ProtocolSplit,
    data_or_subjects: Any,
    run_ids: Sequence[int] | None = None,
    session_ids: Sequence[str] | None = None,
    *,
    require_session_disjoint: bool = False,
    require_subject_disjoint: bool = False,
) -> None:
    """Hard-fail on row, session, or subject leakage for a materialized split."""

    metadata = _coerce_metadata(data_or_subjects, run_ids, session_ids)
    all_indices = np.concatenate((split.train_indices, split.test_indices))
    if int(all_indices.max(initial=-1)) >= metadata.n_samples:
        raise ValueError("split index is outside the metadata array")
    # ProtocolSplit already rejects row intersection; repeat as defense for callers
    # that mutate the NumPy arrays after construction.
    row_overlap = np.intersect1d(split.train_indices, split.test_indices)
    if len(row_overlap):
        raise LeakageError(f"train/test row overlap detected: {row_overlap[:10].tolist()}")
    if require_session_disjoint:
        train_sessions = _compound_sessions(metadata, split.train_indices)
        test_sessions = _compound_sessions(metadata, split.test_indices)
        overlap = train_sessions & test_sessions
        if overlap:
            raise LeakageError(f"train/test session overlap detected: {sorted(overlap)[:10]}")
    if require_subject_disjoint:
        train_subjects = set(metadata.subjects[split.train_indices].tolist())
        test_subjects = set(metadata.subjects[split.test_indices].tolist())
        overlap = train_subjects & test_subjects
        if overlap:
            raise LeakageError(f"train/test subject overlap detected: {sorted(overlap)}")


def within_subject_splits(
    data_or_subjects: Any,
    run_ids: Sequence[int] | None = None,
    session_ids: Sequence[str] | None = None,
) -> tuple[ProtocolSplit, ...]:
    """Hold out each complete recording session within each subject.

    This is deliberately group CV, not a random epoch-level k-fold: epochs from one
    recording can be strongly autocorrelated and may never appear on both sides.
    """

    metadata = _coerce_metadata(
        data_or_subjects, run_ids, session_ids, require_sessions=session_ids is not None
    )
    if metadata.sessions is None:
        if metadata.runs is None:
            raise ValueError("within-subject folds require session_ids or run_ids")
        sessions = np.asarray(
            [f"S{subject:02d}-R{run:02d}" for subject, run in zip(metadata.subjects, metadata.runs)],
            dtype=np.str_,
        )
        metadata = _Metadata(metadata.subjects, metadata.runs, sessions)

    splits: list[ProtocolSplit] = []
    for subject in sorted(np.unique(metadata.subjects).tolist()):
        subject_mask = metadata.subjects == subject
        subject_sessions = sorted(np.unique(metadata.sessions[subject_mask]).tolist())
        if len(subject_sessions) < 2:
            raise ValueError(f"subject {subject} has fewer than two recording sessions")
        for session in subject_sessions:
            test = np.flatnonzero(subject_mask & (metadata.sessions == session))
            train = np.flatnonzero(subject_mask & (metadata.sessions != session))
            split = ProtocolSplit(
                protocol="within_subject",
                fold=f"subject-{subject:02d}_test-{session}",
                train_indices=train,
                test_indices=test,
                held_out_subject=int(subject),
                held_out_sessions=(session,),
            )
            validate_split(split, metadata, require_session_disjoint=True)
            splits.append(split)
    return tuple(splits)


def chronological_cross_session_splits(
    data_or_subjects: Any,
    run_ids: Sequence[int] | None = None,
    session_ids: Sequence[str] | None = None,
) -> tuple[ProtocolSplit, ...]:
    """Train on each subject's first three runs and test on its last run."""

    metadata = _coerce_metadata(data_or_subjects, run_ids, session_ids, require_runs=True)
    if metadata.sessions is None:
        metadata = _Metadata(
            metadata.subjects,
            metadata.runs,
            np.asarray(
                [
                    f"S{subject:02d}-R{run:02d}"
                    for subject, run in zip(metadata.subjects, metadata.runs)
                ],
                dtype=np.str_,
            ),
        )

    splits: list[ProtocolSplit] = []
    assert metadata.runs is not None
    assert metadata.sessions is not None
    for subject in sorted(np.unique(metadata.subjects).tolist()):
        subject_mask = metadata.subjects == subject
        runs = sorted(np.unique(metadata.runs[subject_mask]).tolist())
        if len(runs) != 4:
            raise ValueError(
                f"subject {subject} must have exactly four valid runs, found {runs}"
            )
        train_runs, test_run = runs[:3], runs[-1]
        train = np.flatnonzero(subject_mask & np.isin(metadata.runs, train_runs))
        test = np.flatnonzero(subject_mask & (metadata.runs == test_run))
        held_sessions = tuple(sorted(np.unique(metadata.sessions[test]).tolist()))
        split = ProtocolSplit(
            protocol="chronological_cross_session",
            fold=f"subject-{subject:02d}_train-{train_runs[0]}-{train_runs[-1]}_test-{test_run}",
            train_indices=train,
            test_indices=test,
            held_out_subject=int(subject),
            held_out_sessions=held_sessions,
        )
        validate_split(split, metadata, require_session_disjoint=True)
        splits.append(split)
    return tuple(splits)


def loso_splits(
    data_or_subjects: Any,
    run_ids: Sequence[int] | None = None,
    session_ids: Sequence[str] | None = None,
) -> tuple[ProtocolSplit, ...]:
    """Leave every subject out once, with no target-subject training epochs."""

    metadata = _coerce_metadata(data_or_subjects, run_ids, session_ids)
    subjects = sorted(np.unique(metadata.subjects).tolist())
    if len(subjects) < 2:
        raise ValueError("LOSO requires at least two subjects")
    splits: list[ProtocolSplit] = []
    for subject in subjects:
        test = np.flatnonzero(metadata.subjects == subject)
        train = np.flatnonzero(metadata.subjects != subject)
        if metadata.sessions is None:
            held_sessions: tuple[str, ...] = ()
        else:
            held_sessions = tuple(sorted(np.unique(metadata.sessions[test]).tolist()))
        split = ProtocolSplit(
            protocol="loso",
            fold=f"test-subject-{subject:02d}",
            train_indices=train,
            test_indices=test,
            held_out_subject=int(subject),
            held_out_sessions=held_sessions,
        )
        validate_split(split, metadata, require_subject_disjoint=True)
        splits.append(split)
    return tuple(splits)


# Clear aliases for callers that prefer fold/iterator terminology.
within_subject_folds = within_subject_splits
cross_session_splits = chronological_cross_session_splits
loso_folds = loso_splits


__all__ = [
    "LeakageError",
    "ProtocolSplit",
    "chronological_cross_session_splits",
    "cross_session_splits",
    "loso_folds",
    "loso_splits",
    "validate_split",
    "within_subject_folds",
    "within_subject_splits",
]
