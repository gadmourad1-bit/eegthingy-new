from types import SimpleNamespace

import numpy as np
import pytest

from benchmark.research.config import VALID_SUBJECTS
from benchmark.research.loso import nested_loso_fold, session_calibration_partitions


def _nested_metadata() -> SimpleNamespace:
    subjects: list[int] = []
    sessions: list[str] = []
    for subject in VALID_SUBJECTS:
        for run in range(4):
            # Two rows are enough for split metadata; the fold checks four unique
            # complete session IDs, not any target labels.
            subjects.extend((subject, subject))
            sessions.extend((f"S{subject:02d}-R{run + 1:02d}",) * 2)
    return SimpleNamespace(
        subject_ids=np.asarray(subjects),
        session_ids=np.asarray(sessions),
    )


def test_nested_loso_is_subject_disjoint_and_uses_next_fixed_subject() -> None:
    data = _nested_metadata()
    fold = nested_loso_fold(data, 4)
    assert fold.validation_subject == 5
    assert set(fold.selection_subjects) == set(VALID_SUBJECTS) - {4, 5}
    assert set(fold.available_source_subjects) == set(VALID_SUBJECTS) - {4}
    assert set(data.subject_ids[fold.selection_indices]) == set(fold.selection_subjects)
    assert set(data.subject_ids[fold.validation_indices]) == {5}
    assert set(data.subject_ids[fold.outer_indices]) == {4}
    assert not np.intersect1d(fold.selection_indices, fold.validation_indices).size
    assert not np.intersect1d(fold.selection_indices, fold.outer_indices).size
    assert not np.intersect1d(fold.validation_indices, fold.outer_indices).size

    wrapped = nested_loso_fold(data, VALID_SUBJECTS[-1])
    assert wrapped.validation_subject == VALID_SUBJECTS[0]


def test_nested_loso_rejects_an_incomplete_outer_cohort() -> None:
    data = _nested_metadata()
    keep = data.session_ids != "S04-R04"
    incomplete = SimpleNamespace(
        subject_ids=data.subject_ids[keep],
        session_ids=data.session_ids[keep],
    )
    with pytest.raises(ValueError, match="exactly four sessions"):
        nested_loso_fold(incomplete, 4)


def test_session_calibration_is_independent_and_excludes_prefix_tasks() -> None:
    # Rows from the two sessions are deliberately interleaved in the combined
    # object.  Within each recording, labels alternate task/rest and event times
    # are intentionally out of array order.
    session_ids = np.asarray(["a", "b"] * 10)
    event_samples = np.asarray(
        [50, 1050, 10, 1010, 60, 1060, 20, 1020, 70, 1070,
         30, 1030, 80, 1080, 40, 1040, 90, 1090, 100, 1100]
    )
    # In chronological order each session has six task cues and four rests.
    labels_by_session = {
        "a": {10: 0, 20: -1, 30: 1, 40: -1, 50: 0,
              60: -1, 70: 1, 80: -1, 90: 0, 100: 1},
        "b": {1010: 1, 1020: -1, 1030: 0, 1040: -1, 1050: 1,
              1060: -1, 1070: 0, 1080: -1, 1090: 1, 1100: 0},
    }
    labels = np.asarray(
        [labels_by_session[str(session)][int(sample)]
         for session, sample in zip(session_ids, event_samples, strict=True)]
    )
    data = SimpleNamespace(
        labels=labels,
        session_ids=session_ids,
        event_samples=event_samples,
    )
    partitions = session_calibration_partitions(data, np.arange(len(labels)), 2)
    assert [partition.session_id for partition in partitions] == ["a", "b"]
    for partition in partitions:
        calibration = partition.calibration_indices
        stream = partition.stream_indices
        assert np.all(session_ids[calibration] == partition.session_id)
        assert np.all(session_ids[stream] == partition.session_id)
        assert np.sum(labels[calibration] >= 0) == 2
        assert set(calibration).isdisjoint(stream)
        assert event_samples[calibration].max() < event_samples[stream].min()
        assert set(labels[stream][labels[stream] >= 0]) == {0, 1}
