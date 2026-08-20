from __future__ import annotations

import numpy as np

from benchmark.verification import verify_native_fbms_full_grid as verify


def _heterogeneous_scores() -> dict[str, dict[str, dict[int, float]]]:
    result: dict[str, dict[str, dict[int, float]]] = {}
    for dataset_index, dataset in enumerate(verify.DATASETS):
        subjects = verify.DATASET_SPECS[dataset]["subjects"]
        result[dataset] = {condition: {} for condition in verify.CONDITIONS}
        for index, subject in enumerate(subjects):
            control = 0.45 + 0.002 * ((index + dataset_index) % 11)
            result[dataset][verify.PRIMARY][subject] = control
            result[dataset][verify.CANDIDATE][subject] = (
                control + 0.005 + 0.001 * ((index % 5) - 2)
            )
            for condition in verify.CONDITIONS:
                result[dataset][condition].setdefault(subject, control - 0.01)
    return result


def test_balanced_accuracy_weights_classes_equally() -> None:
    labels = np.asarray([0, 0, 0, 0, 1, 1], dtype=np.int64)
    predictions = np.asarray([0, 0, 0, 1, 1, 0], dtype=np.int64)
    assert verify._balanced_accuracy(labels, predictions) == 0.625


def test_independent_bootstrap_is_deterministic_with_heterogeneous_pairs(
    monkeypatch,
) -> None:
    monkeypatch.setattr(verify, "BOOTSTRAP_REPETITIONS", 1_003)
    monkeypatch.setattr(verify, "BOOTSTRAP_BATCH_SIZE", 127)
    scores = _heterogeneous_scores()

    first_differences, first_intervals = verify._bootstrap(
        subject_scores=scores,
        comparator=verify.PRIMARY,
        seed=20_260_721,
    )
    second_differences, second_intervals = verify._bootstrap(
        subject_scores=scores,
        comparator=verify.PRIMARY,
        seed=20_260_721,
    )

    for dataset in verify.DATASETS:
        assert np.array_equal(first_differences[dataset], second_differences[dataset])
        assert np.unique(first_differences[dataset]).size > 1
    assert first_intervals == second_intervals
    assert (
        first_intervals["equal_dataset_macro"]["one_sided_95_lower"] > 0.0
    )
