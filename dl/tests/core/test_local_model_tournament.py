from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from benchmark import runner as benchmark
from benchmark.local_model_tournament import (
    ARCHITECTURES,
    _sign_flip_p_value,
    strict_load,
)


def test_architecture_roster_is_unique_and_factory_routable() -> None:
    assert len(ARCHITECTURES) == 43
    assert len(ARCHITECTURES) == len(set(ARCHITECTURES))
    for requested in ARCHITECTURES:
        model, variant = benchmark._model_definition(requested)
        if model in {"scope", "free_scope"}:
            benchmark._scope_config(variant)
        if model in {"cardinal", "free_cardinal"}:
            benchmark._cardinal_config(variant)


def test_strict_load_rejects_duplicate_keys_and_nonfinite_values(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a": 1, "a": 2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON"):
        strict_load(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"a": NaN}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite JSON"):
        strict_load(nonfinite)


def test_exact_sign_flip_test_uses_all_subject_assignments() -> None:
    assert _sign_flip_p_value(np.zeros(3)) == 1.0
    assert _sign_flip_p_value(np.ones(3)) == 0.25
