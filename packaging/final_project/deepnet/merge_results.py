"""Merge compatible schema-v2 benchmark artifacts without losing audit traces."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .experiment import _atomic_json, _summary


def _commit_threshold(payload: dict[str, Any]) -> float:
    config = dict(
        payload.get(
            "benchmark_config",
            payload.get("loso_config", payload.get("baseline_config", {})),
        )
    )
    return float(config.get("commit_confidence", 0.85))


def merge_payloads(payloads: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Combine rows only when protocol, data, cohort, and commit policy match."""

    if len(payloads) < 2:
        raise ValueError("at least two result payloads are required")
    first = payloads[0]
    if int(first.get("schema_version", 0)) < 2:
        raise ValueError("only schema-v2 result artifacts can be merged")
    protocol = first.get("protocol")
    contract = first.get("data_contract")
    outer_subjects = first.get("outer_subjects", first.get("subjects"))
    threshold = _commit_threshold(first)
    for payload in payloads[1:]:
        if int(payload.get("schema_version", 0)) < 2:
            raise ValueError("only schema-v2 result artifacts can be merged")
        if payload.get("protocol") != protocol:
            raise ValueError("cannot merge different benchmark protocols")
        if payload.get("data_contract") != contract:
            raise ValueError("cannot merge different preprocessing/data contracts")
        if payload.get("outer_subjects", payload.get("subjects")) != outer_subjects:
            raise ValueError("cannot merge different participant lists")
        if _commit_threshold(payload) != threshold:
            raise ValueError("cannot merge different commit thresholds")

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    for payload in payloads:
        for row in payload.get("folds", []):
            key = (str(row["model"]), int(row["subject"]), int(row.get("seed", 0)))
            if key in seen:
                raise ValueError(f"duplicate result row {key}")
            seen.add(key)
            rows.append(row)

    limitations = list(
        dict.fromkeys(
            str(item)
            for payload in payloads
            for item in payload.get("limitations", [])
        )
    )
    merged = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol,
        "artifact_kind": "combined_compatible_benchmark_results",
        "model": "combined",
        "models": sorted({str(row["model"]) for row in rows}),
        "loso_config": dict(first.get("loso_config", {})),
        "data_contract": contract,
        "subject_order": first.get("subject_order"),
        "outer_subjects": outer_subjects,
        "environment": first.get("environment"),
        "repository": first.get("repository"),
        "limitations": limitations,
        "components": [
            {
                "artifact_kind": payload.get("artifact_kind"),
                "model": payload.get("model"),
                "models": payload.get("models"),
                "created_at": payload.get("created_at"),
            }
            for payload in payloads
        ],
        "folds": rows,
        "summary": _summary(rows),
    }
    return merged


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    _atomic_json(args.output, merge_payloads(payloads))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main", "merge_payloads"]
