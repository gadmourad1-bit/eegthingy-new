"""Participant-level summaries for benchmark JSON artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .experiment import _atomic_json, _summary
from .metrics import classification_metrics


METRICS = (
    "balanced_accuracy",
    "roc_auc",
    "brier",
    "ece",
    "coverage",
    "selective_accuracy",
    "rest_false_commit_rate",
)
AUDIT_METRICS = ("accuracy", "balanced_accuracy", "kappa", *METRICS[1:])


def metrics_from_predictions(
    predictions: Sequence[dict[str, Any]], *, commit_confidence: float
) -> dict[str, float | None]:
    """Recompute every reported metric from the retained per-window trace."""

    task = [item for item in predictions if int(item["label"]) >= 0]
    if not task:
        raise ValueError("prediction trace contains no scored task windows")
    labels = np.asarray([int(item["label"]) for item in task], dtype=np.int64)
    probabilities = np.asarray(
        [
            [float(item["probability_left"]), float(item["probability_right"])]
            for item in task
        ],
        dtype=np.float64,
    )
    committed = np.asarray([bool(item["committed"]) for item in task], dtype=bool)
    metrics = classification_metrics(
        labels, probabilities, commit_confidence=commit_confidence
    ).to_dict()
    metrics["coverage"] = float(committed.mean())
    metrics["selective_accuracy"] = (
        float(np.mean(probabilities[committed].argmax(axis=1) == labels[committed]))
        if np.any(committed)
        else None
    )
    rest = [item for item in predictions if int(item["label"]) < 0]
    metrics["rest_false_commit_rate"] = (
        float(np.mean([bool(item["committed"]) for item in rest])) if rest else None
    )
    return metrics


def validate_prediction_metrics(payload: dict[str, Any]) -> None:
    """Hard-fail if a schema-v2 aggregate cannot be reproduced from its trace."""

    if int(payload.get("schema_version", 0)) < 2:
        return
    config = dict(payload.get("benchmark_config", payload.get("loso_config", {})))
    threshold = float(config.get("commit_confidence", 0.85))
    for row in payload.get("folds", []):
        predictions = row.get("predictions")
        if not isinstance(predictions, list) or not predictions:
            raise ValueError("schema-v2 result row is missing its prediction trace")
        recomputed = metrics_from_predictions(
            predictions, commit_confidence=threshold
        )
        for metric in AUDIT_METRICS:
            expected = row["metrics"].get(metric)
            observed = recomputed.get(metric)
            if expected is None or observed is None:
                if expected is not None or observed is not None:
                    raise ValueError(f"prediction trace disagrees for {metric}")
            elif not np.isclose(float(expected), float(observed), rtol=1e-9, atol=1e-12):
                raise ValueError(f"prediction trace disagrees for {metric}")


def participant_values(
    rows: Sequence[dict[str, Any]], model: str, metric: str
) -> dict[int, float]:
    """Average repeated neural seeds before treating people as observations."""

    selected = [row for row in rows if row["model"] == model]
    result: dict[int, float] = {}
    for subject in sorted({int(row["subject"]) for row in selected}):
        values = [
            row["metrics"].get(metric)
            for row in selected
            if int(row["subject"]) == subject and row["metrics"].get(metric) is not None
        ]
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if len(finite):
            result[subject] = float(finite.mean())
    return result


def bootstrap_mean_ci(
    values: Sequence[float], *, repetitions: int = 20_000, seed: int = 20260717
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0:
        return float("nan"), float("nan")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(array), size=(repetitions, len(array)))
    means = array[indices].mean(axis=1)
    lower, upper = np.quantile(means, (0.025, 0.975))
    return float(lower), float(upper)


def build_report(payload: dict[str, Any]) -> tuple[dict[str, Any], str]:
    validate_prediction_metrics(payload)
    rows = payload["folds"]
    summary = _summary(rows)
    models = sorted({str(row["model"]) for row in rows})
    intervals: dict[str, Any] = {}
    for model in models:
        intervals[model] = {}
        for metric in METRICS:
            values = participant_values(rows, model, metric)
            lower, upper = bootstrap_mean_ci(list(values.values()))
            intervals[model][metric] = {
                "participant_values": {str(key): value for key, value in values.items()},
                "mean_95ci": [lower, upper],
            }

    paired: dict[str, Any] = {}
    if "geoadapt" in models:
        for baseline in ("riemann", "fbcsp"):
            if baseline not in models:
                continue
            paired[baseline] = {}
            for metric in ("balanced_accuracy", "rest_false_commit_rate"):
                proposed = participant_values(rows, "geoadapt", metric)
                reference = participant_values(rows, baseline, metric)
                common = sorted(set(proposed) & set(reference))
                differences = [proposed[subject] - reference[subject] for subject in common]
                lower, upper = bootstrap_mean_ci(differences)
                paired[baseline][metric] = {
                    "mean_difference": float(np.mean(differences)),
                    "participant_bootstrap_95ci": [lower, upper],
                    "n_participants": len(common),
                }

    enriched = dict(payload)
    enriched["summary"] = summary
    enriched["participant_bootstrap"] = intervals
    enriched["paired_differences"] = paired

    protocol = str(payload.get("protocol", ""))
    title = (
        "Initial nested leave-one-subject-out benchmark"
        if protocol.startswith("nested_loso")
        else "Chronological cross-session benchmark"
    )
    lines = [
        f"# {title}",
        "",
        "Participant-level means; neural seeds are averaged within participant before aggregation.",
        "",
        "| Model | Participants | Balanced accuracy | ROC AUC | Brier | Coverage | Selective accuracy | Rest false commits |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in models:
        item = summary[model]

        def cell(metric: str) -> str:
            mean = item[metric]["mean"]
            std = item[metric]["std"]
            if mean is None:
                return "NA"
            return f"{100.0 * mean:.2f} +/- {100.0 * std:.2f}%"

        lines.append(
            f"| {model} | {item['participants']} | {cell('balanced_accuracy')} | "
            f"{cell('roc_auc')} | {cell('brier')} | {cell('coverage')} | "
            f"{cell('selective_accuracy')} | {cell('rest_false_commit_rate')} |"
        )
    lines.extend(["", "## Paired participant-level differences", ""])
    for baseline, values in paired.items():
        for metric, estimate in values.items():
            lower, upper = estimate["participant_bootstrap_95ci"]
            lines.append(
                f"- geoadapt - {baseline}, {metric}: "
                f"{100.0 * estimate['mean_difference']:+.2f} percentage points "
                f"(participant bootstrap 95% CI {100.0 * lower:+.2f} to {100.0 * upper:+.2f})."
            )
    lines.extend(
        [
            "",
            (
                "These intervals describe this "
                f"{len({int(row['subject']) for row in rows})}-participant "
                "healthy-volunteer sample; they are not clinical validation."
            ),
            "",
        ]
    )
    return enriched, "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--update-json", action="store_true")
    args = parser.parse_args(argv)
    payload = json.loads(args.result.read_text(encoding="utf-8"))
    enriched, markdown = build_report(payload)
    if args.update_json:
        _atomic_json(args.result, enriched)
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
