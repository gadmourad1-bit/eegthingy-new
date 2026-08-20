#!/usr/bin/env python3
"""Build presentation-only views from the sealed common-grid CSVs.

This script never opens predictions or recomputes a metric. It validates the
sealed 43-model by 5-dataset tables, pivots their existing numeric strings,
and writes deterministic CSV/Markdown views beside the sealed publication.
"""

from __future__ import annotations

import csv
import json
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parent
COMMON = ROOT / "common_grid_v6"
SEALED = COMMON / "analysis"
DATASET_ORDER = (
    "local_exp4",
    "bnci2014_001",
    "bnci2014_004",
    "cho2017",
    "physionet_mi",
)
MODEL_COUNT = 43


def read_csv(name: str) -> list[dict[str, str]]:
    with (SEALED / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def pct(value: str) -> str:
    return f"{Decimal(value) * 100:.3f}%"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def main() -> None:
    dataset_rows = read_csv("dataset_summary.csv")
    overall_rows = read_csv("overall_summary.csv")
    ranking_rows = read_csv("model_ranking.csv")
    plan = json.loads((COMMON / "plan.json").read_text(encoding="utf-8"))

    architecture_order = plan["architectures"]
    if len(architecture_order) != MODEL_COUNT or len(set(architecture_order)) != MODEL_COUNT:
        raise ValueError("plan does not contain exactly 43 unique architectures")
    if tuple(plan["dataset_order"]) != DATASET_ORDER:
        raise ValueError("unexpected common-grid dataset order")
    if len(dataset_rows) != MODEL_COUNT * len(DATASET_ORDER):
        raise ValueError("dataset_summary.csv is not the complete 43 x 5 grid")
    if len(overall_rows) != MODEL_COUNT or len(ranking_rows) != MODEL_COUNT:
        raise ValueError("overall/ranking table does not contain exactly 43 models")

    by_dataset_model = {(r["dataset"], r["model"]): r for r in dataset_rows}
    overall = {r["model"]: r for r in overall_rows}
    rank = {r["model"]: r for r in ranking_rows}
    expected_models = set(architecture_order)
    if set(overall) != expected_models or set(rank) != expected_models:
        raise ValueError("model identities disagree across sealed inputs")
    if set(by_dataset_model) != {
        (dataset, model) for dataset in DATASET_ORDER for model in architecture_order
    }:
        raise ValueError("dataset_summary.csv has a missing or extra model-dataset cell")

    ranking_rows = sorted(ranking_rows, key=lambda row: int(row["rank"]))
    for row in ranking_rows:
        model = row["model"]
        if Decimal(row["value"]) != Decimal(overall[model]["balanced_accuracy"]):
            raise ValueError(f"balanced-accuracy mismatch for {model}")

    ba_fields = [
        "rank",
        "model",
        *(f"{dataset}_balanced_accuracy" for dataset in DATASET_ORDER),
        "equal_dataset_macro_balanced_accuracy",
        "status",
    ]
    ba_rows: list[dict[str, str]] = []
    for ranked in ranking_rows:
        model = ranked["model"]
        row = {"rank": ranked["rank"], "model": model}
        row.update(
            {
                f"{dataset}_balanced_accuracy": by_dataset_model[(dataset, model)][
                    "balanced_accuracy"
                ]
                for dataset in DATASET_ORDER
            }
        )
        row["equal_dataset_macro_balanced_accuracy"] = ranked["value"]
        row["status"] = ranked["status"]
        ba_rows.append(row)
    write_csv(COMMON / "all_models_balanced_accuracy.csv", ba_fields, ba_rows)

    ba_md_headers = [
        "Rank",
        "Model",
        "Local Exp4 BA",
        "BNCI 2014-001 BA",
        "BNCI 2014-004 BA",
        "Cho2017 BA",
        "PhysioNet MI BA",
        "Equal-dataset BA",
    ]
    ba_md_rows = [
        [
            row["rank"],
            f"`{row['model']}`",
            *(pct(row[f"{dataset}_balanced_accuracy"]) for dataset in DATASET_ORDER),
            pct(row["equal_dataset_macro_balanced_accuracy"]),
        ]
        for row in ba_rows
    ]
    (COMMON / "all_models_balanced_accuracy.md").write_text(
        "# All 43 common-grid models: balanced accuracy\n\n"
        "Values are direct presentation transforms of the sealed "
        "`analysis/dataset_summary.csv` and `analysis/model_ranking.csv`; no metric "
        "was recomputed. Ranking is the sealed primary equal-dataset macro balanced-"
        "accuracy ranking. All evidence is opened-development only.\n\n"
        + markdown_table(ba_md_headers, ba_md_rows),
        encoding="utf-8",
    )

    architecture_index = {model: index for index, model in enumerate(architecture_order)}
    accuracy_order = sorted(
        architecture_order,
        key=lambda model: (-Decimal(overall[model]["accuracy"]), architecture_index[model]),
    )
    accuracy_rank = {model: str(index + 1) for index, model in enumerate(accuracy_order)}
    primary_ba_rank = {row["model"]: row["rank"] for row in ranking_rows}
    acc_fields = [
        "descriptive_accuracy_rank",
        "primary_balanced_accuracy_rank",
        "model",
        *(f"{dataset}_accuracy" for dataset in DATASET_ORDER),
        "equal_dataset_macro_accuracy",
        "status",
    ]
    acc_rows: list[dict[str, str]] = []
    for model in accuracy_order:
        row = {
            "descriptive_accuracy_rank": accuracy_rank[model],
            "primary_balanced_accuracy_rank": primary_ba_rank[model],
            "model": model,
        }
        row.update(
            {
                f"{dataset}_accuracy": by_dataset_model[(dataset, model)]["accuracy"]
                for dataset in DATASET_ORDER
            }
        )
        row["equal_dataset_macro_accuracy"] = overall[model]["accuracy"]
        row["status"] = rank[model]["status"]
        acc_rows.append(row)
    write_csv(COMMON / "all_models_accuracy.csv", acc_fields, acc_rows)

    acc_md_headers = [
        "Accuracy rank",
        "Primary BA rank",
        "Model",
        "Local Exp4 accuracy",
        "BNCI 2014-001 accuracy",
        "BNCI 2014-004 accuracy",
        "Cho2017 accuracy",
        "PhysioNet MI accuracy",
        "Equal-dataset accuracy",
    ]
    acc_md_rows = [
        [
            row["descriptive_accuracy_rank"],
            row["primary_balanced_accuracy_rank"],
            f"`{row['model']}`",
            *(pct(row[f"{dataset}_accuracy"]) for dataset in DATASET_ORDER),
            pct(row["equal_dataset_macro_accuracy"]),
        ]
        for row in acc_rows
    ]
    (COMMON / "all_models_accuracy.md").write_text(
        "# All 43 common-grid models: standard accuracy\n\n"
        "Values are direct presentation transforms of the sealed "
        "`analysis/dataset_summary.csv` and `analysis/overall_summary.csv`; no metric "
        "was recomputed. The descriptive accuracy rank is obtained by sorting the sealed "
        "equal-dataset accuracy column, using frozen plan architecture order only for exact "
        "ties. Balanced accuracy remains the prespecified primary outcome.\n\n"
        + markdown_table(acc_md_headers, acc_md_rows),
        encoding="utf-8",
    )

    top_rows = [
        {
            "rank": row["rank"],
            "model": row["model"],
            "equal_dataset_macro_balanced_accuracy": row["value"],
            "status": row["status"],
        }
        for row in ranking_rows[:10]
    ]
    top_fields = [
        "rank",
        "model",
        "equal_dataset_macro_balanced_accuracy",
        "status",
    ]
    write_csv(COMMON / "top_10_balanced_accuracy.csv", top_fields, top_rows)

    winner_rows: list[dict[str, str]] = []
    for dataset in DATASET_ORDER:
        rows = [by_dataset_model[(dataset, model)] for model in architecture_order]
        maximum = max(Decimal(row["balanced_accuracy"]) for row in rows)
        for row in rows:
            if Decimal(row["balanced_accuracy"]) == maximum:
                winner_rows.append(
                    {
                        "dataset": dataset,
                        "model": row["model"],
                        "balanced_accuracy": row["balanced_accuracy"],
                        "subjects_averaged": row["subjects_averaged"],
                        "aggregation": row["aggregation"],
                    }
                )
    winner_fields = [
        "dataset",
        "model",
        "balanced_accuracy",
        "subjects_averaged",
        "aggregation",
    ]
    write_csv(COMMON / "per_dataset_balanced_accuracy_winners.csv", winner_fields, winner_rows)

    summary_lines = [
        "# Common-grid v6 compact summary",
        "",
        "This view cites only sealed v6 analysis values. Balanced accuracy is the primary "
        "outcome; all results are opened-development evidence, not confirmation or a SOTA "
        "claim.",
        "",
        "## Top 10 equal-dataset rankings",
        "",
        markdown_table(
            ["Rank", "Model", "Equal-dataset macro BA"],
            [
                [row["rank"], f"`{row['model']}`", pct(row["value"])]
                for row in ranking_rows[:10]
            ],
        ).rstrip(),
        "",
        "## Per-dataset winners",
        "",
        markdown_table(
            ["Dataset", "Model", "Balanced accuracy", "Participants"],
            [
                [
                    row["dataset"],
                    f"`{row['model']}`",
                    pct(row["balanced_accuracy"]),
                    row["subjects_averaged"],
                ]
                for row in winner_rows
            ],
        ).rstrip(),
        "",
        "Complete tables: `all_models_balanced_accuracy.csv`/`.md` and "
        "`all_models_accuracy.csv`/`.md`.",
        "",
    ]
    (COMMON / "SUMMARY.md").write_text("\n".join(summary_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
