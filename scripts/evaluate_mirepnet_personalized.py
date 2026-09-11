"""Evaluate fast patient calibration with a frozen MIRepNet encoder.

For each participant, the earliest one or two recordings train only a logistic
feature head. All later recordings are held out for testing. The transformer is
never updated, and test labels are used only for final metrics.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "classifier"), str(ROOT)]

from config import (DEEP_DEVICE, LOCAL_EXP4_VALID_RUNS, MIREPNET_WINDOW_MODE,
                    TARGET_MAPPINGS)  # noqa: E402
import mirepnet  # noqa: E402


def subject_id(path):
    match = re.search(r"subject(\d+)", Path(path).name)
    if not match:
        raise ValueError(f"cannot parse participant from {path}")
    return int(match.group(1))


def run_number(path):
    match = re.search(r"training_(\d+)_mi_raw", Path(path).name)
    if not match:
        raise ValueError(f"cannot parse recording number from {path}")
    return int(match.group(1))


def domain_z(features):
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    return (features - mean) / np.maximum(scale, 1e-5)


def balanced_prediction(scores, classes):
    prediction = np.full(len(scores), classes[0])
    prediction[np.argsort(scores)[-(len(scores) // 2):]] = classes[1]
    return prediction


def metrics(y_true, y_pred):
    labels = sorted(np.unique(y_true).tolist())
    return {
        "n_trials": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
    }


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--calibration-runs", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--validated-local-cohort", action="store_true",
                    help="use the repository's frozen eight-subject Local Exp4 manifest")
    ap.add_argument("--window-mode", choices=("plan-task", "task-repeat"),
                    default=MIREPNET_WINDOW_MODE)
    ap.add_argument("--device", default=DEEP_DEVICE)
    ap.add_argument("--output", type=Path, default=None)
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    started = time.perf_counter()
    files = sorted(args.data_dir.glob("*_mi_raw.fif"))
    if args.validated_local_cohort:
        files = [path for path in files if subject_id(path) in LOCAL_EXP4_VALID_RUNS
                 and run_number(path) in LOCAL_EXP4_VALID_RUNS[subject_id(path)]]
    subjects = sorted(set(map(subject_id, files)))
    model = mirepnet.MIRepNetDecoder.load(str(args.checkpoint), device=args.device)
    results = {str(count): [] for count in args.calibration_runs}

    print(f"Personalized frozen MIRepNet evaluation: {args.checkpoint.name}")
    print(f"calibration recording counts: {args.calibration_runs}")
    print("transformer weights updated: NO\n")

    for subject_index, subject in enumerate(subjects, start=1):
        subject_files = sorted(
            [path for path in files if subject_id(path) == subject], key=run_number
        )
        parts = [mirepnet.process_data(str(path), TARGET_MAPPINGS, args.window_mode)
                 for path in subject_files]
        for count in args.calibration_runs:
            if len(parts) <= count:
                continue
            X_cal = np.concatenate([part[0] for part in parts[:count]])
            y_cal = np.concatenate([part[1] for part in parts[:count]])
            X_test = np.concatenate([part[0] for part in parts[count:]])
            y_test = np.concatenate([part[1] for part in parts[count:]])

            model.set_reference(X_cal)
            features_cal = domain_z(model.extract_features(X_cal))
            # Test-domain EA/z-scaling use signals only, never test labels.
            model.set_reference(X_test)
            features_test = domain_z(model.extract_features(X_test))

            head = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=0.01, max_iter=2000, class_weight="balanced"),
            ).fit(features_cal, y_cal)
            raw_prediction = head.predict(features_test)
            balanced = balanced_prediction(head.decision_function(features_test), head.classes_)
            row = {
                "subject": subject,
                "calibration_files": [path.name for path in subject_files[:count]],
                "test_files": [path.name for path in subject_files[count:]],
                "calibration_trials": int(len(y_cal)),
                "raw_threshold": metrics(y_test, raw_prediction),
                "balanced_protocol_prior": metrics(y_test, balanced),
            }
            results[str(count)].append(row)

        scores = "  ".join(
            f"{count}run={results[str(count)][-1]['balanced_protocol_prior']['accuracy'] * 100:.2f}%"
            for count in args.calibration_runs if results[str(count)]
            and results[str(count)][-1]["subject"] == subject
        )
        print(f"[{subject_index:02d}/{len(subjects):02d}] subject {subject:02d}: {scores}")

    aggregates = {}
    for count, rows in results.items():
        aggregates[count] = {}
        for metric_name in ("raw_threshold", "balanced_protocol_prior"):
            aggregate = {
                "n_subjects": len(rows),
                "n_test_trials": int(sum(row[metric_name]["n_trials"] for row in rows)),
                "macro_accuracy": float(np.mean([row[metric_name]["accuracy"] for row in rows])),
                "macro_balanced_accuracy": float(
                    np.mean([row[metric_name]["balanced_accuracy"] for row in rows])
                ),
                "subjects_at_or_above_80pct": int(sum(
                    row[metric_name]["accuracy"] >= 0.8 for row in rows
                )),
                "subjects_at_or_above_90pct": int(sum(
                    row[metric_name]["accuracy"] >= 0.9 for row in rows
                )),
            }
            aggregates[count][metric_name] = aggregate

    payload = {
        "format": "mirepnet-personalized-frozen-head-v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_seconds": float(time.perf_counter() - started),
        "checkpoint": str(args.checkpoint.resolve()),
        "cohort": "validated_local_exp4" if args.validated_local_cohort else "filesystem_discovery",
        "validated_manifest": LOCAL_EXP4_VALID_RUNS if args.validated_local_cohort else None,
        "window_mode": args.window_mode,
        "transformer_weights_updated": False,
        "test_labels_used_for_training_or_adaptation": False,
        "target_test_signals_used_for_unlabeled_alignment": True,
        "feature_head": "subject-domain-z + balanced logistic regression C=0.01",
        "results": results,
        "aggregate": aggregates,
    }
    output = args.output or (
        ROOT / "results" / "mirepnet" / f"personalized__{args.checkpoint.stem}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("\nAggregate balanced-protocol results")
    for count in map(str, args.calibration_runs):
        row = aggregates[count]["balanced_protocol_prior"]
        print(f"  {count} calibration run(s): macro={row['macro_accuracy'] * 100:.2f}%  "
              f">=80% {row['subjects_at_or_above_80pct']}/{row['n_subjects']}  "
              f">=90% {row['subjects_at_or_above_90pct']}/{row['n_subjects']}")
    print(f"result: {output.resolve()}")
    print(f"total runtime: {payload['run_seconds'] / 60:.2f} minutes")


if __name__ == "__main__":
    main()
