"""Patient-calibrated control experiment for the project's EA + FB-CSP decoder."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "classifier"), str(ROOT)]

from config import TARGET_MAPPINGS  # noqa: E402
import run as classical  # noqa: E402


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
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--calibration-runs", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--output", type=Path,
                    default=ROOT / "results" / "classical-personalized.json")
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    started = time.perf_counter()
    files = sorted(args.data_dir.glob("*_mi_raw.fif"))
    subjects = sorted(set(map(subject_id, files)))
    results = {str(count): [] for count in args.calibration_runs}

    for subject_index, subject in enumerate(subjects, start=1):
        subject_files = sorted(
            [path for path in files if subject_id(path) == subject], key=run_number
        )
        parts = [classical.process_data(str(path), TARGET_MAPPINGS) for path in subject_files]
        for count in args.calibration_runs:
            if len(parts) <= count:
                continue
            X_cal = np.concatenate([part[0] for part in parts[:count]])
            y_cal = np.concatenate([part[1] for part in parts[:count]])
            X_test = np.concatenate([part[0] for part in parts[count:]])
            y_test = np.concatenate([part[1] for part in parts[count:]])
            groups = np.concatenate([
                np.full(len(part[1]), index, dtype=np.int64)
                for index, part in enumerate(parts[:count])
            ])
            decoder = classical.EAFilterBankCSP(n_components=2).fit(
                X_cal, y_cal, groups=groups
            )
            decoder.set_reference(X_test)
            raw_prediction = decoder.predict(X_test)
            scores = decoder.analyze(X_test)[1]
            balanced = balanced_prediction(scores, decoder.classes_)
            results[str(count)].append({
                "subject": subject,
                "calibration_trials": int(len(y_cal)),
                "calibration_files": [path.name for path in subject_files[:count]],
                "test_files": [path.name for path in subject_files[count:]],
                "raw_threshold": metrics(y_test, raw_prediction),
                "balanced_protocol_prior": metrics(y_test, balanced),
            })
        scores = "  ".join(
            f"{count}run={results[str(count)][-1]['balanced_protocol_prior']['accuracy'] * 100:.2f}%"
            for count in args.calibration_runs if results[str(count)]
            and results[str(count)][-1]["subject"] == subject
        )
        print(f"[{subject_index:02d}/{len(subjects):02d}] subject {subject:02d}: {scores}")

    aggregate = {}
    for count, rows in results.items():
        aggregate[count] = {}
        for protocol in ("raw_threshold", "balanced_protocol_prior"):
            aggregate[count][protocol] = {
                "n_subjects": len(rows),
                "macro_accuracy": float(np.mean([row[protocol]["accuracy"] for row in rows])),
                "macro_balanced_accuracy": float(np.mean([
                    row[protocol]["balanced_accuracy"] for row in rows
                ])),
                "subjects_at_or_above_80pct": int(sum(
                    row[protocol]["accuracy"] >= 0.8 for row in rows
                )),
                "subjects_at_or_above_90pct": int(sum(
                    row[protocol]["accuracy"] >= 0.9 for row in rows
                )),
            }
    payload = {
        "format": "classical-ea-fbcsp-personalized-v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_seconds": float(time.perf_counter() - started),
        "test_labels_used_for_training_or_adaptation": False,
        "target_test_signals_used_for_unlabeled_alignment": True,
        "results": results,
        "aggregate": aggregate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("\nAggregate balanced-protocol results")
    for count in map(str, args.calibration_runs):
        row = aggregate[count]["balanced_protocol_prior"]
        print(f"  {count} run(s): macro={row['macro_accuracy'] * 100:.2f}%  "
              f">=80% {row['subjects_at_or_above_80pct']}/{row['n_subjects']}  "
              f">=90% {row['subjects_at_or_above_90pct']}/{row['n_subjects']}")
    print(f"result: {args.output.resolve()}")
    print(f"total runtime: {payload['run_seconds'] / 60:.2f} minutes")


if __name__ == "__main__":
    main()
