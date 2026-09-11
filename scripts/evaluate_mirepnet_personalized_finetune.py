"""Fully fine-tune MIRepNet on early runs, then test later runs per patient."""

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
    ap.add_argument("--subjects", type=int, nargs="+", required=True)
    ap.add_argument("--calibration-runs", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--window-mode", choices=("plan-task", "task-repeat"),
                    default="plan-task")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--output", type=Path, default=None)
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    started = time.perf_counter()
    files = sorted(args.data_dir.glob("*_mi_raw.fif"))
    output = args.output or (
        ROOT / "results" / "mirepnet" /
        f"personalized-full-finetune__{'-'.join(map(str, args.subjects))}.json"
    )
    rows = []

    for index, subject in enumerate(args.subjects, start=1):
        subject_started = time.perf_counter()
        subject_files = sorted(
            [path for path in files if subject_id(path) == subject], key=run_number
        )
        if len(subject_files) <= args.calibration_runs:
            raise SystemExit(f"subject {subject} lacks held-out recordings")
        parts = [mirepnet.process_data(str(path), TARGET_MAPPINGS, args.window_mode)
                 for path in subject_files]
        calibration = parts[:args.calibration_runs]
        testing = parts[args.calibration_runs:]
        X_cal = np.concatenate([part[0] for part in calibration])
        y_cal = np.concatenate([part[1] for part in calibration])
        X_test = np.concatenate([part[0] for part in testing])
        y_test = np.concatenate([part[1] for part in testing])

        decoder = mirepnet.MIRepNetDecoder(
            epochs=args.epochs, batch_size=args.batch_size,
            seed=args.seed, device=args.device,
        ).fit(X_cal, y_cal)
        training_reference_prediction = decoder.predict(X_test)
        decoder.set_reference(X_test)
        target_ea_prediction = decoder.predict(X_test)
        row = {
            "subject": subject,
            "calibration_files": [path.name for path in subject_files[:args.calibration_runs]],
            "test_files": [path.name for path in subject_files[args.calibration_runs:]],
            "calibration_trials": int(len(y_cal)),
            "selected_epoch": decoder.best_epoch_,
            "fit_seconds": decoder.fit_seconds_,
            "training_reference": metrics(y_test, training_reference_prediction),
            "target_unlabeled_ea": metrics(y_test, target_ea_prediction),
            "run_seconds": float(time.perf_counter() - subject_started),
        }
        rows.append(row)
        payload = {
            "format": "mirepnet-personalized-full-finetune-v1",
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "subjects_requested": args.subjects,
            "calibration_runs": args.calibration_runs,
            "epochs": args.epochs,
            "window_mode": args.window_mode,
            "test_labels_used_for_training_or_adaptation": False,
            "results": rows,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"[{index:02d}/{len(args.subjects):02d}] subject {subject:02d}: "
              f"strict={row['training_reference']['accuracy'] * 100:.2f}%  "
              f"EA={row['target_unlabeled_ea']['accuracy'] * 100:.2f}%  "
              f"fit={row['fit_seconds'] / 60:.2f}m")

    accuracies = [row["target_unlabeled_ea"]["accuracy"] for row in rows]
    payload["run_seconds"] = float(time.perf_counter() - started)
    payload["aggregate"] = {
        "n_subjects": len(rows),
        "macro_accuracy": float(np.mean(accuracies)),
        "subjects_at_or_above_80pct": int(sum(value >= 0.8 for value in accuracies)),
        "subjects_at_or_above_90pct": int(sum(value >= 0.9 for value in accuracies)),
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nmacro EA accuracy: {np.mean(accuracies) * 100:.2f}%")
    print(f"result: {output.resolve()}")
    print(f"total runtime: {payload['run_seconds'] / 60:.2f} minutes")


if __name__ == "__main__":
    main()
