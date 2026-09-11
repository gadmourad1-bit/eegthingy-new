"""Diagnostic within-patient stratified CV upper bound for EA + FB-CSP.

This intentionally mixes recordings within each fold and is therefore not a
deployment/generalization estimate. It tests whether labeled class information
is recoverable at all under the current preprocessing contract.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import mne
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "classifier"), str(ROOT)]

from config import TARGET_MAPPINGS  # noqa: E402
import run as classical  # noqa: E402

mne.set_log_level("ERROR")


def subject_id(path):
    match = re.search(r"subject(\d+)", Path(path).name)
    return int(match.group(1)) if match else None


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--output", type=Path,
                    default=ROOT / "results" / "classical-within-patient-upper-bound.json")
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    files = sorted(args.data_dir.glob("*_mi_raw.fif"))
    subjects = sorted({subject_id(path) for path in files})
    rows = []
    for index, subject in enumerate(subjects, start=1):
        parts = [classical.process_data(str(path), TARGET_MAPPINGS)
                 for path in files if subject_id(path) == subject]
        X = np.concatenate([part[0] for part in parts])
        y = np.concatenate([part[1] for part in parts])
        fold_rows = []
        splitter = StratifiedKFold(args.folds, shuffle=True, random_state=7)
        for train, test in splitter.split(X, y):
            decoder = classical.EAFilterBankCSP(n_components=2).fit(X[train], y[train])
            decoder.set_reference(X[test])
            prediction = decoder.predict(X[test])
            fold_rows.append({
                "accuracy": float(accuracy_score(y[test], prediction)),
                "balanced_accuracy": float(balanced_accuracy_score(y[test], prediction)),
            })
        row = {
            "subject": subject,
            "n_trials": int(len(y)),
            "accuracy": float(np.mean([fold["accuracy"] for fold in fold_rows])),
            "balanced_accuracy": float(np.mean([
                fold["balanced_accuracy"] for fold in fold_rows
            ])),
            "folds": fold_rows,
        }
        rows.append(row)
        print(f"[{index:02d}/{len(subjects):02d}] subject {subject:02d}: "
              f"{row['accuracy'] * 100:.2f}%")

    payload = {
        "format": "classical-within-patient-upper-bound-v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "warning": "random trial folds mix recordings; diagnostic upper bound only",
        "subjects": rows,
        "aggregate": {
            "macro_accuracy": float(np.mean([row["accuracy"] for row in rows])),
            "subjects_at_or_above_80pct": int(sum(row["accuracy"] >= 0.8 for row in rows)),
            "subjects_at_or_above_90pct": int(sum(row["accuracy"] >= 0.9 for row in rows)),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"macro={payload['aggregate']['macro_accuracy'] * 100:.2f}%  "
          f">=80% {payload['aggregate']['subjects_at_or_above_80pct']}/{len(rows)}  "
          f">=90% {payload['aggregate']['subjects_at_or_above_90pct']}/{len(rows)}")
    print(f"result: {args.output.resolve()}")


if __name__ == "__main__":
    main()
