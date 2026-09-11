"""Evaluate one existing MIRepNet checkpoint on every participant, serially.

This script never trains or changes checkpoint weights.  It reports the saved
training-domain EA reference (strict) and participant-unlabeled EA separately.
Participants recorded in the checkpoint metadata are marked as training-seen.
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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "classifier"), str(ROOT)]

from config import DEEP_DEVICE, TARGET_MAPPINGS  # noqa: E402
import mirepnet  # noqa: E402


def subject_id(path):
    match = re.search(r"subject(\d+)", Path(path).name)
    if not match:
        raise ValueError(f"cannot parse participant from {path}")
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


def summary(rows, key):
    if not rows:
        return None
    return {
        "n_subjects": len(rows),
        "n_trials": int(sum(row[key]["n_trials"] for row in rows)),
        "macro_accuracy": float(np.mean([row[key]["accuracy"] for row in rows])),
        "macro_balanced_accuracy": float(
            np.mean([row[key]["balanced_accuracy"] for row in rows])
        ),
        "micro_accuracy": float(
            sum(row[key]["accuracy"] * row[key]["n_trials"] for row in rows)
            / sum(row[key]["n_trials"] for row in rows)
        ),
    }


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--subjects", type=int, nargs="+", default=None,
                    help="optional subset; default evaluates every available participant")
    ap.add_argument("--device", default=DEEP_DEVICE)
    ap.add_argument("--output", type=Path, default=None)
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    started = time.perf_counter()
    checkpoint = args.checkpoint.resolve()
    files = sorted(args.data_dir.glob("*_mi_raw.fif"))
    available = sorted(set(map(subject_id, files)))
    subjects = available if args.subjects is None else list(dict.fromkeys(args.subjects))
    missing = sorted(set(subjects) - set(available))
    if missing:
        raise SystemExit(f"no recordings found for participant(s) {missing}")

    model = mirepnet.MIRepNetDecoder.load(str(checkpoint), device=args.device)
    training_subjects = sorted({
        subject_id(name) for name in model.meta_.get("trained_on", [])
        if re.search(r"subject\d+", Path(name).name)
    })
    training_reference = np.asarray(model.ref_white_).copy()
    rows = []

    print(f"MIRepNet checkpoint-only evaluation: {checkpoint.name}")
    print(f"participants: {subjects}")
    print(f"checkpoint training participants: {training_subjects}")
    print("weights updated: NO\n")

    for index, subject in enumerate(subjects, start=1):
        subject_started = time.perf_counter()
        subject_files = [path for path in files if subject_id(path) == subject]
        parts = [mirepnet.process_data(str(path), TARGET_MAPPINGS) for path in subject_files]
        X = np.concatenate([part[0] for part in parts])
        y = np.concatenate([part[1] for part in parts])

        model.ref_white_ = training_reference.copy()
        strict_pred = model.predict(X)
        strict = metrics(y, strict_pred)

        model.set_reference(X)
        deployment_pred = model.predict(X)
        deployment = metrics(y, deployment_pred)

        per_file = []
        offset = 0
        for path, (_, labels) in zip(subject_files, parts):
            count = len(labels)
            per_file.append({
                "file": path.name,
                **metrics(labels, deployment_pred[offset:offset + count]),
            })
            offset += count

        row = {
            "subject": subject,
            "training_seen": subject in training_subjects,
            "files": [path.name for path in subject_files],
            "strict_zero_shot": strict,
            "target_unlabeled_ea": deployment,
            "per_file_target_unlabeled_ea": per_file,
            "run_seconds": float(time.perf_counter() - subject_started),
        }
        rows.append(row)
        status = "SEEN" if row["training_seen"] else "UNSEEN"
        print(f"[{index:02d}/{len(subjects):02d}] subject {subject:02d} {status}: "
              f"strict={strict['accuracy'] * 100:6.2f}%  "
              f"EA={deployment['accuracy'] * 100:6.2f}%  "
              f"n={len(y)}  {row['run_seconds']:.1f}s")

    unseen = [row for row in rows if not row["training_seen"]]
    seen = [row for row in rows if row["training_seen"]]
    payload = {
        "format": "mirepnet-all-participants-evaluation-v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_seconds": float(time.perf_counter() - started),
        "checkpoint": str(checkpoint),
        "weights_updated": False,
        "checkpoint_training_subjects": training_subjects,
        "target_labels_used_for_alignment": False,
        "subjects": rows,
        "aggregate": {
            "all": {
                "strict_zero_shot": summary(rows, "strict_zero_shot"),
                "target_unlabeled_ea": summary(rows, "target_unlabeled_ea"),
            },
            "unseen_only": {
                "strict_zero_shot": summary(unseen, "strict_zero_shot"),
                "target_unlabeled_ea": summary(unseen, "target_unlabeled_ea"),
            },
            "training_seen_only": {
                "strict_zero_shot": summary(seen, "strict_zero_shot"),
                "target_unlabeled_ea": summary(seen, "target_unlabeled_ea"),
            },
        },
    }
    output = args.output or (
        ROOT / "results" / "mirepnet" / f"all-patients__{checkpoint.stem}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("\nUnseen-only macro results:")
    print(f"  strict={payload['aggregate']['unseen_only']['strict_zero_shot']['macro_accuracy'] * 100:.2f}%")
    print(f"  unlabeled EA={payload['aggregate']['unseen_only']['target_unlabeled_ea']['macro_accuracy'] * 100:.2f}%")
    print(f"result: {output.resolve()}")
    print(f"total runtime: {payload['run_seconds'] / 60:.2f} minutes")


if __name__ == "__main__":
    main()
