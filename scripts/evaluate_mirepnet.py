"""Participant-held-out evaluation for the live MIRepNet decoder.

Example (subject 22 is never used for fitting or model selection):

    uv run python scripts/evaluate_mirepnet.py --holdout-subject 22

The script reports both strict zero-shot performance (training-domain EA
reference) and deployment-style performance after fitting EA on the target
participant's *unlabeled* test EEG.  No target labels enter training, early
stopping, preprocessing parameter estimation, or alignment.
"""

from __future__ import annotations

import argparse
import json
import os
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


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--holdout-subject", type=int, required=True,
                    help="participant excluded completely from fine-tuning/model selection")
    ap.add_argument("--train-subjects", type=int, nargs="+", default=None,
                    help="optional explicit training-participant IDs; default uses every non-target participant")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default=DEEP_DEVICE)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--output-dir", type=Path, default=ROOT / "results" / "mirepnet")
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="evaluate an existing fine-tuned checkpoint instead of training")
    return ap


def metrics(y_true, y_pred):
    labels = sorted(np.unique(y_true).tolist())
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
    }


def main(argv=None):
    args = parser().parse_args(argv)
    started = time.perf_counter()
    files = sorted(args.data_dir.glob("*_mi_raw.fif"))
    if args.train_subjects is not None:
        requested = list(dict.fromkeys(args.train_subjects))
        if args.holdout_subject in requested:
            raise SystemExit("held-out participant cannot also be a training participant")
        train_files = [path for path in files if subject_id(path) in requested]
        found = set(map(subject_id, train_files))
        missing = sorted(set(requested) - found)
        if missing:
            raise SystemExit(f"no recordings found for training participant(s) {missing}")
    else:
        train_files = [path for path in files if subject_id(path) != args.holdout_subject]
    test_files = [path for path in files if subject_id(path) == args.holdout_subject]
    if not train_files or not test_files:
        raise SystemExit(f"need both training files and subject {args.holdout_subject} test files")

    print(f"MIRepNet participant-held-out evaluation\n"
          f"  train: {len(train_files)} files from "
          f"{len(set(map(subject_id, train_files)))} participants\n"
          f"  test:  {len(test_files)} files from held-out participant {args.holdout_subject}\n"
          f"  target labels used in fitting: NO")

    test_parts = [mirepnet.process_data(str(path), TARGET_MAPPINGS) for path in test_files]
    X_test = np.concatenate([part[0] for part in test_parts])
    y_test = np.concatenate([part[1] for part in test_parts])

    if args.checkpoint:
        model = mirepnet.MIRepNetDecoder.load(str(args.checkpoint), device=args.device)
        checkpoint = args.checkpoint
    else:
        train_parts = [mirepnet.process_data(str(path), TARGET_MAPPINGS) for path in train_files]
        X_train = np.concatenate([part[0] for part in train_parts])
        y_train = np.concatenate([part[1] for part in train_parts])
        groups = np.concatenate([
            np.full(len(labels), index, dtype=np.int64)
            for index, (_, labels) in enumerate(train_parts)
        ])
        model = mirepnet.MIRepNetDecoder(
            epochs=args.epochs, batch_size=args.batch_size,
            seed=args.seed, device=args.device,
        ).fit(X_train, y_train, groups=groups)
        checkpoint = Path(model.save(sources=[str(path) for path in train_files]))
        print(f"fine-tuned checkpoint: {checkpoint}")

    # First use the training-pool fallback reference: no target data of any kind.
    zero_shot = metrics(y_test, model.predict(X_test))

    # Then reproduce deployment EA: use target signals but never target labels.
    model.set_reference(X_test)
    adapted_pred = model.predict(X_test)
    target_aligned = metrics(y_test, adapted_pred)

    per_file = []
    offset = 0
    for path, (_, labels) in zip(test_files, test_parts):
        n = len(labels)
        per_file.append({"file": path.name, **metrics(labels, adapted_pred[offset:offset + n])})
        offset += n

    payload = {
        "format": "mirepnet-held-out-evaluation-v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_seconds": float(time.perf_counter() - started),
        "holdout_subject": args.holdout_subject,
        "target_labels_used_for_training": False,
        "target_signal_used_for_target_aligned_ea": True,
        "train_files": [path.name for path in train_files],
        "test_files": [path.name for path in test_files],
        "checkpoint": str(Path(checkpoint).resolve()),
        "pretrained_model": mirepnet.MODEL_ID,
        "pretrained_revision": mirepnet.MODEL_REVISION,
        "preprocessing": {
            "band_hz": [mirepnet.FMIN, mirepnet.FMAX],
            "sfreq_hz": mirepnet.SFREQ,
            "samples": mirepnet.N_TIMES,
            "observed_channels": mirepnet.SOURCE_CHANNELS,
            "template_channels": mirepnet.CHANNELS,
            "alignment": "EA on observed channels, then inverse-distance interpolation",
        },
        "strict_zero_shot": zero_shot,
        "target_unlabeled_ea": target_aligned,
        "per_test_file_target_unlabeled_ea": per_file,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_tag = "-".join(map(str, sorted(set(map(subject_id, train_files)))))
    output = args.output_dir / (
        f"train-{train_tag}__test-{args.holdout_subject}__seed-{args.seed}.json"
    )
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("\nStrict zero-shot (no target signals):")
    print(f"  accuracy={zero_shot['accuracy'] * 100:.2f}%  "
          f"balanced={zero_shot['balanced_accuracy'] * 100:.2f}%")
    print("Target-unlabeled EA (deployment protocol):")
    print(f"  accuracy={target_aligned['accuracy'] * 100:.2f}%  "
          f"balanced={target_aligned['balanced_accuracy'] * 100:.2f}%")
    print(f"  confusion={target_aligned['confusion_matrix']}")
    print(f"result: {output}")
    print(f"total runtime: {payload['run_seconds'] / 60:.2f} minutes")


if __name__ == "__main__":
    main()
