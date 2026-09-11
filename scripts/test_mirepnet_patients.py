"""Interactively test selected patients with saved MIRepNet weights.

Three protocols are kept separate:
  zero-shot   -- no target-patient signal or label is used for adaptation;
  unlabeled   -- the completed target EEG batch fits EA, but labels are untouched;
  calibrated  -- the first chronological recordings fit a small patient head and
                 only later recordings are tested.
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

from config import (LOCAL_EXP4_VALID_RUNS, MIREPNET_DEVICE, MIREPNET_WINDOW_MODE,
                    TARGET_MAPPINGS)  # noqa: E402
import mirepnet  # noqa: E402
from mirepnet_excel import export_test_workbook  # noqa: E402


def subject_id(path):
    match = re.search(r"subject(\d+)", Path(path).name)
    if not match:
        raise ValueError(f"cannot parse patient from {path}")
    return int(match.group(1))


def run_number(path):
    match = re.search(r"training_(\d+)_mi_raw", Path(path).name)
    if not match:
        raise ValueError(f"cannot parse recording number from {path}")
    return int(match.group(1))


def score(y_true, y_pred):
    labels = sorted(np.unique(y_true).tolist())
    return {
        "n_trials": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
    }


def append_epoch_rows(rows, run_id, subject, training_seen, protocol, eval_parts,
                      eval_files, prediction, probabilities, classes):
    """Record one flat, Excel-ready row for every tested EEG epoch."""
    prediction = np.asarray(prediction)
    probabilities = np.asarray(probabilities)
    class_index = {int(label): index for index, label in enumerate(classes)}
    offset = 0
    for path, (_, labels) in zip(eval_files, eval_parts):
        for file_epoch, true_label in enumerate(labels, 1):
            test_index = offset + file_epoch - 1
            predicted_label = int(prediction[test_index])
            confidence = float(probabilities[test_index, class_index[predicted_label]])
            rows.append({
                "run_id": run_id,
                "subject": int(subject),
                "training_seen": bool(training_seen),
                "protocol": protocol,
                "file": path.name,
                "epoch_in_file": int(file_epoch),
                "test_epoch": int(test_index + 1),
                "true_label": int(true_label),
                "predicted_label": predicted_label,
                "confidence": confidence,
                "correct": bool(int(true_label) == predicted_label),
            })
        offset += len(labels)


def choose_checkpoint():
    checkpoints = mirepnet.list_checkpoints()
    if not checkpoints:
        raise SystemExit("no saved MIRepNet checkpoints found")
    print("\nSaved MIRepNet weights:")
    for index, path in enumerate(checkpoints, 1):
        print(f"  {index}) {mirepnet.describe_checkpoint(path)}")
    while True:
        try:
            raw = input("checkpoint number> ").strip()
        except (EOFError, KeyboardInterrupt):
            raise SystemExit(130)
        if raw.isdigit() and 1 <= int(raw) <= len(checkpoints):
            return Path(checkpoints[int(raw) - 1])
        print(f"enter 1-{len(checkpoints)}")


def choose_subjects(files):
    available = sorted(set(map(subject_id, files)))
    print("\nAvailable patients:")
    for subject in available:
        patient_files = [path for path in files if subject_id(path) == subject]
        valid = sum(run_number(path) in LOCAL_EXP4_VALID_RUNS.get(subject, ())
                    for path in patient_files)
        outside = len(patient_files) - valid
        tag = (f"{valid} validated" + (f", {outside} outside manifest" if outside else "")
               if valid else "exploratory / outside manifest")
        print(f"  S{subject:02d}: {len(patient_files)} recording(s) [{tag}]")
    while True:
        try:
            raw = input("patients (example 1,5,8 or all)> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            raise SystemExit(130)
        if raw == "all":
            return available
        try:
            chosen = list(dict.fromkeys(
                int(re.sub(r"^s", "", item.strip(), flags=re.IGNORECASE))
                for item in raw.split(",") if item.strip()
            ))
        except ValueError:
            chosen = []
        missing = sorted(set(chosen) - set(available))
        if chosen and not missing:
            return chosen
        if missing:
            print("no recording files found for: " +
                  ", ".join(f"S{subject:02d}" for subject in missing))
            print("available patients: " +
                  ", ".join(f"S{subject:02d}" for subject in available))
        else:
            print("enter comma-separated patients such as 1,5,8 or S01,S05,S08; "
                  "or enter 'all'")


def choose_protocol():
    print("\nTest protocol:")
    print("  1) Strict zero-shot — genuinely unseen; no patient adaptation")
    print("  2) Unlabeled EA     — uses patient EEG signals, never labels")
    print("  3) Calibrated       — first recordings train a small patient head")
    print("  4) Compare all three on the same later recordings")
    mapping = {"1": "zero-shot", "2": "unlabeled", "3": "calibrated", "4": "all"}
    while True:
        try:
            raw = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            raise SystemExit(130)
        if raw in mapping:
            return mapping[raw]
        if raw in mapping.values():
            return raw
        print("enter 1-4")


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--subjects", type=int, nargs="+", default=None)
    ap.add_argument("--protocol", choices=("zero-shot", "unlabeled", "calibrated", "all"),
                    default=None)
    ap.add_argument("--calibration-runs", type=int, default=2)
    ap.add_argument("--window-mode", choices=("plan-task", "task-repeat"),
                    default=MIREPNET_WINDOW_MODE)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--validated-only", action="store_true",
                    help="restrict data to the frozen eight-patient Local Exp4 manifest")
    ap.add_argument("--device", default=MIREPNET_DEVICE)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--xlsx-output", type=Path, default=None)
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    started = time.perf_counter()
    files = sorted(args.data_dir.glob("*_mi_raw.fif"))
    if not files:
        raise SystemExit(f"no patient recordings found in {args.data_dir}")
    if args.validated_only:
        files = [path for path in files if run_number(path) in
                 LOCAL_EXP4_VALID_RUNS.get(subject_id(path), ())]

    checkpoint = args.checkpoint or choose_checkpoint()
    subjects = args.subjects or choose_subjects(files)
    protocol = args.protocol or choose_protocol()
    available = set(map(subject_id, files))
    missing = sorted(set(subjects) - available)
    if missing:
        raise SystemExit(f"no recordings found for patient(s): {missing}")
    if args.calibration_runs < 1:
        raise SystemExit("--calibration-runs must be at least 1")

    model = mirepnet.MIRepNetDecoder.load(str(checkpoint), device=args.device)
    training_reference = np.asarray(model.ref_white_).copy()
    training_subjects = sorted({
        subject_id(name) for name in model.meta_.get("trained_on", [])
        if re.search(r"subject\d+", Path(name).name)
    })
    rows = []
    epoch_rows = []
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\nCheckpoint: {checkpoint.name}")
    print(f"Checkpoint training patients: {training_subjects or 'unknown'}")
    print(f"Selected patients: {subjects}")
    print(f"Protocol: {protocol}; window: {args.window_mode}")
    print(f"Recording scope: {'validated manifest only' if args.validated_only else 'all discovered files'}")
    print("Foundation weights updated: NO\n")

    for index, subject in enumerate(subjects, 1):
        patient_files = sorted(
            [path for path in files if subject_id(path) == subject], key=run_number
        )
        parts = [mirepnet.process_data(str(path), TARGET_MAPPINGS, args.window_mode)
                 for path in patient_files]
        use_calibration_split = protocol in {"calibrated", "all"}
        if use_calibration_split:
            if len(parts) <= args.calibration_runs:
                print(f"S{subject:02d}: skipped; needs more than {args.calibration_runs} recordings")
                continue
            cal_parts = parts[:args.calibration_runs]
            eval_parts = parts[args.calibration_runs:]
            eval_files = patient_files[args.calibration_runs:]
        else:
            cal_parts = []
            eval_parts = parts
            eval_files = patient_files

        X_eval = np.concatenate([X for X, _ in eval_parts])
        y_eval = np.concatenate([y for _, y in eval_parts])
        row = {
            "subject": subject,
            "training_seen": subject in training_subjects,
            "calibration_files": [path.name for path in patient_files[:args.calibration_runs]]
                                 if use_calibration_split else [],
            "test_files": [path.name for path in eval_files],
        }

        if protocol in {"zero-shot", "all"}:
            model.ref_white_ = training_reference.copy()
            probabilities = model.predict_proba(X_eval)
            prediction = model.classes_[probabilities.argmax(axis=1)]
            row["strict_zero_shot"] = score(y_eval, prediction)
            append_epoch_rows(epoch_rows, run_id, subject, row["training_seen"],
                              "strict_zero_shot", eval_parts, eval_files, prediction,
                              probabilities, model.classes_)

        if protocol in {"unlabeled", "all"}:
            model.set_reference(X_eval)
            probabilities = model.predict_proba(X_eval)
            prediction = model.classes_[probabilities.argmax(axis=1)]
            row["unlabeled_ea"] = score(y_eval, prediction)
            append_epoch_rows(epoch_rows, run_id, subject, row["training_seen"],
                              "unlabeled_ea", eval_parts, eval_files, prediction,
                              probabilities, model.classes_)

        if protocol in {"calibrated", "all"}:
            X_cal = np.concatenate([X for X, _ in cal_parts])
            y_cal = np.concatenate([y for _, y in cal_parts])
            model.fit_feature_adapter(X_cal, y_cal)
            probabilities = model.predict_feature_adapter_proba(X_eval)
            adapter_classes = model.feature_adapter_.named_steps["logisticregression"].classes_
            raw_prediction = adapter_classes[probabilities.argmax(axis=1)]
            balanced_prediction = model.predict_feature_adapter(X_eval, balanced_protocol=True)
            row["calibrated_raw"] = score(y_eval, raw_prediction)
            row["calibrated_balanced_block"] = score(y_eval, balanced_prediction)
            append_epoch_rows(epoch_rows, run_id, subject, row["training_seen"],
                              "calibrated_raw", eval_parts, eval_files, raw_prediction,
                              probabilities, adapter_classes)
            append_epoch_rows(epoch_rows, run_id, subject, row["training_seen"],
                              "calibrated_balanced_block", eval_parts, eval_files,
                              balanced_prediction, probabilities, adapter_classes)

        rows.append(row)
        status = "SEEN" if row["training_seen"] else "UNSEEN"
        values = []
        for key, label in (("strict_zero_shot", "zero"), ("unlabeled_ea", "EA"),
                           ("calibrated_raw", "cal"),
                           ("calibrated_balanced_block", "cal-balanced")):
            if key in row:
                values.append(f"{label}={row[key]['accuracy'] * 100:.2f}%")
        print(f"[{index:02d}/{len(subjects):02d}] S{subject:02d} {status}: " + "  ".join(values))

    metric_keys = ("strict_zero_shot", "unlabeled_ea", "calibrated_raw",
                   "calibrated_balanced_block")
    aggregate = {}
    for key in metric_keys:
        values = [row[key]["accuracy"] for row in rows if key in row]
        if values:
            aggregate[key] = {
                "n_subjects": len(values),
                "macro_accuracy": float(np.mean(values)),
                "subjects_at_or_above_80pct": int(sum(value >= 0.8 for value in values)),
                "subjects_at_or_above_90pct": int(sum(value >= 0.9 for value in values)),
            }

    payload = {
        "format": "mirepnet-selected-patient-test-v2",
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_training_subjects": training_subjects,
        "subjects_requested": subjects,
        "protocol": protocol,
        "recording_scope": "validated_local_exp4" if args.validated_only else "all_discovered",
        "window_mode": args.window_mode,
        "calibration_runs": args.calibration_runs if protocol in {"calibrated", "all"} else 0,
        "foundation_weights_updated": False,
        "protocol_contract": {
            "strict_zero_shot_uses_target_adaptation": False,
            "unlabeled_ea_uses_target_signals": True,
            "unlabeled_ea_uses_target_labels": False,
            "calibrated_uses_early_recording_labels": True,
            "test_labels_used_for_adaptation": False,
        },
        "results": rows,
        "epoch_results": epoch_rows,
        "aggregate": aggregate,
        "run_seconds": float(time.perf_counter() - started),
    }
    output = args.output or ROOT / "results" / "mirepnet" / f"selected-patients_{run_id}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    xlsx_output = export_test_workbook(output, args.xlsx_output)
    if aggregate:
        print("\nAggregate:")
        for key, values in aggregate.items():
            print(f"  {key}: {values['macro_accuracy'] * 100:.2f}%  "
                  f">=80% {values['subjects_at_or_above_80pct']}/{values['n_subjects']}  "
                  f">=90% {values['subjects_at_or_above_90pct']}/{values['n_subjects']}")
    print(f"\nSaved detailed result: {output.resolve()}")
    print(f"Saved Excel epoch report: {xlsx_output.resolve()}")
    print(f"Runtime: {payload['run_seconds'] / 60:.2f} minutes")


if __name__ == "__main__":
    main()
