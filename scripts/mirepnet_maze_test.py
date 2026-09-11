"""Test one patient's recorded EEG on a standard TIAGo maze.

The project has four fixed 40-corner mazes (historical seeds 11--14). For each
corner, the test takes an unused EEG epoch whose recorded instruction matches
the turn that the fixed maze requires. MIRepNet inference runs only when the
robot reaches that corner, and the new prediction immediately controls it.
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
sys.path[:0] = [str(ROOT / "classifier"), str(ROOT), str(ROOT / "scripts"),
                str(ROOT / "simulation" / "src")]

from config import MIREPNET_DEVICE, MIREPNET_WINDOW_MODE, TARGET_MAPPINGS  # noqa: E402
import mirepnet  # noqa: E402
from test_mirepnet_patients import choose_checkpoint, run_number, subject_id  # noqa: E402
from tiago_maze.standards import STANDARD_MAZES, get_standard_maze  # noqa: E402


PROTOCOLS = {
    "1": "zero-shot",
    "2": "unlabeled",
    "3": "calibrated",
}


def choose_subject(files):
    available = sorted(set(map(subject_id, files)))
    print("\nPatient to replay in the maze:")
    for subject in available:
        count = sum(subject_id(path) == subject for path in files)
        print(f"  S{subject:02d}: {count} recording(s)")
    while True:
        try:
            raw = input("patient (example 9 or S09)> ").strip()
        except (EOFError, KeyboardInterrupt):
            raise SystemExit(130)
        cleaned = re.sub(r"^s", "", raw, flags=re.IGNORECASE)
        if cleaned.isdigit() and int(cleaned) in available:
            return int(cleaned)
        print("choose one patient from: " + ", ".join(f"S{x:02d}" for x in available))


def choose_protocol():
    print("\nHow should MIRepNet test this patient?")
    print("  1) Zero-shot           — no information from this patient is used first")
    print("  2) Unlabeled EA        — adjusts signal scale/shape; does not read labels")
    print("  3) Calibrated          — learns from the first recordings, tests later ones")
    print("\nBalanced-block mode is not used here because it needs all predictions")
    print("before making decisions; this maze tests one EEG epoch at each corner.")
    while True:
        try:
            raw = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            raise SystemExit(130)
        if raw in PROTOCOLS:
            return PROTOCOLS[raw]
        if raw in PROTOCOLS.values():
            return raw
        print("enter 1-3")


def choose_standard_maze():
    print("\nChoose one of the four original fixed mazes:")
    for number, maze in STANDARD_MAZES.items():
        left = maze.route.count("L")
        right = maze.route.count("R")
        print(f"  {number}) Standard maze {number} — original seed {maze.seed}; "
              f"40 corners ({left} left, {right} right)")
    while True:
        try:
            raw = input("maze number> ").strip()
        except (EOFError, KeyboardInterrupt):
            raise SystemExit(130)
        if raw.isdigit() and int(raw) in STANDARD_MAZES:
            return int(raw)
        print("enter 1, 2, 3, or 4")


def _flatten_epoch_metadata(eval_parts, eval_files):
    rows = []
    test_epoch = 0
    for path, (_, labels) in zip(eval_files, eval_parts):
        for epoch_in_file, label in enumerate(labels, 1):
            test_epoch += 1
            rows.append({
                "_array_index": test_epoch - 1,
                "test_epoch": test_epoch,
                "file": path.name,
                "epoch_in_file": epoch_in_file,
                "true_label": int(label),
            })
    return rows


def _prepare_test(model, protocol, parts, patient_files, calibration_runs):
    """Prepare adaptation only; do not calculate any test predictions."""
    calibrated = protocol == "calibrated"
    if calibrated:
        if len(parts) <= calibration_runs:
            raise SystemExit(
                f"calibrated testing needs more than {calibration_runs} recordings; "
                f"this patient has {len(parts)}"
            )
        cal_parts = parts[:calibration_runs]
        eval_parts = parts[calibration_runs:]
        eval_files = patient_files[calibration_runs:]
    else:
        cal_parts = []
        eval_parts = parts
        eval_files = patient_files

    X_eval = np.concatenate([X for X, _ in eval_parts])
    y_eval = np.concatenate([y for _, y in eval_parts]).astype(int)

    feature_reference = None
    if protocol == "unlabeled":
        model.set_reference(X_eval)
    elif protocol == "calibrated":
        X_cal = np.concatenate([X for X, _ in cal_parts])
        y_cal = np.concatenate([y for _, y in cal_parts])
        model.fit_feature_adapter(X_cal, y_cal)
        # The calibrated head's target-domain normalization is unlabeled. Fit
        # those statistics once, but leave every class prediction uncomputed.
        model.set_reference(X_eval)
        features = model.extract_features(X_eval)
        feature_reference = (
            features.mean(axis=0), np.maximum(features.std(axis=0), 1e-5)
        )

    metadata = _flatten_epoch_metadata(eval_parts, eval_files)
    return X_eval, y_eval, metadata, eval_files, feature_reference


class OnDemandMIRepNet:
    """Run one genuinely new classifier decision when each corner requests it."""

    def __init__(self, model, protocol, X_eval, selected_indices, feature_reference,
                 payload, output):
        self.model = model
        self.protocol = protocol
        self.X_eval = X_eval
        self.selected_indices = list(selected_indices)
        self.feature_reference = feature_reference
        self.payload = payload
        self.output = Path(output)

    def _probabilities(self, array_index):
        X_one = self.X_eval[array_index:array_index + 1]
        if self.protocol != "calibrated":
            return np.asarray(self.model.classes_), self.model.predict_proba(X_one)[0]
        feature = self.model.extract_features(X_one)
        mean, scale = self.feature_reference
        feature = (feature - mean) / scale
        adapter = self.model.feature_adapter_
        classes = np.asarray(adapter.named_steps["logisticregression"].classes_)
        return classes, adapter.predict_proba(feature)[0]

    def _save_progress(self):
        finished = [row for row in self.payload["turns"]
                    if row.get("predicted_label") is not None]
        if finished:
            y_true = np.asarray([row["true_label"] for row in finished])
            y_pred = np.asarray([row["predicted_label"] for row in finished])
            labels = [1, 2]
            self.payload["maze_test"].update({
                "predictions_completed": len(finished),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "balanced_accuracy": (
                    float(balanced_accuracy_score(y_true, y_pred))
                    if len(np.unique(y_true)) == 2 else None
                ),
                "confusion_matrix": confusion_matrix(
                    y_true, y_pred, labels=labels
                ).tolist(),
            })
        self.output.write_text(json.dumps(self.payload, indent=2) + "\n", encoding="utf-8")

    def __call__(self, turn_index, planned_row):
        if planned_row.get("predicted_label") is not None:
            raise RuntimeError(f"maze turn {turn_index + 1} was already predicted")
        started = time.perf_counter()
        classes, probabilities = self._probabilities(self.selected_indices[turn_index])
        predicted = int(classes[int(np.argmax(probabilities))])
        class_index = {int(label): index for index, label in enumerate(classes)}
        confidence = float(probabilities[class_index[predicted]])
        row = dict(planned_row)
        row.update({
            "predicted_label": predicted,
            "confidence": confidence,
            "correct": bool(int(row["true_label"]) == predicted),
            "inference_ms": float((time.perf_counter() - started) * 1000.0),
            "predicted_at": datetime.now().isoformat(timespec="milliseconds"),
        })
        self.payload["turns"][turn_index] = row
        self._save_progress()
        wanted = "LEFT" if int(row["true_label"]) == 1 else "RIGHT"
        guessed = "LEFT" if predicted == 1 else "RIGHT"
        result = "CORRECT" if row["correct"] else "WRONG"
        print(f"corner {turn_index + 1:02d}/40: maze needs {wanted}; "
              f"MIRepNet predicts {guessed} ({confidence * 100:.1f}%) — {result}")
        return row


def select_route_epochs(epoch_rows, required_labels):
    """Choose chronological EEG epochs whose labels match a fixed maze route."""
    queues = {
        label: [dict(row) for row in epoch_rows if int(row["true_label"]) == label]
        for label in (1, 2)
    }
    used = {1: 0, 2: 0}
    selected = []
    for maze_turn, label in enumerate(required_labels, 1):
        label = int(label)
        if used[label] >= len(queues[label]):
            direction = "LEFT" if label == 1 else "RIGHT"
            raise SystemExit(
                f"not enough recorded {direction} EEG epochs for this 40-corner maze"
            )
        row = queues[label][used[label]]
        used[label] += 1
        row["maze_turn"] = maze_turn
        selected.append(row)
    return selected


def build_test(args):
    files = sorted(args.data_dir.glob("*_mi_raw.fif"))
    if not files:
        raise SystemExit(f"no patient recordings found in {args.data_dir}")
    checkpoint = args.checkpoint or choose_checkpoint()
    subject = args.subject if args.subject is not None else choose_subject(files)
    patient_files = sorted(
        [path for path in files if subject_id(path) == subject], key=run_number
    )
    if not patient_files:
        raise SystemExit(f"no recordings found for S{subject:02d}")
    protocol = args.protocol or choose_protocol()
    standard = get_standard_maze(args.maze) if args.maze else get_standard_maze(
        choose_standard_maze()
    )

    print(f"\nLoading MIRepNet weights: {Path(checkpoint).name}")
    print(f"Reading S{subject:02d} EEG recordings...")
    model = mirepnet.MIRepNetDecoder.load(str(checkpoint), device=args.device)
    parts = [
        mirepnet.process_data(str(path), TARGET_MAPPINGS, args.window_mode)
        for path in patient_files
    ]
    X_eval, y_eval, epoch_rows, eval_files, feature_reference = _prepare_test(
        model, protocol, parts, patient_files, args.calibration_runs
    )
    selected = select_route_epochs(epoch_rows, standard.labels)
    turn_count = len(selected)
    selected_indices = [int(row.pop("_array_index")) for row in selected]
    for row in selected:
        row.update({"predicted_label": None, "confidence": None, "correct": None})
    created_at = datetime.now()
    payload = {
        "format": "mirepnet-maze-plan-v1",
        "inference_mode": "on-demand",
        "created_at": created_at.isoformat(timespec="seconds"),
        "subject": f"S{subject:02d}",
        "protocol": protocol,
        "standard_maze": standard.number,
        "maze_seed": standard.seed,
        "maze_route": standard.route,
        "checkpoint": str(Path(checkpoint).resolve()),
        "calibration_files": [path.name for path in patient_files[:args.calibration_runs]]
        if protocol.startswith("calibrated") else [],
        "test_files": [path.name for path in eval_files],
        "label_meaning": {"1": "LEFT", "2": "RIGHT"},
        "eligible_patient_epochs": int(len(y_eval)),
        "maze_test": {
            "epochs": turn_count,
            "predictions_completed": 0,
            "accuracy": None,
            "balanced_accuracy": None,
            "confusion_matrix": None,
            "selection": (
                "chronological EEG queues matched to each required turn of the fixed maze"
            ),
        },
        "turns": selected,
    }
    output = args.output or (
        ROOT / "results" / "mirepnet" /
        f"standard-maze{standard.number}_S{subject:02d}_{created_at:%Y%m%d_%H%M%S}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("\nMAZE TEST READY")
    print(f"  Patient: S{subject:02d}")
    print(f"  Method: {protocol}")
    print(f"  Fixed maze: {standard.number} (original seed {standard.seed}, 40 corners)")
    print(f"  Eligible Patient {subject} test epochs: {len(y_eval)}")
    print("  Predictions calculated now: 0")
    print("  Each prediction will be calculated only when its maze corner is reached.")
    print(f"  Live test record: {output.resolve()}")
    provider = OnDemandMIRepNet(
        model, protocol, X_eval, selected_indices, feature_reference, payload, output
    )
    return output, subject, protocol, standard.number, provider


def launch_maze(plan, subject, protocol, maze_number, provider, args):
    simulation_dir = ROOT / "simulation"
    sim_args = [
        "--decision-plan", str(plan.resolve()),
        "--subject", f"S{subject:02d}",
        "--test", f"mirepnet-{protocol}-maze{maze_number}",
        "--view", args.view,
        "--forward-speed", str(args.forward_speed),
        "--turn-speed", str(args.turn_speed),
    ]
    if args.offscreen:
        sim_args.append("--offscreen")
    if args.frames is not None:
        sim_args.extend(["--frames", str(args.frames)])
    print("\nOpening TIAGo. Patient EEG will be tested one epoch per maze corner...")
    old_path = list(sys.path)
    try:
        sys.path.insert(0, str(simulation_dir / "src"))
        from tiago_maze.__main__ import build_params
        from tiago_maze.game import MazeGame
        params = build_params(sim_args)
        params.decision_provider = provider
        MazeGame(params).run()
    finally:
        sys.path[:] = old_path


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--subject", type=int, default=None)
    ap.add_argument("--protocol", choices=tuple(PROTOCOLS.values()), default=None)
    ap.add_argument("--calibration-runs", type=int, default=2)
    ap.add_argument("--maze", type=int, choices=tuple(STANDARD_MAZES), default=None,
                    help="one of the four original fixed mazes")
    ap.add_argument("--window-mode", choices=("plan-task", "task-repeat"),
                    default=MIREPNET_WINDOW_MODE)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--device", default=MIREPNET_DEVICE)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--view", choices=("third", "first", "top"), default="third")
    ap.add_argument("--forward-speed", type=float, default=1.2)
    ap.add_argument("--turn-speed", type=float, default=1.5)
    ap.add_argument("--no-launch", action="store_true",
                    help="prepare the pending test without calculating predictions")
    ap.add_argument("--offscreen", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--frames", type=int, default=None, help=argparse.SUPPRESS)
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    if args.calibration_runs < 1:
        raise SystemExit("--calibration-runs must be at least 1")
    plan, subject, protocol, maze_number, provider = build_test(args)
    if not args.no_launch:
        launch_maze(plan, subject, protocol, maze_number, provider, args)


if __name__ == "__main__":
    main()
