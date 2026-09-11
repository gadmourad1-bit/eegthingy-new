"""Fine-tune MIRepNet on user-selected patients or individual recordings."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "classifier"), str(ROOT)]

from config import (MIREPNET_DEVICE, MIREPNET_EPOCHS, MIREPNET_SEED,
                    MIREPNET_WINDOW_MODE, TARGET_MAPPINGS)  # noqa: E402
import mirepnet  # noqa: E402


def subject_id(path):
    match = re.search(r"subject(\d+)", Path(path).name, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"cannot parse patient from {path}")
    return int(match.group(1))


def parse_indices(raw, count):
    raw = raw.strip().lower()
    if raw == "all":
        return list(range(count))
    chosen = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, end = token.split("-", 1)
            chosen.extend(range(int(start), int(end) + 1))
        else:
            chosen.append(int(token))
    chosen = list(dict.fromkeys(chosen))
    if not chosen or any(index < 1 or index > count for index in chosen):
        raise ValueError
    return [index - 1 for index in chosen]


def choose_subject_files(files):
    available = sorted(set(map(subject_id, files)))
    print("\nAvailable patients:")
    for subject in available:
        count = sum(subject_id(path) == subject for path in files)
        print(f"  S{subject:02d}: {count} recording(s)")
    while True:
        try:
            raw = input("patients (example 1,5,8 or S01,S05,S08; or all)> ").strip()
            if raw.lower() == "all":
                chosen = available
            else:
                chosen = list(dict.fromkeys(
                    int(re.sub(r"^s", "", item.strip(), flags=re.IGNORECASE))
                    for item in raw.split(",") if item.strip()
                ))
        except (ValueError, EOFError):
            chosen = []
        except KeyboardInterrupt:
            raise SystemExit(130)
        missing = sorted(set(chosen) - set(available))
        if chosen and not missing:
            return [path for path in files if subject_id(path) in chosen]
        if missing:
            print("no recording files found for: " +
                  ", ".join(f"S{subject:02d}" for subject in missing))
        else:
            print("enter one or more available patients")


def choose_individual_files(files):
    print("\nAvailable recordings:")
    for index, path in enumerate(files, 1):
        print(f"  {index:3d}) {path.name}")
    while True:
        try:
            indices = parse_indices(input("files (example 1,3-6 or all)> "), len(files))
            return [files[index] for index in indices]
        except (ValueError, EOFError):
            print(f"enter file numbers between 1 and {len(files)}, ranges, or all")
        except KeyboardInterrupt:
            raise SystemExit(130)


def choose_training_files(files):
    print("\nFine-tuning data selection:")
    print("  1) Select whole patients")
    print("  2) Select individual recording files")
    while True:
        try:
            raw = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            raise SystemExit(130)
        if raw in ("1", "patient", "patients", "subject", "subjects"):
            return choose_subject_files(files)
        if raw in ("2", "file", "files", "recording", "recordings"):
            return choose_individual_files(files)
        print("enter 1 or 2")


def choose_initial_checkpoint():
    checkpoints = mirepnet.list_checkpoints()
    print("\nStarting weights:")
    print("  0) Official pretrained MIRepNet weights (fresh task head)")
    for index, path in enumerate(checkpoints, 1):
        print(f"  {index}) Continue from {mirepnet.describe_checkpoint(path)}")
    while True:
        try:
            raw = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            raise SystemExit(130)
        if raw in ("0", "official", "pretrained", "fresh"):
            return None
        if raw.isdigit() and 1 <= int(raw) <= len(checkpoints):
            return Path(checkpoints[int(raw) - 1])
        print(f"enter 0-{len(checkpoints)}")


def prompt_int(label, default, minimum=1):
    while True:
        try:
            raw = input(f"{label} [{default}]> ").strip()
        except (EOFError, KeyboardInterrupt):
            raise SystemExit(130)
        if not raw:
            return default
        if raw.isdigit() and int(raw) >= minimum:
            return int(raw)
        print(f"enter an integer >= {minimum}")


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    selection = ap.add_mutually_exclusive_group()
    selection.add_argument("--subjects", type=int, nargs="+")
    selection.add_argument("--files", type=Path, nargs="+")
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="continue fine-tuning from this project checkpoint")
    ap.add_argument("--official", action="store_true",
                    help="start from the official pretrained encoder instead of prompting")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--device", default=MIREPNET_DEVICE)
    ap.add_argument("--window-mode", choices=("plan-task", "task-repeat"),
                    default=MIREPNET_WINDOW_MODE)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--output", type=Path, default=None)
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    files = sorted(args.data_dir.glob("*_mi_raw.fif"))
    if not files:
        raise SystemExit(f"no patient recordings found in {args.data_dir}")

    if args.subjects:
        missing = sorted(set(args.subjects) - set(map(subject_id, files)))
        if missing:
            raise SystemExit("no recording files found for: " +
                             ", ".join(f"S{subject:02d}" for subject in missing))
        selected = [path for path in files if subject_id(path) in args.subjects]
    elif args.files:
        selected = [path if path.is_absolute() else args.data_dir / path for path in args.files]
        missing = [path for path in selected if not path.is_file()]
        if missing:
            raise SystemExit("missing recording file(s): " + ", ".join(map(str, missing)))
    else:
        selected = choose_training_files(files)

    initial = args.checkpoint
    if initial is None and not args.official:
        initial = choose_initial_checkpoint()
    if initial is not None and not initial.is_file():
        raise SystemExit(f"checkpoint not found: {initial}")
    epochs = args.epochs or prompt_int("maximum fine-tuning epochs", MIREPNET_EPOCHS)
    batch_size = args.batch_size or prompt_int("batch size", 16)

    print(f"\nFine-tuning on {len(selected)} recording(s) from "
          f"{len(set(map(subject_id, selected)))} patient(s):")
    for path in selected:
        print(f"  - {path.name}")
    print("Starting from: " + (str(initial.resolve()) if initial else "official pretrained MIRepNet"))

    parts = [mirepnet.process_data(str(path), TARGET_MAPPINGS, args.window_mode)
             for path in selected]
    X = np.concatenate([part[0] for part in parts])
    y = np.concatenate([part[1] for part in parts])
    groups = np.concatenate([[index] * len(part[1]) for index, part in enumerate(parts)])
    decoder = mirepnet.MIRepNetDecoder(
        epochs=epochs,
        seed=MIREPNET_SEED,
        device=args.device,
        batch_size=batch_size,
        window_mode=args.window_mode,
        initial_checkpoint=str(initial) if initial else None,
    ).fit(X, y, groups=groups)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output or ROOT / "models" / "mirepnet" / f"mirepnet__finetuned_{stamp}.pt"
    saved = Path(decoder.save(str(output), sources=[str(path) for path in selected])).resolve()
    report = saved.with_suffix(".training.json")
    report.write_text(json.dumps({
        "format": "mirepnet-selected-finetune-v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "checkpoint": str(saved),
        "initialized_from": decoder.initialized_from_,
        "subjects": sorted(set(map(subject_id, selected))),
        "files": [path.name for path in selected],
        "trials": int(len(y)),
        "best_epoch": int(decoder.best_epoch_),
        "best_validation_loss": float(decoder.best_validation_loss_),
        "fit_seconds": float(decoder.fit_seconds_),
        "history": decoder.history_,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved reusable weights: {saved}")
    print(f"Saved training history: {report.resolve()}")
    print("The original checkpoint was not overwritten.")


if __name__ == "__main__":
    main()
