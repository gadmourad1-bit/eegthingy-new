"""Nested leave-one-subject-out evaluation of a MIRepNet feature head.

The transformer checkpoint is frozen.  Each outer test participant is excluded
from both head fitting and inner model selection. Target EEG is used without
labels only for Euclidean Alignment and optional feature standardization.
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "classifier"), str(ROOT)]

from config import DEEP_DEVICE, TARGET_MAPPINGS  # noqa: E402
import mirepnet  # noqa: E402


def subject_id(path):
    match = re.search(r"subject(\d+)", Path(path).name)
    if not match:
        raise ValueError(f"cannot parse participant from {path}")
    return int(match.group(1))


def subject_standardize(features, groups):
    output = np.empty_like(features)
    for group in np.unique(groups):
        mask = groups == group
        mean = features[mask].mean(axis=0)
        scale = features[mask].std(axis=0)
        output[mask] = (features[mask] - mean) / np.maximum(scale, 1e-5)
    return output


def balanced_prediction(scores, classes):
    """Apply the experiment's known approximately 50/50 left/right prior."""
    count_right = len(scores) // 2
    prediction = np.full(len(scores), classes[0])
    if count_right:
        prediction[np.argsort(scores)[-count_right:]] = classes[1]
    return prediction


def candidates():
    rows = []
    # Subject-z variants address domain offsets. Keep the search intentionally
    # small so nested participant-level selection remains practical on CPU.
    for c_value in (0.01, 0.1, 1.0):
        for balance_prior in (False, True):
            rows.append({
                "name": f"logreg-C{c_value:g}-subject-z" +
                        ("-balanced-prior" if balance_prior else ""),
                "kind": "logreg", "c": c_value,
                "subject_standardize": True,
                "balanced_prior": balance_prior,
            })
    for normalized in (False, True):
        suffix = "subject-z" if normalized else "raw"
        rows.append({
            "name": f"shrinkage-lda-{suffix}-balanced-prior",
            "kind": "lda", "subject_standardize": normalized,
            "balanced_prior": True,
        })
    return rows


def estimator(spec):
    if spec["kind"] == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=spec["c"], max_iter=2000, class_weight="balanced"),
        )
    return make_pipeline(
        StandardScaler(),
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    )


def predict(model, features, spec):
    if not spec["balanced_prior"]:
        return model.predict(features)
    scores = model.decision_function(features)
    return balanced_prediction(scores, model.classes_)


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
    ap.add_argument("--device", default=DEEP_DEVICE)
    ap.add_argument("--inner-folds", type=int, default=5)
    ap.add_argument("--output", type=Path, default=None)
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    started = time.perf_counter()
    files = sorted(args.data_dir.glob("*_mi_raw.fif"))
    subjects = sorted(set(map(subject_id, files)))
    model = mirepnet.MIRepNetDecoder.load(str(args.checkpoint), device=args.device)

    feature_parts, label_parts, group_parts = [], [], []
    print(f"Frozen MIRepNet feature extraction from {args.checkpoint.name}")
    for index, subject in enumerate(subjects, start=1):
        parts = [mirepnet.process_data(str(path), TARGET_MAPPINGS)
                 for path in files if subject_id(path) == subject]
        X = np.concatenate([part[0] for part in parts])
        y = np.concatenate([part[1] for part in parts])
        model.set_reference(X)
        features = model.extract_features(X)
        feature_parts.append(features)
        label_parts.append(y)
        group_parts.append(np.full(len(y), subject, dtype=np.int64))
        print(f"features [{index:02d}/{len(subjects):02d}] subject {subject:02d}: "
              f"{features.shape}")

    X = np.concatenate(feature_parts)
    y = np.concatenate(label_parts)
    groups = np.concatenate(group_parts)
    X_subject_z = subject_standardize(X, groups)
    specs = candidates()
    rows = []

    print("\nNested leave-one-subject-out head evaluation")
    for index, subject in enumerate(subjects, start=1):
        outer_test = groups == subject
        outer_train = ~outer_test
        train_groups = groups[outer_train]
        inner = GroupKFold(n_splits=min(args.inner_folds, len(np.unique(train_groups))))
        spec_scores = []
        for spec in specs:
            source = X_subject_z if spec["subject_standardize"] else X
            fold_scores = []
            for inner_train, inner_val in inner.split(source[outer_train], y[outer_train], train_groups):
                train_indices = np.flatnonzero(outer_train)[inner_train]
                val_indices = np.flatnonzero(outer_train)[inner_val]
                head = estimator(spec).fit(source[train_indices], y[train_indices])
                prediction = predict(head, source[val_indices], spec)
                fold_scores.append(balanced_accuracy_score(y[val_indices], prediction))
            spec_scores.append(float(np.mean(fold_scores)))

        best_index = int(np.argmax(spec_scores))
        best = specs[best_index]
        source = X_subject_z if best["subject_standardize"] else X
        head = estimator(best).fit(source[outer_train], y[outer_train])
        prediction = predict(head, source[outer_test], best)
        row = {
            "subject": subject,
            "selected_head": best["name"],
            "inner_balanced_accuracy": spec_scores[best_index],
            **metrics(y[outer_test], prediction),
        }
        rows.append(row)
        print(f"[{index:02d}/{len(subjects):02d}] subject {subject:02d}: "
              f"{row['accuracy'] * 100:6.2f}%  bal={row['balanced_accuracy'] * 100:6.2f}%  "
              f"{best['name']}")

    payload = {
        "format": "mirepnet-nested-loso-feature-head-v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_seconds": float(time.perf_counter() - started),
        "checkpoint": str(args.checkpoint.resolve()),
        "transformer_weights_updated": False,
        "target_labels_used_in_training_or_model_selection": False,
        "target_signal_used_for_unlabeled_alignment": True,
        "subjects": rows,
        "aggregate": {
            "n_subjects": len(rows),
            "n_trials": int(sum(row["n_trials"] for row in rows)),
            "macro_accuracy": float(np.mean([row["accuracy"] for row in rows])),
            "macro_balanced_accuracy": float(np.mean([row["balanced_accuracy"] for row in rows])),
            "micro_accuracy": float(
                sum(row["accuracy"] * row["n_trials"] for row in rows)
                / sum(row["n_trials"] for row in rows)
            ),
            "subjects_at_or_above_80pct": int(sum(row["accuracy"] >= 0.8 for row in rows)),
            "subjects_at_or_above_90pct": int(sum(row["accuracy"] >= 0.9 for row in rows)),
        },
    }
    output = args.output or (
        ROOT / "results" / "mirepnet" / f"nested-loso-features__{args.checkpoint.stem}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nmacro accuracy: {payload['aggregate']['macro_accuracy'] * 100:.2f}%")
    print(f">=80%: {payload['aggregate']['subjects_at_or_above_80pct']}/{len(rows)}")
    print(f">=90%: {payload['aggregate']['subjects_at_or_above_90pct']}/{len(rows)}")
    print(f"result: {output.resolve()}")
    print(f"total runtime: {payload['run_seconds'] / 60:.2f} minutes")


if __name__ == "__main__":
    main()
