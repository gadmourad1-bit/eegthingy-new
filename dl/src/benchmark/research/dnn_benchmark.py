"""Architecture comparison: GeoAdaptNet vs leading EEG neural networks.

Uniform, leakage-controlled chronological protocol on the local cohort: for each
participant, recordings 1--2 train, recording 3 selects the checkpoint (early
stopping / epoch count), and the untouched recording 4 is scored.  Every model is
compared on the same task windows by raw balanced accuracy, so the comparison
isolates the decoder architecture -- no online recentering and no target labels.
Convolutional nets consume broadband epochs; GeoAdaptNet consumes filter-bank
covariances; classical baselines use their native features.  Neural models are run
over several seeds and aggregated within participant first.

Run:  python -m benchmark.research.dnn_benchmark --archs geoadapt,eegnet,shallow,deep,conformer,atcnet \
        --seeds 7,17,27 --device cuda --output src/benchmark/research/results/dnn_compare_local.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from .config import CHANNELS, DEFAULT_DATA_CONFIG, PROJECT_ROOT, SUBJECT_RUNS, VALID_SUBJECTS
from .data import SessionData, SessionKey, load_sessions
from .dnn_baselines import ARCHS, TorchEEGClassifier
from .engine import CovarianceDataset, TrainConfig, predict_proba, set_reproducible_seed, train_model
from .model import GeoAdaptNet

NEURAL_ARCHS = ("geoadapt", *ARCHS.keys())
CLASSICAL_ARCHS = ("fbcsp", "riemann", "geoadapt_anchor")
ALL_ARCHS = (*NEURAL_ARCHS, *CLASSICAL_ARCHS)


def _subject_split(data: SessionData, subject: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chronological task-window rows: (train rec1-2, val rec3, test rec4)."""

    runs = sorted(np.unique(data.run_ids[data.subject_ids == subject]).tolist())
    if runs != list(SUBJECT_RUNS[subject]):
        raise ValueError(f"subject {subject} has runs {runs}, expected {SUBJECT_RUNS[subject]}")
    task = data.labels >= 0

    def rows(selected_runs: Sequence[int]) -> np.ndarray:
        return np.flatnonzero(
            (data.subject_ids == subject) & np.isin(data.run_ids, selected_runs) & task
        )

    return rows(runs[:2]), rows([runs[2]]), rows([runs[3]])


def _geoadapt_probs(
    data: SessionData,
    train_rows: np.ndarray,
    val_rows: np.ndarray,
    test_rows: np.ndarray,
    *,
    seed: int,
    device: str,
    augment: bool,
) -> tuple[np.ndarray, int, float]:
    """Train GeoAdaptNet on covariances and return raw test probabilities."""

    set_reproducible_seed(seed)
    model = GeoAdaptNet(auxiliary_intent=False)
    config = TrainConfig(
        epochs=180,
        patience=25,
        seed=seed,
        device=device,
        lr_swap_prob=0.5 if augment else 0.0,
        select_metric="blend" if augment else "loss",
    )
    started = time.time()
    result = train_model(
        model,
        CovarianceDataset(data.covariances[train_rows], data.labels[train_rows]),
        CovarianceDataset(data.covariances[val_rows], data.labels[val_rows]),
        config,
    )
    fit_seconds = time.time() - started
    probs, _ = predict_proba(
        result.model,
        CovarianceDataset(data.covariances[test_rows], data.labels[test_rows]),
        device=device,
        branch="full",
    )
    return probs, model.parameter_count, fit_seconds


def _conv_probs(
    arch: str,
    data: SessionData,
    train_rows: np.ndarray,
    val_rows: np.ndarray,
    test_rows: np.ndarray,
    *,
    seed: int,
    device: str,
    augment: bool,
) -> tuple[np.ndarray, int, float]:
    """Train a braindecode conv net on broadband epochs and return test probs."""

    epochs = data.broadband_epochs
    clf = TorchEEGClassifier(
        arch,
        n_times=epochs.shape[-1],
        sfreq=DEFAULT_DATA_CONFIG.sfreq,
        seed=seed,
        device=device,
        lr_swap_prob=0.5 if augment else 0.0,
        channels=CHANNELS if augment else None,
    )
    clf.fit(
        epochs[train_rows], data.labels[train_rows], epochs[val_rows], data.labels[val_rows]
    )
    probs = clf.predict_proba(epochs[test_rows])
    return probs, clf.param_count_, clf.fit_seconds_


def _classical_probs(
    arch: str,
    data: SessionData,
    train_rows: np.ndarray,
    val_rows: np.ndarray,
    test_rows: np.ndarray,
) -> tuple[np.ndarray, int, float]:
    """Fit a classical baseline on train+val and return test probabilities."""

    from .baselines import EAFilterBankCSP, RiemannianTangentLogistic

    fit_rows = np.concatenate([train_rows, val_rows])
    started = time.time()
    if arch == "fbcsp":
        estimator = EAFilterBankCSP().fit(data.epochs[fit_rows], data.labels[fit_rows])
        estimator.calibrate(data.epochs[test_rows])
        probs = estimator.predict_proba(data.epochs[test_rows])
    elif arch == "riemann":
        estimator = RiemannianTangentLogistic().fit(data.covariances[fit_rows], data.labels[fit_rows])
        estimator.calibrate(data.covariances[test_rows])
        probs = estimator.predict_proba(data.covariances[test_rows])
    else:  # pragma: no cover
        raise ValueError(arch)
    return probs, 0, time.time() - started


def _score(
    arch: str, data: SessionData, subject: int, seed: int, device: str, augment: bool
) -> dict[str, Any]:
    train_rows, val_rows, test_rows = _subject_split(data, subject)
    if arch == "geoadapt":
        probs, params, secs = _geoadapt_probs(
            data, train_rows, val_rows, test_rows, seed=seed, device=device, augment=augment
        )
    elif arch == "geoadapt_anchor":
        from .tangent_anchor import TangentAnchorClassifier

        started = time.time()
        est = TangentAnchorClassifier().fit(data.covariances[train_rows], data.labels[train_rows])
        probs = est.predict_proba(data.covariances[test_rows])
        params, secs = est.param_count_, time.time() - started
    elif arch in ARCHS:
        probs, params, secs = _conv_probs(
            arch, data, train_rows, val_rows, test_rows, seed=seed, device=device, augment=augment
        )
    else:
        probs, params, secs = _classical_probs(arch, data, train_rows, val_rows, test_rows)
    y = data.labels[test_rows]
    pred = probs.argmax(axis=1)
    return {
        "subject": int(subject),
        "seed": int(seed),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, probs[:, 1])) if len(set(y.tolist())) > 1 else float("nan"),
        "params": int(params),
        "fit_seconds": float(secs),
        "n_test": int(len(y)),
    }


def run_comparison(
    archs: Sequence[str],
    subjects: Sequence[int],
    seeds: Sequence[int],
    *,
    device: str,
    augment: bool = False,
) -> dict[str, Any]:
    data = load_sessions([SessionKey(s, r) for s in subjects for r in SUBJECT_RUNS[s]])
    folds: list[dict[str, Any]] = []
    for arch in archs:
        arch_seeds = seeds if arch in NEURAL_ARCHS else seeds[:1]  # classical is deterministic
        for subject in subjects:
            for seed in arch_seeds:
                record = _score(arch, data, subject, seed, device, augment)
                record["arch"] = arch
                folds.append(record)
                print(f"  {arch:14s} S{subject:<2d} seed={seed} "
                      f"bacc={record['balanced_accuracy']*100:.1f} "
                      f"({record['params']} params, {record['fit_seconds']:.1f}s)", flush=True)
    return {
        "folds": folds,
        "archs": list(archs),
        "subjects": list(subjects),
        "seeds": list(seeds),
        "augment": bool(augment),
    }


def summarize(payload: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for arch in payload["archs"]:
        rows = [f for f in payload["folds"] if f["arch"] == arch]
        by_subject: dict[int, list[float]] = {}
        for f in rows:
            by_subject.setdefault(f["subject"], []).append(f["balanced_accuracy"])
        per_subject = np.array([np.mean(v) for v in by_subject.values()])
        summary[arch] = {
            "balanced_accuracy_mean": float(per_subject.mean()),
            "balanced_accuracy_std": float(per_subject.std()),
            "params": int(np.median([f["params"] for f in rows])),
            "fit_seconds_mean": float(np.mean([f["fit_seconds"] for f in rows])),
            "n_participants": len(by_subject),
        }
    return summary


def _csv(value: str, valid: Sequence[int] | None = None) -> list[int]:
    if value.strip().lower() == "all" and valid is not None:
        return list(valid)
    return [int(v) for v in value.split(",") if v.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archs", default=",".join(ALL_ARCHS))
    parser.add_argument("--subjects", default="all")
    parser.add_argument("--seeds", default="7,17,27")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--augment", action="store_true", help="left/right swap augmentation for all neural nets"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "output" / "research" / "dnn_compare_local.json",
    )
    args = parser.parse_args(argv)

    archs = [a.strip() for a in args.archs.split(",") if a.strip()]
    unknown = set(archs) - set(ALL_ARCHS)
    if unknown:
        raise SystemExit(f"unknown archs {sorted(unknown)}; choose from {ALL_ARCHS}")
    subjects = _csv(args.subjects, VALID_SUBJECTS)
    seeds = _csv(args.seeds)

    payload = run_comparison(archs, subjects, seeds, device=args.device, augment=args.augment)
    payload["summary"] = summarize(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
