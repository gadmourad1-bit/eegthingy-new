"""External within-subject architecture comparison on Cho2017 (GigaDB 100295).

Same architectures and the same 15-channel/125 Hz matched pipeline as the local
benchmark, but on a fully independent 52-subject cohort.  Cho2017 is single-session,
so each subject is evaluated by stratified k-fold cross-validation; within each
training fold a stratified subset is held out for early stopping.  If GeoAdaptNet's
ranking against the leading conv nets survives here, the local result is not an
artefact of the local recordings.

Run:  python -m benchmark.research.external_benchmark --subjects 1-20 --archs geoadapt,eegnet,shallow,deep,conformer,atcnet \
        --folds 5 --seeds 7 --device cuda --output src/benchmark/research/results/dnn_compare_cho2017.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .config import CHANNELS, PROJECT_ROOT, SFREQ
from .dnn_baselines import ARCHS, TorchEEGClassifier
from .engine import CovarianceDataset, TrainConfig, predict_proba, set_reproducible_seed, train_model
from .external_cho2017 import load_cho_subject
from .model import GeoAdaptNet

NEURAL_ARCHS = ("geoadapt", *ARCHS.keys())


def _stratified_val(y: np.ndarray, seed: int, frac: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    val = np.zeros(len(y), dtype=bool)
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        val[idx[: max(1, int(round(frac * len(idx))))]] = True
    return np.flatnonzero(~val), np.flatnonzero(val)


def _fit_predict(
    arch: str,
    cov: np.ndarray,
    broad: np.ndarray,
    y: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    *,
    seed: int,
    device: str,
    augment: bool,
) -> tuple[np.ndarray, int]:
    if arch == "riemann":
        # Non-transductive classical reference (no .calibrate on the test set),
        # fair vs the conv nets.  The decisive L0 fork: can a fixed-band geometric
        # method beat ShallowConvNet at Cho2017's data scale?
        from .baselines import RiemannianTangentLogistic

        estimator = RiemannianTangentLogistic().fit(cov[train], y[train])
        return estimator.predict_proba(cov[test]), 0
    if arch == "geoadapt_anchor":
        # GeoAdaptNet's log-Euclidean tangent features + a convex L2 head (the
        # measured local head-fix): no SGD, no residual, non-transductive.
        from .tangent_anchor import TangentAnchorClassifier

        estimator = TangentAnchorClassifier().fit(cov[train], y[train])
        return estimator.predict_proba(cov[test]), estimator.param_count_
    inner_train, inner_val = _stratified_val(y[train], seed)
    tr, va = train[inner_train], train[inner_val]
    if arch == "geoadapt":
        set_reproducible_seed(seed)
        model = GeoAdaptNet(n_bands=cov.shape[1], auxiliary_intent=False)
        config = TrainConfig(
            epochs=180,
            patience=25,
            seed=seed,
            device=device,
            lr_swap_prob=0.5 if augment else 0.0,
            select_metric="blend" if augment else "loss",
        )
        result = train_model(
            model,
            CovarianceDataset(cov[tr], y[tr]),
            CovarianceDataset(cov[va], y[va]),
            config,
        )
        probs, _ = predict_proba(
            result.model, CovarianceDataset(cov[test], y[test]), device=device, branch="full"
        )
        return probs, model.parameter_count
    if arch in ("geoadapt_fb", "geoadapt_fbsp"):
        from ..shared.augment import left_right_swap_index
        from .filterbank_net import FilterBankSPDClassifier

        clf = FilterBankSPDClassifier(
            n_bands=cov.shape[1],
            sfreq=SFREQ,
            seed=seed,
            device=device,
            reduced_dim=8 if arch == "geoadapt_fbsp" else None,
            lr_swap_index=left_right_swap_index(CHANNELS) if augment else None,
            lr_swap_prob=0.5 if augment else 0.0,
        )
        clf.fit(broad[tr], y[tr], broad[va], y[va])
        return clf.predict_proba(broad[test]), clf.param_count_
    clf = TorchEEGClassifier(
        arch,
        n_times=broad.shape[-1],
        sfreq=SFREQ,
        seed=seed,
        device=device,
        lr_swap_prob=0.5 if augment else 0.0,
        channels=CHANNELS if augment else None,
    )
    clf.fit(broad[tr], y[tr], broad[va], y[va])
    return clf.predict_proba(broad[test]), clf.param_count_


def run(
    archs: Sequence[str],
    subjects: Sequence[int],
    seeds: Sequence[int],
    *,
    folds: int,
    device: str,
    augment: bool,
    bands: str | None = None,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for subject in subjects:
        try:
            data = load_cho_subject(subject, bands=bands)
        except Exception as error:  # a few Cho2017 subjects are known to be unusable
            print(f"  SKIP subject {subject}: {type(error).__name__}: {error}", flush=True)
            skipped.append({"subject": str(subject), "error": f"{type(error).__name__}: {error}"})
            continue
        cov, broad, y = data["covariances"], data["broadband"], data["labels"]
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
        splits = list(splitter.split(np.zeros(len(y)), y))
        for arch in archs:
            for seed in seeds:
                started = time.time()
                truths, probabilities, params = [], [], 0
                for train, test in splits:
                    probs, params = _fit_predict(
                        arch, cov, broad, y, train, test, seed=seed, device=device, augment=augment
                    )
                    truths.append(y[test])
                    probabilities.append(probs)
                truth = np.concatenate(truths)
                prob = np.concatenate(probabilities)
                bacc = balanced_accuracy_score(truth, prob.argmax(axis=1))
                auc = roc_auc_score(truth, prob[:, 1]) if len(set(truth.tolist())) > 1 else float("nan")
                records.append(
                    {
                        "arch": arch,
                        "subject": int(subject),
                        "seed": int(seed),
                        "balanced_accuracy": float(bacc),
                        "roc_auc": float(auc),
                        "params": int(params),
                        "seconds": float(time.time() - started),
                    }
                )
                print(f"  {arch:12s} S{subject:<2d} seed={seed} bacc={bacc*100:.1f} "
                      f"auc={auc*100:.1f} ({time.time()-started:.1f}s)", flush=True)
    return {
        "folds_records": records,
        "archs": list(archs),
        "subjects": list(subjects),
        "scored_subjects": sorted({r["subject"] for r in records}),
        "skipped_subjects": skipped,
        "seeds": list(seeds),
        "n_folds": folds,
        "augment": bool(augment),
        "dataset": "Cho2017",
    }


def summarize(payload: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for arch in payload["archs"]:
        rows = [r for r in payload["folds_records"] if r["arch"] == arch]
        by_subject: dict[int, list[float]] = {}
        for r in rows:
            by_subject.setdefault(r["subject"], []).append(r["balanced_accuracy"])
        per_subject = np.array([np.mean(v) for v in by_subject.values()])
        summary[arch] = {
            "balanced_accuracy_mean": float(per_subject.mean()),
            "balanced_accuracy_std": float(per_subject.std()),
            "params": int(np.median([r["params"] for r in rows])),
            "n_participants": len(by_subject),
        }
    return summary


def _parse_subjects(value: str) -> list[int]:
    out: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if "-" in token:
            lo, hi = token.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        elif token:
            out.append(int(token))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archs", default=",".join(NEURAL_ARCHS))
    parser.add_argument("--subjects", default="1-20")
    parser.add_argument("--seeds", default="7")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--bands", default="default", help="filter-bank preset: default|rich9|rich7")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "output" / "research" / "dnn_compare_cho2017.json",
    )
    args = parser.parse_args(argv)

    archs = [a.strip() for a in args.archs.split(",") if a.strip()]
    subjects = _parse_subjects(args.subjects)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    payload = run(
        archs, subjects, seeds, folds=args.folds, device=args.device,
        augment=args.augment, bands=args.bands,
    )
    payload["bands"] = args.bands
    payload["summary"] = summarize(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
