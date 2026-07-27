"""Fusion of complementary decoders -- the honest route to topping the table.

No single architecture dominates: the geometric decoders have the lowest cross-subject
variance (they hold up on hard participants), ShallowConvNet has the highest mean (it
wins on easy ones), and CSP log-variance vs Riemannian tangent features encode partly
different structure.  Those are complementary *per subject*, so combining them can beat
every individual model even though each is near its own ceiling.

Two combination schemes are evaluated on the same splits as the individual models:

* **feature fusion** -- concatenate per-band CSP log-variance features with frozen-reference
  log-Euclidean tangent features and fit one standardized convex L2 logistic head;
* **probability ensemble** -- average the test probabilities of the member decoders.

Everything is fit on training data only (non-transductive) and scored on the untouched
test recording, so the comparison is fair against the single models.
"""

from __future__ import annotations

import argparse
import json
import warnings
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

from .config import PROJECT_ROOT, SFREQ, SUBJECT_RUNS, VALID_SUBJECTS
from .data import SessionKey, load_sessions
from .dnn_baselines import TorchEEGClassifier
from .tangent_anchor import TangentAnchorClassifier, _tangent

MEMBERS = ("tangent", "fbcsp", "riemann", "shallow")


def _subject_split(data, subject):
    runs = sorted(np.unique(data.run_ids[data.subject_ids == subject]).tolist())
    task = data.labels >= 0

    def rows(sel):
        return np.flatnonzero((data.subject_ids == subject) & np.isin(data.run_ids, sel) & task)

    return rows(runs[:2]), rows([runs[2]]), rows([runs[3]])


def _member_probs(name, data, tr, va, te, seed, device):
    """Test-set probabilities for one member decoder (train-fit only)."""
    from .baselines import EAFilterBankCSP, RiemannianTangentLogistic

    fit = np.concatenate([tr, va])
    if name == "tangent":
        est = TangentAnchorClassifier().fit(data.covariances[fit], data.labels[fit])
        return est.predict_proba(data.covariances[te])
    if name == "riemann":
        est = RiemannianTangentLogistic().fit(data.covariances[fit], data.labels[fit])
        return est.predict_proba(data.covariances[te])
    if name == "fbcsp":
        est = EAFilterBankCSP().fit(data.epochs[fit], data.labels[fit])
        return est.predict_proba(data.epochs[te])
    if name == "shallow":
        clf = TorchEEGClassifier("shallow", n_times=data.broadband_epochs.shape[-1],
                                 sfreq=SFREQ, seed=seed, device=device)
        clf.fit(data.broadband_epochs[tr], data.labels[tr],
                data.broadband_epochs[va], data.labels[va])
        return clf.predict_proba(data.broadband_epochs[te])
    raise ValueError(name)


def _feature_fusion_probs(data, tr, va, te):
    """CSP log-variance features concatenated with log-Euclidean tangent features."""
    from .baselines import EAFilterBankCSP
    import torch
    from .spd import matrix_log

    fit = np.concatenate([tr, va])
    csp = EAFilterBankCSP().fit(data.epochs[fit], data.labels[fit])
    csp_fit = csp._features(data.epochs[fit])
    csp_te = csp._features(data.epochs[te])

    cov_fit = torch.as_tensor(data.covariances[fit], dtype=torch.float64)
    log_ref = matrix_log(cov_fit).mean(dim=0)
    tan_fit = _tangent(data.covariances[fit], log_ref)
    tan_te = _tangent(data.covariances[te], log_ref)

    x_fit = np.hstack([csp_fit, tan_fit])
    x_te = np.hstack([csp_te, tan_te])
    scaler = StandardScaler().fit(x_fit)
    model = LogisticRegression(C=1.0, max_iter=3000).fit(scaler.transform(x_fit), data.labels[fit])
    return model.predict_proba(scaler.transform(x_te))


def run(subjects: Sequence[int], seed: int, device: str) -> dict[str, Any]:
    data = load_sessions([SessionKey(s, r) for s in subjects for r in SUBJECT_RUNS[s]])
    per_subject: dict[str, dict[int, float]] = {}
    for subject in subjects:
        tr, va, te = _subject_split(data, subject)
        y = data.labels[te]
        probs = {m: _member_probs(m, data, tr, va, te, seed, device) for m in MEMBERS}
        probs["fusion_feat"] = _feature_fusion_probs(data, tr, va, te)

        scores = {k: balanced_accuracy_score(y, p.argmax(1)) for k, p in probs.items()}
        # probability ensembles over every member subset of size >= 2
        for r in (2, 3, 4):
            for combo in combinations(MEMBERS, r):
                mean = np.mean([probs[m] for m in combo], axis=0)
                scores["ens:" + "+".join(combo)] = balanced_accuracy_score(y, mean.argmax(1))
        # the full stack including the fused features
        allp = np.mean([probs[m] for m in MEMBERS] + [probs["fusion_feat"]], axis=0)
        scores["ens:ALL+fusion"] = balanced_accuracy_score(y, allp.argmax(1))

        for k, v in scores.items():
            per_subject.setdefault(k, {})[subject] = float(v)
        best = max(scores.items(), key=lambda kv: kv[1])
        print(f"  S{subject:<2d} best={best[0]} {best[1]*100:.1f} | "
              f"tangent {scores['tangent']*100:.1f} fbcsp {scores['fbcsp']*100:.1f} "
              f"shallow {scores['shallow']*100:.1f}", flush=True)

    summary = {
        k: {"mean": float(np.mean(list(v.values())) * 100),
            "std": float(np.std(list(v.values())) * 100)}
        for k, v in per_subject.items()
    }
    return {"per_subject": per_subject, "summary": summary, "seed": seed,
            "subjects": list(subjects)}


EXT_MEMBERS = ("tangent", "riemann", "shallow", "eegnet")


def _ext_member_probs(name, cov, broad, y, tr, va, te, seed, device):
    from .baselines import RiemannianTangentLogistic

    fit = np.concatenate([tr, va])
    if name == "tangent":
        return TangentAnchorClassifier().fit(cov[fit], y[fit]).predict_proba(cov[te])
    if name == "riemann":
        return RiemannianTangentLogistic().fit(cov[fit], y[fit]).predict_proba(cov[te])
    clf = TorchEEGClassifier(name, n_times=broad.shape[-1], sfreq=SFREQ, seed=seed, device=device)
    clf.fit(broad[tr], y[tr], broad[va], y[va])
    return clf.predict_proba(broad[te])


def run_external(subjects: Sequence[int], seed: int, device: str, folds: int = 5) -> dict[str, Any]:
    """Same fusion question on the independent Cho2017 cohort (within-subject CV)."""
    from sklearn.model_selection import StratifiedKFold

    from .external_benchmark import _stratified_val
    from .external_cho2017 import load_cho_subject

    per_subject: dict[str, dict[int, float]] = {}
    for subject in subjects:
        try:
            d = load_cho_subject(subject)
        except Exception as e:
            print(f"  SKIP s{subject}: {e}", flush=True)
            continue
        cov, broad, y = d["covariances"], d["broadband"], d["labels"]
        splits = list(StratifiedKFold(n_splits=folds, shuffle=True, random_state=0).split(np.zeros(len(y)), y))
        truth, acc = [], {m: [] for m in EXT_MEMBERS}
        for train, te in splits:
            it, iv = _stratified_val(y[train], seed)
            tr, va = train[it], train[iv]
            for m in EXT_MEMBERS:
                acc[m].append(_ext_member_probs(m, cov, broad, y, tr, va, te, seed, device))
            truth.append(y[te])
        yy = np.concatenate(truth)
        probs = {m: np.concatenate(acc[m]) for m in EXT_MEMBERS}
        scores = {m: balanced_accuracy_score(yy, p.argmax(1)) for m, p in probs.items()}
        for r in (2, 3, 4):
            for combo in combinations(EXT_MEMBERS, r):
                mean = np.mean([probs[m] for m in combo], axis=0)
                scores["ens:" + "+".join(combo)] = balanced_accuracy_score(yy, mean.argmax(1))
        for k, v in scores.items():
            per_subject.setdefault(k, {})[subject] = float(v)
        best = max(scores.items(), key=lambda kv: kv[1])
        print(f"  s{subject:<2d} best={best[0]} {best[1]*100:.1f}", flush=True)

    summary = {k: {"mean": float(np.mean(list(v.values())) * 100),
                   "std": float(np.std(list(v.values())) * 100)}
               for k, v in per_subject.items()}
    return {"per_subject": per_subject, "summary": summary, "seed": seed,
            "subjects": list(subjects), "dataset": "Cho2017"}


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", default="all")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="cuda")
    p.add_argument("--external", action="store_true", help="run on the Cho2017 cohort instead")
    p.add_argument("--output", type=Path,
                   default=PROJECT_ROOT / "deepnet" / "results" / "fusion_local.json")
    a = p.parse_args(argv)
    if a.external:
        lo_hi = a.subjects.split("-")
        subjects = (list(range(int(lo_hi[0]), int(lo_hi[1]) + 1)) if len(lo_hi) == 2
                    else [int(s) for s in a.subjects.split(",")])
        payload = run_external(subjects, a.seed, a.device)
    else:
        subjects = list(VALID_SUBJECTS) if a.subjects == "all" else [int(s) for s in a.subjects.split(",")]
        payload = run(subjects, a.seed, a.device)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(payload, indent=2))
    print("\n=== ranked (balanced accuracy %) ===")
    for k, v in sorted(payload["summary"].items(), key=lambda kv: -kv[1]["mean"])[:14]:
        print("  %-28s %6.2f +/- %5.2f" % (k, v["mean"], v["std"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
