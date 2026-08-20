"""Deployment-pipeline classical benchmark on the formal Local Exp4 cohort.

Replicates the deployed online pipeline (classifier/run.py) offline:
  - EA + FB-CSP(2) + shrinkage LDA
  - Riemannian LWF covariance + per-group recentering + tangent space + LR
Protocols:
  - personalized: within-subject stratified 5-fold CV over the subject's 4 runs
    (fit on train folds with per-run groups, unsupervised set_reference on the
    validation fold — exactly classifier/run.py run_offline's CV loop)
  - loso: leave-one-subject-out; fit on the other 7 subjects (per-run groups),
    unsupervised set_reference on the held-out subject's data, score all epochs
Formal cohort per dl/src/benchmark/local_outer_refit_benchmark.py: subjects
1,3,4,5,6,7,8,10 (S10 uses runs 5-8).
"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

ROOT = "/Users/admin/Documents/GitHub/eegthingy"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "classifier"))

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from run import process_data, make_decoder
from config import TARGET_MAPPINGS

SUBJECT_RUNS = {1: (1, 2, 3, 4), 3: (1, 2, 3, 4), 4: (1, 2, 3, 4), 5: (1, 2, 3, 4),
                6: (1, 2, 3, 4), 7: (1, 2, 3, 4), 8: (1, 2, 3, 4), 10: (5, 6, 7, 8)}
DECODERS = ("ea", "riemann")

def load_subject(s):
    parts = []
    for r in SUBJECT_RUNS[s]:
        f = f"{ROOT}/data/exp4_subject{s}_training_{r}_mi_raw.fif"
        parts.append(process_data(f, TARGET_MAPPINGS))
    X = np.concatenate([X for X, _ in parts])
    y = np.concatenate([y for _, y in parts])
    g = np.concatenate([[i] * len(yy) for i, (_, yy) in enumerate(parts)])
    return X, y, g

print("loading all subjects…", flush=True)
data = {s: load_subject(s) for s in SUBJECT_RUNS}
for s, (X, y, g) in data.items():
    print(f"  S{s}: {X.shape} labels {np.bincount(y)[1:]}", flush=True)

out = {"personalized": {}, "loso": {}}

for kind in DECODERS:
    out["personalized"][kind] = {}
    for s, (X, y, g) in data.items():
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        accs, baccs = [], []
        for tr, va in cv.split(X, y):
            m = make_decoder(kind).fit(X[tr], y[tr], groups=g[tr])
            m.set_reference(X[va])
            yp = m.predict(X[va])
            accs.append(accuracy_score(y[va], yp))
            baccs.append(balanced_accuracy_score(y[va], yp))
        out["personalized"][kind][s] = {"acc": float(np.mean(accs)), "bacc": float(np.mean(baccs))}
        print(f"personalized {kind} S{s}: acc {np.mean(accs)*100:.2f}%", flush=True)

for kind in DECODERS:
    out["loso"][kind] = {}
    for held in SUBJECT_RUNS:
        others = [x for x in SUBJECT_RUNS if x != held]
        Xtr = np.concatenate([data[s][0] for s in others])
        ytr = np.concatenate([data[s][1] for s in others])
        gtr = np.concatenate([data[s][2] + 10 * i for i, s in enumerate(others)])
        Xte, yte, _ = data[held]
        m = make_decoder(kind).fit(Xtr, ytr, groups=gtr)
        m.set_reference(Xte)
        yp = m.predict(Xte)
        out["loso"][kind][held] = {"acc": float(accuracy_score(yte, yp)),
                                   "bacc": float(balanced_accuracy_score(yte, yp))}
        print(f"loso {kind} S{held}: acc {accuracy_score(yte, yp)*100:.2f}%", flush=True)

def summarize(d):
    vals = [v["acc"] * 100 for v in d.values()]
    return float(np.mean(vals)), float(np.std(vals))

print("\n==== SUMMARY (accuracy %, mean ± SD across 8 subjects) ====")
summary = {}
for proto in ("personalized", "loso"):
    for kind in DECODERS:
        m, sd = summarize(out[proto][kind])
        summary[f"{proto}_{kind}"] = {"mean": m, "sd": sd}
        print(f"{proto:13s} {kind:8s}: {m:.1f} ± {sd:.1f}")

dest = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "generated", "deployment_classical.json")
json.dump({"per_subject": out, "summary": summary}, open(dest, "w"), indent=1)
print(f"saved {dest}")
