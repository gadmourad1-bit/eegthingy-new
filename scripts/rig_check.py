"""Pre-flight rig check: is the recording chain capable of capturing MI today?

Point it at two (or more) fresh training .fif files from a KNOWN-GOOD pilot
(someone from the S1-S8 era). It prints class separability with the epoch-offset
sweep and the per-channel discriminability map. Gate every session day on this:

    PASS  separability >= 75% peaked at offset 0, C3/C4 in the top channels
    FAIL  flat sweep / no channel above ~0.6 AUC -> fix the rig before subjects

Usage:  python scripts/rig_check.py data/exp4_subjectX_training_1_mi_raw.fif \
                                    data/exp4_subjectX_training_2_mi_raw.fif
"""
import sys
import warnings

warnings.filterwarnings("ignore")
import mne
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
except ImportError:
    sys.exit("needs pyriemann")

mne.set_log_level("ERROR")
CH = ["Cz","Pz","C3","C4","T5","T6","Fz","F7","F8","F3","F4","T3","T4","P3","P4"]


def load(f, offset=0.0):
    raw = mne.io.read_raw_fif(f, preload=True)
    raw.pick([c for c in CH if c in raw.ch_names])
    raw.filter(8, 30, method="fir", phase="minimum", fir_design="firwin", verbose=False)
    desc = np.array([str(d).strip().lower() for d in raw.annotations.description])
    raw.annotations.description = desc
    ev_id = {d: (1 if d.startswith("left_hand/task") else 2)
             for d in set(desc) if d.startswith(("left_hand/task", "right_hand/task"))}
    events, _ = mne.events_from_annotations(raw, event_id=ev_id, verbose=False)
    ep = mne.Epochs(raw, events, tmin=offset, tmax=offset + 2.0, baseline=None,
                    preload=True, verbose=False)
    return ep.get_data(copy=True), ep.events[:, -1] - 1


def separability(files, offset=0.0):
    X = []; y = []
    for f in files:
        a, b = load(f, offset); X.append(a); y.append(b)
    X = np.concatenate(X); y = np.concatenate(y)
    if len(set(y.tolist())) < 2 or len(y) < 20:
        return np.nan, X, y
    pipe = make_pipeline(Covariances("oas"), TangentSpace(metric="riemann"),
                         StandardScaler(), LogisticRegression(max_iter=1500))
    return float(np.mean(cross_val_score(pipe, X, y, cv=4)) * 100), X, y


def stream_integrity(files):
    """Digital-integrity gate: frozen samples and missing data (board-level faults)."""
    worst_stale, worst_missing = 0.0, 0.0
    for f in files:
        raw = mne.io.read_raw_fif(f, preload=True)
        raw.pick([c for c in CH if c in raw.ch_names])
        X = raw.get_data()
        stale = max(float((np.diff(x) == 0).mean()) for x in X)
        expected = 389.0  # 10 s prep + 60 cues x 6.33 s
        missing = max(0.0, 1 - (X.shape[1] / raw.info["sfreq"]) / expected)
        worst_stale = max(worst_stale, stale)
        worst_missing = max(worst_missing, missing)
    return worst_stale * 100, worst_missing * 100


def main(files):
    print("rig check on:", ", ".join(files))
    stale, missing = stream_integrity(files)
    print(f"\nstream integrity: worst frozen-sample rate {stale:.1f}%, missing data {missing:.0f}%")
    if stale > 5 or missing > 5:
        print("  ✗ HARD FAIL — the board is freezing or dropping samples (S6/S9/S18/S20")
        print("    signature). Fix the electronics before anything else: batteries, Daisy")
        print("    reseat, scripts/check_board.py. Signal checks below are unreliable.")
    print("\noffset sweep (separability %, should PEAK at 0):")
    accs = {}
    for off in (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0):
        acc, X, y = separability(files, off)
        accs[off] = acc
        bar = "#" * int(max(acc - 50, 0) / 2) if np.isfinite(acc) else ""
        print(f"  {off:+4.0f}s  {acc:5.1f}  {bar}")

    _, X, y = separability(files, 0.0)
    power = np.log(np.var(X, axis=2) + 1e-20)
    aucs = [max(a, 1 - a) for a in
            (roc_auc_score(y, power[:, c]) for c in range(power.shape[1]))]
    order = np.argsort(aucs)[::-1]
    print("\nper-channel discriminability (AUC):")
    print("  " + "  ".join(f"{CH[i]}:{aucs[i]:.2f}" for i in order[:6]))

    peak = accs[0.0]
    peaked = np.isfinite(peak) and peak >= max(v for k, v in accs.items() if k != 0.0) - 3
    c3c4 = max(aucs[2], aucs[3])
    print("\nVERDICT:")
    if peak >= 75 and peaked and c3c4 >= 0.60:
        print(f"  ✓ PASS — separability {peak:.0f}% at cue, C3/C4 AUC {c3c4:.2f}. Rig can see MI.")
    else:
        print(f"  ✗ FAIL — separability {peak:.0f}% (need >=75 at offset 0), C3/C4 AUC {c3c4:.2f}.")
        print("    Do not run subjects. Check, in order: reference/ear-clip electrode contact,")
        print("    fresh gel/paste everywhere, cap placement (C3/C4 over the hand knob),")
        print("    then scripts/check_board.py for the electronics.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1:])
