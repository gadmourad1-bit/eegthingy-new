"""Train EA + FB-CSP (2 comp) on the .fif data and export a portable model.npz
that the lean SBC runtime consumes with numpy/scipy only (no mne/sklearn).

Run on the Mac (needs the full pipeline):
    PYTHONPATH=.:classifier python sbc/export_model.py            # pick files interactively
    PYTHONPATH=.:classifier python sbc/export_model.py data/exp4_subject1_*.fif   # or pass globs

The exported arrays (per band): minimum-phase FIR taps, CSP spatial filters, and
the shrinkage-LDA weights — verified bit-for-bit identical to EAFilterBankCSP.
The EA whitener is NOT exported: it is recomputed live on the SBC from the
calibration block, exactly like set_reference().
"""
import os
import sys
import glob

import numpy as np
import mne

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "classifier")))
mne.set_log_level("ERROR")

import run as R
from config import (FB_BANDS, FB_TRANS, EEG_CHANNELS_TARGETS, EEG_CHANNELS_MAPPING,
                    STRIDE_S, FILTER_WARMUP_S, CALIBRATION_SECONDS, TARGET_MAPPINGS)

OUT = os.path.join(os.path.dirname(__file__), "model.npz")


def main():
    files = sys.argv[1:]
    if not files:
        all_files = R.discover_files()
        if not all_files:
            print(f"no .fif files found in {R.DATA_DIR}"); return
        print("\nAvailable files:")
        for i, f in enumerate(all_files, 1):
            print(f"  {i}) {os.path.basename(f)}")
        print()
        files = R.select_files("training files (e.g. 1,2,3): ", all_files)
        if not files:
            return
    print(f"training on {len(files)} file(s):")
    for f in files:
        print("  -", os.path.basename(f))

    parts = [R.process_data(f, TARGET_MAPPINGS) for f in files]
    X = np.concatenate([a for a, _ in parts])
    y = np.concatenate([b for _, b in parts])
    groups = np.concatenate([[i] * len(b) for i, (_, b) in enumerate(parts)])

    sfreq = float(mne.io.read_raw_fif(files[0], preload=False, verbose=False).info["sfreq"])
    nb, C, T = X.shape[1], X.shape[2], X.shape[3]
    nc = 2

    clf = R.EAFilterBankCSP(n_components=nc).fit(X, y, groups=groups)
    print(f"trained EA + FB-CSP ({nc} comp) on {len(y)} epochs "
          f"({nb} bands x {C} ch x {T} samples, sfreq={sfreq:g})")

    # minimum-phase FIR taps per band (identical to run.bandpass)
    taps = [mne.filter.create_filter(None, sfreq, l, h, method="fir", phase="minimum",
                                     fir_design="firwin", verbose=False, **FB_TRANS)
            for (l, h) in FB_BANDS]
    max_len = max(len(t) for t in taps)
    fir_taps = np.zeros((nb, max_len))
    fir_lens = np.zeros(nb, dtype=int)
    for b, t in enumerate(taps):
        fir_taps[b, :len(t)] = t
        fir_lens[b] = len(t)

    csp_filters = np.stack([clf.csp_.csp_[b].filters_[:nc] for b in range(nb)])   # (nb, nc, C)
    lda_coef = clf.csp_.lda_.coef_[0].astype(np.float64)                          # (nb*nc,)
    lda_intercept = float(clf.csp_.lda_.intercept_[0])
    classes = clf.csp_.classes_.astype(np.int64)

    np.savez(
        OUT,
        fir_taps=fir_taps, fir_lens=fir_lens,
        csp_filters=csp_filters, lda_coef=lda_coef, lda_intercept=lda_intercept,
        classes=classes, sfreq=sfreq, window_n=T, n_components=nc,
        fb_bands=np.array(FB_BANDS, dtype=float),
        warmup_s=FILTER_WARMUP_S, stride_s=STRIDE_S, calibration_seconds=CALIBRATION_SECONDS,
        channels_targets=np.array(EEG_CHANNELS_TARGETS),
        channels_mapping=np.array(EEG_CHANNELS_MAPPING),
    )
    kb = os.path.getsize(OUT) / 1024
    print(f"\nexported -> {OUT}  ({kb:.1f} KB)")
    print("copy this model.npz to the SBC next to the binary.")


if __name__ == "__main__":
    main()
