
import glob
import mne
import numpy as np
import os
import sys
import threading
import time
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from brainflow.board_shim import BoardShim
from config import (DATA_DIR, EEG_CHANNELS_TARGETS, EPOCH_REJECT, EPOCH_TMIN, EPOCH_TMAX,
                    FB_BANDS, FB_TRANS, CSP_COMPONENTS, FILTER_WARMUP_S, STRIDE_S,
                    CALIBRATION_SECONDS, TARGET_MAPPINGS, NORM_CONF_FLOOR)
from mne.decoding import CSP
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from utils.devices import OpenBCI
from ws import WebSocket
from smoother import Smoother

def bandpass(data, sfreq, l_freq, h_freq):
    return mne.filter.filter_data(data, sfreq=sfreq, l_freq=l_freq, h_freq=h_freq,
                                  method='fir', phase='minimum', fir_design='firwin',
                                  verbose=False, **FB_TRANS)

def _band_decomp(feats, coef):
    """Slice boundaries of each band's block in the concatenated feature vector,
    plus that block's mean contribution (feat . coef) to the linear score over the
    training set. The discriminant is linear, so score = sum_band(contrib_b) + bias."""
    slices, start = [], 0
    for f in feats:
        slices.append((start, start + f.shape[1]))
        start += f.shape[1]
    F = np.hstack(feats)
    contribs = np.column_stack([F[:, s:e] @ coef[s:e] for (s, e) in slices])
    return slices, contribs.mean(axis=0)

def build_event_id(raw, target_id_dict):
    """Expand prefix-based target mappings to the exact annotation descriptions
    present in this raw file. Each annotation is assigned the code of the first
    target whose key matches it (exact or hierarchical prefix with '/')."""
    event_id = {}
    for desc in set(raw.annotations.description):
        d = str(desc).strip().lower()
        for prefix, code in target_id_dict.items():
            p = prefix.lower()
            if d == p or d.startswith(p + '/'):
                event_id[str(desc)] = code
                break
    return event_id


def process_data(file_name, target_id_dict):
    print(f"\n--- Loading: {file_name} ---")
    raw = mne.io.read_raw_fif(file_name, preload=True)
    raw.pick(EEG_CHANNELS_TARGETS)
    raw.set_montage('standard_1020', on_missing='ignore')
    raw.annotations.description = np.array([str(d).strip().lower() for d in raw.annotations.description])

    event_id = build_event_id(raw, target_id_dict)
    if not event_id:
        raise ValueError(
            f"No annotations in {file_name} matched any prefix in {target_id_dict}. "
            f"Annotations present: {sorted(set(raw.annotations.description))[:5]}…"
        )
    events, event_id_used = mne.events_from_annotations(raw, event_id=event_id)

    def band_epochs(l_freq, h_freq):
        filtered = raw.copy().filter(l_freq, h_freq, method='fir', phase='minimum',
                                     fir_design='firwin', verbose=False, **FB_TRANS)
        return mne.Epochs(filtered, events, event_id=event_id_used,
                          tmin=EPOCH_TMIN, tmax=EPOCH_TMAX, baseline=None,
                          preload=True, proj=False, on_missing='warn')

    bands = [band_epochs(l, h) for (l, h) in FB_BANDS]
    X = np.stack([b.get_data(copy=False) for b in bands], axis=1)
    y = bands[0].events[:, -1]

    broadband = band_epochs(FB_BANDS[0][0], FB_BANDS[-1][1]).get_data(copy=False)
    p2p = (broadband.max(axis=2) - broadband.min(axis=2)).max(axis=1)
    keep = p2p < EPOCH_REJECT['eeg']
    X, y = X[keep], y[keep]

    print(f"Created {len(y)} epochs ({int(keep.sum())}/{len(keep)} kept) for classes {sorted(set(y.tolist()))}")
    return X, y

def discover_files():
    return sorted(glob.glob(os.path.join(DATA_DIR, '*_mi_raw.fif')))

class FilterBankCSP(BaseEstimator, ClassifierMixin):
    # X is (n_epochs, n_bands, n_channels, n_times): one CSP per band, log-variance
    # features concatenated across bands, classified with shrinkage-LDA.
    def __init__(self, n_components=CSP_COMPONENTS):
        self.n_components = n_components

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.csp_ = []
        feats = []
        for b in range(X.shape[1]):
            csp = CSP(n_components=self.n_components, reg='ledoit_wolf', log=True, norm_trace=False)
            feats.append(csp.fit_transform(X[:, b], y))
            self.csp_.append(csp)
        self.lda_ = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        self.lda_.fit(np.hstack(feats), y)
        self.coef_ = self.lda_.coef_[0]
        self.band_slices_, self.band_mid_ = _band_decomp(feats, self.coef_)
        return self

    def _features(self, X):
        return np.hstack([csp.transform(X[:, b]) for b, csp in enumerate(self.csp_)])

    def predict(self, X):
        return self.lda_.predict(self._features(X))

    def predict_proba(self, X):
        return self.lda_.predict_proba(self._features(X))

    def analyze(self, X):
        """One feature pass -> (probabilities, signed discriminant score, per-band
        centered contribution to that score). band_signal[b] > 0 pushes toward
        classes_[1], < 0 toward classes_[0]; magnitude = how hard band b is voting."""
        F = self._features(X)
        contribs = np.column_stack([F[:, s:e] @ self.coef_[s:e] for (s, e) in self.band_slices_])
        return self.lda_.predict_proba(F), self.lda_.decision_function(F), contribs - self.band_mid_

def build_pipeline():
    return FilterBankCSP()

class EAFilterBankCSP(BaseEstimator, ClassifierMixin):
    """Euclidean Alignment (He et al. 2020) + Filter-bank CSP — the online/offline
    decoder. Per band, each domain is whitened so its mean covariance becomes the
    identity (R^-1/2, R = arithmetic-mean per-trial covariance), removing
    subject/session bias before CSP. fit() aligns per group (subject/session);
    set_reference(X_cal) recomputes the whitener from a calibration batch for the
    online distribution shift. Defaults to 2 CSP components (best cross-subject
    transfer). X is (n_epochs, n_bands, n_channels, n_times).
    """

    def __init__(self, n_components=2):
        self.n_components = n_components

    @staticmethod
    def _whitener(Xb):  # Xb (n, C, T) -> R^{-1/2}
        R = np.einsum("nct,ndt->cd", Xb, Xb) / (len(Xb) * Xb.shape[2])
        w, V = np.linalg.eigh(R)
        w = np.clip(w, 1e-12, None)
        return (V * (w ** -0.5)) @ V.T

    def _align(self, X, whiteners):
        out = np.empty_like(X, dtype=np.float64)
        for b in range(X.shape[1]):
            out[:, b] = np.einsum("cd,ndt->nct", whiteners[b], X[:, b])
        return out

    def fit(self, X, y, groups=None):
        nb = X.shape[1]
        self.ref_white_ = [self._whitener(X[:, b]) for b in range(nb)]  # pooled fallback ref
        if groups is None:
            Xa = self._align(X, self.ref_white_)
        else:
            groups = np.asarray(groups)
            Xa = np.empty_like(X, dtype=np.float64)
            for g in np.unique(groups):
                idx = groups == g
                Xa[idx] = self._align(X[idx], [self._whitener(X[idx, b]) for b in range(nb)])
        self.csp_ = FilterBankCSP(n_components=self.n_components).fit(Xa, y)
        self.classes_ = self.csp_.classes_
        return self

    def set_reference(self, X_cal):
        self.ref_white_ = [self._whitener(X_cal[:, b]) for b in range(X_cal.shape[1])]
        return self

    def predict(self, X):
        return self.csp_.predict(self._align(X, self.ref_white_))

    def predict_proba(self, X):
        return self.csp_.predict_proba(self._align(X, self.ref_white_))

    def analyze(self, X):
        return self.csp_.analyze(self._align(X, self.ref_white_))

class ConfidenceCalibrator:
    """Unsupervised per-side confidence calibration (no labels, model frozen).

    Cross-subject the LDA log-odds shift and one class often reads weaker than the
    other, so a fixed absolute floor stalls the weak hand. This centers the log-odds
    at the subject's own boundary (median of the calibration scores) and normalizes
    each class by its own high-percentile ceiling — so a genuine weak-hand MI reads
    as confident relative to *that* hand's ceiling. Seeded from the calibration
    block; the ceilings then only rise during the run (tracking the subject's best
    MI per side, robust to single artifacts). `center=False` keeps the raw boundary
    (predictions unchanged) and calibrates confidence only.
    """

    def __init__(self, pct=90.0, rise=0.05, min_count=10, min_ceil=1e-3, center=False):
        self.pct, self.rise = pct, rise
        self.min_count, self.min_ceil = min_count, min_ceil
        self.use_center = center
        self.center = 0.0
        self.ceil_pos = 1.0
        self.ceil_neg = 1.0

    def seed(self, scores):
        scores = np.asarray(scores, dtype=float)
        if scores.size == 0:
            return self
        if self.use_center:
            self.center = float(np.median(scores))
        z = scores - self.center
        pos, neg = z[z >= 0], -z[z < 0]
        has_pos, has_neg = pos.size >= self.min_count, neg.size >= self.min_count
        if has_pos:
            self.ceil_pos = max(self.min_ceil, float(np.percentile(pos, self.pct)))
        if has_neg:
            self.ceil_neg = max(self.min_ceil, float(np.percentile(neg, self.pct)))
        if not has_pos:
            self.ceil_pos = self.ceil_neg
        if not has_neg:
            self.ceil_neg = self.ceil_pos
        return self

    def score(self, s):
        """s = raw log-odds toward classes_[1]. Returns (side, conf): side +1 ->
        classes_[1], -1 -> classes_[0]; conf in [0,1] vs that side's rising ceiling."""
        z = s - self.center
        mag = abs(z)
        if z >= 0:
            if mag > self.ceil_pos:
                self.ceil_pos += self.rise * (mag - self.ceil_pos)
            return 1, min(1.0, mag / max(self.ceil_pos, self.min_ceil))
        if mag > self.ceil_neg:
            self.ceil_neg += self.rise * (mag - self.ceil_neg)
        return -1, min(1.0, mag / max(self.ceil_neg, self.min_ceil))


def record_calibration(bci, sfreq, train_idx, window_n, warmup_n, seconds):
    """Stream `seconds` of EEG while the subject imagines the task in the online
    context, then slice it into per-band windows for set_reference()."""
    buffer_n = window_n + warmup_n
    collected, lock = [], threading.Lock()

    def collect(chunk):
        eeg_all = chunk[bci.eeg, :]
        if eeg_all.shape[1] == 0 or max(train_idx) >= eeg_all.shape[0]:
            return
        with lock:
            collected.append(eeg_all[train_idx, :].astype(np.float64) / 1e6)

    bci.callback = collect
    bci.start()
    print(f"\nCalibration: imagine the task(s) in the online context for {seconds}s…")
    for s in range(seconds, 0, -1):
        print(f"  {s:2d}", end="\r", flush=True)
        time.sleep(1)
    bci.stop()
    bci.callback = None

    with lock:
        full = np.hstack(collected) if collected else np.zeros((len(train_idx), 0))
    if full.shape[1] < buffer_n:
        return None

    step = max(1, int(round(0.5 * sfreq)))
    windows = []
    for end in range(buffer_n, full.shape[1] + 1, step):
        buf = full[:, end - buffer_n:end]
        windows.append(np.stack([bandpass(buf, sfreq, l, h)[:, -window_n:] for (l, h) in FB_BANDS], axis=0))
    return np.array(windows)

def parse_indices(raw, n):
    out = []
    for tok in raw.replace(' ', '').split(','):
        if not tok:
            continue
        if not tok.isdigit():
            raise ValueError(f"'{tok}' is not a number")
        i = int(tok)
        if i < 1 or i > n:
            raise ValueError(f"{i} is out of range (1..{n})")
        if i - 1 not in out:
            out.append(i - 1)
    return out

def select_files(prompt, files):
    while True:
        try:
            raw = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if not raw:
            print("pick at least one")
            continue
        try:
            idxs = parse_indices(raw, len(files))
        except ValueError as e:
            print(f"invalid: {e}")
            continue
        if not idxs:
            print("pick at least one")
            continue
        return [files[i] for i in idxs]

def select_device():
    """USB (real Cyton+Daisy) or the brainflow synthetic board. Returns the
    `synthetic` flag for OpenBCI, or None if aborted."""
    print("\nSelect EEG device:")
    print("  1) USB       — OpenBCI Cyton+Daisy")
    print("  2) Synthetic — brainflow test board (no hardware)")
    while True:
        try:
            c = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return None
        if c in ("1", "usb"):
            return False
        if c in ("2", "synthetic", "synth"):
            return True
        print("pick 1 or 2")

def run_offline():
    files = discover_files()
    if not files:
        print(f"no .fif files found in {DATA_DIR}")
        return

    print("\nAvailable files:")
    for i, f in enumerate(files, 1):
        print(f"  {i}) {os.path.basename(f)}")
    print()

    train_files = select_files("training files (e.g. 1,2,3): ", files)
    if train_files is None:
        return
    test_files = select_files("testing files  (e.g. 1,2,3): ", files)
    if test_files is None:
        return

    overlap = set(train_files) & set(test_files)
    if overlap:
        print(f"warning: file(s) used in both train and test: {[os.path.basename(f) for f in overlap]}")

    def load_xy(file_list):
        parts = [process_data(f, TARGET_MAPPINGS) for f in file_list]
        X = np.concatenate([X for X, _ in parts])
        y = np.concatenate([y for _, y in parts])
        groups = np.concatenate([[i] * len(yy) for i, (_, yy) in enumerate(parts)])
        return X, y, groups

    X_train, y_train, g_train = load_xy(train_files)
    X_test, y_test, _ = load_xy(test_files)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    fold_acc = []
    for tr_i, va_i in cv.split(X_train, y_train):
        m = EAFilterBankCSP(n_components=2).fit(X_train[tr_i], y_train[tr_i], groups=g_train[tr_i])
        m.set_reference(X_train[va_i])
        fold_acc.append(accuracy_score(y_train[va_i], m.predict(X_train[va_i])))
    fold_acc = np.array(fold_acc)
    print(f"\n5-fold CV on training pool: "
          f"{fold_acc.mean() * 100:.2f}% +/- {fold_acc.std() * 100:.2f}%")

    clf = EAFilterBankCSP(n_components=2).fit(X_train, y_train, groups=g_train)
    clf.set_reference(X_test)   # unsupervised EA alignment to the target (no labels)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    labels = list(clf.classes_)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print("\nConfusion matrix (rows=true, cols=pred):")
    header = "        " + "".join(f"{c:>8}" for c in labels)
    print(header)
    for c, row in zip(labels, cm):
        print(f"{c:>8}" + "".join(f"{v:>8}" for v in row))

    print("\n" + "=" * 35)
    print(f"Motor Imagery Test Accuracy: {acc * 100:.2f}%")
    print("=" * 35)

def run_online(headless=False):
    synthetic = select_device()
    if synthetic is None:
        return

    files = discover_files()
    if not files:
        print(f"no .fif files found in {DATA_DIR}")
        return

    print("\nAvailable files:")
    for i, f in enumerate(files, 1):
        print(f"  {i}) {os.path.basename(f)}")
    print()

    train_files = select_files("training files (e.g. 1,2,3): ", files)
    if train_files is None:
        return

    print(f"\nTraining on {len(train_files)} file(s):")
    for f in train_files:
        print(f"  - {os.path.basename(f)}")

    parts = [process_data(f, TARGET_MAPPINGS) for f in train_files]
    X_train = np.concatenate([X for X, _ in parts])
    y_train = np.concatenate([y for _, y in parts])
    groups = np.concatenate([[i] * len(y) for i, (_, y) in enumerate(parts)])
    n_bands, n_channels, n_times = X_train.shape[1], X_train.shape[2], X_train.shape[3]

    clf = EAFilterBankCSP(n_components=2).fit(X_train, y_train, groups=groups)
    print(f"\nTrained EA + FB-CSP (2 comp) on {len(X_train)} epochs "
          f"({n_bands} bands x {n_channels} ch x {n_times} samples).")

    raw_full = mne.io.read_raw_fif(train_files[0], preload=False)
    full_names = raw_full.ch_names
    train_sfreq = float(raw_full.info['sfreq'])
    try:
        train_idx = [full_names.index(name) for name in EEG_CHANNELS_TARGETS]
    except ValueError as e:
        print(f"channel mismatch between training data and EEG_CHANNELS_TARGETS: {e}")
        return

    ws = WebSocket()
    ws.start()
    smoother = Smoother(ws, conf_floor=NORM_CONF_FLOOR)

    bci = OpenBCI(interval=STRIDE_S, synthetic=synthetic).open()
    if bci.board is None:
        print("OpenBCI failed to open.")
        ws.stop()
        return

    sfreq = float(BoardShim.get_sampling_rate(bci.board.board_id))
    if abs(sfreq - train_sfreq) > 0.5:
        print(f"warning: live sfreq={sfreq} differs from training sfreq={train_sfreq}; "
              "predictions may degrade")

    window_n = n_times
    warmup_n = int(round(FILTER_WARMUP_S * sfreq))
    buffer_n = window_n + warmup_n
    buffer = np.zeros((len(train_idx), 0), dtype=np.float64)
    buffer_lock = threading.Lock()

    print(f"mode: {'HEADLESS (no GUI)' if headless else 'GUI'}, stride: {STRIDE_S}s, "
          f"classify window: {window_n} samples ({window_n / sfreq:.2f}s), "
          f"buffer: {buffer_n} samples ({buffer_n / sfreq:.2f}s)")

    # EA alignment + unsupervised confidence calibration from the same block.
    calibrator = ConfidenceCalibrator()
    X_cal = record_calibration(bci, sfreq, train_idx, window_n, warmup_n, CALIBRATION_SECONDS)
    if X_cal is not None and len(X_cal) >= 2:
        clf.set_reference(X_cal)
        pcal = clf.predict_proba(X_cal)
        scal = np.log(np.clip(pcal[:, 1], 1e-9, 1.0) / np.clip(pcal[:, 0], 1e-9, 1.0))
        calibrator.seed(scal)
        print(f"aligned + confidence-calibrated on {len(X_cal)} windows "
              f"(boundary={calibrator.center:+.2f}, ceil[{clf.classes_[1]}]={calibrator.ceil_pos:.2f}, "
              f"ceil[{clf.classes_[0]}]={calibrator.ceil_neg:.2f})")
    else:
        print("calibration produced too little data; training reference, uncalibrated confidence.")

    gui = None
    if not headless:
        from gui import GUI
        _, _, train_signal = clf.analyze(X_train)
        cls = clf.classes_
        sig0, sig1 = train_signal[y_train == cls[0]], train_signal[y_train == cls[1]]
        band_sep = np.abs(sig1.mean(0) - sig0.mean(0)) / (np.sqrt(0.5 * (sig0.var(0) + sig1.var(0))) + 1e-9)
        band_abs_max = float(np.percentile(np.abs(train_signal), 99)) or 1.0
        gui = GUI(bci, smoother, clf.classes_, sfreq,
                  band_sep=band_sep, band_abs_max=band_abs_max)

    prev_print_t = None

    def on_chunk(chunk):
        nonlocal buffer, prev_print_t
        eeg_all = chunk[bci.eeg, :]
        if eeg_all.shape[1] == 0:
            return
        if max(train_idx) >= eeg_all.shape[0]:
            print(f"(got {eeg_all.shape[0]} channels, need index {max(train_idx)})")
            return

        eeg = eeg_all[train_idx, :].astype(np.float64, copy=False) / 1e6

        with buffer_lock:
            buffer = np.hstack([buffer, eeg])
            if buffer.shape[1] > buffer_n:
                buffer = buffer[:, -buffer_n:]
            if buffer.shape[1] < buffer_n:
                return
            buf = buffer.copy()

        bands = [bandpass(buf, sfreq, l, h)[:, -window_n:] for (l, h) in FB_BANDS]
        window = np.stack(bands, axis=0)[np.newaxis, ...]

        if headless:
            probs = clf.predict_proba(window)[0]
        else:
            proba, score, band_sig = clf.analyze(window)
            probs = proba[0]
            gui.push_decision(float(score[0]), band_sig[0])
        # subject-normalized confidence; s = log-odds toward classes_[1]
        s = float(np.log(max(probs[1], 1e-9) / max(probs[0], 1e-9)))
        side, conf = calibrator.score(s)
        pred = clf.classes_[1] if side > 0 else clf.classes_[0]
        breakdown = ", ".join(f"{c}={p*100:.1f}%" for c, p in zip(clf.classes_, probs))

        decision, consensus, final = smoother.add(
            prediction=pred,
            confidence=conf,
            probs=dict(zip(clf.classes_, probs)),
        )
        tag = f"{decision} (dwell {smoother.dwell_count}/{smoother.dwell})" if consensus else "—"
        commit = f"  ✓ COMMIT {final}" if final else ""
        now = time.perf_counter()
        dt_ms = (now - prev_print_t) * 1000 if prev_print_t is not None else 0.0
        prev_print_t = now
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        print(f"[{ts}  Δ{dt_ms:6.0f}ms]  {pred}  cal-conf {conf*100:3.0f}%  [{breakdown}]  → {tag}{commit}")

    bci.callback = on_chunk
    bci.start()

    try:
        if headless:
            print("\nrunning headless — decisions broadcasting over websocket. Ctrl-C to stop.\n")
            while True:
                time.sleep(1)
        else:
            gui.mainloop()
    except KeyboardInterrupt:
        print("\nStopping…")
    finally:
        bci.stop()
        bci.close()
        ws.stop()

def menu():
    print("=" * 35)
    print(" 1) Offline  (pick training/testing files)")
    print(" 2) Online   (live, with GUI)")
    print(" 3) Online   (headless, no GUI — for SBC)")
    print(" q) Quit")
    print("=" * 35)
    return input("> ").strip().lower()

def main():
    argv = sys.argv[1:]
    if argv and argv[0] in ('--headless', '--no-gui', '-H'):
        run_online(headless=True)
        return

    while True:
        try:
            choice = menu()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if choice in ('1', 'offline'):
            run_offline()
            return
        if choice in ('2', 'online'):
            run_online(headless=False)
            return
        if choice in ('3', 'headless', 'online-headless'):
            run_online(headless=True)
            return
        if choice in ('q', 'quit', 'exit'):
            return
        print("invalid choice\n")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
