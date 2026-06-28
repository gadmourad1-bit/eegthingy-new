
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
                    CALIBRATION_SECONDS, TARGET_MAPPINGS)
from mne.decoding import CSP
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.base import invsqrtm
from utils.devices import OpenBCI
from ws import WebSocket
from smoother import Smoother
from gui import GUI

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

class FilterBankTangentSpace(BaseEstimator, ClassifierMixin):
    # Per-band Riemannian tangent-space features. Training covariances are recentered
    # per session (groups) so each session sits at the identity; set_reference() then
    # recenters live data by an in-context calibration block, absorbing the
    # train->online distribution shift. X is (n_epochs, n_bands, n_channels, n_times).
    def fit(self, X, y, groups=None):
        self.classes_ = np.unique(y)
        self.cov_, self.ts_, self.ref_white_ = [], [], []
        feats = []
        for b in range(X.shape[1]):
            cov = Covariances('lwf')
            C = cov.transform(X[:, b])
            ts = TangentSpace('riemann')
            aligned = ts.fit_transform(self._recenter(C, groups))
            self.cov_.append(cov)
            self.ts_.append(ts)
            self.ref_white_.append(invsqrtm(mean_riemann(C)))  # fallback ref: training mean
            feats.append(aligned)
        self.lr_ = LogisticRegression(max_iter=2000)
        self.lr_.fit(np.hstack(feats), y)
        self.coef_ = self.lr_.coef_[0]
        self.band_slices_, self.band_mid_ = _band_decomp(feats, self.coef_)
        return self

    def set_reference(self, X_cal):
        for b in range(X_cal.shape[1]):
            self.ref_white_[b] = invsqrtm(mean_riemann(self.cov_[b].transform(X_cal[:, b])))
        return self

    @staticmethod
    def _recenter(C, groups):
        if groups is None:
            W = invsqrtm(mean_riemann(C))
            return np.array([W @ c @ W for c in C])
        out = np.empty_like(C)
        groups = np.asarray(groups)
        for g in np.unique(groups):
            idx = groups == g
            W = invsqrtm(mean_riemann(C[idx]))
            out[idx] = np.array([W @ c @ W for c in C[idx]])
        return out

    def _features(self, X):
        feats = []
        for b in range(X.shape[1]):
            C = self.cov_[b].transform(X[:, b])
            W = self.ref_white_[b]
            feats.append(self.ts_[b].transform(np.array([W @ c @ W for c in C])))
        return np.hstack(feats)

    def predict(self, X):
        return self.lr_.predict(self._features(X))

    def predict_proba(self, X):
        return self.lr_.predict_proba(self._features(X))

    def analyze(self, X):
        F = self._features(X)
        contribs = np.column_stack([F[:, s:e] @ self.coef_[s:e] for (s, e) in self.band_slices_])
        return self.lr_.predict_proba(F), self.lr_.decision_function(F), contribs - self.band_mid_

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
        return np.concatenate([X for X, _ in parts]), np.concatenate([y for _, y in parts])

    X_train, y_train = load_xy(train_files)
    X_test, y_test = load_xy(test_files)

    clf = build_pipeline()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    cv_scores = cross_val_score(build_pipeline(), X_train, y_train, cv=cv)
    print(f"\n5-fold CV on training pool: "
          f"{cv_scores.mean() * 100:.2f}% +/- {cv_scores.std() * 100:.2f}%")

    clf.fit(X_train, y_train)
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

def run_online():
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

    ans = input("\nUse Riemannian recentering? Recommended unless this is the same "
                "session with the cap untouched since training. [Y/n]: ").strip().lower()
    use_recenter = ans not in ("n", "no")

    parts = [process_data(f, TARGET_MAPPINGS) for f in train_files]
    X_train = np.concatenate([X for X, _ in parts])
    y_train = np.concatenate([y for _, y in parts])
    groups = np.concatenate([[i] * len(y) for i, (_, y) in enumerate(parts)])
    n_bands, n_channels, n_times = X_train.shape[1], X_train.shape[2], X_train.shape[3]

    if use_recenter:
        clf = FilterBankTangentSpace().fit(X_train, y_train, groups=groups)
    else:
        clf = build_pipeline().fit(X_train, y_train)
    print(f"\nTrained {'FB tangent-space (recentered)' if use_recenter else 'FBCSP'} on "
          f"{len(X_train)} epochs ({n_bands} bands x {n_channels} ch x {n_times} samples).")

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
    smoother = Smoother(ws)

    bci = OpenBCI(interval=STRIDE_S).open()
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

    print(f"stride: {STRIDE_S}s, classify window: {window_n} samples "
          f"({window_n / sfreq:.2f}s), buffer: {buffer_n} samples ({buffer_n / sfreq:.2f}s)")

    if use_recenter:
        X_cal = record_calibration(bci, sfreq, train_idx, window_n, warmup_n, CALIBRATION_SECONDS)
        if X_cal is not None and len(X_cal) >= 2:
            clf.set_reference(X_cal)
            print(f"recentered on {len(X_cal)} calibration windows.")
        else:
            print("calibration produced too little data; falling back to training reference.")

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

        proba, score, band_sig = clf.analyze(window)
        probs = proba[0]
        idx = int(np.argmax(probs))
        pred = clf.classes_[idx]
        conf = probs[idx]
        gui.push_decision(float(score[0]), band_sig[0])
        breakdown = ", ".join(f"{c}={p*100:.1f}%" for c, p in zip(clf.classes_, probs))

        decision, consensus, final = smoother.add(
            prediction=pred,
            confidence=conf,
            probs=dict(zip(clf.classes_, probs)),
        )
        tag = decision if consensus else "—"
        commit = f"  ✓ COMMIT {final}" if final else ""
        now = time.perf_counter()
        dt_ms = (now - prev_print_t) * 1000 if prev_print_t is not None else 0.0
        prev_print_t = now
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        print(f"[{ts}  Δ{dt_ms:6.0f}ms]  raw: {pred}  conf: {conf*100:.1f}%  [{breakdown}]  → smoothed: {tag}{commit}")

    bci.callback = on_chunk
    bci.start()

    try:
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
    print(" 2) Online   (train on all files, classify live)")
    print(" q) Quit")
    print("=" * 35)
    return input("> ").strip().lower()

def main():
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
            run_online()
            return
        if choice in ('q', 'quit', 'exit'):
            return
        print("invalid choice\n")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
