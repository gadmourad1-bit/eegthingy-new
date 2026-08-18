import glob
import mne
import numpy as np
import os
import sys
import threading
import time
from collections import deque
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from brainflow.board_shim import BoardShim
from config import (DATA_DIR, EEG_CHANNELS_TARGETS, EEG_CHANNELS_MAPPING, EPOCH_REJECT,
                    EPOCH_TMIN, EPOCH_TMAX, FB_BANDS, FB_TRANS, CSP_COMPONENTS, FILTER_WARMUP_S,
                    STRIDE_S, CALIBRATION_TASK_SECONDS, CALIBRATION_REST_SECONDS,
                    TARGET_MAPPINGS, CONF_FLOOR,
                    RECENTER_ALPHA, RECENTER_CLAMP, RECENTER_REST_CONF)
from mne.decoding import CSP
from mne.preprocessing import compute_current_source_density
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix

try:
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    from pyriemann.utils.mean import mean_riemann
    from pyriemann.utils.base import invsqrtm
    _HAS_PYRIEMANN = True
except ImportError:
    _HAS_PYRIEMANN = False

try:
    from mirepnet_pipeline.inference import MIRepNetDecoder
    _HAS_MIREPNET = True
except ImportError:
    _HAS_MIREPNET = False

from utils.devices import OpenBCI
from ws import WebSocket
from smoother import Smoother


def bandpass(data, sfreq, l_freq, h_freq):
    return mne.filter.filter_data(data, sfreq=sfreq, l_freq=l_freq, h_freq=h_freq,
                                  method='fir', phase='zero', fir_design='firwin',
                                  verbose=False, **FB_TRANS)


def _band_decomp(feats, coef):
    slices, start = [], 0
    for f in feats:
        slices.append((start, start + f.shape[1]))
        start += f.shape[1]
    F = np.hstack(feats)
    contribs = np.column_stack([F[:, s:e] @ coef[s:e] for (s, e) in slices])
    return slices, contribs.mean(axis=0)


def build_event_id(raw, target_id_dict):
    event_id = {}
    for desc in set(raw.annotations.description):
        d = str(desc).strip().lower()
        for prefix, code in target_id_dict.items():
            p = prefix.lower()
            if d == p or d.startswith(p + '/'):
                event_id[str(desc)] = code
                break
    return event_id


def process_data(file_name, target_id_dict, apply_car=True, apply_laplacian=False):
    print(f"\n--- Loading: {file_name} ---")
    raw = mne.io.read_raw_fif(file_name, preload=True)
    raw.pick(EEG_CHANNELS_TARGETS)
    raw.set_montage('standard_1020', on_missing='ignore')

    if apply_car:
        raw.set_eeg_reference('average', projection=False)
        print("Applied Common Average Reference (CAR)")

    if apply_laplacian:
        raw = compute_current_source_density(raw)
        print("Applied Surface Laplacian")

    raw.annotations.description = np.array([str(d).strip().lower()
                                            for d in raw.annotations.description])

    event_id = build_event_id(raw, target_id_dict)
    if not event_id:
        raise ValueError(
            f"No annotations in {file_name} matched any prefix in {target_id_dict}. "
            f"Annotations present: {sorted(set(raw.annotations.description))[:5]}…"
        )
    events, event_id_used = mne.events_from_annotations(raw, event_id=event_id)

    def band_epochs(l_freq, h_freq):
        filtered = raw.copy().filter(l_freq, h_freq, method='fir', phase='zero',
                                     fir_design='firwin', verbose=False, **FB_TRANS)
        return mne.Epochs(filtered, events, event_id=event_id_used,
                          tmin=EPOCH_TMIN, tmax=EPOCH_TMAX, baseline=None,
                          preload=True, proj=False, on_missing='warn')

    bands = [band_epochs(l, h) for (l, h) in FB_BANDS]
    X = np.stack([b.get_data(copy=False) for b in bands], axis=1)
    y = bands[0].events[:, -1]

    broadband = band_epochs(FB_BANDS[0][0], FB_BANDS[-1][1]).get_data(copy=False)
    p2p = (broadband.max(axis=2) - broadband.min(axis=2)).max(axis=1)
    keep = p2p < EPOCH_REJECT['eeg'] * 2.0
    X, y = X[keep], y[keep]

    print(f"Created {len(y)} epochs ({int(keep.sum())}/{len(keep)} kept) "
          f"for classes {sorted(set(y.tolist()))}")
    return X, y


def process_data_mirepnet(file_name, target_id_dict, tmin=0.5, tmax=2.5,
                          apply_car=True, apply_laplacian=False,
                          l_freq=8.0, h_freq=30.0):
    print(f"\n--- Loading (MIRepNet): {file_name} ---")
    raw = mne.io.read_raw_fif(file_name, preload=True)
    raw.pick(EEG_CHANNELS_TARGETS)
    raw.set_montage('standard_1020', on_missing='ignore')

    if apply_car:
        raw.set_eeg_reference('average', projection=False)
        print("Applied Common Average Reference (CAR)")

    if apply_laplacian:
        raw = compute_current_source_density(raw)
        print("Applied Surface Laplacian")

    # Band‑pass filter to focus on motor imagery rhythms
    raw.filter(l_freq, h_freq, method='fir', phase='zero',
               fir_design='firwin', verbose=False)

    raw.annotations.description = np.array([str(d).strip().lower()
                                            for d in raw.annotations.description])

    event_id = build_event_id(raw, target_id_dict)
    if not event_id:
        raise ValueError(
            f"No annotations in {file_name} matched any prefix in {target_id_dict}. "
            f"Annotations present: {sorted(set(raw.annotations.description))[:5]}…"
        )
    events, event_id_used = mne.events_from_annotations(raw, event_id=event_id)

    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id_used,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        proj=False,
        on_missing='warn',
    )

    X = epochs.get_data(copy=False)
    y = epochs.events[:, -1]

    # Artifact rejection: peak‑to‑peak threshold on the filtered data
    p2p = (X.max(axis=2) - X.min(axis=2)).max(axis=1)
    keep = p2p < EPOCH_REJECT['eeg'] * 2.0
    X, y = X[keep], y[keep]

    classes, counts = np.unique(y, return_counts=True)
    print(f"Created {len(y)} MIRepNet epochs ({int(keep.sum())}/{len(keep)} kept) "
          f"for classes {classes} counts {counts}")
    print(f"Epoch shape: {X.shape}")

    if len(classes) < 2:
        print(f"WARNING: Only {len(classes)} class(es) found in {file_name}.")

    return X, y


def discover_files():
    return sorted(glob.glob(os.path.join(DATA_DIR, '*_mi_raw.fif')))


class FilterBankCSP(BaseEstimator, ClassifierMixin):
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
        F = self._features(X)
        contribs = np.column_stack([F[:, s:e] @ self.coef_[s:e] for (s, e) in self.band_slices_])
        return self.lda_.predict_proba(F), self.lda_.decision_function(F), contribs - self.band_mid_


class EAFilterBankCSP(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components=2):
        self.n_components = n_components

    @staticmethod
    def _whitener(Xb):
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
        self.ref_white_ = [self._whitener(X[:, b]) for b in range(nb)]
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


class FilterBankTangentSpace(BaseEstimator, ClassifierMixin):
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
            self.ref_white_.append(invsqrtm(mean_riemann(C)))
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


class BoundaryRecenter:
    def __init__(self, alpha=RECENTER_ALPHA, clamp=RECENTER_CLAMP, rest_conf=RECENTER_REST_CONF):
        self.alpha = alpha
        self.clamp = clamp
        self.rest_margin = float(np.log(rest_conf / (1.0 - rest_conf)))
        self.center = 0.0
        self.seed_center = 0.0
        self.offset = 0.0
        self._lock = threading.Lock()

    def seed(self, scores):
        scores = np.asarray(scores, dtype=float)
        if scores.size:
            with self._lock:
                self.center = self.seed_center = float(np.median(scores))
        return self

    def nudge(self, delta):
        with self._lock:
            self.offset += delta

    def shift(self, delta):
        with self._lock:
            self.center += delta
            self.seed_center += delta

    def update(self, s):
        with self._lock:
            if abs(s - self.center) < self.rest_margin:
                self.center += self.alpha * (s - self.center)
                lo, hi = self.seed_center - self.clamp, self.seed_center + self.clamp
                self.center = min(hi, max(lo, self.center))
            return s - self.center - self.offset


def _stream_windows(bci, sfreq, train_idx, window_n, warmup_n, seconds, prompt):
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
    print(prompt)
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


def _stream_windows_raw(bci, sfreq, train_idx, window_n, seconds, prompt):
    buffer_n = window_n
    collected, lock = [], threading.Lock()

    def collect(chunk):
        eeg_all = chunk[bci.eeg, :]
        if eeg_all.shape[1] == 0 or max(train_idx) >= eeg_all.shape[0]:
            return
        with lock:
            collected.append(eeg_all[train_idx, :].astype(np.float64) / 1e6)

    bci.callback = collect
    bci.start()
    print(prompt)
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
        windows.append(full[:, end - buffer_n:end])
    return np.array(windows)


def record_calibration(bci, sfreq, train_idx, window_n, warmup_n):
    X_task = _stream_windows(
        bci, sfreq, train_idx, window_n, warmup_n, CALIBRATION_TASK_SECONDS,
        f"\nCalibration 1/2: imagine the task(s), alternating sides, for {CALIBRATION_TASK_SECONDS}s…")
    X_rest = _stream_windows(
        bci, sfreq, train_idx, window_n, warmup_n, CALIBRATION_REST_SECONDS,
        f"\nCalibration 2/2: relax and imagine NOTHING for {CALIBRATION_REST_SECONDS}s…")
    return X_task, X_rest


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


def select_decoder():
    print("\nSelect decoder:")
    print("  1) EA + FB-CSP     — Euclidean Alignment (fast, default)")
    print("  2) Riemann tangent — Riemannian alignment + tangent space + LR")
    if _HAS_MIREPNET:
        print("  3) MIRepNet        — pretrained 45-ch transformer")
    while True:
        try:
            c = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            c = ""
        if c in ("", "1", "ea"):
            return "ea"
        if c in ("2", "riemann", "ra", "tangent"):
            if not _HAS_PYRIEMANN:
                print("pyriemann not installed — using EA (`uv add pyriemann` to enable Riemann).")
                return "ea"
            return "riemann"
        if c in ("3", "mirepnet", "mi"):
            if not _HAS_MIREPNET:
                print("MIRepNet is not available — check mirepnet_pipeline/inference.py")
                continue
            return "mirepnet"
        print("pick 1, 2 or 3")


def make_decoder(kind, sfreq=None):
    if kind == "riemann":
        return FilterBankTangentSpace()
    if kind == "mirepnet":
        finetuned_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "mirepnet_pipeline", "weights", "MIRepNet_finetuned.pth"
        )
        if os.path.exists(finetuned_path):
            print(f"[MIRepNet] Using fine‑tuned checkpoint: {finetuned_path}")
            return MIRepNetDecoder(sfreq=sfreq, pretrain_path=finetuned_path)
        else:
            print("[MIRepNet] Fine‑tuned checkpoint not found; using original pretrained weights.")
            return MIRepNetDecoder(sfreq=sfreq)
    return EAFilterBankCSP(n_components=2)


def decoder_label(kind):
    if kind == "riemann":
        return "Riemann tangent-space (RA)"
    if kind == "mirepnet":
        return "MIRepNet (pretrained transformer)"
    return "EA + FB-CSP (2 comp)"


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

    decoder_kind = select_decoder()

    if decoder_kind == "mirepnet":
        def load_xy(file_list):
            parts = [process_data_mirepnet(f, TARGET_MAPPINGS) for f in file_list]
            X = np.concatenate([X for X, _ in parts])
            y = np.concatenate([y for _, y in parts])
            groups = np.concatenate([[i] * len(yy) for i, (_, yy) in enumerate(parts)])
            return X, y, groups
    else:
        def load_xy(file_list):
            parts = [process_data(f, TARGET_MAPPINGS) for f in file_list]
            X = np.concatenate([X for X, _ in parts])
            y = np.concatenate([y for _, y in parts])
            groups = np.concatenate([[i] * len(yy) for i, (_, yy) in enumerate(parts)])
            return X, y, groups

    X_train, y_train, g_train = load_xy(train_files)
    X_test, y_test, _ = load_xy(test_files)

    decoder_sfreq = None
    if decoder_kind == "mirepnet":
        tmp_raw = mne.io.read_raw_fif(train_files[0], preload=False)
        decoder_sfreq = float(tmp_raw.info['sfreq'])
        print(f"MIRepNet source sfreq = {decoder_sfreq} Hz")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    fold_acc = []
    for tr_i, va_i in cv.split(X_train, y_train):
        m = make_decoder(decoder_kind, sfreq=decoder_sfreq).fit(X_train[tr_i], y_train[tr_i], groups=g_train[tr_i])
        m.set_reference(X_train[va_i])
        fold_acc.append(accuracy_score(y_train[va_i], m.predict(X_train[va_i])))
    fold_acc = np.array(fold_acc)
    print(f"\n[{decoder_label(decoder_kind)}]  5-fold CV on training pool: "
          f"{fold_acc.mean() * 100:.2f}% +/- {fold_acc.std() * 100:.2f}%")

    clf = make_decoder(decoder_kind, sfreq=decoder_sfreq).fit(X_train, y_train, groups=g_train)
    clf.set_reference(X_test)
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


RECORDINGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "recordings")


def save_online_recording(eeg_chunks, sfreq, out_path):
    if not eeg_chunks:
        print("recording: no EEG captured, nothing saved")
        return
    data = np.hstack(eeg_chunks).astype(np.float64) / 1e6
    n_ch = data.shape[0]
    ch_names = (list(EEG_CHANNELS_MAPPING) if n_ch == len(EEG_CHANNELS_MAPPING)
                else [f"EEG{i + 1}" for i in range(n_ch)])
    raw = mne.io.RawArray(data, mne.create_info(ch_names, sfreq, ["eeg"] * n_ch), verbose=False)
    if "DEAD" in ch_names:
        raw.set_channel_types({"DEAD": "misc"})
    raw.set_montage("standard_1020", on_missing="ignore")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    raw.save(out_path, overwrite=True, verbose=False)
    print(f"recording saved: {raw.n_times} samples ({raw.n_times / sfreq:.1f}s) -> {out_path}")


def save_decision_log(rows, out_path):
    if not rows:
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("time,log_odds,center,margin,pred,conf,committed,conf_floor,bands\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    print(f"decision log saved: {len(rows)} rows -> {out_path}")


def run_online(headless=False):
    synthetic = select_device()
    if synthetic is None:
        return

    decoder_kind = select_decoder()

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

    if decoder_kind == "mirepnet":
        parts = [process_data_mirepnet(f, TARGET_MAPPINGS) for f in train_files]
        X_train = np.concatenate([X for X, _ in parts])
        y_train = np.concatenate([y for _, y in parts])
        groups = np.concatenate([[i] * len(y) for i, (_, y) in enumerate(parts)])
        n_bands, n_channels, n_times = 1, X_train.shape[1], X_train.shape[2]
    else:
        parts = [process_data(f, TARGET_MAPPINGS) for f in train_files]
        X_train = np.concatenate([X for X, _ in parts])
        y_train = np.concatenate([y for _, y in parts])
        groups = np.concatenate([[i] * len(y) for i, (_, y) in enumerate(parts)])
        n_bands, n_channels, n_times = X_train.shape[1], X_train.shape[2], X_train.shape[3]

    raw_full = mne.io.read_raw_fif(train_files[0], preload=False)
    full_names = raw_full.ch_names
    train_sfreq = float(raw_full.info['sfreq'])

    clf = make_decoder(decoder_kind, sfreq=train_sfreq).fit(X_train, y_train, groups=groups)
    print(f"\nTrained {decoder_label(decoder_kind)} on {len(X_train)} epochs "
          f"({n_bands} bands x {n_channels} ch x {n_times} samples).")

    try:
        train_idx = [full_names.index(name) for name in EEG_CHANNELS_TARGETS]
    except ValueError as e:
        print(f"channel mismatch between training data and EEG_CHANNELS_TARGETS: {e}")
        return

    ws = WebSocket()
    ws.start()
    smoother = Smoother(ws, conf_floor=CONF_FLOOR)

    bci = OpenBCI(interval=STRIDE_S, synthetic=synthetic).open()
    if bci.board is None:
        print("OpenBCI failed to open.")
        ws.stop()
        return

    sfreq = float(BoardShim.get_sampling_rate(bci.board.board_id))
    if abs(sfreq - train_sfreq) > 0.5:
        print(f"warning: live sfreq={sfreq} differs from training sfreq={train_sfreq}; "
              "predictions may degrade")

    if decoder_kind == "mirepnet":
        window_n = int(round(1.0 * sfreq))
        warmup_n = 0
    else:
        window_n = n_times
        warmup_n = int(round(FILTER_WARMUP_S * sfreq))
    buffer_n = window_n + warmup_n
    buffer = np.zeros((len(train_idx), 0), dtype=np.float64)
    buffer_lock = threading.Lock()

    print(f"mode: {'HEADLESS (no GUI)' if headless else 'GUI'}, stride: {STRIDE_S}s, "
          f"classify window: {window_n} samples ({window_n / sfreq:.2f}s), "
          f"buffer: {buffer_n} samples ({buffer_n / sfreq:.2f}s)")

    recenter = BoundaryRecenter()
    if decoder_kind == "mirepnet":
        X_task = _stream_windows_raw(
            bci, sfreq, train_idx, window_n, CALIBRATION_TASK_SECONDS,
            f"\nCalibration 1/2: imagine the task(s) for {CALIBRATION_TASK_SECONDS}s…")
        X_rest = _stream_windows_raw(
            bci, sfreq, train_idx, window_n, CALIBRATION_REST_SECONDS,
            f"\nCalibration 2/2: relax and imagine NOTHING for {CALIBRATION_REST_SECONDS}s…")
    else:
        X_task, X_rest = record_calibration(bci, sfreq, train_idx, window_n, warmup_n)

    if X_task is not None and len(X_task) >= 2:
        clf.set_reference(X_task)
        print(f"aligned on {len(X_task)} task windows")
    else:
        print("task calibration produced too little data; using pooled training reference.")

    def log_odds(X):
        p = clf.predict_proba(X)
        return np.log(np.clip(p[:, 1], 1e-9, 1.0) / np.clip(p[:, 0], 1e-9, 1.0))

    if X_rest is not None and len(X_rest) >= 2:
        recenter.seed(log_odds(X_rest))
        print(f"rest baseline from {len(X_rest)} windows: boundary center={recenter.center:+.2f}")
    elif X_task is not None and len(X_task) >= 2:
        recenter.seed(log_odds(X_task))
        print(f"no rest data; boundary seeded from task windows (center={recenter.center:+.2f})")
    else:
        print("no calibration data; boundary center=0.")

    gui = None
    if not headless:
        from gui import GUI
        _, _, train_signal = clf.analyze(X_train)
        cls = clf.classes_
        sig0, sig1 = train_signal[y_train == cls[0]], train_signal[y_train == cls[1]]
        band_sep = np.abs(sig1.mean(0) - sig0.mean(0)) / (np.sqrt(0.5 * (sig0.var(0) + sig1.var(0))) + 1e-9)
        band_abs_max = float(np.percentile(np.abs(train_signal), 99)) or 1.0
        gui = GUI(bci, smoother, clf.classes_, sfreq,
                  band_sep=band_sep, band_abs_max=band_abs_max,
                  recenter=recenter, base_center=recenter.center)

    prev_print_t = None
    record = not headless
    rec_eeg, rec_rows = [], []
    rec_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    prev_mask = None
    recent_sig = deque(maxlen=50)

    def on_chunk(chunk):
        nonlocal buffer, prev_print_t, prev_mask
        eeg_all = chunk[bci.eeg, :]
        if eeg_all.shape[1] == 0:
            return
        if max(train_idx) >= eeg_all.shape[0]:
            print(f"(got {eeg_all.shape[0]} channels, need index {max(train_idx)})")
            return

        if record:
            rec_eeg.append(eeg_all.astype(np.float32))

        eeg = eeg_all[train_idx, :].astype(np.float64, copy=False) / 1e6

        with buffer_lock:
            buffer = np.hstack([buffer, eeg])
            if buffer.shape[1] > buffer_n:
                buffer = buffer[:, -buffer_n:]
            if buffer.shape[1] < buffer_n:
                return
            buf = buffer.copy()

        if decoder_kind == "mirepnet":
            window = buf[np.newaxis, ...]
        else:
            bands = [bandpass(buf, sfreq, l, h)[:, -window_n:] for (l, h) in FB_BANDS]
            window = np.stack(bands, axis=0)[np.newaxis, ...]

        band_mask = None
        if headless:
            probs_raw = clf.predict_proba(window)[0]
            s = float(np.log(max(probs_raw[1], 1e-9) / max(probs_raw[0], 1e-9)))
        else:
            _, score, band_sig = clf.analyze(window)
            sig = band_sig[0]
            band_mask = list(gui.band_on)
            if prev_mask is not None and band_mask != prev_mask and recent_sig:
                hist = np.mean(recent_sig, axis=0)
                delta = sum((float(hist[b]) if band_mask[b] else -float(hist[b]))
                            for b in range(min(len(hist), len(band_mask), len(prev_mask)))
                            if band_mask[b] != prev_mask[b])
                if delta:
                    recenter.shift(delta)
                    print(f"band mask {''.join('1' if m else '0' for m in band_mask)} "
                          f"-> boundary shifted {delta:+.2f}")
            prev_mask = band_mask
            recent_sig.append(sig.copy())
            s = float(score[0]) - sum(float(sig[b]) for b in range(len(sig))
                                      if b < len(band_mask) and not band_mask[b])
        z = recenter.update(s)
        center_used = s - z
        if not headless:
            gui.push_decision(s, center_used, sig)
        p1 = 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))
        probs = np.array([1.0 - p1, p1])
        pred = clf.classes_[int(np.argmax(probs))]
        conf = float(np.max(probs))
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
        print(f"[{ts}  Δ{dt_ms:6.0f}ms]  {pred}  conf {conf*100:3.0f}% (ctr {center_used:+.1f})  [{breakdown}]  → {tag}{commit}")

        if record:
            mask_str = "" if band_mask is None else "".join("1" if b else "0" for b in band_mask)
            rec_rows.append((ts, f"{s:.4f}", f"{center_used:.4f}", f"{z:.4f}",
                             int(pred), f"{conf:.4f}", int(final),
                             f"{smoother.conf_floor:.2f}", mask_str))

    bci.callback = on_chunk
    bci.start()

    if record:
        print(f"\n📼 recording session → recordings/online_{rec_stamp}_raw.fif (+ _decisions.csv)")

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
        if record:
            save_online_recording(rec_eeg, sfreq, os.path.join(RECORDINGS_DIR, f"online_{rec_stamp}_raw.fif"))
            save_decision_log(rec_rows, os.path.join(RECORDINGS_DIR, f"online_{rec_stamp}_decisions.csv"))


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