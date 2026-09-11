
import glob
import json
import mne
import numpy as np
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from brainflow.board_shim import BoardShim
from serial.tools import list_ports
from config import (DATA_DIR, EEG_CHANNELS_TARGETS, EEG_CHANNELS_MAPPING, EPOCH_REJECT,
                    EPOCH_TMIN, EPOCH_TMAX, FB_BANDS, FB_TRANS, CSP_COMPONENTS, FILTER_WARMUP_S,
                    STRIDE_S, CALIBRATION_SECONDS, CALIBRATION_SECONDS_DEEP,
                     DEEP_EPOCHS, DEEP_SEED, DEEP_DEVICE, TARGET_MAPPINGS, CONF_FLOOR,
                     MIREPNET_EPOCHS, MIREPNET_SEED, MIREPNET_DEVICE, MIREPNET_WINDOW_MODE,
                     LOCAL_EXP4_VALID_RUNS,
                     RECENTER_ADAPTIVE, RECENTER_ALPHA, RECENTER_CLAMP, RECENTER_REST_CONF)
from mne.decoding import CSP
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix

try:   # optional: enables the Riemannian decoder; EA works without pyriemann
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    from pyriemann.utils.mean import mean_riemann
    from pyriemann.utils.base import invsqrtm
    _HAS_PYRIEMANN = True
except ImportError:
    _HAS_PYRIEMANN = False
try:   # optional: enables the deep decoders; needs torch + the dl/ package
    import deep as deepmod
    _HAS_DEEP = True
except Exception as _deep_error:   # torch missing, or dl/ unavailable
    _DEEP_IMPORT_ERROR = _deep_error
    _HAS_DEEP = False
try:   # MI-specific foundation model + official pretrained checkpoint
    import mirepnet as mirepmod
    _HAS_MIREPNET = True
except Exception as _mirepnet_error:
    _MIREPNET_IMPORT_ERROR = _mirepnet_error
    _HAS_MIREPNET = False
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


class FilterBankTangentSpace(BaseEstimator, ClassifierMixin):
    """Riemannian alternative to EAFilterBankCSP (same fit/set_reference/predict_proba/
    analyze API). Per band: LWF covariances, Riemannian recentering per group (each
    session recentred to the identity), tangent-space projection, logistic regression.
    set_reference() recentres live data on an in-context calibration block, absorbing the
    train->online shift. X is (n_epochs, n_bands, n_channels, n_times)."""

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
            self.ref_white_.append(invsqrtm(mean_riemann(C)))   # fallback ref: training mean
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
    """Unsupervised adaptive decision-boundary recentering (no labels, model frozen).

    EA aligns covariance but not *where* a subject's log-odds sit, so cross-subject the
    LDA boundary is offset (left-skew; rest reads as a confident class) and drifts
    within a session (the decoder sticks). Seed the neutral from the calibration block,
    then track it from rest-like windows ONLY (so sustained real MI never pulls it),
    clamped near the seed so it can't run away. Decisions and confidence are taken on
    the recentered margin z = s - center; gating that at CONF_FLOOR gives a rest
    dead-zone for free (idle sits near the boundary -> low confidence -> no commit).
    RECENTER_ADAPTIVE=False freezes the center at its calibration seed (no live
    feedback loop); the seed and the recentered decisions still apply.
    """

    def __init__(self, adaptive=RECENTER_ADAPTIVE, alpha=RECENTER_ALPHA,
                 clamp=RECENTER_CLAMP, rest_conf=RECENTER_REST_CONF):
        self.adaptive = adaptive
        self.alpha = alpha
        self.clamp = clamp
        self.rest_margin = float(np.log(rest_conf / (1.0 - rest_conf)))  # |z| below this = rest-like
        self.center = 0.0
        self.seed_center = 0.0

    def seed(self, scores):
        scores = np.asarray(scores, dtype=float)
        if scores.size:
            self.center = self.seed_center = float(np.median(scores))
        return self

    def update(self, s):
        """Track the neutral from rest-like windows only, clamped to seed +/- clamp
        (skipped entirely when frozen); return the recentered margin z = s - center."""
        if self.adaptive and abs(s - self.center) < self.rest_margin:
            self.center += self.alpha * (s - self.center)
            lo, hi = self.seed_center - self.clamp, self.seed_center + self.clamp
            self.center = min(hi, max(lo, self.center))
        return s - self.center


def record_calibration(bci, sfreq, train_idx, window_n, buffer_n, make_window,
                       seconds, deep_mode=False):
    """Stream `seconds` of EEG in the online context, then slice it into decoder
    windows. For the classical decoders these feed set_reference(); for the nets
    they only seed the decision boundary."""
    collected, lock = [], threading.Lock()

    def collect(chunk):
        eeg_all = chunk[bci.eeg, :]
        if eeg_all.shape[1] == 0 or max(train_idx) >= eeg_all.shape[0]:
            return
        with lock:
            collected.append(eeg_all[train_idx, :].astype(np.float64) / 1e6)

    bci.callback = collect
    bci.start()
    what = ("Calibration (boundary only — the network needs no alignment): relax, then "
            "imagine the task(s)" if deep_mode
            else "Calibration: imagine the task(s) in the online context")
    print(f"\n{what} for {seconds}s…")
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
        windows.append(make_window(full[:, end - buffer_n:end])[0])
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


def select_serial_port():
    """Let the operator select the OpenBCI COM port instead of relying only on a probe."""
    ports = list(list_ports.comports())
    print("\nSerial ports visible to Windows:")
    if ports:
        for index, port in enumerate(ports, 1):
            maker = port.manufacturer or "unknown manufacturer"
            print(f"  {index}) {port.device} — {maker} {port.description or ''}")
        likely = [
            port for port in ports
            if "ftdi" in (port.manufacturer or "").lower()
            or "ftdi" in (port.description or "").lower()
            or "usbserial" in port.device.lower()
            or (port.vid, port.pid) == (0x0403, 0x6015)
        ]
        default = likely[0].device if len(likely) == 1 else (ports[0].device if len(ports) == 1 else "")
    else:
        print("  (none detected)")
        default = ""

    prompt = f"OpenBCI serial port{f' [{default}]' if default else ' (example COM5)'}> "
    while True:
        try:
            answer = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if not answer and default:
            return default
        if answer.isdigit() and ports and 1 <= int(answer) <= len(ports):
            return ports[int(answer) - 1].device
        if answer:
            normalized = answer.upper()
            known = {port.device.upper(): port.device for port in ports}
            return known.get(normalized, normalized)
        print("Enter a COM port, its list number, or Ctrl-C to cancel.")


def select_live_maze():
    """Collect the small amount of metadata needed for a headset-driven maze."""
    print("\nLive MIRepNet OpenBCI maze")
    print("The robot stops at every corner and waits for fresh post-stop EEG.")
    print("Imagine LEFT hand = turn left; imagine RIGHT hand = turn right.")
    print("\nChoose one of the four original fixed mazes:")
    print("  1) Standard maze 1 (seed 11)")
    print("  2) Standard maze 2 (seed 12)")
    print("  3) Standard maze 3 (seed 13)")
    print("  4) Standard maze 4 (seed 14)")
    while True:
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if raw in ("1", "2", "3", "4"):
            maze_number = int(raw)
            break
        print("enter 1, 2, 3, or 4")

    try:
        subject = input("subject id [live]> ").strip() or "live"
        test = input(f"test id [live-mirepnet-maze{maze_number}]> ").strip()
        test = test or f"live-mirepnet-maze{maze_number}"
        print("view: 1) third-person  2) first-person  3) top-down")
        view_raw = input("[1]> ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return None
    view = {"1": "third", "2": "first", "3": "top", "": "third"}.get(view_raw)
    if view is None:
        print("unknown view; using third-person")
        view = "third"
    return {"maze_number": maze_number, "subject": subject, "test": test, "view": view}


def launch_live_maze(config, checkpoint, calibrate):
    """Open the Panda3D maze as a localhost client of the live decoder."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sim_dir = os.path.join(project_root, "simulation")
    src_dir = os.path.join(sim_dir, "src")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        item for item in (src_dir, env.get("PYTHONPATH")) if item
    )
    metadata = json.dumps({
        "mode": "live-openbci-mirepnet-maze",
        "checkpoint": os.path.basename(checkpoint) if checkpoint else "session model",
        "calibration": "unlabeled" if calibrate is not False else "none",
    })
    command = [
        sys.executable,
        "-m",
        "tiago_maze",
        "--standard-maze",
        str(config["maze_number"]),
        "--view",
        config["view"],
        "--ws-url",
        "ws://127.0.0.1:8765",
        "--no-manual-keys",
        "--subject",
        config["subject"],
        "--test",
        config["test"],
        "--meta",
        metadata,
    ]
    print(f"\nOpening standard maze {config['maze_number']} for live EEG control...")
    print("At each wall, wait for the prompt, then sustain LEFT- or RIGHT-hand imagery.")
    print("Close the maze window to stop and save the EEG, decisions, CSV report, and trace.")
    return subprocess.Popen(command, cwd=sim_dir, env=env)

def select_decoder():
    """Choose a classical, project neural, or pretrained foundation decoder."""
    print("\nSelect decoder:")
    print("  1) EA + FB-CSP     — Euclidean Alignment (fast, default)")
    print("  2) Riemann tangent — Riemannian alignment + tangent space + LR")
    print("  3) Deep: CardinalFBC + compact dynamics — 32k params, benchmark rank 1")
    print("  4) Deep: Cardinal Sinc dynamics        — 13.5k params, best on local data")
    print("  5) Foundation: MIRepNet — pretrained MI transformer, 5.14M params")
    if not _HAS_DEEP:
        print("     (3 and 4 need torch: `uv pip install torch`)")
    if not _HAS_MIREPNET:
        print("     (5 unavailable: install huggingface-hub + safetensors)")
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
        if c in ("3", "compactdyn", "deep", "4", "sinc"):
            if not _HAS_DEEP:
                print(f"deep decoders unavailable: {_DEEP_IMPORT_ERROR}")
                continue
            return "deep_compactdyn" if c in ("3", "compactdyn", "deep") else "deep_sinc"
        if c in ("5", "mirepnet", "foundation", "fm"):
            if not _HAS_MIREPNET:
                print(f"MIRepNet unavailable: {_MIREPNET_IMPORT_ERROR}")
                continue
            return "mirepnet"
        print("pick 1-5")

def neural_module(kind):
    return mirepmod if kind == "mirepnet" else deepmod


def select_neural_source(kind):
    """Reuse a saved neural model or train a new one from recordings."""
    module = neural_module(kind)
    saved = module.list_checkpoints(kind)
    if not saved:
        print("\nNo saved model for this decoder yet — training a new one.")
        return None
    print(f"\nSaved {decoder_label(kind)} models:")
    for i, path in enumerate(saved, 1):
        print(f"  {i}) {module.describe_checkpoint(path)}")
    print("  t) Train a new one from recordings")
    while True:
        try:
            c = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return None
        if c in ("t", "train", "n", "new"):
            return None
        if c.isdigit() and 1 <= int(c) <= len(saved):
            return saved[int(c) - 1]
        print(f"pick 1-{len(saved)} or t")


def is_deep(kind):
    return kind.startswith("deep_")


def is_foundation(kind):
    return kind == "mirepnet"


def is_neural(kind):
    return is_deep(kind) or is_foundation(kind)

def make_decoder(kind):
    """Instantiate the chosen decoder; all share the fit/set_reference/predict_proba/analyze API."""
    if is_deep(kind):
        return deepmod.DeepDecoder(kind, epochs=DEEP_EPOCHS, seed=DEEP_SEED,
                                   device=DEEP_DEVICE)
    if is_foundation(kind):
        return mirepmod.MIRepNetDecoder(epochs=MIREPNET_EPOCHS, seed=MIREPNET_SEED,
                                        device=MIREPNET_DEVICE)
    return FilterBankTangentSpace() if kind == "riemann" else EAFilterBankCSP(n_components=2)

def decoder_label(kind):
    if is_deep(kind):
        return deepmod.DEEP_MODELS[kind][1]
    if is_foundation(kind):
        return "MIRepNet foundation model (pretrained, 5.14M params)"
    return "Riemann tangent-space (RA)" if kind == "riemann" else "EA + FB-CSP (2 comp)"

def load_epochs(kind, file_name):
    """Front end for the chosen decoder family: the classical four-band 8-30 Hz
    causal path, or the benchmark's 4-40 Hz / 128 Hz / CAR profile for the nets."""
    if is_deep(kind):
        return deepmod.process_data(file_name, TARGET_MAPPINGS)
    if is_foundation(kind):
        return mirepmod.process_data(file_name, TARGET_MAPPINGS, MIREPNET_WINDOW_MODE)
    return process_data(file_name, TARGET_MAPPINGS)


def warn_mirepnet_data_contract(files):
    """Flag recordings outside the repository's frozen Local Exp4 manifest."""
    outside = []
    for path in files:
        match = re.search(r"subject(\d+)_training_(\d+)_mi_raw", os.path.basename(path))
        if not match:
            outside.append(os.path.basename(path))
            continue
        subject, run = map(int, match.groups())
        if run not in LOCAL_EXP4_VALID_RUNS.get(subject, ()):
            outside.append(os.path.basename(path))
    if outside:
        print("\nMIRepNet data-contract warning: the validated Local Exp4 result applies only "
              "to S1/S3/S4/S5/S6/S7/S8 runs 1-4 and S10 runs 5-8.")
        print("Selected recordings outside that frozen manifest:")
        for name in outside:
            print(f"  - {name}")
        print("They can still be explored, but their scores must not be mixed into the "
              "validated eight-patient benchmark.")

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
    if is_foundation(decoder_kind):
        warn_mirepnet_data_contract(train_files + test_files)

    checkpoint = select_neural_source(decoder_kind) if is_foundation(decoder_kind) else None

    def load_xy(file_list):
        parts = [load_epochs(decoder_kind, f) for f in file_list]
        X = np.concatenate([X for X, _ in parts])
        y = np.concatenate([y for _, y in parts])
        groups = np.concatenate([[i] * len(yy) for i, (_, yy) in enumerate(parts)])
        return X, y, groups

    X_train, y_train, g_train = load_xy(train_files)
    X_test, y_test, _ = load_xy(test_files)

    if checkpoint is not None:
        clf = mirepmod.MIRepNetDecoder.load(checkpoint, device=MIREPNET_DEVICE)
        clf.fit_feature_adapter(X_train, y_train)
        raw_pred = clf.predict_feature_adapter(X_test, balanced_protocol=False)
        balanced_pred = clf.predict_feature_adapter(X_test, balanced_protocol=True)
        raw_acc = accuracy_score(y_test, raw_pred)
        acc = accuracy_score(y_test, balanced_pred)
        labels = list(clf.classes_)
        cm = confusion_matrix(y_test, balanced_pred, labels=labels)
        print("\nFrozen MIRepNet encoder + fast patient calibration head")
        print("  transformer weights updated: no")
        print("  target test labels used for adaptation: no")
        print("  completed test EEG used for unlabeled EA/feature scaling: yes")
        print(f"  ordinary decision threshold: {raw_acc * 100:.2f}%")
        print("  balanced-block threshold assumes the known 50/50 trial schedule")
        print("\nConfusion matrix (rows=true, cols=pred):")
        print("        " + "".join(f"{c:>8}" for c in labels))
        for c, row in zip(labels, cm):
            print(f"{c:>8}" + "".join(f"{v:>8}" for v in row))
        print("\n" + "=" * 35)
        print(f"Motor Imagery Test Accuracy: {acc * 100:.2f}%")
        print("=" * 35)
        return

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    fold_acc = []
    for tr_i, va_i in cv.split(X_train, y_train):
        m = make_decoder(decoder_kind).fit(X_train[tr_i], y_train[tr_i], groups=g_train[tr_i])
        m.set_reference(X_train[va_i])
        fold_acc.append(accuracy_score(y_train[va_i], m.predict(X_train[va_i])))
    fold_acc = np.array(fold_acc)
    print(f"\n[{decoder_label(decoder_kind)}]  5-fold CV on training pool: "
          f"{fold_acc.mean() * 100:.2f}% +/- {fold_acc.std() * 100:.2f}%")

    clf = make_decoder(decoder_kind).fit(X_train, y_train, groups=g_train)
    clf.set_reference(X_test)   # unsupervised alignment to the target (no labels)
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
    """Concatenate the raw EEG chunks captured during a live run (channels x samples,
    microvolts) into an MNE Raw and save as *_raw.fif — same format as the training
    files, so the session can be replayed and analyzed with the existing pipeline."""
    if not eeg_chunks:
        print("recording: no EEG captured, nothing saved")
        return
    data = np.hstack(eeg_chunks).astype(np.float64) / 1e6          # microvolts -> volts
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
    """Per-window decoder trace (time, raw log-odds, adaptive center, recentered margin,
    prediction, confidence, committed class) for offline analysis of skew/drift."""
    if not rows:
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("time,log_odds,center,margin,pred,conf,committed\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    print(f"decision log saved: {len(rows)} rows -> {out_path}")

def run_online(headless=False, decoder_kind=None, calibrate=None, live_maze=None):
    synthetic = select_device()
    if synthetic is None:
        return
    serial_port = None if synthetic else select_serial_port()
    if not synthetic and serial_port is None:
        return

    decoder_kind = decoder_kind or select_decoder()
    deep_mode = is_deep(decoder_kind)
    foundation_mode = is_foundation(decoder_kind)
    neural_mode = is_neural(decoder_kind)

    if foundation_mode and calibrate is None:
        print("\nMIRepNet live startup:")
        print("  1) With unlabeled calibration (recommended; alignment + boundary seed)")
        print("  2) Without calibration (use the saved model exactly as-is)")
        while True:
            try:
                raw = input("> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return
            if raw in ("", "1", "yes", "y", "calibrated", "calibration"):
                calibrate = True
                break
            if raw in ("2", "no", "n", "zero-shot", "none"):
                calibrate = False
                break
            print("enter 1 or 2")

    checkpoint = select_neural_source(decoder_kind) if neural_mode else None
    X_train = y_train = None

    if checkpoint is not None:
        module = neural_module(decoder_kind)
        device = MIREPNET_DEVICE if foundation_mode else DEEP_DEVICE
        decoder_class = mirepmod.MIRepNetDecoder if foundation_mode else deepmod.DeepDecoder
        clf = decoder_class.load(checkpoint, device=device)
        meta = clf.meta_
        passes = meta.get("epochs_run")
        pass_note = f"{passes} passes, " if passes is not None else ""
        print(f"\nLoaded {decoder_label(decoder_kind)} from {os.path.basename(checkpoint)}\n"
              f"  trained on {', '.join(meta['trained_on']) or 'unknown'} "
              f"({pass_note}best epoch {meta['best_epoch']}, "
              f"held-out loss {meta['validation_loss']:.4f})")
        n_times = clf.n_times_
        train_sfreq = float(module.SFREQ)
    else:
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

        if foundation_mode:
            warn_mirepnet_data_contract(train_files)

        print(f"\nTraining on {len(train_files)} file(s):")
        for f in train_files:
            print(f"  - {os.path.basename(f)}")

        parts = [load_epochs(decoder_kind, f) for f in train_files]
        X_train = np.concatenate([X for X, _ in parts])
        y_train = np.concatenate([y for _, y in parts])
        groups = np.concatenate([[i] * len(y) for i, (_, y) in enumerate(parts)])
        n_channels, n_times = X_train.shape[-2], X_train.shape[-1]
        shape_note = (f"{n_channels} ch x {n_times} samples" if neural_mode
                      else f"{X_train.shape[1]} bands x {n_channels} ch x {n_times} samples")

        clf = make_decoder(decoder_kind).fit(X_train, y_train, groups=groups)
        print(f"\nTrained {decoder_label(decoder_kind)} on {len(X_train)} epochs ({shape_note}).")
        if neural_mode:
            saved_to = clf.save(sources=train_files)
            checkpoint = saved_to
            print(f"saved for reuse: {os.path.relpath(saved_to, os.path.dirname(DATA_DIR))}")

        train_sfreq = float(mne.io.read_raw_fif(train_files[0], preload=False).info['sfreq'])

    # board EEG order -> the decoder's channel order; the same for every recording
    try:
        train_idx = [EEG_CHANNELS_MAPPING.index(name) for name in EEG_CHANNELS_TARGETS]
    except ValueError as e:
        print(f"channel mismatch between EEG_CHANNELS_MAPPING and EEG_CHANNELS_TARGETS: {e}")
        return

    ws = WebSocket()
    ws.start()
    smoother = Smoother(ws, conf_floor=CONF_FLOOR)

    bci = OpenBCI(interval=STRIDE_S, synthetic=synthetic, serial_port=serial_port).open()
    if bci.board is None:
        print(f"OpenBCI failed to open on {serial_port or 'the selected port'}.")
        print("Close OpenBCI GUI and every other program using that COM port, then retry. "
              "Also confirm the Cyton is powered in PC mode and the Daisy is attached.")
        ws.stop()
        return

    sfreq = float(BoardShim.get_sampling_rate(bci.board.board_id))
    if abs(sfreq - train_sfreq) > 0.5:
        print(f"warning: live sfreq={sfreq} differs from training sfreq={train_sfreq}; "
              "predictions may degrade")

    warmup_n = int(round(FILTER_WARMUP_S * sfreq))
    if neural_mode:
        # Neural epochs are defined at their own rate; the live buffer stays at board rate.
        module = neural_module(decoder_kind)
        window_n = int(round(module.EPOCH_S * sfreq))
        buffer_n = module.buffer_samples(sfreq, FILTER_WARMUP_S)
        make_window = lambda buf: clf.prepare_window(buf, sfreq)
    else:
        window_n = n_times
        buffer_n = window_n + warmup_n
        make_window = lambda buf: np.stack(
            [bandpass(buf, sfreq, l, h)[:, -window_n:] for (l, h) in FB_BANDS],
            axis=0)[np.newaxis, ...]
    buffer = np.zeros((len(train_idx), 0), dtype=np.float64)
    buffer_lock = threading.Lock()

    print(f"mode: {'HEADLESS (no GUI)' if headless else 'GUI'}, stride: {STRIDE_S}s, "
          f"classify window: {window_n} samples ({window_n / sfreq:.2f}s), "
          f"buffer: {buffer_n} samples ({buffer_n / sfreq:.2f}s)")

    # Classical decoders and MIRepNet calibrate their alignment reference and boundary.
    # Cardinal nets have no alignment reference, so their calibration is shorter.
    recenter = BoundaryRecenter()
    cal_seconds = CALIBRATION_SECONDS_DEEP if deep_mode else CALIBRATION_SECONDS
    X_cal = (record_calibration(bci, sfreq, train_idx, window_n, buffer_n, make_window,
                                cal_seconds, deep_mode)
             if calibrate is not False else None)
    if X_cal is not None and len(X_cal) >= 2:
        if not deep_mode:
            clf.set_reference(X_cal)
        pcal = clf.predict_proba(X_cal)
        scal = np.log(np.clip(pcal[:, 1], 1e-9, 1.0) / np.clip(pcal[:, 0], 1e-9, 1.0))
        recenter.seed(scal)
        mode = "adaptive" if recenter.adaptive else "FROZEN at calibration"
        what = "boundary seeded" if deep_mode else "aligned"
        print(f"{what} on {len(X_cal)} windows (boundary center={recenter.center:+.2f}, {mode})")
    elif calibrate is False:
        print("live calibration skipped; using the saved alignment and decision center=0.")
    else:
        print("calibration produced too little data; using the trained model as-is, center=0.")

    gui = None
    if not headless:
        from gui import GUI
        band_sep, band_abs_max = None, 1.0
        if getattr(clf, "has_bands", True) and X_train is not None:
            _, _, train_signal = clf.analyze(X_train)
            cls = clf.classes_
            sig0, sig1 = train_signal[y_train == cls[0]], train_signal[y_train == cls[1]]
            band_sep = np.abs(sig1.mean(0) - sig0.mean(0)) / (np.sqrt(0.5 * (sig0.var(0) + sig1.var(0))) + 1e-9)
            band_abs_max = float(np.percentile(np.abs(train_signal), 99)) or 1.0
        gui = GUI(bci, smoother, clf.classes_, sfreq,
                  band_sep=band_sep, band_abs_max=band_abs_max)

    prev_print_t = None

    # GUI and live-maze modes record raw EEG; every MIRepNet mode records each
    # decision window. A live maze needs both sources for a defensible audit.
    record_eeg = not headless or live_maze is not None
    record_decisions = not headless or foundation_mode
    rec_eeg, rec_rows = [], []
    rec_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def on_chunk(chunk):
        nonlocal buffer, prev_print_t
        eeg_all = chunk[bci.eeg, :]
        if eeg_all.shape[1] == 0:
            return
        if max(train_idx) >= eeg_all.shape[0]:
            print(f"(got {eeg_all.shape[0]} channels, need index {max(train_idx)})")
            return

        if record_eeg:
            rec_eeg.append(eeg_all.astype(np.float32))     # all EEG channels, microvolts

        eeg = eeg_all[train_idx, :].astype(np.float64, copy=False) / 1e6

        with buffer_lock:
            buffer = np.hstack([buffer, eeg])
            if buffer.shape[1] > buffer_n:
                buffer = buffer[:, -buffer_n:]
            if buffer.shape[1] < buffer_n:
                return
            buf = buffer.copy()

        window = make_window(buf)

        if headless:
            probs_raw = clf.predict_proba(window)[0]
        else:
            proba, score, band_sig = clf.analyze(window)
            probs_raw = proba[0]
            gui.push_decision(float(score[0]), band_sig[0])
        s = float(np.log(max(probs_raw[1], 1e-9) / max(probs_raw[0], 1e-9)))
        z = recenter.update(s)                            # recenter on the subject's neutral
        p1 = 1.0 / (1.0 + np.exp(-z))
        probs = np.array([1.0 - p1, p1])                  # recentered probabilities
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
        print(f"[{ts}  Δ{dt_ms:6.0f}ms]  {pred}  conf {conf*100:3.0f}% (ctr {recenter.center:+.1f})  [{breakdown}]  → {tag}{commit}")

        if record_decisions:
            rec_rows.append((ts, f"{s:.4f}", f"{recenter.center:.4f}", f"{z:.4f}",
                             int(pred), f"{conf:.4f}", int(final)))

    bci.callback = on_chunk
    bci.start()

    if record_eeg:
        print(f"\n📼 recording session → recordings/online_{rec_stamp}_raw.fif (+ _decisions.csv)")
    elif record_decisions:
        print(f"\nrecording MIRepNet decision windows → recordings/online_{rec_stamp}_decisions.xlsx")

    maze_process = None
    try:
        if live_maze is not None:
            maze_process = launch_live_maze(live_maze, checkpoint, calibrate)
        if headless:
            if maze_process is not None:
                print("\nlive maze running — close its window or press Ctrl-C here to stop.\n")
                while maze_process.poll() is None:
                    time.sleep(0.5)
            else:
                print("\nrunning headless — decisions broadcasting over websocket. Ctrl-C to stop.\n")
                while True:
                    time.sleep(1)
        else:
            gui.mainloop()
    except KeyboardInterrupt:
        print("\nStopping…")
    finally:
        if maze_process is not None and maze_process.poll() is None:
            maze_process.terminate()
            try:
                maze_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                maze_process.kill()
        bci.stop()
        bci.close()
        ws.stop()
        if record_eeg:
            save_online_recording(rec_eeg, sfreq, os.path.join(RECORDINGS_DIR, f"online_{rec_stamp}_raw.fif"))
        if record_decisions:
            save_decision_log(rec_rows, os.path.join(RECORDINGS_DIR, f"online_{rec_stamp}_decisions.csv"))
        if foundation_mode:
            try:
                from scripts.mirepnet_excel import export_online_workbook
                excel_path = export_online_workbook(
                    rec_rows,
                    os.path.join(RECORDINGS_DIR, f"online_{rec_stamp}_decisions.xlsx"),
                    metadata={
                        "mode": ("live maze" if live_maze is not None else
                                 ("headless" if headless else "GUI")),
                        "input_device": "synthetic" if synthetic else "OpenBCI USB",
                        "checkpoint": os.path.abspath(checkpoint) if checkpoint else "session model",
                        "decoder": decoder_label(decoder_kind),
                        "stride_seconds": STRIDE_S,
                        "calibration": "unlabeled" if calibrate is not False else "none",
                        "standard_maze": (live_maze or {}).get("maze_number", ""),
                    },
                )
                print(f"Excel decision report saved: {excel_path}")
            except Exception as error:
                print(f"Excel decision report failed: {error}")

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
    if argv and argv[0] == '--mirepnet-online':
        run_online(headless=False, decoder_kind='mirepnet')
        return
    if argv and argv[0] == '--mirepnet-headless':
        run_online(headless=True, decoder_kind='mirepnet')
        return
    if argv and argv[0] == '--mirepnet-live-maze':
        live_maze = select_live_maze()
        if live_maze is not None:
            run_online(headless=True, decoder_kind='mirepnet', live_maze=live_maze)
        return
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
