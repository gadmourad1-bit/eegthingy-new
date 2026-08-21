"""Deep-learning decoders for the live system.

Wraps the two benchmark-winning in-house networks (dl/) behind the same
fit / set_reference / predict_proba / analyze interface the classical online
decoders expose, so classifier/run.py can drive either family.

Unlike EA+FB-CSP and the Riemannian tangent decoder, these networks carry no
covariance alignment reference, so set_reference() is a no-op: there is nothing
for a calibration block to whiten. What they DO need is the benchmark's own
front end -- 4-40 Hz, 125 -> 128 Hz, common-average reference, 2.0 s epochs of
256 samples, per-channel standardisation -- which differs from the classical
8-30 Hz four-band causal path and is implemented here.
"""
import os
import sys

import mne
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, os.path.join(_ROOT, "dl", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from sklearn.base import BaseEstimator, ClassifierMixin

from benchmark.baselines import make_model
from benchmark.config import LOCAL_EXP4_15_CHANNELS, LOCAL_EXP4_PREPROCESSING
from benchmark.data import (_atlas_unit_positions, apply_channel_scaler,
                            fit_channel_scaler)
from benchmark.training import TrainConfig, fit_model

def resolve_device(preference="auto"):
    """Pick the training/inference device: CUDA, then Apple Metal (MPS), then CPU.

    These are PyTorch networks, so Apple-silicon acceleration is torch's MPS
    backend. (MLX is a separate framework and would require reimplementing the
    architectures, which would break parity with the benchmarked models.)
    """
    if preference and preference != "auto":
        return preference
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


def device_label(device):
    if device == "cuda":
        return f"CUDA ({torch.cuda.get_device_name(0)})"
    return "Apple Metal (MPS)" if device == "mps" else "CPU"


# menu key -> (registry name, label)
DEEP_MODELS = {
    "deep_compactdyn": ("cardinal_fbc_compactdyn_scale025_extended",
                        "CardinalFBC + compact dynamics (32k params, grid rank 1)"),
    "deep_sinc": ("cardinal_dynamics_sinc_extended",
                  "Cardinal Sinc dynamics (13.5k params, best on local data)"),
}

# the online montage in acquisition order, and its modern standard_1005 names
SOURCE_CHANNELS = ("Cz", "Pz", "C3", "C4", "T5", "T6", "Fz",
                   "F7", "F8", "F3", "F4", "T3", "T4", "P3", "P4")
CHANNELS = LOCAL_EXP4_15_CHANNELS          # T5->P7, T6->P8, T3->T7, T4->T8
FMIN = float(LOCAL_EXP4_PREPROCESSING["fmin_hz"])
FMAX = float(LOCAL_EXP4_PREPROCESSING["fmax_hz"])
SFREQ = float(LOCAL_EXP4_PREPROCESSING["sfreq_hz"])
N_TIMES = int(LOCAL_EXP4_PREPROCESSING["n_times"])
EPOCH_S = float(LOCAL_EXP4_PREPROCESSING["tmax_seconds_exclusive"])
RESAMPLE_PAD = 100      # see prepare_window(); fixed so fidelity is buffer-length independent
_FILT = dict(l_trans_bandwidth=float(LOCAL_EXP4_PREPROCESSING["l_trans_bandwidth_hz"]),
             h_trans_bandwidth=float(LOCAL_EXP4_PREPROCESSING["h_trans_bandwidth_hz"]))


def _bandpass(data, sfreq):
    """Benchmark front-end filter: zero-phase FIR over the whole array."""
    return mne.filter.filter_data(data, sfreq=sfreq, l_freq=FMIN, h_freq=FMAX,
                                  method="fir", phase="zero", fir_design="firwin",
                                  pad="reflect_limited", verbose=False, **_FILT)


def process_data(file_name, target_id_dict):
    """Load one training recording under the benchmark's local profile.

    Returns X (n_epochs, 15, 256) float32 in volts and y (class codes), matching
    dl/src/benchmark/data.py: filter -> resample -> CAR -> epoch -> demean.
    """
    print(f"\n--- Loading: {file_name} ---")
    raw = mne.io.read_raw_fif(file_name, preload=True, verbose=False)
    raw.pick(list(SOURCE_CHANNELS))
    raw.rename_channels(dict(zip(SOURCE_CHANNELS, CHANNELS)))
    raw.annotations.description = np.array(
        [str(d).strip().lower() for d in raw.annotations.description])

    event_id = {}
    for desc in set(raw.annotations.description):
        for prefix, code in target_id_dict.items():
            p = prefix.lower()
            if desc == p or desc.startswith(p + "/"):
                event_id[str(desc)] = code
                break
    if not event_id:
        raise ValueError(f"No annotations in {file_name} matched {target_id_dict}")

    raw.filter(FMIN, FMAX, picks="eeg", method="fir", phase="zero",
               fir_design="firwin", pad="reflect_limited", verbose="ERROR", **_FILT)
    raw.resample(SFREQ, npad="auto", verbose="ERROR")
    raw.set_eeg_reference(ref_channels="average", projection=False, verbose="ERROR")

    events, used = mne.events_from_annotations(raw, event_id=event_id, verbose=False)
    epochs = mne.Epochs(raw, events, event_id=used, tmin=0.0,
                        tmax=EPOCH_S - 1.0 / SFREQ, baseline=None, preload=True,
                        proj=False, reject_by_annotation=False, verbose="ERROR")
    X = epochs.get_data(copy=True).astype(np.float32)[:, :, :N_TIMES]
    X = X - X.mean(axis=2, keepdims=True)
    y = epochs.events[:, -1]
    print(f"Created {len(y)} epochs ({X.shape[1]} ch x {X.shape[2]} samples) "
          f"for classes {sorted(set(y.tolist()))}")
    return np.ascontiguousarray(X), y


class DeepDecoder(BaseEstimator, ClassifierMixin):
    """Online wrapper around a benchmark network.

    X is (n_epochs, n_channels, n_times) volts under the benchmark front end.
    The net is trained here from the selected recordings (no checkpoints ship
    with the results bundle), using the frozen benchmark recipe with a held-out
    split for early stopping.
    """

    has_bands = False

    def __init__(self, kind, epochs=None, seed=7, device="auto", val_fraction=0.2):
        self.kind = kind
        self.epochs = epochs
        self.seed = seed
        self.device = device
        self.val_fraction = val_fraction

    def fit(self, X, y, groups=None):
        name = DEEP_MODELS[self.kind][0]
        self.classes_ = np.unique(y)
        y_idx = np.searchsorted(self.classes_, y).astype(np.int64)
        X = np.asarray(X, dtype=np.float32)

        self.mean_, self.std_ = fit_channel_scaler(X, CHANNELS)
        Xs = apply_channel_scaler(X, self.mean_, self.std_)
        self.positions_ = _atlas_unit_positions(CHANNELS)

        rng = np.random.default_rng(self.seed)
        val = np.zeros(len(y_idx), dtype=bool)
        for c in np.unique(y_idx):
            idx = np.flatnonzero(y_idx == c)
            rng.shuffle(idx)
            val[idx[:max(1, int(round(self.val_fraction * len(idx))))]] = True

        train_device = resolve_device(self.device)
        cfg = TrainConfig(seed=self.seed, device=train_device)
        if self.epochs is not None:
            cfg = TrainConfig(**{**cfg.__dict__, "epochs": int(self.epochs)})

        model = make_model(name, n_channels=X.shape[1], n_outputs=len(self.classes_),
                           n_times=X.shape[2], sfreq=SFREQ, channel_names=CHANNELS,
                           channel_positions=torch.as_tensor(self.positions_,
                                                             dtype=torch.float32))
        n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"training {name} ({n_par:,} parameters) on {device_label(train_device)}: "
              f"{int((~val).sum())} epochs fitted, {int(val.sum())} held out, "
              f"up to {cfg.epochs} passes…")
        history = fit_model(model, Xs[~val], y_idx[~val], Xs[val], y_idx[val],
                            self.positions_, config=cfg)
        self.device_ = torch.device(train_device)
        self.model_ = model.to(self.device_).eval()
        self.positions_t_ = torch.as_tensor(self.positions_, dtype=torch.float32,
                                            device=self.device_)
        self.n_parameters_ = n_par
        self.history_ = history
        print(f"trained: {history['epochs_run']} passes run, best epoch "
              f"{history['best_epoch']}, held-out loss "
              f"{history['best_validation_loss']:.4f} ({history['fit_seconds']:.0f}s)")
        return self

    def set_reference(self, X_cal):
        """No-op: these networks have no covariance alignment reference."""
        return self

    def _logits(self, X):
        Xs = apply_channel_scaler(np.asarray(X, dtype=np.float32), self.mean_, self.std_)
        with torch.no_grad():
            out = self.model_(torch.as_tensor(Xs, dtype=torch.float32, device=self.device_),
                              self.positions_t_)
        return out.detach().float().cpu().numpy().astype(np.float64)

    def predict_proba(self, X):
        z = self._logits(X)
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self._logits(X), axis=1)]

    def analyze(self, X):
        """(probabilities, signed score, per-band contributions).

        Band contributions are zero: there is no filter bank to attribute the
        score to, so the GUI's band panel stays flat for these decoders.
        """
        from config import FB_BANDS
        proba = self.predict_proba(X)
        score = np.log(np.clip(proba[:, 1], 1e-12, 1.0)
                       / np.clip(proba[:, 0], 1e-12, 1.0))
        return proba, score, np.zeros((len(proba), len(FB_BANDS)))

    def prepare_window(self, buf, sfreq):
        """Raw live buffer (channels x samples, volts) -> (1, 15, 256) window."""
        x = _bandpass(np.asarray(buf, dtype=np.float64), sfreq)
        if abs(sfreq - SFREQ) > 1e-6:
            x = mne.filter.resample(x, up=SFREQ, down=sfreq, npad=RESAMPLE_PAD,
                                    verbose=False)
        x = x - x.mean(axis=0, keepdims=True)
        x = x[:, -N_TIMES:]
        x = x - x.mean(axis=1, keepdims=True)
        return x[np.newaxis, ...].astype(np.float32)


def buffer_samples(sfreq, warmup_s):
    """Live buffer length: one epoch plus filter warm-up, at the source rate."""
    return int(round((EPOCH_S + warmup_s) * sfreq))
