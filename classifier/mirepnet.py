"""MIRepNet foundation-model decoder for the live EEG system.

This module implements the downstream architecture and preprocessing contract
released by Liu et al. (Knowledge-Based Systems, 2026).  The pretrained tensors
come from the official MIT-licensed checkpoint converted by Braindecode:
``braindecode/mirepnet-pretrained``.

Local Exp4 compatibility is explicit rather than implicit:

* 8--30 Hz FIR filtering and 125 -> 250 Hz resampling;
* Euclidean Alignment (EA) on the *observed* 15-channel domain;
* inverse-distance interpolation to MIRepNet's fixed 45-channel template;
* 4-second / 1,000-sample inputs.

The official source applies EA before channel interpolation.  Keeping that
order is important: interpolating first creates a rank-deficient 45-channel
covariance because this headset observes only 15 independent channels.
"""

from __future__ import annotations

import copy
import hashlib
import math
import os
import random
import re
import time
from datetime import datetime

import mne
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config import EEG_CHANNELS_TARGETS, EPOCH_REJECT, FB_TRANS, MIREPNET_WINDOW_MODE


MODEL_ID = "braindecode/mirepnet-pretrained"
MODEL_REVISION = "857f1e3642976be9f2ca6883e5508c3f7c91d86f"
MODEL_FILE = "model.safetensors"
MODEL_SHA256 = "a3a7eb0f100b703cd40ee2ed634dbc04d1f6ae04931404299dc2f81c4175559b"

SFREQ = 250.0
N_TIMES = 1000
EPOCH_S = N_TIMES / SFREQ
FMIN, FMAX = 8.0, 30.0
RESAMPLE_PAD = 100

CHANNELS = [
    "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8",
    "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8",
    "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",
]

# The headset uses legacy 10-20 temporal names; these are their modern aliases.
SOURCE_CHANNELS = [
    {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}.get(ch, ch).upper()
    for ch in EEG_CHANNELS_TARGETS
]

# Exact 2-D coordinates used by the official MIRepNet interpolation pipeline.
_ROWS = {
    "F": {"F7": -0.5, "F5": -0.4, "F3": -0.3, "F1": -0.15, "FZ": 0.0,
          "F2": 0.15, "F4": 0.3, "F6": 0.4, "F8": 0.5},
    "FC": {"FT7": -0.6, "FC5": -0.5, "FC3": -0.4, "FC1": -0.2,
           "FCZ": 0.0, "FC2": 0.2, "FC4": 0.4, "FC6": 0.5, "FT8": 0.6},
    "C": {"T7": -1.0, "C5": -0.7, "C3": -0.4, "C1": -0.2, "CZ": 0.0,
          "C2": 0.2, "C4": 0.4, "C6": 0.7, "T8": 1.0},
    "CP": {"TP7": -0.6, "CP5": -0.5, "CP3": -0.4, "CP1": -0.2,
           "CPZ": 0.0, "CP2": 0.2, "CP4": 0.4, "CP6": 0.5, "TP8": 0.6},
    "P": {"P7": -0.5, "P5": -0.4, "P3": -0.3, "P1": -0.15, "PZ": 0.0,
          "P2": 0.15, "P4": 0.3, "P6": 0.4, "P8": 0.5},
}
_Y = {"F": 0.7, "FC": 0.6, "C": 0.5, "CP": 0.4, "P": 0.3}
CHANNEL_POSITIONS = {
    name: (x, _Y[row]) for row, entries in _ROWS.items() for name, x in entries.items()
}

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
CHECKPOINT_DIR = os.path.join(_ROOT, "models", "mirepnet")


def resolve_device(preference="auto"):
    preference = str(preference or "auto").lower()
    if preference != "auto":
        requested = torch.device(preference)
        if requested.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "MIRepNet was configured for CUDA, but this PyTorch installation "
                "cannot see a CUDA device. Use --device auto or install a CUDA build of PyTorch."
            )
        if requested.type == "mps" and not (
                torch.backends.mps.is_available() and torch.backends.mps.is_built()):
            raise RuntimeError(
                "MIRepNet was configured for MPS, but this PyTorch installation cannot use MPS."
            )
        if requested.type not in {"cpu", "cuda", "mps"}:
            raise ValueError(f"unsupported MIRepNet device {preference!r}")
        return str(requested)
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


def device_label(device):
    device = torch.device(device)
    if device.type == "cuda":
        index = torch.cuda.current_device() if device.index is None else device.index
        return f"CUDA:{index} ({torch.cuda.get_device_name(index)})"
    return "Apple Metal (MPS)" if device.type == "mps" else "CPU"


def interpolation_matrix(source_channels=SOURCE_CHANNELS):
    """Official inverse-distance template mapping, shape (45, source channels)."""
    source_channels = [str(ch).upper() for ch in source_channels]
    missing = [ch for ch in source_channels if ch not in CHANNEL_POSITIONS]
    if missing:
        raise ValueError(f"MIRepNet has no interpolation coordinates for {missing}")
    source_pos = np.asarray([CHANNEL_POSITIONS[ch] for ch in source_channels], dtype=np.float64)
    W = np.zeros((len(CHANNELS), len(source_channels)), dtype=np.float64)
    for row, channel in enumerate(CHANNELS):
        if channel in source_channels:
            W[row, source_channels.index(channel)] = 1.0
        else:
            pos = np.asarray(CHANNEL_POSITIONS[channel], dtype=np.float64)
            distance = np.linalg.norm(source_pos - pos, axis=1)
            weights = 1.0 / (distance + 1e-6)
            W[row] = weights / weights.sum()
    return W


def _whitener(X):
    """EA reference on observed channels, following the official np.cov method."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 3 or len(X) == 0:
        raise ValueError("EA expects non-empty (trials, channels, samples) data")
    centered = X - X.mean(axis=2, keepdims=True)
    cov = np.einsum("nct,ndt->ncd", centered, centered) / max(1, X.shape[2] - 1)
    reference = cov.mean(axis=0)
    values, vectors = np.linalg.eigh(reference)
    floor = max(float(values.max()) * 1e-7, 1e-15)
    values = np.clip(values, floor, None)
    return (vectors * values**-0.5) @ vectors.T


def _align(X, whitener):
    return np.einsum("cd,ndt->nct", whitener, np.asarray(X, dtype=np.float64))


def domain_z(features):
    """Standardize a feature batch without labels (target-domain adaptation)."""
    features = np.asarray(features, dtype=np.float32)
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    return (features - mean) / np.maximum(scale, 1e-5)


def _bandpass(data, sfreq):
    return mne.filter.filter_data(
        np.asarray(data, dtype=np.float64), sfreq=sfreq, l_freq=FMIN, h_freq=FMAX,
        method="fir", phase="zero", fir_design="firwin", pad="reflect_limited",
        verbose=False, **FB_TRANS,
    )


def process_data(file_name, target_id_dict, window_mode="plan-task"):
    """Load one FIF as unaligned 15-channel, 4-second MIRepNet trials.

    ``plan-task`` preserves the original bridge (2 s plan + 2 s task).
    ``task-repeat`` uses the audited Exp4 0--2 s task interval twice, retaining
    MIRepNet's required 1,000-sample input without including planning activity.
    """
    print(f"\n--- Loading MIRepNet input: {file_name} ---")
    raw = mne.io.read_raw_fif(file_name, preload=True, verbose=False)
    raw.pick(list(EEG_CHANNELS_TARGETS))
    raw.rename_channels(dict(zip(EEG_CHANNELS_TARGETS, SOURCE_CHANNELS)))
    raw.annotations.description = np.asarray(
        [str(desc).strip().lower() for desc in raw.annotations.description]
    )

    # Imported lazily because classifier/run.py imports this module.
    from run import build_event_id

    event_id = build_event_id(raw, target_id_dict)
    if not event_id:
        raise ValueError(f"No annotations in {file_name} matched {target_id_dict}")

    raw.filter(FMIN, FMAX, picks="eeg", method="fir", phase="zero",
               fir_design="firwin", pad="reflect_limited", verbose="ERROR", **FB_TRANS)
    raw.resample(SFREQ, npad="auto", verbose="ERROR")
    events, used = mne.events_from_annotations(raw, event_id=event_id, verbose=False)

    if window_mode not in {"plan-task", "task-repeat"}:
        raise ValueError(f"unknown MIRepNet window mode {window_mode!r}")
    tmin = -2.0 if window_mode == "plan-task" else 0.0
    tmax = 2.0 - 1.0 / SFREQ
    epochs = mne.Epochs(
        raw, events, event_id=used, tmin=tmin, tmax=tmax,
        baseline=None, preload=True, proj=False, reject_by_annotation=False,
        on_missing="warn", verbose="ERROR",
    )
    X = epochs.get_data(copy=True).astype(np.float32)
    if window_mode == "task-repeat":
        X = np.concatenate([X, X], axis=2)
    X = X[:, :, :N_TIMES]
    X -= X.mean(axis=2, keepdims=True)
    y = epochs.events[:, -1]

    peak_to_peak = np.ptp(X, axis=2).max(axis=1)
    keep = peak_to_peak < float(EPOCH_REJECT["eeg"])
    X, y = np.ascontiguousarray(X[keep]), y[keep]
    print(f"Created {len(y)} MIRepNet trials ({int(keep.sum())}/{len(keep)} kept; "
          f"{X.shape[1]} observed ch -> 45 template ch; {N_TIMES} samples @ {SFREQ:g} Hz; "
          f"window={window_mode})")
    return X, y


class _MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, dropout=0.5):
        super().__init__()
        self.embed_dim, self.num_heads = embed_dim, num_heads
        self.keys = nn.Linear(embed_dim, embed_dim)
        self.queries = nn.Linear(embed_dim, embed_dim)
        self.values = nn.Linear(embed_dim, embed_dim)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch, tokens, _ = x.shape
        head_dim = self.embed_dim // self.num_heads
        reshape = lambda z: z.reshape(batch, tokens, self.num_heads, head_dim).transpose(1, 2)
        q, k, v = reshape(self.queries(x)), reshape(self.keys(x)), reshape(self.values(x))
        attention = torch.softmax((q @ k.transpose(-2, -1)) / math.sqrt(self.embed_dim), dim=-1)
        out = self.att_drop(attention) @ v
        return self.projection(out.transpose(1, 2).reshape(batch, tokens, self.embed_dim))


class _ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)


def _transformer_block(embed_dim=256):
    return nn.Sequential(
        _ResidualAdd(nn.Sequential(
            nn.LayerNorm(embed_dim), _MultiHeadAttention(embed_dim, 8, 0.5), nn.Dropout(0.5)
        )),
        _ResidualAdd(nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4), nn.GELU(), nn.Dropout(0.5),
                nn.Linear(embed_dim * 4, embed_dim),
            ), nn.Dropout(0.5),
        )),
    )


class _PatchEmbedding(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(1, 25))
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(45, 1))
        self.bn = nn.BatchNorm2d(128)
        self.elu = nn.ELU()
        self.pool = nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15))
        self.dropout = nn.Dropout(0.5)
        self.projection = nn.Conv2d(128, embed_dim, kernel_size=(1, 1))

    def forward(self, x):
        x = self.dropout(self.pool(self.elu(self.bn(self.conv2(self.conv1(x[:, None]))))))
        x = self.projection(x)
        return x.flatten(2).transpose(1, 2)


class MIRepNetModel(nn.Module):
    """Downstream MIRepNet architecture with Braindecode-compatible key names."""
    def __init__(self, n_outputs=2, embed_dim=256, depth=6):
        super().__init__()
        self.embedding = _PatchEmbedding(embed_dim)
        self.transformer = nn.Sequential(*[_transformer_block(embed_dim) for _ in range(depth)])
        self.final_layer = nn.Linear(embed_dim, n_outputs)

    def forward(self, x, return_features=False):
        features = self.transformer(self.embedding(x)).mean(dim=1)
        return features if return_features else self.final_layer(features)


_PRETRAINED_STATE = None


def pretrained_state(path=None):
    """Load and verify the official pretrained state, caching it in memory."""
    global _PRETRAINED_STATE
    if _PRETRAINED_STATE is not None and path is None:
        return _PRETRAINED_STATE
    if path is None:
        path = os.environ.get("MIREPNET_WEIGHTS") or hf_hub_download(
            repo_id=MODEL_ID, filename=MODEL_FILE, revision=MODEL_REVISION
        )
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    if digest != MODEL_SHA256:
        raise ValueError(f"MIRepNet weight SHA-256 mismatch: {digest} != {MODEL_SHA256}")
    state = load_file(path, device="cpu")
    if path == os.environ.get("MIREPNET_WEIGHTS"):
        return state
    _PRETRAINED_STATE = state
    return state


def _validation_mask(y, groups, fraction, seed):
    """Hold out complete recordings when possible; fall back to stratified trials."""
    y, groups = np.asarray(y), None if groups is None else np.asarray(groups)
    rng = np.random.default_rng(seed)
    mask = np.zeros(len(y), dtype=bool)
    if groups is not None and len(np.unique(groups)) >= 3:
        unique = np.unique(groups)
        rng.shuffle(unique)
        n_val = max(1, int(round(fraction * len(unique))))
        mask = np.isin(groups, unique[:n_val])
        if set(np.unique(y[mask])) == set(np.unique(y)):
            return mask
    mask[:] = False
    for label in np.unique(y):
        indices = np.flatnonzero(y == label)
        rng.shuffle(indices)
        mask[indices[:max(1, int(round(fraction * len(indices))))]] = True
    return mask


class MIRepNetDecoder(BaseEstimator, ClassifierMixin):
    has_bands = False

    def __init__(self, epochs=20, seed=7, device="auto", val_fraction=0.15,
                 batch_size=16, encoder_lr=1e-4, head_lr=1e-3,
                 weight_decay=1e-4, patience=5, freeze_epochs=2,
                 window_mode=MIREPNET_WINDOW_MODE, initial_checkpoint=None):
        self.epochs = epochs
        self.seed = seed
        self.device = device
        self.val_fraction = val_fraction
        self.batch_size = batch_size
        self.encoder_lr = encoder_lr
        self.head_lr = head_lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.freeze_epochs = freeze_epochs
        self.window_mode = window_mode
        self.initial_checkpoint = initial_checkpoint

    def _domain_transform(self, X, whitener):
        aligned = _align(X, whitener)
        return np.einsum("kc,nct->nkt", self.interpolation_, aligned).astype(np.float32)

    def fit(self, X, y, groups=None):
        X, y = np.asarray(X, dtype=np.float32), np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError(f"live MIRepNet expects two classes, got {self.classes_.tolist()}")
        if X.shape[1:] != (len(SOURCE_CHANNELS), N_TIMES):
            raise ValueError(f"expected (*, {len(SOURCE_CHANNELS)}, {N_TIMES}), got {X.shape}")

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        torch.use_deterministic_algorithms(True, warn_only=True)

        self.interpolation_ = interpolation_matrix()
        self.ref_white_ = _whitener(X)
        Xa = np.empty((len(X), len(CHANNELS), N_TIMES), dtype=np.float32)
        if groups is None:
            Xa[:] = self._domain_transform(X, self.ref_white_)
        else:
            groups = np.asarray(groups)
            for group in np.unique(groups):
                idx = groups == group
                Xa[idx] = self._domain_transform(X[idx], _whitener(X[idx]))

        validation = _validation_mask(y, groups, self.val_fraction, self.seed)
        if not validation.any() or (~validation).sum() < 2:
            raise ValueError("MIRepNet could not construct a non-empty train/validation split")
        y_index = np.searchsorted(self.classes_, y).astype(np.int64)

        model = MIRepNetModel(n_outputs=len(self.classes_))
        if self.initial_checkpoint:
            blob = torch.load(self.initial_checkpoint, map_location="cpu", weights_only=False)
            if blob.get("format") != "eeg-online-mirepnet-v1":
                raise ValueError(f"{self.initial_checkpoint} is not an online MIRepNet checkpoint")
            if tuple(blob.get("source_channels", ())) != tuple(SOURCE_CHANNELS):
                raise ValueError("MIRepNet checkpoint source montage does not match this project")
            old_classes = np.asarray(blob.get("classes", []))
            if not np.array_equal(old_classes, self.classes_):
                raise ValueError(
                    f"checkpoint classes {old_classes.tolist()} do not match selected data "
                    f"{self.classes_.tolist()}"
                )
            model.load_state_dict(blob["state_dict"], strict=True)
            initialization = os.path.abspath(self.initial_checkpoint)
        else:
            state = pretrained_state()
            encoder_state = {k: v for k, v in state.items() if not k.startswith("final_layer.")}
            incompatible = model.load_state_dict(encoder_state, strict=False)
            if (set(incompatible.missing_keys) != {"final_layer.weight", "final_layer.bias"}
                    or incompatible.unexpected_keys):
                raise RuntimeError(f"pretrained MIRepNet state is incompatible: {incompatible}")
            initialization = MODEL_ID

        device = torch.device(resolve_device(self.device))
        model.to(device)
        train_ds = TensorDataset(torch.from_numpy(Xa[~validation]), torch.from_numpy(y_index[~validation]))
        val_ds = TensorDataset(torch.from_numpy(Xa[validation]), torch.from_numpy(y_index[validation]))
        generator = torch.Generator().manual_seed(self.seed)
        pin_memory = device.type == "cuda"
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True,
                      generator=generator, num_workers=0,
                      pin_memory=pin_memory)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False,
                    num_workers=0, pin_memory=pin_memory)

        encoder_params = [p for name, p in model.named_parameters() if not name.startswith("final_layer.")]
        optimizer = torch.optim.AdamW([
            {"params": encoder_params, "lr": self.encoder_lr},
            {"params": model.final_layer.parameters(), "lr": self.head_lr},
        ], weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, self.epochs))
        criterion = nn.CrossEntropyLoss()

        best_state, best_loss, best_epoch, stale = None, float("inf"), -1, 0
        history, started = [], time.perf_counter()
        print(f"fine-tuning MIRepNet (5.14M params) on {device_label(device)}: "
              f"{len(train_ds)} train / {len(val_ds)} recording-held-out validation trials")
        for epoch in range(int(self.epochs)):
            frozen = epoch < int(self.freeze_epochs)
            for parameter in encoder_params:
                parameter.requires_grad_(not frozen)
            model.train()
            train_loss, train_correct, train_count = 0.0, 0, 0
            for xb, yb in train_loader:
                xb = xb.to(device, non_blocking=pin_memory)
                yb = yb.to(device, non_blocking=pin_memory)
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += float(loss.detach()) * len(yb)
                train_correct += int((logits.argmax(1) == yb).sum())
                train_count += len(yb)

            model.eval()
            val_loss, val_correct, val_count = 0.0, 0, 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device, non_blocking=pin_memory)
                    yb = yb.to(device, non_blocking=pin_memory)
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    val_loss += float(loss) * len(yb)
                    val_correct += int((logits.argmax(1) == yb).sum())
                    val_count += len(yb)
            scheduler.step()
            row = {
                "epoch": epoch + 1,
                "train_loss": train_loss / train_count,
                "train_accuracy": train_correct / train_count,
                "validation_loss": val_loss / val_count,
                "validation_accuracy": val_correct / val_count,
                "encoder_frozen": frozen,
            }
            history.append(row)
            print(f"  epoch {epoch + 1:02d}: train {row['train_accuracy'] * 100:5.1f}% "
                  f"val {row['validation_accuracy'] * 100:5.1f}% loss {row['validation_loss']:.4f}" +
                  (" [head warm-up]" if frozen else ""))
            if row["validation_loss"] < best_loss - 1e-5:
                best_loss, best_epoch = row["validation_loss"], epoch + 1
                best_state, stale = copy.deepcopy(model.state_dict()), 0
            else:
                stale += 1
                if stale >= int(self.patience) and not frozen:
                    break

        model.load_state_dict(best_state)
        self.model_ = model.to(device).eval()
        self.device_ = device
        self.n_times_ = N_TIMES
        self.n_parameters_ = sum(p.numel() for p in model.parameters())
        self.history_ = history
        self.best_epoch_ = best_epoch
        self.best_validation_loss_ = best_loss
        self.fit_seconds_ = time.perf_counter() - started
        self.initialized_from_ = initialization
        print(f"MIRepNet selected epoch {best_epoch}; validation loss {best_loss:.4f}; "
              f"fit time {self.fit_seconds_:.0f}s")
        return self

    def set_reference(self, X_cal):
        """Fit target-domain EA from unlabeled calibration/test windows."""
        self.ref_white_ = _whitener(np.asarray(X_cal, dtype=np.float32))
        return self

    def _logits(self, X):
        Xt = self._domain_transform(np.asarray(X, dtype=np.float32), self.ref_white_)
        outputs = []
        with torch.no_grad():
            for start in range(0, len(Xt), self.batch_size):
                batch = torch.from_numpy(Xt[start:start + self.batch_size]).to(self.device_)
                outputs.append(self.model_(batch).detach().cpu())
        return torch.cat(outputs).numpy().astype(np.float64)

    def predict_proba(self, X):
        logits = self._logits(X)
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self._logits(X).argmax(axis=1)]

    def extract_features(self, X):
        """Return the 256-D encoder representation after the active EA reference."""
        Xt = self._domain_transform(np.asarray(X, dtype=np.float32), self.ref_white_)
        outputs = []
        with torch.no_grad():
            for start in range(0, len(Xt), self.batch_size):
                batch = torch.from_numpy(Xt[start:start + self.batch_size]).to(self.device_)
                outputs.append(self.model_(batch, return_features=True).detach().cpu())
        return torch.cat(outputs).numpy().astype(np.float32)

    def fit_feature_adapter(self, X_cal, y_cal):
        """Fit the fast patient-specific head while leaving MIRepNet frozen."""
        self.set_reference(X_cal)
        features = domain_z(self.extract_features(X_cal))
        self.feature_adapter_ = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.01, max_iter=2000, class_weight="balanced"),
        ).fit(features, np.asarray(y_cal))
        return self

    def predict_feature_adapter(self, X, balanced_protocol=False):
        """Predict an offline target batch with unlabeled EA/feature adaptation.

        ``balanced_protocol`` enforces the experiment's known 50/50 left/right
        trial schedule. It is appropriate for a completed balanced evaluation
        block, not an unconstrained live stream.
        """
        if not hasattr(self, "feature_adapter_"):
            raise ValueError("fit_feature_adapter must be called before adapter prediction")
        self.set_reference(X)
        features = domain_z(self.extract_features(X))
        if not balanced_protocol:
            return self.feature_adapter_.predict(features)
        scores = self.feature_adapter_.decision_function(features)
        classes = self.feature_adapter_.named_steps["logisticregression"].classes_
        prediction = np.full(len(scores), classes[0])
        prediction[np.argsort(scores)[-(len(scores) // 2):]] = classes[1]
        return prediction

    def predict_feature_adapter_proba(self, X):
        """Return patient-head probabilities for a completed offline test batch."""
        if not hasattr(self, "feature_adapter_"):
            raise ValueError("fit_feature_adapter must be called before adapter prediction")
        self.set_reference(X)
        features = domain_z(self.extract_features(X))
        return self.feature_adapter_.predict_proba(features)

    def analyze(self, X):
        from config import FB_BANDS
        proba = self.predict_proba(X)
        score = np.log(np.clip(proba[:, 1], 1e-12, 1.0) /
                       np.clip(proba[:, 0], 1e-12, 1.0))
        return proba, score, np.zeros((len(proba), len(FB_BANDS)))

    def prepare_window(self, buf, sfreq):
        x = _bandpass(buf, sfreq)
        if abs(float(sfreq) - SFREQ) > 1e-6:
            x = mne.filter.resample(x, up=SFREQ, down=float(sfreq), npad=RESAMPLE_PAD,
                                    verbose=False)
        if self.window_mode == "task-repeat":
            task = x[:, -(N_TIMES // 2):]
            x = np.concatenate([task, task], axis=1)
        else:
            x = x[:, -N_TIMES:]
        if x.shape[1] != N_TIMES:
            raise ValueError(f"MIRepNet live window has {x.shape[1]} samples, needs {N_TIMES}")
        x -= x.mean(axis=1, keepdims=True)
        return x[np.newaxis].astype(np.float32)

    def save(self, path=None, sources=()):
        if path is None:
            path = os.path.join(CHECKPOINT_DIR, f"mirepnet__{_tag(sources)}.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "format": "eeg-online-mirepnet-v1",
            "state_dict": {k: v.detach().cpu() for k, v in self.model_.state_dict().items()},
            "classes": np.asarray(self.classes_),
            "source_channels": SOURCE_CHANNELS,
            "template_channels": CHANNELS,
            "interpolation": self.interpolation_,
            "reference_whitener": self.ref_white_,
            "n_parameters": int(self.n_parameters_),
            "best_epoch": int(self.best_epoch_),
            "validation_loss": float(self.best_validation_loss_),
            "fit_seconds": float(self.fit_seconds_),
            "window_mode": self.window_mode,
            "trained_on": [os.path.basename(item) for item in sources],
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "pretrained_repo": MODEL_ID,
            "pretrained_revision": MODEL_REVISION,
            "pretrained_sha256": MODEL_SHA256,
            "initialized_from": getattr(self, "initialized_from_", MODEL_ID),
            "training_history": list(getattr(self, "history_", [])),
        }, path)
        return path

    @classmethod
    def load(cls, path, device="auto"):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if blob.get("format") != "eeg-online-mirepnet-v1":
            raise ValueError(f"{path} is not an online MIRepNet checkpoint")
        if tuple(blob["source_channels"]) != tuple(SOURCE_CHANNELS):
            raise ValueError("MIRepNet checkpoint source montage does not match this project")
        self = cls(device=device, window_mode=blob.get("window_mode", MIREPNET_WINDOW_MODE))
        self.classes_ = np.asarray(blob["classes"])
        self.interpolation_ = np.asarray(blob["interpolation"])
        self.ref_white_ = np.asarray(blob["reference_whitener"])
        self.n_times_, self.n_parameters_ = N_TIMES, int(blob["n_parameters"])
        self.best_epoch_ = int(blob["best_epoch"])
        self.best_validation_loss_ = float(blob["validation_loss"])
        self.fit_seconds_ = float(blob.get("fit_seconds", 0.0))
        model = MIRepNetModel(n_outputs=len(self.classes_))
        model.load_state_dict(blob["state_dict"])
        self.device_ = torch.device(resolve_device(device))
        self.model_ = model.to(self.device_).eval()
        self.meta_ = blob
        return self


def buffer_samples(sfreq, warmup_s, window_mode=MIREPNET_WINDOW_MODE):
    signal_seconds = EPOCH_S / 2.0 if window_mode == "task-repeat" else EPOCH_S
    return int(round((signal_seconds + warmup_s) * float(sfreq)))


def _tag(sources):
    names = sorted(os.path.basename(path) for path in sources)
    subjects = sorted({int(match.group(1)) for name in names
                       if (match := re.search(r"subject(\d+)", name))})
    digest = hashlib.sha1("|".join(names).encode()).hexdigest()[:8]
    return f"{len(subjects)}subjects_{len(names)}runs_{digest}" if names else "session"


def list_checkpoints(kind="mirepnet"):
    if not os.path.isdir(CHECKPOINT_DIR):
        return []
    paths = [os.path.join(CHECKPOINT_DIR, name) for name in os.listdir(CHECKPOINT_DIR)
             if name.startswith("mirepnet__") and name.endswith(".pt")]
    return sorted(paths, key=os.path.getmtime, reverse=True)


def describe_checkpoint(path):
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        return (f"{os.path.basename(path)}  [{len(blob.get('trained_on', []))} recordings; "
                f"val loss {blob['validation_loss']:.3f}; "
                f"{blob['trained_at'].replace('T', ' ')[:16]}]")
    except Exception as error:
        return f"{os.path.basename(path)}  [unreadable: {error}]"
