"""Leading braindecode EEG networks wrapped for the GeoAdaptNet comparison.

Each net is trained under the same protected chronological protocol as
GeoAdaptNet: fit on the training recordings, select the checkpoint on a held-out
recording (never the outer test), and score raw balanced accuracy on the untouched
test recording.  Input is broadband epochs ``(n_epochs, n_channels, n_times)`` with
per-channel standardisation fit on the training set only.  These are architecture
baselines -- no covariance path and no online recentering, so the comparison
isolates the decoder architecture.

References: EEGNet (Lawhern 2018), ShallowConvNet / DeepConvNet (Schirrmeister
2017), EEG-Conformer (Song 2023), ATCNet (Altaheri 2023).
"""

from __future__ import annotations

import time
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

import braindecode.models as bdm

from ..shared.augment import left_right_swap_index


# key -> (display label, braindecode class, extra kwargs)
ARCHS: dict[str, tuple[str, Any, dict[str, Any]]] = {
    "eegnet": ("EEGNet", bdm.EEGNet, {}),
    "shallow": ("ShallowConvNet", bdm.ShallowFBCSPNet, {}),
    "deep": ("DeepConvNet", bdm.Deep4Net, {}),
    "conformer": ("EEG-Conformer", bdm.EEGConformer, {}),
    "atcnet": ("ATCNet", bdm.ATCNet, {}),
}
DNN_BASELINES = tuple(ARCHS)


class TorchEEGClassifier:
    """A braindecode model with a small, leakage-controlled training loop.

    ``fit`` takes an explicit validation set (the protocol's selection recording)
    and early-stops on its loss, restoring the best weights -- it never inspects
    the outer test recording.
    """

    def __init__(
        self,
        arch: str = "eegnet",
        *,
        n_times: int = 251,
        sfreq: float = 125.0,
        n_epochs: int = 200,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        patience: int = 40,
        seed: int = 7,
        device: str = "cuda",
        lr_swap_prob: float = 0.0,
        channels: Sequence[str] | None = None,
        verbose: bool = False,
    ) -> None:
        if arch not in ARCHS:
            raise ValueError(f"unknown arch {arch!r}; choose from {DNN_BASELINES}")
        if lr_swap_prob > 0.0 and channels is None:
            raise ValueError("lr_swap_prob>0 requires channel names for the mirror index")
        self.arch = arch
        self.n_times = int(n_times)
        self.sfreq = float(sfreq)
        self.n_epochs = int(n_epochs)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.patience = int(patience)
        self.seed = int(seed)
        self.device = device
        self.lr_swap_prob = float(lr_swap_prob)
        self.channels = tuple(channels) if channels is not None else None
        self.verbose = bool(verbose)

    def _resolve_device(self) -> torch.device:
        if self.device in (None, "auto"):
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _build(self, n_chans: int, n_times: int, n_classes: int) -> nn.Module:
        label, cls, kw = ARCHS[self.arch]
        model = cls(n_chans=n_chans, n_outputs=n_classes, n_times=n_times, sfreq=self.sfreq, **kw)
        model.eval()
        with torch.no_grad():
            probe = model(torch.zeros(2, n_chans, n_times))
        summed = torch.exp(probe).sum(dim=1)
        self._log_softmax = bool(torch.allclose(summed, torch.ones_like(summed), atol=1e-3))
        return model

    def _standardize_fit(self, x: np.ndarray) -> None:
        self.mean_ = x.mean(axis=(0, 2), keepdims=True)
        self.std_ = x.std(axis=(0, 2), keepdims=True) + 1e-6

    def _prep(self, x: np.ndarray) -> torch.Tensor:
        x = (np.asarray(x, dtype=np.float32) - self.mean_) / self.std_
        return torch.from_numpy(x)

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
    ) -> "TorchEEGClassifier":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        device = self._resolve_device()
        self.device_ = device

        x_train = np.asarray(x_train, dtype=np.float32)
        self.classes_ = np.unique(y_train)
        train_idx = np.searchsorted(self.classes_, y_train)
        val_idx = np.searchsorted(self.classes_, y_val)
        n_chans, n_times = x_train.shape[1], x_train.shape[2]

        self._standardize_fit(x_train)
        x_tr = self._prep(x_train).to(device)
        y_tr = torch.from_numpy(train_idx.astype(np.int64)).to(device)
        x_va = self._prep(x_val).to(device)
        y_va = torch.from_numpy(val_idx.astype(np.int64)).to(device)

        self.model_ = self._build(n_chans, n_times, len(self.classes_)).to(device)
        loss_fn = nn.NLLLoss() if self._log_softmax else nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        swap_index = None
        if self.lr_swap_prob > 0.0 and len(self.classes_) == 2:
            swap_index = torch.as_tensor(
                left_right_swap_index(self.channels), dtype=torch.long, device=device
            )

        best_val, best_state, stale = float("inf"), None, 0
        generator = torch.Generator(device=device).manual_seed(self.seed)
        started = time.time()
        for epoch in range(self.n_epochs):
            self.model_.train()
            order = torch.randperm(len(x_tr), device=device, generator=generator)
            for start in range(0, len(x_tr), self.batch_size):
                batch = order[start : start + self.batch_size]
                xb, yb = x_tr[batch], y_tr[batch]
                if swap_index is not None:
                    # Left/right electrode swap + label flip on a random subset,
                    # the raw-epoch analogue of the covariance augmentation.
                    do = torch.rand(len(xb), device=device, generator=generator) < self.lr_swap_prob
                    if torch.any(do):
                        xb = xb.clone()
                        xb[do] = xb[do][:, swap_index, :]
                        yb = yb.clone()
                        yb[do] = 1 - yb[do]
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(self.model_(xb), yb)
                loss.backward()
                optimizer.step()
            self.model_.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(self.model_(x_va), y_va))
            if val_loss < best_val - 1e-4:
                best_val, stale = val_loss, 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.model_.state_dict().items()}
            else:
                stale += 1
                if stale >= self.patience:
                    break
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.fit_seconds_ = time.time() - started
        self.epochs_run_ = epoch + 1
        self.param_count_ = sum(p.numel() for p in self.model_.parameters())
        if self.verbose:
            print(f"    [{ARCHS[self.arch][0]}] {self.epochs_run_} ep, val={best_val:.3f}, "
                  f"{self.fit_seconds_:.1f}s, {self.param_count_} params")
        return self

    def _logits(self, x: np.ndarray) -> torch.Tensor:
        self.model_.eval()
        x_t = self._prep(x).to(self.device_)
        chunks = []
        with torch.no_grad():
            for start in range(0, len(x_t), 256):
                chunks.append(self.model_(x_t[start : start + 256]).cpu())
        return torch.cat(chunks, dim=0)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        out = self._logits(x)
        proba = torch.exp(out) if self._log_softmax else torch.softmax(out, dim=1)
        return proba.numpy()

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.classes_[self.predict_proba(x).argmax(axis=1)]


__all__ = ["ARCHS", "DNN_BASELINES", "TorchEEGClassifier"]
