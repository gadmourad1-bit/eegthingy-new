import os
import sys

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config import EEG_CHANNELS_TARGETS
from mirepnet_pipeline.model import AdaptedMIRepNet
from mirepnet_pipeline.preprocess import prepare_for_mirepnet


class MIRepNetDecoder(BaseEstimator, ClassifierMixin):
    """
    Wrapper around the adapted MIRepNet transformer.
    Expects raw EEG windows of shape (n_epochs, n_channels, n_times).
    Uses native 15 channels, Euclidean Alignment, then a learnable adapter maps
    to 45 channels for the pretrained backbone.
    """

    def __init__(self, sfreq, source_channels=None, pretrain_path=None,
                 n_classes=2, device='cpu'):
        self.sfreq = sfreq
        self.source_channels = list(source_channels) if source_channels is not None else list(EEG_CHANNELS_TARGETS)
        self.n_classes = n_classes
        self.device = device
        self.num_channels = len(self.source_channels)

        if pretrain_path is None:
            raise ValueError("A fine-tuned checkpoint path must be provided.")

        # Load adapted model (includes adapter + base 45ch backbone + head)
        self.model = AdaptedMIRepNet(
            in_channels=self.num_channels,
            n_classes=n_classes,
            pretrain_path=pretrain_path if pretrain_path.endswith('.pth') and os.path.exists(pretrain_path) else None,
        )
        # If we didn't pass pretrain_path (because it's original), load manually
        if pretrain_path is not None and os.path.exists(pretrain_path):
            state = torch.load(pretrain_path, map_location='cpu')
            self.model.load_state_dict(state, strict=False)

        self.model.to(device)
        self.model.eval()

        self.ea_whitener = None
        self.classes_ = np.arange(n_classes)

    def fit(self, X, y, groups=None):
        self.classes_ = np.unique(y)
        _, _, self.ea_whitener = prepare_for_mirepnet(
            X,
            self.source_channels,
            self.sfreq,
            fit_ea=True,
            project_to_template=False,
        )
        return self

    def set_reference(self, X_cal):
        _, _, self.ea_whitener = prepare_for_mirepnet(
            X_cal,
            self.source_channels,
            self.sfreq,
            fit_ea=True,
            project_to_template=False,
        )
        return self

    def _prepare(self, X):
        if self.ea_whitener is None:
            raise RuntimeError("Preprocessor not initialised. Call fit() or set_reference() first.")
        X_prep, _, _ = prepare_for_mirepnet(
            X,
            self.source_channels,
            self.sfreq,
            ea_whitener=self.ea_whitener,
            fit_ea=False,
            project_to_template=False,
        )
        return torch.tensor(X_prep, dtype=torch.float32, device=self.device)

    def predict_proba(self, X):
        X_t = self._prepare(X)
        with torch.no_grad():
            _, logits = self.model(X_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

    def analyze(self, X):
        probs = self.predict_proba(X)
        if self.n_classes == 2:
            score = np.log(np.clip(probs[:, 1], 1e-9, 1.0) /
                           np.clip(probs[:, 0], 1e-9, 1.0))
        else:
            top2 = np.sort(probs, axis=1)[:, ::-1][:, :2]
            score = top2[:, 0] - top2[:, 1]
        score = np.atleast_1d(score)
        band_sig = np.zeros((probs.shape[0], 1))
        return probs, score, band_sig