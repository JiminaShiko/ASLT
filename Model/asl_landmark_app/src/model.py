"""The landmark MLP: 71 -> 128 -> 64 -> 24 letters, small enough to run on CPU.

The input width comes from the meta sidecar at load time, never assumed here.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class LandmarkNet(nn.Module):
    def __init__(self, in_dim: int = 71, hidden=(128, 64), n_classes: int = 24,
                 dropout: float = 0.25):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def build(in_dim: int, n_classes: int, hidden=(128, 64), dropout=0.25) -> LandmarkNet:
    return LandmarkNet(in_dim, hidden, n_classes, dropout)


@torch.no_grad()
def predict_probs(model: nn.Module, feats: np.ndarray, batch: int = 8192) -> np.ndarray:
    """Softmax probabilities for an (N, D) feature matrix."""
    model.eval()
    n_classes = list(model.net)[-1].out_features
    out = np.empty((len(feats), n_classes), dtype=np.float32)
    t = torch.from_numpy(np.ascontiguousarray(feats, dtype=np.float32))
    for i in range(0, len(t), batch):
        logits = model(t[i:i + batch])
        out[i:i + batch] = torch.softmax(logits, dim=1).numpy()
    return out


def load_model(path, in_dim: int, n_classes: int, hidden=(128, 64),
               dropout: float = 0.25) -> nn.Module:
    m = build(in_dim, n_classes, hidden, dropout)
    state = torch.load(path, map_location="cpu", weights_only=True)
    m.load_state_dict(state)
    m.eval()
    return m
