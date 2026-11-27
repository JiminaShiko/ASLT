"""Runtime side of the J/Z stroke head (weights written by `train_jz.py`).

The head reads a window, not a frame, so it cannot be folded into the per-frame
letter classifier; `MotionDecider` owns the window and asks this for the answer.
Set `ASL_JZ_MODEL=off` to fall back to the old trajectory rule.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

if __package__:
    from . import config
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import config  # noqa: E402

DEFAULT_WEIGHTS = "jz_motion.pt"
ENV = "ASL_JZ_MODEL"


def weights_path() -> Path:
    return config.MODELS_DIR / DEFAULT_WEIGHTS


def available() -> bool:
    """False when the head was never trained or was switched off."""
    if os.environ.get(ENV, "").strip().lower() in ("off", "0", "false", "no"):
        return False
    return weights_path().is_file()


class JZModel:
    """One loaded head: `predict(window_features)` -> {letter: probability}."""

    def __init__(self, path: Path | None = None):
        import torch

        path = Path(path or weights_path())
        blob = torch.load(path, map_location="cpu", weights_only=False)
        self.meta = blob["meta"]
        self.labels = list(self.meta["labels"])
        self.thresholds = {k.lower(): float(v)
                           for k, v in self.meta["thresholds"].items()}
        self.mean = np.asarray(blob["mean"], dtype=np.float32)
        self.std = np.asarray(blob["std"], dtype=np.float32)
        hidden = tuple(self.meta.get("hidden") or (128, 64))
        dropout = float(self.meta.get("dropout", 0.2))
        self.nets = []
        # `state_dicts` is the shipped ensemble; `state_dict` is the older
        # single-seed file, still loadable so an un-retrained model works.
        states = blob.get("state_dicts") or [blob["state_dict"]]
        for state in states:
            net = self._build(int(self.meta["feature_dim"]), hidden, dropout,
                              len(self.labels))
            net.load_state_dict(state)
            net.eval()
            self.nets.append(net)
        self.torch = torch
        self.path = path

    @classmethod
    def load(cls) -> "JZModel | None":
        """The shipped head, or None when it is absent or disabled."""
        if not available():
            return None
        try:
            return cls()
        except Exception:            # noqa: BLE001 - never break the camera for this
            return None

    @staticmethod
    def _build(dim: int, hidden, dropout: float, classes: int):
        import torch
        layers: list = []
        prev = dim
        for h in hidden:
            layers += [torch.nn.Linear(prev, int(h)), torch.nn.ReLU(),
                       torch.nn.Dropout(dropout)]
            prev = int(h)
        layers.append(torch.nn.Linear(prev, classes))
        return torch.nn.Sequential(*layers)

    def predict(self, feats: np.ndarray) -> dict[str, float]:
        """Mean softmax over the shipped seeds."""
        x = (np.asarray(feats, dtype=np.float32) - self.mean) / self.std
        acc = None
        with self.torch.no_grad():
            for net in self.nets:
                logits = net(self.torch.from_numpy(x[None, :]))
                p = self.torch.softmax(logits, 1).numpy()[0]
                acc = p if acc is None else acc + p
        p = acc / len(self.nets)
        return {name: float(v) for name, v in zip(self.labels, p)}

    def decide(self, feats: np.ndarray, allowed) -> tuple[str | None, dict]:
        """Highest-probability allowed letter that clears its own threshold.

        Both classes come from the same softmax, so `other` is the refusal class:
        a stroke has to be more J than *not a J* as well as above threshold.
        """
        probs = self.predict(feats)
        other = probs.get("other", 0.0)
        best, best_p = None, 0.0
        for letter in allowed:
            p = probs.get(letter, 0.0)
            if p < self.thresholds.get(letter.lower(), 1.01):
                continue
            if p <= other:
                continue
            if p > best_p:
                best, best_p = letter, p
        return best, probs
