"""Stage 4b - word model: standardise -> 1D-CNN -> BiGRU -> classifier.

Mean/std are TRAIN-split buffers stored in the checkpoint, so live cannot drift from training.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import config


@dataclass
class Arch:
    in_dim: int = 0
    num_classes: int = 0
    hidden: int = 96
    gru_layers: int = 2
    conv_channels: tuple[int, ...] = (64, 96)
    kernel: int = 5
    dropout: float = config.DROPOUT

    def as_dict(self) -> dict:
        return {"in_dim": self.in_dim, "num_classes": self.num_classes,
                "hidden": self.hidden, "gru_layers": self.gru_layers,
                "conv_channels": list(self.conv_channels),
                "kernel": self.kernel, "dropout": self.dropout}


class WordNet(nn.Module):
    """[B, T, D] -> [B, num_classes] logits."""

    def __init__(self, arch: Arch,
                 mean: np.ndarray | None = None,
                 std: np.ndarray | None = None) -> None:
        super().__init__()
        self.arch = arch
        d = arch.in_dim
        self.register_buffer("mu", torch.ones(d) if mean is None
                             else torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("sd", torch.ones(d) if std is None
                             else torch.as_tensor(std, dtype=torch.float32)
                             .clamp_min(1e-6))
        chans, layers = [], []
        c_in = d
        for c_out in arch.conv_channels:
            chans += [nn.Conv1d(c_in, c_out, arch.kernel,
                                padding=arch.kernel // 2),
                      nn.GroupNorm(8, c_out), nn.GELU(), nn.MaxPool1d(2)]
            c_in = c_out
        self.conv = nn.Sequential(*chans)
        self.gru = nn.GRU(c_in, arch.hidden, num_layers=arch.gru_layers,
                          batch_first=True, bidirectional=True,
                          dropout=arch.dropout if arch.gru_layers > 1 else 0.0)
        self.drop = nn.Dropout(arch.dropout)
        self.fc = nn.Linear(2 * arch.hidden, arch.num_classes)

    def _h0(self, h: torch.Tensor) -> torch.Tensor:
        """Explicit zero hidden state; without it ONNX bakes the export batch size in."""
        dirs = 2 if self.gru.bidirectional else 1
        return h.new_zeros(self.gru.num_layers * dirs, h.size(0),
                           self.gru.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mu) / self.sd                  # [B,T,D]
        h = self.conv(x.transpose(1, 2))             # [B,C,T/2^k]
        h = h.transpose(1, 2)                        # [B,T',C]
        out, _ = self.gru(h, self._h0(h))            # [B,T',2H]
        # max over time: averaging would blur the frame where the sign peaks
        pooled = out.max(1).values                   # [B,2H]
        return self.fc(self.drop(pooled))

    @torch.no_grad()
    def predict(self, x: np.ndarray) -> np.ndarray:
        """[B,T,D] or [T,D] numpy -> [B,K] probabilities (CPU, no autocast)."""
        self.eval()
        a = np.asarray(x, np.float32)
        if a.ndim == 2:
            a = a[None]
        return torch.softmax(self(torch.from_numpy(a)), dim=1).numpy()

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def standardise(feats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """NaN-aware per-feature mean/std over [N, T, D] training features."""
    a = np.asarray(feats, np.float64)
    mu = a.mean(axis=(0, 1)).astype(np.float32)
    sd = a.std(axis=(0, 1)).astype(np.float32)
    return mu, np.where(sd < 1e-6, 1.0, sd).astype(np.float32)


def build(arch: Arch, mean=None, std=None) -> WordNet:
    return WordNet(arch, mean, std)


def save(model: WordNet, path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    torch.save({"state_dict": model.state_dict(), "arch": model.arch.as_dict(),
                "meta": meta}, tmp)
    tmp.replace(path)


class Ensemble(nn.Module):
    """Average of independently trained nets, as probabilities.

    The members were trained with different regularization, so their errors are
    only partly shared; averaging their softmax output measured better than any
    single member on the held-out VAL split (see meta.json -> ensemble_eval).
    The live path and the exported ONNX graph share this class, so they cannot
    drift apart.
    """

    def __init__(self, nets: list[WordNet]) -> None:
        super().__init__()
        self.nets = nn.ModuleList(nets)

    @property
    def arch(self):
        return self.nets[0].arch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        total = None
        for n in self.nets:
            p = torch.softmax(n(x), dim=1)
            total = p if total is None else total + p
        return total / len(self.nets)

    @torch.no_grad()
    def predict(self, x: np.ndarray) -> np.ndarray:
        """[B,T,D] or [T,D] numpy -> [B,K] probabilities."""
        self.eval()
        a = np.asarray(x, np.float32)
        if a.ndim == 2:
            a = a[None]
        return self(torch.from_numpy(a)).numpy()

    def n_params(self) -> int:
        return sum(n.n_params() for n in self.nets)


def load(path: Path) -> tuple[WordNet, dict]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    arch = Arch(**obj["arch"])
    model = WordNet(arch)
    model.load_state_dict(obj["state_dict"])
    model.eval()
    return model, obj.get("meta", {})


def load_release(path: Path) -> tuple[nn.Module, dict]:
    """The shipped artifact: the primary checkpoint plus its meta['ensemble'] members.

    Members are file names next to `path`, so moving models/ moves the ensemble.
    """
    net, meta = load(path)
    names = list(meta.get("ensemble") or [])
    if not names:
        return net, meta
    nets = [net]
    for name in names:
        member = path.parent / str(name)
        if not member.is_file():
            raise FileNotFoundError(
                f"{path.name} lists ensemble member {name!r}, which is missing "
                f"from {path.parent}")
        n, m = load(member)
        if list(m.get("labels") or []) != list(meta.get("labels") or []):
            raise ValueError(f"ensemble member {name} has a different label set")
        if (n.arch.in_dim, n.arch.num_classes) != (net.arch.in_dim,
                                                   net.arch.num_classes):
            raise ValueError(f"ensemble member {name} has a different shape")
        nets.append(n)
    return Ensemble(nets), meta


if __name__ == "__main__":
    import json
    from . import features
    arch = Arch(in_dim=features.FEATURE_DIM, num_classes=22)
    m = build(arch)
    x = torch.randn(4, config.NUM_FRAMES, arch.in_dim)
    y = m(x)
    print(json.dumps({
        "params": m.n_params(),
        "in": list(x.shape), "out": list(y.shape),
        "blocks": [f"{i} -> {o.out_channels} (pool /2)"
                   for i, o in enumerate(m.conv) if isinstance(o, nn.Conv1d)],
        "gru": [f"{m.arch.conv_channels[-1]} -> {m.arch.hidden} bi x"
                f"{m.arch.gru_layers}"],
        "pooled": 2 * m.arch.hidden,
        "feature_dim": features.FEATURE_DIM,
    }, indent=1))
