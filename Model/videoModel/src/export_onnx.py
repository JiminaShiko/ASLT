"""Stage 6b - ONNX export (opset 17, dynamic batch) plus a parity check on real clips, not noise."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import config, features, model, train as T, vocab


class _Probs(nn.Module):
    """softmax wrapper so the exported graph returns probabilities."""

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, x):
        return torch.softmax(self.net(x), dim=1)


def export(weights: Path | None = None, out: Path | None = None,
           log=None) -> tuple[Path, dict]:
    log = log or config.logger_for("onnx", config.LOGS_DIR / "onnx.log")
    weights = weights or config.FOR_RELEASE
    out = out or config.ONNX_PATH
    if not weights.is_file():
        config.logger_for("onnx").error(
            f"{weights} is missing - run python -m src.train first")
        raise SystemExit(2)
    net, meta = model.load(weights)
    if meta.get("smoke"):
        log.warning("%s was produced by a smoke run; the ONNX file will be "
                    "just as useless", weights)
    dim = int(net.arch.in_dim)
    fver = (meta.get("feature_version")
            or features.version_for_dim(dim) or config.FEATURE_VERSION)
    labels = meta.get("labels") or vocab.load_vocab().labels
    dummy = torch.zeros(1, config.NUM_FRAMES, dim)
    config.ensure_dirs()
    tmp = out.with_suffix(".onnx.part")
    # the shipped graph returns probabilities; live's commit rule expects them
    torch.onnx.export(
        _Probs(net), dummy, str(tmp), opset_version=17,
        input_names=["features"], output_names=["probabilities"],
        dynamic_axes={"features": {0: "batch"},
                      "probabilities": {0: "batch"}},
        dynamo=False)
    tmp.replace(out)
    # tracing leaves the module in train mode; that dropout would corrupt parity
    net.eval()
    log.info("exported %s (%.1f MB) from %s", out,
             out.stat().st_size / 1e6, weights)

    bt = None
    for s in ("test", "val", "train"):
        bt = T.make_batcher(s, labels, None, None, version=fver)
        if bt is not None:
            src = s
            break
    if bt is None:
        log.error("no cached landmarks to test against; export is unverified")
        return out, {"verified": False}
    n = min(64, len(bt))
    X = np.stack([bt.row(i, False, None) for i in range(n)]).astype(np.float32)
    with torch.no_grad():
        torch_out = _Probs(net)(torch.from_numpy(X)).numpy()
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"features": X})[0]
    diff = float(np.abs(onnx_out - torch_out).max())
    same_top1 = int((onnx_out.argmax(1) == torch_out.argmax(1)).sum())
    res = {"verified": True, "source_split": src, "n_samples": int(n),
           "max_prob_difference": round(diff, 8),
           "same_top1": f"{same_top1}/{n}",
           "opset": 17, "input": [1, config.NUM_FRAMES, dim],
           "feature_version": int(fver),
           "output": [1, len(labels)], "onnx_mb":
               round(out.stat().st_size / 1e6, 2),
           "weights_mb": round(weights.stat().st_size / 1e6, 2)}
    log.info("parity: %d/%d identical top-1, max probability difference %.2e",
             same_top1, n, diff)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "onnx_parity.json").write_text(
        json.dumps(res, indent=1), encoding="utf-8")
    if diff > 1e-4 or same_top1 != n:
        log.error("ONNX and torch disagree - do not ship this export")
        return out, res
    log.info("OK: the ONNX file is the same model")
    return out, res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Export + verify the ONNX model.")
    ap.add_argument("--weights", default=None,
                    help="checkpoint to export (default models/words_core20.pt)")
    ap.add_argument("--out", default=None,
                    help="target .onnx path (default models/words_core20.onnx)")
    a = ap.parse_args()
    export(Path(a.weights) if a.weights else None,
           Path(a.out) if a.out else None)
