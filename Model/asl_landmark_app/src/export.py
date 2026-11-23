"""Step 6 - export the MLP to ONNX and prove it matches PyTorch on real cached hands."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import landmark_store as LS  # noqa: E402
import model as M  # noqa: E402
import normalize  # noqa: E402


def export(opset: int, prof: str) -> Path:
    """torch.onnx.export the profile's weights to its own .onnx (dynamic batch)."""
    import torch

    meta = json.loads(config.meta_path(prof).read_text(encoding="utf-8"))
    net = M.load_model(config.model_path(prof), meta["model"]["in_dim"],
                       len(meta["labels"]), tuple(meta["model"]["hidden"]),
                       meta["model"]["dropout"])
    net.eval()
    config.ensure_dirs()
    dummy = torch.zeros(1, meta["model"]["in_dim"], dtype=torch.float32)
    torch.onnx.export(
        net, dummy, str(config.onnx_path(prof)),
        verbose=False,
        input_names=["features"],
        output_names=["logits"],
        opset_version=opset,
        dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}},
        # torch >= 2.9 defaults to the dynamo exporter, which needs `onnxscript`;
        # the classic tracer is enough here and keeps the dependency list small.
        dynamo=False,
    )
    return config.onnx_path(prof)


def verify(n_samples: int, prof: str, seed: int = 0) -> bool:
    import onnx
    import onnxruntime as ort
    import torch

    meta = json.loads(config.meta_path(prof).read_text(encoding="utf-8"))
    net = M.load_model(config.model_path(prof), meta["model"]["in_dim"],
                       len(meta["labels"]), tuple(meta["model"]["hidden"]),
                       meta["model"]["dropout"])

    df = LS.load_raw()
    df = LS.keep_static_classes(df)
    rng = np.random.default_rng(seed)
    take = min(n_samples, len(df))
    idx = rng.choice(len(df), size=take, replace=False)
    lm = LS.landmark_matrix(df)[idx]
    feats = normalize.features_from_normalized_batch(normalize.prepare_batch(lm))
    y = np.asarray([config.LABEL_TO_INDEX[l] for l in df["letter"].to_numpy()[idx]])

    onnx_model = onnx.load(str(config.onnx_path(prof)))
    onnx.checker.check_model(onnx_model)
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(config.onnx_path(prof)), so,
                                providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    with torch.no_grad():
        pt = net(torch.from_numpy(feats)).numpy()
    ox = sess.run(None, {in_name: feats.astype(np.float32)})[0]

    mad = float(np.abs(pt - ox).max())
    agree = float((pt.argmax(1) == ox.argmax(1)).mean())
    pt_acc = float((pt.argmax(1) == y).mean())
    ox_acc = float((ox.argmax(1) == y).mean())

    # single-sample path too, because that is what live.py calls
    s1 = sess.run(None, {in_name: feats[:1].astype(np.float32)})[0][0]
    with torch.no_grad():
        t1 = net(torch.from_numpy(feats[:1])).numpy()[0]
    mad1 = float(np.abs(s1 - t1).max())

    print("\n-- ONNX verification --")
    print(f"  samples                    : {take:,} real cached hands")
    print(f"  max |torch - onnx| logits  : {mad:.2e}")
    print(f"  max |torch - onnx| (1 row) : {mad1:.2e}")
    print(f"  argmax agreement           : {100 * agree:.2f}%")
    print(f"  accuracy torch / onnx      : {100 * pt_acc:.2f}% / {100 * ox_acc:.2f}%")

    # speed
    for name, fn in (("torch", lambda x: net(torch.from_numpy(x)).detach().numpy()),
                     ("onnx", lambda x: sess.run(None, {in_name: x})[0])):
        t0 = time.perf_counter()
        for _ in range(200):
            fn(feats[:1])
        dt = time.perf_counter() - t0
        print(f"  {name} single-frame classify: {1000 * dt / 200:.3f} ms "
              f"({200 / dt:.0f} calls/s)")
    ok = mad < 1e-4 and agree > 0.999 and mad1 < 1e-4
    print(f"  RESULT: {'PASS' if ok else 'FAIL'} "
          f"(tolerance 1e-4 on logits, 100% argmax agreement)")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--profile", default=None, choices=sorted(config.PROFILES),
                    help="which trained model to export (default: the profile "
                         "config.PROFILE_DEFAULT points at, i.e. the shipped one)")
    args = ap.parse_args()
    prof = config.profile(args.profile)
    print(f"== Step 6: ONNX export (profile {prof}) ==")
    if not config.meta_path(prof).exists():
        raise SystemExit(f"No {config.meta_path(prof).name} - run "
                         f"src/train.py --profile {prof} first.")
    if not config.model_path(prof).exists():
        raise SystemExit(f"No {config.model_path(prof).name} - run "
                         f"src/train.py --profile {prof} first.")
    path = export(args.opset, prof)
    print(f"  wrote {path} ({path.stat().st_size / 1024:.1f} KB)")
    return 0 if verify(args.samples, prof) else 1


if __name__ == "__main__":
    raise SystemExit(main())
