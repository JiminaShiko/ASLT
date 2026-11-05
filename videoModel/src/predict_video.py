"""Run a saved classifier on one video file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.config import ANNOTATION_FPS, ROOT, TARGET_WORDS, WLASL_METADATA, canonicalize_gloss
from src.model_architectures import build_model
from src.preprocess import extract_frames

CHECKPOINT_DIR = ROOT / "models"


def load_wlasl_record(video_id: str, metadata_path: Path = WLASL_METADATA) -> dict:
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    for entry in data:
        gloss = canonicalize_gloss(entry.get("gloss", ""))
        if gloss not in TARGET_WORDS:
            continue
        for instance in entry.get("instances", []):
            if str(instance.get("video_id", "")).strip() == video_id:
                return {
                    "dataset": "wlasl",
                    "gloss": gloss,
                    "frame_start": instance.get("frame_start", 1),
                    "frame_end": instance.get("frame_end", -1),
                    "annotation_fps": instance.get("fps", ANNOTATION_FPS),
                }
    raise ValueError(f"WLASL video ID {video_id!r} was not found in the target vocabulary.")


def prepare_video(video_path: Path, wlasl_id: str | None = None) -> tuple[torch.Tensor, str | None]:
    record = (
        load_wlasl_record(wlasl_id)
        if wlasl_id
        else {"dataset": "custom", "frame_start": None, "frame_end": None, "annotation_fps": None}
    )
    frames = extract_frames(video_path, record)
    tensor = torch.from_numpy(np.ascontiguousarray(frames))
    tensor = tensor.permute(0, 3, 1, 2).float().div_(255.0).unsqueeze(0)
    return tensor, record.get("gloss")


def load_model(model_name: str, device: torch.device) -> tuple[torch.nn.Module, list[str]]:
    checkpoint_path = CHECKPOINT_DIR / f"{model_name}.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Trained checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get("architecture") != model_name:
        raise ValueError(f"Checkpoint architecture does not match {model_name!r}.")
    class_names = checkpoint["class_names"]
    model = build_model(model_name, len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, class_names


def predict(
    model: torch.nn.Module,
    video: torch.Tensor,
    class_names: list[str],
    device: torch.device,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    if top_k < 1:
        raise ValueError("top_k must be at least one.")
    with torch.inference_mode():
        scores = torch.softmax(model(video.to(device)), dim=1)[0].cpu()
    indices = torch.argsort(scores, descending=True)[: min(top_k, len(class_names))]
    return [(class_names[int(index)], float(scores[index])) for index in indices]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict the sign in a video clip.")
    parser.add_argument("video", type=Path, help="Path to a supported video file.")
    parser.add_argument("--model", choices=("cnn_gru", "cnn3d"), default="cnn3d")
    parser.add_argument("--wlasl-id", help="Use the WLASL annotated signing interval for a full source video.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--json", action="store_true", dest="json_output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.video.is_file():
        raise SystemExit(f"Video file not found: {args.video}")
    if args.top_k < 1:
        raise SystemExit("--top-k must be at least one.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable.")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    model, class_names = load_model(args.model, device)
    try:
        video, reference_gloss = prepare_video(args.video, args.wlasl_id)
        results = predict(model, video, class_names, device, args.top_k)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error

    if args.wlasl_id:
        reference = f"{args.wlasl_id} ({reference_gloss})"
    else:
        reference = None
    if args.json_output:
        print(
            json.dumps(
                {
                    "model": args.model,
                    "reference": reference,
                    "predictions": [
                        {"label": label, "score": score}
                        for label, score in results
                    ],
                }
            )
        )
        return
    print(f"Model: {args.model}")
    print(f"Video: {args.video}")
    if reference:
        print(f"WLASL reference: {reference}")
    print("Top predictions (softmax scores, not calibrated confidence):")
    for label, score in results:
        print(f"  {label}: {score:.1%}")


if __name__ == "__main__":
    main()