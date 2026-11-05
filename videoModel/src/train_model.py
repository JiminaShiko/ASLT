"""Train and evaluate classifiers on the prepared video dataset."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.config import FRAME_HEIGHT, FRAME_WIDTH, NUM_FRAMES, PROCESSED_DIR, ROOT, TARGET_WORDS

DEFAULT_OUTPUT_DIR = PROCESSED_DIR / "training_artifacts"
MODEL_DIR = ROOT / "models"
SPLITS = ("train", "val", "test")


class ProcessedVideoDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], class_to_index: dict[str, int], root: Path = PROCESSED_DIR):
        self.rows = rows
        self.class_to_index = class_to_index
        self.root = root

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        path = self.root / row["processed_path"]
        with np.load(path) as archive:
            frames = archive["frames"]
        expected = (NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, 3)
        if frames.shape != expected:
            raise ValueError(f"Unexpected frame shape in {path}: {frames.shape}; expected {expected}.")
        video = torch.from_numpy(np.ascontiguousarray(frames))
        return video.permute(0, 3, 1, 2).float().div_(255.0), self.class_to_index[row["gloss"]]


class FrameCNNGRU(nn.Module):
    """Frame-wise spatial CNN followed by a temporal GRU."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1),
        )
        self.gru = nn.GRU(64, 64, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(64, num_classes))

    def forward(self, videos: torch.Tensor) -> torch.Tensor:
        batch, frames, channels, height, width = videos.shape
        features = self.encoder(videos.reshape(batch * frames, channels, height, width))
        features = features.flatten(1).reshape(batch, frames, -1)
        sequence, _ = self.gru(features)
        return self.head(sequence[:, -1])


class VideoCNN3D(nn.Module):
    """Compact 3D CNN over spatial and temporal dimensions."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 8, 3, padding=1, bias=False), nn.BatchNorm3d(8), nn.ReLU(inplace=True), nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(8, 16, 3, padding=1, bias=False), nn.BatchNorm3d(16), nn.ReLU(inplace=True), nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, padding=1, bias=False), nn.BatchNorm3d(32), nn.ReLU(inplace=True), nn.AdaptiveAvgPool3d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(32, num_classes))

    def forward(self, videos: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(videos.permute(0, 2, 1, 3, 4)))


def build_model(name: str, class_count: int) -> nn.Module:
    models = {"cnn_gru": FrameCNNGRU, "cnn3d": VideoCNN3D}
    if name not in models:
        raise ValueError(f"Unknown model: {name}")
    return models[name](class_count)


def load_manifest(path: Path, class_names: list[str]) -> dict[str, list[dict[str, str]]]:
    splits = {name: [] for name in SPLITS}
    with path.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            if row.get("split") in splits and row.get("gloss") in class_names:
                splits[row["split"]].append(row)
    if any(not splits[name] for name in SPLITS):
        raise ValueError("The manifest must contain train, validation, and test records for the selected classes.")
    missing = set(class_names) - {row["gloss"] for row in splits["train"]}
    if missing:
        raise ValueError(f"No training examples for: {sorted(missing)}")
    return splits


def compute_metrics(y_true: list[int], y_pred: list[int], class_names: list[str]) -> tuple[dict, np.ndarray]:
    present = sorted(set(y_true))
    report = classification_report(
        y_true, y_pred, labels=present, target_names=[class_names[i] for i in present],
        output_dict=True, zero_division=0,
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "test_support": dict(Counter(class_names[i] for i in y_true)),
        "classes_without_test_examples": [name for i, name in enumerate(class_names) if i not in present],
        "per_class": report,
    }
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    return metrics, matrix


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device, class_names: list[str]):
    model.eval()
    truth, predictions = [], []
    loss_total = 0.0
    count = 0
    for videos, labels in loader:
        videos, labels = videos.to(device), labels.to(device)
        logits = model(videos)
        loss_total += loss_fn(logits, labels).item() * labels.size(0)
        count += labels.size(0)
        truth.extend(labels.cpu().tolist())
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
    metrics, matrix = compute_metrics(truth, predictions, class_names)
    metrics["loss"] = loss_total / count
    return metrics, matrix


def save_confusion(matrix: np.ndarray, names: list[str], path: Path, title: str) -> None:
    fig, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=axis)
    axis.set(xticks=np.arange(len(names)), yticks=np.arange(len(names)), xticklabels=names, yticklabels=names,
             xlabel="Predicted sign", ylabel="True sign", title=title)
    plt.setp(axis.get_xticklabels(), rotation=55, ha="right")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            if matrix[row, col]:
                axis.text(col, row, str(matrix[row, col]), ha="center", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def train_model(name: str, class_names: list[str], splits: dict[str, list[dict[str, str]]],
                device: torch.device, output: Path, epochs: int, patience: int,
                batch_size: int, learning_rate: float, seed: int) -> tuple[dict, list[dict]]:
    mapping = {label: index for index, label in enumerate(class_names)}
    datasets = {key: ProcessedVideoDataset(rows, mapping) for key, rows in splits.items()}
    generator = torch.Generator().manual_seed(seed)
    loaders = {key: DataLoader(ds, batch_size=batch_size, shuffle=key == "train", num_workers=0,
                               generator=generator if key == "train" else None)
               for key, ds in datasets.items()}
    model = build_model(name, len(class_names)).to(device)
    counts = Counter(row["gloss"] for row in splits["train"])
    weights = torch.tensor([len(splits["train"]) / (len(class_names) * counts[label]) for label in class_names],
                           dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    history, best_state, best_f1, best_epoch, stale = [], None, -1.0, 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total = 0.0, 0
        for videos, labels in loaders["train"]:
            videos, labels = videos.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(videos), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)
        val, _ = evaluate(model, loaders["val"], loss_fn, device, class_names)
        history.append({"epoch": epoch, "train_loss": total_loss / total,
                        "val_loss": val["loss"], "val_macro_f1": val["macro_f1"]})
        print(f"{name} epoch {epoch}/{epochs}: train_loss={total_loss / total:.4f}, "
              f"val_loss={val['loss']:.4f}, val_macro_f1={val['macro_f1']:.4f}")
        if val["macro_f1"] > best_f1:
            best_f1, best_epoch, stale = val["macro_f1"], epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= patience:
            print(f"{name}: early stopping after epoch {epoch}.")
            break
    if best_state is None:
        raise RuntimeError(f"No best checkpoint was produced for {name}.")
    model.load_state_dict(best_state)
    test, matrix = evaluate(model, loaders["test"], loss_fn, device, class_names)
    val, _ = evaluate(model, loaders["val"], loss_fn, device, class_names)
    test.update({"model": name, "best_epoch": best_epoch, "validation_macro_f1": val["macro_f1"],
                 "validation_count": len(splits["val"]), "history": history})
    checkpoint = {
        "architecture": name,
        "model_state_dict": best_state,
        "class_names": class_names,
        "input_shape": [NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, 3],
        "best_epoch": best_epoch,
        "normalization": "uint8 RGB scaled to float32 [0, 1]",
        "seed": seed,
    }
    torch.save(checkpoint, output / f"{name}.pt")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, MODEL_DIR / f"{name}.pt")
    with (output / f"metrics_{name}.json").open("w", encoding="utf-8") as file:
        json.dump(test, file, indent=2, allow_nan=False)
    save_confusion(matrix, class_names, output / f"confusion_matrix_{name}.png", f"{name}: held-out test")
    return test, history


def write_artifacts(output: Path, results: dict[str, dict], histories: dict[str, list[dict]],
                    class_names: list[str], train_count: int, val_count: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, history in histories.items():
        axes[0].plot([p["epoch"] for p in history], [p["train_loss"] for p in history], label=f"{name} train")
        axes[0].plot([p["epoch"] for p in history], [p["val_loss"] for p in history], linestyle="--", label=f"{name} val")
        axes[1].plot([p["epoch"] for p in history], [p["val_macro_f1"] for p in history], label=name)
    axes[0].set(title="Training and validation loss", xlabel="Epoch", ylabel="Loss")
    axes[1].set(title="Validation macro F1", xlabel="Epoch", ylabel="F1", ylim=(0, 1))
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "training_curves.png", dpi=150)
    plt.close(fig)

    names = list(results)
    fig, axis = plt.subplots(figsize=(7, 4))
    x = np.arange(len(names))
    width = 0.24
    for offset, key in enumerate(("accuracy", "macro_f1", "weighted_f1")):
        axis.bar(x + (offset - 1) * width, [results[name][key] for name in names], width, label=key)
    axis.set(xticks=x, xticklabels=names, ylim=(0, 1), ylabel="Score", title="Held-out test comparison")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "model_comparison.png", dpi=150)
    plt.close(fig)

    fields = ["model", "accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "best_epoch"]
    with (output / "comparison.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for name, result in results.items():
            writer.writerow({key: name if key == "model" else result[key] for key in fields})

    styles = getSampleStyleSheet()
    story = [Paragraph("Video Sign Classifier: Evaluation Report", styles["Title"]), Spacer(1, 10),
             Paragraph(f"Official splits: {train_count} train, {val_count} validation, "
                       f"{sum(results[names[0]]['test_support'].values())} test records.", styles["BodyText"]),
             Spacer(1, 8),
             Paragraph("The comparison uses a frame-wise CNN + GRU and a compact 3D CNN. "
                       "Frames are scaled to [0, 1]; class-weighted cross-entropy is used. "
                       "Checkpoints are selected by validation macro F1 and evaluated on the held-out test split.",
                       styles["BodyText"]), Spacer(1, 10)]
    table_data = [["Model", "Accuracy", "Macro P", "Macro R", "Macro F1", "Weighted F1", "Epoch"]]
    for name, result in results.items():
        table_data.append([name, *(f"{result[key]:.3f}" for key in
                            ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1")),
                           str(result["best_epoch"])])
    table = Table(table_data, repeatRows=1)
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.lightblue),
                               ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                               ("FONTSIZE", (0, 0), (-1, -1), 8),
                               ("PADDING", (0, 0), (-1, -1), 5)]))
    story.extend([table, Spacer(1, 8)])
    missing = ", ".join(results[names[0]]["classes_without_test_examples"]) or "None"
    story.append(Paragraph(f"No test examples: {missing}. The current test split has no eat examples, "
                           "so its performance is not measured. Macro averages cover classes represented in test.",
                           styles["BodyText"]))
    story.extend([Spacer(1, 8), Paragraph("Limitations and error sources", styles["Heading2"]),
                  Paragraph("The data is small and imbalanced, particularly for eat, thanks, and drink. "
                            "Some WLASL clips are missing or undecodable. Uniform 16-frame sampling can "
                            "miss brief motion; signer, background, framing, and signing speed vary. "
                            "These small models are trained from scratch, so results are baselines, not "
                            "production-level performance. Confirm with the instructor whether dataset "
                            "licenses permit sharing trained weights.", styles["BodyText"]),
                  Spacer(1, 10), Image(str(output / "model_comparison.png"), width=6.8 * inch, height=3.4 * inch),
                  Image(str(output / "training_curves.png"), width=6.8 * inch, height=2.8 * inch)])
    for name in names:
        story.extend([Spacer(1, 8), Paragraph(f"Test confusion matrix: {name}", styles["Heading2"]),
                      Image(str(output / f"confusion_matrix_{name}.png"), width=6.2 * inch, height=5.0 * inch)])
    SimpleDocTemplate(str(output / "model_evaluation_report.pdf"), pagesize=letter).build(story)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate sign-video classifiers.")
    parser.add_argument("--models", default="cnn_gru,cnn3d", help="Comma-separated: cnn_gru, cnn3d, or both.")
    parser.add_argument("--classes", default="", help="Optional comma-separated subset, e.g. eat,drink,thanks.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.epochs, args.patience, args.batch_size) < 1:
        raise SystemExit("epochs, patience, and batch-size must be positive.")
    requested = {word.strip().casefold() for word in args.classes.split(",") if word.strip()}
    unknown = requested - set(TARGET_WORDS)
    if unknown:
        raise SystemExit(f"Unknown classes: {sorted(unknown)}")
    classes = [word for word in TARGET_WORDS if not requested or word in requested]
    models = [name.strip() for name in args.models.split(",") if name.strip()]
    if not models or len(set(models)) != len(models) or set(models) - {"cnn_gru", "cnn3d"}:
        raise SystemExit("Choose distinct model names from cnn_gru and cnn3d.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable.")
    device = torch.device("cuda" if args.device == "cuda" or
                          (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    splits = load_manifest(PROCESSED_DIR / "manifest.csv", classes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device={device}; records={len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}; "
          f"classes={len(classes)}")
    results, histories = {}, {}
    for name in models:
        result, history = train_model(name, classes, splits, device, args.output_dir,
                                       args.epochs, args.patience, args.batch_size,
                                       args.learning_rate, args.seed)
        results[name], histories[name] = result, history
        print(f"{name} test: accuracy={result['accuracy']:.4f}, macro_f1={result['macro_f1']:.4f}, "
              f"weighted_f1={result['weighted_f1']:.4f}")
    write_artifacts(args.output_dir, results, histories, classes,
                    len(splits["train"]), len(splits["val"]))
    print(f"Artifacts saved to {args.output_dir.resolve()}")
    print(f"Standalone model checkpoints saved to {MODEL_DIR.resolve()}")


if __name__ == "__main__":
    main()