"""Train and evaluate classifiers on the prepared video dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.config import FRAME_HEIGHT, FRAME_WIDTH, NUM_FRAMES, PROCESSED_DIR, ROOT, TARGET_WORDS
from src.model_architectures import FrameCNNGRU, VideoCNN3D, build_model

DEFAULT_OUTPUT_DIR = PROCESSED_DIR / "training_artifacts"
MODEL_DIR = ROOT / "models"
SPLITS = ("train", "val", "test")


def prepare_data_if_needed(seed_manifest: Path) -> bool:
    """Download missing datasets, preprocess, and seed before the first run.

    Returns True when data was freshly prepared; training should then ignore
    previous checkpoints because they were fitted to different data.
    """
    raw_manifest = PROCESSED_DIR / "manifest.csv"
    if raw_manifest.is_file():
        return False
    print("No processed manifest found; preparing datasets automatically.")

    from src import download_data as data_download
    from src import preprocess as data_preprocess
    from src import seed_dataset as data_seed

    print("Read and accept the WLASL C-UDA terms before proceeding: "
          "https://dxli94.github.io/WLASL/")
    if not data_download.WLASL_METADATA.is_file():
        data_download.download_file(data_download.WLASL_METADATA_URL,
                                    data_download.WLASL_METADATA)
    entries = data_download.load_wlasl_entries()
    data_download.download_wlasl_videos(entries)

    if data_download.manual_asl_citizen_present():
        print(f"ASL Citizen: using the manual dataset at {data_download.ASL_CITIZEN_DIR}.")
    elif not data_download.ASL_CITIZEN_METADATA.is_file():
        print("Read and accept the ASL Citizen license terms before proceeding.")
        data_download.download_asl_citizen()

    data_preprocess.main()
    report = data_seed.build_seed(
        raw_manifest, {"train": 20, "val": 5, "test": 5}, seed_manifest.parent
    )
    print(f"Seed prepared: {report['records']} clips, {report['bytes'] / 1e6:.1f} MB.")
    return True


def augment_frames(frames: np.ndarray) -> np.ndarray:
    """Light train-only augmentation: mirror, temporal shift, brightness jitter."""
    rng = np.random.default_rng()
    if rng.random() < 0.5:
        frames = frames[:, :, ::-1, :]
    if rng.random() < 0.5:
        shift = int(rng.integers(-2, 3))
        if shift > 0:
            frames = np.concatenate(
                [frames[shift:], np.repeat(frames[-1:], shift, axis=0)], axis=0
            )
        elif shift < 0:
            frames = np.concatenate(
                [np.repeat(frames[:1], -shift, axis=0), frames[:shift]], axis=0
            )
    jitter = float(rng.uniform(0.85, 1.15))
    frames = np.clip(frames.astype(np.float32) * jitter, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frames)


class ProcessedVideoDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], class_to_index: dict[str, int], root: Path = PROCESSED_DIR,
                 include_seed_flag: bool = False, augment: bool = False):
        self.rows = rows
        self.class_to_index = class_to_index
        self.root = root
        self.include_seed_flag = include_seed_flag
        self.augment = augment

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
        if self.augment:
            frames = augment_frames(frames)
        video = torch.from_numpy(np.ascontiguousarray(frames))
        result = (video.permute(0, 3, 1, 2).float().div_(255.0), self.class_to_index[row["gloss"]])
        return (*result, row.get("is_seed") == "1") if self.include_seed_flag else result


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


def load_incremental_splits(class_names: list[str], seed_manifest: Path, raw_manifest: Path) -> dict[str, list[dict[str, str]]]:
    """Merge available current data with the compact seed; deduplicate source clips."""
    rows: dict[str, dict[str, str]] = {}
    for path, is_seed in ((raw_manifest, False), (seed_manifest, True)):
        if not path.is_file():
            continue
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                if row.get("split") not in SPLITS or row.get("gloss") not in class_names:
                    continue
                if is_seed and row.get("sha256"):
                    seed_clip = PROCESSED_DIR / row["processed_path"]
                    if not seed_clip.is_file():
                        raise FileNotFoundError(f"Seed clip listed in manifest is missing: {seed_clip}")
                    checksum = hashlib.sha256(seed_clip.read_bytes()).hexdigest()
                    if checksum.casefold() != row["sha256"].casefold():
                        raise ValueError(f"Seed clip checksum mismatch: {seed_clip}")
                key = f"{row.get('dataset','')}:{row.get('video_id','')}:{row['split']}"
                # Seed manifest wins because it remains usable after source files are removed.
                if key not in rows or is_seed:
                    row["is_seed"] = "1" if is_seed else "0"
                    rows[key] = row
    splits = {name: [] for name in SPLITS}
    for row in rows.values():
        splits[row["split"]].append(row)
    if any(not splits[name] for name in SPLITS):
        raise ValueError("Current data plus seed must contain train, validation, and test records.")
    missing = set(class_names) - {row["gloss"] for row in splits["train"]}
    if missing:
        raise ValueError(f"No training examples for: {sorted(missing)}")
    return splits


def initialize_from_checkpoint(model: nn.Module, checkpoint: dict, old_classes: list[str], class_names: list[str]) -> None:
    """Copy compatible weights and map old output rows by label, not by assumed index."""
    old_state = checkpoint["model_state_dict"]
    new_state = model.state_dict()
    old_index = {label: i for i, label in enumerate(old_classes)}
    new_index = {label: i for i, label in enumerate(class_names)}
    for key, value in old_state.items():
        if key not in new_state:
            continue
        if new_state[key].shape == value.shape:
            new_state[key] = value
        elif key.endswith(("head.1.weight", "head.1.bias", "head.2.weight", "head.2.bias")):
            for label in old_classes:
                if label in new_index:
                    new_state[key][new_index[label]].copy_(value[old_index[label]])
    model.load_state_dict(new_state)


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
                batch_size: int, learning_rate: float, seed: int,
                prior_checkpoint: dict | None = None, old_classes: list[str] | None = None,
                distillation_weight: float = 0.5, distillation_temperature: float = 2.0,
                max_old_recall_drop: float = 0.05) -> tuple[dict, list[dict]]:
    mapping = {label: index for index, label in enumerate(class_names)}
    datasets = {key: ProcessedVideoDataset(rows, mapping, include_seed_flag=(key == "train"),
                                           augment=(key == "train"))
                for key, rows in splits.items()}
    generator = torch.Generator().manual_seed(seed)
    loaders = {key: DataLoader(ds, batch_size=batch_size, shuffle=key == "train", num_workers=0,
                               generator=generator if key == "train" else None)
               for key, ds in datasets.items()}
    model = build_model(name, len(class_names)).to(device)
    teacher = None
    old_classes = old_classes or []
    if prior_checkpoint is not None:
        if prior_checkpoint.get("architecture") != name:
            raise ValueError(f"Cannot initialize {name} from a different architecture checkpoint.")
        initialize_from_checkpoint(model, prior_checkpoint, old_classes, class_names)
        teacher = build_model(name, len(old_classes)).to(device)
        teacher.load_state_dict(prior_checkpoint["model_state_dict"])
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
    counts = Counter(row["gloss"] for row in splits["train"])
    weights = torch.tensor([len(splits["train"]) / (len(class_names) * counts[label]) for label in class_names],
                           dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    history, best_state, best_f1, best_epoch, stale = [], None, -1.0, 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total = 0.0, 0
        for videos, labels, seed_flags in loaders["train"]:
            videos, labels = videos.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(videos)
            loss = loss_fn(logits, labels)
            # Distill only on retained, labeled old-word examples. New examples are not
            # forced to imitate the teacher's old vocabulary outputs.
            if teacher is not None and old_classes:
                old_rows = torch.tensor([class_names[int(label)] in old_classes for label in labels],
                                        dtype=torch.bool, device=device)
                replay_rows = seed_flags.to(device) & old_rows
                if replay_rows.any():
                    with torch.no_grad():
                        teacher_logits = teacher(videos[replay_rows])
                    student_columns = torch.tensor([mapping[label] for label in old_classes], device=device)
                    temperature = distillation_temperature
                    distill = nn.functional.kl_div(
                        nn.functional.log_softmax(logits[replay_rows][:, student_columns] / temperature, dim=1),
                        nn.functional.softmax(teacher_logits / temperature, dim=1), reduction="batchmean")
                    loss = loss + distillation_weight * temperature * temperature * distill
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
        "class_order_version": 1,
        "incremental_from": old_classes if prior_checkpoint is not None else [],
        "distillation": {"weight": distillation_weight, "temperature": distillation_temperature,
                         "replay_seed_only": True},
    }
    if teacher is not None and old_classes:
        old_validation = [row for row in splits["val"] if row["gloss"] in old_classes]
        support = {label: sum(row["gloss"] == label for row in old_validation) for label in old_classes}
        if any(count == 0 for count in support.values()):
            missing = [label for label, count in support.items() if count == 0]
            raise RuntimeError(f"Candidate rejected: retained validation anchors are missing old labels {missing}.")
        old_mapping = {label: index for index, label in enumerate(old_classes)}
        anchor_loader = DataLoader(ProcessedVideoDataset(old_validation, old_mapping), batch_size=batch_size,
                                   shuffle=False, num_workers=0)
        truth, before, after = [], [], []
        model.eval()
        with torch.inference_mode():
            for videos, labels in anchor_loader:
                videos = videos.to(device)
                teacher_logits = teacher(videos)
                candidate_logits = model(videos)
                truth.extend(labels.tolist())
                before.extend(teacher_logits.argmax(dim=1).cpu().tolist())
                predicted = candidate_logits.argmax(dim=1).cpu().tolist()
                after.extend([old_mapping.get(class_names[index], -1) for index in predicted])
        baseline_recall = float(recall_score(truth, before, labels=list(range(len(old_classes))),
                                             average="macro", zero_division=0))
        candidate_recall = float(recall_score(truth, after, labels=list(range(len(old_classes))),
                                              average="macro", zero_division=0))
        drop = baseline_recall - candidate_recall
        test["old_word_acceptance"] = {"baseline_macro_recall": baseline_recall,
                                       "candidate_macro_recall": candidate_recall,
                                       "drop": drop, "maximum_drop": max_old_recall_drop,
                                       "support": support}
        if drop > max_old_recall_drop:
            raise RuntimeError(f"Candidate rejected: old-word macro recall dropped by {drop:.3f} "
                               f"(limit {max_old_recall_drop:.3f}); previous checkpoint remains active.")
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
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed-manifest", type=Path, default=PROCESSED_DIR / "seed" / "manifest.csv",
                        help="Compact replay/evaluation manifest produced by src.seed_dataset.")
    parser.add_argument("--distillation-weight", type=float, default=0.5)
    parser.add_argument("--distillation-temperature", type=float, default=2.0)
    parser.add_argument("--old-class-recall-drop", type=float, default=0.05,
                        help="Maximum permitted absolute macro-recall drop on retained old-word validation anchors.")
    parser.add_argument("--fresh", action="store_true", help="Ignore existing checkpoints and train from scratch.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.epochs, args.patience, args.batch_size) < 1:
        raise SystemExit("epochs, patience, and batch-size must be positive.")
    requested = {word.strip().casefold() for word in args.classes.split(",") if word.strip()}
    models = [name.strip() for name in args.models.split(",") if name.strip()]
    if not models or len(set(models)) != len(models) or set(models) - {"cnn_gru", "cnn3d"}:
        raise SystemExit("Choose distinct model names from cnn_gru and cnn3d.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable.")
    if args.distillation_weight < 0 or args.distillation_temperature <= 0 or not 0 <= args.old_class_recall_drop <= 1:
        raise SystemExit("Distillation weight must be nonnegative, temperature positive, and recall-drop in [0, 1].")
    if prepare_data_if_needed(args.seed_manifest) and not args.fresh:
        print("Data was freshly prepared; previous checkpoints are ignored for this run.")
        args.fresh = True
    reference_checkpoint = None
    if not args.fresh:
        reference_path = MODEL_DIR / f"{models[0]}.pt"
        if reference_path.is_file():
            reference_checkpoint = torch.load(reference_path, map_location="cpu", weights_only=True)
    old_vocab = list(reference_checkpoint.get("class_names", [])) if reference_checkpoint else []
    unknown = requested - set(TARGET_WORDS) - set(old_vocab)
    if unknown:
        raise SystemExit(f"Unknown classes (add them to TARGET_WORDS first): {sorted(unknown)}")
    additions = [word for word in TARGET_WORDS if (not requested or word in requested) and word not in old_vocab]
    # Existing classes keep their checkpoint indices. New labels are appended in config order.
    classes = old_vocab + additions if old_vocab else [word for word in TARGET_WORDS if not requested or word in requested]
    if requested and old_vocab:
        omitted = set(old_vocab) - requested
        if omitted:
            raise SystemExit(f"Incremental updates cannot drop existing labels: {sorted(omitted)}")
    device = torch.device("cuda" if args.device == "cuda" or
                          (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    splits = load_incremental_splits(classes, args.seed_manifest, PROCESSED_DIR / "manifest.csv")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device={device}; records={len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}; "
          f"classes={len(classes)}")
    results, histories = {}, {}
    for name in models:
        prior_path = MODEL_DIR / f"{name}.pt"
        prior = None
        old_classes = []
        if not args.fresh and prior_path.is_file():
            prior = torch.load(prior_path, map_location="cpu", weights_only=True)
            old_classes = list(prior.get("class_names", []))
            if classes[:len(old_classes)] != old_classes:
                raise SystemExit(f"{name} checkpoint vocabulary is incompatible with the selected class order.")
        result, history = train_model(name, classes, splits, device, args.output_dir,
                                       args.epochs, args.patience, args.batch_size,
                                       args.learning_rate, args.seed, prior, old_classes,
                                       args.distillation_weight, args.distillation_temperature,
                                       args.old_class_recall_drop)
        results[name], histories[name] = result, history
        print(f"{name} test: accuracy={result['accuracy']:.4f}, macro_f1={result['macro_f1']:.4f}, "
              f"weighted_f1={result['weighted_f1']:.4f}")
    write_artifacts(args.output_dir, results, histories, classes,
                    len(splits["train"]), len(splits["val"]))
    print(f"Artifacts saved to {args.output_dir.resolve()}")
    print(f"Standalone model checkpoints saved to {MODEL_DIR.resolve()}")


if __name__ == "__main__":
    main()
