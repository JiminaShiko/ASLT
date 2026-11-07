import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.train_model import FrameCNNGRU, ProcessedVideoDataset, VideoCNN3D, augment_frames, compute_metrics


class ModelTrainingTests(unittest.TestCase):
    def test_video_models_return_class_logits(self):
        videos = torch.zeros((2, 16, 3, 112, 112))
        for model in (FrameCNNGRU(3, pretrained=False), VideoCNN3D(3)):
            model.eval()
            with torch.inference_mode():
                logits = model(videos)
            self.assertEqual(tuple(logits.shape), (2, 3))

    def test_augment_frames_preserves_clip_shape(self):
        frames = np.random.randint(0, 255, (16, 112, 112, 3), dtype=np.uint8)
        for _ in range(20):
            augmented = augment_frames(frames)
            self.assertEqual(augmented.shape, frames.shape)
            self.assertEqual(augmented.dtype, np.uint8)

    def test_processed_dataset_loads_expected_tensor(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            np.savez_compressed(root / "sample.npz", frames=np.zeros((16, 112, 112, 3), dtype=np.uint8))
            dataset = ProcessedVideoDataset(
                [{"processed_path": "sample.npz", "gloss": "hello"}], {"hello": 0}, root
            )
            video, label = dataset[0]
            self.assertEqual(tuple(video.shape), (16, 3, 112, 112))
            self.assertEqual(video.dtype, torch.float32)
            self.assertEqual(label, 0)

    def test_metrics_mark_classes_without_test_examples(self):
        metrics, matrix = compute_metrics([1, 1], [1, 0], ["eat", "hello"])
        self.assertEqual(metrics["classes_without_test_examples"], ["eat"])
        self.assertEqual(metrics["test_support"], {"hello": 2})
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(matrix.shape, (2, 2))


if __name__ == "__main__":
    unittest.main()