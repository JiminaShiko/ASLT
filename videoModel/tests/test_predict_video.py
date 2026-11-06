import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import av
import numpy as np
import torch
from torch import nn

from streamlit_app import LivePredictionState, LiveVideoProcessor
from src.predict_video import frames_to_tensor, load_model, load_wlasl_record, predict
from src.train_model import build_model


class FixedClassifier(nn.Module):
    def forward(self, videos):
        return torch.tensor([[0.1, 2.0, 1.0]], device=videos.device)


class PredictVideoTests(unittest.TestCase):
    def test_frames_to_tensor_matches_video_model_input(self):
        frames = np.full((16, 112, 112, 3), 255, dtype=np.uint8)
        video = frames_to_tensor(frames)
        self.assertEqual(tuple(video.shape), (1, 16, 3, 112, 112))
        self.assertEqual(video.dtype, torch.float32)
        self.assertTrue(torch.all(video == 1.0))

    def test_frames_to_tensor_rejects_unexpected_shape(self):
        with self.assertRaises(ValueError):
            frames_to_tensor(np.zeros((15, 112, 112, 3), dtype=np.uint8))

    def test_live_processor_predicts_after_collecting_video_window(self):
        state = LivePredictionState()
        processor = LiveVideoProcessor(
            FixedClassifier(),
            ["hello", "thanks", "bye"],
            torch.device("cpu"),
            state,
        )
        frame = av.VideoFrame.from_ndarray(
            np.zeros((32, 32, 3), dtype=np.uint8),
            format="bgr24",
        )

        for _ in range(16):
            processor.recv(frame)

        results, error, frames_seen, _ = state.snapshot()
        self.assertIsNone(error)
        self.assertEqual(frames_seen, 16)
        self.assertEqual([label for label, _ in results[:2]], ["thanks", "bye"])

    def test_load_wlasl_record_reads_target_annotation(self):
        with tempfile.TemporaryDirectory() as temporary:
            metadata = Path(temporary) / "metadata.json"
            metadata.write_text(
                json.dumps(
                    [
                        {
                            "gloss": "drink",
                            "instances": [
                                {"video_id": "70173", "frame_start": 5710, "frame_end": 5831, "fps": 25}
                            ],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            record = load_wlasl_record("70173", metadata)
        self.assertEqual(record["gloss"], "drink")
        self.assertEqual(record["frame_start"], 5710)
        self.assertEqual(record["frame_end"], 5831)

    def test_predict_returns_top_scores(self):
        video = torch.zeros((1, 16, 3, 112, 112))
        results = predict(FixedClassifier(), video, ["hello", "thanks", "bye"], torch.device("cpu"), 2)
        self.assertEqual([label for label, _ in results], ["thanks", "bye"])
        self.assertGreater(results[0][1], results[1][1])

    def test_load_model_uses_standalone_checkpoint_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            model_dir = Path(temporary)
            source_model = build_model("cnn3d", 3)
            torch.save(
                {
                    "architecture": "cnn3d",
                    "class_names": ["eat", "drink", "thanks"],
                    "model_state_dict": source_model.state_dict(),
                },
                model_dir / "cnn3d.pt",
            )
            with patch("src.predict_video.CHECKPOINT_DIR", model_dir):
                model, class_names = load_model("cnn3d", torch.device("cpu"))
        self.assertEqual(class_names, ["eat", "drink", "thanks"])
        self.assertFalse(model.training)


if __name__ == "__main__":
    unittest.main()