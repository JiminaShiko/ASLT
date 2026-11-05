import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from src.predict_video import load_model, load_wlasl_record, predict
from src.train_model import build_model


class FixedClassifier(nn.Module):
    def forward(self, videos):
        return torch.tensor([[0.1, 2.0, 1.0]], device=videos.device)


class PredictVideoTests(unittest.TestCase):
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