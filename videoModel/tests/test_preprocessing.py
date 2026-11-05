import csv
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from src import download_data, preprocess
from src.config import canonicalize_gloss


class VideoPreprocessingTests(unittest.TestCase):
    def test_manual_download_mode_reports_missing_wlasl_videos(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wlasl_dir = root / "WLASL"
            video_dir = wlasl_dir / "videos"
            video_dir.mkdir(parents=True)
            (video_dir / "available.mp4").write_bytes(b"video")

            old_wlasl_dir = download_data.WLASL_DIR
            old_video_dir = download_data.WLASL_VIDEOS
            download_data.WLASL_DIR = wlasl_dir
            download_data.WLASL_VIDEOS = video_dir
            try:
                download_data.report_manual_wlasl_videos(
                    [
                        {
                            "gloss": "hello",
                            "instances": [
                                {"video_id": "available", "split": "train"},
                                {"video_id": "missing", "split": "test"},
                            ],
                        }
                    ]
                )
            finally:
                download_data.WLASL_DIR = old_wlasl_dir
                download_data.WLASL_VIDEOS = old_video_dir

            with (wlasl_dir / "download_failures.csv").open(
                newline="", encoding="utf-8"
            ) as file:
                failures = list(csv.DictReader(file))
            self.assertEqual([row["video_id"] for row in failures], ["missing"])

    def test_manual_download_mode_is_available_as_cli_flag(self):
        with patch(
            "sys.argv",
            [
                "download_data",
                "--manual",
                "--confirm-wlasl-cuda",
                "--confirm-asl-citizen-license",
            ],
        ):
            args = download_data.parse_args()
        self.assertTrue(args.manual)

    def test_project_vocab_maps_wlasl_thank_you_label(self):
        self.assertEqual(canonicalize_gloss("Thank-you"), "thanks")
        self.assertEqual(canonicalize_gloss("HELLO!"), "hello")

    def test_wlasl_annotation_frames_are_mapped_to_video_fps(self):
        record = {
            "dataset": "wlasl",
            "frame_start": 26,
            "frame_end": 50,
            "annotation_fps": 25,
        }
        self.assertEqual(preprocess.frame_interval(record, 100, 25.0), (25, 49))

    def test_asl_citizen_metadata_finds_target_word_video(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "fixture.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(
                    "ASL_Citizen/train.csv",
                    "video_id,gloss,signer_id\nclip-01,Thank You,signer-1\n",
                )
                archive.writestr("ASL_Citizen/videos/clip-01.mp4", b"video")
                archive.writestr("ASL_Citizen/videos/other.mp4", b"other")

            with zipfile.ZipFile(archive_path) as archive:
                rows, _ = download_data.read_asl_citizen_metadata(archive)

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["gloss"], "thanks")
            self.assertEqual(rows[0]["split"], "train")
            self.assertEqual(
                rows[0]["archive_member"],
                "ASL_Citizen/videos/clip-01.mp4",
            )
            old_video_dir = download_data.ASL_CITIZEN_VIDEOS
            download_data.ASL_CITIZEN_VIDEOS = Path(temporary) / "selected"
            try:
                with zipfile.ZipFile(archive_path) as archive:
                    extracted, errors = (
                        download_data.extract_selected_asl_citizen_videos(
                            archive, rows
                        )
                    )
            finally:
                download_data.ASL_CITIZEN_VIDEOS = old_video_dir

            self.assertEqual(extracted, 1)
            self.assertEqual(errors, [])
            self.assertEqual(
                (Path(temporary) / "selected" / "clip-01.mp4").read_bytes(),
                b"video",
            )

    def test_video_frames_are_saved_with_expected_shape(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_dir = root / "datasets"
            video_dir = dataset_dir / "videos"
            video_dir.mkdir(parents=True)
            video_path = video_dir / "sample.avi"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                10.0,
                (32, 32),
            )
            if not writer.isOpened():
                self.skipTest("OpenCV MJPG video writer is unavailable.")
            for value in (0, 60, 120, 180, 240):
                frame = np.full((32, 32, 3), value, dtype=np.uint8)
                writer.write(frame)
            writer.release()

            old_processed_dir = preprocess.PROCESSED_DIR
            processed_dir = dataset_dir / "processed"
            preprocess.PROCESSED_DIR = processed_dir
            try:
                record = {
                    "dataset": "asl_citizen",
                    "video_id": "sample",
                    "gloss": "hello",
                    "split": "train",
                    "signer_id": "signer-1",
                    "video_path": video_path,
                    "frame_start": None,
                    "frame_end": None,
                    "annotation_fps": None,
                }
                result = preprocess.preprocess_record(record)
            finally:
                preprocess.PROCESSED_DIR = old_processed_dir

            output_path = processed_dir / result["processed_path"]
            with np.load(output_path) as saved:
                frames = saved["frames"]
            self.assertEqual(frames.shape, (16, 112, 112, 3))
            self.assertEqual(frames.dtype, np.uint8)
            self.assertEqual(result["split"], "train")

    def test_extract_frames_uses_decoded_count_instead_of_container_metadata(self):
        source_frames = [
            np.full((8, 8, 3), index, dtype=np.uint8)
            for index in range(30)
        ]

        class FakeCapture:
            def __init__(self, _path):
                self.index = 0

            def isOpened(self):
                return True

            def get(self, property_id):
                return 15.0 if property_id == cv2.CAP_PROP_FPS else 31

            def read(self):
                if self.index >= len(source_frames):
                    return False, None
                frame = source_frames[self.index]
                self.index += 1
                return True, frame

            def release(self):
                pass

        with patch("src.preprocess.cv2.VideoCapture", FakeCapture):
            frames = preprocess.extract_frames(
                Path("camera.webm"),
                {
                    "dataset": "custom",
                    "frame_start": None,
                    "frame_end": None,
                    "annotation_fps": None,
                },
            )

        self.assertEqual(frames.shape, (16, 112, 112, 3))
        self.assertEqual(int(frames[0, 0, 0, 0]), 0)
        self.assertEqual(int(frames[-1, 0, 0, 0]), 29)


if __name__ == "__main__":
    unittest.main()
