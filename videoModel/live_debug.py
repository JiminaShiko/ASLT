"""In-memory live camera diagnostic for Holistic vs Hands + Pose.

Run from videoModel with ``python live_debug.py``. Frames are never written.
Both trackers receive each captured frame; Tab changes which tracker's
landmarks are drawn. Press q or Esc to quit and print the comparison table.
"""
from __future__ import annotations

import argparse
import math
import os
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from src import config, landmarks, predictor


class LatestFrame:
    """Continuously drain the camera and retain one in-memory frame only."""
    def __init__(self, cap):
        self.cap = cap
        self.lock = threading.Lock()
        self.item = None
        self.count = 0
        self.stopped = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stopped:
            ok, frame = self.cap.read()
            captured = time.perf_counter()
            if ok:
                with self.lock:
                    self.count += 1
                    self.item = (self.count, captured, frame)

    def read(self):
        with self.lock:
            return self.item

    def close(self):
        self.stopped = True
        self.thread.join(timeout=2)


class HolisticTracker:
    def __init__(self, detection, tracking):
        from mediapipe.tasks.python import vision
        self.vision = vision
        self.detection = detection
        self.tracking = tracking
        self.det = None
        self.ts = -1
        self._create()

    def _create(self):
        from mediapipe.tasks.python.core.base_options import BaseOptions
        model = landmarks.model_path()
        if not model.is_file():
            raise SystemExit(f"Holistic task model is missing at {model}; diagnostic will not download or write a model file.")
        opts = self.vision.HolisticLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model)),
            running_mode=self.vision.RunningMode.VIDEO,
            min_face_detection_confidence=self.detection,
            min_face_landmarks_confidence=self.tracking,
            min_pose_detection_confidence=self.detection,
            min_pose_landmarks_confidence=self.tracking,
            min_hand_landmarks_confidence=self.tracking,
            output_face_blendshapes=False, output_segmentation_mask=False,
        )
        self.det = self.vision.HolisticLandmarker.create_from_options(opts)

    def reset(self):
        self.det.close()
        self.ts = -1
        self._create()

    def close(self):
        self.det.close()

    def detect(self, rgb):
        import mediapipe as mp
        ts = max(int(time.perf_counter() * 1000), self.ts + 1)
        self.ts = ts
        res = self.det.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
        return landmarks.pack_result(res)


class HandsPoseTracker:
    """Legacy solutions provide the requested model-complexity control."""
    def __init__(self, complexity, detection, tracking):
        import mediapipe as mp
        self.mp = mp
        self.complexity, self.detection, self.tracking = complexity, detection, tracking
        self.reset()

    def reset(self):
        if hasattr(self, "hands"):
            self.hands.close()
            self.pose.close()
        self.hands = self.mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=2, model_complexity=self.complexity,
            min_detection_confidence=self.detection,
            min_tracking_confidence=self.tracking)
        self.pose = self.mp.solutions.pose.Pose(
            static_image_mode=False, model_complexity=self.complexity,
            smooth_landmarks=True, min_detection_confidence=self.detection,
            min_tracking_confidence=self.tracking)

    def close(self):
        self.hands.close()
        self.pose.close()

    def detect(self, rgb):
        out = np.full((config.P_TOTAL, 3), np.nan, np.float32)
        h, w = rgb.shape[:2]
        p = self.pose.process(rgb)
        if p.pose_landmarks:
            lm = p.pose_landmarks.landmark
            for row, idx in enumerate(config.POSE_UPPER):
                out[row] = (lm[idx].x, lm[idx].y, lm[idx].z)
        hs = self.hands.process(rgb)
        if hs.multi_hand_landmarks:
            handed = hs.multi_handedness or []
            for i, hand in enumerate(hs.multi_hand_landmarks):
                # Legacy Hands labels are from the unmirrored input image.
                label = handed[i].classification[0].label.lower() if i < len(handed) else "right"
                # Solutions labels assume mirrored selfie input; restore the
                # physical side because this diagnostic feeds the raw frame.
                side = "right" if label == "left" else "left"
                start = config.COLS_LHAND.start if side == "left" else config.COLS_RHAND.start
                out[start:start + config.N_HAND] = [(q.x, q.y, q.z) for q in hand.landmark]
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--model-complexity", type=int, choices=(0, 1, 2), default=1,
                    help="Hands + Pose complexity; Holistic Tasks uses its fixed bundled model")
    ap.add_argument("--detection-confidence", type=float, default=.3)
    ap.add_argument("--tracking-confidence", type=float, default=.3)
    ap.add_argument("--redetect-after", type=int, default=30,
                    help="recreate a tracker after this many consecutive frames without a hand; 0 disables")
    ap.add_argument("--tracker", choices=("holistic", "hands_pose"), default="holistic",
                    help="tracker whose skeleton is displayed; both are measured when available")
    ap.add_argument("--weights", default=None,
                    help="optional word-model checkpoint; defaults to models/words_core20.pt")
    ap.add_argument("--no-mirror", action="store_true")
    args = ap.parse_args()
    for name in ("detection_confidence", "tracking_confidence"):
        if not 0 <= getattr(args, name) <= 1:
            ap.error(f"--{name.replace('_', '-')} must be between 0 and 1")

    import mediapipe as mp
    legacy_available = hasattr(mp, "solutions")

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if os.name == "nt" else 0)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    stream = LatestFrame(cap)
    negotiated_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    negotiated_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    camera_buffer = cap.get(cv2.CAP_PROP_BUFFERSIZE)
    holistic = HolisticTracker(args.detection_confidence, args.tracking_confidence)
    word_model = predictor.WordPredictor(weights=Path(args.weights) if args.weights else None)
    hp = (HandsPoseTracker(args.model_complexity, args.detection_confidence,
                           args.tracking_confidence) if legacy_available else None)
    if args.tracker == "hands_pose" and hp is None:
        holistic.close()
        stream.close(); cap.release()
        raise SystemExit("--tracker hands_pose requires legacy mp.solutions, which is absent from installed MediaPipe 1.2.2. No alternate tracker is selected silently.")
    trackers = {"Holistic": holistic}
    if hp is not None:
        trackers["Hands+Pose"] = hp
    selected = "Hands+Pose" if args.tracker == "hands_pose" else "Holistic"
    stats = {k: {"n": 0, "found": 0, "above_n": 0, "above_found": 0,
                 "face_found": 0, "cost": [], "jitter": [], "missing": 0,
                 "last_tip": None}
             for k in trackers}
    prev_capture = None
    cap_intervals = deque(maxlen=60)
    seen = -1
    try:
        while True:
            item = stream.read()
            if item is None or item[0] == seen:
                time.sleep(.001)
                continue
            seq, captured, bgr = item
            seen = seq
            if prev_capture is not None:
                cap_intervals.append(captured - prev_capture)
            prev_capture = captured
            rgb = landmarks.bgr_to_rgb(bgr)
            rows = {}
            for name, tracker in trackers.items():
                t0 = time.perf_counter()
                row = tracker.detect(rgb)
                elapsed = (time.perf_counter() - t0) * 1000
                rows[name] = row
                s = stats[name]
                hand = bool(np.isfinite(row[config.COLS_LHAND.start, :2]).all() or
                            np.isfinite(row[config.COLS_RHAND.start, :2]).all())
                face_found = bool(np.isfinite(row[config.COLS_FACE, :2]).all(axis=1).any())
                shoulders = row[[config.POSE_L_SHOULDER, config.POSE_R_SHOULDER], 1]
                wrists = row[[config.COLS_LHAND.start, config.COLS_RHAND.start], 1]
                above = bool(np.isfinite(shoulders).all() and np.isfinite(wrists).any() and
                             np.any(wrists[np.isfinite(wrists)] < np.mean(shoulders)))
                s["n"] += 1; s["found"] += int(hand); s["cost"].append(elapsed)
                s["face_found"] += int(face_found)
                s["above_n"] += int(above); s["above_found"] += int(above and hand)
                if hand:
                    tips = [row[i, :2] for i in (config.COLS_LHAND.start + 8,
                                                 config.COLS_RHAND.start + 8)
                            if np.isfinite(row[i, :2]).all()]
                    tip = np.mean(tips, axis=0) if tips else None
                    if tip is not None and s["last_tip"] is not None:
                        s["jitter"].append(float(np.linalg.norm(tip - s["last_tip"]) * math.hypot(*rgb.shape[:2])))
                    s["last_tip"] = tip
                    s["missing"] = 0
                else:
                    s["missing"] += 1; s["last_tip"] = None
                    if args.redetect_after and s["missing"] >= args.redetect_after:
                        tracker.reset(); s["missing"] = 0

            # The word network was trained on Holistic landmarks. Keep feeding
            # it that layout even if the debug display is switched to another
            # tracker; no Hands+Pose landmarks enter the classifier.
            word_result = word_model.update(rows["Holistic"])
            row = rows[selected]
            h, w = bgr.shape[:2]
            display = cv2.flip(bgr, 1) if not args.no_mirror else bgr.copy()
            # Coordinates stay normalized on the captured frame; the same
            # horizontal flip is applied to the image and every landmark.
            for hand, color in (("left", (0, 220, 255)), ("right", (0, 255, 0))):
                start = config.COLS_LHAND.start if hand == "left" else config.COLS_RHAND.start
                points = row[start:start + 21, :2]
                for a, b in ((0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)):
                    if np.isfinite(points[[a,b], :]).all():
                        xa, xb = points[a,0], points[b,0]
                        if not args.no_mirror: xa, xb = 1-xa, 1-xb
                        cv2.line(display, (int(xa*w),int(points[a,1]*h)), (int(xb*w),int(points[b,1]*h)), color, 2, cv2.LINE_AA)
                tip = points[8]
                if np.isfinite(tip).all():
                    x = 1-tip[0] if not args.no_mirror else tip[0]
                    cv2.circle(display, (int(x*w), int(tip[1]*h)), 7, (0,0,255), -1)
            if selected == "Holistic":
                for x, y in row[config.COLS_FACE, :2]:
                    if np.isfinite((x, y)).all():
                        px = 1-x if not args.no_mirror else x
                        cv2.circle(display, (int(px*w), int(y*h)), 1, (255, 160, 220), -1)
            pose_links = ((11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),(15,17),(15,19),(16,18),(16,20))
            for a, b in pose_links:
                if np.isfinite(row[[a,b], :2]).all():
                    xa, xb = row[a,0], row[b,0]
                    if not args.no_mirror: xa, xb = 1-xa, 1-xb
                    cv2.line(display, (int(xa*w),int(row[a,1]*h)), (int(xb*w),int(row[b,1]*h)), (255,180,0), 2, cv2.LINE_AA)
            fps = 1 / np.mean(cap_intervals) if cap_intervals else 0
            latency = (time.perf_counter() - captured) * 1000
            cv2.putText(display, f"{selected} | capture {fps:.1f} fps | frame {1000/fps if fps else 0:.1f} ms | e2e {latency:.1f} ms | Tab switch | q quit",
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .55, (255,255,255), 2, cv2.LINE_AA)
            face_now = bool(np.isfinite(rows["Holistic"][config.COLS_FACE, :2]).all(axis=1).any())
            cv2.putText(display, f"Holistic face: {'FOUND' if face_now else 'NOT FOUND'}",
                        (8, 48), cv2.FONT_HERSHEY_SIMPLEX, .5,
                        (70, 230, 90) if face_now else (40, 80, 255), 2, cv2.LINE_AA)
            cv2.putText(display, f"frame {w}x{h}; detection {args.detection_confidence:.2f}; tracking {args.tracking_confidence:.2f}",
                        (8, 68), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1, cv2.LINE_AA)
            label = word_result.get("display") or "show a sign"
            cv2.putText(display, f"{word_result['state']}  |  {label.upper()}",
                        (8, 96), cv2.FONT_HERSHEY_SIMPLEX, .7,
                        (70, 230, 90) if label not in ("uncertain", "show a sign") else (40, 180, 255),
                        2, cv2.LINE_AA)
            top3 = "   ".join(f"{word}: {prob:.2f}" for word, prob in word_result.get("top3", [])[:3])
            if top3:
                cv2.putText(display, top3, (8, 122), cv2.FONT_HERSHEY_SIMPLEX,
                            .55, (255, 255, 255), 1, cv2.LINE_AA)
            if word_result.get("text"):
                cv2.putText(display, "Words: " + word_result["text"][-50:],
                            (8, h - 14), cv2.FONT_HERSHEY_SIMPLEX, .6,
                            (70, 230, 90), 2, cv2.LINE_AA)
            cv2.imshow("Live tracking diagnostic (memory only)", display)
            key = cv2.waitKey(1) & 0xff
            if key in (27, ord('q')):
                break
            if key == 9 and hp is not None:
                selected = "Hands+Pose" if selected == "Holistic" else "Holistic"
    finally:
        stream.close(); cap.release()
        holistic.close()
        if hp is not None:
            hp.close()
        word_model.close()
        cv2.destroyAllWindows()

    print("\nCurrent pipeline: MediaPipe Tasks HolisticLandmarker, VIDEO mode, detect_for_video; monotonic perf_counter timestamps in ms. Camera BGR -> contiguous RGB; no resize/crop in software, requested size via OpenCV (camera may negotiate another size). Normalized landmarks map directly to that frame; mirror is a consistent frame flip and x -> 1-x. Camera queue requested at 1 and a latest-frame reader drains stale frames. Holistic complexity is fixed by the bundled Tasks model; detection/tracking thresholds are passed as Tasks detection/landmark confidence options. Hands+Pose comparison uses legacy MediaPipe Solutions VIDEO stream, RGB input, no timestamps, same resolution, same mirror mapping; complexity and both thresholds are configurable.")
    print("\nCurrent pipeline: MediaPipe Tasks HolisticLandmarker, VIDEO mode, detect_for_video; monotonic perf_counter timestamps in ms. Camera BGR -> contiguous RGB; no resize/crop in software, requested size via OpenCV (camera may negotiate another size). Normalized landmarks map directly to that frame; mirror is a consistent frame flip and x -> 1-x. Camera queue requested at 1 and a latest-frame reader drains stale frames. Holistic complexity is fixed by the bundled Tasks model; detection/landmark confidence thresholds are configurable. Hands+Pose, when available, uses legacy MediaPipe Solutions stream mode, RGB input, no timestamps, same resolution and mirror mapping; complexity and both thresholds are configurable.")
    print(f"Camera resolution: requested {args.width}x{args.height}, negotiated {negotiated_width}x{negotiated_height}; OpenCV buffer property at start: {camera_buffer:.0f}; frames captured: {stream.count} (no frame data saved).")
    print("Tracker       hand found overall    hand found above shoulders    face found overall    tip movement px/frame    processing ms/frame")
    for name, s in stats.items():
        overall = 100*s['found']/s['n'] if s['n'] else 0
        above = 100*s['above_found']/s['above_n'] if s['above_n'] else float('nan')
        jitter = float(np.median(s['jitter'])) if s['jitter'] else float('nan')
        cost = float(np.median(s['cost'])) if s['cost'] else float('nan')
        face = 100*s['face_found']/s['n'] if s['n'] and name == "Holistic" else float('nan')
        face_text = f"{face:.1f}%" if math.isfinite(face) else "n/a"
        print(f"{name:<14}{overall:>7.1f}%{(above if math.isfinite(above) else 0):>25.1f}%{face_text:>21}{jitter:>24.1f}{cost:>26.1f}")
    if hp is None:
        print("Hands+Pose      unavailable: installed MediaPipe 1.2.2 is Tasks-only; legacy mp.solutions was removed. No packages or model files were installed/downloaded.")


if __name__ == "__main__":
    main()
