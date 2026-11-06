# preprocess.py - shared preprocessing for training AND the app.
# MediaPipe finds the hand, the background is removed, and the result is a
# 96 x 96 gray image with a bright hand on a black background.
# Verified 2026-11-05: finds hands in dataset photos (90-100% of frames),
# in camera-like photos down to 1/5 scale, and returns "no hand" for the
# "nothing" class (full frame is kept).
import os
import threading
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from PIL import Image

IMG_SIZE = 96
TASK_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
TASK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

_landmarker = None
# The app predicts from the Streamlit page thread AND the live video thread,
# and one HandLandmarker cannot be used from two threads at the same time.
_lock = threading.RLock()

# The live webcam uses MediaPipe VIDEO mode: it remembers the hand between
# frames, so the tracking is much more stable than a fresh detection on every
# frame. It is a separate landmarker because VIDEO mode needs frame timestamps.
_landmarker_video = None
_video_lock = threading.RLock()
_video_timestamp = 0


def get_landmarker():
    global _landmarker
    if _landmarker is None:
        if not os.path.exists(TASK_PATH):
            print("Downloading hand_landmarker.task (about 8 MB) ...")
            try:
                urllib.request.urlretrieve(TASK_URL, TASK_PATH)
            except Exception as error:
                print("Download failed:", error)
                print("Download it manually from:", TASK_URL)
                print("and save it as:", TASK_PATH)
                raise SystemExit
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=TASK_PATH),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.3)
        _landmarker = vision.HandLandmarker.create_from_options(options)
    return _landmarker


def _brighten(rgb, gamma=0.35):
    # Gamma < 1 lifts the shadows, so backlit / dark hands become detectable.
    table = ((np.arange(256) / 255.0) ** gamma * 255).astype("uint8")
    return cv2.LUT(rgb, table)


def _detect(rgb):
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    with _lock:
        result = get_landmarker().detect(image)
    if not result.hand_landmarks:
        return None
    return [(p.x, p.y) for p in result.hand_landmarks[0]]


def get_video_landmarker():
    global _landmarker_video
    if _landmarker_video is None:
        if not os.path.exists(TASK_PATH):
            get_landmarker()   # downloads hand_landmarker.task if needed
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=TASK_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.3)
        _landmarker_video = vision.HandLandmarker.create_from_options(options)
    return _landmarker_video


def _detect_video(rgb):
    """VIDEO mode detection. Timestamps must be strictly increasing, so every
    call (including retries) gets its own timestamp."""
    global _video_timestamp
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    with _video_lock:
        _video_timestamp = max(_video_timestamp + 1, int(time.monotonic() * 1000))
        result = get_video_landmarker().detect_for_video(image, _video_timestamp)
    if not result.hand_landmarks:
        return None
    return [(p.x, p.y) for p in result.hand_landmarks[0]]


def find_hand_landmarks_video(rgb):
    """Live webcam detection: VIDEO mode first (tracks the hand between
    frames), then the photo detector with all its retries as a fallback."""
    landmarks = _detect_video(rgb)
    if landmarks is None:
        landmarks = _detect_video(_brighten(rgb))
    if landmarks is None:
        landmarks = find_hand_landmarks(rgb)
    return landmarks


def find_hand_landmarks(rgb):
    """Normalized (x, y) list of the 21 hand landmarks, or None."""
    landmarks = _detect(rgb)
    if landmarks is None:
        # Retry once on a brightened copy (the coordinates are still valid
        # for the original image).
        landmarks = _detect(_brighten(rgb))
    if landmarks is None:
        # Last retries at other scales: a hand that fills the frame is hard
        # for the detector because it has no context, so shrink it first;
        # a small hand is easier to find after an upscale. The normalized
        # landmark coordinates do not depend on the scale of the input.
        height, width = rgb.shape[:2]
        longest = max(height, width)
        scales = []
        if longest >= 160:
            scales.append(0.6)
        if longest <= 800:
            scales.append(1.6)
        for scale in scales:
            resized = cv2.resize(rgb, None, fx=scale, fy=scale,
                                 interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
            landmarks = _detect(resized)
            if landmarks is None:
                landmarks = _detect(_brighten(resized))
            if landmarks is not None:
                break
    return landmarks


def remove_background(crop, crop_points=None):
    """Return the crop with a black background and a BRIGHT hand.

    crop_points: hand landmarks converted to crop pixel coordinates. They are
    guaranteed to be ON the hand, so they decide which side of the Otsu
    threshold is the hand, which connected component is the hand, and the final
    mask is limited to the region around them - a bright background (window,
    white wall) can never end up in the model's input.
    (A single centre pixel is not reliable - it can land in the gap between
    two fingers.)"""
    h, w = crop.shape
    _, mask = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    points = []
    if crop_points is not None:
        points = [(int(x), int(y)) for x, y in crop_points
                  if 0 <= x < w and 0 <= y < h]
    if points:
        bright_votes = sum(1 for x, y in points if mask[y, x] > 0)
        hand_is_bright = bright_votes * 2 >= len(points)
    else:
        hand_is_bright = mask[h // 2, w // 2] > 0

    if not hand_is_bright:
        crop = cv2.bitwise_not(crop)
        _, mask = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count > 1:
        votes = {}
        for x, y in points:
            label = int(labels[y, x])
            if label > 0:
                votes[label] = votes.get(label, 0) + 1
        if votes:
            hand_label = max(votes, key=votes.get)
            hand_mask = np.where(labels == hand_label, 255, 0).astype(np.uint8)
        else:
            centre_label = labels[h // 2, w // 2]
            if centre_label > 0:
                hand_mask = np.where(labels == centre_label, 255, 0).astype(np.uint8)
            else:
                biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                hand_mask = np.where(labels == biggest, 255, 0).astype(np.uint8)
    else:
        hand_mask = mask

    # The landmarks are ON the hand: keep only the part of the mask around
    # them, so background that Otsu merged with the hand is cut away.
    hull_mask = None
    if len(points) >= 3:
        hull = cv2.convexHull(np.array(points, dtype=np.int32).reshape(-1, 1, 2))
        hull_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(hull_mask, hull, 255)
        pad = max(3, int(0.12 * max(h, w)))   # cover the flesh around the landmarks
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1))
        hull_mask = cv2.dilate(hull_mask, kernel)
        hand_mask = cv2.bitwise_and(hand_mask, hull_mask)

    hand_fraction = (hand_mask > 0).mean()
    if not (0.08 <= hand_fraction <= 0.85):
        if hull_mask is None:
            return crop
        # Fall back to the landmark region alone (the brightness is already
        # normalized above), so the model still gets a bright hand on black.
        hand_mask = hull_mask
    out = crop.copy()
    out[hand_mask == 0] = 0
    return out


def preprocess_image(image, target=IMG_SIZE, video=False):
    """PIL RGB image -> (target x target uint8 numpy array, hand_found bool).
    hand_found is False when MediaPipe found no hand: the full frame is kept
    (that is exactly what the model's "nothing" class was trained on).
    video=True (live webcam) uses MediaPipe VIDEO mode with the photo
    detector as a fallback."""
    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape

    if video:
        landmarks = find_hand_landmarks_video(rgb)
    else:
        landmarks = find_hand_landmarks(rgb)
    if landmarks is None:
        crop = gray
    else:
        xs = [p[0] for p in landmarks]
        ys = [p[1] for p in landmarks]
        bx0, by0 = int(min(xs) * width), int(min(ys) * height)
        bw, bh = int((max(xs) - min(xs)) * width), int((max(ys) - min(ys)) * height)
        margin = int(max(bw, bh) * 0.25)
        cx, cy = bx0 + bw // 2, by0 + bh // 2
        half = max(bw, bh) // 2 + margin
        x0c, x1c = max(0, cx - half), min(width, cx + half)
        y0c, y1c = max(0, cy - half), min(height, cy + half)
        crop = gray[y0c:y1c, x0c:x1c]
        crop_points = [(p[0] * width - x0c, p[1] * height - y0c) for p in landmarks]
        crop = remove_background(crop, crop_points)

    scale = target / max(crop.shape)
    new = cv2.resize(crop, (int(round(crop.shape[1] * scale)), int(round(crop.shape[0] * scale))),
                     interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas = np.zeros((target, target), dtype=np.uint8)
    y0p, x0p = (target - new.shape[0]) // 2, (target - new.shape[1]) // 2
    canvas[y0p:y0p + new.shape[0], x0p:x0p + new.shape[1]] = new
    return canvas, landmarks is not None


def pad_to_square(gray, target=IMG_SIZE):
    """Aspect-preserving resize of a gray image onto a black square.
    Used for the ASL "nothing" class, which is a raw full frame without a hand."""
    scale = target / max(gray.shape)
    new = cv2.resize(gray, (int(round(gray.shape[1] * scale)), int(round(gray.shape[0] * scale))),
                     interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas = np.zeros((target, target), dtype=np.uint8)
    y0, x0 = (target - new.shape[0]) // 2, (target - new.shape[1]) // 2
    canvas[y0:y0 + new.shape[0], x0:x0 + new.shape[1]] = new
    return canvas


def selftest():
    """Process sample images and save them so a human can check the output.
    Expected: bright hand on a BLACK background; the "nothing" image stays a
    full frame."""
    out_dir = os.path.join("training_results", "preprocess_check")
    os.makedirs(out_dir, exist_ok=True)

    alpha_folder = os.path.join("dataset", "asl_alphabet")
    if not (os.path.isdir(alpha_folder) and len(os.listdir(alpha_folder)) > 0):
        import kagglehub
        alpha_folder = kagglehub.dataset_download("grassknoted/asl-alphabet")

    train_dir = None
    for root, dirs, files in os.walk(alpha_folder):
        if "A" in dirs and "nothing" in dirs:
            train_dir = root
            break
    if train_dir is None:
        raise SystemExit("Could not find the ASL Alphabet class folders for the selftest.")

    checks = [("A", True), ("L", True), ("nothing", False)]
    for cls, expect_hand in checks:
        class_dir = os.path.join(train_dir, cls)
        files = sorted(os.listdir(class_dir))[:20]
        saved = False
        for file_name in files:
            path = os.path.join(class_dir, file_name)
            image = Image.open(path)
            landmarks = find_hand_landmarks(np.array(image.convert("RGB")))
            if (landmarks is not None) != expect_hand:
                continue
            processed, hand_found = preprocess_image(image)
            out_path = os.path.join(out_dir, f"check_{cls}.png")
            cv2.imwrite(out_path, processed)
            black_frac = (processed < 10).mean()
            print(f"{cls:8s} file={file_name} hand_found={str(hand_found):5s} "
                  f"black_background={black_frac * 100:4.0f}%  ->  {out_path}")
            saved = True
            break
        if not saved:
            print(f"{cls:8s} no suitable frame found in the first 20 files - skipped")

    print("\nSELFTEST DONE. Open training_results/preprocess_check/ and verify:")
    print("  - check_A.png and check_L.png: bright hand on a black background")
    print("  - check_nothing.png: the full photo (no cropping, because there is no hand)")


if __name__ == "__main__":
    selftest()
