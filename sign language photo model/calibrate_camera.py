"""Teach the model YOUR hand.

This tool captures webcam frames of every class through the EXACT same
pipeline the live webcam uses (shrink to 640, video-mode hand tracking,
whole-frame prediction, background removal) and saves the resulting 96 x 96
model inputs into dataset/user_calibration/<class>/.

Afterwards run:  python train_model.py --calibrate-only
which fine-tunes sign_photo_model.keras on these images (plus a slice of the
original training data so nothing is forgotten).

Controls per class: pose your hand in the green box exactly like in the app
(same distance, same angle, same way YOU sign it), click the preview window
and press ENTER or SPACE to start the 6-second hold.
- q  = quit early (everything captured so far is kept)
- b  = go back and redo the previous class (its images are replaced)

If you sign with your LEFT hand in the app (checkbox "I sign with my LEFT
hand"), run:  python calibrate_camera.py --left
"""
import argparse
import csv
import os
import time

import cv2
import numpy as np
from PIL import Image

import preprocess
import tensorflow as tf

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sign_photo_model.keras")
CLASS_NAMES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["space", "del", "nothing"]
CALIBRATION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "dataset", "user_calibration")

DETECT_MAX_SIDE = 640   # same as app.py
BOX_SIZE = 0.45         # same as app.py
HOLD_SECONDS = 6.0
MAX_PER_CLASS = 12      # saved 96 x 96 images per class

PROMPTS = {
    "space": "space (your open-hand sign for a space)",
    "del": "del (your sign for deleting a letter)",
    "nothing": "nothing (take your hand AWAY from the camera)",
}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--left", action="store_true",
                    help="flip frames like the app does when 'I sign with my LEFT hand' is on")
args = parser.parse_args()

model = tf.keras.models.load_model(MODEL_PATH)


def predict(processed):
    batch = np.asarray(processed, dtype="float32").reshape(1, 96, 96, 1)
    return model(batch, training=False).numpy()[0]


def top1(probs):
    order = np.argsort(probs)[::-1]
    return CLASS_NAMES[int(order[0])], float(probs[order[0]])


def draw_guide(frame, text, color=(0, 200, 0)):
    h, w = frame.shape[:2]
    side = int(min(h, w) * BOX_SIZE)
    x0, y0 = (w - side) // 2, (h - side) // 2
    cv2.rectangle(frame, (x0, y0), (x0 + side, y0 + side), color, 2)
    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return frame


cam = cv2.VideoCapture(0)
if not cam.isOpened():
    cam = cv2.VideoCapture(1)
if not cam.isOpened():
    raise SystemExit("No camera found.")
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

WIN = "calibration"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

print("A preview window opened. Sign each class the way YOU sign it in the app:")
print("same distance, same angle, in the green box. Press ENTER/SPACE to start")
print("the 6-second hold. q = quit, b = redo the previous class.")

counts = {name: 0 for name in CLASS_NAMES}
index = 0
while index < len(CLASS_NAMES):
    name = CLASS_NAMES[index]
    prompt = PROMPTS.get(name, f"letter {name}")
    class_dir = os.path.join(CALIBRATION_DIR, name)
    print(f"\n>>> NEXT: {prompt}  (click the window, then ENTER/SPACE; b = redo previous) <<<")

    while True:
        ok, frame = cam.read()
        if not ok:
            continue
        view = draw_guide(frame.copy(), f"NEXT: {name}   ENTER/SPACE = start, b = back, q = quit")
        cv2.imshow(WIN, view)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            cam.release()
            cv2.destroyAllWindows()
            print(f"\nQuit early. Images kept in {CALIBRATION_DIR}")
            raise SystemExit
        if key == ord("b"):
            if index > 0:
                index -= 1
                print(f"\nGoing back to redo {CLASS_NAMES[index]}")
            break
        if key in (13, 32):  # Enter or Space
            break
    if key == ord("b"):
        continue

    # A redo replaces this class's images
    if os.path.isdir(class_dir):
        for old in os.listdir(class_dir):
            os.remove(os.path.join(class_dir, old))
    os.makedirs(class_dir, exist_ok=True)

    print(f"  HOLD {name} ...")
    end = time.monotonic() + HOLD_SECONDS
    tick = 0
    saved = 0
    while time.monotonic() < end and saved < MAX_PER_CLASS:
        ok, frame = cam.read()
        if not ok:
            continue
        tick += 1
        left = max(0.0, end - time.monotonic())
        view = draw_guide(frame.copy(), f"HOLD {name}  {left:.0f}s  saved {saved}", (0, 0, 255))
        cv2.imshow(WIN, view)
        key = cv2.waitKey(5) & 0xFF
        if key == ord("q"):
            cam.release()
            cv2.destroyAllWindows()
            print(f"\nQuit early. Images kept in {CALIBRATION_DIR}")
            raise SystemExit
        if key == ord("b"):
            print("  aborted - redoing this class")
            break
        if tick % 10 != 0:
            continue

        # The app's exact live pipeline: shrink to 640, then whole-frame
        # detection with video-mode tracking. The LEFT-hand checkbox flips
        # the frame before the prediction, so calibration flips it too.
        height, width = frame.shape[:2]
        longest = max(height, width)
        if longest > DETECT_MAX_SIDE:
            scale = DETECT_MAX_SIDE / longest
            frame = cv2.resize(frame, (int(width * scale), int(height * scale)),
                               interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if args.left:
            rgb = cv2.flip(rgb, 1)

        processed, hand_found = preprocess.preprocess_image(Image.fromarray(rgb), video=True)
        if name == "nothing":
            keep = not hand_found          # nothing = no hand in the frame
        else:
            keep = hand_found
        if not keep:
            continue

        cv2.imwrite(os.path.join(class_dir, f"{saved:03d}.png"), processed)
        saved += 1
        if name != "nothing":
            guessed, conf = top1(predict(processed))
            print(f"  {name} {saved}: current model says {guessed} ({conf * 100:.0f}%)")
        else:
            print(f"  {name} {saved}: saved an empty frame")
    if key == ord("b"):
        continue

    counts[name] = saved
    index += 1

cam.release()
cv2.destroyAllWindows()

print("\nCalibration finished. Images per class:")
for name, count in counts.items():
    if count:
        print(f"  {name:8s} {count}")
print("\nNow teach the model your hand with:")
print('cd "D:\\sign language project\\ASTL project\\ASLT\\sign language photo model" && python train_model.py --calibrate-only')
