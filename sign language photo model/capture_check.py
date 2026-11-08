"""Capture real webcam frames while the user signs problem letters, and save
what the app's live pipeline actually feeds to the model (raw, live-processed,
photo-processed) plus predictions. Run from the photo model folder."""
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
OUT_DIR = os.path.join(os.environ.get("TEMP", "/tmp"), "sign_capture")
os.makedirs(OUT_DIR, exist_ok=True)

model = tf.keras.models.load_model(MODEL_PATH)


def predict(processed):
    batch = np.asarray(processed, dtype="float32").reshape(1, 96, 96, 1)
    return model(batch, training=False).numpy()[0]


def top3(probs):
    if probs is None:
        return "", "", ""
    order = np.argsort(probs)[::-1][:3]
    return tuple(f"{CLASS_NAMES[int(i)]} {probs[int(i)] * 100:.0f}%" for i in order)


cam = cv2.VideoCapture(0)
if not cam.isOpened():
    cam = cv2.VideoCapture(1)
if not cam.isOpened():
    raise SystemExit("No camera found.")

print("Camera opened. Keep your hand at the SAME distance and angle as when")
print("the app misread the letters. Hold each letter still when prompted.")
time.sleep(1.0)

letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
log_rows = []
frame_id = 0
for letter in letters:
    print(f"\n>>> SIGN NOW: {letter}  (hold it still for ~4 seconds) <<<")
    end = time.monotonic() + 4.0
    tick = 0
    while time.monotonic() < end:
        ok, frame = cam.read()
        if not ok:
            continue
        tick += 1
        if tick % 12 != 0:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        proc_v, hand_v = preprocess.preprocess_image(Image.fromarray(rgb), video=True)
        proc_p, hand_p = preprocess.preprocess_image(Image.fromarray(rgb), video=False)
        probs_v = predict(proc_v) if hand_v else None
        probs_p = predict(proc_p) if hand_p else None
        cv2.imwrite(os.path.join(OUT_DIR, f"f{frame_id:03d}_{letter}_raw.jpg"), frame)
        if hand_v:
            cv2.imwrite(os.path.join(OUT_DIR, f"f{frame_id:03d}_{letter}_live.png"), proc_v)
        if hand_p:
            cv2.imwrite(os.path.join(OUT_DIR, f"f{frame_id:03d}_{letter}_photo.png"), proc_p)
        t1, t2, t3 = top3(probs_v)
        p1, p2, p3 = top3(probs_p)
        log_rows.append([frame_id, letter, hand_v, t1, t2, t3, hand_p, p1, p2, p3])
        print(f"  frame {frame_id} held={letter} live_hand={hand_v} live_top=({t1}, {t2}, {t3}) "
              f"photo_top=({p1}, {p2}, {p3})")
        frame_id += 1

cam.release()
with open(os.path.join(OUT_DIR, "log.csv"), "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "held", "live_hand", "live_1", "live_2", "live_3",
                     "photo_hand", "photo_1", "photo_2", "photo_3"])
    writer.writerows(log_rows)
print("\nDONE. Files and log.csv are in", OUT_DIR)
