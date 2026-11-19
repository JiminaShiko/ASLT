# evaluate_official_photos.py - test the model on the 29 official ASL Alphabet photos
# (A_test.jpg ... nothing_test.jpg). These photos are NOT used for training.
import os

import numpy as np
import tensorflow as tf
from PIL import Image

import preprocess

MODEL_PATH = "sign_photo_model.keras"
ASL_DIR = os.path.join("dataset", "asl_alphabet")

CLASS_NAMES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["space", "del", "nothing"]
CLASS_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def find_asl_folder():
    if os.path.isdir(ASL_DIR) and len(os.listdir(ASL_DIR)) > 0:
        return ASL_DIR
    import kagglehub
    return kagglehub.dataset_download("grassknoted/asl-alphabet")


def find_test_images(folder):
    found = []
    for root, _, files in os.walk(folder):
        for file_name in files:
            if not file_name.lower().endswith(IMAGE_EXTENSIONS):
                continue
            if "_test" not in file_name.lower():
                continue
            path = os.path.join(root, file_name)
            name = file_name.split("_")[0]
            name = name.lower() if name.lower() in ["space", "del", "nothing"] else name.upper()
            if name in CLASS_ID:
                found.append((path, name))
    return sorted(found)


def main():
    folder = find_asl_folder()
    photos = find_test_images(folder)
    if not photos:
        print("No *_test images found inside", folder)
        raise SystemExit

    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model: {MODEL_PATH}")
    print(f"Found {len(photos)} official test photos\n")

    correct = 0
    for path, real_name in photos:
        image = Image.open(path).convert("RGB")
        processed, hand_found = preprocess.preprocess_image(image)
        batch = processed.astype("float32").reshape(1, preprocess.IMG_SIZE, preprocess.IMG_SIZE, 1)
        prediction = model(batch, training=False).numpy()[0]

        guessed = CLASS_NAMES[int(np.argmax(prediction))]
        confidence = float(prediction[int(np.argmax(prediction))])
        ok = guessed == real_name
        correct += int(ok)
        status = "OK  " if ok else "WRONG"
        hand = "hand" if hand_found else "NO HAND"
        print(f"{status} {real_name:8s} -> {guessed:8s} {confidence * 100:6.2f}%  ({hand})  {os.path.basename(path)}")

    print("\n===================================================================")
    print(f"Correct: {correct} / {len(photos)}")
    print("===================================================================")


if __name__ == "__main__":
    main()
