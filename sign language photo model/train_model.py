# Sign Language Photo Model - train a MobileNetV2 transfer-learning CNN
# on Sign Language MNIST + ASL Alphabet, both preprocessed with preprocess.py
# (MediaPipe hand crop + black background). Two training phases:
#   1. only the new classification head is trained (frozen base)
#   2. the top blocks of the base are fine-tuned with a small learning rate
# The better of the two phase checkpoints (by validation accuracy) becomes
# sign_photo_model.keras, the file that app.py loads.
import os
import re
import shutil
import string

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from PIL import Image
from tensorflow.keras import layers, models  # type: ignore

import preprocess

#------------------------------------------
# 2. Settings
#------------------------------------------
DATA_DIR = "dataset"
RESULTS_DIR = "training_results"
MODEL_PATH = "sign_photo_model.keras"
CACHE_FILE = os.path.join(DATA_DIR, "photo_data_cache.npz")   # prepared 96 x 96 images
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")     # temporary, deleted after training
PHASE1_PATH = os.path.join(CHECKPOINT_DIR, "phase1.keras")
PHASE2_PATH = os.path.join(CHECKPOINT_DIR, "phase2.keras")

SIGN_MNIST_DIR = os.path.join(DATA_DIR, "sign_mnist")
ASL_ALPHABET_DIR = os.path.join(DATA_DIR, "asl_alphabet")

IMG_SIZE = preprocess.IMG_SIZE   # 96 x 96 gray
MAX_ALPHABET_PER_CLASS = None    # images per ASL class used (None = all 3000, slower, a bit better)
EPOCHS_PHASE1 = 12               # maximum epochs of phase 1 (early stopping can stop sooner)
EPOCHS_PHASE2 = 12               # maximum epochs of phase 2
BATCH_SIZE = 64

CLASS_NAMES = list(string.ascii_uppercase) + ["space", "del", "nothing"]
CLASS_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

np.random.seed(42)
tf.random.set_seed(42)

# Download a dataset from Kaggle (or use the local folder if it already has files)
def get_dataset_folder(local_folder, kaggle_name):
    if os.path.isdir(local_folder) and len(os.listdir(local_folder)) > 0:
        return local_folder
    print(f"Downloading {kaggle_name} from Kaggle........")
    try:
        import kagglehub
        return kagglehub.dataset_download(kaggle_name)
    except Exception as error:
        print("Automatic download failed:", error)
        print("Please download the dataset manually from https://www.kaggle.com/datasets/" + kaggle_name)
        print("and extract it inside the folder:", local_folder)
        raise SystemExit

def find_file(folder, file_name):
    for root, subfolders, files in os.walk(folder):
        if file_name in files:
            return os.path.join(root, file_name)
    print("Cannot find", file_name, "inside", folder)
    raise SystemExit

# Load ASL Alphabet photos with the SAME preprocessing the app uses:
# MediaPipe hand crop, background removal, 96 x 96 black canvas.
# Frames where MediaPipe cannot find a hand are SKIPPED (about 5% of the
# training images and 15-20% of the harder val/test frames): the app also
# reports "nothing" when it cannot find a hand, so those frames must not be
# taught as letters. "nothing" is trained on the full frame, because that class
# means "there is no hand".
def preprocess_asl_split(pairs, part_name):
    images, labels, skipped = [], [], 0
    for i, (path, label) in enumerate(pairs):
        image = Image.open(path).convert("RGB")
        if CLASS_NAMES[label] == "nothing":
            images.append(preprocess.pad_to_square(np.array(image.convert("L"))))
            labels.append(label)
        else:
            processed, hand_found = preprocess.preprocess_image(image)
            if hand_found:
                images.append(processed)
                labels.append(label)
            else:
                skipped += 1
        if (i + 1) % 1000 == 0:
            print(f"  preprocessed {i + 1} of {len(pairs)} {part_name} images")
    print(f"  {part_name}: kept {len(images)}, skipped {skipped} frames where no hand was found")
    return np.array(images, dtype="uint8"), np.array(labels)

# Sign MNIST: the CSV images have a LIGHT background, the opposite of the ASL
# hand crops and of the app. Invert them, stretch the contrast, apply CLAHE and
# add a black margin, so the input looks like the app's bright-hand-on-black.
def resize_mnist(rows):
    images = np.zeros((len(rows), IMG_SIZE, IMG_SIZE), dtype="uint8")
    inner = int(IMG_SIZE * 0.875)          # 84 x 84 hand, 6 px black margin
    offset = (IMG_SIZE - inner) // 2
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    for i, row in enumerate(rows):
        inverted = 255 - row.reshape(28, 28).astype("uint8")
        up = cv2.resize(inverted, (inner, inner), interpolation=cv2.INTER_CUBIC)
        low, high = np.percentile(up, 2), np.percentile(up, 98)
        up = np.clip((up.astype("float32") - low) * 255.0 / max(high - low, 1), 0, 255).astype("uint8")
        images[i, offset:offset + inner, offset:offset + inner] = clahe.apply(up)
    return images

def number_in_name(path):
    numbers = re.findall(r"\d+", os.path.basename(path))
    return int(numbers[-1]) if numbers else 0

#------------------------------------------
# 3. Load the Data (use the saved cache if it exists)
#------------------------------------------
if os.path.exists(CACHE_FILE):
    print("Loading the saved data from", CACHE_FILE, "........")
    cache = np.load(CACHE_FILE)
    x_train, y_train = cache["x_train"], cache["y_train"]
    x_val, y_val = cache["x_val"], cache["y_val"]
    x_test_mnist, y_test_mnist = cache["x_test_mnist"], cache["y_test_mnist"]
    x_test_alpha, y_test_alpha = cache["x_test_alpha"], cache["y_test_alpha"]
else:
    print("Building the preprocessed data cache (one time, about 30-35 minutes)........")

    mnist_folder = get_dataset_folder(SIGN_MNIST_DIR, "datamunge/sign-language-mnist")
    mnist_train = pd.read_csv(find_file(mnist_folder, "sign_mnist_train.csv"))
    mnist_test = pd.read_csv(find_file(mnist_folder, "sign_mnist_test.csv"))
    print("Sign MNIST train:", mnist_train.shape, "| test:", mnist_test.shape)

    def mnist_labels(label_column):
        return np.array([CLASS_ID[chr(65 + int(label))] for label in label_column])

    x_mnist_train = resize_mnist(mnist_train.drop(columns="label").to_numpy())
    y_mnist_train = mnist_labels(mnist_train["label"])
    x_test_mnist = resize_mnist(mnist_test.drop(columns="label").to_numpy())
    y_test_mnist = mnist_labels(mnist_test["label"])

    order = np.random.permutation(len(x_mnist_train))
    n_val = int(0.1 * len(order))
    val_index, train_index = order[:n_val], order[n_val:]
    x_mnist_val, y_mnist_val = x_mnist_train[val_index], y_mnist_train[val_index]
    x_mnist_train, y_mnist_train = x_mnist_train[train_index], y_mnist_train[train_index]

    alpha_folder = get_dataset_folder(ASL_ALPHABET_DIR, "grassknoted/asl-alphabet")

    train_files = {}
    official_test = []
    skipped = 0
    for root, subfolders, files in os.walk(alpha_folder):
        for file_name in files:
            if not file_name.lower().endswith(IMAGE_EXTENSIONS):
                continue
            path = os.path.join(root, file_name)
            is_test = "test" in os.path.relpath(path, alpha_folder).lower()
            name = file_name.split("_")[0] if is_test else os.path.basename(root)
            name = name.lower() if name.lower() in ["space", "del", "nothing"] else name.upper()
            if name not in CLASS_ID:
                skipped += 1
                continue
            if is_test:
                official_test.append((path, CLASS_ID[name]))
            else:
                train_files.setdefault(name, []).append(path)

    print("ASL Alphabet classes found:", len(train_files), "| skipped files:", skipped)

    # Temporal split per class: first 70% train, next 15% validation, last 15% test.
    # (The images are video frames, so a random split would leak almost identical
    # images into the test set and the accuracy would look too good.)
    alpha_split = {"train": [], "val": [], "test": []}
    for name, paths in train_files.items():
        paths = sorted(paths, key=number_in_name)
        if MAX_ALPHABET_PER_CLASS is not None and len(paths) > MAX_ALPHABET_PER_CLASS:
            keep = np.linspace(0, len(paths) - 1, MAX_ALPHABET_PER_CLASS).astype(int)
            paths = [paths[i] for i in keep]
        n_train = int(0.70 * len(paths))
        n_val = int(0.15 * len(paths))
        alpha_split["train"] += [(p, CLASS_ID[name]) for p in paths[:n_train]]
        alpha_split["val"] += [(p, CLASS_ID[name]) for p in paths[n_train:n_train + n_val]]
        alpha_split["test"] += [(p, CLASS_ID[name]) for p in paths[n_train + n_val:]]

    alpha_split["test"] += official_test

    arrays = {}
    for part in ["train", "val", "test"]:
        print(f"Preprocessing ASL Alphabet {part} images (MediaPipe)........")
        arrays[part] = preprocess_asl_split(alpha_split[part], part)

    x_test_alpha, y_test_alpha = arrays["test"]

    x_train = np.concatenate([x_mnist_train, arrays["train"][0]])
    y_train = np.concatenate([y_mnist_train, arrays["train"][1]])
    x_val = np.concatenate([x_mnist_val, arrays["val"][0]])
    y_val = np.concatenate([y_mnist_val, arrays["val"][1]])

    np.savez_compressed(CACHE_FILE, x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val,
                        x_test_mnist=x_test_mnist, y_test_mnist=y_test_mnist,
                        x_test_alpha=x_test_alpha, y_test_alpha=y_test_alpha)
    print("Data saved in:", CACHE_FILE)

print("\nTrain images:     ", x_train.shape)
print("Validation images:", x_val.shape)
print("Test images (Sign MNIST):  ", x_test_mnist.shape)
print("Test images (ASL Alphabet):", x_test_alpha.shape)

#------------------------------------------
# 7. Check the Classes
#------------------------------------------
counts = np.bincount(y_train, minlength=len(CLASS_NAMES))
print("\nTraining images per class:")
for name, count in zip(CLASS_NAMES, counts):
    print(f"  {name:8s} {count}")

x_train = x_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
x_val = x_val.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
x_test_mnist = x_test_mnist.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
x_test_alpha = x_test_alpha.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

#------------------------------------------
# 9. Class Weights
#------------------------------------------
total = len(y_train)
class_weight = {i: total / (len(CLASS_NAMES) * counts[i]) for i in range(len(CLASS_NAMES)) if counts[i] > 0}

#------------------------------------------
# 10. TensorFlow Datasets
#------------------------------------------
def make_dataset(x, y, shuffle):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        dataset = dataset.shuffle(10000, seed=42)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.map(lambda a, b: (tf.cast(a, tf.float32), b))
    return dataset.prefetch(tf.data.AUTOTUNE)

train_data = make_dataset(x_train, y_train, True)
val_data = make_dataset(x_val, y_val, False)
test_data_mnist = make_dataset(x_test_mnist, y_test_mnist, False)
test_data_alpha = make_dataset(x_test_alpha, y_test_alpha, False)

#------------------------------------------
# 11. Model: MobileNetV2 transfer learning
#------------------------------------------
# Gray input is repeated 3x to fake an RGB image for the pretrained base.
# Augmentation (NO horizontal flip - mirroring is handled in the app instead,
# because mirrored ASL signs look like different letters).
# The base expects values in [-1, 1], which the Rescaling(1/127.5, offset=-1)
# layer produces from 0-255 input.
base = tf.keras.applications.MobileNetV2(
    include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3))
base.trainable = False

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
x = layers.RandomRotation(0.10, fill_mode="constant", fill_value=0.0)(inputs)
x = layers.RandomTranslation(0.1, 0.1, fill_mode="constant", fill_value=0.0)(x)
x = layers.RandomZoom((0.0, 0.3), (0.0, 0.3), fill_mode="constant", fill_value=0.0)(x)
x = layers.RandomContrast(0.2)(x)
x = layers.RandomBrightness(0.2, value_range=(0, 255))(x)
x = layers.Concatenate()([x, x, x])
x = layers.Rescaling(1.0 / 127.5, offset=-1.0)(x)
x = base(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(len(CLASS_NAMES), activation="softmax")(x)

model = models.Model(inputs, outputs)
model.summary()

#------------------------------------------
# 12. Phase 1: train only the new head
#------------------------------------------
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

phase1_callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4, min_delta=0.001,
                                     restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2,
                                         min_lr=1e-6, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(PHASE1_PATH, monitor="val_accuracy",
                                       save_best_only=True, verbose=1),
]

print("\nPhase 1: training the classification head (frozen base)........\n")
history1 = model.fit(train_data, validation_data=val_data, epochs=EPOCHS_PHASE1,
                     class_weight=class_weight, callbacks=phase1_callbacks)

#------------------------------------------
# 13. Phase 2: fine-tune the top of the base
#------------------------------------------
base.trainable = True
for layer in base.layers[:100]:   # keep the first 100 layers frozen
    layer.trainable = False
# BatchNormalization must stay frozen: updating its running statistics on the
# augmented grayscale images made the model much worse at the start of phase 2.
for layer in base.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

phase2_callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, min_delta=0.001,
                                     restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2,
                                         min_lr=1e-6, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(PHASE2_PATH, monitor="val_accuracy",
                                       save_best_only=True, verbose=1),
]

print("\nPhase 2: fine-tuning the top blocks of the base........\n")
history2 = model.fit(train_data, validation_data=val_data, epochs=EPOCHS_PHASE2,
                     class_weight=class_weight, callbacks=phase2_callbacks)

#------------------------------------------
# 14. Keep the better of the two phase checkpoints
#------------------------------------------
phase1 = tf.keras.models.load_model(PHASE1_PATH)
phase2 = tf.keras.models.load_model(PHASE2_PATH)
acc1 = phase1.evaluate(val_data, verbose=0)[1]
acc2 = phase2.evaluate(val_data, verbose=0)[1]
print(f"\nPhase 1 validation accuracy: {acc1 * 100:.2f}%")
print(f"Phase 2 validation accuracy: {acc2 * 100:.2f}%")

best = phase2 if acc2 >= acc1 else phase1
best.save(MODEL_PATH)
shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)
print("Best model saved as:", MODEL_PATH, "(from phase", "2" if acc2 >= acc1 else "1", ")")

#------------------------------------------
# 15. Test the Best Model
#------------------------------------------
loss_mnist, acc_mnist = best.evaluate(test_data_mnist, verbose=0)
loss_alpha, acc_alpha = best.evaluate(test_data_alpha, verbose=0)

x_test_all = np.concatenate([x_test_mnist, x_test_alpha])
y_test_all = np.concatenate([y_test_mnist, y_test_alpha])

predictions = best.predict(make_dataset(x_test_all, y_test_all, False), verbose=0)
predicted = np.argmax(predictions, axis=1)
acc_all = float((predicted == y_test_all).mean())

print("\n===================================================================")
print("TEST RESULTS")
print("===================================================================")
print(f"Sign MNIST test accuracy:    {acc_mnist * 100:.2f}%  ({len(y_test_mnist)} images)")
print(f"ASL Alphabet test accuracy:  {acc_alpha * 100:.2f}%  ({len(y_test_alpha)} images)")
print(f"All test images together:    {acc_all * 100:.2f}%  ({len(y_test_all)} images)")

print("\nAccuracy per class:")
lines = []
for i, name in enumerate(CLASS_NAMES):
    mask = y_test_all == i
    if mask.sum() > 0:
        line = f"  {name:8s} {(predicted[mask] == i).mean() * 100:6.2f}%   ({mask.sum()} test images)"
        print(line)
        lines.append(line)

#------------------------------------------
# 16. Save Charts and Results
#------------------------------------------
# Both phases on one chart (concatenated epochs)
acc = history1.history["accuracy"] + history2.history["accuracy"]
val_acc = history1.history["val_accuracy"] + history2.history["val_accuracy"]
train_loss = history1.history["loss"] + history2.history["loss"]
val_loss = history1.history["val_loss"] + history2.history["val_loss"]

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(acc, label="Train")
plt.plot(val_acc, label="Validation")
plt.title("Model Accuracy (both phases)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_loss, label="Train")
plt.plot(val_loss, label="Validation")
plt.title("Model Loss (both phases)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "training_curves.png"), dpi=150)
plt.close()

matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int)
for real, guess in zip(y_test_all, predicted):
    matrix[real, guess] += 1

plt.figure(figsize=(10, 9))
plt.imshow(matrix, cmap="Blues")
plt.colorbar(label="Number of images")
plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=90)
plt.yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
plt.title("Confusion Matrix (all test images)")
plt.xlabel("Predicted class")
plt.ylabel("Real class")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=150)
plt.close()

with open(os.path.join(RESULTS_DIR, "test_results.txt"), "w") as file:
    file.write(f"Sign MNIST test accuracy:   {acc_mnist * 100:.2f}% ({len(y_test_mnist)} images)\n")
    file.write(f"ASL Alphabet test accuracy: {acc_alpha * 100:.2f}% ({len(y_test_alpha)} images)\n")
    file.write(f"All test images together:   {acc_all * 100:.2f}% ({len(y_test_all)} images)\n\n")
    file.write("Accuracy per class:\n" + "\n".join(lines) + "\n")

print("\nBest model saved as:", MODEL_PATH)
print("Charts and results saved in:", RESULTS_DIR)
