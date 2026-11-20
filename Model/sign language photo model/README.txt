SIGN LANGUAGE PHOTO MODEL (SIGN LANGUAGE MNIST + ASL ALPHABET)
===============================================================
This project uses:
-Python
-TensorFlow / Keras
-Sign Language MNIST dataset and ASL Alphabet dataset (downloaded automatically)
-MobileNetV2 transfer learning (TensorFlow / Keras)
-MediaPipe Hand Landmarker (finds the hand and removes the background)
-streamlit (photo upload, camera photo and live webcam)

PROJECT FILES
==============
train_model.py                 trains the model
calibrate_camera.py            captures YOUR hand for every class and fine-tunes the model on it
preprocess.py                  shared hand detection / crop / background removal (used by training AND the app)
app.py                         the Streamlit app
evaluate_official_photos.py    tests the official ASL Alphabet test photos (28 files)

HTML TRANSLATOR API (LANDMARK MODEL)
====================================
The new website page at ../../Front-end/user_homepage/translation.html uses the
API blueprint in this folder's api.py. The root ASLT app registers that
blueprint, so run the website and translator together from the Backend folder:
	cd ../..                    # repository root
	python -m pip install -r requirements.txt
	cd Backend
	python app.py

The translator uses the separate landmark recogniser in
../asl_landmark_app; it does NOT use sign_photo_model.keras. Its camera frame
gate, smoothing, J/Z motion rules and live overlay are shared with
asl_landmark_app/src/live.py. Sign in to the website and open the Translation
page. Use localhost or HTTPS so the browser can access the camera.

For the old standalone HTML page only, api.py can still be run as a separate
local bridge from this folder. Do not run it at the same time as the root
website on port 5000.

The photo-model Streamlit app below (this folder's python app.py) is a separate
application and still uses sign_photo_model.keras.

COMMANDS
=========
All commands below must be run in THIS folder (the folder that contains app.py), except the dependency install in step 1, which runs from the repository root.
In PowerShell or Command Prompt, open it with:
	cd "D:\sign language project\ASTL project\ASLT\sign language photo model"
(If the project is saved somewhere else, use the folder where this README is.)

1. Install packages (the merged requirements.txt is at the repository root):
	cd ../..
	python -m pip install -r requirements.txt
	cd "Model/sign language photo model"
2. Train the model (downloads both datasets automatically the first time):
	python train_model.py
2b. Teach the model YOUR hand (optional but recommended - see CALIBRATION below):
	python calibrate_camera.py
	python train_model.py --calibrate-only
3. Run Streamlit:
	streamlit run app.py

WHAT THE MODEL DOES
====================
- Input: a gray image of a hand sign (96 x 96 pixels, hand cropped and background removed)
- Output: one of 29 classes: A-Z, space, del, nothing
- Sign Language MNIST (28 x 28, letters without J and Z) and ASL Alphabet (200 x 200 colour photos, 29 classes)
  are both converted to 96 x 96 images by preprocess.py and trained together in one MobileNetV2 model.
- The first time preprocess.py runs it downloads hand_landmarker.task (about 8 MB).

TRAIN / VALIDATION / TEST DATA
===============================
- Train (the model learns from it):
    Sign MNIST: 90% of the official train file
    ASL Alphabet: first 70% of every class
- Validation (used during training to stop at the best epoch):
    Sign MNIST: 10% of the official train file
    ASL Alphabet: next 15% of every class
- Test (never used for training, only for the final result):
    Sign MNIST: the official test file (7,172 images)
    ASL Alphabet: last 15% of every class + the official test images
- ASL Alphabet frames where MediaPipe cannot find a hand are skipped (about 5% of the
  training images and 15-20% of the harder val/test frames).
  The app also reports "nothing" when it cannot find a hand, so those frames are not taught as letters.
- Sign MNIST images (light background) are inverted and contrast-stretched, so they look
  like the app's bright hand on a black background.
- The results of the two test sets are shown separately, because ASL Alphabet images are video frames
  and its accuracy is usually higher than the real accuracy with a new person or camera.

FILES CREATED BY THE TRAINING
==============================
sign_photo_model.keras                  the trained model (used by app.py)
dataset/photo_data_cache.npz            the prepared 96 x 96 images (the next training starts faster)
dataset/user_calibration/               YOUR hand for every class (made by calibrate_camera.py)
training_results/checkpoints/           temporary phase weights, deleted automatically after training
training_results/training_curves.png    accuracy and loss charts
training_results/confusion_matrix.png   which letters are confused
training_results/test_results.txt       accuracy of the test sets and of every class

CALIBRATION (TEACH THE MODEL YOUR HAND)
========================================
The training photos are other people's hands, so the app can misread YOUR hand
(for example fist letters like A / E / M / N can all look like S at your camera
angle). Calibration fixes this:

1.  Run the capture tool. A preview window opens with the same green box as the
    app. For each class, sign it the way YOU sign it in the app (same distance,
    same angle), click the window and press ENTER or SPACE, then hold the sign
    for 6 seconds. "nothing" = take your hand away. q quits (what was captured
    is kept), b redoes the previous class.
        python calibrate_camera.py
    (If you sign with your LEFT hand in the app, add:  --left)
2.  Teach the model your hand. This fine-tunes sign_photo_model.keras on your
    images plus a slice of the original training data, so nothing is forgotten:
        python train_model.py --calibrate-only
    The old and new accuracies are printed (your hand / Sign MNIST test /
    ASL Alphabet test), so you can check nothing got worse.
3.  Restart the app and try the letters in the Live Webcam tab.
Running plain "python train_model.py" also fine-tunes at the end automatically
when dataset/user_calibration has images. To undo the calibration, delete the
dataset/user_calibration folder and retrain.

THE APP
========
Tab 1 - Upload Photo: upload an image of a hand sign, the app shows the letter, its meaning and the top 5 predictions.
Tab 2 - Camera Photo: take a photo with the camera and get the prediction.
Tab 3 - Live Webcam: click START and show the hand sign inside the green box, like the training photos.
        The green box is a guide only - the app reads the whole picture, so a hand outside the box still works.
        Keep the palm or the back of your hand turned towards the camera: a hand seen from the side looks
        like a different letter.
        The app shows the sign name, the percentage and the top 5 predictions.
        When the same letter is seen for a moment, it is typed into the text box.
        "space" adds a space, "del" deletes a letter, "nothing" means no hand.
        Turn on "I sign with my LEFT hand" if you are left-handed, and "Mirror the webcam" for the selfie view.
Tips: use good light, keep the whole hand in the centre. The app shows the processed 96 x 96 image that the
model sees, which makes it easy to check that the hand was found correctly.

SETTINGS (top of train_model.py)
=================================
MAX_ALPHABET_PER_CLASS = None   use fewer images for a faster training (None = all images)
EPOCHS_PHASE1 = 12              epochs with the frozen MobileNetV2 base
EPOCHS_PHASE2 = 12              epochs of fine-tuning the last MobileNetV2 layers
IMG_SIZE = 96                   image size (defined in preprocess.py, used by training and the app)
If the automatic download fails, download the datasets manually and extract them into
dataset/sign_mnist/ and dataset/asl_alphabet/
