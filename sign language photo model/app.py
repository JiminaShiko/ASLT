#--------------------------------------------------
# 1. Import Libraries
#--------------------------------------------------
import threading
import time
from collections import deque

import av
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

# Import Image for image processing and ImageOps for image operations such as grayscale
from PIL import Image, ImageOps
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer

import preprocess

#--------------------------------------------------
# 2. Model Path and Settings
#--------------------------------------------------
# Define the path of the trained sign language photo model
MODEL_PATH = "sign_photo_model.keras"

# The image size of the model (must be the same as in preprocess.py)
IMG_SIZE = preprocess.IMG_SIZE

# The 29 classes of the model (must be the same list as in train_model.py)
CLASS_NAMES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["space", "del", "nothing"]

# What the special signs mean
MEANINGS = {
    "space": "SPACE - adds a space between words",
    "del": "DELETE - removes the last typed letter",
    "nothing": "No sign - there is no hand sign in the image",
    "-": "Waiting for a sign",
}

# Live webcam settings
MIN_CONFIDENCE = 0.60      # Predictions below 60% are not typed
STABLE_FRAMES = 10         # The same letter must be predicted 10 times in a row before it is typed
PREDICT_EVERY = 3          # Predict only every 3rd video frame (faster)

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

#--------------------------------------------------
# 3. Streamlit Page Configuration
#--------------------------------------------------
st.set_page_config(
    page_title = "Sign Language Classifier",
    page_icon = "🤟",
    layout = "centered"
)

#--------------------------------------------------
# 4. Load Trained Model
#--------------------------------------------------
@st.cache_resource # Cache the loaded model so Streamlit does not reload it every time the page changes
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

#--------------------------------------------------
# 5. Helper Functions
#--------------------------------------------------
# Predict the probabilities of the 29 classes for one processed 96 x 96 image
def predict_probabilities(model, processed):
    # Shape (1, 96, 96, 1). The values stay 0-255 because the model normalizes them itself.
    batch = np.asarray(processed, dtype = "float32").reshape(1, IMG_SIZE, IMG_SIZE, 1)

    return model(batch, training = False).numpy()[0]

# Change the typed text with a predicted class
def apply_to_text(text, name):
    if name == "space":
        return text + " "
    if name == "del":
        return text[:-1]
    if name == "nothing":
        return text
    return text + name

# Text that explains what the sign means
def meaning_of(name):
    return MEANINGS.get(name, f"Letter {name}")

#--------------------------------------------------
# 6. Page Title and Description
#--------------------------------------------------
st.title("Sign Language Letter Classification")

st.write(
    "Show an ASL hand sign with a photo or with the webcam. "
    "The neural network predicts the letter and can type it for you."
)

#--------------------------------------------------
# 7. Load Model with Error Handling
#--------------------------------------------------
try:
    model = load_model()

except Exception:
    # Display error message telling the user to train the model first
    st.error(
        "Model not found. Please run python train_model.py first."
    )

    # Stop the Streamlit application
    st.stop()

# The text that is typed from the signs (kept while the page changes)
if "typed_text" not in st.session_state:
    st.session_state["typed_text"] = ""

#--------------------------------------------------
# 8. Show the Prediction of a Photo
#--------------------------------------------------
# This function shows the result of one photo (upload or camera photo)
def show_photo_result(image, key, mirror = False):
    # Un-mirror selfie photos, so the model sees the same image as the user sees
    if mirror:
        image = ImageOps.mirror(image)

    # Detect the hand, crop around it and remove the background (the same as the training images)
    processed, hand_found = preprocess.preprocess_image(image)

    if hand_found:
        prediction = predict_probabilities(model, processed)
    else:
        # No hand means "nothing" (the same rule as the live webcam)
        prediction = np.zeros(len(CLASS_NAMES), dtype = "float32")
        prediction[CLASS_NAMES.index("nothing")] = 1.0
        st.info(
            "No hand was found in this photo, so it is treated as \"nothing\". "
            "Take the photo again with good light and the whole hand visible."
        )

    # Find the index with the highest probability
    predicted_index = int(np.argmax(prediction))
    predicted_name = CLASS_NAMES[predicted_index]

    # Get the probability of the predicted class
    confidence = float(prediction[predicted_index])

    st.subheader("Prediction")

    st.success(
        f"Predicted Sign: **{predicted_name.upper()}**"
    )

    st.write(
        f"Meaning: **{meaning_of(predicted_name)}**"
    )

    st.write(
        f"Confidence: **{confidence * 100:.2f}%**"
    )

    if confidence < MIN_CONFIDENCE:
        st.warning("The model is not sure. Try a clearer photo of the hand in the centre of the image.")

    # Show the image that goes into the neural network
    st.subheader(f"Processed {IMG_SIZE} x {IMG_SIZE} Image")
    st.image(processed, width = 200)

    # Type the predicted letter (only when the model is confident enough)
    if confidence >= MIN_CONFIDENCE:
        if st.button("Add to text", key = "add_" + key):
            st.session_state["typed_text"] = apply_to_text(st.session_state["typed_text"], predicted_name)

    st.write("Typed text:")
    st.code(st.session_state["typed_text"] if st.session_state["typed_text"] else " ")

    if st.button("Clear text", key = "clear_" + key):
        st.session_state["typed_text"] = ""
        st.rerun()

    # Show the 5 most probable classes
    st.subheader("Top 5 Predictions")

    top_five = np.argsort(prediction)[::-1][:5]

    for index in top_five:
        st.write(
            f"{CLASS_NAMES[index].upper()}: {prediction[index] * 100:.2f}%"
        )
        st.progress(float(prediction[index]))

#--------------------------------------------------
# 9. Live Webcam Processor
#--------------------------------------------------
# This class receives every frame of the webcam, predicts the sign inside the green box
# and writes the letter on the video
class SignProcessor(VideoProcessorBase):

    def __init__(self):
        self.model = None                       # The model is set from the main page
        self.mirror = False                     # Mirror the video (selfie view), set from the main page
        self.left_hand = False                  # The user signs with the left hand, set from the main page
        self.lock = threading.Lock()            # The video and the page run in different threads
        self.frame_count = 0
        self.letter = "-"                       # Last predicted class
        self.confidence = 0.0                   # Probability of the last predicted class
        self.recent = deque(maxlen = STABLE_FRAMES)
        self.last_added = None                  # Last class that was typed
        self.text = ""                          # Typed text
        self.prob_ema = None                    # Smoothed probabilities (removes flicker between frames)
        self.last_processed = None              # Last 96 x 96 image that went into the model

    def recv(self, frame):
        image = frame.to_ndarray(format = "bgr24")

        height, width = image.shape[:2]

        # Green box in the centre: the hand must be inside it
        size = int(min(height, width) * 0.45)
        x0 = (width - size) // 2
        y0 = (height - size) // 2

        self.frame_count += 1

        if self.model is not None and self.frame_count % PREDICT_EVERY == 0:
            # Crop the box BEFORE any mirroring, because the model was trained on normal (not selfie) photos.
            box = image[y0:y0 + size, x0:x0 + size]

            # A left hand looks like a mirrored right hand, so flip it before the prediction.
            if self.left_hand:
                box = cv2.flip(box, 1)

            processed, hand_found = preprocess.preprocess_image(
                Image.fromarray(cv2.cvtColor(box, cv2.COLOR_BGR2RGB))
            )

            with self.lock:
                self.last_processed = processed

            if hand_found:
                prediction = predict_probabilities(self.model, processed)

                # Smooth the probabilities with the previous frames, so the letter does not flicker
                if self.prob_ema is None:
                    self.prob_ema = prediction
                else:
                    self.prob_ema = 0.6 * self.prob_ema + 0.4 * prediction
                prediction = self.prob_ema

                index = int(np.argmax(prediction))
                name = CLASS_NAMES[index]
                confidence = float(prediction[index])
            else:
                self.prob_ema = None
                name = "nothing"
                confidence = 1.0

            with self.lock:
                self.letter = name
                self.confidence = confidence

                # "?" = not sure
                self.recent.append(name if confidence >= MIN_CONFIDENCE else "?")

                # The same class many times in a row = a stable sign
                stable = self.recent[0] if len(self.recent) == STABLE_FRAMES and len(set(self.recent)) == 1 else None

                if stable == "nothing":
                    # The hand was removed, so the same letter can be typed again
                    self.last_added = None
                elif stable is not None and stable != "?" and stable != self.last_added:
                    self.text = apply_to_text(self.text, stable)
                    self.last_added = stable

        # Mirror the video for the selfie view. This happens AFTER the prediction,
        # so the model always sees the normal (not mirrored) photo.
        if self.mirror:
            image = cv2.flip(image, 1)

        # Draw the box and the prediction on the video
        colour = (0, 200, 0) if self.confidence >= MIN_CONFIDENCE else (0, 165, 255)
        cv2.rectangle(image, (x0, y0), (x0 + size, y0 + size), colour, 3)
        cv2.putText(image, f"{self.letter.upper()}  {self.confidence * 100:.0f}%", (x0, max(30, y0 - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 2)

        return av.VideoFrame.from_ndarray(image, format = "bgr24")

#--------------------------------------------------
# 10. Tabs: Upload Photo, Camera Photo, Live Webcam
#--------------------------------------------------
tab_upload, tab_camera, tab_live = st.tabs(["Upload Photo", "Camera Photo", "Live Webcam"])

#--------------------------------------------------
# 11. Tab 1 - Upload a Photo
#--------------------------------------------------
with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload a hand sign image",          # Label for the uploader
        type = ["png", "jpg", "jpeg"]        # Allow PNG, JPG, and JPEG image format
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        st.subheader("Uploaded Image")
        st.image(image, width = 200)

        show_photo_result(image, "upload")

#--------------------------------------------------
# 12. Tab 2 - Take a Photo with the Camera
#--------------------------------------------------
with tab_camera:
    st.write("Put your hand in the centre of the picture and take a photo.")

    # Some cameras save mirrored (selfie) photos - turn this on if the predictions are wrong
    mirror_camera = st.checkbox("Mirror the image before predicting", key = "mirror_camera")

    camera_file = st.camera_input("Take a photo of your hand sign")

    if camera_file is not None:
        image = Image.open(camera_file)

        show_photo_result(image, "camera", mirror_camera)

#--------------------------------------------------
# 13. Tab 3 - Live Webcam
#--------------------------------------------------
with tab_live:
    st.write(
        "Click START and show your hand sign inside the green box. "
        "When the same letter is seen for a moment, it is typed below. "
        "For a double letter (for example LL), remove your hand for a moment between the two letters."
    )

    # Turn this on to see the webcam like a mirror (selfie view)
    mirror_live = st.checkbox("Mirror the webcam (selfie view)", key = "mirror_live")

    # The model was trained on right hands. Left-handed users flip their input before the prediction.
    left_hand_live = st.checkbox("I sign with my LEFT hand", key = "left_hand_live")

    # Start the webcam. Every frame is processed by the SignProcessor.
    live = webrtc_streamer(
        key = "sign-live",
        video_processor_factory = SignProcessor,
        rtc_configuration = RTC_CONFIGURATION,
        media_stream_constraints = {"video": True, "audio": False},
        async_processing = True
    )

    st.subheader("Live Prediction")
    live_letter = st.empty()
    st.subheader("Typed Text")
    live_text = st.empty()
    st.subheader(f"Model Input ({IMG_SIZE} x {IMG_SIZE})")
    live_debug = st.empty()

    # Clear the typed text
    if st.button("Clear live text", key = "clear_live") and live.video_processor:
        with live.video_processor.lock:
            live.video_processor.text = ""
            live.video_processor.last_added = None

    # Show the prediction and the typed text while the webcam is running
    while live.state.playing:
        processor = live.video_processor

        if processor is not None:
            processor.model = model
            processor.mirror = mirror_live
            processor.left_hand = left_hand_live

            with processor.lock:
                letter, confidence, text = processor.letter, processor.confidence, processor.text
                processed = processor.last_processed

            live_letter.write(f"**{meaning_of(letter)}**  ({confidence * 100:.0f}%)")
            live_text.code(text if text else " ")

            if processed is not None:
                live_debug.image(processed, width = 150)

        time.sleep(0.3)
