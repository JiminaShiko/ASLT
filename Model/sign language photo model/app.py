#--------------------------------------------------
# 1. Import Libraries
#--------------------------------------------------
import os
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
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sign_photo_model.keras")

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
STABLE_FRAMES = 4          # The same letter must be predicted 4 times in a row before it is typed
PREDICT_INTERVAL = 0.3     # Seconds between two predictions (the work runs in its own thread)
DETECT_MAX_SIDE = 640      # Frames bigger than this are shrunk before hand detection
NOTHING_TICKS = 2          # Missed detections in a row before "no hand" is shown
BOX_SIZE = 0.45            # The green guide box is this fraction of the smaller frame side

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
# This class receives every frame of the webcam. The heavy work (hand
# detection + neural network) runs in its own background thread on the
# newest frame, so the video never waits for a prediction and stays smooth.
# Predictions use the WHOLE frame, like the photo tabs (no box).
class SignProcessor(VideoProcessorBase):

    def __init__(self):
        self.model = None                       # The model is set from the main page
        self.mirror = False                     # Mirror the video (selfie view), set from the main page
        self.left_hand = False                  # The user signs with the left hand, set from the main page
        self.lock = threading.Lock()            # The video and the page run in different threads
        self.letter = "-"                       # Last predicted class ("-" = waiting for a sign)
        self.confidence = 0.0                   # Probability of the last predicted class
        self.hand_found = False                 # Did MediaPipe find a hand in the last frame?
        self.top_names = []                     # Top 5 class names of the newest prediction
        self.top_probs = []                     # Top 5 probabilities of the newest prediction
        self.recent = deque(maxlen = STABLE_FRAMES)
        self.last_added = None                  # Last class that was typed
        self.misses = 0                         # Detections in a row without a hand
        self.text = ""                          # Typed text
        self.prob_ema = None                    # Smoothed probabilities (removes flicker between frames)
        self.last_processed = None              # Last 96 x 96 image that went into the model
        self.latest_frame = None                # Newest frame waiting for a prediction
        self.last_frame_time = 0.0              # time.monotonic() value of the newest frame
        self.next_predict_time = 0.0            # time.monotonic() value of the next prediction
        self.worker = None                      # Background prediction thread

    def recv(self, frame):
        image = frame.to_ndarray(format = "bgr24")

        if self.model is not None:
            # Hand the frame to the background thread. recv itself stays fast,
            # so the video keeps its frame rate.
            with self.lock:
                self.latest_frame = image.copy()
            self.last_frame_time = time.monotonic()
            if self.worker is None or not self.worker.is_alive():
                self.worker = threading.Thread(target = self.predict_loop, daemon = True)
                self.worker.start()

        # Mirror the video for the selfie view. The prediction thread works on
        # the un-mirrored copy, because the model was trained on normal photos.
        if self.mirror:
            image = cv2.flip(image, 1)

        # Draw the green guide box and the prediction label
        with self.lock:
            letter, confidence = self.letter, self.confidence

        height, width = image.shape[:2]
        size = int(min(height, width) * BOX_SIZE)
        x0 = (width - size) // 2
        y0 = (height - size) // 2
        colour = (0, 200, 0) if letter != "-" and confidence >= MIN_CONFIDENCE else (0, 165, 255)
        cv2.rectangle(image, (x0, y0), (x0 + size, y0 + size), colour, 3)

        if letter != "-":
            label = "NO SIGN" if letter == "nothing" else f"{letter.upper()}  {confidence * 100:.0f}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(image, (10, 10), (10 + tw + 24, 10 + th + 20), (0, 0, 0), -1)
            cv2.rectangle(image, (10, 10), (10 + tw + 24, 10 + th + 20), colour, 2)
            cv2.putText(image, label, (22, 10 + th + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)

        return av.VideoFrame.from_ndarray(image, format = "bgr24")

    def predict_loop(self):
        # Runs in its own thread until the webcam has been silent for 5 seconds.
        while True:
            if time.monotonic() - self.last_frame_time > 5.0:
                return

            if time.monotonic() < self.next_predict_time:
                time.sleep(0.02)
                continue

            with self.lock:
                frame = self.latest_frame
                self.latest_frame = None
                model = self.model
                left_hand = self.left_hand

            if frame is None or model is None:
                time.sleep(0.02)
                continue

            started = time.monotonic()
            self.predict_frame(frame, model, left_hand)
            self.next_predict_time = started + PREDICT_INTERVAL

    def predict_frame(self, image, model, left_hand):
        if left_hand:
            # A left hand looks like a mirrored right hand, so flip it first.
            image = cv2.flip(image, 1)

        # Shrink very large frames (the 96 x 96 model input needs no more detail)
        height, width = image.shape[:2]
        longest = max(height, width)
        if longest > DETECT_MAX_SIDE:
            scale = DETECT_MAX_SIDE / longest
            image = cv2.resize(image, (int(width * scale), int(height * scale)),
                               interpolation = cv2.INTER_AREA)

        # Predict from the whole frame, exactly like the photo tabs. The green
        # box is only a guide for the user: cropping to the box can cut a hand
        # in half when it touches the edge, and the model then reads a partial
        # hand as a different letter.
        processed, hand_found = preprocess.preprocess_image(
            Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),
            video = True
        )

        if hand_found:
            self.misses = 0
            prediction = predict_probabilities(model, processed)

            # Smooth the probabilities with the previous frames, so the letter
            # does not flicker - but when a clearly different sign appears,
            # switch at once instead of blending it with the old sign for a
            # second or two (that made the display show the previous letter).
            if self.prob_ema is None:
                self.prob_ema = prediction
            else:
                old_name = int(np.argmax(self.prob_ema))
                new_name = int(np.argmax(prediction))
                if new_name != old_name and prediction[new_name] >= MIN_CONFIDENCE:
                    self.prob_ema = prediction
                else:
                    self.prob_ema = 0.6 * self.prob_ema + 0.4 * prediction
            prediction = self.prob_ema

            index = int(np.argmax(prediction))
            name = CLASS_NAMES[index]
            confidence = float(prediction[index])
        else:
            self.misses += 1
            if self.misses < NOTHING_TICKS:
                # One failed detection is not "no hand" yet: keep the last sign,
                # so a single bad frame does not reset the display or the typing.
                with self.lock:
                    self.last_processed = processed
                return

            # The hand is really gone: "nothing" (the same rule as the photo tabs)
            self.prob_ema = None
            prediction = np.zeros(len(CLASS_NAMES), dtype = "float32")
            prediction[CLASS_NAMES.index("nothing")] = 1.0
            name = "nothing"
            confidence = 1.0

        with self.lock:
            self.letter = name
            self.confidence = confidence
            self.hand_found = hand_found
            self.last_processed = processed

            top_five = np.argsort(prediction)[::-1][:5]
            self.top_names = [CLASS_NAMES[i] for i in top_five]
            self.top_probs = [float(prediction[i]) for i in top_five]

            # "?" = not sure
            self.recent.append(name if confidence >= MIN_CONFIDENCE else "?")

            if name == "nothing":
                # The hand was removed, so the same letter can be typed again
                self.last_added = None
            else:
                # The same class many times in a row = a stable sign
                stable = self.recent[0] if len(self.recent) == STABLE_FRAMES and len(set(self.recent)) == 1 else None
                if stable is not None and stable != "?" and stable != self.last_added:
                    self.text = apply_to_text(self.text, stable)
                    self.last_added = stable

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
        "Click START and show your hand sign inside the green box, like in the training photos. "
        "Keep the palm or the back of your hand turned towards the camera - a hand seen from the side "
        "looks like a different letter. "
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
    live_sign = st.empty()
    live_meaning = st.empty()
    live_confidence = st.empty()
    live_warning = st.empty()
    live_top = st.empty()
    st.subheader(f"Model Input ({IMG_SIZE} x {IMG_SIZE})")
    live_debug = st.empty()
    st.subheader("Typed Text")
    live_text = st.empty()

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
                letter, confidence = processor.letter, processor.confidence
                hand_found = processor.hand_found
                text = processor.text
                processed = processor.last_processed
                top_names = list(processor.top_names)
                top_probs = list(processor.top_probs)

            if letter == "-":
                live_sign.info("Waiting for a sign - show your hand to the webcam")
                live_meaning.write("")
                live_confidence.write("")
                live_warning.write("")
                live_top.write("")
            else:
                live_sign.success(f"Predicted Sign: **{letter.upper()}**")
                live_meaning.write(f"Meaning: **{meaning_of(letter)}**")
                live_confidence.write(f"Confidence: **{confidence * 100:.2f}%**")

                if not hand_found:
                    live_warning.warning(
                        "The camera does not see a hand. Improve the light, move closer, "
                        "turn the palm or the back of the hand towards the camera, "
                        "and keep the whole hand inside the green box."
                    )
                elif confidence < MIN_CONFIDENCE:
                    live_warning.warning("The model is not sure. Show the whole hand clearly and closer to the webcam.")
                else:
                    live_warning.write("")

                top_md = "\n\n".join(
                    f"{name.upper()}: {prob * 100:.2f}%" for name, prob in zip(top_names, top_probs)
                )
                live_top.markdown(f"**Top 5 Predictions**\n\n{top_md}")

            live_text.code(text if text else " ")

            if processed is not None:
                live_debug.image(processed, width = 150)

        time.sleep(0.2)
