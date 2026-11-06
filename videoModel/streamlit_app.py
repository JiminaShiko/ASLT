"""Standalone Streamlit interface for testing the video sign classifiers."""

from __future__ import annotations

import tempfile
import threading
from collections import deque
from pathlib import Path

import av
import cv2
import numpy as np
import streamlit as st
import torch
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer

from src.config import FRAME_HEIGHT, FRAME_WIDTH, NUM_FRAMES
from src.predict_video import frames_to_tensor, load_model, predict, prepare_video


class LivePredictionState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.results: list[tuple[str, float]] = []
        self.error: str | None = None
        self.frames_seen = 0
        self.top_k = 5

    def update(
        self,
        results: list[tuple[str, float]],
        frames_seen: int,
        error: str | None = None,
    ) -> None:
        with self.lock:
            self.results = results
            self.frames_seen = frames_seen
            self.error = error

    def set_error(self, error: str) -> None:
        with self.lock:
            self.error = error

    def set_top_k(self, top_k: int) -> None:
        with self.lock:
            self.top_k = top_k

    def snapshot(self) -> tuple[list[tuple[str, float]], str | None, int, int]:
        with self.lock:
            return self.results.copy(), self.error, self.frames_seen, self.top_k


class LiveVideoProcessor(VideoProcessorBase):
    def __init__(
        self,
        model: torch.nn.Module,
        class_names: list[str],
        device: torch.device,
        state: LivePredictionState,
    ) -> None:
        self.model = model
        self.class_names = class_names
        self.device = device
        self.state = state
        self.frames: deque[np.ndarray] = deque(maxlen=NUM_FRAMES)
        self.frames_seen = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        sampled = cv2.resize(
            rgb,
            (FRAME_WIDTH, FRAME_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )
        self.frames.append(sampled)
        self.frames_seen += 1

        should_predict = self.frames_seen == NUM_FRAMES or (
            self.frames_seen > NUM_FRAMES
            and (self.frames_seen - NUM_FRAMES) % 8 == 0
        )
        if should_predict:
            try:
                video = frames_to_tensor(np.stack(self.frames))
                _, _, _, top_k = self.state.snapshot()
                results = predict(
                    self.model,
                    video,
                    self.class_names,
                    self.device,
                    top_k,
                )
                self.state.update(results, self.frames_seen)
            except Exception as error:
                self.state.update(
                    [],
                    self.frames_seen,
                    f"{type(error).__name__}: {error}",
                )

        results, error, _, _ = self.state.snapshot()
        if results:
            label, score = results[0]
            overlay = f"{label} ({score:.1%})"
            color = (0, 200, 0)
        elif error:
            overlay = "Prediction error"
            color = (0, 0, 255)
        else:
            overlay = f"Collecting frames: {len(self.frames)}/{NUM_FRAMES}"
            color = (0, 200, 0)
        cv2.putText(
            image,
            overlay,
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )
        return av.VideoFrame.from_ndarray(image, format="bgr24")


def choose_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was selected, but it is not available.")
        return torch.device("cuda")
    if name == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@st.cache_resource
def cached_model(model_name: str, device_name: str) -> tuple[torch.nn.Module, list[str]]:
    return load_model(model_name, choose_device(device_name))


def show_predictions(results: list[tuple[str, float]]) -> None:
    if not results:
        st.info("Waiting for enough webcam frames to make a prediction.")
        return
    st.subheader("Top predictions")
    for label, score in results:
        st.write(f"**{label}** — {score:.1%}")
        st.progress(score)


def main() -> None:
    st.set_page_config(page_title="ASLT Video Model", page_icon="🤟", layout="centered")
    st.title("ASL Video Sign Classifier")
    st.write(
        "Test the video model independently. It samples 16 frames from each clip "
        "and predicts one of the model's trained word classes."
    )
    st.caption("Prediction scores are softmax scores, not calibrated confidence.")

    model_name = st.sidebar.selectbox("Model architecture", ("cnn3d", "cnn_gru"))
    device_options = ["auto", "cpu"]
    if torch.cuda.is_available():
        device_options.insert(1, "cuda")
    device_name = st.sidebar.selectbox("Device", device_options)
    try:
        model, class_names = cached_model(model_name, device_name)
        device = choose_device(device_name)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        st.error(str(error))
        st.stop()

    top_k = st.sidebar.slider(
        "Number of predictions",
        min_value=1,
        max_value=min(5, len(class_names)),
        value=min(5, len(class_names)),
    )

    clip_tab, webcam_tab = st.tabs(["Test a video clip", "Live webcam"])
    with clip_tab:
        uploaded = st.file_uploader(
            "Choose a sign-language video",
            type=["mp4", "m4v", "mov", "avi", "mkv", "webm"],
        )
        wlasl_id = st.text_input(
            "Optional WLASL video ID",
            help="For a full-length WLASL source video, use its ID to select the annotated signing interval.",
        )
        if uploaded is not None:
            video_bytes = uploaded.getvalue()
            st.video(video_bytes)
            if st.button("Predict video", type="primary"):
                suffix = Path(uploaded.name).suffix or ".mp4"
                try:
                    with tempfile.TemporaryDirectory(prefix="aslt-video-") as directory:
                        video_path = Path(directory) / f"upload{suffix}"
                        video_path.write_bytes(video_bytes)
                        video, reference_gloss = prepare_video(
                            video_path,
                            wlasl_id.strip() or None,
                        )
                    results = predict(model, video, class_names, device, top_k)
                    if reference_gloss:
                        st.caption(f"WLASL reference: {wlasl_id.strip()} ({reference_gloss})")
                    show_predictions(results)
                except (OSError, ValueError) as error:
                    st.error(f"Could not process this video: {error}")

    with webcam_tab:
        st.write(
            "Start the camera and hold a sign in view. Predictions use a rolling "
            "16-frame window and refresh every 8 incoming frames."
        )
        state_key = f"live_state_{model_name}_{device.type}"
        if state_key not in st.session_state:
            st.session_state[state_key] = LivePredictionState()
        live_state: LivePredictionState = st.session_state[state_key]
        live_state.set_top_k(top_k)

        rtc_configuration = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )
        webrtc_streamer(
            key=f"live-{model_name}-{device.type}",
            video_processor_factory=lambda: LiveVideoProcessor(
                model,
                class_names,
                device,
                live_state,
            ),
            rtc_configuration=rtc_configuration,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        @st.fragment(run_every=1)
        def live_results() -> None:
            results, error, frames_seen, _ = live_state.snapshot()
            st.caption(f"Frames received: {frames_seen}")
            if error:
                st.error(f"Live prediction failed: {error}")
            show_predictions(results)

        live_results()


if __name__ == "__main__":
    main()
