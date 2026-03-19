import streamlit as st
from PIL import Image, ImageOps, ImageDraw
import numpy as np
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import time
import threading

st.set_page_config(page_title="Smart Light Controller", page_icon="💡")

@st.cache_resource
def load():
    from tf_keras.models import load_model
    model = load_model("keras_model.h5", compile=False)
    class_names = open("labels.txt", "r").readlines()
    return model, class_names

model, class_names = load()

def classify(image):
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    img = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    data[0] = (np.asarray(img).astype(np.float32) / 127.5) - 1
    prediction = model.predict(data)
    index = np.argmax(prediction)
    return class_names[index][2:].strip(), prediction[0][index]

def is_occupied(class_name):
    return "occupied" in class_name.lower() or "people" in class_name.lower() or "person" in class_name.lower()

# Session state for energy tracking
if "lights_off_since" not in st.session_state:
    st.session_state.lights_off_since = None
if "total_seconds_saved" not in st.session_state:
    st.session_state.total_seconds_saved = 0

WATT = 60  # assumed bulb wattage

def energy_saved(seconds):
    kwh = (WATT * seconds) / 3600 / 1000
    return kwh

# UI
st.title("💡 Smart Light Controller")
st.write("SDG 7.3 — Energy Efficiency: AI that controls lights based on room occupancy.")

# --- Upload section ---
st.subheader("📸 Upload a photo")
uploaded_file = st.file_uploader("Upload a photo of the room", type=["jpg", "jpeg", "png", "bmp", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    class_name, confidence = classify(image)
    occupied = is_occupied(class_name)

    st.divider()

    if occupied:
        st.markdown("## 💡 Lights: ON")
        st.success(f"Room is **occupied** — lights should be on.")
        if st.session_state.lights_off_since is not None:
            saved = time.time() - st.session_state.lights_off_since
            st.session_state.total_seconds_saved += saved
            st.session_state.lights_off_since = None
    else:
        st.markdown("## 🌑 Lights: OFF")
        st.warning(f"Room is **empty** — lights should be off.")
        if st.session_state.lights_off_since is None:
            st.session_state.lights_off_since = time.time()

    st.info(f"Detected: **{class_name}** — Confidence: {confidence:.2%}")

    # Energy savings
    st.divider()
    st.subheader("⚡ Energy Savings")
    current_saving = 0
    if st.session_state.lights_off_since is not None:
        current_saving = time.time() - st.session_state.lights_off_since

    total = st.session_state.total_seconds_saved + current_saving
    kwh = energy_saved(total)
    minutes = int(total // 60)
    seconds = int(total % 60)

    st.metric("Lights off for", f"{minutes}m {seconds}s")
    st.metric("Energy saved", f"{kwh * 1000:.2f} Wh")
    st.metric("CO₂ avoided", f"{kwh * 0.233:.4f} kg")

# --- Webcam section ---
st.divider()
st.subheader("📷 Live Webcam")
st.write("The model classifies a frame every 10 seconds.")

last_prediction = {"class": "Waiting...", "confidence": 0.0}
last_run = {"time": 0}
lock = threading.Lock()

def video_frame_callback(frame):
    img = frame.to_image()

    now = time.time()
    if now - last_run["time"] >= 10:
        last_run["time"] = now
        class_name, confidence = classify(img.copy())
        with lock:
            last_prediction["class"] = class_name
            last_prediction["confidence"] = confidence

    draw = ImageDraw.Draw(img)
    with lock:
        label = f"{last_prediction['class']} ({last_prediction['confidence']:.2%})"
    draw.text((10, 10), label, fill=(0, 255, 0))

    return av.VideoFrame.from_image(img)

webrtc_streamer(
    key="light-controller",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
        ]
    },
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
)

with lock:
    webcam_occupied = is_occupied(last_prediction["class"])

if last_prediction["class"] != "Waiting...":
    if webcam_occupied:
        st.markdown("## 💡 Lights: ON")
        st.success(f"Room is **occupied**")
    else:
        st.markdown("## 🌑 Lights: OFF")
        st.warning(f"Room is **empty**")
    st.info(f"Detected: **{last_prediction['class']}** — Confidence: {last_prediction['confidence']:.2%}")