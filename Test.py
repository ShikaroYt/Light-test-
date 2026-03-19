import streamlit as st
from streamlit_autorefresh import st_autorefresh
from PIL import Image, ImageOps
import numpy as np
import time

st.set_page_config(page_title="Smart Light Controller", page_icon="💡")

st_autorefresh(interval=1000)

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

WATT = 60

# Session state
if "lights_off_since" not in st.session_state:
    st.session_state.lights_off_since = None
if "total_seconds_saved" not in st.session_state:
    st.session_state.total_seconds_saved = 0
if "last_class" not in st.session_state:
    st.session_state.last_class = None
if "last_confidence" not in st.session_state:
    st.session_state.last_confidence = None

# UI
st.title("💡 Smart Light Controller")
st.write("SDG 7.3 — Energy Efficiency: AI that controls lights based on room occupancy.")

uploaded_file = st.file_uploader("Upload a photo of the room", type=["jpg", "jpeg", "png", "bmp", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", width='stretch')

    class_name, confidence = classify(image)
    st.session_state.last_class = class_name
    st.session_state.last_confidence = confidence
    occupied = is_occupied(class_name)

    st.divider()

    if occupied:
        st.markdown("## 💡 Lights: ON")
        st.success("Room is **occupied** — lights should be on.")
        if st.session_state.lights_off_since is not None:
            st.session_state.total_seconds_saved += time.time() - st.session_state.lights_off_since
            st.session_state.lights_off_since = None
    else:
        st.markdown("## 🌑 Lights: OFF")
        st.warning("Room is **empty** — lights should be off.")
        if st.session_state.lights_off_since is None:
            st.session_state.lights_off_since = time.time()

    st.info(f"Detected: **{class_name}** — Confidence: {confidence:.2%}")

else:
    # No image uploaded — stop the timer and reset
    if st.session_state.lights_off_since is not None:
        st.session_state.total_seconds_saved += time.time() - st.session_state.lights_off_since
        st.session_state.lights_off_since = None
    st.session_state.last_class = None
    st.session_state.last_confidence = None

# Energy savings (always visible)
st.divider()
st.subheader("⚡ Energy Savings")

current_saving = 0
if st.session_state.lights_off_since is not None:
    current_saving = time.time() - st.session_state.lights_off_since

total = st.session_state.total_seconds_saved + current_saving
kwh = (WATT * total) / 3600 / 1000
minutes = int(total // 60)
seconds = int(total % 60)

st.metric("Lights off for", f"{minutes}m {seconds}s")
st.metric("Energy saved", f"{kwh * 1000:.2f} Wh")
st.metric("CO₂ avoided", f"{kwh * 0.233:.4f} kg")