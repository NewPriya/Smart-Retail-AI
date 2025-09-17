import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile
import os

# -------------------------------
# Load YOLO Model
# -------------------------------
MODEL_PATH = "models/best.pt"

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🛒 Smart Retail AI - Shelf Monitoring")
st.write("Upload an image of a supermarket shelf, and the AI will detect missing/misplaced products.")

# File uploader
uploaded_file = st.file_uploader("Upload Shelf Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Shelf Image", use_container_width=True)

    # Save to temp file for YOLO
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        image.save(tmp_file.name)
        temp_path = tmp_file.name

    st.write("🔍 Running detection...")

    # Run YOLO prediction
    results = model.predict(source=temp_path, conf=0.4, save=False)

    # Plot detections on image
    res_plotted = results[0].plot()  # numpy array (BGR)
    res_rgb = Image.fromarray(res_plotted[..., ::-1])  # convert to RGB for Streamlit

    st.image(res_rgb, caption="AI Detection Result", use_container_width=True)

    # Show counts
    counts = results[0].boxes.cls.cpu().numpy()
    names = model.names
    detected_classes = [names[int(c)] for c in counts]

    st.subheader("📊 Detected Products:")
    if detected_classes:
        st.write({c: detected_classes.count(c) for c in set(detected_classes)})
    else:
        st.write("⚠️ No products detected.")
