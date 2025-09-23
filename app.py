import subprocess, sys

# Always ensure only headless OpenCV is present
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "opencv-python"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-U", "opencv-python-headless"], check=False)

import streamlit as st
import os
import sys
import tempfile
import requests
import numpy as np
from PIL import Image

# -------------------------------
# Force headless OpenCV (fixes libGL.so.1 error on Streamlit Cloud)
# -------------------------------
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
try:
    import cv2
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "opencv-python"])
    subprocess.run([sys.executable, "-m", "pip", "install", "opencv-python-headless"])
    import cv2

from ultralytics import YOLO

# -------------------------------
# Handle resource paths
# -------------------------------
def resource_path(relative_path):
    """Get absolute path to resource, works both for dev and PyInstaller exe"""
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

# -------------------------------
# Download YOLO model from Hugging Face if missing
# -------------------------------
MODEL_DIR = "models"
MODEL_PATH = resource_path(os.path.join(MODEL_DIR, "best.pt"))
HF_URL = "https://huggingface.co/priyatech3031/smart-retail-yolo/resolve/main/best.pt"

def download_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        st.info("📥 Downloading YOLO model from Hugging Face (first run only)... please wait.")
        response = requests.get(HF_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        st.success("✅ Model downloaded successfully.")

# -------------------------------
# Load YOLO Model (cached)
# -------------------------------
@st.cache_resource
def load_model():
    download_model()
    return YOLO(MODEL_PATH)

model = load_model()

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Smart Retail AI", layout="wide")

st.title("🛒 Smart Retail AI - Shelf Monitoring")
st.write("Upload an image of a supermarket shelf, and the AI will detect missing/misplaced products.")

uploaded_file = st.file_uploader("Upload Shelf Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Shelf Image", use_container_width=True)

    # Save to temp file for YOLO
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        image.save(tmp_file.name)
        temp_path = tmp_file.name

    st.write("🔍 Running detection...")
    results = model.predict(source=temp_path, conf=0.4, save=False)

    # Plot detections
    res_plotted = results[0].plot()  # numpy array (BGR)
    res_rgb = Image.fromarray(res_plotted[..., ::-1])  # convert to RGB
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

if __name__ == "__main__":
    st.write("✅ App deployed successfully on Streamlit Cloud.")
