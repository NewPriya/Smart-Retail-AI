import streamlit as st
import tempfile
import numpy as np
from PIL import Image
from ultralytics import YOLO
import requests
import os

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(page_title="Smart Retail AI", layout="wide")
st.title("🛒 Smart Retail AI - Shelf Monitoring")
st.write("Upload an image of a supermarket shelf, and the AI will detect missing/misplaced products.")

# -------------------------------
# Load YOLO model (download from Hugging Face if missing)
# -------------------------------
@st.cache_resource
def load_model():
    HF_URL = "https://huggingface.co/priyatech3031/smart-retail-yolo/resolve/main/best.pt"
    local_path = "best.pt"

    # Download only if not present locally
    if not os.path.exists(local_path):
        st.info("📥 Downloading model weights from Hugging Face... please wait (first run only).")
        r = requests.get(HF_URL, stream=True)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        st.success("✅ Model downloaded successfully!")

    return YOLO(local_path)

model = load_model()

# -------------------------------
# File uploader
# -------------------------------
uploaded_file = st.file_uploader("📂 Upload Shelf Image", type=["jpg", "jpeg", "png"])

# -------------------------------
# Resize Helper
# -------------------------------
def resize_for_inference(img: Image.Image, max_dim: int = 1280) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_dim:
        return img
    scale = max_dim / max(w, h)
    new_size = (int(w * scale), int(h * scale))
    return img.resize(new_size, Image.LANCZOS)

# -------------------------------
# Inference
# -------------------------------
if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Shelf Image", width=700)

    # Resize
    inf_image = resize_for_inference(image)

    # Save to temp file for YOLO
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    inf_image.save(tmp.name, format="JPEG")

    with st.spinner("🔍 Running detection... please wait."):
        results = model.predict(source=tmp.name, conf=0.4, save=False)

    # Plot detections
    if results:
        res_plotted = results[0].plot()
        if isinstance(res_plotted, np.ndarray):
            res_rgb = Image.fromarray(res_plotted[..., ::-1])  # BGR → RGB
            st.image(res_rgb, caption="AI Detection Result", width=700)

        # Show detected products
        detected_classes = []
        try:
            boxes = results[0].boxes
            if boxes is not None and hasattr(boxes, "cls"):
                cls_np = boxes.cls.cpu().numpy()
                names = getattr(model, "names", {})
                detected_classes = [names.get(int(c), str(int(c))) for c in cls_np]
        except Exception as e:
            st.warning(f"Could not extract detection classes: {e}")

        st.subheader("📊 Detected Products:")
        if detected_classes:
            st.write({c: detected_classes.count(c) for c in sorted(set(detected_classes))})
        else:
            st.write("⚠️ No products detected.")

    tmp.close()
