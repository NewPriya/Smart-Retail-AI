import streamlit as st
import os
import sys
import tempfile
import requests
import numpy as np
from PIL import Image
from ultralytics import YOLO

# -------------------------------
# Quick runtime check for OpenCV (friendly error message)
# -------------------------------
try:
    import cv2  # noqa: F401
except Exception as e:
    st.set_page_config(page_title="Smart Retail AI", layout="wide")
    st.title("🛒 Smart Retail AI - Shelf Monitoring")
    st.error(
        "Failed to import OpenCV (cv2).\n\n"
        "Please install the headless OpenCV wheel for Windows:\n\n"
        "`pip install opencv-python-headless==4.12.0.88`\n\n"
        f"Error detail: {e}"
    )
    st.stop()

# -------------------------------
# Helper: resource path compatible with PyInstaller
# -------------------------------
def resource_path(relative_path: str) -> str:
    """Get absolute path to resource, works both for dev + PyInstaller exe"""
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

# -------------------------------
# Model download / paths
# -------------------------------
MODEL_DIR = "models"
MODEL_FILENAME = "best.pt"
MODEL_PATH = resource_path(os.path.join(MODEL_DIR, MODEL_FILENAME))
HF_URL = "https://huggingface.co/priyatech3031/smart-retail-yolo/resolve/main/best.pt"

def download_model(dest_path: str = MODEL_PATH, url: str = HF_URL, chunk_size: int = 8192):
    """Stream-download model from HF to avoid memory spikes and provide progress."""
    if os.path.exists(dest_path):
        return
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    st.info("📥 Downloading YOLO model from Hugging Face (first run only)... please wait.")
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            downloaded = 0
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            st.progress(min(downloaded / total, 1.0))
        st.success("✅ Model downloaded successfully.")
    except Exception as e:
        # If download fails, ensure partial file removed
        try:
            if os.path.exists(dest_path):
                os.remove(dest_path)
        except Exception:
            pass
        st.error(f"Model download failed: {e}")
        raise

# -------------------------------
# Load model (cached across reruns)
# -------------------------------
@st.cache_resource
def load_model():
    download_model()
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to load YOLO model from {MODEL_PATH}: {e}")
        raise

    # Try to move model to GPU if available
    try:
        import torch  # local import so app still runs if torch missing
        if torch.cuda.is_available():
            try:
                model.to("cuda")
                st.info("Using GPU for inference.")
            except Exception:
                # not fatal, continue on CPU
                st.info("GPU detected but failed to move model to GPU; continuing on CPU.")
    except Exception:
        # torch not installed or other issue — fine, keep CPU
        pass

    return model

# instantiate model (cached)
model = load_model()

# -------------------------------
# Streamlit UI & logic
# -------------------------------
st.set_page_config(page_title="Smart Retail AI", layout="wide")
st.title("🛒 Smart Retail AI - Shelf Monitoring")
st.write("Upload an image of a supermarket shelf, and the AI will detect missing/misplaced products.")

uploaded_file = st.file_uploader("Upload Shelf Image", type=["jpg", "jpeg", "png"])

def resize_for_inference(img: Image.Image, max_dim: int = 1280) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_dim:
        return img
    scale = max_dim / max(w, h)
    new_size = (int(w * scale), int(h * scale))
    return img.resize(new_size, Image.LANCZOS)

if uploaded_file is not None:
    # Display original
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Shelf Image", use_container_width=True)

    # Resize to reasonable size for faster inference (keeps aspect ratio)
    inf_image = resize_for_inference(image, max_dim=1280)

    # Save to a temporary file for the YOLO .predict API (cleaned up later)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    try:
        inf_image.save(tmp.name, format="JPEG")
        temp_path = tmp.name

        with st.spinner("🔍 Running detection..."):
            try:
                results = model.predict(source=temp_path, conf=0.4, save=False)
            except Exception as e:
                st.error(f"Inference failed: {e}")
                results = None

        if results:
            # Plot detections (ultralytics returns BGR image)
            try:
                res_plotted = results[0].plot()
                # If plot returns None or not an ndarray, handle safely
                if isinstance(res_plotted, np.ndarray):
                    res_rgb = Image.fromarray(res_plotted[..., ::-1])  # BGR -> RGB
                    st.image(res_rgb, caption="AI Detection Result", use_container_width=True)
                else:
                    st.warning("Plot returned unexpected type; skipping visualization.")
            except Exception as e:
                st.warning(f"Could not render plotted image: {e}")

            # Extract detected classes robustly
            detected_classes = []
            try:
                boxes = results[0].boxes
                if boxes is not None and hasattr(boxes, "cls"):
                    cls = boxes.cls
                    # cls may be a tensor or list-like
                    if hasattr(cls, "cpu"):
                        cls_np = cls.cpu().numpy()
                    else:
                        cls_np = np.array(cls)
                    names = getattr(model, "names", {})
                    detected_classes = [names.get(int(c), str(int(c))) if isinstance(names, dict) else names[int(c)] for c in cls_np]
            except Exception as e:
                st.warning(f"Could not extract detection classes: {e}")

            st.subheader("📊 Detected Products:")
            if detected_classes:
                st.write({c: detected_classes.count(c) for c in sorted(set(detected_classes))})
            else:
                st.write("⚠️ No products detected.")
    finally:
        # cleanup temp file
        try:
            tmp.close()
        except Exception:
            pass
        try:
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)
        except Exception:
            pass

if __name__ == "__main__":
    st.write("✅ App ready.")
