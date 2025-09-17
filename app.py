import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image

# -------------------------------
# Load YOLO model
# -------------------------------
MODEL_PATH = "models/best.pt"
model = YOLO(MODEL_PATH)

st.title("🛒 Smart Retail AI - Shelf Monitoring")
st.write("Upload an image of a supermarket shelf, and the AI will detect missing/misplaced products.")

# -------------------------------
# Upload Image
# -------------------------------
uploaded_file = st.file_uploader("Upload Shelf Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert file to image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Shelf Image", use_container_width=True)

    st.write("🔍 Running AI model on the image...")

    # -------------------------------
    # Run YOLO inference
    # -------------------------------
    results = model.predict(source=np.array(image))

    # Annotated image with detections
    annotated_img = results[0].plot()  # YOLO auto-draws boxes
    st.image(annotated_img, caption="Detections", use_container_width=True)

    # -------------------------------
    # Stock Count
    # -------------------------------
    class_names = model.names
    counts = {}
    for box in results[0].boxes:
        cls_id = int(box.cls)
        cls_name = class_names[cls_id]
        counts[cls_name] = counts.get(cls_name, 0) + 1

    st.subheader("📊 Stock Count")
    st.json(counts)
