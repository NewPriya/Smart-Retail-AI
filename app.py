import streamlit as st
import os
from PIL import Image

st.set_page_config(page_title="Smart Retail AI", layout="wide")

st.title("🛒 Smart Retail AI - Shelf Monitoring")

st.write("Upload an image of a supermarket shelf, and the AI will detect missing/misplaced products.")

# File uploader
uploaded_file = st.file_uploader("Upload Shelf Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Shelf Image", use_column_width=True)

    # Placeholder for YOLOv8 inference
    st.write("🔍 Running AI model on the image... (model integration coming next)")
