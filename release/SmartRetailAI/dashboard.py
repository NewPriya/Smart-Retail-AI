import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
from collections import Counter
import pandas as pd
import altair as alt
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import sys
import torch
import ultralytics

# -----------------------
# DEBUG INFO BANNER
# -----------------------
st.set_page_config(page_title="Smart Retail AI", layout="wide")

st.sidebar.info(
    f"🛠️ Debug Info\n\n"
    f"- Python: {sys.version.split()[0]}\n"
    f"- Torch: {torch.__version__}\n"
    f"- Ultralytics: {ultralytics.__version__}"
)

# -----------------------
# CONFIGURE YOUR EMAIL
# -----------------------
SENDER_EMAIL = "priyadarshini985@gmail.com"
APP_PASSWORD = "pzdr yajs agjq xlqh"  # ⚠️ better move to st.secrets
RECEIVER_EMAIL = "aruncorp01@gmail.com"
SMTP_SERVER, SMTP_PORT = "smtp.gmail.com", 587

def send_email_alert(store, product, count, threshold):
    subject = f"⚠️ {store} - Low Stock Alert: {product}"
    body = f"""
    Hello,

    The Smart Retail AI system has detected low stock in {store}:

    Product: {product}
    Current Stock: {count}
    Threshold: {threshold}

    Please restock soon.

    Regards,
    Smart Retail AI
    """

    msg = MIMEMultipart()
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_EMAIL
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, APP_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
        server.quit()
        st.success(f"📧 Email sent: {subject}")
    except Exception as e:
        st.error(f"Email sending failed: {e}")

# -----------------------
# YOLO MODEL (safe load)
# -----------------------
model = None
try:
    model = YOLO("models/best.pt")  # or "best.pt" if not in models/
except Exception as e:
    st.error(f"❌ Could not load YOLO model: {e}")
    st.warning("Dashboard is running, but detections are disabled.")

# -----------------------
# STREAMLIT CONFIG
# -----------------------
st.title("🛒 Smart Retail AI - Multi-Store Dashboard")
st.write("Monitor stock levels, get alerts, and download reports per store.")

# Sidebar: multi-store selection
st.sidebar.header("🏪 Store Settings")
stores = ["Store A", "Store B", "Store C"]
selected_store = st.sidebar.selectbox("Select Store", stores)

# Sidebar: thresholds
st.sidebar.subheader("⚙️ Stock Thresholds")
thresholds = {
    "Coke": st.sidebar.number_input("Coke minimum stock", 0, 50, 5),
    "Pepsi": st.sidebar.number_input("Pepsi minimum stock", 0, 50, 5),
    "Lays": st.sidebar.number_input("Lays minimum stock", 0, 50, 5),
    "Oreo": st.sidebar.number_input("Oreo minimum stock", 0, 50, 5),
}

# Upload / webcam
uploaded_file = st.file_uploader("Upload Image/Video", type=["jpg", "png", "mp4", "avi"])
use_webcam = st.checkbox("📹 Use Webcam")

# Keep history per store
if "history" not in st.session_state:
    st.session_state["history"] = {store: [] for store in stores}

# -----------------------
# HELPERS
# -----------------------
def process_frame(frame):
    if model is None:
        return frame, {}
    results = model(frame)
    frame = results[0].plot()
    names = results[0].names
    classes = results[0].boxes.cls.tolist()
    counts = Counter([names[int(c)] for c in classes])
    return frame, dict(counts)

def show_charts(counts):
    if counts:
        df = pd.DataFrame(list(counts.items()), columns=["Product", "Count"])
        chart = (
            alt.Chart(df)
            .mark_bar(cornerRadius=8)
            .encode(
                x=alt.X("Product", sort="-y"),
                y="Count",
                color=alt.Color("Product", legend=None),
                tooltip=["Product", "Count"],
            )
            .properties(width=500, height=400)
        )
        st.altair_chart(chart, use_container_width=True)

def update_trend(store, counts):
    counts["time"] = time.strftime("%H:%M:%S")
    st.session_state["history"][store].append(counts)
    history_df = pd.DataFrame(st.session_state["history"][store])
    if len(history_df) > 1:
        melted = history_df.melt("time", var_name="Product", value_name="Count")
        trend_chart = (
            alt.Chart(melted)
            .mark_line(point=True)
            .encode(
                x="time",
                y="Count",
                color="Product",
                tooltip=["time", "Product", "Count"],
            )
            .properties(width=700, height=400)
        )
        st.subheader(f"📊 {store} - Stock Trend Over Time")
        st.altair_chart(trend_chart, use_container_width=True)

def check_alerts(store, counts):
    for item, count in counts.items():
        if item in thresholds and count < thresholds[item]:
            st.error(f"⚠️ {store} - Low stock: {item} ({count} < {thresholds[item]})")
            send_email_alert(store, item, count, thresholds[item])

def save_report(store):
    history_df = pd.DataFrame(st.session_state["history"][store])
    if not history_df.empty:
        csv_path = f"{store.replace(' ', '_')}_report.csv"
        history_df.to_csv(csv_path, index=False)
        st.success(f"✅ Report saved as {csv_path}")
        with open(csv_path, "rb") as f:
            st.download_button("⬇️ Download Report", f, file_name=csv_path)

# -----------------------
# MAIN LOGIC
# -----------------------
if uploaded_file:
    temp_path = tempfile.NamedTemporaryFile(delete=False).name
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    if uploaded_file.type.startswith("image"):
        img = cv2.imread(temp_path)
        frame, counts = process_frame(img)
        st.image(frame, channels="BGR")

        st.subheader(f"📊 {selected_store} - Detected Products")
        st.json(counts)

        show_charts(counts)
        update_trend(selected_store, counts)
        check_alerts(selected_store, counts)
        save_report(selected_store)

    else:  # Video
        cap = cv2.VideoCapture(temp_path)
        stframe = st.empty()
        count_display = st.empty()
        chart_display = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame, counts = process_frame(frame)
            stframe.image(frame, channels="BGR", use_container_width=True)
            count_display.json(counts)
            with chart_display.container():
                show_charts(counts)
                update_trend(selected_store, counts)
                check_alerts(selected_store, counts)
        cap.release()
        save_report(selected_store)

elif use_webcam:
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    count_display = st.empty()
    chart_display = st.empty()
    st.info(f"{selected_store} Webcam running... close tab to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame, counts = process_frame(frame)
        stframe.image(frame, channels="BGR", use_container_width=True)
        count_display.json(counts)
        with chart_display.container():
            show_charts(counts)
            update_trend(selected_store, counts)
            check_alerts(selected_store, counts)
    cap.release()
    save_report(selected_store)
