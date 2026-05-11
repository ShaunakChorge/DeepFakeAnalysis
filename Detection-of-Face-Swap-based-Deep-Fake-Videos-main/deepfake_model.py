import tensorflow as tf
import numpy as np
import cv2
import tempfile
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import streamlit as st
import gdown

# Google Drive file ID
file_id = "1UiYjPQBC-mZO4qETov4C6oFlnSNeSf2i"  
model_path = "deepfake_model.h5"

# Function to download model
@st.cache_resource  # ✅ Compatible with older Streamlit versions
def load_model():
    gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)
    return tf.keras.models.load_model(model_path)

# Load the model
model = load_model()

# Function to detect deepfakes in a video
def detect_deep_fake(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Unable to open video file."}

    frame_scores = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame = cv2.resize(frame, (224, 224))  # Resize for model
        frame = frame.astype("float32") / 255.0  # Normalize
        frame = np.expand_dims(frame, axis=0)  # Add batch dimension

        prediction = model.predict(frame)[0][0]  # Model output
        frame_scores.append(prediction)

    cap.release()

    confidence_score = np.mean(frame_scores) * 100  # Convert to percentage
    detection = "Fake" if confidence_score > 50 else "Real"

    return {
        "detection": detection,
        "confidence_score": round(confidence_score, 2),
        "frame_scores": frame_scores
    }

# Function to generate a detailed PDF report
def generate_pdf_report(video_name, detection_result):
    report_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name

    c = canvas.Canvas(report_path, pagesize=letter)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(200, 750, "Deep Fake Detection Report")

    c.setFont("Helvetica", 12)
    c.drawString(50, 720, f"Video Name: {video_name}")
    c.drawString(50, 700, f"Detection Result: {detection_result['detection']}")
    c.drawString(50, 680, f"Confidence Score: {detection_result['confidence_score']}%")

    c.drawString(50, 650, "Interpretation:")
    c.setFont("Helvetica", 10)
    c.drawString(70, 630, "- Above 80%: High confidence that the video is a deepfake.")
    c.drawString(70, 610, "- 50% to 80%: Moderate confidence of deepfake content.")
    c.drawString(70, 590, "- Below 50%: The video appears genuine.")

    # Add Frame Scores in a column format
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, 560, "Frame-wise Analysis:")

    c.setFont("Helvetica", 10)
    y_position = 540
    for i, score in enumerate(detection_result["frame_scores"][:40]):  # Show up to 40 frame scores
        c.drawString(70, y_position, f"Frame {i+1}: {round(score * 100, 2)}%")
        y_position -= 15
        if y_position < 50:  # New page if space runs out
            c.showPage()
            c.setFont("Helvetica", 10)
            y_position = 750

    c.save()
    return report_path
