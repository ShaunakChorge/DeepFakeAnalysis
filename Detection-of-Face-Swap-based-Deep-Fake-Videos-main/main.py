import streamlit as st
import cv2
import os
import time
import numpy as np
from deepfake_model import detect_deep_fake, generate_pdf_report
import tempfile
import gdown

st.title("🔍 Deep Fake Detection Application")

st.sidebar.title("Navigation")
page = st.sidebar.radio("📌 Select a page:",
                        ("Home", "Upload Video", "Detection Results", "How It Works", "Feedback", "About"))

if page == "Home":
    st.header("🏠 Welcome to the Deep Fake Detection Application!")

    st.write("""
    🔍 **What is Deepfake?**  
    Deepfake technology uses AI to manipulate videos by swapping faces, mimicking voices, and creating highly realistic synthetic media.  
    These videos are often used for entertainment, but they can also be **misused for misinformation, fraud, and identity theft**.
    """)

    st.image("deepfake_example.png", caption="Example of a deepfake video")

    st.subheader("🔹 Why Detect Deepfakes?")
    st.markdown("""
    - 🚨 **Prevent misinformation** in media and social platforms.  
    - 🏦 **Combat fraud** in banking, identity verification, and financial sectors.  
    - 🎥 **Ensure video authenticity** in legal evidence and journalism.  
    - 📢 **Protect personal privacy** from unauthorized deepfake use.  
    """)

    import streamlit as st

    st.subheader("🎭 Real vs. Deepfake Process")

    col1, col2 = st.columns(2)

    with col1:
        st.image("face swap.jpg", caption="Deepfake Generation Process", use_container_width=True)

    with col2:
        st.markdown("""
        🔹 **Deepfake Creation Steps**  
        - **Detect & Crop Faces**: Extracts face from source and target  
        - **Feature Extraction**: Identifies landmarks, key points, depth maps  
        - **DeepFake Model**: AI swaps features & generates new face  
        - **Blending & Postprocessing**: Merges the new face into the target video  
        """)

    st.write("This visualization explains how deepfake videos are generated.")

    st.subheader("🔹 Features of This App")
    st.markdown("""
    - 🎥 **Upload & Analyze Videos** with ease.  
    - 📊 **Confidence Scores & Graphs** for transparency.  
    - 🧠 **AI-based Frame-by-Frame Analysis** for accurate results.  
    - 📄 **Downloadable PDF Report** summarizing detection findings.  
    - 🛡️ **User-friendly & Secure!**  
    """)

    st.success("🚀 Ready to try? Go to 'Upload Video' and start detecting deepfakes now!")


elif page == "Upload Video":
    st.header("📤 Upload Video for Detection")
    uploaded_file = st.file_uploader("🎬 Choose a video file...", type=["mp4", "mov", "avi", ".mpeg"])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        st.success("✅ Video uploaded successfully! Showing Preview...")

        cap = cv2.VideoCapture(video_path)
        frame_placeholder = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame, channels="RGB", use_container_width=True)
            time.sleep(0.03)

        cap.release()

        if st.button("🚀 Analyze Video"):
            st.success("Processing your video... ⏳")
            result = detect_deep_fake(video_path)

            if "error" in result:
                st.error(result["error"])
            else:
                st.session_state["detection_result"] = result
                st.session_state["video_name"] = uploaded_file.name
                # Display a notification after analysis
                st.balloons()  # This will show balloons to indicate success
                st.success(f"🎉 Detection Complete! The video '{uploaded_file.name}' has been analyzed.")
                st.rerun()

elif page == "Detection Results":
    st.header("📊 Detection Results")

    if "detection_result" in st.session_state:
        result = st.session_state["detection_result"]
        video_name = st.session_state.get("video_name", "Uploaded Video")

        st.write("**⚠️ Deep Fake Detection:**", result["detection"])
        st.write("**✅ Confidence Score:**", result["confidence_score"], "%")

        st.subheader("📈 Frame-by-Frame Analysis")
        st.line_chart(result["frame_scores"])

        st.subheader("📢 Interpretation of Results")
        if result["confidence_score"] > 80:
            st.error("⚠️ High confidence that this video is a deepfake!")
        elif 50 < result["confidence_score"] <= 80:
            st.warning("⚠️ Moderate confidence of deepfake content detected.")
        else:
            st.success("✅ The video appears genuine.")

        # 🔹 **Download PDF Report Button**
        if st.button("📥 Generate PDF Report"):
            pdf_path = generate_pdf_report(video_name, result)
            with open(pdf_path, "rb") as file:
                st.download_button("📄 Click to Download Report", file, file_name="DeepFake_Report.pdf")

    else:
        st.warning("⚠️ No detection results available. Please analyze a video first.")

elif page == "How It Works":
    st.header("🛠 How Deep Fake Detection Works")
    st.write("""
    Our system uses:
    - 🧠 **CNN** for facial feature analysis  
    - 🎞 **Frame-wise Confidence Scoring**  
    - 🔬 **Deep Learning Models** trained on real and fake datasets  
    """)
    st.image("model_architecture.png", caption="Model Architecture")

elif page == "Feedback":
    st.header("💬 User Feedback")
    feedback = st.text_area("✍️ Your feedback here:")
    if st.button("📩 Submit Feedback"):
        st.success("🙏 Thank you for your feedback!")

# About Page
elif page == "About":
    st.header("📚 About This Application")
    st.write(
        "This Deep Fake Detection Application helps users identify deepfake videos using state-of-the-art AI models. "
        "It provides an interactive interface for video analysis and insights."
    )
    st.write("For more details, visit our [GitHub Repository](link).")

st.sidebar.write("")


