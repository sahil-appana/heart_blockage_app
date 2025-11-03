import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from utils import preprocess_image
import requests
from streamlit_lottie import st_lottie
from openai import OpenAI
import matplotlib.pyplot as plt

# ========== SAFE OPENAI CLIENT SETUP ==========
api_key = st.secrets.get("OPENAI_API_KEY", None)
client = None
if api_key:
    client = OpenAI(api_key=api_key)

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="🫀 Heart Blockage Detection AI",
    page_icon="❤️",
    layout="wide"
)

# ========== HEADER & STYLES ==========
st.markdown("""
    <style>
        .main-title {
            font-size: 42px;
            font-weight: 700;
            color: #d90429;
            text-align: center;
            margin-bottom: 5px;
        }
        .sub-title {
            font-size: 18px;
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }
        .ai-box {
            padding: 20px;
            border-radius: 12px;
            margin-top: 25px;
            font-size: 16px;
            line-height: 1.6;
            color: white;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.2);
        }
        .footer {
            text-align: center;
            color: #888;
            font-size: 13px;
            margin-top: 60px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🫀 Heart Blockage Detection using VNet + AI Agent</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Upload a heart or angiogram image to detect potential blockages and view AI-driven severity analysis</div>", unsafe_allow_html=True)

# ========== LOAD MODEL ==========
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model/vnet_best_model.h5", compile=False)
    return model

model = load_model()

# ========== LOAD LOTTIE ANIMATIONS ==========
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

success_anim = load_lottie_url("https://assets8.lottiefiles.com/packages/lf20_0yfsb3a1.json")
error_anim = load_lottie_url("https://assets2.lottiefiles.com/private_files/lf30_wqypnpu5.json")
heart_anim = load_lottie_url("https://assets4.lottiefiles.com/packages/lf20_qp1q7mct.json")

# ========== SIDEBAR ==========
st.sidebar.header("⚙️ App Configuration")
st.sidebar.markdown("**Model:** VNet (50 Epochs)")
st.sidebar.markdown("**Validation Accuracy:** ~99.2%")
st.sidebar.markdown("**Developer:** Satya Sahil")
st.sidebar.divider()

use_ai_agent = st.sidebar.toggle("🤖 Enable AI Detailed Reasoning", value=True)
st.sidebar.info("📁 Upload only medical scan images (JPG, PNG, JPEG).")

# ========== MAIN CONTENT ==========
uploaded_file = st.file_uploader("📤 Upload a Heart Scan Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    col1, col2 = st.columns(2)

    with col1:
        image = Image.open(uploaded_file).convert("RGB")
        image = np.array(image)
        st.image(image, caption="🩻 Uploaded Image", use_container_width=True)

    with col2:
        st.info("✅ Image loaded successfully. Click **Predict Blockage** below to start AI analysis.")

    st.markdown("---")

    if st.button("🔍 Predict Blockage", use_container_width=True):
        with st.spinner("Analyzing the image using AI model... Please wait ⏳"):
            processed = preprocess_image(image)
            prediction = model.predict(processed)

            mask = (prediction[0, :, :, 0] > 0.5).astype(np.uint8)
            mask_resized = cv2.resize(mask * 255, (image.shape[1], image.shape[0]))

            overlay = image.copy()
            overlay[:, :, 1] = np.maximum(overlay[:, :, 1], mask_resized)
            overlay = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

        confidence = np.mean(prediction) * 100
        blockage_area = np.sum(mask)
        total_area = mask.size
        blockage_ratio = blockage_area / total_area

        st.markdown("---")
        st.subheader("🧠 Prediction Result:")
        st_lottie(heart_anim, height=120, key="heart_anim")

        if np.sum(mask) > 1000:
            st.error(f"⚠️ Blockage Detected — Confidence: {confidence:.2f}%")
            st_lottie(error_anim, height=180, key="error_anim")
            result_text = "blockage"
        else:
            st.success(f"✅ No Significant Blockage Detected — Confidence: {confidence:.2f}%")
            st_lottie(success_anim, height=180, key="success_anim")
            result_text = "clear"

        # ======== BLOCKAGE SEVERITY ESTIMATION ========
        if result_text == "blockage":
            if blockage_ratio < 0.05:
                severity = "🟢 Mild Blockage"
                severity_msg = "Small narrowing detected, minimal restriction of blood flow."
            elif blockage_ratio < 0.15:
                severity = "🟠 Moderate Blockage"
                severity_msg = "Moderate narrowing in one or more vessel regions — requires medical review."
            else:
                severity = "🔴 Severe Blockage"
                severity_msg = "Significant arterial blockage detected — immediate cardiology consultation advised."

            st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #ffcccc, #ff9999);
                    padding: 15px;
                    border-radius: 12px;
                    color: #000;
                    font-size: 16px;
                    margin-top: 10px;
                    box-shadow: 0 3px 10px rgba(0,0,0,0.15);
                ">
                <b>🩸 Blockage Severity:</b> {severity}<br><br>
                {severity_msg}<br><br>
                <b>Blockage Ratio:</b> {(blockage_ratio * 100):.2f}% of image area
                </div>
            """, unsafe_allow_html=True)

            # Pie chart visualization
            labels = ["Blocked", "Healthy"]
            sizes = [blockage_ratio * 100, 100 - blockage_ratio * 100]
            fig, ax = plt.subplots()
            ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90, colors=["#ff6b6b", "#8ee000"])
            ax.axis("equal")
            st.pyplot(fig)

        # ========== VISUAL OUTPUTS ==========
        st.markdown("### 🩻 Visualization Results")
        colA, colB, colC = st.columns(3)
        with colA:
            st.image(image, caption="Original Image", use_container_width=True)
        with colB:
            st.image(mask * 255, caption="Predicted Mask", use_container_width=True)
        with colC:
            st.image(overlay, caption="Overlay Visualization", use_container_width=True)

        # Save blockage overlay prediction
        result_image = Image.fromarray(overlay)
        result_image.save("predicted_blockage_image.png")
        st.download_button(
            label="📥 Download Predicted Image",
            data=open("predicted_blockage_image.png", "rb").read(),
            file_name="predicted_blockage_result.png",
            mime="image/png"
        )

        # ========== AI REASONING (OpenAI Integration) ==========
        if use_ai_agent and client:
            with st.spinner("🧠 Generating detailed AI interpretation..."):
                prompt = f"""
                You are a cardiology AI assistant analyzing a heart scan model output.
                The model result is: {result_text}.
                Confidence: {confidence:.1f}%.
                Blockage Ratio: {(blockage_ratio * 100):.2f}%.
                Severity: {severity if result_text == 'blockage' else 'No Blockage'}.
                Provide a clear, medically accurate summary and advice.
                """
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}]
                    )
                    ai_explanation = response.choices[0].message.content
                except Exception as e:
                    ai_explanation = f"(AI explanation unavailable — {e})"
        elif use_ai_agent and not client:
            ai_explanation = "⚠️ OpenAI API key not found. Please add your key in Streamlit Secrets."
        else:
            ai_explanation = "AI reasoning is disabled. Enable it from the sidebar for detailed medical explanation."

        color_gradient = (
            "linear-gradient(135deg, #ff4d4d, #b30000);"
            if result_text == "blockage"
            else "linear-gradient(135deg, #06d6a0, #118ab2);"
        )

        st.markdown(f"""
            <div class='ai-box' style="background: {color_gradient}">
                <b>AI Summary:</b><br>
                {ai_explanation}
            </div>
        """, unsafe_allow_html=True)

# ========== FOOTER ==========
st.markdown("<div class='footer'>© 2025 Heart Blockage Detection | Developed by <b>Satya Sahil</b> | Powered by TensorFlow, OpenAI & Streamlit</div>", unsafe_allow_html=True)
