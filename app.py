import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from utils import preprocess_image
import requests
from streamlit_lottie import st_lottie
from openai import OpenAI

api_key = st.secrets["OPENAI_API_KEY"]
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
st.markdown("<div class='sub-title'>Upload a heart or angiogram image to detect potential blockages and get AI-driven detailed analysis</div>", unsafe_allow_html=True)

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

# ========== OPENAI CLIENT ==========
if use_ai_agent:
    try:
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    except Exception:
        st.warning("⚠️ OpenAI API key not found. Add it in Streamlit Secrets to use AI reasoning.")

# ========== MAIN CONTENT ==========
uploaded_file = st.file_uploader("📤 Upload a Heart Scan Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    col1, col2 = st.columns(2)

    with col1:
        image = np.array(Image.open(uploaded_file))
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

        # ========== VISUAL OUTPUTS ==========
        st.markdown("### 🩻 Visualization Results")
        colA, colB, colC = st.columns(3)
        with colA:
            st.image(image, caption="Original Image", use_container_width=True)
        with colB:
            st.image(mask * 255, caption="Predicted Mask", use_container_width=True)
        with colC:
            st.image(overlay, caption="Overlay Visualization", use_container_width=True)

        st.markdown("---")
        st.subheader("🤖 AI Assistant Analysis")

        # Base message before AI reasoning
        if result_text == "blockage":
            base_message = f"""
                The scan indicates potential blockage with confidence {confidence:.1f}%.
                Detected regions may represent restricted blood flow or vessel narrowing.
                """
        else:
            base_message = f"""
                The scan shows no visible blockage with confidence {confidence:.1f}%.
                Vessel structures appear uniform and healthy.
                """

        # AI detailed reasoning
        if use_ai_agent and "client" in locals():
            with st.spinner("🧠 Generating detailed AI interpretation..."):
                prompt = f"""
                You are a cardiology AI assistant analyzing a heart scan model output.
                The model result is: {result_text}.
                Confidence: {confidence:.1f}%.
                Provide a short, professional explanation describing what this means medically, 
                how reliable it might be, and what the next steps should be.
                Make the tone human, informative, and medically neutral.
                """
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}]
                    )
                    ai_explanation = response.choices[0].message.content
                except Exception as e:
                    ai_explanation = f"(AI explanation unavailable — {e})"
        else:
            ai_explanation = "AI reasoning is disabled. Enable it from the sidebar for detailed medical explanation."

        # Display AI output
        color_gradient = "linear-gradient(135deg, #ff4d4d, #b30000);" if result_text == "blockage" else "linear-gradient(135deg, #06d6a0, #118ab2);"

        st.markdown(f"""
            <div class='ai-box' style="background: {color_gradient}">
                <b>AI Summary:</b><br>{base_message}<br><br>
                <b>AI Detailed Analysis:</b> {ai_explanation}
            </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("<div class='footer'>© 2025 Heart Blockage Detection | Developed by <b>Satya Sahil</b> | Powered by TensorFlow, OpenAI & Streamlit</div>", unsafe_allow_html=True)
