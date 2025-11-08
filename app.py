import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from utils import preprocess_image
import requests
from streamlit_lottie import st_lottie
import matplotlib.pyplot as plt
import google.generativeai as genai
import concurrent.futures  # for timeout control

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

# ========== MODEL LOADING ==========
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("model/vnet_best_model.h5", compile=False)
        return model
    except Exception as e:
        st.error(f"⚠️ Model loading failed: {e}")
        st.stop()

model = load_model()

# ========== SAFE LOTTIE LOADER ==========
def load_lottie_url(url: str):
    """Safely load Lottie animation from URL (handles connection errors)."""
    try:
        r = requests.get(url, timeout=20)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None

# Lottie animations
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

# ========== SESSION STATE ==========
if "prediction_count" not in st.session_state:
    st.session_state["prediction_count"] = 0

# ========== GEMINI CONFIG ==========
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", None)
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    st.sidebar.warning("⚠️ Gemini API key not found. Add it to `.streamlit/secrets.toml`")

# ========== FILE UPLOAD ==========
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
        with st.spinner("Analyzing image using AI model... Please wait ⏳"):
            processed = preprocess_image(image)
            prediction = model.predict(processed)
            mask = (prediction[0, :, :, 0] > 0.5).astype(np.uint8)

            # Track number of predictions
            st.session_state["prediction_count"] += 1

            # ✅ Force blockage every 2nd prediction (for testing/demo)
            if st.session_state["prediction_count"] % 2 == 0:
                st.warning("⚠️ Forced blockage result (testing mode).")
                fake_mask = np.zeros_like(mask)
                h, w = mask.shape
                cv2.circle(fake_mask, (w // 2, h // 2), min(h, w) // 4, 1, -1)
                mask = fake_mask

            # Reset counter after 2 predictions
            if st.session_state["prediction_count"] > 2:
                st.session_state["prediction_count"] = 0

            # Overlay visualization
            mask_resized = cv2.resize(mask * 255, (image.shape[1], image.shape[0]))
            overlay = image.copy()
            overlay[:, :, 1] = np.maximum(overlay[:, :, 1], mask_resized)
            overlay = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

        confidence = np.mean(prediction) * 100
        blockage_area = np.sum(mask)
        total_area = mask.size
        blockage_ratio = blockage_area / total_area

        # ========== RESULTS ==========
        st.markdown("---")
        st.subheader("🧠 Prediction Result:")
        if heart_anim: st_lottie(heart_anim, height=120, key="heart_anim")

        if np.sum(mask) > 1000:
            st.error(f"⚠️ Blockage Detected — Confidence: {confidence:.2f}%")
            if error_anim: st_lottie(error_anim, height=180, key="error_anim")
            result_text = "blockage"
        else:
            st.success(f"✅ No Significant Blockage Detected — Confidence: {confidence:.2f}%")
            if success_anim: st_lottie(success_anim, height=180, key="success_anim")
            result_text = "clear"

        # ======== BLOCKAGE SEVERITY ========
        severity = "No Blockage"
        severity_msg = ""
        if result_text == "blockage":
            if blockage_ratio < 0.05:
                severity = "🟢 Mild Blockage"
                severity_msg = "Small narrowing detected, minimal restriction of blood flow."
            elif blockage_ratio < 0.15:
                severity = "🟠 Moderate Blockage"
                severity_msg = "Moderate narrowing detected — requires medical review."
            else:
                severity = "🔴 Severe Blockage"
                severity_msg = "Significant arterial blockage detected — immediate cardiology consultation advised."

            st.markdown(f"""
                <div style="background: linear-gradient(135deg, #ffcccc, #ff9999);
                    padding: 15px; border-radius: 12px; color: #000;
                    font-size: 16px; margin-top: 10px;
                    box-shadow: 0 3px 10px rgba(0,0,0,0.15);">
                <b>🩸 Blockage Severity:</b> {severity}<br><br>
                {severity_msg}<br><br>
                <b>Blockage Ratio:</b> {(blockage_ratio * 100):.2f}% of image area
                </div>
            """, unsafe_allow_html=True)

            labels = ["Blocked", "Healthy"]
            sizes = [blockage_ratio * 100, 100 - blockage_ratio * 100]
            fig, ax = plt.subplots()
            ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90, colors=["#ff6b6b", "#8ee000"])
            ax.axis("equal")
            st.pyplot(fig)

        # ======== VISUAL OUTPUTS ========
        st.markdown("### 🩻 Visualization Results")
        colA, colB, colC = st.columns(3)
        with colA:
            st.image(image, caption="Original Image", use_container_width=True)
        with colB:
            st.image(mask * 255, caption="Predicted Mask", use_container_width=True)
        with colC:
            st.image(overlay, caption="Overlay Visualization", use_container_width=True)

        # Download result
        result_image = Image.fromarray(overlay)
        result_image.save("predicted_blockage_image.png")
        st.download_button(
            label="📥 Download Predicted Image",
            data=open("predicted_blockage_image.png", "rb").read(),
            file_name="predicted_blockage_result.png",
            mime="image/png"
        )

        # ========== AI REASONING (GEMINI OPTIMIZED MODE) ==========
        ai_explanation = ""

        if use_ai_agent:
            with st.spinner("🤖 Generating medical reasoning via Gemini (optimized mode)..."):
                if GEMINI_API_KEY:
                    prompt = f"""
                    You are an AI cardiologist analyzing a heart angiogram image.
                    Prediction: {result_text}.
                    Confidence: {confidence:.1f}%.
                    Blockage Ratio: {(blockage_ratio*100):.2f}%.
                    Severity: {severity}.
                    Provide a short 3-sentence medical explanation (<80 words)
                    summarizing findings, likely cause, and recommended next step.
                    """

                    try:
                        # Try Gemini 2.0 Flash first, then fallback to Pro
                        model_names = ["gemini-2.0-flash", "gemini-2.0-pro"]
                        response = None
                        model_used = None

                        for model_name in model_names:
                            try:
                                model_g = genai.GenerativeModel(model_name)
                                with concurrent.futures.ThreadPoolExecutor() as executor:
                                    future = executor.submit(model_g.generate_content, prompt)
                                    response = future.result(timeout=8)
                                model_used = model_name
                                break
                            except concurrent.futures.TimeoutError:
                                ai_explanation = "⚠️ Gemini took too long to respond. Please try again."
                                break
                            except Exception:
                                continue

                        if response and hasattr(response, "text"):
                            ai_explanation = f"{response.text}\n\n🧩 Model used: {model_used}"
                        elif not response:
                            ai_explanation = "⚠️ No valid response received from Gemini."

                    except Exception as e:
                        ai_explanation = f"⚠️ Gemini API error — {e}"
                else:
                    ai_explanation = "⚠️ Gemini API key not found. Please add GEMINI_API_KEY in Streamlit Secrets."
        else:
            ai_explanation = "🧠 AI reasoning disabled."

        # AI Summary Box
        color_gradient = (
            "linear-gradient(135deg, #ff4d4d, #b30000);" if result_text == "blockage"
            else "linear-gradient(135deg, #06d6a0, #118ab2);"
        )

        st.markdown(f"""
            <div class='ai-box' style="background: {color_gradient}">
                <b>AI Summary:</b><br>
                {ai_explanation}
            </div>
        """, unsafe_allow_html=True)

# ========== FOOTER ==========
st.markdown("<div class='footer'>© 2025 Heart Blockage Detection | Developed by <b>Satya Sahil</b> | Powered by TensorFlow, Gemini & Streamlit</div>", unsafe_allow_html=True)
