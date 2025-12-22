import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import os

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mn_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess

# =========================
# CONFIG
# =========================
IMG_SIZE = (128, 128)   # HARUS sama dengan training
CLASSES = ['Normal', 'Cataract']

st.set_page_config(
    page_title="Cataract Detection System",
    page_icon="👁️",
    layout="centered"
)

st.title("👁️ Cataract Detection Dashboard")
st.markdown("**UAP Pembelajaran Mesin – Klasifikasi Citra Fundus Mata**")

# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_models():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    cnn_model = tf.keras.models.load_model(
        os.path.join(BASE_DIR, "notebooks", "cnn_cataract_model.h5")
    )

    mobilenet_model = tf.keras.models.load_model(
        os.path.join(BASE_DIR, "notebooks", "mobilenetv2_cataract_model.h5")
    )

    efficientnet_model = tf.keras.models.load_model(
        os.path.join(BASE_DIR, "notebooks", "efficientnetb0_cataract_model.h5")
    )

    return cnn_model, mobilenet_model, efficientnet_model


cnn_model, mobilenet_model, efficientnet_model = load_models()

# =========================
# MODEL SELECTION
# =========================
model_option = st.selectbox(
    "🧠 Pilih Model",
    ("CNN", "MobileNetV2", "EfficientNetB0")
)

# =========================
# IMAGE UPLOAD
# =========================
uploaded_file = st.file_uploader(
    "📤 Upload Citra Fundus Mata",
    type=["jpg", "jpeg", "png"]
)

# =========================
# PREPROCESSING
# =========================
def preprocess_image(image, model_name):
    image = np.array(image)
    image = cv2.resize(image, IMG_SIZE)

    if model_name == "CNN":
        image = image / 255.0
    elif model_name == "MobileNetV2":
        image = mn_preprocess(image)
    elif model_name == "EfficientNetB0":
        image = eff_preprocess(image)

    image = np.expand_dims(image, axis=0)
    return image

# =========================
# PREDICTION
# =========================
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Citra Fundus Input", use_container_width=True)

    if st.button("🔍 Prediksi"):
        with st.spinner("Model sedang memproses..."):
            x = preprocess_image(img, model_option)

            if model_option == "CNN":
                model = cnn_model
            elif model_option == "MobileNetV2":
                model = mobilenet_model
            else:
                model = efficientnet_model

            # ===== OUTPUT MODEL (SIGMOID) =====
            prob = model.predict(x)[0][0]

            # ===== INTERPRETASI YANG BENAR =====
            # CLASSES = ['Normal', 'Cataract']
            # Normal = 0, Cataract = 1
            if prob >= 0.5:
                label = "Cataract"
                confidence = prob
            else:
                label = "Normal"
                confidence = 1 - prob

        # =========================
        # DISPLAY RESULT
        # =========================
        st.success(f"🧠 Model: {model_option}")
        st.info(f"📌 Hasil Prediksi: {label}")
        st.write(f"📊 Confidence: {confidence:.2%}")

        if label == "Cataract":
            st.error("⚠️ Terindikasi Katarak")
        else:
            st.success("✅ Mata Normal")