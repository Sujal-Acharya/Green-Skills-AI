import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import os

# Set page config
st.set_page_config(
    page_title="Forest Fire Detection",
    page_icon="🌲",
    layout="centered",
    initial_sidebar_state="auto"
)

# Load model
@st.cache_resource
def load_fire_model():
    current_dir = os.path.dirname(__file__)
    model_path = os.path.join(current_dir, 'Forest_fire_detection_model.h5')
    return load_model(model_path)

model = load_fire_model()

# Custom CSS for better UI
st.markdown("""
    <style>
        .main {
            background-color: #f0f2f6;
        }
        .stButton>button {
            background-color: #ff7043;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 0.5em 2em;
        }
        .stFileUploader {
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# App title and description
st.title("🌲 Forest Fire Detection")
st.markdown(
    """
    <div style='font-size:18px; color:#333;'>
        Upload a forest image to detect the presence of wildfire using a deep learning model.<br>
        <b>Supported formats:</b> JPG, JPEG, PNG
    </div>
    """, unsafe_allow_html=True
)

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear image of a forest area."
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.markdown("---")

    with st.spinner("Analyzing image..."):
        # Preprocess image
        img_for_model = image.resize((64, 64))
        img_array = np.array(img_for_model) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)[0][0]
        prob = float(prediction)
        if prob > 0.5:
            st.error("🔥 **Wildfire Detected!**")
            st.progress(int(prob * 100))
        else:
            st.success("✅ **No Wildfire Detected.**")
            st.progress(int((1 - prob) * 100))

        st.markdown(
            f"<div style='font-size:16px;'>Detection Confidence: <b>{prob*100:.2f}%</b></div>",
            unsafe_allow_html=True
        )
else:
    st.info("Please upload a forest image to start detection.")