"""Streamlit interface for Mango Disease Detector."""
import streamlit as st
from PIL import Image
from predict import load_model_and_classes
from patch_utils import patch_based_predict
from gradcam import generate_gradcam

st.set_page_config(page_title="Mango Disease Detector", page_icon="🥭")
st.title("🥭 Mango Disease Detector")
st.caption("Patch-based CNN classification with Grad-CAM explainability")

@st.cache_resource
def load_detector():
    return load_model_and_classes()

try:
    model, class_names = load_detector()
except Exception as exc:
    st.error("Model not found. Train it first with: python train.py --dataset ./dataset")
    st.exception(exc)
    st.stop()

uploaded = st.file_uploader("Upload a mango leaf image", type=["jpg", "jpeg", "png", "bmp"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded leaf", use_container_width=True)
    with st.spinner("Analyzing image patches..."):
        result = patch_based_predict(model, image, class_names)
    st.subheader(f"Prediction: {result['predicted_class']}")
    st.caption(f"Analyzed {result['num_patches']} image patches")
    st.bar_chart(result["probabilities"])
    with st.spinner("Generating Grad-CAM..."):
        heatmap = generate_gradcam(model, image)
    st.image(heatmap, caption="Grad-CAM attention map", use_container_width=True)
else:
    st.info("Upload a mango leaf image to begin.")
