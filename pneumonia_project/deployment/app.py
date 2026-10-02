import os
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# ---------- Model loader (cached so it loads once per session) ----------
@st.cache_resource
def load_pneumonia_model():
    model_path = os.path.join(os.path.dirname(__file__), "best_model_vgg16_aug.h5")
    return load_model(model_path)

model = load_pneumonia_model()

# ---------- Decision threshold ----------
# Model was trained with rescale=1./255 and outputs shifted-low probabilities.
# 0.3 gave clean separation on the diagnostic run.
THRESHOLD = 0.3

# ---------- UI ----------
st.title("Pneumonia Detection App")
st.write("""
Upload a chest X-ray image (JPG, PNG, or DICOM) to get a prediction.
The model will classify the image as **Pneumonia** or **Healthy** and show the confidence.
""")

uploaded_file = st.file_uploader(
    "Choose a chest X-ray...",
    type=["jpg", "jpeg", "png", "dcm"]
)

# ---------- Helpers ----------
def load_dicom_as_pil(file_obj):
    """Read a DICOM file and return a PIL RGB image."""
    import pydicom
    ds = pydicom.dcmread(file_obj)
    arr = ds.pixel_array.astype(np.float32)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255.0
    arr = arr.astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")

def preprocess_for_model(pil_image, target_size=(128, 128)):
    """Resize + rescale to [0, 1], matching how the model was trained."""
    img = pil_image.resize(target_size)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

# ---------- Main flow ----------
if uploaded_file is not None:
    try:
        if uploaded_file.name.lower().endswith(".dcm"):
            image = load_dicom_as_pil(uploaded_file)
        else:
            image = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"Could not read the uploaded file: {e}")
        st.stop()

    # Smaller, centered display
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption="Uploaded X-ray", width=250)

    # Predict
    img_array = preprocess_for_model(image)
    pred_prob = float(model.predict(img_array)[0][0])

    pred_class = "Pneumonia" if pred_prob > THRESHOLD else "Healthy"
    confidence = pred_prob if pred_class == "Pneumonia" else 1 - pred_prob

    st.subheader("Prediction Result")
    st.markdown(f"**Class:** {pred_class}")
    st.markdown(f"**Confidence:** {confidence:.2%}")
    st.caption(f"Raw pneumonia probability (class = 1): {pred_prob:.4f}   |   Threshold: {THRESHOLD}")
