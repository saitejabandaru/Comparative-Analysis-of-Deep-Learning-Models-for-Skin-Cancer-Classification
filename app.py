import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

MODEL_PATH = "xception_best_model2.h5"
IMG_SIZE = (299, 299)

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

CLASS_FULL_NAMES = {
    "akiec": "Actinic keratoses / intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}

st.set_page_config(page_title="Skin Cancer Classifier", page_icon="🩺", layout="centered")

@st.cache_resource
def get_model():
    return load_model(MODEL_PATH, compile=False)

def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    arr = np.array(image).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

st.title("🩺 Skin Cancer Classification")
st.write("Upload a dermoscopic image to predict the lesion class.")

try:
    model = get_model()
    st.success("Model loaded successfully")
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Predict"):
        with st.spinner("Running prediction..."):
            x = preprocess_image(image)
            probs = model.predict(x, verbose=0)[0]
            pred_idx = int(np.argmax(probs))
            pred_class = CLASS_NAMES[pred_idx]
            confidence = float(probs[pred_idx])

        st.subheader("Prediction")
        st.write(f"**Predicted Class:** {pred_class}")
        st.write(f"**Meaning:** {CLASS_FULL_NAMES[pred_class]}")
        st.write(f"**Confidence:** {confidence * 100:.2f}%")

        st.subheader("All class probabilities")
        sorted_results = sorted(zip(CLASS_NAMES, probs), key=lambda t: t[1], reverse=True)
        for cls, prob in sorted_results:
            st.write(f"**{cls}** — {CLASS_FULL_NAMES[cls]}: {prob * 100:.2f}%")
            st.progress(float(prob))

st.caption("For educational/demo use only. Not medical advice.")